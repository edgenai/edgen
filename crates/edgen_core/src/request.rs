/* Copyright 2023- The Binedge, Lda team. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use core::fmt::{Debug, Display, Formatter};
use std::ops::AddAssign;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use derive_more::Deref;
use once_cell::sync::Lazy;
use serde::Serialize;
use thiserror::Error;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};
use tokio::{select, spawn};
use tracing::{error, info, warn};

use crate::settings::{DevicePolicy, SETTINGS};

/// Global request manager.
pub static REQUEST_QUEUE: Lazy<RequestManager> = Lazy::new(RequestManager::new);

#[derive(Serialize, Error, Debug)]
pub enum QueueError {
    #[error("the queue has already been closed")]
    Closed(String),
    #[error("an error has occurred while waiting in queue {0}")]
    Enqueue(String),
    #[error("cannot fulfill request, model does not fit in total memory")]
    Unfulfillable,
    #[error("no device in the system is present that can fulfill the requirements")]
    NoSuchDevice,
    #[error("must enqueue with a specific device")]
    Unspecified,
}

/// An object for orchestrating the execution of several parallel requests.
///
/// Each request should be submitted to a queue, with a queue existing for each backend device. Request execution
/// is throttled if the memory limit of a device is reached, to avoid *OOM* errors.
pub struct RequestManager {
    /// The sender used to push new items into the queue.
    item_tx: UnboundedSender<QueueItem>,

    /// The fast track sender used to push recursive request items into the queue.
    ft_item_tx: UnboundedSender<QueueItem>,

    /// The sender used to notify the queue of resources being freed.
    free_tx: UnboundedSender<()>,

    /// The join handle the the queue's thread.
    thread: JoinHandle<()>,

    /// Every device found in the system.
    devices: Arc<Vec<Device>>,
}

impl RequestManager {
    /// Create a new request manager.
    fn new() -> Self {
        let mut devices = vec![];
        for hw_device in memonitor::list_all_devices().iter() {
            if let memonitor::DeviceKind::GPU(memonitor::GPUKind::Integrated) = hw_device.kind() {
                continue;
            }

            info!(
                "Found device: {} @ {:?}",
                hw_device.name(),
                hw_device.backend()
            );

            let device = match hw_device.backend() {
                memonitor::BackendId::CPU => DeviceId::CPU,
                memonitor::BackendId::Vulkan => DeviceId::Vulkan(hw_device.local_id()),
                memonitor::BackendId::CUDA => DeviceId::Cuda(hw_device.local_id()),
                _ => {
                    unimplemented!()
                }
            };

            devices.push(Device {
                id: device,
                mm_global_id: hw_device.global_index(),
                max_memory: hw_device.current_memory_stats().total,
                reserved_memory: Arc::new(AtomicUsize::new(0)),
            })
        }
        let devices: Arc<Vec<Device>> = Arc::new(devices);

        let (item_tx, item_rx) = unbounded_channel();
        let (ft_item_tx, ft_item_rx) = unbounded_channel();
        let (drop_tx, drop_rx) = unbounded_channel();
        let thread = spawn(run_queue(item_rx, ft_item_rx, drop_rx, devices.clone()));

        Self {
            item_tx,
            ft_item_tx,
            free_tx: drop_tx,
            thread,
            devices,
        }
    }

    /// Enqueue a request in its appropriate queue, given its requirements.
    ///
    /// This functions will only return once the request has reached the end of the queue, when it is safe to execute
    /// the request without running out of memory.
    ///
    /// # Returns
    ///
    /// A [`Ticket`] necessary to call a generation function.
    pub async fn enqueue(&self, requirements: Passport) -> Result<Ticket, QueueError> {
        if requirements.device == DeviceId::Any {
            return Err(QueueError::Unspecified);
        }

        if requirements.free() {
            info!("\"Free\" request clear for immediate execution");
            return Ok(Ticket {
                content: Some(TicketContent::Free),
                device: requirements.device,
            });
        }

        let (required_host, required_device) = requirements.memory();

        let (idx, device) = {
            let mut matched = None;
            let mut matched_idx = None;
            for (idx, device) in self.devices.iter().enumerate() {
                if device.id == requirements.device {
                    matched = Some(device);
                    matched_idx = Some(idx);
                    break;
                }
            }

            if let Some(device) = matched {
                let idx = matched_idx.expect("There should also be a matching index");
                if device.id == DeviceId::CPU {
                    if device.max_memory < required_host + required_device {
                        return Err(QueueError::Unfulfillable);
                    } else {
                        (idx, device)
                    }
                } else if self.devices[0].max_memory < required_host
                    || device.max_memory < required_device
                {
                    return Err(QueueError::Unfulfillable);
                } else {
                    (idx, device)
                }
            } else {
                return Err(QueueError::NoSuchDevice);
            }
        };

        info!(
            "Enqueuing request for device {:?} ({:?})",
            device.id, requirements.request
        );

        let (ticket_tx, ticket_rx) = oneshot::channel();
        let waiter = QueueItem::Normal {
            passport: requirements,
            ticket_tx,
            device_idx: idx,
        };
        self.item_tx
            .send(waiter)
            .map_err(move |e| QueueError::Closed(e.to_string()))?;

        let ticket = ticket_rx
            .await
            .map_err(move |e| QueueError::Enqueue(e.to_string()))?;

        info!("Request clear for execution: {ticket:?}");

        Ok(ticket)
    }

    /// Given a list of possible passports and a function that calculates the allocation
    /// requirements for each device, return either the best passport or a device based on the
    /// configured device selection policy.
    pub async fn pick(
        &self,
        mut passports: Vec<Passport>,
        local_size: impl Fn(DeviceId) -> (usize, usize),
    ) -> Result<QueueSelection, QueueError> {
        let mut occupancy_passports: Vec<(f32, Passport)> = passports
            .drain(..)
            .map(|p| {
                let mut device = None;
                for d in self.devices.iter() {
                    if d.id == p.device {
                        device = Some(d);
                        break;
                    }
                }
                let occupancy = device
                    .expect("A matching device should always be found")
                    .occupancy();
                (occupancy, p)
            })
            .collect();
        // Sort by occupancy
        occupancy_passports.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        for passport in passports {
            if passport.free() {
                return Ok(QueueSelection::Passport(passport));
            }
        }
        if let Some((occupancy, _)) = occupancy_passports.first() {
            if occupancy_passports.len() == self.devices.len() || *occupancy < 0.9 {
                return Ok(QueueSelection::Passport(
                    occupancy_passports
                        .drain(..)
                        .next()
                        .expect("There should be at least one element in the vector")
                        .1,
                ));
            }
        }

        let filtered_devices = self.devices.iter().skip(1).filter(|d| {
            let mut keep = true;
            for (_, p) in &occupancy_passports {
                if d.id == p.device {
                    keep = false;
                    break;
                }
            }
            keep
        });

        let policy = SETTINGS.read().await.read().await.gpu_policy;
        let device = match policy {
            DevicePolicy::AlwaysCpu {
                overflow_to_device: false,
            } => DeviceId::CPU,
            DevicePolicy::AlwaysCpu {
                overflow_to_device: true,
            } => {
                let (host_memory0, host_memory1) = local_size(DeviceId::CPU);

                if host_memory0 + host_memory1 < self.devices[0].available_memory() {
                    DeviceId::CPU
                } else {
                    let mut device_id = DeviceId::Any;
                    for device in filtered_devices {
                        let (required_host, required_device) = local_size(device.id);

                        if required_host < self.devices[0].available_memory()
                            && required_device < device.available_memory()
                        {
                            device_id = device.id;
                            break;
                        }
                    }
                    device_id
                }
            }
            DevicePolicy::AlwaysDevice {
                overflow_to_cpu: false,
            } => {
                if self.devices.len() == 1 {
                    warn!("Device policy set to always execute on device, without CPU overflow, but no device was found in the system");
                    return Ok(QueueSelection::Device(DeviceId::CPU));
                }

                let mut device_id = DeviceId::Any;
                for device in filtered_devices {
                    let (required_host, required_device) = local_size(device.id);

                    if required_host < self.devices[0].max_memory
                        && required_device < device.max_memory
                    {
                        if device_id == DeviceId::Any {
                            device_id = device.id;
                        }

                        if required_host < self.devices[0].available_memory()
                            && required_device < device.available_memory()
                        {
                            device_id = device.id;
                            break;
                        }
                    }
                }

                if device_id == DeviceId::Any {
                    if occupancy_passports.is_empty() {
                        return Err(QueueError::Unfulfillable);
                    } else {
                        return Ok(QueueSelection::Passport(occupancy_passports.remove(0).1));
                    }
                }

                device_id
            }
            DevicePolicy::AlwaysDevice {
                overflow_to_cpu: true,
            } => {
                let mut device_id = DeviceId::Any;
                for device in filtered_devices {
                    let (required_host, required_device) = local_size(device.id);

                    if required_host < self.devices[0].available_memory()
                        && required_device < device.available_memory()
                    {
                        device_id = device.id;
                        break;
                    }
                }

                let (host_memory0, host_memory1) = local_size(DeviceId::CPU);
                if device_id == DeviceId::Any
                    && host_memory0 + host_memory1 < self.devices[0].available_memory()
                {
                    DeviceId::CPU
                } else {
                    device_id
                }
            }
        };

        Ok(QueueSelection::Device(device))
    }

    /// Notify the queue that some resource has been freed.
    pub fn notify_free(&self, _device_id: DeviceId) -> Result<(), QueueError> {
        self.free_tx
            .send(())
            .map_err(|e| QueueError::Closed(e.to_string()))
    }

    /// List every device found.
    pub fn all_devices(&self) -> Vec<DeviceId> {
        self.devices.iter().map(|device| device.id).collect()
    }

    /// Register the provided resource user.
    pub fn register_user(&self, user: Box<dyn ResourceUser>) {
        if let Err(e) = self.ft_item_tx.send(QueueItem::RegisterUser(user)) {
            error!("Failed to register resource user: {e}")
        }
    }
}

impl Drop for RequestManager {
    fn drop(&mut self) {
        self.thread.abort();
    }
}

/// An abstraction over a backend device.
///
/// A backend in this context, is a device API, like Vulkan, CUDA or Metal.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize)]
pub enum DeviceId {
    CPU,
    Vulkan(usize),
    Cuda(usize),
    Metal(usize),
    Any,
}

impl DeviceId {
    /// Return the device's id relative to its backend (matches the CUDA device index, Vulkan's order of finding
    /// devices, etc.).
    pub fn local_id(&self) -> usize {
        match self {
            DeviceId::Vulkan(id) => *id,
            DeviceId::Cuda(id) => *id,
            DeviceId::Metal(id) => *id,
            _ => 0,
        }
    }

    /// Return the name of a device provided its id in the backend, the backend id and a default name for the case
    /// where the backend wasn't found.
    fn get_name(local_id: usize, backend_id: memonitor::BackendId, default_name: &str) -> String {
        let mut name = default_name.to_string();
        for backend in memonitor::list_backends().iter() {
            if backend.id() == backend_id {
                let id = backend.device_indexes()[local_id];
                name = memonitor::list_all_devices()[id].name().to_string();
                break;
            }
        }
        name
    }

    /// Return the name of the device.
    pub fn name(&self) -> String {
        match self {
            DeviceId::CPU => memonitor::list_all_devices()[0].name().to_string(),
            DeviceId::Vulkan(local_id) => {
                Self::get_name(*local_id, memonitor::BackendId::Vulkan, "VULKAN_NOT_FOUND")
            }
            DeviceId::Cuda(local_id) => {
                Self::get_name(*local_id, memonitor::BackendId::CUDA, "CUDA_NOT_FOUND")
            }
            DeviceId::Metal(_local_id) => todo!(), // Self::get_name(*local_id, "TODO", "METAL_NOT_FOUND"),
            DeviceId::Any => "NONE".to_string(),
        }
    }
}

#[derive(Deref)]
struct Device {
    #[deref]
    id: DeviceId,
    mm_global_id: usize,
    max_memory: usize,
    reserved_memory: Arc<AtomicUsize>,
}

impl Device {
    fn available_memory(&self) -> usize {
        let real = memonitor::list_all_devices()[self.mm_global_id]
            .current_memory_stats()
            .available
            - self.reserved_memory.load(Ordering::SeqCst);

        // Reserve 15% of VRAM for the system and unaccounted runtime allocations
        let reserved = (self.max_memory as f64 * 0.15) as usize;
        if reserved < real {
            real - reserved
        } else {
            0
        }
    }

    fn reserve_memory(&self, amount: usize) -> ReservedMemory {
        ReservedMemory::reserve(amount, self.reserved_memory.clone())
    }

    fn occupancy(&self) -> f32 {
        (self.max_memory - self.available_memory()) as f32 / self.max_memory as f32
    }
}

/// The main loop for a queue.
async fn run_queue(
    mut item_rx: UnboundedReceiver<QueueItem>,
    mut ft_item_rx: UnboundedReceiver<QueueItem>,
    mut drop_rx: UnboundedReceiver<()>,
    devices: Arc<Vec<Device>>,
) {
    let mut users = vec![];

    // Attempting to make a `select!` that prioritizes items coming from ft_item_rx, unfortunately if both receivers
    // receive an item at once inside the select, ft_item_rx cannot be prioritized
    loop {
        while let Ok(item) = ft_item_rx.try_recv() {
            match item {
                QueueItem::Normal { .. } => {
                    queue_normal(&mut drop_rx, devices.as_slice(), &users, item).await
                }
                QueueItem::RegisterUser(user) => users.push(user),
            }
        }

        select! {
            item = ft_item_rx.recv() => {
                if let Some(item) = item {
                    match item {
                        QueueItem::Normal { .. } => {
                            queue_normal(&mut drop_rx, devices.as_slice(), &users, item).await
                        }
                        QueueItem::RegisterUser(user) => users.push(user),
                    }
                } else {
                    break;
                }
            }
            item = item_rx.recv() => {
                if let Some(item) = item {
                    queue_normal(&mut drop_rx, devices.as_slice(), &users, item).await;
                } else {
                    break;
                }
            }
        }
    }
}

async fn queue_normal(
    drop_rx: &mut UnboundedReceiver<()>,
    devices: &[Device],
    users: &[Box<dyn ResourceUser>],
    item: QueueItem,
) {
    if let QueueItem::Normal {
        passport,
        device_idx,
        ticket_tx: signal_tx,
    } = item
    {
        let (host_memory, device_memory) = passport.memory();

        let mut interval = interval(Duration::from_millis(2000));
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

        let mut requirements =
            MemoryRequirements::new(host_memory, device_memory, devices, device_idx);
        while !requirements.met() {
            if signal_tx.is_closed() {
                break;
            }

            let mut ordered_ids: Vec<(usize, usize)> = users
                .iter()
                .map(|user: &Box<dyn ResourceUser>| user.allocs())
                .enumerate()
                .collect();
            // Sort by decreasing allocation count
            ordered_ids.sort_by(|a, b| b.1.cmp(&a.1));

            let mut freed = FreedMemory {
                host_memory: 0,
                device_memory: 0,
            };
            for (id, _) in ordered_ids {
                if let Some((host_needed, device_needed)) = requirements.cached_met_with(&freed) {
                    // Clear drop alert channel
                    while let Ok(()) = drop_rx.try_recv() {}

                    freed += users[id]
                        .request_memory(host_needed, device_needed, devices[device_idx].id)
                        .await;
                } else {
                    break;
                }
            }

            select! {
                Some(_) = drop_rx.recv() => {},
                _ = interval.tick() => {},
            }
        }

        let ticket = Ticket {
            content: Some(passport.into_reservation(&devices[0], &devices[device_idx])),
            device: devices[device_idx].id,
        };

        let _ = signal_tx.send(ticket);
    }
}

/// The objects passed into [`Queue`]s.
enum QueueItem {
    /// A standard generation request to be executed.
    Normal {
        /// The request's passport.
        passport: Passport,

        /// The device index of the device the request is executed on.
        device_idx: usize,

        /// A oneshot channel used to signal the item has been through the queue.
        ticket_tx: oneshot::Sender<Ticket>,
    },

    /// A request to register a new resource user.
    RegisterUser(Box<dyn ResourceUser>),
}

struct MemoryRequirements<'a> {
    /// Required amount of host memory.
    host_memory: usize,

    /// Required amount of device memory.
    device_memory: usize,

    devices: &'a [Device],

    /// The [`memonitor`] device id of the device the request is executed on.
    /// If [`None`], the request is executed fully on the CPU/host.
    device_idx: Option<usize>,

    /// Cached value for available host memory.
    cached_available_host: Option<usize>,

    /// Cached value for available device memory.
    cached_available_device: Option<usize>,
}

impl<'a> MemoryRequirements<'a> {
    fn new(
        required_host: usize,
        required_device: usize,
        devices: &'a [Device],
        device_idx: usize,
    ) -> Self {
        let id = if device_idx == 0 {
            None
        } else {
            Some(device_idx)
        };

        Self {
            host_memory: required_host,
            device_memory: required_device,
            devices,
            device_idx: id,
            cached_available_host: None,
            cached_available_device: None,
        }
    }

    /// Updates cached available memory values and checks if requirements have been met.
    fn met(&mut self) -> bool {
        if let Some(id) = &self.device_idx {
            let available_host = self.devices[0].available_memory();
            let available_device = self.devices[*id].available_memory();
            self.cached_available_host = Some(available_host);
            self.cached_available_device = Some(available_device);
            self.host_memory < available_host && self.device_memory < available_device
        } else {
            let available = self.devices[0].available_memory();
            self.cached_available_host = Some(available);
            self.host_memory + self.device_memory < available
        }
    }

    /// Checks if requirements have been met, given that some memory has been freed.
    ///
    /// This function assumes [`MemoryRequirements::met`] has been called previously.
    ///
    /// # Return
    ///
    /// [`None`] if requirements have been met, otherwise the memory currently required.
    fn cached_met_with(&self, freed: &FreedMemory) -> Option<(usize, usize)> {
        if self.device_idx.is_some() {
            let available_host = self.cached_available_host.unwrap_or(0);
            let available_device = self.cached_available_device.unwrap_or(0);

            let currently_required_host = if self.host_memory < available_host + freed.host_memory {
                0
            } else {
                self.host_memory - (available_host + freed.host_memory)
            };
            let currently_required_device =
                if self.device_memory < available_device + freed.device_memory {
                    0
                } else {
                    self.device_memory - (available_device + freed.device_memory)
                };

            if 0 < currently_required_host || 0 < currently_required_device {
                Some((currently_required_host, currently_required_device))
            } else {
                None
            }
        } else {
            let available = self.cached_available_host.unwrap_or(0);
            if self.host_memory + self.device_memory
                < available + freed.host_memory + freed.device_memory
            {
                None
            } else {
                Some((
                    self.host_memory + self.device_memory
                        - (available + freed.host_memory + freed.device_memory),
                    0,
                ))
            }
        }
    }
}

#[derive(Debug, Serialize)]
/// The requirements to execute a request.
///
/// This should be acquired by querying a generation endpoint.
pub struct Passport {
    /// A basic description of the request.
    request: Request,

    /// The device the request should be executed on. Cannot be [`DeviceId::Any`].
    device: DeviceId,
}

impl Passport {
    /// Create a new passport containing a requests requirements.
    pub fn new(request: Request, device: DeviceId) -> Self {
        Self { request, device }
    }

    /// Return true if this passport's request requires multiple allocations.
    pub fn staged(&self) -> bool {
        matches!(&self.request, Request::Staged { .. })
    }

    /// Return true if this passport's request requires no allocations.
    pub fn free(&self) -> bool {
        matches!(&self.request, Request::Free)
    }

    /// Return a tuple containing the host and device memory requirements for this passport, in that order and
    /// adjusted for safety.
    fn memory(&self) -> (usize, usize) {
        match self.request {
            Request::Staged {
                host_memory,
                device_memory,
            } => (
                (host_memory as f64 * 1.2) as usize,
                (device_memory as f64 * 1.2) as usize,
            ),
            Request::Final {
                host_memory,
                device_memory,
            } => (
                (host_memory as f64 * 1.15) as usize,
                (device_memory as f64 * 1.15) as usize,
            ),
            Request::Free => (0, 0),
        }
    }

    /// Turn this passport into a reservation of host and device memory.
    fn into_reservation(self, host: &Device, device: &Device) -> TicketContent {
        match self.request {
            Request::Staged {
                host_memory,
                device_memory,
            } => TicketContent::Staged {
                host_memory: host.reserve_memory(host_memory),
                device_memory: device.reserve_memory(device_memory),
            },
            Request::Final {
                host_memory,
                device_memory,
            } => TicketContent::Final {
                host_memory: host.reserve_memory(host_memory),
                device_memory: device.reserve_memory(device_memory),
            },
            Request::Free => TicketContent::Free,
        }
    }
}

/// A basic description of a request.
#[derive(Debug, Eq, PartialEq, Serialize)]
pub enum Request {
    /// A complex request that may require multiple allocations, where estimating some allocations depend on some
    /// resource already being allocated.
    Staged {
        /// Required amount of host memory for this request.
        host_memory: usize,

        /// Required amount of device memory for this request.
        device_memory: usize,
    },

    /// A normal request.
    Final {
        /// Required amount of host memory for this request.
        host_memory: usize,

        /// Required amount of device memory for this request.
        device_memory: usize,
    },

    /// A request that does not require any allocations.
    Free,
}

/// A "ticket" that allows a generation request to be executed.
///
/// This object should be required as an argument for all generations functions.
#[derive(Debug)]
pub struct Ticket {
    /// The inner contents of the ticket.
    content: Option<TicketContent>,

    /// The device the request executes on.
    device: DeviceId,
}

impl Ticket {
    /// Consume the ticket, signaling that the required resources for the request have been allocated.
    pub fn consume(&mut self) -> bool {
        self.content.take().is_some()
    }

    /// Return the device this ticket is valid for.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Return true if this ticket only allows for a staging allocation.
    pub fn staged(&self) -> bool {
        matches!(self.content, Some(TicketContent::Staged { .. }))
    }

    /// Return true if this ticket requires no allocations.
    pub fn free(&self) -> bool {
        matches!(self.content, Some(TicketContent::Free))
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if self.content.is_some() {
            warn!("Unconsumed ticket: {self:?}")
        }
    }
}

/// The inner content of a [`Ticket`].
enum TicketContent {
    /// A ticket for one of many allocations needed for the request.
    Staged {
        /// Reserved host memory for this ticket.
        host_memory: ReservedMemory,

        /// Reserved device memory for this ticket.
        device_memory: ReservedMemory,
    },

    /// A normal ticket.
    Final {
        /// Reserved host memory for this ticket.
        host_memory: ReservedMemory,

        /// Reserved device memory for this ticket.
        device_memory: ReservedMemory,
    },

    /// No resource allocations are necessary for this ticket's request.
    Free,
}

impl Debug for TicketContent {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            TicketContent::Staged {
                host_memory,
                device_memory,
            } => write!(
                f,
                "Staged {{ host_reserved: {}MiB, device_reserved: {}MiB }}",
                host_memory.amount / 1024 / 1024,
                device_memory.amount / 1024 / 1024
            ),
            TicketContent::Final {
                host_memory,
                device_memory,
            } => write!(
                f,
                "Final {{ host_reserved: {}MiB, device_reserved: {}MiB }}",
                host_memory.amount / 1024 / 1024,
                device_memory.amount / 1024 / 1024
            ),
            TicketContent::Free => write!(f, "Free"),
        }
    }
}

/// A chunk of "reserved" memory.
struct ReservedMemory {
    /// The amount of memory reserved, in bytes.
    amount: usize,

    /// A reference to a queue's reserved memory counter.
    counter: Arc<AtomicUsize>,
}

impl ReservedMemory {
    fn reserve(amount: usize, counter: Arc<AtomicUsize>) -> Self {
        if 0 < amount {
            counter.fetch_add(amount, Ordering::SeqCst);
        }

        Self { amount, counter }
    }
}

impl Drop for ReservedMemory {
    fn drop(&mut self) {
        if 0 < self.amount {
            self.counter.fetch_sub(self.amount, Ordering::SeqCst);
        }
    }
}

/// A resource user, which can coordinate with the request manager.
#[async_trait::async_trait]
pub trait ResourceUser: Send + Sync {
    /// This user's current number of allocations.
    fn allocs(&self) -> usize;

    /// Request that the user frees some memory.
    ///
    /// The user is not forced to free anything, but it should whenever it can.
    ///
    /// # Arguments
    ///
    /// * `host_memory` - The minimum amount of host memory to be freed.
    /// * `device_memory` - The minimum amount of device memory to be freed.
    /// * `device_id` - The id of the device from which memory should freed.
    ///
    /// # Returns
    ///
    /// The amount of memory the user was able to free.
    async fn request_memory(
        &self,
        host_memory: usize,
        device_memory: usize,
        device_id: DeviceId,
    ) -> FreedMemory;
}

/// Memory freed by a call to [`ResourceUser::request_memory`].
pub struct FreedMemory {
    /// Amount of host memory freed.
    pub host_memory: usize,

    /// Amount of device memory freed.
    pub device_memory: usize,
}

impl AddAssign for FreedMemory {
    fn add_assign(&mut self, rhs: Self) {
        self.host_memory += rhs.host_memory;
        self.device_memory += rhs.device_memory;
    }
}

impl Display for FreedMemory {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "(host: {}MiB, device: {}MiB)",
            self.host_memory / 1024 / 1024,
            self.device_memory / 1024 / 1024
        )
    }
}

/// A selection made by a call to [`RequestManager::pick`].
pub enum QueueSelection {
    /// Execute the request using a provided passport.
    ///
    /// Some allocations have probably already been made for this type of selection.
    Passport(Passport),

    /// Execute the request on a specified device.
    ///
    /// It's up to the backend to generate a new [`Passport`].
    Device(DeviceId),
}