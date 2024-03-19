use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use once_cell::sync::Lazy;
use serde::Serialize;
use thiserror::Error;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio::{select, spawn};
use tracing::{error, warn};

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
}

/// An object for orchestrating the execution of several parallel requests.
///
/// Each request should be submitted to a queue, with a queue existing for each backend device. Request execution
/// is throttled if the memory limit of a device is reached, to avoid *OOM* errors.
pub struct RequestManager {
    /// The collection of queues, each mapped to a device.
    queues: DashMap<Device, Arc<Queue>>,
}

impl RequestManager {
    /// Create a new request manager.
    fn new() -> Self {
        memonitor::init();

        let queues = DashMap::new();

        for hw_device in memonitor::list_all_devices().iter() {
            if let memonitor::DeviceKind::GPU(memonitor::GPUKind::Integrated) = hw_device.kind() {
                continue;
            }

            let device = match hw_device.backend_name() {
                memonitor::CPU_NAME => Device::CPU,
                memonitor::VULKAN_NAME => Device::Vulkan(hw_device.local_id()),
                &_ => {
                    unimplemented!()
                }
            };

            let queue = Queue::new(
                device,
                hw_device.global_id(),
                hw_device.current_memory_stats().total,
            );
            queues.insert(device, Arc::new(queue));
        }

        Self { queues }
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
        let required_memory = match requirements.request {
            Request::Model(required_memory) => (required_memory as f64 * 1.1) as usize,
            Request::Regular(required_memory) => (required_memory as f64 * 1.05) as usize,
            Request::Free => {
                return Ok(Ticket {
                    content: Some(TicketContent::Free),
                    device: requirements.device,
                });
            }
        };

        let queue = if requirements.device == Device::Any {
            let mut capable_devices = Vec::with_capacity(self.queues.len());
            for device in &self.queues {
                if required_memory < device.max_memory {
                    capable_devices.push(*device.key());
                }
            }
            if capable_devices.is_empty() {
                return Err(QueueError::Unfulfillable);
            }

            // TODO decide device based on current policy
            self.queues.get(&capable_devices[0]).unwrap().clone()
        } else {
            // Unwrap should never fail
            let queue = self.queues.get(&requirements.device).unwrap();
            if queue.max_memory < required_memory {
                return Err(QueueError::Unfulfillable);
            } else {
                queue.clone()
            }
        };

        queue.enqueue(required_memory).await
    }

    /// Given a function that calculates the allocation requirements for each device, return a device based on the
    /// configured device selection policy.
    pub async fn pick_device(
        &self,
        local_size: impl Fn(Device) -> usize,
    ) -> Result<Device, QueueError> {
        let policy = SETTINGS.read().await.read().await.gpu_policy;
        let device = match policy {
            DevicePolicy::AlwaysCpu {
                overflow_to_device: false,
            } => Device::CPU,
            DevicePolicy::AlwaysCpu {
                overflow_to_device: true,
            } => {
                if local_size(Device::CPU)
                    < memonitor::list_all_devices()[0]
                        .current_memory_stats()
                        .available
                {
                    Device::CPU
                } else {
                    let mut device = Device::Any;
                    for queue in self.queues.iter() {
                        if *queue.key() == Device::CPU {
                            continue;
                        }

                        if local_size(*queue.key())
                            < memonitor::list_all_devices()[queue.device_id]
                                .current_memory_stats()
                                .available
                        {
                            device = *queue.key();
                            break;
                        }
                    }
                    device
                }
            }
            DevicePolicy::AlwaysDevice {
                overflow_to_cpu: false,
            } => {
                if self.queues.len() == 1 {
                    // Only CPU is available.
                    return Err(QueueError::NoSuchDevice);
                }

                let mut device = Device::Any;
                for queue in self.queues.iter() {
                    if *queue.key() == Device::CPU {
                        continue;
                    }

                    if local_size(*queue.key())
                        < memonitor::list_all_devices()[queue.device_id]
                            .current_memory_stats()
                            .available
                    {
                        device = *queue.key();
                        break;
                    }
                }

                if device == Device::Any {
                    return Err(QueueError::Unfulfillable);
                }
                warn!("{:?}", device);
                device
            }
            DevicePolicy::AlwaysDevice {
                overflow_to_cpu: true,
            } => {
                let mut device = Device::Any;
                for queue in self.queues.iter() {
                    if *queue.key() == Device::CPU {
                        continue;
                    }

                    if local_size(*queue.key())
                        < memonitor::list_all_devices()[queue.device_id]
                            .current_memory_stats()
                            .available
                    {
                        device = *queue.key();
                        break;
                    }
                }

                if device == Device::Any
                    && local_size(Device::CPU)
                        < memonitor::list_all_devices()[0]
                            .current_memory_stats()
                            .available
                {
                    Device::CPU
                } else {
                    device
                }
            }
        };

        Ok(device)
    }

    /// Notify a queue that some resource has been freed.
    pub fn notify_free(&self, device: &Device) -> Result<(), QueueError> {
        self.queues
            .get(device)
            .ok_or(QueueError::NoSuchDevice)?
            .notify_free()
    }

    /// List every device found.
    pub fn all_devices() -> Vec<Device> {
        memonitor::list_all_devices()
            .iter()
            .map(|device| match device.backend_name() {
                memonitor::CPU_NAME => Device::CPU,
                memonitor::VULKAN_NAME => Device::Vulkan(device.local_id()),
                &_ => unimplemented!(),
            })
            .collect()
    }

    /// Register all the provided resource users in the appropriate queues.
    pub fn register_users(&self, users: Vec<(Device, Box<dyn ResourceUser>)>) {
        for (device, user) in users {
            if let Some(queue) = self.queues.get(&device) {
                queue.register_user(user);
            }
        }
    }
}

/// An abstraction over a backend device.
///
/// A backend in this context, is a device API, like Vulkan, CUDA or Metal.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Device {
    CPU,
    Vulkan(usize),
    Cuda(usize),
    Metal(usize),
    Any,
}

impl Device {
    /// Return the device's id relative to its backend (matches the CUDA device index, Vulkan's order of finding
    /// devices, etc).
    pub fn id(&self) -> usize {
        match self {
            Device::Vulkan(id) => *id,
            Device::Cuda(id) => *id,
            Device::Metal(id) => *id,
            _ => 0,
        }
    }

    /// Return the name of a device provided its id in the backend, the backend name and a default name for the case
    /// where the backend wasn't found.
    fn get_name(local_id: usize, backend_name: &str, default_name: &str) -> String {
        let mut name = default_name.to_string();
        for backend in memonitor::list_backends().iter() {
            if backend.name() == backend_name {
                let id = backend.device_ids()[local_id];
                name = memonitor::list_all_devices()[id].name().to_string();
                break;
            }
        }
        name
    }

    /// Return the name of the device.
    pub fn name(&self) -> String {
        match self {
            Device::CPU => memonitor::list_all_devices()[0].name().to_string(),
            Device::Vulkan(local_id) => {
                Self::get_name(*local_id, memonitor::VULKAN_NAME, "VULKAN_NOT_FOUND")
            }
            Device::Cuda(local_id) => Self::get_name(*local_id, "TODO", "CUDA_NOT_FOUND"),
            Device::Metal(local_id) => Self::get_name(*local_id, "TODO", "METAL_NOT_FOUND"),
            Device::Any => "NONE".to_string(),
        }
    }
}

/// A request queue for a device.
///
/// Request execution is throttled if the memory limit of the queue's device is reached, to avoid *OOM* errors.
struct Queue {
    tx: UnboundedSender<QueueItem>,
    free_tx: UnboundedSender<()>,
    thread: JoinHandle<()>,
    transient_memory: Arc<AtomicUsize>,
    device: Device,
    device_id: usize,
    max_memory: usize,
}

impl Queue {
    /// Create a new request queue given a device, the device's index for [`memonitor::list_all_devices`] and the
    /// device's memory capacity.
    fn new(device: Device, device_id: usize, max_memory: usize) -> Self {
        let (tx, rx) = unbounded_channel();
        let (free_tx, free_rx) = unbounded_channel();
        let transient_memory = Arc::new(AtomicUsize::new(0));
        let thread = spawn(run_queue(rx, free_rx, device_id, transient_memory.clone()));

        Self {
            tx,
            free_tx,
            thread,
            transient_memory,
            device,
            device_id,
            max_memory,
        }
    }

    /// Enqueue a request.
    ///
    /// This function will only return once the request has reached the end of the queue, when it is safe to be
    /// executed without running out of memory.
    ///
    /// # Returns
    ///
    /// A [`Ticket`] necessary to call a generation function.
    async fn enqueue(&self, required_memory: usize) -> Result<Ticket, QueueError> {
        let (os_tx, os_rx) = oneshot::channel();
        let waiter = QueueItem::Normal {
            required_memory,
            signal_tx: os_tx,
        };
        self.tx
            .send(waiter)
            .map_err(move |e| QueueError::Closed(e.to_string()))?;
        os_rx
            .await
            .map_err(move |e| QueueError::Enqueue(e.to_string()))?;

        self.transient_memory
            .fetch_add(required_memory, Ordering::SeqCst);
        let ticket = Ticket {
            content: Some(TicketContent::Regular {
                ticket_memory: required_memory,
                transient_memory: self.transient_memory.clone(),
            }),
            device: self.device,
        };

        Ok(ticket)
    }

    /// Notify the queue that a resource has been freed on its device.
    fn notify_free(&self) -> Result<(), QueueError> {
        self.free_tx
            .send(())
            .map_err(|e| QueueError::Closed(e.to_string()))
    }

    /// Register a resource user for this device.
    fn register_user(&self, user: Box<dyn ResourceUser>) {
        if let Err(e) = self.tx.send(QueueItem::RegisterUser(user)) {
            error!("Failed to register resource user: {e}")
        }
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        self.thread.abort();
    }
}

/// The main loop for a queue.
async fn run_queue(
    mut rx: UnboundedReceiver<QueueItem>,
    mut free_rx: UnboundedReceiver<()>,
    device_id: usize,
    transient_memory: Arc<AtomicUsize>,
) {
    let mut users = vec![];
    while let Some(request) = rx.recv().await {
        match request {
            QueueItem::Normal {
                required_memory,
                signal_tx,
            } => {
                // Empty free alert channel
                while let Ok(()) = free_rx.try_recv() {}

                let mut interval = tokio::time::interval(Duration::from_millis(2000));
                interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

                let mut available = memonitor::list_all_devices()[device_id]
                    .current_memory_stats()
                    .available
                    - transient_memory.load(Ordering::SeqCst);
                while available < required_memory {
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

                    let mut freed = 0;
                    for (id, _) in ordered_ids {
                        if required_memory - available < freed {
                            break;
                        }
                        freed += users[id]
                            .request_memory(required_memory - available - freed)
                            .await;
                    }

                    select! {
                        Some(_) = free_rx.recv() => {},
                        _ = interval.tick() => {},
                    }

                    available = memonitor::list_all_devices()[device_id]
                        .current_memory_stats()
                        .available
                        - transient_memory.load(Ordering::SeqCst);
                }
                let _ = signal_tx.send(());
            }
            QueueItem::RegisterUser(user) => users.push(user),
            QueueItem::Close => break,
        }
    }
}

/// The objects passed into [`Queue`]s.
enum QueueItem {
    /// A standard generation request to be executed.
    Normal {
        required_memory: usize,
        signal_tx: oneshot::Sender<()>,
    },
    /// A request to register a new resource user.
    RegisterUser(Box<dyn ResourceUser>),
    /// A request to close the queue.
    Close,
}

#[derive(Debug)]
/// The requirements to execute a request.
///
/// This should be acquired by querying a generation endpoint.
pub struct Passport {
    /// A basic description of the request.
    request: Request,
    /// The device the request should be executed on.
    device: Device,
}

impl Passport {
    /// Create a new passport containing a requests requirements.
    pub fn new(request: Request, device: Device) -> Self {
        Self { request, device }
    }
}

/// A basic description of a request.
#[derive(Debug, Eq, PartialEq)]
pub enum Request {
    /// A complex request that may require multiple allocations, where estimating some allocations depend on some
    /// resource already being allocated.
    Model(usize),
    /// A normal request.
    Regular(usize),
    /// A request that does not require any allocations.
    Free,
}

/// A "ticket" that allows a request to be executed.
///
/// This object should be required as an argument for all generations functions.
#[derive(Debug)]
pub struct Ticket {
    /// The inner contents of the ticket.
    content: Option<TicketContent>,
    /// The device the request executes on.
    device: Device,
}

impl Ticket {
    /// Consume the ticket, signaling that the required resources for the request have been allocated.
    pub fn consume(&mut self) -> bool {
        self.content.take().is_some()
    }

    /// Return the device this ticket is valid for.
    pub fn device(&self) -> Device {
        self.device
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
#[derive(Debug)]
enum TicketContent {
    Regular {
        ticket_memory: usize,
        transient_memory: Arc<AtomicUsize>,
    },
    Free,
}

impl Drop for TicketContent {
    fn drop(&mut self) {
        if let TicketContent::Regular {
            ticket_memory,
            transient_memory,
        } = self
        {
            transient_memory.fetch_sub(*ticket_memory, Ordering::SeqCst);
        }
    }
}

/// A resource user, which can coordinate with the request manager.
#[async_trait::async_trait]
pub trait ResourceUser: Send {
    /// This user's current number of allocations.
    fn allocs(&self) -> usize;

    /// Request that the user frees some memory.
    ///
    /// The user is not forced to free anything, but it should whenever it can.
    ///
    /// # Parameters
    ///
    /// * `memory` - The minimum amount of memory to be freed.
    ///
    /// # Returns
    ///
    /// The amount of memory the user was able to free.
    async fn request_memory(&self, memory: usize) -> usize;
}
