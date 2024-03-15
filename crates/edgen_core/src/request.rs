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
use tracing::warn;

use crate::settings::{DevicePolicy, SETTINGS};

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

pub struct RequestManager {
    queues: DashMap<Device, Arc<Queue>>,
}

impl RequestManager {
    fn new() -> Self {
        memonitor::init();

        let queues = DashMap::new();

        for hw_device in memonitor::list_all_devices().iter() {
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

    pub fn notify_free(&self, device: &Device) -> Result<(), QueueError> {
        self.queues
            .get(device)
            .ok_or(QueueError::NoSuchDevice)?
            .notify_free()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Device {
    CPU,
    Vulkan(usize),
    Cuda(usize),
    Metal(usize),
    Any,
}

struct Queue {
    tx: UnboundedSender<QueueWaiter>,
    free_tx: UnboundedSender<()>,
    thread: JoinHandle<()>,
    transient_memory: Arc<AtomicUsize>,
    device: Device,
    device_id: usize,
    max_memory: usize,
}

impl Queue {
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

    async fn enqueue(&self, required_memory: usize) -> Result<Ticket, QueueError> {
        let (os_tx, os_rx) = oneshot::channel();
        let waiter = QueueWaiter::Normal {
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

    fn notify_free(&self) -> Result<(), QueueError> {
        self.free_tx
            .send(())
            .map_err(|e| QueueError::Closed(e.to_string()))
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        self.thread.abort();
    }
}

async fn run_queue(
    mut rx: UnboundedReceiver<QueueWaiter>,
    mut free_rx: UnboundedReceiver<()>,
    device_id: usize,
    transient_memory: Arc<AtomicUsize>,
) {
    while let Some(request) = rx.recv().await {
        match request {
            QueueWaiter::Normal {
                required_memory,
                signal_tx,
            } => {
                // Empty free alert channel
                while let Ok(()) = free_rx.try_recv() {}

                let mut interval = tokio::time::interval(Duration::from_millis(2000));
                interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

                while memonitor::list_all_devices()[device_id]
                    .current_memory_stats()
                    .available
                    - transient_memory.load(Ordering::SeqCst)
                    < required_memory
                {
                    if signal_tx.is_closed() {
                        break;
                    }

                    select! {
                        Some(_) = free_rx.recv() => {},
                        _ = interval.tick() => {},
                    }
                }
                let _ = signal_tx.send(());
            }
            QueueWaiter::Close => break,
        }
    }
}

enum QueueWaiter {
    Normal {
        required_memory: usize,
        signal_tx: oneshot::Sender<()>,
    },
    Close,
}

#[derive(Debug)]
pub struct Passport {
    request: Request,
    device: Device,
}

impl Passport {
    pub fn new(request: Request, device: Device) -> Self {
        Self { request, device }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum Request {
    Model(usize),
    Regular(usize),
    Free,
}

#[derive(Debug)]
pub struct Ticket {
    content: Option<TicketContent>,
    device: Device,
}

impl Ticket {
    pub fn consume(&mut self) -> bool {
        self.content.take().is_some()
    }

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
