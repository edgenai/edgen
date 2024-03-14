use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use thiserror::Error;
use tokio::spawn;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::sleep;

#[derive(Serialize, Error, Debug)]
enum QueueError {
    #[error("the queue has already been closed")]
    Closed(String),
    #[error("an error has occurred while waiting in queue {0}")]
    Enqueue(String),
    #[error("cannot fulfill request, model does not fit in memory")]
    Unfulfillable,
}

struct RequestManager {
    queues: DashMap<Device, Arc<Queue>>,
}

impl RequestManager {
    fn new() -> Self {
        memonitor::init();

        let queues = DashMap::new();

        for device in memonitor::list_all_devices().iter() {
            let backend = match device.backend_name() {
                memonitor::CPU_NAME => Device::CPU,
                memonitor::VULKAN_NAME => Device::Vulkan(device.local_id()),
                &_ => {
                    unimplemented!()
                }
            };

            let queue = Queue::new(device.global_id(), device.current_memory_stats().total);
            queues.insert(backend, Arc::new(queue));
        }

        Self { queues }
    }

    pub async fn enqueue(&self, requirements: Passport) -> Result<Ticket, QueueError> {
        let required_memory = match requirements.request {
            Request::Model(required_memory) => (required_memory as f64 * 1.2) as usize,
            Request::Regular(required_memory) => (required_memory as f64 * 1.1) as usize,
            Request::Free => {
                return Ok(Ticket {
                    content: Some(TicketContent::Free),
                })
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
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    Vulkan(usize),
    Cuda(usize),
    Metal(usize),
    Any,
}

struct Queue {
    tx: UnboundedSender<QueueWaiter>,
    thread: JoinHandle<()>,
    transient_memory: Arc<AtomicUsize>,
    max_memory: usize,
}

impl Queue {
    fn new(device_id: usize, max_memory: usize) -> Self {
        let (tx, rx) = unbounded_channel();
        let transient_memory = Arc::new(AtomicUsize::new(0));
        let thread = spawn(run_queue(rx, device_id, transient_memory.clone()));

        Self {
            tx,
            thread,
            transient_memory,
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

        let ticket = Ticket {
            content: Some(TicketContent::Regular {
                ticket_memory: required_memory,
                transient_memory: self.transient_memory.clone(),
            }),
        };

        Ok(ticket)
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        self.thread.abort();
    }
}

async fn run_queue(
    mut rx: UnboundedReceiver<QueueWaiter>,
    device_id: usize,
    transient_memory: Arc<AtomicUsize>,
) {
    while let Some(request) = rx.recv().await {
        match request {
            QueueWaiter::Normal {
                required_memory,
                signal_tx,
            } => {
                while memonitor::list_all_devices()[device_id]
                    .current_memory_stats()
                    .available
                    - transient_memory.load(Ordering::SeqCst)
                    < required_memory
                {
                    sleep(Duration::from_millis(1000)).await; // TODO this is a terrible way to wait for memory to be freed
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

pub struct Passport {
    request: Request,
    device: Device,
}

impl Passport {
    pub fn new(request: Request, device: Device) -> Self {
        Self { request, device }
    }
}

#[derive(Eq, PartialEq)]
pub enum Request {
    Model(usize),
    Regular(usize),
    Free,
}

pub struct Ticket {
    content: Option<TicketContent>,
}

impl Ticket {
    pub fn consume(&mut self) -> bool {
        self.content.take().is_some()
    }
}

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
