use serde::Serialize;
use thiserror::Error;
use tokio::spawn;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

#[derive(Serialize, Error, Debug)]
enum QueueError {
    #[error("the queue has already been closed")]
    Closed(String),
    #[error("an error has occurred while waiting in queue {0}")]
    Enqueue(String),
}

enum Request {
    Generate {
        size: usize,
        tx: oneshot::Sender<()>,
    },
    Close,
}

struct RequestQueue {
    tx: UnboundedSender<Request>,
    thread: JoinHandle<()>,
}

impl RequestQueue {
    fn new() -> Self {
        let (tx, rx) = unbounded_channel();
        let thread = spawn(run(rx));

        Self { tx, thread }
    }

    pub async fn enqueue(&self, allocation_size: usize) -> Result<(), QueueError> {
        let (os_tx, os_rx) = oneshot::channel();
        let request = Request::Generate {
            size: allocation_size,
            tx: os_tx,
        };
        self.tx
            .send(request)
            .map_err(move |e| QueueError::Closed(e.to_string()))?;
        os_rx
            .await
            .map_err(move |e| QueueError::Enqueue(e.to_string()))?;
        Ok(())
    }
}

impl Drop for RequestQueue {
    fn drop(&mut self) {
        self.thread.abort();
    }
}

async fn run(mut rx: UnboundedReceiver<Request>) {
    while let Some(request) = rx.recv().await {
        match request {
            Request::Generate { .. } => {}
            Request::Close => break,
        }
    }
}
