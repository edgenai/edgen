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
 
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures::executor::block_on;
use serde::Serialize;
use smol::future::Future;
use thiserror::Error;
use tokio::spawn;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use utoipa::ToSchema;

#[derive(Serialize, Error, ToSchema, Debug)]
pub enum WhisperError {
    #[error("{mime:?} mime is unsupported, this executor supports: {supported:?}")]
    UnsupportedMime { mime: String, supported: String },
    #[error("audio has unsupported sample rate {value:?}, executor supports: {supported:?}")]
    UnsupportedSampleRate { value: u32, supported: String },
    #[error("could not parse input data: {0}")]
    Parsing(String),
    #[error("failed to run the internal executor: {0}")]
    Internal(String),
    #[error("failed to load model: {0}")]
    ModelInitialization(String),
    #[error("failed to create a new session: {0}")]
    SessionInitialization(String),
    #[error("failed to prepare the execution: {0}")]
    Other(String),
}

pub trait Whisper {
    fn decode<'a>(
        &'a self,
        data: &'a [f32],
    ) -> Box<dyn Future<Output = Result<String, WhisperError>> + Send + Unpin + 'a>;

    fn new_session<'a>(
        &'a self,
        callback: Box<dyn Fn(String) + Send + 'static>,
    ) -> Box<dyn Future<Output = Result<WhisperSession, WhisperError>> + Send + Unpin + 'a>;
}

pub trait WhisperRunner {
    fn forward_decode(
        &mut self,
        data: &[f32],
    ) -> impl Future<Output = Result<String, WhisperError>> + Send;
}

#[derive(Serialize, Error, ToSchema, Debug)]
pub enum DecodeSessionError {
    #[error("the session has already been closed or aborted")]
    SessionOver,
    #[error("failed to send input to worker thread: {0}")]
    Send(String),
    #[error("error occurred in the session runner: {0}")]
    SessionRunner(#[from] SessionRunnerError),
    #[error("failed to join runner thread: {0}")]
    Join(String),
}

pub struct WhisperSession {
    current_result: Arc<RwLock<String>>,
    tx: Option<UnboundedSender<Vec<f32>>>,
    rx: Option<UnboundedReceiver<f32>>,
    runner: Option<JoinHandle<Result<(), SessionRunnerError>>>,
    closed: Arc<AtomicBool>,
}

// TODO turn session into a receiver-transmitter pair
impl WhisperSession {
    /// Creates a new [`WhisperSession`].
    ///
    /// ## Arguments
    /// * `runner` - A *Whisper* backend instance that implements [`WhisperRunner`]
    /// * `callback` - A callback function (of type [`Fn(String)`]) that is called for every item
    /// processed, receiving the result of the item as input
    pub fn new<F>(runner: impl WhisperRunner + Send + 'static, callback: F) -> Self
    where
        F: Fn(String) + Send + 'static,
    {
        let (tx, rx) = mpsc::unbounded_channel();
        let (status_tx, status_rx) = mpsc::unbounded_channel();
        let mut session = Self {
            current_result: Arc::new(RwLock::new("".to_string())),
            tx: Some(tx),
            rx: Some(status_rx),
            runner: None,
            closed: Arc::new(AtomicBool::new(false)),
        };

        session.runner = Some(spawn(run_session(
            runner,
            rx,
            status_tx,
            callback,
            session.current_result.clone(),
            session.closed.clone(),
        )));

        session
    }

    /// Push an item to the session queue to be processed.
    ///
    /// Does nothing if the provided slice is empty.
    ///
    /// ## Arguments
    /// * `data` - A [`f32`] normalized slice of *PCM* data to be pushed to the queue
    pub fn push(&self, data: &[f32]) -> Result<(), DecodeSessionError> {
        if data.is_empty() {
            return Ok(());
        }

        self.unchecked_push(data)
    }

    /// Push an item to the session queue to be processed.
    ///
    /// ## Arguments
    /// * `data` -A [`f32`] normalized slice of *PCM* data to be pushed to the queue
    fn unchecked_push(&self, data: &[f32]) -> Result<(), DecodeSessionError> {
        if let Some(tx) = &self.tx {
            tx.send(data.to_vec())
                .map_err(move |e| DecodeSessionError::Send(e.to_string()))?;
            Ok(())
        } else {
            Err(DecodeSessionError::SessionOver)
        }
    }

    /// Returns the current cumulative [`String`] result of the session.
    pub async fn result(&self) -> String {
        let locked = self.current_result.read().await;
        locked.clone()
    }

    /// Wait for every item submitted to the session queue up to this point to processed.
    pub async fn sync(&mut self) -> Result<(), DecodeSessionError> {
        if self.closed() {
            return Err(DecodeSessionError::SessionOver);
        }

        let empty: Vec<f32> = vec![];
        self.unchecked_push(&empty)?;

        if let Some(rx) = &mut self.rx {
            let _ = rx.recv().await;
        } else {
            // Should never happen, since we check if the session is already closed above.
            return Err(DecodeSessionError::SessionOver);
        }

        Ok(())
    }

    /// Terminates the session and waits for the remaining items in the queue to be processed and
    /// joining with the worker thread.
    pub async fn end(&mut self) -> Result<(), DecodeSessionError> {
        {
            let _ = self.rx.take();
        }

        let empty: Vec<f32> = vec![];
        self.unchecked_push(&empty)?;

        if let Some(tx) = self.tx.take() {
            tx.closed().await;
        }

        if let Some(runner) = self.runner.take() {
            runner
                .await
                .map_err(move |e| DecodeSessionError::Join(e.to_string()))??;
        }

        Ok(())
    }

    /// Abruptly ends the session, joining with the worker thread as soon as possible. Any remaining
    /// items in the queue are ignored.
    pub async fn abort(&mut self) -> Result<(), DecodeSessionError> {
        if self.closed() {
            return Err(DecodeSessionError::SessionOver);
        }

        self.closed.store(true, Ordering::Relaxed);

        if let Some(tx) = self.tx.take() {
            // Make sure that the runner iterates at least once more
            tx.send(vec![])
                .map_err(move |e| DecodeSessionError::Send(e.to_string()))?;
        }

        if let Some(runner) = self.runner.take() {
            runner
                .await
                .map_err(move |e| DecodeSessionError::Join(e.to_string()))??;
        }

        Ok(())
    }

    /// Checks if the session has been terminated.
    pub fn closed(&self) -> bool {
        self.tx.is_none()
            || self.rx.is_none()
            || self.runner.is_none()
            || self.closed.load(Ordering::Relaxed)
    }
}

impl Drop for WhisperSession {
    fn drop(&mut self) {
        let e = block_on(self.abort());
        match e {
            // Nothing to do, session got terminated previously
            Err(DecodeSessionError::SessionOver) => { /* nothing */ }
            Err(DecodeSessionError::Send(_)) => {
                println!("Failed to send end signal: {e:?}");
            }
            Err(DecodeSessionError::SessionRunner(_)) => {
                println!("Error occurred while session was running: {e:?}");
            }
            Err(DecodeSessionError::Join(_)) => {
                // todo should this panic?
                println!("Failed to join runner thread: {e:?}");
            }
            _ => { /* nothing */ }
        }
    }
}

#[derive(Serialize, Error, ToSchema, Debug)]
pub enum SessionRunnerError {
    #[error("could not run the executor: {0}")]
    Executor(#[from] WhisperError),
}

/// The runtime of the session runner thread.
///
/// ## Arguments
/// * `runner` -
/// * `rx` -
/// * `tx` -
/// * `callback` -
/// * `current_result` -  
/// * `closed` -
async fn run_session<R, F>(
    mut runner: R,
    mut rx: UnboundedReceiver<Vec<f32>>,
    tx: UnboundedSender<f32>,
    callback: F,
    current_result: Arc<RwLock<String>>,
    closed: Arc<AtomicBool>,
) -> Result<(), SessionRunnerError>
where
    R: WhisperRunner,
    F: Fn(String),
{
    while let Some(data) = rx.recv().await {
        if closed.load(Ordering::Relaxed) {
            rx.close();
            break;
        }

        if data.is_empty() {
            if tx.send(0.0).is_err() {
                // session.end() was called, probably
                break;
            } else {
                continue;
            }
        }

        let segment = runner.forward_decode(&data).await?;
        {
            let mut locked = current_result.write().await;
            let len = locked.len();
            locked.insert_str(len, &segment);
        }
        callback(segment);
    }

    closed.store(true, Ordering::Relaxed);
    rx.close();

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::future::Future;

    use thiserror::Error;

    use crate::whisper::{
        DecodeSessionError, Whisper, WhisperError, WhisperRunner, WhisperSession,
    };

    #[derive(Error, Debug)]
    enum TestError {
        #[error("whisper error: {0}")]
        Whisper(#[from] WhisperError),
        #[error("decode session error: {0}")]
        Session(#[from] DecodeSessionError),
    }

    struct TestWhisper {}

    impl TestWhisper {
        async fn async_decode(&self, _data: &[f32]) -> Result<String, WhisperError> {
            Ok("decode".to_string())
        }

        async fn async_new_session(
            &self,
            callback: Box<dyn Fn(String) + Send + 'static>,
        ) -> Result<WhisperSession, WhisperError> {
            Ok(WhisperSession::new(TestRunner {}, callback))
        }
    }

    impl Whisper for TestWhisper {
        fn decode<'a>(
            &'a self,
            data: &'a [f32],
        ) -> Box<dyn Future<Output = Result<String, WhisperError>> + Send + Unpin + 'a> {
            let fut = Box::pin(self.async_decode(data));
            Box::new(fut)
        }

        fn new_session<'a>(
            &'a self,
            callback: Box<dyn Fn(String) + Send + 'static>,
        ) -> Box<dyn Future<Output = Result<WhisperSession, WhisperError>> + Send + Unpin + 'a>
        {
            let fut = Box::pin(self.async_new_session(callback));
            Box::new(fut)
        }
    }

    struct TestRunner {}

    impl WhisperRunner for TestRunner {
        async fn forward_decode(&mut self, _data: &[f32]) -> Result<String, WhisperError> {
            Ok("forward".to_string())
        }
    }

    #[tokio::test]
    async fn decode() -> Result<(), TestError> {
        let test = TestWhisper {};

        let e: Vec<f32> = vec![];
        let res = test.decode(&e).await?;

        assert_eq!(res, "decode".to_string());

        Ok(())
    }

    #[tokio::test]
    async fn session() -> Result<(), TestError> {
        let test = TestWhisper {};

        let mut session = test
            .new_session(Box::new(move |e| assert_eq!(e, "forward".to_string())))
            .await?;

        let e: Vec<f32> = vec![1.0];

        for _ in 0..3 {
            session.push(&e)?;
        }
        session.sync().await?;
        assert_eq!(session.result().await, "forwardforwardforward".to_string());

        for _ in 0..2 {
            session.push(&e)?;
        }
        session.end().await?;
        assert_eq!(
            session.result().await,
            "forwardforwardforwardforwardforward".to_string()
        );

        Ok(())
    }
}
