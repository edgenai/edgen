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

//! A brutally simple session management framework for Edgen's transport-agnostic, Protobuf-based
//! message-passing protocol.
//!
//! The heart of this crate is the [`Server`], which can open many [`Session`]s backed by an
//! arbitrary [`EnvelopeHandler`].

extern crate alloc;
/*
use alloc::sync::Arc;

use dashmap::DashMap;
use smol::channel::{Receiver, Sender};
use smol::future::Future;
use smol::lock::{OnceCell, Semaphore};
use time::OffsetDateTime;
use tracing::info;
use uuid::Uuid;

use edgen_proto::Envelope;
*/
pub mod llm;
pub mod whisper;

pub mod settings;

pub mod perishable;

/*

/// A [sans-IO][sans-io], fully-asynchronous, task-oriented server for routing [`Envelope`]s to and
/// from [`Session`]s.
///
/// [sans-io]: https://sans-io.readthedocs.io/
#[derive(Clone)]
pub struct Server {
    _inner: Arc<ServerInner>,
}

impl Server {
    /// Opens a new session on this server, dispatching incoming envelopes to the given handler.
    pub fn open_session_with_handler(&self, _handler: impl EnvelopeHandler) -> Session {
        let session_id = Uuid::new_v4();

        info!("New virtual session: {}", session_id);
        todo!();
        /*let session = Session {
            inner: Arc::new(SessionState {
                uuid: session_id,
                bound_server: self.clone(),
                task_semaphore: Arc::new(Semaphore::new(Session::MAX_CONCURRENT_TASKS)),
            })
        };

        self.inner.open_sessions.insert(session_id, session.clone());

        session*/
    }

    /// Removes a session from this server.
    #[allow(unused)]
    pub(crate) fn drop_session(&self, session_id: Uuid) {
        self._inner._open_sessions.remove(&session_id);
    }
}

struct ServerInner {
    _open_sessions: DashMap<Uuid, Session, ahash::RandomState>,
}

struct SessionState {
    uuid: Uuid,

    /// The server that contains this session.
    _bound_server: Server,

    /// A semaphore limiting the number of concurrent asynchronous tasks that this session can
    /// spawn to [`Session::MAX_CONCURRENT_TASKS`].
    _task_semaphore: Arc<Semaphore>,

    /// When set, the timestamp at which the session was closed. When present, downstream tasks
    /// should gracefully terminate.
    dead_from: OnceCell<OffsetDateTime>,

    /// The channel from the main thread to the primary task thread for this session.
    process_tx: Sender<Envelope>,

    /// The channel from the primary task thread for this session to the main thread.
    outgoing_rx: Receiver<Envelope>,

    /// If present, bound callback handlers waiting for one or more messages on a given channel.
    _bound_channels: DashMap<Uuid, Sender<Envelope>, ahash::RandomState>,
}

/// A transport-agnostic session within a [`Server`].
///
/// Typically, you'll want to wire this into some other sender/receiver pair for
/// [`Envelope`][edgen_proto::Envelope]s, such as a WebSocket or HTTP connection, via
/// [`Session::receive_incoming`][Session::receive_incoming] and [`Session::next_outgoing`].
///
/// When this structure is `Drop`ped, all downstream tasks associated with it are cancelled;
/// **the server only maintains a weak reference to session channels, and this structure is the only
/// thing actually keeping the session alive.**
pub struct Session {
    /// The shared inner state of this session.
    inner: Arc<SessionState>,
}

impl Session {
    /// The maximum number of concurrent tasks this session can have outstanding on its [`Server`].
    ///
    /// Beyond this limit, new requests may start to block as old ones finish.
    pub const MAX_CONCURRENT_TASKS: usize = 32;

    /// Returns the UUID of this session.
    pub fn id(&self) -> Uuid {
        self.inner.uuid
    }

    /// Resolves, eventually, to the next outgoing [`Envelope`] for this session.
    ///
    /// This envelope is outbound from this session, and is destined for another peer.
    ///
    /// Resolves to `None` if the session has been dropped.
    pub async fn next_outgoing(&self) -> Option<Envelope> {
        self.inner.outgoing_rx.recv().await.ok()
    }

    /// Forwards an incoming envelope to this session's server, dispatching upstream routing
    /// handlers on a new task thread.
    ///
    /// This function may yield if [`MAX_CONCURRENT_TASKS`] would be exceeded, until at least one
    /// such task thread is available.
    pub async fn receive_incoming(&self, envelope: Envelope) {
        self.inner.process_tx.send(envelope).await.ok();
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.inner.dead_from.set(OffsetDateTime::now_utc());
    }
}

/// Whether a handler has consumed an envelope, or wishes to pass it to the next handler.
pub enum SinkMode {
    /// This handler has successfully processed the envelope, and no subsequent handlers should be
    /// invoked.
    Sink,

    /// This handler has indicated that this envelope should be passed to the next downstream
    /// handler.
    Pass(Envelope),
}

/// An asynchronous handler for incoming [`Envelope`]s.
pub trait EnvelopeHandler {
    fn handle(&self, envelope: Envelope) -> impl Future<Output=SinkMode> + Send;
}

impl<F, Fut> EnvelopeHandler for F
    where
        F: Fn(Envelope) -> Fut + Send + Sync + 'static,
        Fut: Future<Output=SinkMode> + Send + 'static,
{
    #[inline]
    fn handle(&self, envelope: Envelope) -> impl Future<Output=SinkMode> + Send {
        (self)(envelope)
    }
}

pub struct OrElse<A, B> {
    a: A,
    b: B,
}

impl<A, B> EnvelopeHandler for OrElse<A, B> where A: EnvelopeHandler, B: EnvelopeHandler, OrElse<A, B>: Send + Sync + 'static {
    #[inline]
    fn handle(&self, envelope: Envelope) -> impl Future<Output=SinkMode> + Send {
        async move {
            match self.a.handle(envelope).await {
                SinkMode::Pass(envelope) => self.b.handle(envelope).await,
                SinkMode::Sink => SinkMode::Sink,
            }
        }
    }
}

pub trait EnvelopeHandlerExt: EnvelopeHandler {
    #[inline]
    fn or_else<B>(self, other: B) -> OrElse<Self, B> where Self: Sized, B: EnvelopeHandler {
        OrElse { a: self, b: other }
    }
}

impl<T> EnvelopeHandlerExt for T where T: EnvelopeHandler {}

#[cfg(test)]
mod test {
    use super::*;
}

*/
