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

use std::mem::take;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use blake3::Hasher;
use dashmap::DashMap;
use futures::executor::block_on;
use futures::Stream;
use llama_cpp::standard_sampler::StandardSampler;
use llama_cpp::{
    CompletionHandle, EmbeddingsParams, LlamaModel, LlamaParams, LlamaSession, SessionParams,
    TokensToStrings,
};
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::task::JoinHandle;
use tokio::time::{interval, Instant, MissedTickBehavior};
use tokio::{fs, select, spawn};
use tracing::{error, info, warn};

use edgen_core::cleanup_interval;
use edgen_core::llm::{
    default_context_settings, inactive_llm_session_ttl, inactive_llm_ttl, CompletionArgs,
    LLMEndpoint, LLMEndpointError, ASSISTANT_TAG, SYSTEM_TAG, TOOL_TAG, USER_TAG,
};
use edgen_core::perishable::{ActiveSignal, Perishable, PerishableReadGuard, PerishableWriteGuard};
use edgen_core::request::{
    DeviceId, FreedMemory, Passport, Request, ResourceUser, Ticket, REQUEST_QUEUE,
};
use edgen_core::settings::SETTINGS;

// TODO this should be in settings
const SINGLE_MESSAGE_LIMIT: usize = 4096;

/// A large language model endpoint, implementing [`LLMEndpoint`] using a [`llama_cpp`] backend.
pub struct LlamaCppEndpoint {
    /// A map of the models currently loaded into memory, with their path as the key.
    models: Arc<DashMap<ModelKey, UnloadingModel>>,

    /// A background thread that periodically removes models from the `models` collection, if they
    /// are not loaded at the time.
    cleanup_thread: JoinHandle<()>,
}

impl LlamaCppEndpoint {
    /// Gets the [`UnloadingModel`] loaded from the specified path. If the model isn't already
    /// loaded, first initialise it and add it to the `models` collection.
    async fn get(
        &self,
        model_path: impl AsRef<Path> + Send,
        device: DeviceId,
    ) -> Result<dashmap::mapref::one::Ref<ModelKey, UnloadingModel>, LLMEndpointError> {
        let key = ModelKey::new(&model_path, device);

        if !self.models.contains_key(&key) {
            let model = UnloadingModel::new(model_path, device).await?;
            self.models.insert(key.clone(), model);
        }

        // PANIC SAFETY: Just inserted the element if it isn't already inside the map, so must be present in the map
        Ok(self.models.get(&key).unwrap())
    }
}

#[async_trait::async_trait]
impl LLMEndpoint for LlamaCppEndpoint {
    async fn chat_completions(
        &self,
        model_path: impl AsRef<Path> + Send + Sync,
        prompt: &str,
        args: &CompletionArgs,
        mut ticket: Ticket,
    ) -> Result<String, LLMEndpointError> {
        let model = self.get(&model_path, ticket.device()).await?;
        model.staging_check(prompt, &args, &mut ticket).await?;
        model.chat_completions(prompt, args.clone(), ticket).await
    }

    async fn stream_chat_completions(
        &self,
        model_path: impl AsRef<Path> + Send + Sync,
        prompt: &str,
        args: &CompletionArgs,
        mut ticket: Ticket,
    ) -> Result<Box<dyn Stream<Item = String> + Unpin + Send>, LLMEndpointError> {
        let model = self.get(&model_path, ticket.device()).await?;
        model.staging_check(prompt, &args, &mut ticket).await?;
        model
            .stream_chat_completions(prompt, args.clone(), ticket)
            .await
    }

    async fn embeddings(
        &self,
        model_path: impl AsRef<Path> + Send + Sync,
        inputs: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, LLMEndpointError> {
        let model = self.get(model_path, DeviceId::CPU).await?;
        model.embeddings(inputs).await
    }

    async fn completion_requirements(
        &self,
        model_path: impl AsRef<Path> + Send + Sync,
        device: DeviceId,
        prompt: &str,
        args: &CompletionArgs,
    ) -> Result<Passport, LLMEndpointError> {
        let key = ModelKey::new(&model_path, device);

        // If mmap is enabled by default, the model won't occupy a relevant amount of space in
        // memory, so it can just be instantiated immediately.
        if let Some(model) = self.models.get(&key) {
            if model.loaded().await
                || (model.device == DeviceId::CPU && LlamaParams::default().use_mmap)
            {
                model.completion_requirements(prompt, args).await
            } else {
                let (host_size, device_size) = if model.device == DeviceId::CPU {
                    (file_size(model_path).await?, 0)
                } else {
                    (0, file_size(model_path).await?)
                };

                Ok(Passport::new(
                    Request::Staged {
                        host_memory: host_size,
                        device_memory: device_size,
                    },
                    device,
                ))
            }
        } else {
            match device {
                DeviceId::CPU if LlamaParams::default().use_mmap => {
                    let model = self.get(model_path, device).await?;
                    model.completion_requirements(prompt, args).await
                }
                DeviceId::Any => {
                    // TODO look for loaded models with this path for every device

                    let size = file_size(&model_path).await?;
                    let use_mmap = LlamaParams::default().use_mmap;
                    let device_pick = REQUEST_QUEUE
                        .pick_device(|d| {
                            match d {
                                DeviceId::CPU => {
                                    if use_mmap {
                                        (0, 0)
                                    } else {
                                        (size, 0)
                                    }
                                }
                                // DeviceId::Vulkan(_) => {}
                                DeviceId::Cuda(_) => (0, size),
                                _ => (usize::MAX, usize::MAX),
                            }
                        })
                        .await?;
                    if device_pick == DeviceId::CPU && use_mmap {
                        let model = self.get(model_path, DeviceId::CPU).await?;
                        model.completion_requirements(prompt, args).await
                    } else {
                        let (host_size, device_size) = if device_pick == DeviceId::CPU {
                            (size, 0)
                        } else {
                            (0, size)
                        };

                        Ok(Passport::new(
                            Request::Staged {
                                host_memory: host_size,
                                device_memory: device_size,
                            },
                            device_pick,
                        ))
                    }
                }
                _ => Ok(Passport::new(
                    Request::Staged {
                        host_memory: 0,
                        device_memory: file_size(model_path).await?,
                    },
                    device,
                )),
            }
        }
    }

    fn resource_user(&self) -> Box<dyn ResourceUser> {
        let resource_user = LlamaResourceUser {
            models: self.models.clone(),
        };
        Box::new(resource_user)
    }

    fn reset(&self) {
        self.models.clear();
    }
}

impl Default for LlamaCppEndpoint {
    fn default() -> Self {
        let models: Arc<DashMap<ModelKey, UnloadingModel>> = Default::default();
        let models_clone = models.clone();
        let cleanup_thread = spawn(async move {
            let mut interval = interval(cleanup_interval());
            interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

            loop {
                interval.tick().await;
                models_clone.retain(move |_, model| block_on(model.loaded()));
            }
        });

        Self {
            models,
            cleanup_thread,
        }
    }
}

impl Drop for LlamaCppEndpoint {
    fn drop(&mut self) {
        self.cleanup_thread.abort()
    }
}

/// A hashable key used to identify a model instance.
///
/// To be used as the key in a map collection.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct ModelKey {
    /// The path string of the model.
    path: String,

    /// The device where the model is loaded.
    device: DeviceId,
}

impl ModelKey {
    fn new(path: impl AsRef<Path>, device: DeviceId) -> Self {
        Self {
            path: path.as_ref().to_string_lossy().to_string(),
            device,
        }
    }
}

/// Return the size of a file, given its path, retrieved from its metadata.
async fn file_size(path: impl AsRef<Path> + Send) -> Result<usize, LLMEndpointError> {
    if path.as_ref().is_file() {
        let model_file = fs::File::open(path)
            .await
            .map_err(|e| LLMEndpointError::Load(format!("Failed to open model file: {e}")))?;
        if let Err(e) = model_file.sync_all().await {
            warn!("Failed to sync file metadata: {e}");
        }
        let metadata = model_file
            .metadata()
            .await
            .map_err(|e| LLMEndpointError::Load(format!("Failed to read file metadata: {e}")))?;

        Ok(metadata.len() as usize)
    } else {
        Err(LLMEndpointError::Load("File not found".to_string()))
    }
}

/// A resource user for this backend.
struct LlamaResourceUser {
    /// A reference to the map with all models the Llama backend has currently loaded.
    models: Arc<DashMap<ModelKey, UnloadingModel>>,
}

#[async_trait::async_trait]
impl ResourceUser for LlamaResourceUser {
    fn allocs(&self) -> usize {
        self.models
            .iter()
            .map(|model| {
                let m = if model.key().device == DeviceId::CPU && LlamaParams::default().use_mmap {
                    0
                } else {
                    1
                };
                model.sessions() + m
            })
            .reduce(|a, b| a + b)
            .unwrap_or(0)
    }

    async fn request_memory(
        &self,
        host_memory: usize,
        device_memory: usize,
        device_id: DeviceId,
    ) -> FreedMemory {
        let mut droppable: Vec<(ModelKey, Instant)> = self
            .models
            .iter()
            .filter(|model| {
                model.key().device == device_id
                    // Filter models that only have sessions in flight
                    && !(model.sessions() > 0 && model.droppable_sessions() == 0)
            })
            .map(|model| (model.key().clone(), model.created()))
            .collect();

        // Sort by oldest to most recent
        droppable.sort_by(|a, b| a.1.cmp(&b.1));

        let mut freed = FreedMemory {
            host_memory: 0,
            device_memory: 0,
        };
        for (key, _) in droppable {
            if host_memory < freed.host_memory && device_memory < freed.device_memory {
                break;
            }

            if let Some(model) = self.models.get(&key) {
                if !(model.key().device == DeviceId::CPU && LlamaParams::default().use_mmap)
                    && model.sessions() == 0
                {
                    if model.key().device == DeviceId::CPU {
                        freed.host_memory += model.size();
                    } else {
                        freed.device_memory += model.size();
                    }
                    model.unload().await;
                } else {
                    freed += model
                        .request_memory(
                            host_memory - freed.host_memory,
                            device_memory - freed.device_memory,
                        )
                        .await;
                }
            }
        }

        info!("Llama backend has freed {freed} bytes from {device_id:?}");

        freed
    }
}

/// A [`LlamaModel`] (as well as its associated [`LlamaSession`]s) that unloads itself from memory after not being used
/// for a period of time.
struct UnloadingModel {
    /// This model's [`Perishable`] inner [`LlamaModel`].
    model: Perishable<LlamaModel>,

    /// The path to the models file.
    path: PathBuf,

    /// The device where the model is loaded in.
    device: DeviceId,

    /// A map storing this model's idle sessions.
    sessions: Arc<DashMap<SessionId, UnloadingSession>>,

    /// The background thread which cleans up old sessions and reinserts sessions coming from streams.
    maintenance_thread: JoinHandle<()>,

    /// The sender used by streams to insert sessions back into the model's session collection.
    finished_tx: UnboundedSender<(SessionId, UnloadingSession)>,

    /// The current number of sessions in flight owned by the model.
    sessions_in_flight: Arc<AtomicUsize>,

    /// The instant when the model was created.
    created: Instant,

    /// The size of the model, in bytes.
    size: usize,
}

impl UnloadingModel {
    /// Creates a new instance of this model, provided it's [`Path`].
    ///
    /// This function is lazy and does not actually load the model into system memory, the model must be accessed in
    /// order to be loaded.
    async fn new(
        model_path: impl AsRef<Path> + Send,
        device: DeviceId,
    ) -> Result<Self, LLMEndpointError> {
        let sessions: Arc<DashMap<SessionId, UnloadingSession>> = Default::default();
        let (tx, mut rx) = unbounded_channel();

        let sessions_clone = sessions.clone();
        let maintenance_thread = spawn(async move {
            let mut interval = interval(cleanup_interval());
            interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

            loop {
                select! {
                    _ = interval.tick() => sessions_clone.retain(move |_, session| block_on(session.loaded())),
                    item = rx.recv() => {
                        if let Some((id, session)) = item {
                            sessions_clone.insert(id, session);
                        }
                    }
                }
            }
        });

        let model = Perishable::with_ttl(inactive_llm_ttl());
        model
            .set_callback(Some(move || {
                if let Err(e) = REQUEST_QUEUE.notify_free(device) {
                    error!("Failed to notify request queue: {e}");
                }
            }))
            .await;

        Ok(Self {
            model,
            path: model_path.as_ref().to_path_buf(),
            device,
            sessions,
            maintenance_thread,
            finished_tx: tx,
            sessions_in_flight: Arc::new(AtomicUsize::new(0)),
            created: Instant::now(),
            size: file_size(model_path).await?,
        })
    }

    /// Returns **`true`** if this model is currently loaded in system memory, **`false`** otherwise.
    async fn loaded(&self) -> bool {
        self.model.is_alive().await
    }

    /// Unload this model from memory.
    async fn unload(&self) {
        self.model.kill().await;
    }

    /// Request for resource to be freed by the model.
    ///
    /// The model's sessions are sorted by oldest to newest and freed in order until either the memory target is met,
    /// or there are no more sessions to be freed.
    async fn request_memory(&self, host_memory: usize, device_memory: usize) -> FreedMemory {
        let mut droppable: Vec<(SessionId, Instant)> = self
            .sessions
            .iter()
            .map(|session| (session.key().clone(), session.created()))
            .collect();

        droppable.sort_by(|a, b| a.1.cmp(&b.1));

        let mut freed = FreedMemory {
            host_memory: 0,
            device_memory: 0,
        };
        for (key, _) in droppable {
            if host_memory < freed.host_memory && device_memory < freed.device_memory {
                break;
            }

            if let Some(session) = self.sessions.get(&key) {
                if let Ok(size) = session.size().await {
                    if self.device == DeviceId::CPU {
                        freed.host_memory += size;
                    } else {
                        freed.host_memory += (size * 2) / 3;
                        freed.device_memory += size / 3;
                    }
                    session.unload().await;
                } else {
                    warn!("Session is already uninitialised")
                }
            }
        }

        freed
    }

    /// Verify the ticket, if it is a staging ticket, return an error containing a passport with
    /// updated requirements, otherwise return ok.
    async fn staging_check(
        &self,
        prompt: &str,
        args: &CompletionArgs,
        ticket: &mut Ticket,
    ) -> Result<(), LLMEndpointError> {
        if ticket.staged() {
            ticket.consume();
            let passport = self.completion_requirements(prompt, args).await?;
            Err(LLMEndpointError::Retry(passport))
        } else {
            Ok(())
        }
    }

    /// Estimate and return the resources required to compute a request, given its arguments.
    async fn completion_requirements(
        &self,
        prompt: &str,
        args: &CompletionArgs,
    ) -> Result<Passport, LLMEndpointError> {
        let (key, _, _) = SessionId::chat(&prompt);
        if self.sessions.contains_key(&key) {
            Ok(Passport::new(Request::Free, self.device))
        } else {
            let params = from_completion_args(args.clone()).await;
            let (_signal, model) = self.get_or_init().await?;
            let estimated = model.estimate_session_size(&params);
            Ok(Passport::new(
                Request::Final {
                    host_memory: estimated.host_memory,
                    device_memory: estimated.device_memory,
                },
                self.device,
            ))
        }
    }

    /// Either takes an existing chat [`LlamaSession`] compatible with the provided prompt from the
    /// `sessions` collection, or creates a new one.
    ///
    /// The matching [`SessionId`] and the new context derived from `prompt` are also returned.
    async fn take_chat_session<'a>(
        &self,
        prompt: &'a str,
        args: CompletionArgs,
    ) -> Result<(UnloadingSession, SessionId, &'a str), LLMEndpointError> {
        let (id, old_context, new_context) = SessionId::chat(prompt);

        let session = if let Some((_, session)) = self.sessions.remove(&id) {
            info!("Matching session found, continuing");
            if !session.loaded().await {
                let (_signal, mut guard) = session.get_or_init(None).await?;
                guard
                    .advance_context_async(old_context)
                    .await
                    .map_err(move |e| LLMEndpointError::Advance(e.to_string()))?;
            }
            session
        } else {
            info!("No matching session found, creating new one");
            let (_signal, model) = self.get_or_init().await?;
            let session = UnloadingSession::new(args, model.clone(), self.device).await;
            {
                let (_signal, mut guard) = session.get_or_init(None).await?;
                guard
                    .advance_context_async(old_context)
                    .await
                    .map_err(move |e| LLMEndpointError::Advance(e.to_string()))?;
            }
            session
        };

        Ok((session, id, new_context))
    }

    /// Computes the full chat completions for the provided [`CompletionArgs`].
    async fn chat_completions(
        &self,
        prompt: &str,
        args: CompletionArgs,
        mut ticket: Ticket,
    ) -> Result<String, LLMEndpointError> {
        let (_model_signal, model_guard) = self.get_or_init().await?;

        if args.one_shot {
            info!("Allocating one-shot LLM session");
            let _in_flight_ref = InFlightRef::new(&self.sessions_in_flight);

            let params = from_completion_args(args).await;
            let res = {
                let mut session = model_guard
                    .create_session(params)
                    .map_err(move |e| LLMEndpointError::SessionCreationFailed(e.to_string()))?;
                ticket.consume();

                session
                    .advance_context_async(prompt)
                    .await
                    .map_err(move |e| LLMEndpointError::Advance(e.to_string()))?;

                let sampler = StandardSampler::default();
                let handle = session
                    .start_completing_with(sampler, SINGLE_MESSAGE_LIMIT)
                    .map_err(|e| LLMEndpointError::Advance(e.to_string()))?;

                handle.into_string_async().await
            };

            REQUEST_QUEUE.notify_free(self.device)?;

            Ok(res)
        } else {
            let (session, mut id, new_context) = self.take_chat_session(prompt, args).await?;

            let (_session_signal, handle) = {
                let (session_signal, mut session_guard) = session.get_or_init(Some(ticket)).await?;

                session_guard
                    .advance_context_async(new_context)
                    .await
                    .map_err(move |e| LLMEndpointError::Advance(e.to_string()))?;
                id.advance(new_context);

                let sampler = StandardSampler::default();
                let handle = session_guard
                    .start_completing_with(sampler, SINGLE_MESSAGE_LIMIT)
                    .map_err(|e| LLMEndpointError::Advance(e.to_string()))?;

                (session_signal, handle)
            };

            let res = handle.into_string_async().await;

            self.sessions.insert(id, session);

            Ok(res)
        }
    }

    /// Return a [`Box`]ed [`Stream`] of chat completions computed for the provided
    /// [`CompletionArgs`].
    async fn stream_chat_completions(
        &self,
        prompt: &str,
        args: CompletionArgs,
        mut ticket: Ticket,
    ) -> Result<Box<dyn Stream<Item = String> + Unpin + Send>, LLMEndpointError> {
        let (model_signal, model_guard) = self.get_or_init().await?;

        let in_flight_ref = InFlightRef::new(&self.sessions_in_flight);
        if args.one_shot {
            info!("Allocating one-shot LLM session");
            let params = from_completion_args(args).await;
            let session = model_guard
                .create_session(params)
                .map_err(move |e| LLMEndpointError::SessionCreationFailed(e.to_string()))?;
            let sampler = StandardSampler::default();
            ticket.consume();

            Ok(Box::new(
                CompletionStream::new_oneshot(
                    session,
                    prompt,
                    model_signal,
                    self.device,
                    sampler,
                    in_flight_ref,
                )
                .await?,
            ))
        } else {
            let (session, id, new_context) = self.take_chat_session(prompt, args).await?;

            let sampler = StandardSampler::default();
            let tx = self.finished_tx.clone();

            Ok(Box::new(
                CompletionStream::new(
                    session,
                    id,
                    new_context,
                    model_signal,
                    sampler,
                    tx,
                    in_flight_ref,
                    ticket,
                )
                .await?,
            ))
        }
    }

    /// Compute and return embeddings vectors for the provided vector of inputs.
    async fn embeddings(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>, LLMEndpointError> {
        info!("Allocating one-shot LLM embeddings session");
        let _in_flight_ref = InFlightRef::new(&self.sessions_in_flight);
        let threads = SETTINGS.read().await.read().await.auto_threads(false);
        let mut params = EmbeddingsParams::default();
        params.n_threads = threads;
        params.n_threads_batch = threads;

        let (_model_signal, model_guard) = self.get_or_init().await?;
        let res = model_guard
            .embeddings_async(&inputs, params)
            .await
            .map_err(move |e| LLMEndpointError::Embeddings(e.to_string()))?;

        REQUEST_QUEUE.notify_free(self.device)?;

        Ok(res)
    }

    /// Helper function to acquire a read guard to a [`LlamaModel`] (and its associated
    /// [`ActiveSignal`]).
    async fn get_or_init(
        &self,
    ) -> Result<(ActiveSignal, PerishableReadGuard<LlamaModel>), LLMEndpointError> {
        let path = self.path.clone();
        let device = self.device;

        // This should never be called, but just in case
        if !self.loaded().await {
            self.sessions.clear();
        }

        self.model
            .get_or_try_init(move || async move {
                info!(
                    "Loading {} into \"{}\"'s memory",
                    path.to_string_lossy(),
                    device.name()
                );

                let mut args = LlamaParams::default();

                if device == DeviceId::CPU {
                    args.n_gpu_layers = 0;
                } else {
                    args.main_gpu = device.local_id() as u32;
                    args.n_gpu_layers = i32::MAX as u32;
                }

                LlamaModel::load_from_file_async(path, args)
                    .await
                    .map_err(move |e| LLMEndpointError::Load(e.to_string()))
            })
            .await
    }

    /// Return the total amount of sessions currently owned by the model.
    fn sessions(&self) -> usize {
        self.sessions.len() + self.sessions_in_flight.load(Ordering::SeqCst)
    }

    /// Return the amount of sessions currently owned by the model, that can be dropped (are not in flight).
    fn droppable_sessions(&self) -> usize {
        self.sessions.len()
    }

    /// Return the instant when the model was created.
    fn created(&self) -> Instant {
        self.created
    }

    /// Return the size in bytes of the model.
    fn size(&self) -> usize {
        self.size
    }
}

impl Drop for UnloadingModel {
    fn drop(&mut self) {
        self.maintenance_thread.abort()
    }
}

/// A Llama session that automatically unloads itself from memory after some time.
///
/// If the session gets unloaded before a request that would match is submitted, all of the request's context will
/// have to be processed again.
struct UnloadingSession {
    /// This sessions [`Perishable`] inner [`LlamaSession`].
    session: Perishable<LlamaSession>,

    /// The parameter used to create the session.
    params: SessionParams,

    /// The model that owns this session.
    model: LlamaModel,

    /// The instant when the session was created.
    created: Instant,
}

impl UnloadingSession {
    /// Create a new unloading session.
    async fn new(args: CompletionArgs, model: LlamaModel, device: DeviceId) -> Self {
        let session = Perishable::with_ttl(inactive_llm_session_ttl());
        session
            .set_callback(Some(move || {
                if let Err(e) = REQUEST_QUEUE.notify_free(device) {
                    error!("Failed to notify request queue: {e}");
                }
            }))
            .await;

        let params = from_completion_args(args).await;
        Self {
            session,
            params,
            model,
            created: Instant::now(),
        }
    }

    /// Returns **`true`** if this session is currently loaded in system memory, **`false`** otherwise.
    async fn loaded(&self) -> bool {
        self.session.is_alive().await
    }

    /// Unload this session from memory.
    async fn unload(&self) {
        self.session.kill().await;
    }

    /// Helper function to acquire a write guard to a [`LlamaSession`] (and its associated
    /// [`ActiveSignal`]).
    async fn get_or_init(
        &self,
        ticket: Option<Ticket>,
    ) -> Result<(ActiveSignal, PerishableWriteGuard<LlamaSession>), LLMEndpointError> {
        let params = self.params.clone();
        let model = self.model.clone();
        let res = self
            .session
            .get_or_try_init_mut(move || async move {
                info!("Allocating new LLM session");
                model
                    .create_session(params)
                    .map_err(move |e| LLMEndpointError::SessionCreationFailed(e.to_string()))
            })
            .await;
        if let Some(mut ticket) = ticket {
            ticket.consume();
        }
        res
    }

    /// Return the instant when the session was created.
    fn created(&self) -> Instant {
        self.created
    }

    /// Return the current size of this session in memory.
    async fn size(&self) -> Result<usize, LLMEndpointError> {
        let (_signal, session) = self.get_or_init(None).await?;
        Ok(session.memory_size())
    }
}

/// Return a [`SessionParams`] converted from a [`CompletionArgs`].
async fn from_completion_args(args: CompletionArgs) -> SessionParams {
    let mut params = SessionParams::default();
    let default_settings = default_context_settings().await;

    // TODO handle optional params
    //params.seed = args.seed;
    params.n_threads = default_settings.threads;
    params.n_threads_batch = default_settings.threads;
    params.n_ctx = args.context_hint.unwrap_or(default_settings.size);

    params
}

/// An object representing an unique identifier for a session context.
#[derive(Default, Clone)]
struct SessionId {
    /// The context [`Hasher`].
    hasher: Hasher,

    /// The length of the current context.
    len: usize,
}

impl SessionId {
    /// Creates a [`SessionId`] from a prompt.
    ///
    /// This function makes a few assumptions about the provided prompt, given that it is dedicated
    /// to chat sessions. It is assumed that the prompt ends with [`ASSISTANT_TAG`], and that it
    /// probably contains [`ASSISTANT_TAG`], [`USER_TAG`], [`TOOL_TAG`] and/or [`SYSTEM_TAG`].
    ///
    /// Besides returning the new [`SessionId`] instance, the new context in the prompt is also
    /// returned, found based to the positions of the tags.
    ///
    /// # Note
    ///
    /// The new [`SessionId`] returned by this function must be advanced using the returned new context,
    /// before being advanced with inference content. The reason it isn't already advance with the
    /// new context, is for the purpose of finding matching [`SessionId`]s in the endpoint.
    fn chat(prompt: &str) -> (Self, &str, &str) {
        let idx = if prompt.ends_with(ASSISTANT_TAG) {
            if let Some(start) = prompt[..prompt.len() - ASSISTANT_TAG.len()].rfind(ASSISTANT_TAG) {
                // Another assistant tag is found, the is previous context
                if let Some(tag_idx) = find_any(
                    &prompt[start + ASSISTANT_TAG.len()..],
                    &[ASSISTANT_TAG, USER_TAG, TOOL_TAG, SYSTEM_TAG],
                ) {
                    start + ASSISTANT_TAG.len() + tag_idx
                } else {
                    // This should be unreachable
                    error!("Could not find any tags after the last assistant message");
                    0
                }
            } else {
                // No other assistant tag is found, this is the first prompt
                0
            }
        } else {
            error!("Chat prompt doesn't end with the assistant tag");
            0
        };

        let old_context = &prompt[..idx];
        let new_context = &prompt[idx..];

        let mut hasher = Hasher::new();
        hasher.update(old_context.as_bytes());

        let id = Self {
            hasher,
            len: old_context.len(),
        };
        (id, old_context, new_context)
    }

    /// A function to advance the current session context with the provided [`str`] slice.
    ///
    /// ## Notes
    /// The received slice should contain only new dialogue, as old dialogue is already hashed in this
    /// [`SessionId`] and "stored" in the corresponding [`LlamaSession`].
    fn advance(&mut self, new_context: &str) {
        self.hasher.update(new_context.as_bytes());
        self.len += new_context.len();
    }
}

impl core::hash::Hash for SessionId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let blake_hash = self.hasher.finalize();
        for byte in blake_hash.as_bytes() {
            state.write_u8(*byte);
        }
    }
}

impl PartialEq for SessionId {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.hasher.finalize() == other.hasher.finalize()
    }
}

impl Eq for SessionId {}

/// Helper function that finds the first of several substrings in a string, returning the index if
/// one was found
///
/// # Note
/// Internally, calls `find` from [`core::str`].
fn find_any(text: &str, patterns: &[&str]) -> Option<usize> {
    let mut idxs = vec![];
    for pattern in patterns {
        if let Some(idx) = text.find(pattern) {
            idxs.push(idx);
        }
    }

    if idxs.len() > 0 {
        let mut min = usize::MAX;
        for idx in idxs {
            if idx < min {
                min = idx;
            }
        }
        Some(min)
    } else {
        None
    }
}

/// Small helper object used to keep track of session references.
///
/// Upon creation, the provided counter is incremented once, and upon dropping the counter is decremented once.
struct InFlightRef {
    /// The inner counter to increment and decrement.
    allocations: Arc<AtomicUsize>,
}

impl InFlightRef {
    /// Create a new reference.
    ///
    /// Will increment the provided counter once.
    fn new(allocations: &Arc<AtomicUsize>) -> Self {
        allocations.fetch_add(1, Ordering::SeqCst);
        Self {
            allocations: allocations.clone(),
        }
    }
}

impl Drop for InFlightRef {
    fn drop(&mut self) {
        self.allocations.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A [`Stream`] of [`Token`]s returned by a [`LlamaCppSession::stream_complete`] call.
struct CompletionStream {
    /// Handle to the model completions handle.
    handle: TokensToStrings<CompletionHandle>,

    /// The session used for generation completions.
    session: SessionOption,

    /// The `session`'s id.
    session_id: Option<SessionId>,

    /// A sender used to send both `session` and `session_id` once generation is completion
    finished_tx: Option<UnboundedSender<(SessionId, UnloadingSession)>>,

    /// A reference signaling that there is a session in flight.
    _in_flight_ref: InFlightRef,

    /// The object signaling that `model` is currently active.
    _model_signal: ActiveSignal,

    /// The object signaling that `session` is currently active.
    _session_signal: Option<ActiveSignal>,
}

impl CompletionStream {
    /// Constructs a new [`CompletionStream`].
    ///
    /// Once the stream finishes, the respective session is sent back to its owning model.
    ///
    /// ## Arguments
    /// * `session` - The session used to generate completions.
    /// * `session_id` - The [`SessionId`] associated with `session`.
    /// * `new_context` - The context used to advance the session.
    /// * `model_signal` - The `model`'s associated [`ActiveSignal`].
    /// * `sampler` - The [`StandardSampler`] used to generate completions.
    /// * `finished_tx` - An [`UnboundedSender`] used to send both `session` and `session_id` once
    /// generation finishes.
    /// * `in_flight_ref` - The reference used to signal that there is a session in flight.
    /// * `ticket` - The ticket necessary for generation.
    async fn new(
        session: UnloadingSession,
        mut session_id: SessionId,
        new_context: &str,
        model_signal: ActiveSignal,
        sampler: StandardSampler,
        finished_tx: UnboundedSender<(SessionId, UnloadingSession)>,
        in_flight_ref: InFlightRef,
        ticket: Ticket,
    ) -> Result<Self, LLMEndpointError> {
        let (session_signal, handle) = {
            let (session_signal, mut session_guard) = session.get_or_init(Some(ticket)).await?;

            session_guard
                .advance_context_async(new_context)
                .await
                .map_err(move |e| LLMEndpointError::Advance(e.to_string()))?;
            session_id.advance(new_context);

            (
                session_signal,
                session_guard
                    .start_completing_with(sampler, SINGLE_MESSAGE_LIMIT)
                    .map_err(|e| LLMEndpointError::Advance(e.to_string()))?,
            )
        };

        Ok(Self {
            handle: handle.into_strings(),
            session: SessionOption::Perishable(session),
            session_id: Some(session_id),
            finished_tx: Some(finished_tx),
            _in_flight_ref: in_flight_ref,
            _model_signal: model_signal,
            _session_signal: Some(session_signal),
        })
    }

    /// Constructs a new [`CompletionStream`].
    ///
    /// ## Arguments
    /// * `session` - The session used to generate completions.
    /// * `new_context` - The context used to advance the session.
    /// * `model_signal` - The `model`'s associated [`ActiveSignal`].
    /// * `device` - The device where `session` is loaded in.
    /// * `sampler` - The [`StandardSampler`] used to generate completions.
    /// * `in_flight_ref` - The reference used to signal that there is a session in flight.
    async fn new_oneshot(
        mut session: LlamaSession,
        new_context: &str,
        model_signal: ActiveSignal,
        device: DeviceId,
        sampler: StandardSampler,
        in_flight_ref: InFlightRef,
    ) -> Result<Self, LLMEndpointError> {
        session
            .advance_context_async(new_context)
            .await
            .map_err(move |e| LLMEndpointError::Advance(e.to_string()))?;
        let handle = session
            .start_completing_with(sampler, SINGLE_MESSAGE_LIMIT)
            .map_err(|e| LLMEndpointError::Advance(e.to_string()))?;

        Ok(Self {
            handle: handle.into_strings(),
            session: SessionOption::OneShot {
                _session: session,
                device,
            },
            session_id: None,
            finished_tx: None,
            _in_flight_ref: in_flight_ref,
            _model_signal: model_signal,
            _session_signal: None,
        })
    }
}

impl Stream for CompletionStream {
    type Item = String;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match std::pin::pin!(&mut self.handle).poll_next(cx) {
            Poll::Ready(Some(val)) => {
                if let Some(id) = &mut self.session_id {
                    id.advance(&val);
                }
                Poll::Ready(Some(val))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for CompletionStream {
    fn drop(&mut self) {
        let mut notify = None;

        if let Some(id) = self.session_id.take() {
            match self.session.take() {
                SessionOption::OneShot { device, .. } => notify = Some(device),
                SessionOption::Perishable(session) => {
                    if let Some(channel) = self.finished_tx.take() {
                        channel.send((id, session)).unwrap_or_else(move |e| {
                            error!("Failed to send session to maintenance thread: {e}")
                        });
                    }
                }
                SessionOption::None => {}
            }
        }

        // Make sure the notification is only sent after the session has been dropped
        if let Some(device) = notify {
            let _ = REQUEST_QUEUE.notify_free(device);
        }
    }
}

/// A type which may contain an isolated session, an unloading session to be sent back to a model, or nothing.
#[derive(Default)]
enum SessionOption {
    /// An isolated session.
    OneShot {
        /// The session's handle.
        _session: LlamaSession,

        /// The device where `session` is loaded in.
        device: DeviceId,
    },

    /// An unloading session that should be sent back to a model.
    Perishable(UnloadingSession),

    /// Nothing.
    #[default]
    None,
}

impl SessionOption {
    /// Return the options value and replace it with [`SessionOption::None`].
    fn take(&mut self) -> Self {
        take(self)
    }
}
