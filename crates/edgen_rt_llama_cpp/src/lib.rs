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
use tokio::time::{interval, MissedTickBehavior};
use tokio::{select, spawn};
use tracing::{error, info};

use edgen_core::cleanup_interval;
use edgen_core::llm::{
    default_context_settings, inactive_llm_session_ttl, inactive_llm_ttl, CompletionArgs,
    LLMEndpoint, LLMEndpointError, ASSISTANT_TAG, SYSTEM_TAG, TOOL_TAG, USER_TAG,
};
use edgen_core::perishable::{ActiveSignal, Perishable, PerishableReadGuard, PerishableWriteGuard};
use edgen_core::settings::{DevicePolicy, SETTINGS};

// TODO this should be in settings
const SINGLE_MESSAGE_LIMIT: usize = 4096;

/// A large language model endpoint, implementing [`LLMEndpoint`] using a [`llama_cpp`] backend.
pub struct LlamaCppEndpoint {
    /// A map of the models currently loaded into memory, with their path as the key.
    models: Arc<DashMap<String, UnloadingModel>>,

    /// A background thread that periodically removes models from the `models` collection, if they
    /// are not loaded at the time.
    cleanup_thread: JoinHandle<()>,
}

impl LlamaCppEndpoint {
    /// Gets the [`UnloadingModel`] loaded from the specified path. If the model isn't already
    /// loaded, first initialise it and add it to the `models` collection.
    async fn get(
        &self,
        model_path: impl AsRef<Path>,
    ) -> dashmap::mapref::one::Ref<String, UnloadingModel> {
        let key = model_path.as_ref().to_string_lossy().to_string();

        if !self.models.contains_key(&key) {
            let model = UnloadingModel::new(model_path).await;
            self.models.insert(key.clone(), model);
        }

        // PANIC SAFETY: Just inserted the element if it isn't already inside the map, so must be present in the map
        self.models.get(&key).unwrap()
    }
}

#[async_trait::async_trait]
impl LLMEndpoint for LlamaCppEndpoint {
    async fn chat_completions(
        &self,
        model_path: impl AsRef<Path> + Send,
        args: CompletionArgs,
    ) -> Result<String, LLMEndpointError> {
        let model = self.get(model_path).await;
        model.chat_completions(args).await
    }

    async fn stream_chat_completions(
        &self,
        model_path: impl AsRef<Path> + Send,
        args: CompletionArgs,
    ) -> Result<Box<dyn Stream<Item = String> + Unpin + Send>, LLMEndpointError> {
        let model = self.get(model_path).await;
        model.stream_chat_completions(args).await
    }

    async fn embeddings(
        &self,
        model_path: impl AsRef<Path> + Send,
        inputs: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, LLMEndpointError> {
        let model = self.get(model_path).await;
        model.embeddings(inputs).await
    }

    fn reset(&self) {
        self.models.clear();
    }
}

impl Default for LlamaCppEndpoint {
    fn default() -> Self {
        let models: Arc<DashMap<String, UnloadingModel>> = Default::default();
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

/// A [`LlamaModel`] (as well as its associated [`LlamaSession`]s) that unloads itself from memory after not being used
/// for a period of time.
struct UnloadingModel {
    model: Perishable<LlamaModel>,
    path: PathBuf,
    sessions: Arc<DashMap<SessionId, Perishable<LlamaSession>>>,
    maintenance_thread: JoinHandle<()>,
    finished_tx: UnboundedSender<(SessionId, Perishable<LlamaSession>)>,
}

impl UnloadingModel {
    /// Creates a new instance of this model, provided it's [`Path`].
    ///
    /// This function is lazy and does not actually load the model into system memory, the model must be accessed in
    /// order to be loaded.
    async fn new(model_path: impl AsRef<Path>) -> Self {
        let sessions: Arc<DashMap<SessionId, Perishable<LlamaSession>>> = Default::default();
        let (tx, mut rx) = unbounded_channel();

        let sessions_clone = sessions.clone();
        let maintenance_thread = spawn(async move {
            let mut interval = interval(cleanup_interval());
            interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

            loop {
                select! {
                    _ = interval.tick() => sessions_clone.retain(move |_, session| block_on(session.is_alive())),
                    item = rx.recv() => {
                        if let Some((id, session)) = item {
                            sessions_clone.insert(id, session);
                        }
                    }
                }
            }
        });

        Self {
            model: Perishable::with_ttl(inactive_llm_ttl()),
            path: model_path.as_ref().to_path_buf(),
            sessions,
            maintenance_thread,
            finished_tx: tx,
        }
    }

    /// Returns **`true`** if this model is currently loaded in system memory, **`false`** otherwise.
    async fn loaded(&self) -> bool {
        self.model.is_alive().await
    }

    /// Either takes an existing chat [`LlamaSession`] compatible with the provided prompt from the
    /// `sessions` collection, or creates a new one.
    ///
    /// The matching [`SessionId`] and the new context derived from `prompt` are also returned.
    async fn take_chat_session<'a>(
        &self,
        prompt: &'a str,
    ) -> (Perishable<LlamaSession>, SessionId, &'a str) {
        let (id, new_context) = SessionId::chat(prompt);

        let session_perishable = if let Some((_, session)) = self.sessions.remove(&id) {
            info!("Matching session found, continuing");
            session
        } else {
            info!("No matching session found, creating new one");
            Perishable::with_ttl(inactive_llm_session_ttl())
        };

        (session_perishable, id, new_context)
    }

    /// Computes the full chat completions for the provided [`CompletionArgs`].
    async fn chat_completions(&self, args: CompletionArgs) -> Result<String, LLMEndpointError> {
        let (_model_signal, model_guard) = get_or_init_model(&self.model, &self.path).await?;

        if args.one_shot {
            info!("Allocating one-shot LLM session");
            let mut params = SessionParams::default();
            let default_settings = default_context_settings().await;

            // TODO handle optional params
            //params.seed = args.seed;
            params.n_threads = default_settings.threads;
            params.n_threads_batch = default_settings.threads;
            params.n_ctx = args.context_hint.unwrap_or(default_settings.size);

            let mut session = model_guard
                .create_session(params)
                .map_err(move |e| LLMEndpointError::SessionCreationFailed(e.to_string()))?;

            session
                .advance_context_async(args.prompt)
                .await
                .map_err(move |e| LLMEndpointError::Advance(e.to_string()))?;

            let sampler = StandardSampler::default();
            let handle = session.start_completing_with(sampler, SINGLE_MESSAGE_LIMIT);

            Ok(handle.into_string_async().await)
        } else {
            let (session, mut id, new_context) = self.take_chat_session(&args.prompt).await;

            let (_session_signal, handle) = {
                let (session_signal, mut session_guard) =
                    get_or_init_session(&session, model_guard.clone()).await?;

                session_guard
                    .advance_context_async(new_context)
                    .await
                    .map_err(move |e| LLMEndpointError::Advance(e.to_string()))?;
                id.advance(new_context);

                let sampler = StandardSampler::default();
                let handle = session_guard.start_completing_with(sampler, SINGLE_MESSAGE_LIMIT);

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
        args: CompletionArgs,
    ) -> Result<Box<dyn Stream<Item = String> + Unpin + Send>, LLMEndpointError> {
        let (model_signal, model_guard) = get_or_init_model(&self.model, &self.path).await?;

        if args.one_shot {
            info!("Allocating one-shot LLM session");
            let mut params = SessionParams::default();
            let default_settings = default_context_settings().await;

            // TODO handle optional params
            //params.seed = args.seed;
            params.n_threads = default_settings.threads;
            params.n_threads_batch = default_settings.threads;
            params.n_ctx = args.context_hint.unwrap_or(default_settings.size);

            let session = model_guard
                .create_session(params)
                .map_err(move |e| LLMEndpointError::SessionCreationFailed(e.to_string()))?;
            let sampler = StandardSampler::default();

            Ok(Box::new(
                CompletionStream::new_oneshot(session, &args.prompt, model_signal, sampler).await?,
            ))
        } else {
            let (session, id, new_context) = self.take_chat_session(&args.prompt).await;

            let sampler = StandardSampler::default();
            let tx = self.finished_tx.clone();

            Ok(Box::new(
                CompletionStream::new(
                    session,
                    id,
                    new_context,
                    model_guard.clone(),
                    model_signal,
                    sampler,
                    tx,
                )
                .await?,
            ))
        }
    }

    async fn embeddings(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>, LLMEndpointError> {
        let threads = SETTINGS.read().await.read().await.auto_threads(false);
        let mut params = EmbeddingsParams::default();
        params.n_threads = threads;
        params.n_threads_batch = threads;

        let (_model_signal, model_guard) = get_or_init_model(&self.model, &self.path).await?;
        model_guard
            .embeddings_async(&inputs, params)
            .await
            .map_err(move |e| LLMEndpointError::Embeddings(e.to_string()))
    }
}

impl Drop for UnloadingModel {
    fn drop(&mut self) {
        self.maintenance_thread.abort()
    }
}

/// Helper function to acquire a read guard to a [`LlamaModel`] (and its associated
/// [`ActiveSignal`]).
async fn get_or_init_model(
    model: &Perishable<LlamaModel>,
    path: impl AsRef<Path>,
) -> Result<(ActiveSignal, PerishableReadGuard<LlamaModel>), LLMEndpointError> {
    let path = path.as_ref().to_path_buf();
    model
        .get_or_try_init(move || async move {
            info!("Loading {} into memory", path.to_string_lossy());
            let mut args = LlamaParams::default();

            match SETTINGS.read().await.read().await.gpu_policy {
                DevicePolicy::AlwaysCpu { .. } => {
                    args.n_gpu_layers = 0;
                }
                DevicePolicy::AlwaysDevice { .. } => {
                    args.n_gpu_layers = i32::MAX as u32;
                }
                _ => {
                    unimplemented!()
                }
            }

            LlamaModel::load_from_file_async(path, args)
                .await
                .map_err(move |e| LLMEndpointError::Load(e.to_string()))
        })
        .await
}

/// Helper function to acquire a write guard to a [`LlamaSession`] (and its associated
/// [`ActiveSignal`]).
async fn get_or_init_session(
    session: &Perishable<LlamaSession>,
    model: LlamaModel,
) -> Result<(ActiveSignal, PerishableWriteGuard<LlamaSession>), LLMEndpointError> {
    session
        .get_or_try_init_mut(move || async move {
            info!("Allocating new LLM session");
            let mut params = SessionParams::default();
            let default_settings = default_context_settings().await;

            // TODO handle optional params
            //params.seed = args.seed;
            params.n_threads = default_settings.threads;
            params.n_threads_batch = default_settings.threads;
            params.n_ctx = default_settings.size;

            model
                .create_session(params)
                .map_err(move |e| LLMEndpointError::SessionCreationFailed(e.to_string()))
        })
        .await
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
    fn chat(prompt: &str) -> (Self, &str) {
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
        (id, new_context)
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

/// A [`Stream`] of [`Token`]s returned by a [`LlamaCppSession::stream_complete`] call.
struct CompletionStream {
    /// Handle to the model completions handle.
    handle: TokensToStrings<CompletionHandle>,

    /// The session used for generation completions.
    session: SessionOption,

    /// The `session`'s id.
    session_id: Option<SessionId>,

    /// A sender used to send both `session` and `session_id` once generation is completion
    finished_tx: Option<UnboundedSender<(SessionId, Perishable<LlamaSession>)>>,

    /// The object signaling that `model` is currently active.
    _model_signal: ActiveSignal,

    /// The object signaling that `session` is currently active.
    _session_signal: Option<ActiveSignal>,
}

impl CompletionStream {
    /// Constructs a new [`CompletionStream`].
    ///
    /// ## Arguments
    /// * `session` - The session used to generate completions.
    /// * `session_id` - The [`SessionId`] associated with `session`.
    /// * `new_context` - The context used to advance the session.
    /// * `model` - The [`LlamaModel`] that `session` is associated with.
    /// * `model_signal` - The `model`'s associated [`ActiveSignal`].
    /// * `sample` - The [`StandardSampler`] used to generate completions.
    /// * `end_token` - An [`UnboundedSender`] used to send both `session` and `session` once
    /// generation finishes.
    async fn new(
        session: Perishable<LlamaSession>,
        mut session_id: SessionId,
        new_context: &str,
        model: LlamaModel,
        model_signal: ActiveSignal,
        sampler: StandardSampler,
        finished_tx: UnboundedSender<(SessionId, Perishable<LlamaSession>)>,
    ) -> Result<Self, LLMEndpointError> {
        let (session_signal, handle) = {
            let (session_signal, mut session_guard) = get_or_init_session(&session, model).await?;

            session_guard
                .advance_context_async(new_context)
                .await
                .map_err(move |e| LLMEndpointError::Advance(e.to_string()))?;
            session_id.advance(new_context);

            (
                session_signal,
                session_guard.start_completing_with(sampler, SINGLE_MESSAGE_LIMIT),
            )
        };

        Ok(Self {
            handle: handle.into_strings(),
            session: SessionOption::Perishable(session),
            session_id: Some(session_id),
            finished_tx: Some(finished_tx),
            _model_signal: model_signal,
            _session_signal: Some(session_signal),
        })
    }

    async fn new_oneshot(
        mut session: LlamaSession,
        new_context: &str,
        model_signal: ActiveSignal,
        sampler: StandardSampler,
    ) -> Result<Self, LLMEndpointError> {
        session
            .advance_context_async(new_context)
            .await
            .map_err(move |e| LLMEndpointError::Advance(e.to_string()))?;
        let handle = session.start_completing_with(sampler, SINGLE_MESSAGE_LIMIT);

        Ok(Self {
            handle: handle.into_strings(),
            session: SessionOption::OneShot(session),
            session_id: None,
            finished_tx: None,
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
        if let Some(id) = self.session_id.take() {
            if let SessionOption::Perishable(session) = self.session.take() {
                if let Some(channel) = self.finished_tx.take() {
                    channel.send((id, session)).unwrap_or_else(move |e| {
                        error!("Failed to send session to maintenance thread: {e}")
                    });
                }
            }
        }
    }
}

#[derive(Default)]
enum SessionOption {
    OneShot(LlamaSession),
    Perishable(Perishable<LlamaSession>),
    #[default]
    None,
}

impl SessionOption {
    fn take(&mut self) -> Self {
        take(self)
    }
}
