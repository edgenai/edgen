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

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dashmap::DashMap;
use futures::executor::block_on;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};
use tokio::{select, spawn};
use tracing::info;
use uuid::Uuid;
use whisper_cpp::{WhisperModel, WhisperParams, WhisperSampling, WhisperSession};

use edgen_core::cleanup_interval;
use edgen_core::perishable::{ActiveSignal, Perishable, PerishableReadGuard, PerishableWriteGuard};
use edgen_core::settings::SETTINGS;
use edgen_core::whisper::{
    inactive_whisper_session_ttl, inactive_whisper_ttl, parse_pcm, TranscriptionArgs,
    WhisperEndpoint, WhisperEndpointError,
};

/// A large language model endpoint, implementing [`WhisperEndpoint`] using a [`whisper_cpp`] backend.
pub struct WhisperCppEndpoint {
    /// A map of the models currently loaded into memory, with their path as the key.
    models: Arc<DashMap<String, UnloadingModel>>,

    /// A background thread that periodically removes models from the `models` collection, if they
    /// are not loaded at the time.
    cleanup_thread: JoinHandle<()>,
}

impl WhisperCppEndpoint {
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

    async fn async_transcription(
        &self,
        model_path: impl AsRef<Path>,
        args: TranscriptionArgs,
    ) -> Result<String, WhisperEndpointError> {
        let pcm = parse_pcm(&args.file)?;
        let model = self.get(model_path).await;
        model.transcription(args.session, pcm).await
    }
}

impl WhisperEndpoint for WhisperCppEndpoint {
    fn transcription<'a>(
        &'a self,
        model_path: impl AsRef<Path> + Send + 'a,
        args: TranscriptionArgs,
    ) -> Box<dyn Future<Output = Result<String, WhisperEndpointError>> + Send + Unpin + 'a> {
        let pinned = Box::pin(self.async_transcription(model_path, args));
        Box::new(pinned)
    }

    fn reset(&self) {
        self.models.clear();
    }
}

impl Default for WhisperCppEndpoint {
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

impl Drop for WhisperCppEndpoint {
    fn drop(&mut self) {
        self.cleanup_thread.abort()
    }
}

/// A [`WhisperModel`] (as well as its associated [`WhisperSession`]s) that unloads itself from
/// memory after not being used for a period of time.
struct UnloadingModel {
    model: Perishable<WhisperModel>,
    path: PathBuf,
    sessions: Arc<DashMap<Uuid, Perishable<WhisperSession>>>,
    maintenance_thread: JoinHandle<()>,
    finished_tx: UnboundedSender<(Uuid, Perishable<WhisperSession>)>,
}

impl UnloadingModel {
    /// Creates a new instance of this model, provided it's [`Path`].
    ///
    /// This function is lazy and does not actually load the model into system memory, the model must be accessed in
    /// order to be loaded.
    async fn new(model_path: impl AsRef<Path>) -> Self {
        let sessions: Arc<DashMap<Uuid, Perishable<WhisperSession>>> = Default::default();
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
            model: Perishable::with_ttl(inactive_whisper_ttl()),
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

    /// Either takes an existing chat [`WhisperSession`] matching the provided [`Uuid`], or creates
    /// a new one.
    async fn take_session(&self, uuid: Uuid) -> Perishable<WhisperSession> {
        let session_perishable = if let Some((_, session)) = self.sessions.remove(&uuid) {
            info!("Matching session found, continuing");
            session
        } else {
            info!("No matching session found, creating new one");
            Perishable::with_ttl(inactive_whisper_session_ttl())
        };

        session_perishable
    }

    async fn transcription(
        &self,
        uuid: Option<Uuid>,
        pcm: Vec<f32>,
    ) -> Result<String, WhisperEndpointError> {
        let (_model_signal, model_guard) = get_or_init_model(&self.model, &self.path).await?;

        let mut params = WhisperParams::new(WhisperSampling::default_greedy());
        let threads = SETTINGS.read().await.read().await.auto_threads(false);

        params.thread_count = threads;

        if let Some(uuid) = uuid {
            let session = self.take_session(uuid).await;

            let (_session_signal, mut session_guard) = {
                let (session_signal, mut session_guard) =
                    get_or_init_session(&session, model_guard.clone()).await?;

                (session_signal, session_guard)
            };

            session_guard
                .full(params, &pcm)
                .await
                .map_err(move |e| WhisperEndpointError::Advance(e.to_string()))?;

            let mut res = "".to_string();
            for i in 0..session_guard.segment_count() {
                res += &*session_guard
                    .segment_text(i)
                    .map_err(move |e| WhisperEndpointError::Decode(e.to_string()))?;
            }

            Ok(res)
        } else {
            let mut session = model_guard
                .new_session()
                .await
                .map_err(move |e| WhisperEndpointError::Session(e.to_string()))?;

            session
                .full(params, &pcm)
                .await
                .map_err(move |e| WhisperEndpointError::Advance(e.to_string()))?;

            let mut res = "".to_string();
            for i in 0..session.segment_count() {
                res += &*session
                    .segment_text(i)
                    .map_err(move |e| WhisperEndpointError::Decode(e.to_string()))?;
            }

            Ok(res)
        }
    }
}

impl Drop for UnloadingModel {
    fn drop(&mut self) {
        self.maintenance_thread.abort()
    }
}

/// Helper function to acquire a read guard to a [`WhisperModel`] (and its associated
/// [`ActiveSignal`]).
async fn get_or_init_model(
    model: &Perishable<WhisperModel>,
    path: impl AsRef<Path>,
) -> Result<(ActiveSignal, PerishableReadGuard<WhisperModel>), WhisperEndpointError> {
    let path = path.as_ref().to_path_buf();
    model
        .get_or_try_init(move || async move {
            WhisperModel::new_from_file(path, false)
                .map_err(move |e| WhisperEndpointError::Load(e.to_string()))
        })
        .await
}

/// Helper function to acquire a write guard to a [`WhisperSession`] (and its associated
/// [`ActiveSignal`]).
async fn get_or_init_session(
    session: &Perishable<WhisperSession>,
    model: WhisperModel,
) -> Result<(ActiveSignal, PerishableWriteGuard<WhisperSession>), WhisperEndpointError> {
    session
        .get_or_try_init_mut(move || async move {
            model
                .new_session()
                .await
                .map_err(move |e| WhisperEndpointError::Session(e.to_string()))
        })
        .await
}
