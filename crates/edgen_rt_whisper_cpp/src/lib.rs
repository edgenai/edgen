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

use std::path::{Path, PathBuf};
use std::sync::Arc;

use dashmap::DashMap;
use futures::executor::block_on;
use tokio::spawn;
use tokio::task::JoinHandle;
use tokio::time::{interval, MissedTickBehavior};
use tracing::info;
use uuid::Uuid;
use whisper_cpp::{WhisperModel, WhisperParams, WhisperSampling, WhisperSession};

use edgen_core::perishable::{ActiveSignal, Perishable, PerishableReadGuard, PerishableWriteGuard};
use edgen_core::settings::{DevicePolicy, SETTINGS};
use edgen_core::whisper::{
    inactive_whisper_session_ttl, inactive_whisper_ttl, parse, TranscriptionArgs, WhisperEndpoint,
    WhisperEndpointError,
};
use edgen_core::{cleanup_interval, BoxedFuture};

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

    /// Helper `async` function that returns the transcription for the specified model and
    /// [`TranscriptionArgs`]
    async fn async_transcription(
        &self,
        model_path: impl AsRef<Path>,
        args: TranscriptionArgs,
    ) -> Result<(String, Option<Uuid>), WhisperEndpointError> {
        let pcm = parse::pcm(&args.file)?;
        let model = self.get(model_path).await;
        model
            .transcription(args.create_session, args.session, pcm)
            .await
    }
}

impl WhisperEndpoint for WhisperCppEndpoint {
    fn transcription<'a>(
        &'a self,
        model_path: impl AsRef<Path> + Send + 'a,
        args: TranscriptionArgs,
    ) -> BoxedFuture<Result<(String, Option<Uuid>), WhisperEndpointError>> {
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
}

impl UnloadingModel {
    /// Creates a new instance of this model, provided it's [`Path`].
    ///
    /// This function is lazy and does not actually load the model into system memory, the model must be accessed in
    /// order to be loaded.
    async fn new(model_path: impl AsRef<Path>) -> Self {
        let sessions: Arc<DashMap<Uuid, Perishable<WhisperSession>>> = Default::default();

        let sessions_clone = sessions.clone();
        let maintenance_thread = spawn(async move {
            let mut interval = interval(cleanup_interval());
            interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

            loop {
                interval.tick().await;
                sessions_clone.retain(move |_, session| block_on(session.is_alive()));
            }
        });

        Self {
            model: Perishable::with_ttl(inactive_whisper_ttl()),
            path: model_path.as_ref().to_path_buf(),
            sessions,
            maintenance_thread,
        }
    }

    /// Returns **`true`** if this model is currently loaded in system memory, **`false`** otherwise.
    async fn loaded(&self) -> bool {
        self.model.is_alive().await
    }

    /// Computes the full transcription for the provided *PCM*;
    async fn transcription(
        &self,
        create_session: bool,
        uuid: Option<Uuid>,
        pcm: Vec<f32>,
    ) -> Result<(String, Option<Uuid>), WhisperEndpointError> {
        let (_model_signal, model_guard) = get_or_init_model(&self.model, &self.path).await?;

        let mut params = WhisperParams::new(WhisperSampling::default_greedy());
        let threads = SETTINGS.read().await.read().await.auto_threads(false);

        params.thread_count = threads;

        let uuid = if let Some(uuid) = uuid {
            Some(uuid)
        } else {
            if create_session {
                let uuid = Uuid::new_v4();
                self.sessions
                    .insert(uuid, Perishable::with_ttl(inactive_whisper_session_ttl()));
                Some(uuid)
            } else {
                None
            }
        };

        if let Some(uuid) = uuid {
            let session = self
                .sessions
                .get(&uuid)
                .ok_or(WhisperEndpointError::SessionNotFound)?;

            let (_session_signal, mut session_guard) =
                get_or_init_session(session.value(), model_guard.clone()).await?;
            // Perishable uses a tokio RwLock internally, which guarantees fair access, so we
            // shouldn't have to worry about thread ordering

            params.no_context = false;
            session_guard
                .advance(params, &pcm)
                .await
                .map_err(move |e| WhisperEndpointError::Advance(e.to_string()))?;
            let res = session_guard
                .new_context()
                .map_err(move |e| WhisperEndpointError::Advance(e.to_string()))?;

            if create_session {
                Ok((res, Some(uuid)))
            } else {
                Ok((res, None))
            }
        } else {
            info!("Allocating oneshot whisper session");
            let mut session = model_guard
                .new_session()
                .await
                .map_err(move |e| WhisperEndpointError::SessionCreationFailed(e.to_string()))?;

            params.no_context = true;
            session
                .advance(params, &pcm)
                .await
                .map_err(move |e| WhisperEndpointError::Advance(e.to_string()))?;
            let res = session
                .new_context()
                .map_err(move |e| WhisperEndpointError::Advance(e.to_string()))?;

            Ok((res, None))
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
            info!("Loading {} into memory", path.to_string_lossy());

            let device = match SETTINGS.read().await.read().await.gpu_policy {
                DevicePolicy::AlwaysCpu { .. } => None,
                DevicePolicy::AlwaysDevice { .. } => Some(0),
                _ => {
                    unimplemented!()
                }
            };

            WhisperModel::new_from_file(path, device)
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
            info!("Allocating new whisper session");
            model
                .new_session()
                .await
                .map_err(move |e| WhisperEndpointError::SessionCreationFailed(e.to_string()))
        })
        .await
}
