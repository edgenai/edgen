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

use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dashmap::DashMap;
use directories::ProjectDirs;
use futures::executor::block_on;
use notify::{Config, Event, EventHandler, EventKind, PollWatcher, RecursiveMode, Watcher};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_yaml::{from_slice, to_string};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{error, info};
use uuid::Uuid;

/// The file extension of a YAML file, which is the format used to store settings.
const FILE_EXTENSION: &str = ".yaml";
const FILE_NAME: &str = "edgen.conf";

// TODO look for a better way to do this, since [Settings] already uses a lock
pub static SETTINGS: Lazy<RwLock<StaticSettings>> = Lazy::new(Default::default);

/// The configuration, and data directories for Edgen.
pub static PROJECT_DIRS: Lazy<ProjectDirs> =
    Lazy::new(|| ProjectDirs::from("com", "EdgenAI", "Edgen").unwrap());
pub static CONFIG_FILE: Lazy<PathBuf> = Lazy::new(|| build_config_file_path());

/// Create project dirs if they don't exist
pub async fn create_project_dirs() -> Result<(), std::io::Error> {
    let config_dir = PROJECT_DIRS.config_dir();

    let chat_completions_str = SETTINGS
        .read()
        .await
        .read()
        .await
        .chat_completions_models_dir
        .to_string();

    let chat_completions_dir = PathBuf::from(&chat_completions_str);

    let audio_transcriptions_str = SETTINGS
        .read()
        .await
        .read()
        .await
        .audio_transcriptions_models_dir
        .to_string();

    let audio_transcriptions_dir = PathBuf::from(&audio_transcriptions_str);

    if !config_dir.is_dir() {
        std::fs::create_dir_all(&config_dir)?;
    }

    if !chat_completions_dir.is_dir() {
        std::fs::create_dir_all(&chat_completions_dir)?;
    }

    if !audio_transcriptions_dir.is_dir() {
        std::fs::create_dir_all(&audio_transcriptions_dir)?;
    }

    Ok(())
}

/// Create the default config file if it does not exist
pub fn create_default_config_file() -> Result<(), SettingsError> {
    block_on(async {
        StaticSettings { inner: None }.init().await
    })?;

    Ok(())
}

/// Get path to the config file
pub fn get_config_file_path() -> PathBuf {
    CONFIG_FILE.to_path_buf()
}

fn build_config_file_path() -> PathBuf {
    let config_dir = PROJECT_DIRS.config_dir();
    let filename = FILE_NAME.to_string() + FILE_EXTENSION;
    config_dir.join(Path::new(&filename))
}

#[derive(Error, Debug, Serialize)]
pub enum SettingsError {
    #[error("failed to read the settings file: {0}")]
    Read(String),
    #[error("failed to write the settings file: {0}")]
    Write(String),
    #[error("failed to create directory: {0}")]
    Directory(String),
    #[error("failed to deserialize settings from file: {0}")]
    Deserialize(String),
    #[error("failed to serialize settings from struct: {0}")]
    Serialize(String),
    #[error("failed to create file watcher: {0}")]
    Watcher(String),
    #[error("failed to watch settings file: {0}")]
    WatchFile(String),
    #[error("global settings have already been initialised")]
    AlreadyInitialised,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettingsParams {
    // TODO make a different thread settings for each endpoint
    /// The number of threads each individual endpoint session can use.
    pub threads: u32,

    // TODO should this be a vector instead?
    /// The default URI that *Edgen* will receive requests in.
    pub default_uri: String,

    // TODO temporary, until the model parameter in incoming requests can be parsed into local paths
    pub chat_completions_models_dir: String,
    /// The chat completion model that Edgen will use when the user does not provide a model
    pub chat_completions_model_name: String,
    /// The chat completion model repo that Edgen will use for download
    pub chat_completions_model_repo: String,

    // TODO temporary, until the model parameter in incoming requests can be parsed into local paths
    pub audio_transcriptions_models_dir: String,
    /// The audio transcription model that Edgen will use when the user does not provide a model
    pub audio_transcriptions_model_name: String,
    /// The audio transcription repo that Edgen will use for downloads
    pub audio_transcriptions_model_repo: String,
}

impl SettingsParams {
    pub fn auto_threads(&self, physical: bool) -> u32 {
        let max_threads = if physical {
            num_cpus::get_physical()
        } else {
            num_cpus::get()
        };
        let max_threads = max_threads as u32;

        if self.threads == 0 || self.threads > max_threads {
            max_threads
        } else {
            self.threads
        }
    }
}

impl Default for SettingsParams {
    fn default() -> Self {
        let data_dir = PROJECT_DIRS.data_dir();
        let chat_completions_dir = data_dir.join(Path::new("models/chat/completions"));
        let audio_transcriptions_dir = data_dir.join(Path::new("models/audio/transcriptions"));

        let chat_completions_str = chat_completions_dir.into_os_string().into_string().unwrap();
        let audio_transcriptions_str = audio_transcriptions_dir
            .into_os_string()
            .into_string()
            .unwrap();

        let cpus = num_cpus::get_physical();
        let threads = if cpus > 1 { cpus - 1 } else { 1 };

        // if changed, please update docs at docs/src/app/documentation/configuration/page.mdx
        Self {
            threads: threads as u32,
            default_uri: "http://127.0.0.1:33322".to_string(),
            chat_completions_model_name: "neural-chat-7b-v3-3.Q4_K_M.gguf".to_string(),
            chat_completions_model_repo: "TheBloke/neural-chat-7B-v3-3-GGUF".to_string(),
            audio_transcriptions_model_name: "ggml-distil-small.en.bin".to_string(),
            audio_transcriptions_model_repo: "distil-whisper/distil-small.en".to_string(),
            chat_completions_models_dir: chat_completions_str,
            audio_transcriptions_models_dir: audio_transcriptions_str,
        }
    }
}

pub struct SettingsInner {
    params: SettingsParams,
    changed_params: SettingsParams,
    path: PathBuf,
}

impl SettingsInner {
    async fn load_or_create(
        directory: impl AsRef<Path>,
        name: &str,
    ) -> Result<(Self, bool), SettingsError> {
        let filename = name.to_string() + FILE_EXTENSION;
        let path = directory.as_ref().join(filename);

        let is_new = !path.exists();
        let params = if is_new {
            info!("Creating new settings file: {}", path.to_string_lossy());

            tokio::fs::create_dir_all(directory)
                .await
                .map_err(move |e| SettingsError::Write(e.to_string()))?;
            tokio::fs::write(&path, "")
                .await
                .map_err(move |e| SettingsError::Write(e.to_string()))?;
            SettingsParams::default()
        } else {
            info!("Loading existing settings file: {}", path.to_string_lossy());

            let yaml = tokio::fs::read(&path)
                .await
                .map_err(move |e| SettingsError::Read(e.to_string()))?;
            from_slice(&yaml).map_err(move |e| SettingsError::Deserialize(e.to_string()))?
        };
        let changed_params = params.clone();

        Ok((
            Self {
                params,
                changed_params,
                path,
            },
            is_new,
        ))
    }

    async fn save(&self) -> Result<(), SettingsError> {
        let text =
            to_string(&self.params).map_err(move |e| SettingsError::Serialize(e.to_string()))?;
        tokio::fs::write(&self.path, text)
            .await
            .map_err(move |e| SettingsError::Write(e.to_string()))?;

        Ok(())
    }

    pub async fn apply(&mut self) -> Result<(), SettingsError> {
        info!("Applying new settings");

        // TODO the update handler already does this, is it needed?
        self.params = self.changed_params.clone();
        self.save().await?;

        Ok(())
    }
}

impl Deref for SettingsInner {
    type Target = SettingsParams;

    fn deref(&self) -> &Self::Target {
        &self.params
    }
}

impl DerefMut for SettingsInner {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.changed_params
    }
}

pub struct Settings {
    inner: Arc<RwLock<SettingsInner>>,
    _watcher: PollWatcher, // we use a PollWatcher because it observes the path, not the inode
    handler: UpdateHandler,
}

impl Settings {
    pub async fn load_or_create(
        directory: impl AsRef<Path>,
        name: &str,
    ) -> Result<Self, SettingsError> {
        let (inner, is_new) = SettingsInner::load_or_create(directory, name).await?;
        let inner = Arc::new(RwLock::new(inner));

        let handler = UpdateHandler::new(inner.clone());
        let watcher_config =
            Config::default().with_poll_interval(std::time::Duration::from_secs(3));
        let mut watcher = PollWatcher::new(handler.clone(), watcher_config)
            .map_err(move |e| SettingsError::Watcher(e.to_string()))?;

        {
            let locked = inner.read().await;

            watcher
                .watch(&locked.path, RecursiveMode::NonRecursive)
                .map_err(move |e| SettingsError::WatchFile(e.to_string()))?;

            if is_new {
                locked.save().await?;
            }
        }

        let res = Self {
            inner,
            _watcher: watcher,
            handler,
        };

        Ok(res)
    }

    pub fn add_change_callback<F>(&self, callback: F) -> CallbackHandle
    where
        F: FnMut() + Send + Sync + 'static,
    {
        CallbackHandle::new(callback, &self.handler)
    }
}

impl Deref for Settings {
    type Target = Arc<RwLock<SettingsInner>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Clone)]
struct UpdateHandler {
    settings: Arc<RwLock<SettingsInner>>,
    callbacks: Arc<DashMap<Uuid, Box<dyn FnMut() + Send + Sync>>>,
}

impl UpdateHandler {
    fn new(settings: Arc<RwLock<SettingsInner>>) -> Self {
        Self {
            settings,
            callbacks: Default::default(),
        }
    }

    fn add_callback<F>(&self, callback: F) -> Uuid
    where
        F: FnMut() + Send + Sync + 'static,
    {
        let uuid = Uuid::new_v4();

        self.callbacks.insert(uuid, Box::new(callback));

        uuid
    }

    fn remove_callback(&self, uuid: Uuid) {
        let _ = self.callbacks.remove(&uuid);
    }
}

fn read_with_retry(path: &PathBuf) -> Result<Vec<u8>, ()> {
    for i in 0..10 {
        match std::fs::read(path) {
            Ok(yaml) => return Ok(yaml),
            Err(e) => {
                if i == 9 {
                    error!("cannot read config: {}", e);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }

    Err(())
}

impl EventHandler for UpdateHandler {
    fn handle_event(&mut self, event: notify::Result<Event>) {
        if let Ok(Event {
            kind: EventKind::Modify(_),
            paths,
            ..
        }) = event
        {
            assert_eq!(paths.len(), 1, "There should be 1 and only 1 path");
            let path = &paths[0];

            let yaml = read_with_retry(path).unwrap();

            let params: SettingsParams = from_slice(&yaml).expect("Failed to parse YAML");
            {
                let mut locked = self.settings.blocking_write();
                locked.params = params.clone();
                locked.changed_params = params;
            }

            for mut item in self.callbacks.iter_mut() {
                let callback = item.as_mut();
                callback();
            }
        }
    }
}

pub struct CallbackHandle {
    handler: UpdateHandler,
    uuid: Uuid,
}

impl CallbackHandle {
    fn new<F>(callback: F, handler: &UpdateHandler) -> Self
    where
        F: FnMut() + Send + Sync + 'static,
    {
        let handler = handler.clone();
        let uuid = handler.add_callback(callback);

        Self { handler, uuid }
    }
}

impl Drop for CallbackHandle {
    fn drop(&mut self) {
        self.handler.remove_callback(self.uuid);
    }
}

#[derive(Default)]
pub struct StaticSettings {
    inner: Option<Settings>,
}

impl StaticSettings {
    pub async fn init(&mut self) -> Result<(), SettingsError> {
        if self.inner.is_none() {
            let directory = PROJECT_DIRS.config_dir();
            let name = FILE_NAME;
            self.inner = Some(Settings::load_or_create(directory, name).await?);
            Ok(())
        } else {
            Ok(())
        }
    }
}

impl Deref for StaticSettings {
    type Target = Settings;

    fn deref(&self) -> &Self::Target {
        if let Some(settings) = &self.inner {
            settings
        } else {
            panic!("Settings have not been initialised yet")
        }
    }
}

impl DerefMut for StaticSettings {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if let Some(settings) = &mut self.inner {
            settings
        } else {
            panic!("Settings have not been initialised yet")
        }
    }
}

#[cfg(test)]
mod tests {
    // use std::sync::atomic::{AtomicBool, Ordering};
    // use std::sync::Arc;

    use crate::settings::*;

    const TEST_FILE: &str = "tfile";

    // Trying to avoid doing too many disk writes in unit tests by performing every test using the
    // same file.
    #[tokio::test]
    async fn all() {
        let tmp = tempfile::tempdir().expect("Failed to create temporary directory");

        // New
        {
            let (settings, _) = SettingsInner::load_or_create(tmp.path(), TEST_FILE)
                .await
                .expect("Failed to create settings object");

            let filename = TEST_FILE.to_string() + FILE_EXTENSION;
            let path = tmp.path().join(filename);
            assert!(path.exists(), "Settings file was not created");

            let params = SettingsParams::default();
            assert_eq!(
                params.threads, settings.threads,
                "Settings do not match default parameters"
            );

            settings.save().await.expect("Failed to save settings");
        }

        // Change, load and callback
        {
            let settings = Settings::load_or_create(tmp.path(), TEST_FILE)
                .await
                .expect("Failed to load settings object");

            let params = SettingsParams::default();
            assert_eq!(
                params.threads,
                settings.read().await.threads,
                "Settings do not match default parameters"
            );

            settings.write().await.threads = 1000000;

            assert_eq!(
                params.threads,
                settings.read().await.threads,
                "Settings were changed without getting applied"
            );

            /* With the PollWatcher this is not working.
             * We need to find a better test method.

            let flag = Arc::new(AtomicBool::new(false));
            let cloned_flag = flag.clone();

            let _unused =
                settings.add_change_callback(move || cloned_flag.store(true, Ordering::SeqCst));
            */

            settings
                .write()
                .await
                .apply()
                .await
                .expect("Failed to apply settings");

            // The watcher must read the file to update the settings, so we must give it some time
            // tokio::time::sleep(std::time::Duration::from_millis(1)).await;

            assert_eq!(
                1000000,
                settings.read().await.threads,
                "Settings were not applied"
            );

            // If this fails with "File not found", should try increasing the sleep duration
            // assert!(flag.load(Ordering::SeqCst), "Callback was not called")
        }
        {
            let (settings, _) = SettingsInner::load_or_create(tmp.path(), TEST_FILE)
                .await
                .expect("Failed to load settings object");

            assert_eq!(1000000, settings.threads, "Settings were not saved");
        }
    }
}
