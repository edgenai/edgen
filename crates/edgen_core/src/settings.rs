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
use tracing::{error, info, warn};
use uuid::Uuid;

/// The file extension of a YAML file, which is the format used to store settings.
const FILE_EXTENSION: &str = ".yaml";
const FILE_NAME: &str = "edgen.conf";

// TODO look for a better way to do this, since [Settings] already uses a lock
pub static SETTINGS: Lazy<RwLock<StaticSettings>> = Lazy::new(Default::default);

/// The configuration, and data directories for Edgen.
pub static PROJECT_DIRS: Lazy<ProjectDirs> =
    Lazy::new(|| ProjectDirs::from("com", "EdgenAI", "Edgen").unwrap());
pub static CONFIG_FILE: Lazy<PathBuf> = Lazy::new(build_config_file_path);

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
        std::fs::create_dir_all(config_dir)?;
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
    block_on(async { StaticSettings { inner: None }.init().await })?;

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

/// Helper to get the chat completions model directory.
pub async fn chat_completions_dir() -> String {
    SETTINGS
        .read()
        .await
        .read()
        .await
        .chat_completions_models_dir
        .trim()
        .to_string()
}

/// Helper to get the audio transcriptions model directory.
pub async fn audio_transcriptions_dir() -> String {
    SETTINGS
        .read()
        .await
        .read()
        .await
        .audio_transcriptions_models_dir
        .trim()
        .to_string()
}

/// Helper to get the embeddings model directory.
pub async fn embeddings_dir() -> String {
    SETTINGS
        .read()
        .await
        .read()
        .await
        .embeddings_models_dir
        .trim()
        .to_string()
}

/// Helper to get the chat completions model name.
pub async fn chat_completions_name() -> String {
    SETTINGS
        .read()
        .await
        .read()
        .await
        .chat_completions_model_name
        .trim()
        .to_string()
}

/// Helper to get the audio transcriptions model name.
pub async fn audio_transcriptions_name() -> String {
    SETTINGS
        .read()
        .await
        .read()
        .await
        .audio_transcriptions_model_name
        .trim()
        .to_string()
}

/// Helper to get the embeddings model name.
pub async fn embeddings_name() -> String {
    SETTINGS
        .read()
        .await
        .read()
        .await
        .embeddings_model_name
        .trim()
        .to_string()
}

/// Helper to get the chat completions repo.
pub async fn chat_completions_repo() -> String {
    SETTINGS
        .read()
        .await
        .read()
        .await
        .chat_completions_model_repo
        .trim()
        .to_string()
}

/// Helper to get the audio transcriptions model repo.
pub async fn audio_transcriptions_repo() -> String {
    SETTINGS
        .read()
        .await
        .read()
        .await
        .audio_transcriptions_model_repo
        .trim()
        .to_string()
}

/// Helper to get the embeddings model repo.
pub async fn embeddings_repo() -> String {
    SETTINGS
        .read()
        .await
        .read()
        .await
        .embeddings_model_repo
        .trim()
        .to_string()
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

/// A device allocation/execution policy.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
#[serde(rename_all = "snake_case")]
pub enum DevicePolicy {
    /// Always allocate and run on the system CPU.
    AlwaysCpu {
        /// Upon reaching the host memory limit, allocate and execute on a device if possible
        overflow_to_device: bool,
    },

    /// Always allocated and run on acceleration hardware.
    AlwaysDevice {
        /// Upon reaching the device memory limit, allocate on system memory and execute on the CPU
        /// if possible
        overflow_to_cpu: bool,
    },
    // TODO add other policies like: modelthreshold, devicememorythreshold, requestbased, etc
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

    // TODO temporary, until the model parameter in incoming requests can be parsed into local paths
    pub embeddings_models_dir: String,
    /// The embeddings model that Edgen will use when the user does not provide a model
    pub embeddings_model_name: String,
    /// The embeddings repo that Edgen will use for downloads
    pub embeddings_model_repo: String,

    /// The policy used to decided if models/session should be allocated and run on acceleration
    /// hardware.
    pub gpu_policy: DevicePolicy,

    /// The maximum size, in bytes, any request can have. This is most relevant in requests with files, such as audio
    /// transcriptions.
    pub max_request_size: usize,

    /// The default maximum number of tokens a Large Language Model context will have (where applicable).
    pub llm_default_context_size: u32,
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
        let chat_completions_dir =
            data_dir.join(&join_path_components(&["models", "chat", "completions"]));
        let audio_transcriptions_dir = data_dir.join(&join_path_components(&[
            "models",
            "audio",
            "transcriptions",
        ]));
        let embeddings_dir = data_dir.join(&join_path_components(&["models", "embeddings"]));

        let chat_completions_str = chat_completions_dir.into_os_string().into_string().unwrap();
        let audio_transcriptions_str = audio_transcriptions_dir
            .into_os_string()
            .into_string()
            .unwrap();
        let embeddings_str = embeddings_dir.into_os_string().into_string().unwrap();

        let cpus = num_cpus::get_physical();
        let threads = if cpus > 1 { cpus - 1 } else { 1 };

        // if changed, please update docs at docs/src/app/documentation/configuration/page.mdx
        Self {
            threads: threads as u32,
            default_uri: "http://127.0.0.1:33322".to_string(),
            chat_completions_model_name: "neural-chat-7b-v3-3.Q4_K_M.gguf".to_string(),
            chat_completions_model_repo: "TheBloke/neural-chat-7B-v3-3-GGUF".to_string(),
            chat_completions_models_dir: chat_completions_str,
            audio_transcriptions_model_name: "ggml-distil-small.en.bin".to_string(),
            audio_transcriptions_model_repo: "distil-whisper/distil-small.en".to_string(),
            audio_transcriptions_models_dir: audio_transcriptions_str,
            embeddings_model_name: "nomic-embed-text-v1.5.f16.gguf".to_string(),
            embeddings_model_repo: "nomic-ai/nomic-embed-text-v1.5-GGUF".to_string(),
            embeddings_models_dir: embeddings_str,
            // TODO detect if the system has acceleration hardware to decide the default
            gpu_policy: DevicePolicy::AlwaysDevice {
                overflow_to_cpu: true,
            },
            max_request_size: 1024 * 1014 * 100, // 100 MB
            llm_default_context_size: 4096,
        }
    }
}

fn join_path_components(comps: &[&str]) -> PathBuf {
    comps.iter().collect::<PathBuf>()
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
    _watcher: PollWatcher,
    // we use a PollWatcher because it observes the path, not the inode
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
                    warn!("cannot read config: {}", e);
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

            let yaml = read_with_retry(path);

            // a user may have deleted the config file by accident,
            // ignore it until it is readable again.
            if yaml.is_err() {
                return;
            }
            let yaml = yaml.unwrap();

            // likewise, a user may have invalidated the config by accident,
            // ignore it until it is readable again.
            let params: Result<SettingsParams, serde_yaml::Error> = from_slice(&yaml);
            if params.is_err() {
                warn!("cannot parse config: {:?}", params.unwrap_err());
                return;
            }
            let params = params.unwrap();

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
    use std::ffi::OsString;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use crate::settings::*;

    const TEST_FILE: &str = "tfile";

    #[test]
    fn test_join_path_components() {
        for i in 0..4 {
            let (have, expected) = if i == 0 {
                (
                    join_path_components(&["path", "to", "my", "file.ext"]).into_os_string(),
                    #[cfg(target_family = "windows")]
                    OsString::from("path\\to\\my\\file.ext"),
                    #[cfg(not(target_family = "windows"))]
                    OsString::from("path/to/my/file.ext"),
                )
            } else if i == 1 {
                (
                    #[cfg(target_family = "windows")]
                    join_path_components(&["c:\\absolute", "path", "to", "my", "file.ext"])
                        .into_os_string(),
                    #[cfg(not(target_family = "windows"))]
                    join_path_components(&["/absolute", "path", "to", "my", "file.ext"])
                        .into_os_string(),
                    #[cfg(target_family = "windows")]
                    OsString::from("c:\\absolute\\path\\to\\my\\file.ext"),
                    #[cfg(not(target_family = "windows"))]
                    OsString::from("/absolute/path/to/my/file.ext"),
                )
            } else if i == 2 {
                (
                    join_path_components(&["file.ext"]).into_os_string(),
                    OsString::from("file.ext"),
                )
            } else {
                (OsString::from(""), OsString::from(""))
            };
            println!("Path from components: {:?}", have);
            assert_eq!(have, expected);
        }
    }

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

            tokio::time::sleep(std::time::Duration::from_secs(1)).await;

            settings.write().await.threads = 1000000;

            assert_eq!(
                params.threads,
                settings.read().await.threads,
                "Settings were changed without getting applied"
            );

            let flag = Arc::new(AtomicBool::new(false));
            let cloned_flag = flag.clone();

            let _unused = settings.add_change_callback(move || {
                println!("settings callback running");
                cloned_flag.store(true, Ordering::SeqCst)
            });

            settings
                .write()
                .await
                .apply()
                .await
                .expect("Failed to apply settings");

            // The watcher must read the file to update the settings, so we must give it some time
            tokio::time::sleep(std::time::Duration::from_secs(4)).await;

            assert_eq!(
                1000000,
                settings.read().await.threads,
                "Settings were not applied"
            );

            // If this fails with "File not found", should try increasing the sleep duration
            assert!(flag.load(Ordering::SeqCst), "Callback was not called")
        }
        {
            let (settings, _) = SettingsInner::load_or_create(tmp.path(), TEST_FILE)
                .await
                .expect("Failed to load settings object");

            assert_eq!(1000000, settings.threads, "Settings were not saved");
        }
    }
}
