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

use std::path::PathBuf;

use serde_derive::Serialize;
use thiserror::Error;
use tracing::warn;
use utoipa::ToSchema;

use crate::status;
use crate::types::Endpoint;

#[derive(Serialize, Error, ToSchema, Debug, PartialEq)]
pub enum ModelError {
    #[error("the provided model file name does does not exist, or isn't a file: ({0})")]
    FileNotFound(String),
    #[error("no repository is available for the specified model: ({0:?})")]
    UnknownModel(ModelKind),
    #[error("error checking remote repository: ({0})")]
    API(String),
    /// error resulting from tokio::JoinError
    #[error("model could not be preloaded because of an internal error (JoinError)")]
    JoinError(String),
    #[error("model was not preloaded before use")]
    NotPreloaded,
}

#[derive(Serialize, ToSchema, Debug, Clone, PartialEq)]
pub enum ModelKind {
    LLM,
    Whisper,
}

#[derive(Debug, PartialEq)]
enum ModelQuantization {
    Default,
}

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
pub struct Model {
    kind: ModelKind,
    quantization: ModelQuantization,
    name: String,
    repo: String,
    dir: PathBuf,
    path: PathBuf,
    preloaded: bool,
}

impl Model {
    pub fn new(kind: ModelKind, model_name: &str, repo: &str, dir: &PathBuf) -> Self {
        let quantization = ModelQuantization::Default;

        let path = dir.join(model_name);

        Self {
            kind,
            quantization,
            name: model_name.to_string(),
            repo: repo.to_string(),
            dir: dir.to_path_buf(),
            path: path,
            preloaded: false,
        }
    }

    /// Checks if a file of the model is already present locally, and if not, downloads it.
    pub async fn preload(&mut self, ep: Endpoint) -> Result<(), ModelError> {
        if self.path.is_file() {
            self.preloaded = true;
            return Ok(());
        }

        if self.name.is_empty() || self.repo.is_empty() {
            return Err(ModelError::UnknownModel(self.kind.clone()));
        }

        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(self.dir.clone())
            .build()
            .map_err(move |e| ModelError::API(e.to_string()))?;
        let api = api.model(self.repo.to_string());

        // progress observer
        let download = hf_hub::Cache::new(self.dir.clone())
            .model(self.repo.to_string())
            .get(&self.name)
            .is_none();
        let size = if download {
            self.get_size(&api).await
        } else {
            None
        };

        let progress_handle = observe_download(ep, &self.dir, size, download).await;

        let name = self.name.clone();
        let download_handle = tokio::spawn(async move {
            if download {
                report_start_of_download(ep).await;
            }

            let path = api
                .get(&name)
                .map_err(move |e| ModelError::API(e.to_string()));

            if download {
                report_end_of_download(ep).await;
            }

            return path;
        });

        let _ = progress_handle
            .await
            .map_err(|e| ModelError::JoinError(e.to_string()))?;
        let path = download_handle
            .await
            .map_err(|e| ModelError::JoinError(e.to_string()))?;

        self.path = path?;
        self.preloaded = true;

        Ok(())
    }

    // get size of the remote file when we download.
    async fn get_size(&self, api: &hf_hub::api::sync::ApiRepo) -> Option<u64> {
        match reqwest::Client::new()
            .get(api.url(&self.name))
            .header("Content-Range", "bytes 0-0")
            .header("Range", "bytes 0-0")
            .send()
            .await
        {
            Ok(metadata) => metadata.content_length(),
            Err(e) => {
                warn!("no metadata for model {}: {:?}", self.name, e);
                None
            }
        }
    }

    /// Returns a [`PathBuf`] pointing to the local model file.
    pub fn file_path(&self) -> Result<PathBuf, ModelError> {
        if self.preloaded {
            return Ok(self.path.clone());
        }

        Err(ModelError::NotPreloaded)
    }
}

async fn observe_download(
    ep: Endpoint,
    dir: &PathBuf,
    size: Option<u64>,
    download: bool,
) -> tokio::task::JoinHandle<()> {
    match ep {
        Endpoint::ChatCompletions => {
            status::observe_chat_completions_progress(dir, size, download).await
        }
        Endpoint::AudioTranscriptions => {
            status::observe_audio_transcriptions_progress(dir, size, download).await
        }
        Endpoint::Embeddings => todo!(),
    }
}

async fn report_start_of_download(ep: Endpoint) {
    match ep {
        Endpoint::ChatCompletions => status::set_chat_completions_download(true).await,
        Endpoint::AudioTranscriptions => status::set_audio_transcriptions_download(true).await,
        Endpoint::Embeddings => todo!(),
    }
}

async fn report_end_of_download(ep: Endpoint) {
    match ep {
        Endpoint::ChatCompletions => {
            status::set_chat_completions_progress(100).await;
            status::set_chat_completions_download(false).await;
        }
        Endpoint::AudioTranscriptions => {
            status::set_audio_transcriptions_progress(100).await;
            status::set_audio_transcriptions_download(false).await;
        }
        Endpoint::Embeddings => todo!(),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use std::path::PathBuf;

    use hf_hub;

    #[test]
    fn llm_new() {
        let model = "model";
        let repo = "repo";
        let dir = PathBuf::from("dir");
        let m = Model::new(ModelKind::LLM, model, repo, &dir);
        assert_eq!(
            m,
            Model {
                kind: ModelKind::LLM,
                quantization: ModelQuantization::Default,
                name: model.to_string(),
                repo: repo.to_string(),
                dir: dir.clone(),
                path: dir.join(model),
                preloaded: false,
            }
        );
        assert_eq!(m.file_path(), Err(ModelError::NotPreloaded));
    }

    #[test]
    fn whisper_new() {
        let model = "model";
        let repo = "repo";
        let dir = PathBuf::from("dir");
        let m = Model::new(ModelKind::Whisper, model, repo, &dir);
        assert_eq!(
            m,
            Model {
                kind: ModelKind::Whisper,
                quantization: ModelQuantization::Default,
                name: model.to_string(),
                repo: repo.to_string(),
                dir: dir.clone(),
                path: dir.join(model),
                preloaded: false,
            }
        );
        assert_eq!(m.file_path(), Err(ModelError::NotPreloaded));
    }

    #[tokio::test]
    async fn preload() {
        let model = "dummy.gguf";
        let repo = "dummy";
        let dir = PathBuf::from("resources");
        let mut m = Model::new(ModelKind::LLM, model, repo, &dir);
        m.preload(Endpoint::ChatCompletions)
            .await
            .expect("model preload failed");
        assert_eq!(
            m,
            Model {
                kind: ModelKind::LLM,
                quantization: ModelQuantization::Default,
                name: model.to_string(),
                repo: repo.to_string(),
                dir: dir.clone(),
                path: dir.join(model),
                preloaded: true,
            }
        );
        assert_eq!(m.file_path(), Ok(m.path));
    }

    #[tokio::test]
    #[ignore]
    // This test tries to connect to huggingface
    // Therefore, we usually ignore it
    async fn get_size() {
        let model = "tinyllama-1.1b-chat-v1.0.Q2_K.gguf";
        let repo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
        let dir = PathBuf::from("dir");
        let m = Model::new(ModelKind::LLM, model, repo, &dir);
        assert_eq!(
            m,
            Model {
                kind: ModelKind::LLM,
                quantization: ModelQuantization::Default,
                name: model.to_string(),
                repo: repo.to_string(),
                dir: dir.clone(),
                path: dir.join(model),
                preloaded: false,
            }
        );
        let api = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(dir.clone())
            .build()
            .expect("ApiBuilder::new() failed");
        let api = api.model(repo.to_string());
        let sz = m.get_size(&api).await;
        assert!(sz.is_some());
        assert_eq!(sz, Some(483116416u64));
    }
}
