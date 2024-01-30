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
use utoipa::ToSchema;

#[derive(Serialize, Error, ToSchema, Debug)]
pub enum ModelError {
    #[error("the provided model file name does does not exist, or isn't a file: ({0})")]
    FileNotFound(String),
    #[error("no repository is available for the specified model: ({0:?})")]
    UnknownModel(ModelKind),
    #[error("error checking remote repository: ({0})")]
    API(String),
    #[error("model was not preloaded before use")]
    NotPreloaded,
}

#[derive(Serialize, ToSchema, Debug, Clone, PartialEq)]
pub enum ModelKind {
    LLM,
    Whisper,
    Unknown,
}

enum ModelQuantization {
    Default,
}

#[allow(dead_code)]
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
    pub async fn preload(&mut self) -> Result<(), ModelError> {
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
        let path = api
            .get(&self.name)
            .map_err(move |e| ModelError::API(e.to_string()))?;

        self.path = path;
        self.preloaded = true;

        Ok(())
    }

    /// Returns a [`PathBuf`] pointing to the local model file.
    pub fn file_path(&self) -> Result<PathBuf, ModelError> {
        if self.preloaded {
            return Ok(self.path.clone());
        }

        Err(ModelError::NotPreloaded)
    }
}
