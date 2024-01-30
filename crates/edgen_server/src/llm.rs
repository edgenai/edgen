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
 
use futures::Stream;
use once_cell::sync::Lazy;
use serde_derive::Serialize;
use thiserror::Error;

use edgen_core::llm::{CompletionArgs, LLMEndpoint};
use edgen_rt_llama_cpp::LlamaCppEndpoint;

use crate::model::Model;
use crate::util::StoppingStream;

static ENDPOINT: Lazy<LlamaCppEndpoint> = Lazy::new(Default::default);

#[derive(Serialize, Error, Debug)]
pub enum LLMEndpointError {
    #[error("the provided model file name does does not exist, or isn't a file: ({0})")]
    FileNotFound(String),
    #[error("there is no session associated with the provided uuid ({0})")]
    SessionNotFound(String),
    #[error("failed to run inference: {0}")]
    Inference(#[from] edgen_core::llm::LLMEndpointError),
    #[error("failed to load model: {0}")]
    Model(#[from] crate::model::ModelError),
}

// TODO use this
#[allow(dead_code)]
pub async fn chat_completion(model: Model, context: String) -> Result<String, LLMEndpointError> {
    let args = CompletionArgs {
        prompt: context,
        seed: 0,
        frequency_penalty: 0.0,
    };

    Ok(ENDPOINT.chat_completions(model.file_path()?, args).await?)
}

pub async fn chat_completion_stream(
    model: Model,
    context: String,
) -> Result<StoppingStream<Box<dyn Stream<Item = String> + Unpin + Send>>, LLMEndpointError> {
    let args = CompletionArgs {
        prompt: context,
        seed: 0,
        frequency_penalty: 0.0,
    };

    let stream = ENDPOINT
        .stream_chat_completions(model.file_path()?, args)
        .await?;

    Ok(StoppingStream::wrap_with_stop_words(
        stream,
        vec![
            "<|ASSISTANT|>".to_string(),
            "<|USER|>".to_string(),
            "<|TOOL|>".to_string(),
            "<|SYSTEM|>".to_string(),
        ],
    ))
}

pub async fn reset_environment() {
    ENDPOINT.reset()
}
