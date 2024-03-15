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
use tracing::info;

use edgen_core::llm::{CompletionArgs, LLMEndpoint, LLMEndpointError};
use edgen_core::request::{Device, REQUEST_QUEUE};
use edgen_rt_llama_cpp::LlamaCppEndpoint;

use crate::model::Model;
use crate::util::StoppingStream;

static ENDPOINT: Lazy<LlamaCppEndpoint> = Lazy::new(Default::default);

pub async fn chat_completion(
    model: Model,
    prompt: String,
    args: CompletionArgs,
) -> Result<String, LLMEndpointError> {
    let model_path = model
        .file_path()
        .map_err(move |e| LLMEndpointError::Load(e.to_string()))?;
    let passport = ENDPOINT
        .requirements_of(&model_path, Device::Any, &prompt, &args)
        .await?;

    let ticket = REQUEST_QUEUE.enqueue(passport).await?;

    ENDPOINT
        .chat_completions(model_path, Device::Any, &prompt, args, ticket)
        .await
}

pub async fn chat_completion_stream(
    model: Model,
    prompt: String,
    args: CompletionArgs,
) -> Result<StoppingStream<Box<dyn Stream<Item = String> + Unpin + Send>>, LLMEndpointError> {
    let model_path = model
        .file_path()
        .map_err(move |e| LLMEndpointError::Load(e.to_string()))?;
    let passport = ENDPOINT
        .requirements_of(&model_path, Device::Any, &prompt, &args)
        .await?;

    let ticket = REQUEST_QUEUE.enqueue(passport).await?;

    let stream = ENDPOINT
        .stream_chat_completions(model_path, Device::Any, &prompt, args, ticket)
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

pub async fn embeddings(
    model: Model,
    input: Vec<String>,
) -> Result<Vec<Vec<f32>>, LLMEndpointError> {
    ENDPOINT
        .embeddings(
            model
                .file_path()
                .map_err(move |e| LLMEndpointError::Load(e.to_string()))?,
            Device::CPU,
            input,
        )
        .await
}

pub async fn reset_environment() {
    ENDPOINT.reset()
}
