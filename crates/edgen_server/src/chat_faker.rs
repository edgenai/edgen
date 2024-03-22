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

//! Endpoint for the chat faker model RT

use futures::Stream;
use once_cell::sync::Lazy;

use edgen_core::llm::{CompletionArgs, LLMEndpoint, LLMEndpointError};
use edgen_rt_chat_faker::ChatFakerEndpoint;

use crate::model::Model;
use crate::util::StoppingStream;

static ENDPOINT: Lazy<ChatFakerEndpoint> = Lazy::new(Default::default);

pub async fn chat_completion(
    model: Model,
    args: CompletionArgs,
) -> Result<String, LLMEndpointError> {
    ENDPOINT
        .chat_completions(
            model
                .file_path()
                .map_err(move |e| LLMEndpointError::Load(e.to_string()))?,
            args,
        )
        .await
}

pub async fn chat_completion_stream(
    model: Model,
    args: CompletionArgs,
) -> Result<StoppingStream<Box<dyn Stream<Item = String> + Unpin + Send>>, LLMEndpointError> {
    let stream = ENDPOINT
        .stream_chat_completions(
            model
                .file_path()
                .map_err(move |e| LLMEndpointError::Load(e.to_string()))?,
            args,
        )
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
            input,
        )
        .await
}

// Not needed. Just for completeness.
#[allow(dead_code)]
pub async fn reset_environment() {
    ENDPOINT.reset()
}
