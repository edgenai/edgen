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

use edgen_core::llm::{CompletionArgs, LLMEndpoint, LLMEndpointError};
use edgen_rt_llama_cpp::LlamaCppEndpoint;

use crate::model::Model;
use crate::util::StoppingStream;

static ENDPOINT: Lazy<LlamaCppEndpoint> = Lazy::new(Default::default);

pub async fn chat_completion(model: Model, context: String) -> Result<String, LLMEndpointError> {
    let args = CompletionArgs {
        prompt: context,
        seed: 0,
        frequency_penalty: 0.0,
    };

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
    context: String,
) -> Result<StoppingStream<Box<dyn Stream<Item = String> + Unpin + Send>>, LLMEndpointError> {
    let args = CompletionArgs {
        prompt: context,
        seed: 0,
        frequency_penalty: 0.0,
    };

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

pub async fn reset_environment() {
    ENDPOINT.reset()
}
