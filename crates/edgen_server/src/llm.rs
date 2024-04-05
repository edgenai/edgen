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
use edgen_core::request::{DeviceId, REQUEST_QUEUE};
use edgen_rt_llama_cpp::LlamaCppEndpoint;

use crate::model::Model;
use crate::util::StoppingStream;

static ENDPOINT: Lazy<LlamaCppEndpoint> = Lazy::new(move || {
    let endpoint = LlamaCppEndpoint::default();
    REQUEST_QUEUE.register_user(endpoint.resource_user());
    endpoint
});

pub async fn chat_completion(
    model: Model,
    prompt: String,
    args: CompletionArgs,
) -> Result<String, LLMEndpointError> {
    let model_path = model
        .file_path()
        .map_err(move |e| LLMEndpointError::Load(e.to_string()))?;
    let mut passport = ENDPOINT
        .completion_requirements(&model_path, DeviceId::Any, &prompt, &args)
        .await?;

    loop {
        let ticket = REQUEST_QUEUE.enqueue(passport).await?;

        let res = ENDPOINT
            .chat_completions(&model_path, &prompt, &args, ticket)
            .await;

        match res {
            Ok(completion) => {
                return Ok(completion);
            }
            Err(LLMEndpointError::Retry(new_passport)) => passport = new_passport,
            Err(e) => {
                return Err(e);
            }
        }
    }
}

pub async fn chat_completion_stream(
    model: Model,
    prompt: String,
    args: CompletionArgs,
) -> Result<StoppingStream<Box<dyn Stream<Item = String> + Unpin + Send>>, LLMEndpointError> {
    let model_path = model
        .file_path()
        .map_err(move |e| LLMEndpointError::Load(e.to_string()))?;
    let mut passport = ENDPOINT
        .completion_requirements(&model_path, DeviceId::Any, &prompt, &args)
        .await?;

    loop {
        let ticket = REQUEST_QUEUE.enqueue(passport).await?;

        let res = ENDPOINT
            .stream_chat_completions(&model_path, &prompt, &args, ticket)
            .await;

        match res {
            Ok(stream) => {
                return Ok(StoppingStream::wrap_with_stop_words(
                    stream,
                    vec![
                        "<|ASSISTANT|>".to_string(),
                        "<|USER|>".to_string(),
                        "<|TOOL|>".to_string(),
                        "<|SYSTEM|>".to_string(),
                    ],
                ));
            }
            Err(LLMEndpointError::Retry(new_passport)) => passport = new_passport,
            Err(e) => {
                return Err(e);
            }
        }
    }
}

pub async fn embeddings(
    model: Model,
    input: Vec<String>,
) -> Result<Vec<Vec<f32>>, LLMEndpointError> {
    let model_path = model
        .file_path()
        .map_err(move |e| LLMEndpointError::Load(e.to_string()))?;
    let mut passport = ENDPOINT
        .embedding_requirements(&model_path, DeviceId::Any, &input)
        .await?;

    loop {
        let ticket = REQUEST_QUEUE.enqueue(passport).await?;

        let res = ENDPOINT.embeddings(&model_path, &input, ticket).await;

        match res {
            Ok(embeddings) => {
                return Ok(embeddings);
            }
            Err(LLMEndpointError::Retry(new_passport)) => passport = new_passport,
            Err(e) => {
                return Err(e);
            }
        }
    }
}

pub async fn reset_environment() {
    ENDPOINT.reset()
}
