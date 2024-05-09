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

use either::Either;
use futures::Stream;
use once_cell::sync::Lazy;
use std::ops::Deref;

use edgen_core::llm::{ChatMessage, CompletionArgs, ContentPart, LLMEndpoint, LLMEndpointError};
use edgen_rt_llama_cpp::{LlamaCppEndpoint, LlavaCppEndpoint};

use crate::model::Model;
use crate::util::StoppingStream;

static ENDPOINT: Lazy<LlamaCppEndpoint> = Lazy::new(Default::default);
static MM_ENDPOINT: Lazy<LlavaCppEndpoint> = Lazy::new(Default::default);

fn has_image(args: &CompletionArgs) -> bool {
    for message in args.messages.deref() {
        match message {
            ChatMessage::User {
                content: Either::Right(content),
                ..
            } => {
                for part in content {
                    match part {
                        ContentPart::Text { .. } => continue,
                        ContentPart::ImageUrl { .. } => return true,
                        ContentPart::ImageData { .. } => return true,
                    }
                }
            }
            _ => {}
        }
    }
    false
}

pub async fn chat_completion(
    model: Model,
    args: CompletionArgs,
) -> Result<String, LLMEndpointError> {
    if has_image(&args) {
        MM_ENDPOINT
            .chat_completions(
                model
                    .file_path()
                    .map_err(move |e| LLMEndpointError::Load(e.to_string()))?,
                args,
            )
            .await
    } else {
        ENDPOINT
            .chat_completions(
                model
                    .file_path()
                    .map_err(move |e| LLMEndpointError::Load(e.to_string()))?,
                args,
            )
            .await
    }
}

pub async fn chat_completion_stream(
    model: Model,
    args: CompletionArgs,
) -> Result<StoppingStream<Box<dyn Stream<Item = String> + Unpin + Send>>, LLMEndpointError> {
    let stream = if has_image(&args) {
        MM_ENDPOINT
            .stream_chat_completions(
                model
                    .file_path()
                    .map_err(move |e| LLMEndpointError::Load(e.to_string()))?,
                args,
            )
            .await?
    } else {
        ENDPOINT
            .stream_chat_completions(
                model
                    .file_path()
                    .map_err(move |e| LLMEndpointError::Load(e.to_string()))?,
                args,
            )
            .await?
    };

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

pub async fn reset_environment() {
    ENDPOINT.reset()
}
