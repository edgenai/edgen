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

use core::time::Duration;
use std::path::Path;

use futures::Stream;
use serde::Serialize;
use thiserror::Error;

use crate::BoxedFuture;

/// The context tag marking the start of generated dialogue.
pub const ASSISTANT_TAG: &str = "<|ASSISTANT|>";

/// The context tag marking the start of user dialogue.
pub const USER_TAG: &str = "<|USER|>";

/// The context tag marking the start of a tool's output.
pub const TOOL_TAG: &str = "<|TOOL|>";

/// The context tag marking the start of system information.
pub const SYSTEM_TAG: &str = "<|SYSTEM|>";

#[derive(Serialize, Error, Debug)]
pub enum LLMEndpointError {
    #[error("failed to advance context: {0}")]
    Advance(String),
    #[error("failed to load the model: {0}")]
    Load(String),
    #[error("failed to create a new session: {0}")]
    SessionCreationFailed(String),
    #[error("failed to create embeddings: {0}")]
    Embeddings(String), // Embeddings may involve session creation, advancing, and other things, so it should have its own error
}

#[derive(Debug, Clone)]
pub struct CompletionArgs {
    pub prompt: String,
    pub one_shot: bool,
    pub seed: Option<u32>,
    pub frequency_penalty: f32,
}

impl Default for CompletionArgs {
    fn default() -> Self {
        Self {
            prompt: "".to_string(),
            one_shot: false,
            seed: None,
            frequency_penalty: 0.0,
        }
    }
}

/// A large language model endpoint, that is, an object that provides various ways to interact with
/// a large language model.
pub trait LLMEndpoint {
    /// Given a prompt with several arguments, return a [`Box`]ed [`Future`] which may eventually
    /// contain the prompt completion in [`String`] form.
    fn chat_completions<'a>(
        &'a self,
        model_path: impl AsRef<Path> + Send + 'a,
        args: CompletionArgs,
    ) -> BoxedFuture<Result<String, LLMEndpointError>>;

    /// Given a prompt with several arguments, return a [`Box`]ed [`Future`] which may eventually
    /// contain a [`Stream`] of [`String`] chunks of the prompt completion, acquired as they get
    /// processed.
    fn stream_chat_completions<'a>(
        &'a self,
        model_path: impl AsRef<Path> + Send + 'a,
        args: CompletionArgs,
    ) -> BoxedFuture<Result<Box<dyn Stream<Item = String> + Unpin + Send>, LLMEndpointError>>;

    fn embeddings<'a>(
        &'a self,
        model_path: impl AsRef<Path> + Send + 'a,
        inputs: Vec<String>,
    ) -> BoxedFuture<Result<Vec<Vec<f32>>, LLMEndpointError>>;

    /// Unloads everything from memory.
    fn reset(&self);
}

/// Return the [`Duration`] for which a large language model lives while not being used before
/// being unloaded from memory.
pub fn inactive_llm_ttl() -> Duration {
    // TODO this should come from the settings
    Duration::from_secs(5 * 60)
}

/// Return the [`Duration`] for which a large language model session lives while not being used
/// before being unloaded from memory.
pub fn inactive_llm_session_ttl() -> Duration {
    // TODO this should come from the settings
    Duration::from_secs(2 * 60)
}
