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

use crate::request::Passport;
use futures::Stream;
use serde::Serialize;
use thiserror::Error;

use crate::settings::SETTINGS;

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
    pub one_shot: bool,
    pub seed: Option<u32>,
    pub frequency_penalty: f32,
    pub context_hint: Option<u32>,
}

impl Default for CompletionArgs {
    fn default() -> Self {
        Self {
            one_shot: false,
            seed: None,
            frequency_penalty: 0.0,
            context_hint: None,
        }
    }
}

/// A large language model endpoint, that is, an object that provides various ways to interact with
/// a large language model.
#[async_trait::async_trait]
pub trait LLMEndpoint {
    /// Given a prompt with several arguments, return the prompt completion in [`String`] form.
    async fn chat_completions(
        &self,
        model_path: impl AsRef<Path> + Send,
        prompt: &str,
        args: CompletionArgs,
    ) -> Result<String, LLMEndpointError>;

    /// Given a prompt with several arguments, return a [`Box`]ed [`Stream`] of [`String`] chunks of the prompt completion,
    /// acquired as they get processed.
    async fn stream_chat_completions(
        &self,
        model_path: impl AsRef<Path> + Send,
        prompt: &str,
        args: CompletionArgs,
    ) -> Result<Box<dyn Stream<Item = String> + Unpin + Send>, LLMEndpointError>;

    /// Runs embeddings inference for the given inputs, returning the result.
    async fn embeddings(
        &self,
        model_path: impl AsRef<Path> + Send,
        inputs: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, LLMEndpointError>;

    /// Return an estimation of the resources required to process inference given its arguments.
    async fn requirements_of(
        &self,
        model_path: impl AsRef<Path> + Send,
        prompt: &str,
        args: &CompletionArgs,
    ) -> Result<Passport, LLMEndpointError>;

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

/// Default LLM context settings retrived from [` ].
pub struct ContextSettings {
    pub threads: u32,
    pub size: u32,
}

/// Returns some default parameters for Large Language Models contexts retrieved from the global settings.
pub async fn default_context_settings() -> ContextSettings {
    let global_guard = SETTINGS.read().await;
    let instance_guard = global_guard.read().await;

    ContextSettings {
        threads: instance_guard.auto_threads(false),
        size: instance_guard.llm_default_context_size,
    }
}
