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

//! A fake model RT for chat completions that answers with predefined strings

use std::path::Path;
use std::sync::Arc;

use dashmap::DashMap;
use futures::Stream;
use tracing::info;

use edgen_core::llm::{CompletionArgs, LLMEndpoint, LLMEndpointError};

pub const CAPITAL: &str = "The capital of Canada is Ottawa.";
pub const CAPITAL_OF_PORTUGAL: &str = "The capital of Portugal is Lisbon.";
pub const DEFAULT_ANSWER: &str = "The answer is 42.";
pub const LONG_ANSWER: &str = "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people’s hats off—then, I account it high time to get to sea as soon as I can. There is nothing surprising in this. If they but knew it, almost all men in their degree, some time or other, cherish very nearly the same feelings towards the ocean with me.";

struct ChatFakerModel {}

impl ChatFakerModel {
    async fn new(_path: impl AsRef<Path>) -> Self {
        Self {}
    }

    async fn chat_completions(&self, args: &CompletionArgs) -> Result<String, LLMEndpointError> {
        info!("faking chat completions");
        Ok(completions_for(&args.prompt))
    }

    async fn stream_chat_completions(
        &self,
        args: &CompletionArgs,
    ) -> Result<Box<dyn Stream<Item = String> + Unpin + Send>, LLMEndpointError> {
        info!("faking stream chat completions");
        let msg = completions_for(&args.prompt);
        let toks = streamify(&msg);
        Ok(Box::new(futures::stream::iter(toks.into_iter())))
    }

    //TODO: implement
    async fn embeddings(&self, _inputs: &[String]) -> Result<Vec<Vec<f32>>, LLMEndpointError> {
        info!("faking emeddings");
        Ok(vec![])
    }
}

fn completions_for(prompt: &str) -> String {
    let prompt = prompt.to_lowercase();
    if prompt.contains("capital") {
        if prompt.contains("portugal") {
            return CAPITAL_OF_PORTUGAL.to_string();
        } else {
            return CAPITAL.to_string();
        }
    } else if prompt.contains("long") {
        return LONG_ANSWER.to_string();
    } else {
        return DEFAULT_ANSWER.to_string();
    }
}

fn streamify(msg: &str) -> Vec<String> {
    msg.split_whitespace().map(|s| s.to_string()).collect()
}

/// Faking a large language model endpoint, implementing [`LLMEndpoint`].
pub struct ChatFakerEndpoint {
    /// A map of the models currently loaded into memory, with their path as the key.
    models: Arc<DashMap<String, ChatFakerModel>>,
}

impl ChatFakerEndpoint {
    // This is not strictly needed because we have no Unloading models.
    // Anyway, it looks more like a real model.
    async fn get(
        &self,
        model_path: impl AsRef<Path>,
    ) -> dashmap::mapref::one::Ref<String, ChatFakerModel> {
        let key = model_path.as_ref().to_string_lossy().to_string();

        if !self.models.contains_key(&key) {
            let model = ChatFakerModel::new(model_path).await;
            self.models.insert(key.clone(), model);
        }

        // PANIC SAFETY: Just inserted the element if it isn't already inside the map, so must be present in the map
        self.models.get(&key).unwrap()
    }
}

#[async_trait::async_trait]
impl LLMEndpoint for ChatFakerEndpoint {
    async fn chat_completions(
        &self,
        model_path: impl AsRef<Path> + Send,
        args: CompletionArgs,
    ) -> Result<String, LLMEndpointError> {
        let model = self.get(model_path).await;
        model.chat_completions(&args).await
    }

    async fn stream_chat_completions(
        &self,
        model_path: impl AsRef<Path> + Send,
        args: CompletionArgs,
    ) -> Result<Box<dyn Stream<Item = String> + Unpin + Send>, LLMEndpointError> {
        let model = self.get(model_path).await;
        model.stream_chat_completions(&args).await
    }

    async fn embeddings(
        &self,
        model_path: impl AsRef<Path> + Send,
        inputs: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, LLMEndpointError> {
        let model = self.get(model_path).await;
        model.embeddings(&inputs).await
    }

    fn reset(&self) {
        self.models.clear();
    }
}

impl Default for ChatFakerEndpoint {
    fn default() -> Self {
        let models: Arc<DashMap<String, ChatFakerModel>> = Default::default();
        Self { models }
    }
}
