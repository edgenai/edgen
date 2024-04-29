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

use core::fmt::{Display, Formatter};
use core::time::Duration;
use std::collections::HashMap;
use std::path::Path;

use derive_more::{Deref, DerefMut, From};
use either::Either;
use futures::Stream;
use serde::Serialize;
use thiserror::Error;

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
    #[error("unsuitable endpoint for model: {0}")]
    UnsuitableEndpoint(String),
}

/// The plaintext or image content of a [`ChatMessage`] within a [`CreateChatCompletionRequest`].
///
/// This can be plain text or a URL to an image.
#[derive(Debug)]
pub enum ContentPart {
    /// Plain text.
    Text {
        /// The plain text.
        text: String,
    },
    /// A URL to an image.
    ImageUrl {
        /// The URL.
        url: String,

        /// A description of the image behind the URL, if any.
        detail: Option<String>,
    },
}

impl Display for ContentPart {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ContentPart::Text { text } => write!(f, "{}", text),
            ContentPart::ImageUrl { url, detail } => {
                if let Some(detail) = detail {
                    write!(f, "<IMAGE {}> ({})", url, detail)
                } else {
                    write!(f, "<IMAGE {}>", url)
                }
            }
        }
    }
}

/// A description of a function provided to a large language model, to assist it in interacting
/// with the outside world.
///
/// This is included in [`AssistantToolCall`]s within [`ChatMessage`]s.
#[derive(Debug)]
pub struct AssistantFunctionStub {
    /// The name of the function from the assistant's point of view.
    pub name: String,

    /// The arguments passed into the function.
    pub arguments: String,
}

/// A description of a function that an assistant called.
///
/// This is included in [`ChatMessage`]s when the `tool_calls` field is present.
#[derive(Debug)]
pub struct AssistantToolCall {
    /// A unique identifier for the invocation of this function.
    pub id: String,

    /// The type of the invoked tool.
    ///
    /// OpenAI currently specifies this to always be `function`, but more variants may be added
    /// in the future.
    pub type_: String,

    /// The invoked function.
    pub function: AssistantFunctionStub,
}

/// A chat message in a multi-user dialogue.
///
/// This is as context for a [`CreateChatCompletionRequest`].
#[derive(Debug)]
pub enum ChatMessage {
    /// A message from the system. This is typically used to set the initial system prompt; for
    /// example, "you are a helpful assistant".
    System {
        /// The content of the message, if any.
        content: Option<String>,

        /// If present, a name for the system.
        name: Option<String>,
    },
    /// A message from a user.
    User {
        /// The content of the message. This can be a sequence of multiple plain text or image
        /// parts.
        content: Either<String, Vec<ContentPart>>,

        /// If present, a name for the user.
        name: Option<String>,
    },
    /// A message from an assistant.
    Assistant {
        /// The plaintext message of the message, if any.
        content: Option<String>,

        /// The name of the assistant, if any.
        name: Option<String>,

        /// If the assistant used any tools in generating this message, the tools that the assistant
        /// used.
        tool_calls: Option<Vec<AssistantToolCall>>,
    },
    /// A message from a tool accessible by other peers in the dialogue.
    Tool {
        /// The plaintext that the tool generated, if any.
        content: Option<String>,

        /// A unique identifier for the specific invocation that generated this message.
        tool_call_id: String,
    },
}

/*

/// A tool made available to an assistant that invokes a named function.
///
/// This is included in [`ToolStub`]s within [`CreateChatCompletionRequest`]s.
#[derive(Debug)]
pub struct FunctionStub<'a> {
    /// A human-readable description of what the tool does.
    pub description: Option<Cow<'a, str>>,

    /// The name of the tool.
    pub name: Cow<'a, str>,

    /// A [JSON schema][json-schema] describing the parameters that the tool accepts.
    ///
    /// [json-schema]: https://json-schema.org/
    pub parameters: serde_json::Value,
}

/// A tool made available to an assistant.
///
/// At present, this can only be a [`FunctionStub`], but this enum is marked `#[non_exhaustive]`
/// for the (likely) event that more variants are added in the future.
///
/// This is included in [`CreateChatCompletionRequest`]s.
#[derive(Debug)]
#[non_exhaustive]
pub enum ToolStub<'a> {
    /// A named function that can be invoked by an assistant.
    Function {
        /// The named function.
        function: FunctionStub<'a>,
    },
}

*/

/// A sequence of chat messages in a [`CreateChatCompletionRequest`].
///
/// This implements [`Display`] to generate a transcript of the chat messages compatible with most
/// LLaMa-based models.
#[derive(Debug, Default, Deref, DerefMut, From)]
pub struct ChatMessages(
    #[deref]
    #[deref_mut]
    pub Vec<ChatMessage>,
);

impl Display for ChatMessages {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for message in &self.0 {
            match message {
                ChatMessage::System {
                    content: Some(data),
                    ..
                } => {
                    write!(f, "{SYSTEM_TAG}{data}")?;
                }
                ChatMessage::User {
                    content: Either::Left(data),
                    ..
                } => {
                    write!(f, "{USER_TAG}{data}")?;
                }
                ChatMessage::User {
                    content: Either::Right(data),
                    ..
                } => {
                    write!(f, "{USER_TAG}")?;

                    for part in data {
                        write!(f, "{part}")?;
                    }
                }
                ChatMessage::Assistant {
                    content: Some(data),
                    ..
                } => {
                    write!(f, "{ASSISTANT_TAG}{data}")?;
                }
                ChatMessage::Tool {
                    content: Some(data),
                    ..
                } => {
                    write!(f, "{TOOL_TAG}{data}")?;
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// A request to generate chat completions for the provided context.
#[derive(Debug)]
pub struct CompletionArgs {
    /// The messages that have been sent in the dialogue so far.
    pub messages: ChatMessages,

    /// A number in `[-2.0, 2.0]`. A higher number decreases the likelihood that the model
    /// repeats itself.
    pub frequency_penalty: Option<f32>,

    /// A map of token IDs to `[-100.0, +100.0]`. Adds a percentage bias to those tokens before
    /// sampling; a value of `-100.0` prevents the token from being selected at all.
    ///
    /// You could use this to, for example, prevent the model from emitting profanity.
    pub logit_bias: Option<HashMap<u32, f32>>,

    /// The maximum number of tokens to generate. If `None`, terminates at the first stop token
    /// or the end of sentence.
    pub max_tokens: Option<u32>,

    /// How many choices to generate for each token in the output. `1` by default. You can use
    /// this to generate several sets of completions for the same prompt.
    pub n: Option<u32>,

    /// A number in `[-2.0, 2.0]`. Positive values "increase the model's likelihood to talk about
    /// new topics."
    pub presence_penalty: Option<f32>,

    /// An RNG seed for the session. Random by default.
    pub seed: Option<u32>,

    /// A stop phrase or set of stop phrases.
    ///
    /// The server will pause emitting completions if it appears to be generating a stop phrase,
    /// and will terminate completions if a full stop phrase is detected.
    ///
    /// Stop phrases are never emitted to the client.
    pub stop: Option<Either<String, Vec<String>>>,

    /// The sampling temperature, in `[0.0, 2.0]`. Higher values make the output more random.
    pub temperature: Option<f32>,

    /// Nucleus sampling. If you set this value to 10%, only the top 10% of tokens are used for
    /// sampling, preventing sampling of very low-probability tokens.
    pub top_p: Option<f32>,

    /// A list of tools made available to the model.
    // pub tools: Option<Vec<ToolStub<'a>>>,

    /// If present, the tool that the user has chosen to use.
    ///
    /// OpenAI states:
    ///
    /// - `none` prevents any tool from being used,
    /// - `auto` allows any tool to be used, or
    /// - you can provide a description of the tool entirely instead of a name.
    // pub tool_choice: Option<Either<Cow<'a, str>, ToolStub<'a>>>,

    /// Indicate if this is an isolated request, with no associated past or future context. This may allow for
    /// optimisations in some implementations. Default: `false`
    pub one_shot: Option<bool>,

    /// A hint for how big a context will be.
    ///
    /// # Warning
    /// An unsound hint may severely drop performance and/or inference quality, and in some cases even cause Edgen
    /// to crash. Do not set this value unless you know what you are doing.
    pub context_hint: Option<u32>,
}

/// A large language model endpoint, that is, an object that provides various ways to interact with
/// a large language model.
#[async_trait::async_trait]
pub trait LLMEndpoint {
    /// Given a prompt with several arguments, return a prompt completion in [`String`] form.
    async fn chat_completions(
        &self,
        model_path: impl AsRef<Path> + Send,
        args: CompletionArgs,
    ) -> Result<String, LLMEndpointError>;

    /// Given a prompt with several arguments, return a [`Stream`] of [`String`] chunks of the
    /// prompt completion, acquired as they get processed.
    async fn stream_chat_completions(
        &self,
        model_path: impl AsRef<Path> + Send,
        args: CompletionArgs,
    ) -> Result<Box<dyn Stream<Item = String> + Unpin + Send>, LLMEndpointError>;

    async fn embeddings(
        &self,
        model_path: impl AsRef<Path> + Send,
        inputs: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, LLMEndpointError>;

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
