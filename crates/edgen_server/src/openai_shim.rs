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

//! JSON structures and Axum endpoints compatible with [OpenAI's API][openai], providing a thin
//! shim between an HTTP REST API server to Edgen's Protobuf-based messaging system.
//!
//! [openai]: https://beta.openai.com/docs/api-reference

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

use axum::http::StatusCode;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response, Sse};
use axum::Json;
use axum_typed_multipart::{FieldData, TryFromMultipart, TypedMultipart};
use derive_more::{Deref, DerefMut, From};
use edgen_core::llm::{CompletionArgs, LLMEndpointError};
use either::Either;
use futures::{Stream, StreamExt, TryStream};
use serde_derive::{Deserialize, Serialize};
use thiserror::Error;
use time::OffsetDateTime;
use tinyvec::{tiny_vec, TinyVec};
use tracing::error;
use utoipa::ToSchema;
use uuid::Uuid;

use edgen_core::settings::SETTINGS;
use edgen_core::whisper::WhisperEndpointError;

use crate::model::{Model, ModelError, ModelKind};

/// The plaintext or image content of a [`ChatMessage`] within a [`CreateChatCompletionRequest`].
///
/// This can be plain text or a URL to an image.
///
/// See [the documentation for creating chat completions][openai] for more details.
///
/// [openai]: https://platform.openai.com/docs/api-reference/chat/create
#[derive(Serialize, Deserialize, ToSchema)]
#[serde(tag = "type")]
pub enum ContentPart<'a> {
    /// Plain text.
    #[serde(rename = "text")]
    Text {
        /// The plain text.
        text: Cow<'a, str>,
    },
    /// A URL to an image.
    #[serde(rename = "image_url")]
    ImageUrl {
        /// The URL.
        url: Cow<'a, str>,

        /// A description of the image behind the URL, if any.
        detail: Option<Cow<'a, str>>,
    },
}

impl<'a> Display for ContentPart<'a> {
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
///
/// See [the documentation for creating chat completions][openai] for more details.
///
/// [openai]: https://platform.openai.com/docs/api-reference/chat/create
#[derive(Serialize, Deserialize, ToSchema)]
pub struct AssistantFunctionStub<'a> {
    /// The name of the function from the assistant's point of view.
    pub name: Cow<'a, str>,

    /// The arguments passed into the function.
    pub arguments: Cow<'a, str>,
}

/// A description of a function that an assistant called.
///
/// This is included in [`ChatMessage`]s when the `tool_calls` field is present.
#[derive(Serialize, Deserialize, ToSchema)]
pub struct AssistantToolCall<'a> {
    /// A unique identifier for the invocation of this function.
    pub id: Cow<'a, str>,

    /// The type of the invoked tool.
    ///
    /// OpenAI currently specifies this to always be `function`, but more variants may be added
    /// in the future.
    #[serde(rename = "type")]
    pub type_: Cow<'a, str>,

    /// The invoked function.
    pub function: AssistantFunctionStub<'a>,
}

/// A chat message in a multi-user dialogue.
///
/// This is as context for a [`CreateChatCompletionRequest`].
///
/// See [the documentation for creating chat completions][openai] for more details.
///
/// [openai]: https://platform.openai.com/docs/api-reference/chat/create
#[derive(Serialize, Deserialize, ToSchema)]
#[serde(tag = "role")]
pub enum ChatMessage<'a> {
    /// A message from the system. This is typically used to set the initial system prompt; for
    /// example, "you are a helpful assistant".
    #[serde(rename = "system")]
    System {
        /// The content of the message, if any.
        content: Option<Cow<'a, str>>,

        /// If present, a name for the system.
        name: Option<Cow<'a, str>>,
    },
    /// A message from a user.
    #[serde(rename = "user")]
    User {
        /// The content of the message. This can be a sequence of multiple plain text or image
        /// parts.
        #[serde(with = "either::serde_untagged")]
        #[schema(value_type = String)]
        content: Either<Cow<'a, str>, Vec<ContentPart<'a>>>,

        /// If present, a name for the user.
        name: Option<Cow<'a, str>>,
    },
    /// A message from an assistant.
    #[serde(rename = "assistant")]
    Assistant {
        /// The plaintext message of the message, if any.
        content: Option<Cow<'a, str>>,

        /// The name of the assistant, if any.
        name: Option<Cow<'a, str>>,

        /// If the assistant used any tools in generating this message, the tools that the assistant
        /// used.
        tool_calls: Option<Vec<AssistantToolCall<'a>>>,
    },
    /// A message from a tool accessible by other peers in the dialogue.
    #[serde(rename = "tool")]
    Tool {
        /// The plaintext that the tool generated, if any.
        content: Option<Cow<'a, str>>,

        /// A unique identifier for the specific invocation that generated this message.
        tool_call_id: Cow<'a, str>,
    },
}

/// A tool made available to an assistant that invokes a named function.
///
/// This is included in [`ToolStub`]s within [`CreateChatCompletionRequest`]s.
///
/// See [the documentation for creating chat completions][openai] for more details.
///
/// [openai]: https://platform.openai.com/docs/api-reference/chat/create
#[derive(Serialize, Deserialize, ToSchema)]
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
///
/// See [the documentation for creating chat completions][openai] for more details.
///
/// [openai]: https://platform.openai.com/docs/api-reference/chat/create
#[derive(Serialize, Deserialize, ToSchema)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum ToolStub<'a> {
    /// A named function that can be invoked by an assistant.
    #[serde(rename = "function")]
    Function {
        /// The named function.
        function: FunctionStub<'a>,
    },
}

/// A sequence of chat messages in a [`CreateChatCompletionRequest`].
///
/// This implements [`Display`] to generate a transcript of the chat messages compatible with most
/// LLaMa-based models.
#[derive(Serialize, Deserialize, Default, Deref, DerefMut, From, ToSchema)]
pub struct ChatMessages<'a>(
    #[deref]
    #[deref_mut]
    Vec<ChatMessage<'a>>,
);

impl<'a> Display for ChatMessages<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for message in &self.0 {
            match message {
                ChatMessage::System {
                    content: Some(data),
                    ..
                } => {
                    write!(f, "<|SYSTEM|>{data}")?;
                }
                ChatMessage::User {
                    content: Either::Left(data),
                    ..
                } => {
                    write!(f, "<|USER|>{data}")?;
                }
                ChatMessage::User {
                    content: Either::Right(data),
                    ..
                } => {
                    write!(f, "<|USER|>")?;

                    for part in data {
                        write!(f, "{part}")?;
                    }
                }
                ChatMessage::Assistant {
                    content: Some(data),
                    ..
                } => {
                    write!(f, "<|ASSISTANT|>{data}")?;
                }
                ChatMessage::Tool {
                    content: Some(data),
                    ..
                } => {
                    write!(f, "<|TOOL|>{data}")?;
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// A request to generate chat completions for the provided context.
///
/// An `axum` handler, [`chat_completions`][chat_completions], is provided to handle this request.
///
/// See [the documentation for creating chat completions][openai] for more details.
///
/// [chat_completions]: fn.chat_completions.html
/// [openai]: https://platform.openai.com/docs/api-reference/chat/create
#[derive(Serialize, Deserialize, ToSchema)]
pub struct CreateChatCompletionRequest<'a> {
    /// The messages that have been sent in the dialogue so far.
    #[serde(default)]
    pub messages: ChatMessages<'a>,

    /// The model to use for generating completions.
    pub model: Cow<'a, str>,

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
    pub n: Option<f32>,

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
    #[serde(default, with = "either::serde_untagged_optional")]
    #[schema(value_type = String)]
    pub stop: Option<Either<Cow<'a, str>, Vec<Cow<'a, str>>>>,

    /// If `true`, emit [`ChatCompletionChunk`]s instead of a single [`ChatCompletion`].
    ///
    /// You can use this to live-stream completions to a client.
    pub stream: Option<bool>,

    /// The format of the response stream.
    ///
    /// This is always assumed to be JSON, which is non-conformant with the OpenAI spec.
    pub response_format: Option<serde_json::Value>,

    /// The sampling temperature, in `[0.0, 2.0]`. Higher values make the output more random.
    pub temperature: Option<f32>,

    /// Nucleus sampling. If you set this value to 10%, only the top 10% of tokens are used for
    /// sampling, preventing sampling of very low-probability tokens.
    pub top_p: Option<f32>,

    /// A list of tools made available to the model.
    pub tools: Option<Vec<ToolStub<'a>>>,

    /// If present, the tool that the user has chosen to use.
    ///
    /// OpenAI states:
    ///
    /// - `none` prevents any tool from being used,
    /// - `auto` allows any tool to be used, or
    /// - you can provide a description of the tool entirely instead of a name.
    #[serde(default, with = "either::serde_untagged_optional")]
    #[schema(value_type = String)]
    pub tool_choice: Option<Either<Cow<'a, str>, ToolStub<'a>>>,

    /// A unique identifier for the _end user_ creating this request. This is used for telemetry
    /// and user tracking, and is unused within Edgen.
    pub user: Option<Cow<'a, str>>,

    /// Indicate if this is an isolated request, with no associated past or future context. This may allow for
    /// optimisations in some implementations. Default: `false`
    pub one_shot: Option<bool>,
}

/// A message in a chat completion.
///
/// This is included in [`ChatCompletion`]s.
///
/// See [the documentation for creating chat completions][openai] for more details.
///
/// [openai]: https://platform.openai.com/docs/api-reference/chat/create
#[derive(Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionChoice<'a> {
    /// The plaintext of the generated message.
    pub message: ChatMessage<'a>,

    /// If present, the reason that generation terminated at this choice.
    ///
    /// This can be:
    ///
    /// - `length`, indicating that the length cutoff was reached, or
    /// - `stop`, indicating that a stop word was reached.
    pub finish_reason: Option<Cow<'a, str>>,

    /// The index of this choice.
    pub index: i32,
}

/// Statistics about a completed chat completion.
///
/// See [the documentation for creating chat completions][openai] for more details.
///
/// [openai]: https://platform.openai.com/docs/api-reference/completions/object
#[derive(Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionUsage {
    /// The number of generated tokens.
    pub completion_tokens: u32,

    /// The number of tokens in the prompt.
    pub prompt_tokens: u32,

    /// `completion_tokens` + `prompt_tokens`; the total number of tokens in the dialogue
    /// so far.
    pub total_tokens: u32,
}

/// A fully generated chat completion.
#[derive(Serialize, Deserialize, ToSchema)]
pub struct ChatCompletion<'a> {
    /// A unique identifier for this completion.
    pub id: Cow<'a, str>,

    /// The tokens generated by the model.
    pub choices: Vec<ChatCompletionChoice<'a>>,

    /// The UNIX timestamp at which the completion was generated.
    pub created: i64,

    /// The model that generated the completion.
    pub model: Cow<'a, str>,

    /// A unique identifier for the backend configuration that generated the completion.
    pub system_fingerprint: Cow<'a, str>,

    /// The object type. This is always `text_completion`.
    pub object: Cow<'a, str>,

    /// Usage information about this completion.
    pub usage: ChatCompletionUsage,
}

/// A delta-encoded difference for an ongoing, stream-mode chat completion.
#[derive(Serialize, Deserialize, Default, ToSchema)]
pub struct ChatCompletionChunkDelta<'a> {
    /// If present, new content added to the end of the completion stream.
    pub content: Option<Cow<'a, str>>,

    /// If present, `content` is being generated under a new role.
    pub role: Option<Cow<'a, str>>,
}

/// A chunk of a stream-mode chat completion.
#[derive(Serialize, Deserialize, Default, ToSchema)]
pub struct ChatCompletionChunkChoice<'a> {
    /// The delta-encoded difference to append to the completion stream.
    pub delta: ChatCompletionChunkDelta<'a>,

    /// If present, this choice terminated the completion stream. The following variants
    /// are available:
    ///
    /// - `length`, indicating that the length cutoff was reached, or
    /// - `stop`, indicating that a stop word was reached.
    pub finish_reason: Option<Cow<'a, str>>,

    /// The index of this choice. If `n` was set in [`CreateChatCompletionRequest`], this is
    /// which stream this choice belongs to.
    pub index: u32,
}

/// A chunk generated in streaming mode from a [`CreateChatCompletionRequest`].
#[derive(Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionChunk<'a> {
    /// A unique identifier for this chunk.
    pub id: Cow<'a, str>,

    /// The tokens generated by the model.
    #[schema(value_type = [ChatCompletionChunkChoice])]
    pub choices: TinyVec<[ChatCompletionChunkChoice<'a>; 1]>,

    /// The UNIX timestamp at which the chunk was generated.
    pub created: i64,

    /// The model that generated the chunk.
    pub model: Cow<'a, str>,

    /// A unique identifier for the backend configuration that generated the chunk.
    pub system_fingerprint: Cow<'a, str>,

    /// The object type. This is always `text_completion`.
    pub object: Cow<'a, str>,
}

/// An error condition raised by the chat completion API.
///
/// This is **not normative** with OpenAI's specification, which does not document any specific
/// failure modes.
#[derive(Serialize, Error, ToSchema, Debug)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "error")]
pub enum ChatCompletionError {
    /// The provided model could not be found on the local system.
    #[error("no such model: {model_name}")]
    NoSuchModel {
        /// The name of the model.
        model_name: String,
    },

    /// The provided model name contains prohibited characters.
    #[error("model {model_name} could not be fetched from the system: {reason}")]
    ProhibitedName {
        /// The name of the model provided.
        model_name: String,

        /// A human-readable error message.
        reason: Cow<'static, str>,
    },

    /// An error occurred on the other side of an FFI boundary.
    #[error("an error occurred on the other side of a C FFI boundary; check `tracing`")]
    Ffi,

    /// An error occurred while processing the request to this endpoint.
    #[error("an error occurred while processing the request: {0}")]
    Endpoint(#[from] LLMEndpointError),
}

impl IntoResponse for ChatCompletionError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(self)).into_response()
    }
}

/// The return type of [`chat_completions`].
///
/// Contains either a [`Stream`] of [`Event`]s or the [`Json`] of a [`ChatCompletion`].
#[derive(ToSchema)]
enum ChatCompletionResponse<'a, S>
where
    S: TryStream<Ok = Event> + Send + 'static,
{
    Stream(Sse<S>),
    Full(Json<ChatCompletion<'a>>),
}

impl<'a, S, E> IntoResponse for ChatCompletionResponse<'a, S>
where
    S: Stream<Item = Result<Event, E>> + Send + 'static,
    E: Into<axum::BoxError>,
{
    fn into_response(self) -> Response {
        match self {
            ChatCompletionResponse::Stream(stream) => stream.into_response(),
            ChatCompletionResponse::Full(full) => full.into_response(),
        }
    }
}

/// POST `/v1/chat/completions`: generate chat completions for the provided context, optionally
/// streaming those completions in real-time.
///
/// See [the original OpenAI API specification][openai], which this endpoint is compatible with.
///
/// [openai]: https://platform.openai.com/docs/api-reference/chat/create
///
/// Generates completions for the given [`CreateChatCompletionRequest`] body.
/// If `stream` is enabled, streams a number of newline-separated, JSON-encoded
/// [`ChatCompletionChunk`]s to the client using [server-sent events][sse]. Otherwise, returns a
/// single JSON-encoded [`ChatCompletion`].
///
/// [sse]: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events
///
/// On failure, may raise a `500 Internal Server Error` with a JSON-encoded [`ChatCompletionError`]
/// to the peer.
#[utoipa::path(
post,
path = "/chat/completions",
request_body = CreateChatCompletionRequest,
responses(
(status = 200, description = "OK", body = ChatCompletionResponse),
(status = 500, description = "unexpected internal server error", body = ChatCompletionError)
),
)]
pub async fn chat_completions(
    Json(req): Json<CreateChatCompletionRequest<'_>>,
) -> Result<impl IntoResponse, ChatCompletionError> {
    // For MVP1, the model string in the request is *always* ignored.
    let model_name = SETTINGS
        .read()
        .await
        .read()
        .await
        .chat_completions_model_name
        .trim()
        .to_string();
    let repo = SETTINGS
        .read()
        .await
        .read()
        .await
        .chat_completions_model_repo
        .trim()
        .to_string();
    let dir = SETTINGS
        .read()
        .await
        .read()
        .await
        .chat_completions_models_dir
        .trim()
        .to_string();

    if model_name.is_empty() {
        return Err(ChatCompletionError::ProhibitedName {
            model_name: model_name,
            reason: Cow::Borrowed("Empty model name in config"),
        });
    }
    if dir.is_empty() {
        return Err(ChatCompletionError::ProhibitedName {
            model_name: dir,
            reason: Cow::Borrowed("Empty model directory in config"),
        });
    }

    let mut model = Model::new(ModelKind::LLM, &model_name, &repo, &PathBuf::from(&dir));

    model
        .preload()
        .await
        .map_err(move |_| ChatCompletionError::NoSuchModel {
            model_name: model_name.to_string(),
        })?;

    let untokenized_context = format!("{}<|ASSISTANT|>", req.messages);

    let mut args = CompletionArgs {
        prompt: untokenized_context,
        seed: req.seed,
        ..Default::default()
    };

    if let Some(one_shot) = req.one_shot {
        args.one_shot = one_shot;
    }

    if let Some(frequency_penalty) = req.frequency_penalty {
        args.frequency_penalty = frequency_penalty;
    }

    let stream_response = req.stream.unwrap_or(false);

    let fp = format!("edgen-{}", cargo_crate_version!());
    let response = if stream_response {
        let completions_stream =
            crate::llm::chat_completion_stream(model, args)
                .await?
                .map(move |chunk| {
                    Event::default().json_data(ChatCompletionChunk {
                        id: Uuid::new_v4().to_string().into(),
                        choices: tiny_vec![ChatCompletionChunkChoice {
                            index: 0,
                            finish_reason: None,
                            delta: ChatCompletionChunkDelta {
                                content: Some(Cow::Owned(chunk)),
                                role: None,
                            },
                        }],
                        created: OffsetDateTime::now_utc().unix_timestamp(),
                        model: Cow::Borrowed("main"),
                        system_fingerprint: Cow::Borrowed(&fp), // use macro for version
                        object: Cow::Borrowed("text_completion"),
                    })
                });

        ChatCompletionResponse::Stream(Sse::new(completions_stream))
    } else {
        let content_str = crate::llm::chat_completion(model, args).await?;
        let response = ChatCompletion {
            id: Uuid::new_v4().to_string().into(),
            choices: vec![ChatCompletionChoice {
                message: ChatMessage::Assistant {
                    content: Some(Cow::Owned(content_str)),
                    name: None,
                    tool_calls: None,
                },
                finish_reason: None,
                index: 0,
            }],
            created: OffsetDateTime::now_utc().unix_timestamp(),
            model: Cow::Borrowed("main"),
            object: Cow::Borrowed("text_completion"),
            system_fingerprint: Cow::Owned(fp),
            usage: ChatCompletionUsage {
                completion_tokens: 0,
                prompt_tokens: 0,
                total_tokens: 0,
            },
        };

        ChatCompletionResponse::Full(Json(response))
    };

    Ok(response)
}

/// A request to transcribe an audio file into text in either the specified language, or whichever
/// language is automatically detected, if none is specified.
///
/// An `axum` handler, [`create_transcription`][create_transcription], is provided to handle this request.
///
/// See [the documentation for creating transcriptions][openai] for more details. This request has
/// two additional optional parameters, which are **not normative** with OpenAI's specification,
/// `create_session` and `session` to deal with functionality specific to **Edgen**.
///
/// [create_transcription]: fn.create_transcription.html
/// [openai]: https://platform.openai.com/docs/api-reference/audio/createTranscription
#[derive(TryFromMultipart, ToSchema)]
#[try_from_multipart(strict)]
pub struct CreateTranscriptionRequest {
    /// The audio file object (not file name) to transcribe, in one of the following formats:
    /// **`aac`**, **`flac`**, **`mp3`**, **`m4a`**, **`m4b`**, **`ogg`**, **`oga`**, **`mogg`**,
    /// **`wav`**. TODO check working formats. webm
    #[form_data(limit = "unlimited")]
    #[schema(value_type = Vec < u8 >)]
    pub file: FieldData<axum::body::Bytes>,

    /// ID of the model to use.
    pub model: String,

    /// The language of the input audio. Supplying the input language in ISO-639-1 format will
    /// improve accuracy and latency.
    pub language: Option<String>,

    /// An optional text to guide the model's style or continue a previous audio segment. The prompt
    /// should match the audio language.
    pub prompt: Option<String>,

    /// The format of the transcript output, in one of these options: json, text, srt, verbose_json,
    /// or vtt. TODO whats this?
    pub response_format: Option<String>,

    /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more
    /// random, while lower values like 0.2 will make it more focused and deterministic. If set to 0,
    /// the model will use log probability to automatically increase the temperature until certain
    /// thresholds are hit.
    pub temperature: Option<f32>,

    /// Should a new session be created from this request. This may be useful for things like live
    /// transcriptions where continuous audio is submitted across several requests.
    ///
    /// If `true`, the response will contain a session [`Uuid`].
    ///
    /// The value of this member is ignored if `session` has some value.
    pub create_session: Option<bool>,

    /// The [`Uuid`] of an existing audio session.
    pub session: Option<Uuid>,
}

/// The return type of [`create_transcription`].
#[derive(Serialize, Deserialize, ToSchema)]
pub struct TranscriptionResponse {
    /// The transcribed text of the audio.
    pub text: String,

    /// The [`Uuid`] of a newly created session, present only if `create_session` in
    /// [`CreateTranscriptionRequest`] is set to `true`. This additional member is **not normative**
    /// with OpenAI's specification, as it is intended for **Edgen** specific functinality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session: Option<Uuid>,
}

/// POST `/v1/audio/transcriptions`: transcribes audio into text.
///
/// See [the original OpenAI API specification][openai], which this endpoint is compatible with.
///
/// [openai]: https://platform.openai.com/docs/api-reference/audio/createTranscription
///
/// On failure, may raise a `500 Internal Server Error` with a JSON-encoded [`WhisperEndpointError`]
/// to the peer.
#[utoipa::path(
post,
path = "/audio/transcriptions",
request_body = CreateTranscriptionRequest,
responses(
(status = 200, description = "OK", body = TranscriptionResponse),
(status = 500, description = "unexpected internal server error", body = TranscriptionError)
),
)]
pub async fn create_transcription(
    req: TypedMultipart<CreateTranscriptionRequest>,
) -> Result<impl IntoResponse, TranscriptionError> {
    // For MVP1, the model string in the request is *always* ignored.
    let model_name = SETTINGS
        .read()
        .await
        .read()
        .await
        .audio_transcriptions_model_name
        .trim()
        .to_string();
    let repo = SETTINGS
        .read()
        .await
        .read()
        .await
        .audio_transcriptions_model_repo
        .trim()
        .to_string();
    let dir = SETTINGS
        .read()
        .await
        .read()
        .await
        .audio_transcriptions_models_dir
        .trim()
        .to_string();

    // invalid
    if model_name.is_empty() {
        return Err(TranscriptionError::ProhibitedName {
            model_name,
            reason: Cow::Borrowed("Empty model name in config"),
        });
    }
    if dir.is_empty() {
        return Err(TranscriptionError::ProhibitedName {
            model_name: dir,
            reason: Cow::Borrowed("Empty model directory in config"),
        });
    }

    let mut model = Model::new(ModelKind::Whisper, &model_name, &repo, &PathBuf::from(&dir));

    model.preload().await?;

    let (text, session) = crate::whisper::create_transcription(
        &req.file.contents,
        model,
        req.language.as_deref(),
        req.prompt.as_deref(),
        req.temperature,
        req.create_session.unwrap_or(false),
        req.session,
    )
    .await?;

    Ok(Json(TranscriptionResponse { text, session }))
}

/// An error condition raised by the audio transcription API.
///
/// This is **not normative** with OpenAI's specification, which does not document any specific
/// failure modes.
#[derive(Serialize, Error, ToSchema, Debug)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "error")]
pub enum TranscriptionError {
    /// The provided model could not be found on the local system.
    #[error("no such model: {model_name}")]
    NoSuchModel {
        /// The name of the model.
        model_name: String,
    },

    /// The provided model name contains prohibited characters.
    #[error("model {model_name} could not be fetched from the system: {reason}")]
    ProhibitedName {
        /// The name of the model provided.
        model_name: String,

        /// A human-readable error message.
        reason: Cow<'static, str>,
    },

    /// The provided model could not be preloaded.
    #[error("failed to preload the model: {0}")]
    Preload(#[from] ModelError),

    /// An error occurred on the other side of an FFI boundary.
    #[error("an error occurred on the other side of a C FFI boundary; check `tracing`")]
    Ffi,

    /// An error occurred while processing the request to this endpoint.
    #[error("an error occurred while processing the request: {0}")]
    Endpoint(#[from] WhisperEndpointError),
}

impl IntoResponse for TranscriptionError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(self)).into_response()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn deserialize_chat_completion() {
        let content = r#"
            {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo-0613",
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello there, how may I assist you today?"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21
                }
            }
        "#;

        let _completion: ChatCompletion = serde_json::from_str(content).unwrap();
    }

    #[test]
    fn deserialize_chat_completion_chunks() {
        let chunks = &[
            r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":" today"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"?"},"finish_reason":null}]}"#,
            r#"{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-3.5-turbo-0613", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        ];

        for chunk in chunks {
            let _chunk: ChatCompletionChunk = serde_json::from_str(chunk).unwrap();
        }
    }

    #[test]
    fn deserialize_chat_completion_request() {
        let request = r#"
            {
                "model": "gpt-3.5-turbo",
                    "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": "Hello!"
                    }
                ]
            }
        "#;

        let _request: CreateChatCompletionRequest = serde_json::from_str(request).unwrap();
    }
}
