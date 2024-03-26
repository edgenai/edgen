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

//! A batteries-included, transport-agnostic server for interacting with large AI models.

#![deny(unsafe_code)]
#![warn(missing_docs)]

use core::future::IntoFuture;
use std::process::exit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use futures::executor::block_on;
use tokio::select;
use tokio::sync::oneshot;
use tokio::task::JoinSet;
use tower_http::cors::CorsLayer;
use tracing::{error, info};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use utoipa::OpenApi;

use edgen_core::settings;
use edgen_core::settings::SETTINGS;
use openai_shim as chat;
use openai_shim as audio;

#[macro_use]
pub mod misc;

mod chat_faker;
pub mod cli;
pub mod graceful_shutdown;
mod llm;
mod model;
pub mod model_man;
pub mod openai_shim;
mod routes;
pub mod status;
pub mod types;
pub mod util;
mod whisper;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "Edgen API",
        description = "Edgen API with OpenAI-compliant and proprietary endpoints.",
    ),
    paths(
        misc::edgen_version,
        chat::chat_completions,
        audio::create_transcription
    ),
    components(schemas(
        misc::Version,
        openai_shim::CreateChatCompletionRequest,
        openai_shim::ChatCompletion,
        openai_shim::ChatCompletionChoice,
        openai_shim::ChatCompletionUsage,
        openai_shim::ChatCompletionChunk,
        openai_shim::ChatCompletionChunkDelta,
        openai_shim::ChatCompletionChunkChoice,
        openai_shim::ChatCompletionError,
        openai_shim::ChatMessage,
        openai_shim::ChatMessages,
        openai_shim::ContentPart,
        openai_shim::ToolStub,
        openai_shim::FunctionStub,
        openai_shim::AssistantFunctionStub,
        openai_shim::AssistantToolCall,
        openai_shim::CreateEmbeddingsRequest,
        openai_shim::EmbeddingsResponse,
        openai_shim::Embedding,
        openai_shim::EmbeddingsUsage,
        openai_shim::CreateTranscriptionRequest,
        openai_shim::TranscriptionResponse,
        openai_shim::TranscriptionError,
        model::ModelError,
        model::ModelKind,
    ))
)]
struct ApiDoc;

/// Result for main functions
pub type EdgenResult = Result<(), types::EdgenError>;

/// Main entry point for the server process
pub fn start(command: &cli::TopLevel) -> EdgenResult {
    match &command.subcommand {
        None => serve(&cli::Serve::default())?,
        Some(cli::Command::Serve(serve_args)) => serve(serve_args)?,
        Some(cli::Command::Config(config_args)) => config(config_args)?,
        Some(cli::Command::Version(_)) => version()?,
        Some(cli::Command::Oasgen(oasgen_args)) => oasgen(oasgen_args)?,
    };

    Ok(())
}

/// Prints the edgen version to stdout
pub fn version() -> EdgenResult {
    println!(cargo_crate_version!());

    Ok(())
}

fn config(config_args: &cli::Config) -> EdgenResult {
    match &config_args.subcommand {
        cli::ConfigCommand::Reset(_) => config_reset()?,
    };

    Ok(())
}

/// Resets the configuration file to the default settings.
pub fn config_reset() -> EdgenResult {
    let path = settings::get_config_file_path();

    if path.is_file() {
        std::fs::remove_file(&path)?;
    }

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async { settings::create_default_config_file() })?;

    Ok(())
}

/// Generates the OpenAPI Spec.
pub fn oasgen(args: &cli::Oasgen) -> EdgenResult {
    if args.json {
        println!("{}", ApiDoc::openapi().to_pretty_json().unwrap());
    } else {
        println!("{}", ApiDoc::openapi().to_yaml().unwrap());
    }

    Ok(())
}

// This in-between step appears pointless right now.
// However, synchronous code that we need before
// tokio::main should go here.
fn serve(args: &cli::Serve) -> EdgenResult {
    start_server(args)
}

#[tokio::main]
async fn start_server(args: &cli::Serve) -> EdgenResult {
    let format = tracing_subscriber::fmt::layer().compact();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or(
        tracing_subscriber::EnvFilter::default()
            .add_directive(tracing_subscriber::filter::LevelFilter::INFO.into()),
    );
    let console_layer = console_subscriber::ConsoleLayer::builder()
        .with_default_env()
        .spawn();
    tracing_subscriber::registry()
        .with(console_layer)
        .with(format)
        .with(filter)
        .init();

    SETTINGS
        .write()
        .await
        .init()
        .await
        .expect("Failed to initialise settings. Please make sure the configuration file valid, or reset it via the system tray and restart Edgen.\nThe following error occurred");

    settings::create_project_dirs().await.unwrap();

    while run_server(args).await? {
        info!("Settings have been updated, resetting environment")
    }

    Ok(())
}

async fn run_server(args: &cli::Serve) -> Result<bool, types::EdgenError> {
    status::set_chat_completions_active_model(
        &SETTINGS
            .read()
            .await
            .read()
            .await
            .chat_completions_model_name,
    )
    .await;

    status::set_audio_transcriptions_active_model(
        &SETTINGS
            .read()
            .await
            .read()
            .await
            .audio_transcriptions_model_name,
    )
    .await;

    let http_app = routes::routes()
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(
            SETTINGS.read().await.read().await.max_request_size,
        ));

    let uri_vector = if !args.uri.is_empty() {
        info!("Overriding default URI");
        args.uri.clone()
    } else {
        info!("Using default URI");
        vec![SETTINGS.read().await.read().await.default_uri.clone()]
    };

    let mut all_listeners = JoinSet::new();
    let mut reset_channels = vec![];

    for uri in &uri_vector {
        let listener = match uri {
            uri if uri.starts_with("unix://") => Err(types::EdgenError::GenericError(
                "unix:// URIs are not supported".to_string(),
            )),
            uri if uri.starts_with("http://") => {
                let addr = uri.strip_prefix("http://").unwrap();
                Ok(tokio::net::TcpListener::bind(addr).await?)
            }
            uri if uri.starts_with("ws://") => {
                let addr = uri.strip_prefix("ws://").unwrap();

                Ok(tokio::net::TcpListener::bind(addr).await?)
            }
            _ => Err(types::EdgenError::GenericError(format!(
                "Unsupported URI schema: {uri}. unix://, http://, and ws:// are supported."
            ))),
        }?;

        info!("Listening in on: {uri}");

        let http_app = http_app.clone();
        let (reset_tx, reset_rx) = oneshot::channel::<()>();
        reset_channels.push(reset_tx);

        all_listeners.spawn(async move {
            let mut reset_rx = reset_rx;
            select! {
                bind_res = axum::serve(listener, http_app).into_future() => {
                    bind_res
                        .unwrap_or_else(|err| {
                            error!("Could not bind HTTP server: {err}");
                        });
                    },
                _ = &mut reset_rx => {}
                _ = graceful_shutdown::global_shutdown_starts() => {}
            }
        });
    }

    let reset_flag = Arc::new(AtomicBool::new(false));
    let flag_clone = reset_flag.clone();

    let _callback_handle = SETTINGS.read().await.add_change_callback(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let _guard = rt.enter();
        flag_clone.store(true, Ordering::SeqCst);
        reset_channels.clear();
        block_on(crate::llm::reset_environment());
        block_on(crate::whisper::reset_environment());
        block_on(async {
            status::set_chat_completions_active_model(
                &SETTINGS
                    .read()
                    .await
                    .read()
                    .await
                    .chat_completions_model_name,
            )
            .await;
            status::set_audio_transcriptions_active_model(
                &SETTINGS
                    .read()
                    .await
                    .read()
                    .await
                    .audio_transcriptions_model_name,
            )
            .await;
        });
    });

    loop {
        select! {
            _ = graceful_shutdown::global_shutdown_ends() => {
                error!("Global shutdown grace period has ended; exiting abnormally");

                exit(1) // last resort
            }
            _ = all_listeners.join_next() => {
                info!("Thread has exited");

                if all_listeners.is_empty() {
                    info!("All threads have exited; exiting normally");

                    break;
                }
            }
        }
    }

    Ok(reset_flag.load(Ordering::SeqCst))
}

#[cfg(test)]
mod tests {
    use axum::routing::post;
    use axum::Router;
    use axum_test::multipart;
    use axum_test::TestServer;
    use levenshtein;
    use serde_json::from_str;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    use edgen_rt_chat_faker as chat_faker;

    use crate::openai_shim::{ChatCompletion, ChatMessage, TranscriptionResponse};

    use super::*;

    fn completion_streaming_request() -> String {
        r#"
            {
                "model": "fake-model.fake",
                "stream": true,
                "messages": [
                    {
                        "role": "user",
                        "content": "what is the result of 1 + 2?"
                    }
                ]
            }
        "#
        .to_string()
    }

    fn completion_long_streaming_request() -> String {
        r#"
            {
                "model": "fake-model.fake",
                "stream": true,
                "messages": [
                    {
                        "role": "user",
                        "content": "answer with a long sermon."
                    }
                ]
            }
        "#
        .to_string()
    }

    fn completion_request() -> String {
        r#"
            {
                "model": "fake-model.fake",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": "What is the capital of Portugal?"
                    }
                ]
            }
        "#
        .to_string()
    }

    fn frost() -> String {
        " The woods are lovely, dark and deep, \
         but I have promises to keep \
         and miles to go before I sleep, \
         and miles to go before I sleep."
            .to_string()
    }

    async fn init_settings_for_test() {
        SETTINGS
            .write()
            .await
            .init()
            .await
            .expect("Failed to initialise settings");
    }

    async fn create_chat_fake_model_file() {
        let path_string = settings::chat_completions_dir().await;
        let path = Path::new(&path_string).join("fake-model.fake");
        if !path.exists() {
            let mut file = File::create(path).expect("cannot create fake model");
            file.write_all(b"this is for testing")
                .expect("cannot write to fake model");
        }
    }

    fn poor_mans_stream_processor(stream: &str) -> String {
        let mut answer = String::new();
        let mut next_one = false;
        for p in stream.split("\"") {
            if next_one && p != ":" {
                next_one = false;
                if answer.len() > 0 {
                    answer += " ";
                }
                answer += p;
            }
            if p == "content" {
                next_one = true;
            }
        }
        answer
    }

    #[tokio::test]
    async fn test_axum_completions_stream() {
        init_settings_for_test().await;
        create_chat_fake_model_file().await;

        let router =
            Router::new().route("/v1/chat/completions", post(openai_shim::chat_completions));

        let server = TestServer::new(router).expect("cannot instantiate TestServer");

        let req: openai_shim::CreateChatCompletionRequest =
            from_str(&completion_streaming_request()).unwrap();
        let response = server
            .post("/v1/chat/completions")
            .content_type(&"application/json")
            .json(&req)
            .await;

        response.assert_status_ok();
        assert!(response.text().len() > 0);

        // is there no support for reading sse streams in axum_test?
        assert!(response.text().starts_with("data:"));
        let answer = poor_mans_stream_processor(&response.text());
        assert_eq!(answer, chat_faker::DEFAULT_ANSWER, "wrong answer");
    }

    #[tokio::test]
    async fn test_axum_completions_long_stream() {
        init_settings_for_test().await;
        create_chat_fake_model_file().await;

        let router =
            Router::new().route("/v1/chat/completions", post(openai_shim::chat_completions));

        let server = TestServer::new(router).expect("cannot instantiate TestServer");

        let req: openai_shim::CreateChatCompletionRequest =
            from_str(&completion_long_streaming_request()).unwrap();
        let response = server
            .post("/v1/chat/completions")
            .content_type(&"application/json")
            .json(&req)
            .await;

        response.assert_status_ok();
        assert!(response.text().len() > 0);

        // is there no support for reading sse streams in axum_test?
        assert!(response.text().starts_with("data:"));
        let answer = poor_mans_stream_processor(&response.text());
        assert_eq!(answer, chat_faker::LONG_ANSWER, "wrong answer");
    }

    #[tokio::test]
    async fn test_axum_completions() {
        init_settings_for_test().await;
        create_chat_fake_model_file().await;

        let router =
            Router::new().route("/v1/chat/completions", post(openai_shim::chat_completions));

        let server = TestServer::new(router).expect("cannot instantiate TestServer");

        let req: openai_shim::CreateChatCompletionRequest =
            from_str(&completion_request()).unwrap();
        let response = server
            .post("/v1/chat/completions")
            .content_type(&"application/json")
            .json(&req)
            .await;

        response.assert_status_ok();
        let mut answer = String::new();
        let completion: ChatCompletion = serde_json::from_str(&response.text()).unwrap();
        for choice in completion.choices {
            if let ChatMessage::Assistant { content, .. } = choice.message {
                if let Some(content) = content {
                    answer += &content;
                }
            }
        }
        assert_eq!(answer, chat_faker::CAPITAL_OF_PORTUGAL, "wrong answer");
    }

    #[tokio::test]
    #[ignore]
    //TODO This test expects speech-to-text (a.k.a. /audio/speech) to be implemented
    async fn test_axum_audio_round_trip() {
        init_settings_for_test().await;

        let router = Router::new()
            .route(
                "/v1/audio/transcriptions",
                post(openai_shim::create_transcription),
            )
            .route(
                "/v1/audio/speech",
                post(openai_shim::create_transcription), // speech)
            );

        let server = TestServer::new(router).expect("cannot instantiate TestServer");

        let input = "hello world";

        let stream = server
            .post("/v1/audio/speech")
            .content_type(&"application/json")
            .json(&input) // should be form with model, voice and input
            .await;

        stream.assert_status_ok();

        let txt = server
            .post("/v1/audio/transcriptions")
            .content_type(&"application/json")
            .json(&stream.into_bytes()) // should be multipart with model and stream
            .await;

        assert_eq!(txt.text(), input);
    }

    #[tokio::test]
    // Note that the model must exist in the model path,
    // otherwise the test fails.
    async fn test_axum_transcriptions() {
        init_settings_for_test().await;

        let router = Router::new().route(
            "/v1/audio/transcriptions",
            post(openai_shim::create_transcription),
        );

        let server = TestServer::new(router).expect("cannot instantiate TestServer");

        let sound = include_bytes!("../resources/frost.wav");
        let mp = multipart::MultipartForm::new()
            .add_text("model", "default")
            .add_part(
                "file",
                multipart::Part::bytes(sound.as_slice()).file_name(&"frost.wav"),
            );
        let resp = server
            .post("/v1/audio/transcriptions")
            .content_type(&"multipart/form-data")
            .multipart(mp)
            .await;

        resp.assert_status_ok();

        let expected_text = frost();
        let actual_text = resp.json::<TranscriptionResponse>().text;

        // Calculate Levenshtein distance
        let distance = levenshtein::levenshtein(&expected_text, &actual_text);

        // Calculate similarity percentage
        let similarity_percentage =
            100.0 - ((distance as f64 / expected_text.len() as f64) * 100.0);

        // Assert that the similarity is at least 90%
        println!("test      : '{}'", actual_text);
        println!("similarity: {}", similarity_percentage);
        assert!(
            similarity_percentage >= 90.0,
            "Text similarity is less than 90%"
        );
    }
}
