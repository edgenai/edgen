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

use axum::Router;
use tower_http::cors::CorsLayer;

use futures::executor::block_on;
use tokio::select;
use tokio::sync::oneshot;
use tokio::task::JoinSet;
use tracing::{error, info};
use utoipa::OpenApi;

use edgen_core::settings;
use edgen_core::settings::SETTINGS;
use edgen_core::whisper::{DecodeSessionError, SessionRunnerError, WhisperError};
use openai_shim as chat;
use openai_shim as audio;

#[cfg(test)]
use levenshtein;
#[macro_use]
pub mod misc;

pub mod cli;
pub mod graceful_shutdown;
mod llm;
mod model;
pub mod openai_shim;
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
        openai_shim::CreateTranscriptionRequest,
        whisper::WhisperEndpointError,
        whisper::AudioError,
        WhisperError,
        DecodeSessionError,
        SessionRunnerError,
        model::ModelError,
        model::ModelKind,
    ))
)]
struct ApiDoc;

/// Result for main functions
pub type EdgenResult = Result<(), String>;

/// Main entry point for the server process
pub fn start(command: &cli::TopLevel) -> EdgenResult {
    // if the project dirs do not exist, try to create them.
    // if that fails, exit.
    settings::create_project_dirs().unwrap();

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
        std::fs::remove_file(&path).unwrap();
    }

    settings::create_default_config_file().unwrap();

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
    console_subscriber::init();

    SETTINGS
        .write()
        .await
        .init()
        .await
        .expect("Failed to initialise settings");

    while run_server(args).await {
        info!("Settings have been updated, resetting environment")
    }

    Ok(())
}

async fn run_server(args: &cli::Serve) -> bool {
    let http_app = Router::new()
        .route(
            "/v1/chat/completions",
            axum::routing::post(openai_shim::chat_completions),
        )
        .route(
            "/v1/audio/transcriptions",
            axum::routing::post(openai_shim::create_transcription),
        )
        .route("/v1/misc/version", axum::routing::get(misc::edgen_version))
        .layer(CorsLayer::permissive());

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
            uri if uri.starts_with("unix://") => {
                error!("unix:// URIs are not yet supported");

                exit(1)
            }
            uri if uri.starts_with("http://") => {
                let addr = uri.strip_prefix("http://").unwrap();

                tokio::net::TcpListener::bind(addr)
                    .await
                    .unwrap_or_else(|err| {
                        error!("Could not bind to TCP socket at {addr}: {err}");

                        exit(1)
                    })
            }
            uri if uri.starts_with("ws://") => {
                let addr = uri.strip_prefix("ws://").unwrap();

                tokio::net::TcpListener::bind(addr)
                    .await
                    .unwrap_or_else(|err| {
                        error!("Could not bind to TCP socket at {addr}: {err}");

                        exit(1)
                    })
            }
            _ => {
                error!("Unsupported URI schema: {uri}. unix://, http://, and ws:// are supported.");

                exit(1)
            }
        };

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

                            exit(1)
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
        flag_clone.store(true, Ordering::SeqCst);
        reset_channels.clear();
        block_on(crate::llm::reset_environment());
        block_on(crate::whisper::reset_environment());
    });

    loop {
        select! {
            _ = graceful_shutdown::global_shutdown_ends() => {
                error!("Global shutdown grace period has ended; exiting abnormally");

                exit(1)
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

    reset_flag.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::post;
    use axum::Router;
    use axum_test::multipart;
    use axum_test::TestServer;
    use serde_json::from_str;

    fn completion_streaming_request() -> String {
        r#"
            {
                "model": "gpt-3.5-turbo",
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

    fn completion_request() -> String {
        r#"
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

    #[tokio::test]
    #[ignore]
    // TODO this is hanging inside LlamaModel::load_from_file_async
    // Use a small model for this test.
    // phi-2.Q2_K.gguf is quite fast.
    // Note that the model must exist in the model path,
    // otherwise the test fails.
    async fn test_axum_completions_stream() {
        init_settings_for_test().await;

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
        assert!(response.text().starts_with("data:"));
    }

    #[tokio::test]
    #[ignore]
    //TODO This test should pass with non-streaming completions!
    async fn test_axum_completions() {
        init_settings_for_test().await;

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
        let _completion: openai_shim::ChatCompletion = from_str(&response.text()).unwrap();
        // assert something with completions
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
    async fn test_axum_transcriptions() {
        init_settings_for_test().await;

        let router = Router::new().route(
            "/v1/audio/transcriptions",
            post(openai_shim::create_transcription),
        );

        let server = TestServer::new(router).expect("cannot instantiate TestServer");

        let sound = include_bytes!("../resources/frost.wav");
        let mp = multipart::MultipartForm::new()
            .add_text("model", "ignore")
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
        let actual_text = resp.text();

        // Calculate Levenshtein distance
        let distance = levenshtein::levenshtein(&expected_text, &actual_text);

        // Calculate similarity percentage
        let similarity_percentage =
            100.0 - ((distance as f64 / expected_text.len() as f64) * 100.0);

        // Assert that the similarity is at least 90%
        assert!(
            similarity_percentage >= 90.0,
            "Text similarity is less than 90%"
        );
    }
}
