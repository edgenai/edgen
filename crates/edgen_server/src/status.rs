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

//! Edgen AI service status.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock};
use utoipa::ToSchema;

/// Recent Activity on a specific endpoint, e.g. Completions or Download.
#[derive(ToSchema, Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
pub enum Activity {
    /// Last activity was ChatCompletions
    ChatCompletions,
    /// Last activity was AudioTranscriptions
    AudioTranscriptions,
    /// Last activity was a model download
    Download,
    /// No known activity was recently performed
    Unknown,
}

/// Result of the last activity on a specific endpoint, e.g. Success or Failed.
#[derive(ToSchema, Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
pub enum ActivityResult {
    /// Last activity finished successfully
    Success,
    /// Last activity failed
    Failed,
    /// Result of the activity is unkown or there was no activity
    Unknown,
}

/// Current Endpoint status.
#[derive(ToSchema, Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
pub struct AIStatus {
    active_model: String,
    last_activity: Activity,
    last_activity_result: ActivityResult,
    completions_ongoing: bool,
    download_ongoing: bool,
    last_errors: Vec<String>,
}

impl Default for AIStatus {
    fn default() -> AIStatus {
        AIStatus {
            active_model: "unknown".to_string(),
            last_activity: Activity::Unknown,
            last_activity_result: ActivityResult::Unknown,
            completions_ongoing: false,
            download_ongoing: false,
            last_errors: vec![],
        }
    }
}

/// Get a protected chat completions status.
/// Call read() or write() on the returned value to get either read or write access.
pub fn get_chat_completions_status() -> &'static RwLock<AIStatus> {
    &AISTATES.endpoints[EP_CHAT_COMPLETIONS]
}

/// Get a protected audio transcriptions status.
/// Call read() or write() on the returned value to get either read or write access.
pub fn get_audio_transcriptions_status() -> &'static RwLock<AIStatus> {
    &AISTATES.endpoints[EP_AUDIO_TRANSCRIPTIONS]
}

// axum provides shared state but using this shared state would force us
// to pass the state on to all function that may change the state.
static AISTATES: Lazy<AIStates> = Lazy::new(Default::default);

const EP_CHAT_COMPLETIONS: usize = 0;
const EP_AUDIO_TRANSCRIPTIONS: usize = 1;

struct AIStates {
    endpoints: Vec<RwLock<AIStatus>>,
}

impl Default for AIStates {
    fn default() -> AIStates {
        AIStates {
            endpoints: vec![RwLock::new(Default::default()),
                            RwLock::new(Default::default())]
        }
    }
}

/// GET `/v1/chat/completions/status`: returns the current status of the /chat/completions endpoint.
///
/// The status is returned as json value AIStatus.
/// For any error, the version endpoint returns "internal server error".
pub async fn chat_completions_status() -> Response {
    let rwstate = get_chat_completions_status();
    let locked = rwstate.read().await;
    Json(locked.clone()).into_response()
}

#[allow(dead_code)]
fn internal_server_error(msg: &str) -> Response {
    eprintln!("[ERROR] {}", msg);
    StatusCode::INTERNAL_SERVER_ERROR.into_response()
}
