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

use std::path::PathBuf;

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
#[derive(ToSchema, Deserialize, Serialize, Clone, Debug, PartialEq)]
pub struct AIStatus {
    active_model: String,
    last_activity: Activity,
    last_activity_result: ActivityResult,
    completions_ongoing: bool,
    download_ongoing: bool,
    download_progress: f64,
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
            download_progress: 0.0,
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

/// Set download ongoing
pub async fn set_chat_completions_download(ongoing: bool) {
    let rwstate = get_chat_completions_status();
    let mut state = rwstate.write().await;
    state.download_ongoing = ongoing;
}

/// Set download progress
pub async fn set_chat_completions_progress(progress: f64) {
    let rwstate = get_chat_completions_status();
    let mut state = rwstate.write().await;
    state.download_progress = progress;
}

/// Set download progress
pub async fn observe_chat_completions_progress(tempdir: &PathBuf) -> tokio::task::JoinHandle<()> {
    observe_progress(tempdir).await
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

#[allow(dead_code)]
fn internal_server_error(msg: &str) -> Response {
    eprintln!("[ERROR] {}", msg);
    StatusCode::INTERNAL_SERVER_ERROR.into_response()
}

// This is a mess
async fn observe_progress(tempfile: &PathBuf) -> tokio::task::JoinHandle<()> {
        let tmp = tempfile.join("tmp");
        let progress_handle = tokio::spawn(async move {
            let mut d = tokio::fs::metadata(&tmp).await;
            for _ in 0 .. 3 {
                if d.is_ok() {
                    break;
                };
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                d = tokio::fs::metadata(&tmp).await;
            };
            if d.is_err() {
                return;
            };
            tokio::time::sleep(std::time::Duration::from_millis(500)).await; // sloppy
            let mut t = None;
            for _ in 0 .. 10 {
                let es = std::fs::read_dir(&tmp).unwrap(); // .await.unwrap();
                let mut e = None;
                for x in es {
                    let y = x.unwrap();
                    println!("file: {:?}", y.path());
                    e = Some(y);
                    break;
                };
                if e.is_some() {
                    t = e;
                    break;
                }
            };
          
            if t.is_none() {
                return;
            }
            let f = t.unwrap();
            
            let mut m = tokio::fs::metadata(&f.path()).await;
            for _ in 0 .. 3 {
                if m.is_ok() {
                    break;
                };
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                m = tokio::fs::metadata(&f.path()).await;
            };
            while m.is_ok() {
                let z = m.unwrap();
                let s = z.len() as f64;
                let t = 1173610336 as f64;
                let p = (s * 100.0) / t;
                crate::status::set_chat_completions_progress(p).await;
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                m = tokio::fs::metadata(&f.path()).await;
            }
        });

        progress_handle
}
