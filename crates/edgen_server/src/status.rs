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

use std::collections::VecDeque;
use std::error::Error;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{error, info, warn};
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
    download_progress: u64,
    last_errors: VecDeque<String>,
}

impl Default for AIStatus {
    fn default() -> AIStatus {
        AIStatus {
            active_model: "unknown".to_string(),
            last_activity: Activity::Unknown,
            last_activity_result: ActivityResult::Unknown,
            completions_ongoing: false,
            download_ongoing: false,
            download_progress: 0,
            last_errors: VecDeque::from([]),
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
    if ongoing {
        info!("starting model download");
    } else {
        info!("model download finished");
    };
    let rwstate = get_chat_completions_status();
    let mut state = rwstate.write().await;
    state.download_ongoing = ongoing;
}

/// Set download progress
pub async fn set_chat_completions_progress(progress: u64) {
    let rwstate = get_chat_completions_status();
    let mut state = rwstate.write().await;
    state.download_progress = progress;
}

/// Observe download progress
pub async fn observe_chat_completions_progress(
    datadir: &PathBuf,
    size: Option<u64>,
    download: bool,
) -> tokio::task::JoinHandle<()> {
    observe_progress(datadir, size, download).await
}

/// Add an error to the last errors
pub async fn add_chat_completions_error<E>(e: E)
where
    E: Error,
{
    let rwstate = get_chat_completions_status();
    let mut state = rwstate.write().await;
    if state.last_errors.len() > 32 {
        state.last_errors.pop_front();
    }
    state.last_errors.push_back(format!("{:?}", e));
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
            endpoints: vec![
                RwLock::new(Default::default()),
                RwLock::new(Default::default()),
            ],
        }
    }
}

#[allow(dead_code)]
fn internal_server_error(msg: &str) -> Response {
    eprintln!("[ERROR] {}", msg);
    StatusCode::INTERNAL_SERVER_ERROR.into_response()
}

// helper function to observe download progress.
// It spawns a new tokio task which
// - waits for the tmp directory to appear in dir
// - waits for the tempfile to appear in that directory
// - repeatedly reads the size of this tempfile
// -   calculates the percentage relative to size
// -   sets the percentage in the status.download_progress
// - until the tempfile disappears or no progress was made for 1 minute.
// TODO: This code should go to the module manager.
async fn observe_progress(
    datadir: &PathBuf,
    size: Option<u64>,
    download: bool,
) -> tokio::task::JoinHandle<()> {
    let tmp = datadir.join("tmp");

    let progress_handle = tokio::spawn(async move {
        if !download {
            info!("progress observer: no download necessary, file is already there");
            return;
        }

        if size.is_none() {
            warn!("progress observer: unknown file size. No progress reported on download");
            return;
        }
        let size = size.unwrap();

        if !have_tempdir(&tmp).await {
            return;
        }

        let t = wait_for_tempfile(&tmp).await;
        if t.is_none() {
            return;
        }

        let f = t.unwrap();

        let mut m = tokio::fs::metadata(&f.path()).await;
        let mut last_size = 0;
        let mut timestamp = Instant::now();
        while m.is_ok() {
            let s = m.unwrap().len() as u64;
            let t = size;
            let p = (s * 100) / t;

            if size > last_size {
                last_size = size;
                timestamp = Instant::now();
            } else if Instant::now().duration_since(timestamp) > Duration::from_secs(60) {
                warn!("progress observer: no download progress in a minute. Giving up");
                return;
            };

            set_chat_completions_progress(p).await;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            m = tokio::fs::metadata(&f.path()).await;
        }
    });

    progress_handle
}

async fn have_tempdir(tmp: &PathBuf) -> bool {
    let mut d = tokio::fs::metadata(&tmp).await;
    for _ in 0..10 {
        if d.is_ok() {
            break;
        };
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        d = tokio::fs::metadata(&tmp).await;
    }
    if d.is_err() {
        error!(
            "progress observer: can't read tmp directory ({:?}). Giving up",
            d
        );
        add_chat_completions_error(d.unwrap_err()).await;
        return false;
    };

    true
}

// TODO: we use the first file we find in the tmp directory.
//       we should instead *know* the name of the file.
async fn wait_for_tempfile(tmp: &PathBuf) -> Option<std::fs::DirEntry> {
    for _ in 0..30 {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let es = std::fs::read_dir(&tmp);
        if es.is_err() {
            error!(
                "progress observer: cannot read tmp directory ({:?}). Giving up",
                es
            );
            add_chat_completions_error(es.unwrap_err()).await;
            return None;
        };
        for e in es.unwrap() {
            if e.is_ok() {
                return Some(e.unwrap());
            }
        }
    }

    None
}
