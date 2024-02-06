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

/// GET `/v1/chat/completions/status`: returns the current status of the /chat/completions endpoint.
///
/// The status is returned as json value AIStatus.
/// For any error, the version endpoint returns "internal server error".
pub async fn chat_completions_status() -> Response {
    let state = get_chat_completions_status().read().await;
    Json(state.clone()).into_response()
}

/// GET `/v1/audio/transcriptions/status`: returns the current status of the /audio/transcriptions endpoint.
///
/// The status is returned as json value AIStatus.
/// For any error, the version endpoint returns "internal server error".
pub async fn audio_transcriptions_status() -> Response {
    let state = get_audio_transcriptions_status().read().await;
    Json(state.clone()).into_response()
}

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

// axum provides shared state but using this shared state would force us
// to pass the state on to all function that may change the state.
static AISTATES: Lazy<AIStates> = Lazy::new(Default::default);

const EP_CHAT_COMPLETIONS: usize = 0;
const EP_AUDIO_TRANSCRIPTIONS: usize = 1;

const MAX_ERRORS: usize = 32;

/// Get a protected chat completions status.
/// Call read() or write() on the returned value to get either read or write access.
pub fn get_chat_completions_status() -> &'static RwLock<AIStatus> {
    get_status(EP_CHAT_COMPLETIONS)
}

/// Get a protected audio transcriptions status.
/// Call read() or write() on the returned value to get either read or write access.
pub fn get_audio_transcriptions_status() -> &'static RwLock<AIStatus> {
    get_status(EP_AUDIO_TRANSCRIPTIONS)
}

fn get_status(idx: usize) -> &'static RwLock<AIStatus> {
    &AISTATES.endpoints[idx]
}

/// Reset the chat completions status to its defaults
pub async fn reset_chat_completions_status() {
    reset_status(EP_CHAT_COMPLETIONS).await;
}

/// Reset the audio transcriptions status to its defaults
pub async fn reset_audio_transcriptions_status() {
    reset_status(EP_AUDIO_TRANSCRIPTIONS).await;
}

async fn reset_status(idx: usize) {
    let mut status = get_status(idx).write().await;
    *status = AIStatus::default();
}

/// Set chat completions active model
pub async fn set_chat_completions_active_model(model: &str) {
    set_active_model(EP_CHAT_COMPLETIONS, model).await;
}

/// Set audio transcriptions active model
pub async fn set_audio_transcriptions_active_model(model: &str) {
    set_active_model(EP_AUDIO_TRANSCRIPTIONS, model).await;
}

async fn set_active_model(idx: usize, model: &str) {
    let mut state = get_status(idx).write().await;
    state.active_model = model.to_string();
}

/// Set chat completions download ongoing
pub async fn set_chat_completions_download(ongoing: bool) {
    if ongoing {
        info!("starting chat completions model download");
    } else {
        info!("chat completions model download finished");
    };
    set_download(EP_CHAT_COMPLETIONS, ongoing).await;
}

/// Set audio transcriptions download ongoing
pub async fn set_audio_transcriptions_download(ongoing: bool) {
    if ongoing {
        info!("starting chat completions model download");
    } else {
        info!("chat completions model download finished");
    };
    set_download(EP_AUDIO_TRANSCRIPTIONS, ongoing).await;
}

async fn set_download(idx: usize, ongoing: bool) {
    let mut state = get_status(idx).write().await;
    state.download_ongoing = ongoing;
}

/// Set chat completions download progress
pub async fn set_chat_completions_progress(progress: u64) {
    set_progress(EP_CHAT_COMPLETIONS, progress).await;
}

/// Set audio transcriptions download progress
pub async fn set_audio_transcriptions_progress(progress: u64) {
    set_progress(EP_AUDIO_TRANSCRIPTIONS, progress).await;
}

async fn set_progress(idx: usize, progress: u64) {
    let mut state = get_status(idx).write().await;
    state.download_progress = progress;
}

/// Observe chat completions download progress
pub async fn observe_chat_completions_progress(
    datadir: &PathBuf,
    size: Option<u64>,
    download: bool,
) -> tokio::task::JoinHandle<()> {
    observe_progress(EP_CHAT_COMPLETIONS, datadir, size, download).await
}

/// Observe audio transcptions download progress
pub async fn observe_audio_transcriptions_progress(
    datadir: &PathBuf,
    size: Option<u64>,
    download: bool,
) -> tokio::task::JoinHandle<()> {
    observe_progress(EP_AUDIO_TRANSCRIPTIONS, datadir, size, download).await
}

/// Add an error to the last errors in chat completions
pub async fn add_chat_completions_error<E>(e: E)
where
    E: Error,
{
    add_error(EP_CHAT_COMPLETIONS, e).await;
}

/// Add an error to the last errors in audio transcriptions
pub async fn add_audio_transcriptions_error<E>(e: E)
where
    E: Error,
{
    add_error(EP_AUDIO_TRANSCRIPTIONS, e).await;
}

async fn add_error<E>(idx: usize, e: E)
where
    E: Error,
{
    let rwstate = get_status(idx);
    let mut state = rwstate.write().await;
    if state.last_errors.len() > MAX_ERRORS {
        state.last_errors.pop_front();
    }
    state.last_errors.push_back(format!("{:?}", e));
}

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
    idx: usize,
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

        if !have_tempdir(idx, &tmp).await {
            return;
        }

        let t = wait_for_tempfile(idx, &tmp).await;
        if t.is_none() {
            return;
        }

        let f = t.unwrap();

        let mut m = tokio::fs::metadata(&f.path()).await;
        let mut last_size = 0;
        let mut timestamp = Instant::now();
        while m.is_ok() {
            let s = m.unwrap().len() as u64;
            let p = (s * 100) / size;

            if s > last_size {
                last_size = s;
                timestamp = Instant::now();
            } else if Instant::now().duration_since(timestamp) > Duration::from_secs(180) {
                warn!("progress observer: no download progress in three minutes. Giving up");
                return;
            };

            set_progress(idx, p).await;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            m = tokio::fs::metadata(&f.path()).await;
        }
    });

    progress_handle
}

async fn have_tempdir(idx: usize, tmp: &PathBuf) -> bool {
    if !tmp.exists() {
        let r = std::fs::create_dir(tmp);
        if r.is_err() {
            error!(
                "progress observer: cannot create tmp directory ({:?}). Giving up",
                r
            );
            add_error(idx, r.unwrap_err()).await;
        }
    }
    return tmp.exists();
}

// TODO: we use the first file we find in the tmp directory.
//       we should instead *know* the name of the file.
async fn wait_for_tempfile(idx: usize, tmp: &PathBuf) -> Option<std::fs::DirEntry> {
    for _ in 0..30 {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let es = std::fs::read_dir(&tmp);
        if es.is_err() {
            error!(
                "progress observer: cannot read tmp directory ({:?}). Giving up",
                es
            );
            add_error(idx, es.unwrap_err()).await;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Error, ErrorKind};

    use axum::routing::get;
    use axum::Router;
    use axum_test::TestServer;

    fn default_status_json() -> String {
        "{\"active_model\":\"unknown\",\
          \"last_activity\":\"Unknown\",\
          \"last_activity_result\":\"Unknown\",\
          \"completions_ongoing\":false,\
          \"download_ongoing\":false,\
          \"download_progress\":0,\
          \"last_errors\":[]\
         }"
        .to_string()
    }

    #[test]
    fn test_serialize_status() {
        let state = AIStatus::default();
        assert_eq!(
            serde_json::to_string(&state).unwrap(),
            default_status_json()
        );
    }

    #[test]
    fn test_deserialize_status() {
        let expected = AIStatus::default();
        let state = serde_json::from_str::<AIStatus>(&default_status_json()).unwrap();
        assert_eq!(state, expected);
    }

    // This test should not be split into sub-tests.
    // The problem is that tests run in parallel
    // and we are testing one common resource, the global status.
    #[tokio::test]
    async fn test_chat_completions_status() {
        reset_chat_completions_status().await;

        // default
        let mut expected = AIStatus::default();

        {
            let status = get_chat_completions_status().read().await;
            assert_eq!(*status, AIStatus::default());
        }

        // download ongoing
        expected.download_ongoing = true;
        set_chat_completions_download(true).await;

        {
            let status = get_chat_completions_status().read().await;
            assert_eq!(*status, expected);
        }

        // download progress
        expected.download_progress = 42;
        set_chat_completions_progress(42).await;

        {
            let status = get_chat_completions_status().read().await;
            assert_eq!(*status, expected);
        }

        // errors
        let e1 = Error::new(ErrorKind::Interrupted, "couldn't finish");
        expected.last_errors.push_back(format!("{:?}", e1));
        add_chat_completions_error(e1).await;

        {
            let status = get_chat_completions_status().read().await;
            assert_eq!(*status, expected);
        }

        let e2 = Error::new(ErrorKind::NotFound, "I still haven't found");
        expected.last_errors.push_back(format!("{:?}", e2));
        add_chat_completions_error(e2).await;

        assert_eq!(expected.last_errors.len(), 2);

        {
            let status = get_chat_completions_status().read().await;
            assert_eq!(*status, expected);
        }

        let e3 = Error::new(ErrorKind::PermissionDenied, "verboten");
        expected.last_errors.push_back(format!("{:?}", e3));
        add_chat_completions_error(e3).await;

        assert_eq!(expected.last_errors.len(), 3);

        {
            let status = get_chat_completions_status().read().await;
            assert_eq!(*status, expected);
        }

        // make sure there are at most MAX_ERRORS
        for i in 0..29 {
            let message = format!("{} times verboten", i + 1);
            let e = Error::new(ErrorKind::PermissionDenied, message);
            expected.last_errors.push_back(format!("{:?}", e));
            add_chat_completions_error(e).await;
        }

        assert_eq!(expected.last_errors.len(), MAX_ERRORS);

        {
            let status = get_chat_completions_status().read().await;
            assert_eq!(*status, expected);
        }

        for i in 0..10 {
            let message = format!("{} times more verboten", i + 1);
            let e = Error::new(ErrorKind::PermissionDenied, message);
            expected.last_errors.pop_front();
            expected.last_errors.push_back(format!("{:?}", e));
            add_chat_completions_error(e).await;
        }

        assert_eq!(expected.last_errors.len(), MAX_ERRORS);

        {
            // since we don't know the exact order in which tokio runs the tasks
            // the order of errors in the deques is random.
            // therefore, we sort them before asserting equality.
            let status = get_chat_completions_status().read().await;
            let mut v1 = Vec::from(status.last_errors.clone());
            let mut v2 = Vec::from(expected.last_errors.clone());
            assert_eq!(v1.sort(), v2.sort());
        }

        // axum router
        let router =
            Router::new().route("/v1/chat/completions/status", get(chat_completions_status));

        let server = TestServer::new(router).expect("cannot instantiate TestServer");

        let response = server.get("/v1/chat/completions/status").await;

        response.assert_status_ok();
        assert!(response.text().len() > 0);
        assert_eq!(response.json::<AIStatus>().active_model, "unknown");

        let model = "shes-a-model-and-shes-looking-good".to_string();
        set_chat_completions_active_model(&model).await;

        let response = server.get("/v1/chat/completions/status").await;

        response.assert_status_ok();
        assert!(response.text().len() > 0);
        assert_eq!(response.json::<AIStatus>().active_model, model);
    }

    // This test should not be split into sub-tests.
    // The problem is that tests run in parallel
    // and we are testing one common resource, the global status.
    #[tokio::test]
    async fn test_audio_transcriptions_status() {
        reset_audio_transcriptions_status().await;

        // default
        let mut expected = AIStatus::default();

        {
            let status = get_audio_transcriptions_status().read().await;
            assert_eq!(*status, AIStatus::default());
        }

        // download ongoing
        expected.download_ongoing = true;
        set_audio_transcriptions_download(true).await;

        {
            let status = get_audio_transcriptions_status().read().await;
            assert_eq!(*status, expected);
        }

        // download progress
        expected.download_progress = 42;
        set_audio_transcriptions_progress(42).await;

        {
            let status = get_audio_transcriptions_status().read().await;
            assert_eq!(*status, expected);
        }

        // errors
        let e1 = Error::new(ErrorKind::Interrupted, "couldn't finish");
        expected.last_errors.push_back(format!("{:?}", e1));
        add_audio_transcriptions_error(e1).await;

        {
            let status = get_audio_transcriptions_status().read().await;
            assert_eq!(*status, expected);
        }

        let e2 = Error::new(ErrorKind::NotFound, "I still haven't found");
        expected.last_errors.push_back(format!("{:?}", e2));
        add_audio_transcriptions_error(e2).await;

        assert_eq!(expected.last_errors.len(), 2);

        {
            let status = get_audio_transcriptions_status().read().await;
            assert_eq!(*status, expected);
        }

        let e3 = Error::new(ErrorKind::PermissionDenied, "verboten");
        expected.last_errors.push_back(format!("{:?}", e3));
        add_audio_transcriptions_error(e3).await;

        assert_eq!(expected.last_errors.len(), 3);

        {
            let status = get_audio_transcriptions_status().read().await;
            assert_eq!(*status, expected);
        }

        // make sure there are at most MAX_ERRORS
        for i in 0..29 {
            let message = format!("{} times verboten", i + 1);
            let e = Error::new(ErrorKind::PermissionDenied, message);
            expected.last_errors.push_back(format!("{:?}", e));
            add_audio_transcriptions_error(e).await;
        }

        assert_eq!(expected.last_errors.len(), MAX_ERRORS);

        {
            let status = get_audio_transcriptions_status().read().await;
            assert_eq!(*status, expected);
        }

        for i in 0..10 {
            let message = format!("{} times more verboten", i + 1);
            let e = Error::new(ErrorKind::PermissionDenied, message);
            expected.last_errors.pop_front();
            expected.last_errors.push_back(format!("{:?}", e));
            add_audio_transcriptions_error(e).await;
        }

        assert_eq!(expected.last_errors.len(), MAX_ERRORS);

        {
            // since we don't know the exact order in which tokio runs the tasks
            // the order of errors in the deques is random.
            // therefore, we sort them before asserting equality.
            let status = get_audio_transcriptions_status().read().await;
            let mut v1 = Vec::from(status.last_errors.clone());
            let mut v2 = Vec::from(expected.last_errors.clone());
            assert_eq!(v1.sort(), v2.sort());
        }

        // axum router
        let router = Router::new().route(
            "/v1/audio/transcriptions/status",
            get(audio_transcriptions_status),
        );

        let server = TestServer::new(router).expect("cannot instantiate TestServer");

        let response = server.get("/v1/audio/transcriptions/status").await;

        response.assert_status_ok();
        assert!(response.text().len() > 0);
        assert_eq!(response.json::<AIStatus>().active_model, "unknown");

        let model = "shes-a-model-and-shes-looking-good".to_string();
        set_audio_transcriptions_active_model(&model).await;

        let response = server.get("/v1/audio/transcriptions/status").await;

        response.assert_status_ok();
        assert!(response.text().len() > 0);
        assert_eq!(response.json::<AIStatus>().active_model, model);
    }
}
