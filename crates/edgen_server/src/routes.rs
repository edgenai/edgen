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

//! Contains all routes served by Edgen

use axum::{
    http::{uri::Uri, Method, StatusCode},
    response::IntoResponse,
    routing::{delete, get, post},
    Router,
};

use tracing::warn;

use crate::misc;
use crate::model_man;
use crate::openai_shim;
use crate::status;

pub fn routes() -> Router {
    Router::new()
        // -- AI endpoints -----------------------------------------------------
        // ---- Chat -----------------------------------------------------------
        .route("/v1/chat/completions", post(openai_shim::chat_completions))
        // ---- Embeddings -----------------------------------------------------
        .route("/v1/embeddings", post(openai_shim::create_embeddings))
        // ---- Audio ----------------------------------------------------------
        .route(
            "/v1/audio/transcriptions",
            post(openai_shim::create_transcription),
        )
        // -- AI status endpoints ----------------------------------------------
        // ---- Chat -----------------------------------------------------------
        .route(
            "/v1/chat/completions/status",
            get(status::chat_completions_status),
        )
        // ---- Audio ----------------------------------------------------------
        .route(
            "/v1/audio/transcriptions/status",
            get(status::audio_transcriptions_status),
        )
        // -- Model Manager ----------------------------------------------------
        .route("/v1/models", get(model_man::list_models))
        .route("/v1/models/:model", get(model_man::retrieve_model))
        .route("/v1/models/:model", delete(model_man::delete_model))
        // -- Miscellaneous services -------------------------------------------
        .route("/v1/misc/version", get(misc::edgen_version))
        // -- Catch-all route to log all requests ------------------------------
        .fallback(catch_all)
}

async fn catch_all(method: Method, uri: Uri) -> impl IntoResponse {
    // Log the requested path for debugging or information purposes
    warn!("Unknown route requested: {} {}", method, uri);

    // Return a 404 Not Found status code without any body to mimic a non-existent endpoint
    StatusCode::NOT_FOUND
}

#[cfg(test)]
mod test {
    use super::catch_all;
    use axum::http::StatusCode;
    use axum::Router;
    use axum_test::TestServer;

    #[tokio::test]
    async fn test_get_any_path() {
        let router = Router::new().fallback(catch_all);

        let server = TestServer::new(router).expect("cannot instantiate TestServer");

        let resp = server.get("/v1/does/not_exist").await;

        assert_eq!(resp.status_code(), StatusCode::NOT_FOUND);

        let resp = server.get("/v0/misc/version").await;

        assert_eq!(resp.status_code(), StatusCode::NOT_FOUND);

        let resp = server.get("/misc/version").await;

        assert_eq!(resp.status_code(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_post_any_path() {
        let router = Router::new().fallback(catch_all);

        let server = TestServer::new(router).expect("cannot instantiate TestServer");

        let resp = server.post("/v1/misc/version").await;

        assert_eq!(resp.status_code(), StatusCode::NOT_FOUND);

        let resp = server.post("/v1/does/not_exist").await;

        assert_eq!(resp.status_code(), StatusCode::NOT_FOUND);

        let resp = server.post("/v0/misc/version").await;

        assert_eq!(resp.status_code(), StatusCode::NOT_FOUND);

        let resp = server.post("/misc/version").await;

        assert_eq!(resp.status_code(), StatusCode::NOT_FOUND);
    }
}
