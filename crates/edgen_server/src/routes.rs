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

use axum::Router;

use crate::misc;
use crate::openai_shim;
use crate::status;

pub fn routes() -> Router {
    Router::new()
        // -- AI endpoints -----------------------------------------------------
        // ---- Chat -----------------------------------------------------------
        .route(
            "/v1/chat/completions",
            axum::routing::post(openai_shim::chat_completions),
        )
        // ---- Audio ----------------------------------------------------------
        .route(
            "/v1/audio/transcriptions",
            axum::routing::post(openai_shim::create_transcription),
        )
        // -- AI status endpoints ----------------------------------------------
        // ---- Chat -----------------------------------------------------------
        .route(
            "/v1/chat/completions/status",
            axum::routing::get(status::chat_completions_status),
        )
        // ---- Audio ----------------------------------------------------------
        .route(
            "/v1/audio/transcriptions/status",
            axum::routing::get(status::audio_transcriptions_status),
        )
        // -- Miscellaneous services -------------------------------------------
        .route("/v1/misc/version", axum::routing::get(misc::edgen_version))
}
