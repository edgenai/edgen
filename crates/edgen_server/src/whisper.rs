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

use once_cell::sync::Lazy;

use edgen_core::whisper::{TranscriptionArgs, WhisperEndpoint, WhisperEndpointError};
use edgen_rt_whisper_cpp::WhisperCppEndpoint;

use crate::model::Model;

static ENDPOINT: Lazy<WhisperCppEndpoint> = Lazy::new(Default::default);

pub async fn create_transcription(
    file: &[u8],
    model: Model,
    language: Option<&str>,
    prompt: Option<&str>,
    temperature: Option<f32>,
) -> Result<String, WhisperEndpointError> {
    let args = TranscriptionArgs {
        file: file.to_vec(),
        language: language.map(move |s| s.to_string()),
        prompt: prompt.map(move |s| s.to_string()),
        temperature,
        session: None,
    };

    ENDPOINT
        .transcription(
            model
                .file_path()
                .map_err(move |e| WhisperEndpointError::Load(e.to_string()))?,
            args,
        )
        .await
}

pub async fn reset_environment() {
    ENDPOINT.reset()
}
