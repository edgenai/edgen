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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Model, ModelKind};
    use edgen_core::settings::SETTINGS;
    use levenshtein;
    use std::path::PathBuf;

    async fn init_settings_for_test() {
        SETTINGS
            .write()
            .await
            .init()
            .await
            .expect("Failed to initialise settings");
    }

    fn frost() -> String {
        " The woods are lovely, dark and deep, \
         but I have promises to keep \
         and miles to go before I sleep, \
         and miles to go before I sleep."
            .to_string()
    }

    #[tokio::test]
    #[ignore] // this test hangs sometimes
    async fn test_audio_transcriptions() {
        init_settings_for_test().await;
        let model_name = "ggml-distil-small.en.bin".to_string();
        let repo = "distil-whisper/distil-small.en".to_string();
        let dir = SETTINGS
            .read()
            .await
            .read()
            .await
            .audio_transcriptions_models_dir
            .to_string();
        let mut model = Model::new(ModelKind::Whisper, &model_name, &repo, &PathBuf::from(&dir));
        assert!(model.preload().await.is_ok());

        let sound = include_bytes!("../resources/frost.wav");
        let response = create_transcription(sound, model, None, None, None).await;

        assert!(response.is_ok(), "cannot create transcription");

        let expected_text = frost();
        let actual_text = response.unwrap();

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
