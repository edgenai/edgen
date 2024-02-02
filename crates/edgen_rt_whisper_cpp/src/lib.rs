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

use std::future::Future;

use edgen_core::settings::SETTINGS;
use thiserror::Error;
use whisper_cpp::{WhisperModel, WhisperParams, WhisperSampling};

use edgen_core::whisper::{Whisper, WhisperError, WhisperRunner, WhisperSession};

#[derive(Error, Debug)]
enum WhisperCppError {
    #[error("failed initialize whisper.cpp model: {0}")]
    Initialization(#[from] whisper_cpp::WhisperError),
}

pub struct WhisperCpp {
    model: WhisperModel,
}

impl WhisperCpp {
    pub fn load<P>(model_path: P) -> Result<Self, WhisperError>
    where
        P: AsRef<std::path::Path>,
    {
        Ok(Self {
            model: WhisperModel::new_from_file(model_path, false)
                .map_err(move |e| WhisperError::ModelInitialization(e.to_string()))?,
        })
    }

    async fn async_decode(&self, data: &[f32]) -> Result<String, WhisperError> {
        let mut session: whisper_cpp::WhisperSession = self
            .model
            .new_session()
            .await
            .map_err(move |e| WhisperError::SessionInitialization(e.to_string()))?;

        let mut params = WhisperParams::new(WhisperSampling::default_greedy());
        params.thread_count = SETTINGS.read().await.read().await.auto_threads(false);

        session
            .full(params, data)
            .await
            .map_err(move |e| WhisperError::Internal(e.to_string()))?;

        let mut res = "".to_string();
        for i in 0..session.segment_count() {
            res += &*session
                .segment_text(i)
                .map_err(move |e| WhisperError::Internal(e.to_string()))?;
        }

        Ok(res)
    }

    async fn async_new_session(
        &self,
        callback: Box<dyn Fn(String) + Send + 'static>,
    ) -> Result<WhisperSession, WhisperError> {
        Ok(WhisperSession::new(
            WhisperCppRunner {
                session: self
                    .model
                    .new_session()
                    .await
                    .map_err(move |e| WhisperError::SessionInitialization(e.to_string()))?,
            },
            callback,
        ))
    }
}

impl Whisper for WhisperCpp {
    fn decode<'a>(
        &'a self,
        data: &'a [f32],
    ) -> Box<dyn Future<Output = Result<String, WhisperError>> + Send + Unpin + 'a> {
        // todo are the 2 boxes really needed?
        let fut = Box::pin(self.async_decode(data));
        Box::new(fut)
    }

    fn new_session<'a>(
        &'a self,
        callback: Box<dyn Fn(String) + Send + 'static>,
    ) -> Box<dyn Future<Output = Result<WhisperSession, WhisperError>> + Send + Unpin + 'a> {
        // todo are the 2 boxes really needed?
        let fut = Box::pin(self.async_new_session(callback));
        Box::new(fut)
    }
}

struct WhisperCppRunner {
    session: whisper_cpp::WhisperSession,
}

impl WhisperRunner for WhisperCppRunner {
    async fn forward_decode(&mut self, data: &[f32]) -> Result<String, WhisperError> {
        let params = WhisperParams::new(WhisperSampling::default_greedy());

        self.session
            .full(params, data)
            .await
            .map_err(move |e| WhisperError::Internal(e.to_string()))?;

        let _segment_count = self.session.segment_count();
        /*self
        .session
        .segment_text(segment_count - 1)
        .map_err(move |e| WhisperError::Internal(e.to_string()))*/

        let mut res = "".to_string();
        for i in 0..self.session.segment_count() {
            // we should review if this is correct!
            res += &*self
                .session
                .segment_text(i)
                .map_err(move |e| WhisperError::Internal(e.to_string()))?;
        }

        Ok(res)
    }
}
