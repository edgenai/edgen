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

#[cfg(test)]
mod tests {
    /*
    use alsa::pcm::{Access, Format, HwParams};
    use alsa::{Direction, ValueOr, PCM};
    use tokio::sync::mpsc::error::TryRecvError;

    use crate::*;

    #[derive(Error, Debug)]
    enum TestError {
        #[error("whisper error: {0}")]
        Whisper(#[from] edgen_core::whisper::WhisperError),
        #[error("decode session error: {0}")]
        Session(#[from] edgen_core::whisper::DecodeSessionError),
        #[error("whisper.cpp error: {0}")]
        WhisperCpp(#[from] WhisperCppError),
        #[error("alsa error: {0}")]
        Alsa(#[from] alsa::Error),
        #[error("failed to write test file: {0}")]
        File(#[from] std::io::Error),
    }

    #[tokio::test]
    async fn live_transcription() -> Result<(), TestError> {

        //TODO change to use env variables
        let model = WhisperCpp::load("/home/pedro/dev/models/ggml-base.en.bin")?;

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut session = model
            .new_session(Box::new(move |str| {
                println!("{str}");
                tx.send(str).unwrap();
            }))
            .await?;

        let _rate;
        let pcm = PCM::new("default", Direction::Capture, false)?;
        {
            // For this example, we assume 44100Hz, one channel, 16 bit audio.
            let hwp = HwParams::any(&pcm)?;
            hwp.set_channels(1)?;
            hwp.set_rate(16000, ValueOr::Nearest)?;
            hwp.set_format(Format::s16())?;
            hwp.set_access(Access::RWInterleaved)?;
            pcm.hw_params(&hwp)?;

            _rate = hwp.get_rate()?; // is there any side effect that justifies this assignment?
        }
        pcm.start()?;

        let io = pcm.io_i16()?;
        let mut buf = [0i16; 8192 * 10];

        loop {
            let mut stop = false;

            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
            let _size_read = io.readi(&mut buf)?;
            let samples: Vec<_> = buf[..].iter().map(|v| *v as f32 / 32768.).collect();

            session.push(&samples)?;
            println!("aaaaaa");

            loop {
                println!("huh?");
                let output = rx.try_recv();
                match output {
                    Ok(str) => {
                        println!("{str}");
                        if str.contains("stop") | str.contains("Stop") | str.contains("STOP") {
                            session.end().await?;
                            stop = true;
                            break;
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(_) => {
                        panic!()
                    }
                }
            }

            if stop {
                break;
            }
        }
        Ok(())
    }
    */

    /*
    static ENDPOINT: OnceCell<Arc<RwLock<WhisperEndpoint<WhisperCpp>>>> = OnceCell::const_new();

    #[tokio::test]
    async fn server() -> Result<(), TestError> {
        use axum::Router;
        use axum::routing::post;
        use axum::http::StatusCode;
        use axum::Json;

        async fn creation_helper() -> Arc<RwLock<WhisperEndpoint<WhisperCpp>>> {
            Arc::new(RwLock::new(WhisperEndpoint::default()))
        }

        async fn request_helper(payload: Json<edgen_core::whisper::StandaloneForm>) -> (StatusCode, String) {
            let mut locked = ENDPOINT.get_or_init(creation_helper).await.write().await;
            locked.standalone(payload).await
        }

        let app = Router::new()
            .route("/", post(request_helper));

        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
        axum::serve(listener, app).await.unwrap();

        Ok(())
    }

    */
}
