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
 
use std::sync::Arc;

use dashmap::DashMap;
use once_cell::sync::Lazy;
use rubato::Resampler;
use serde_derive::Serialize;
use thiserror::Error;
use time::Duration;
use utoipa::ToSchema;
use uuid::Uuid;

use edgen_core::whisper::{DecodeSessionError, Whisper, WhisperError, WhisperSession};
use edgen_rt_whisper_cpp::WhisperCpp;

use crate::model::Model;
use crate::model::ModelError;
use crate::util::{Perishable, PerishableReadGuard};

static ENDPOINT: Lazy<WhisperEndpoint> = Lazy::new(Default::default);

/// The number of seconds that a `whisper` model will remain loaded for before being automatically
/// unloaded.
pub const WHISPER_INACTIVE_TTL: Duration = Duration::seconds(5 * 60);

pub async fn create_transcription(
    file: &[u8],
    model: Model,
    language: Option<&str>,
    prompt: Option<&str>,
    temperature: Option<f32>,
) -> Result<String, WhisperEndpointError> {
    ENDPOINT
        .standalone_decode(file, model, language, prompt, temperature)
        .await
}

pub async fn reset_environment() {
    ENDPOINT.reset()
}

#[derive(Serialize, Error, ToSchema, Debug)]
pub enum WhisperEndpointError {
    #[error("the provided model file name does does not exist, or isn't a file: ({0})")]
    FileNotFound(String),
    #[error("there is no session associated with the provided uuid ({0})")]
    SessionNotFound(String),
    #[error("internal error: {0}")]
    Internal(#[from] WhisperError),
    #[error("error in decode session: {0}")]
    Session(#[from] DecodeSessionError),
    #[error("failed to parse audio data: {0}")]
    Audio(#[from] AudioError),
    #[error("failed to load model: {0}")]
    Model(#[from] ModelError),
}

struct WhisperInstance {
    model: Box<dyn Whisper + Send + Sync>,
    _sessions: DashMap<Uuid, WhisperSession>,
}

impl WhisperInstance {
    fn new(model: impl Whisper + Send + Sync + 'static) -> Self {
        Self {
            model: Box::new(model),
            _sessions: DashMap::new(),
        }
    }

    async fn decode(&self, data: &[f32]) -> Result<String, WhisperEndpointError> {
        Ok(self.model.decode(data).await?)
    }

    #[allow(dead_code)]
    async fn new_session(
        &self,
        callback: impl Fn(String) + Send + 'static,
    ) -> Result<Uuid, WhisperEndpointError> {
        let session = self.model.new_session(Box::new(callback)).await?;

        let uuid = Uuid::new_v4();

        self._sessions.insert(uuid, session);

        Ok(uuid)
    }

    #[allow(dead_code)]
    async fn advance_decode(&self, uuid: Uuid, data: &[f32]) -> Result<(), WhisperEndpointError> {
        let session = self
            ._sessions
            .get(&uuid)
            .ok_or(WhisperEndpointError::SessionNotFound(uuid.to_string()))?;

        session.push(data)?;

        Ok(())
    }
}

#[derive(Default)]
pub struct WhisperEndpoint {
    instances: DashMap<String, Arc<Perishable<WhisperInstance>>>,
}

impl WhisperEndpoint {
    fn get_or_create(
        &self,
        model: &Model,
    ) -> Result<Arc<Perishable<WhisperInstance>>, WhisperEndpointError> {
        let path = model.file_path()?;
        let key = path.to_string_lossy();

        if !self.instances.contains_key(key.as_ref()) {
            self.instances.insert(
                key.to_string(),
                Arc::new(Perishable::with_ttl(WHISPER_INACTIVE_TTL)),
            );
        }

        Ok(self.instances.get(key.as_ref()).expect("Model instance not found. This should never happen as a new instance is added if there isn't one").value().clone())
    }

    async fn lock<'a>(
        instance: &'a Arc<Perishable<WhisperInstance>>,
        model: &Model,
    ) -> Result<PerishableReadGuard<'a, WhisperInstance>, WhisperEndpointError> {
        let path = model.file_path()?;

        instance
            .get_or_try_init(move || async move {
                //let model_path = PROJECT_DIRS.config_dir().join(&path);

                if !path.is_file() {
                    return Err(WhisperEndpointError::FileNotFound(
                        path.to_string_lossy().to_string(),
                    ));
                }

                // TODO change this when more backends exist
                let model = WhisperCpp::load(path)?;

                Ok(WhisperInstance::new(model))
            })
            .await
    }

    pub async fn standalone_decode(
        &self,
        audio: &[u8],
        model: Model,
        _language: Option<&str>,   // currently not used
        _prompt: Option<&str>,     // currently not used
        _temperature: Option<f32>, // currently not used
    ) -> Result<String, WhisperEndpointError> {
        let instance = self.get_or_create(&model)?;
        let locked = Self::lock(&instance, &model).await?;

        let pcm = to_pcm(audio)?;

        locked.decode(&pcm).await
    }

    fn reset(&self) {
        self.instances.clear()
    }
}

#[derive(Serialize, Error, ToSchema, Debug)]
pub enum AudioError {
    #[error("failed to parse mime data: {0}")]
    Parse(String),
    #[error("failed to initialise resampler: {0}")]
    ResamplerInit(String),
    #[error("failed to resample the input audio: {0}")]
    Resample(String),
}

fn to_pcm(audio_file: &[u8]) -> Result<Vec<f32>, AudioError> {
    use rubato::{SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
    use symphonia::core::audio::Signal;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::errors::Error;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    /// The optimal sample rate for whisper models.
    const OPTIMAL_SAMPLE_RATE: u32 = 16000;

    // Initialisation.
    let cursor = std::io::Cursor::new(audio_file.to_vec());
    let stream = MediaSourceStream::new(Box::new(cursor), Default::default());

    let hint = Hint::new();

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, stream, &fmt_opts, &meta_opts)
        .map_err(move |e| AudioError::Parse(format!("failed to probe audio data: {e}")))?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or(AudioError::Parse("codec is null".to_string()))?;

    let dec_opts: DecoderOptions = Default::default();

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)
        .map_err(move |e| AudioError::Parse(format!("failed to initialize decoder: {e}")))?;

    let track_id = track.id;

    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(move || AudioError::Parse("could not get sample rate".to_string()))?;

    let mut samples = vec![];

    // Decoding loop.
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::ResetRequired) => {
                break;
            }
            Err(Error::IoError(e)) => {
                // TODO this isnt ideal, but gonna have to wait for symphonia to be updated
                // https://github.com/pdeljanov/Symphonia/issues/134#issuecomment-1146990539
                if e.kind() == std::io::ErrorKind::UnexpectedEof && e.to_string() == "end of stream"
                {
                    break;
                } else {
                    return Err(AudioError::Parse(format!("unexpected end of file: {e:#?}")));
                }
            }
            Err(e) => {
                return Err(AudioError::Parse(format!(
                    "failed to acquire next packet: {e}"
                )));
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = decoder
            .decode(&packet)
            .map_err(move |e| AudioError::Parse(format!("failed to decode packet: {e}")))?;

        let mut sample_slice = decoded.make_equivalent::<f32>();
        decoded.convert(&mut sample_slice);
        samples.extend_from_slice(sample_slice.chan(0));
    }

    // Resample the pcm data if necessary.
    if sample_rate != OPTIMAL_SAMPLE_RATE {
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let mut resampler = SincFixedIn::<f64>::new(
            OPTIMAL_SAMPLE_RATE as f64 / sample_rate as f64,
            2.0,
            params,
            samples.len(),
            1,
        )
        .map_err(move |e| AudioError::ResamplerInit(e.to_string()))?;

        let pre: Vec<_> = samples.drain(..).map(move |x| x as f64).collect();
        let mut resampled = resampler
            .process(&[pre], None)
            .map_err(move |e| AudioError::Resample(e.to_string()))?;
        samples = resampled[0].drain(..).map(move |x| x as f32).collect();
    }

    Ok(samples)
}
