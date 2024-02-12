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

use std::path::Path;
use std::time::Duration;

use crate::BoxedFuture;
use rubato::Resampler;
use serde::Serialize;
use smol::future::Future;
use thiserror::Error;
use tracing::info;
use utoipa::ToSchema;
use uuid::Uuid;

#[derive(Serialize, Error, Debug)]
pub enum WhisperEndpointError {
    #[error("failed to advance context: {0}")]
    Advance(String),
    #[error("failed to decode result: {0}")]
    Decode(String),
    #[error("failed to load the model: {0}")]
    Load(String),
    #[error("failed to create a session: {0}")]
    SessionCreationFailed(String),
    #[error("no matching session found")]
    SessionNotFound,
    #[error("failed to parse audio file data: {0}")]
    Audio(#[from] AudioError),
}

pub struct TranscriptionArgs {
    pub file: Vec<u8>,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub temperature: Option<f32>,
    pub create_session: bool,
    pub session: Option<Uuid>,
}

pub trait WhisperEndpoint {
    /// Given an audio segment with several arguments, return a [`Box`]ed [`Future`] which may
    /// eventually contain its transcription in [`String`] form.
    fn transcription<'a>(
        &'a self,
        model_path: impl AsRef<Path> + Send + 'a,
        args: TranscriptionArgs,
    ) -> BoxedFuture<Result<(String, Option<Uuid>), WhisperEndpointError>>;

    /// Unloads everything from memory.
    fn reset(&self);
}

/// Return the [`Duration`] for which a whisper model lives while not being used before being
/// unloaded from memory.
pub fn inactive_whisper_ttl() -> Duration {
    // TODO this should come from the settings
    Duration::from_secs(5 * 60)
}

/// Return the [`Duration`] for which a whisper model session lives while not being used before
/// being unloaded from memory.
pub fn inactive_whisper_session_ttl() -> Duration {
    // TODO this should come from the settings
    Duration::from_secs(2 * 60)
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

/// Parse an audio file and convert it into a *PCM* audio segment, using the optimal sample rate
/// for whisper models.
pub fn parse_pcm(audio_file: &[u8]) -> Result<Vec<f32>, AudioError> {
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

    info!("Parsing audio file ({} bytes)", audio_file.len());

    // Initialisation.
    let cursor = std::io::Cursor::new(audio_file.to_vec());
    let stream = MediaSourceStream::new(Box::new(cursor), Default::default());

    let hint = Hint::new();

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    // TODO this gets stuck in a loop for some invalid files
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
