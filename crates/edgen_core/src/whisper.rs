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

use serde::Serialize;
use thiserror::Error;
use utoipa::ToSchema;
use uuid::Uuid;

use crate::BoxedFuture;

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

#[async_trait::async_trait]
pub trait WhisperEndpoint {
    /// Given an audio segment with several arguments, return a transcription in [`String`] form.
    async fn transcription(
        &self,
        model_path: impl AsRef<Path> + Send,
        args: TranscriptionArgs,
    ) -> Result<(String, Option<Uuid>), WhisperEndpointError>;

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

pub mod parse {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };
    use symphonia::core::audio::Signal;
    use symphonia::core::codecs::{CodecType, DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::errors::Error;
    use symphonia::core::formats::{FormatOptions, Track};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;
    use tracing::info;

    use crate::whisper::AudioError;

    /// Parse an audio file and convert it into a *PCM* audio segment, using the optimal sample rate
    /// for whisper models.
    pub fn pcm(audio_file: &[u8]) -> Result<Vec<f32>, AudioError> {
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

        debug_track_data(track);

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
                    if e.kind() == std::io::ErrorKind::UnexpectedEof
                        && e.to_string() == "end of stream"
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

    /// Debug track information through `tracing::info!`.
    fn debug_track_data(track: &Track) {
        let mut track_data = vec![];

        // track.codec_params.codec
        track_data.push(format!("codec: {}", codec_name(track.codec_params.codec)));

        if let Some(sample_rate) = track.codec_params.sample_rate {
            track_data.push(format!("sample rate: {sample_rate}"));
        }

        if let Some(sample_format) = track.codec_params.sample_format {
            track_data.push(format!("sample format: {sample_format:?}"));
        }

        if let Some(channels) = track.codec_params.channels {
            track_data.push(format!("channels: {}", channels.count()));
        }

        info!("Audio file data: {track_data:?}");
    }

    /// Returns the name of the given codec as a `str`.
    fn codec_name(codec: CodecType) -> &'static str {
        use symphonia::core::codecs::{
            CODEC_TYPE_AAC, CODEC_TYPE_AC4, CODEC_TYPE_ADPCM_G722, CODEC_TYPE_ADPCM_G726,
            CODEC_TYPE_ADPCM_G726LE, CODEC_TYPE_ADPCM_IMA_QT, CODEC_TYPE_ADPCM_IMA_WAV,
            CODEC_TYPE_ADPCM_MS, CODEC_TYPE_ALAC, CODEC_TYPE_ATRAC1, CODEC_TYPE_ATRAC3,
            CODEC_TYPE_ATRAC3PLUS, CODEC_TYPE_ATRAC9, CODEC_TYPE_DCA, CODEC_TYPE_EAC3,
            CODEC_TYPE_FLAC, CODEC_TYPE_MONKEYS_AUDIO, CODEC_TYPE_MP1, CODEC_TYPE_MP2,
            CODEC_TYPE_MP3, CODEC_TYPE_MUSEPACK, CODEC_TYPE_OPUS, CODEC_TYPE_PCM_ALAW,
            CODEC_TYPE_PCM_F32BE, CODEC_TYPE_PCM_F32BE_PLANAR, CODEC_TYPE_PCM_F32LE,
            CODEC_TYPE_PCM_F32LE_PLANAR, CODEC_TYPE_PCM_F64BE, CODEC_TYPE_PCM_F64BE_PLANAR,
            CODEC_TYPE_PCM_F64LE, CODEC_TYPE_PCM_F64LE_PLANAR, CODEC_TYPE_PCM_MULAW,
            CODEC_TYPE_PCM_S16BE, CODEC_TYPE_PCM_S16BE_PLANAR, CODEC_TYPE_PCM_S16LE,
            CODEC_TYPE_PCM_S16LE_PLANAR, CODEC_TYPE_PCM_S24BE, CODEC_TYPE_PCM_S24BE_PLANAR,
            CODEC_TYPE_PCM_S24LE, CODEC_TYPE_PCM_S24LE_PLANAR, CODEC_TYPE_PCM_S32BE,
            CODEC_TYPE_PCM_S32BE_PLANAR, CODEC_TYPE_PCM_S32LE, CODEC_TYPE_PCM_S32LE_PLANAR,
            CODEC_TYPE_PCM_S8, CODEC_TYPE_PCM_S8_PLANAR, CODEC_TYPE_PCM_U16BE,
            CODEC_TYPE_PCM_U16BE_PLANAR, CODEC_TYPE_PCM_U16LE, CODEC_TYPE_PCM_U16LE_PLANAR,
            CODEC_TYPE_PCM_U24BE, CODEC_TYPE_PCM_U24BE_PLANAR, CODEC_TYPE_PCM_U24LE,
            CODEC_TYPE_PCM_U24LE_PLANAR, CODEC_TYPE_PCM_U32BE, CODEC_TYPE_PCM_U32BE_PLANAR,
            CODEC_TYPE_PCM_U32LE, CODEC_TYPE_PCM_U32LE_PLANAR, CODEC_TYPE_PCM_U8,
            CODEC_TYPE_PCM_U8_PLANAR, CODEC_TYPE_SPEEX, CODEC_TYPE_TTA, CODEC_TYPE_VORBIS,
            CODEC_TYPE_WAVPACK, CODEC_TYPE_WMA,
        };

        match codec {
            CODEC_TYPE_NULL => "NULL",
            CODEC_TYPE_PCM_S32LE => "PCM i32 LE Interleaved",
            CODEC_TYPE_PCM_S32LE_PLANAR => "PCM i32 LE Planar",
            CODEC_TYPE_PCM_S32BE => "PCM i32 BE Interleaved",
            CODEC_TYPE_PCM_S32BE_PLANAR => "PCM i32 BE Planar",
            CODEC_TYPE_PCM_S24LE => "PCM i24 LE Interleaved",
            CODEC_TYPE_PCM_S24LE_PLANAR => "PCM i24 LE Planar",
            CODEC_TYPE_PCM_S24BE => "PCM i24 BE Interleaved",
            CODEC_TYPE_PCM_S24BE_PLANAR => "PCM i24 BE Planar",
            CODEC_TYPE_PCM_S16LE => "PCM i16 LE Interleaved",
            CODEC_TYPE_PCM_S16LE_PLANAR => "PCM i16 LE Planar",
            CODEC_TYPE_PCM_S16BE => "PCM i16 BE Interleaved",
            CODEC_TYPE_PCM_S16BE_PLANAR => "PCM i16 BE Planar",
            CODEC_TYPE_PCM_S8 => "PCM i8 Interleaved",
            CODEC_TYPE_PCM_S8_PLANAR => "PCM i8 Planar",
            CODEC_TYPE_PCM_U32LE => "PCM u32 LE Interleaved",
            CODEC_TYPE_PCM_U32LE_PLANAR => "PCM u32 LE Planar",
            CODEC_TYPE_PCM_U32BE => "PCM u32 BE Interleaved",
            CODEC_TYPE_PCM_U32BE_PLANAR => "PCM u32 BE Planar",
            CODEC_TYPE_PCM_U24LE => "PCM u24 LE Interleaved",
            CODEC_TYPE_PCM_U24LE_PLANAR => "PCM u24 LE Planar",
            CODEC_TYPE_PCM_U24BE => "PCM u24 BE Interleaved",
            CODEC_TYPE_PCM_U24BE_PLANAR => "PCM u24 BE Planar",
            CODEC_TYPE_PCM_U16LE => "PCM u16 LE Interleaved",
            CODEC_TYPE_PCM_U16LE_PLANAR => "PCM u16 LE Planar",
            CODEC_TYPE_PCM_U16BE => "PCM u16 BE Interleaved",
            CODEC_TYPE_PCM_U16BE_PLANAR => "PCM u16 BE Planar",
            CODEC_TYPE_PCM_U8 => "PCM u8 Interleaved",
            CODEC_TYPE_PCM_U8_PLANAR => "PCM u8 Planar",
            CODEC_TYPE_PCM_F32LE => "PCM f32 LE Interleaved",
            CODEC_TYPE_PCM_F32LE_PLANAR => "PCM f32 LE Planar",
            CODEC_TYPE_PCM_F32BE => "PCM f32 BE Interleaved",
            CODEC_TYPE_PCM_F32BE_PLANAR => "PCM f32 BE Planar",
            CODEC_TYPE_PCM_F64LE => "PCM f64 LE Interleaved",
            CODEC_TYPE_PCM_F64LE_PLANAR => "PCM f64 LE Planar",
            CODEC_TYPE_PCM_F64BE => "PCM f64 BE Interleaved",
            CODEC_TYPE_PCM_F64BE_PLANAR => "PCM f64 BE Planar",
            CODEC_TYPE_PCM_ALAW => "PCM A-law",
            CODEC_TYPE_PCM_MULAW => "PCM Mu-law",
            CODEC_TYPE_ADPCM_G722 => "ADPCM G.722",
            CODEC_TYPE_ADPCM_G726 => "ADPCM G.726",
            CODEC_TYPE_ADPCM_G726LE => "ADPCM G.726 LE",
            CODEC_TYPE_ADPCM_MS => "ADPCM Microsoft",
            CODEC_TYPE_ADPCM_IMA_WAV => "ADPCM IDA WAV",
            CODEC_TYPE_ADPCM_IMA_QT => "ADPCM IDA QuickTime",
            CODEC_TYPE_VORBIS => "Vorbis",
            CODEC_TYPE_MP1 => "MP1",
            CODEC_TYPE_MP2 => "MP2",
            CODEC_TYPE_MP3 => "MP3",
            CODEC_TYPE_AAC => "Advanced Audio Encoding",
            CODEC_TYPE_OPUS => "Opus",
            CODEC_TYPE_SPEEX => "Speex",
            CODEC_TYPE_MUSEPACK => "Musepack",
            CODEC_TYPE_ATRAC1 => "ATRAC1",
            CODEC_TYPE_ATRAC3 => "ATRAC3",
            CODEC_TYPE_ATRAC3PLUS => "ATRAC3+",
            CODEC_TYPE_ATRAC9 => "ATRAC9",
            CODEC_TYPE_EAC3 => "AC-3, E-AC-3, Dolby Digital (ATSC A/52)",
            CODEC_TYPE_AC4 => "Dolby AC-4 (ETSI TS 103 190)",
            CODEC_TYPE_DCA => "DTS Coherent Acoustics",
            CODEC_TYPE_WMA => "Windows Media Audio",
            CODEC_TYPE_FLAC => "Free Lossless Audio Codec",
            CODEC_TYPE_WAVPACK => "WavPack",
            CODEC_TYPE_MONKEYS_AUDIO => "Monkey's Audio (APE)",
            CODEC_TYPE_ALAC => "Apple Lossless Audio Codec",
            CODEC_TYPE_TTA => "True Audio (TTA)",
            _ => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn parse_audio_succeeds() {
        let sound = include_bytes!("../../edgen_server/resources/frost.wav");
        assert!(parse::pcm(sound).is_ok(), "cannot parse audio file");
    }

    #[test]
    fn parse_audio_fails() {
        let sound: Vec<u8> = vec![0, 1, 2, 3];
        assert!(parse::pcm(&sound).is_err(), "can parse non-audio file");
    }
}
