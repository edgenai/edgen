use std::io::BufWriter;
use std::io::{Cursor, IntoInnerError};
use std::path::Path;

use candle_core::backend::BackendDevice;
use candle_core::{CudaDevice, DType, Device, IndexOp, Module, Tensor, D};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use candle_transformers::models::{stable_diffusion, wuerstchen};
use image::{ImageBuffer, ImageError, ImageFormat, Rgb};
use rand::random;
use thiserror::Error;
use tokenizers::Tokenizer;
use tracing::{debug, info, info_span, warn};

use edgen_core::image_generation::{
    ImageGenerationArgs, ImageGenerationEndpoint, ImageGenerationEndpointError, ModelFiles,
};
use edgen_core::settings::{DevicePolicy, SETTINGS};

#[derive(Error, Debug)]
enum CandleError {
    #[error("The prompt is too long, {len} > max-tokens ({max})")]
    PromptTooLong { len: usize, max: usize },
    #[error("No clip2 available for the configuration")]
    Clip2Unavailable,
    #[error("Output has wrong number of dimensions, got {dims}, but expected {expected}")]
    BadDims { dims: usize, expected: usize },
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("Could not create image buffer from tensor output")]
    BadOutput,
    #[error(transparent)]
    EncodeProcessFailed(#[from] ImageError),
    #[error(transparent)]
    EncodeWriteFailed(#[from] IntoInnerError<BufWriter<Cursor<Vec<u8>>>>),
}

fn sd_text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer: impl AsRef<Path>,
    clip_weights: impl AsRef<Path>,
    config: &stable_diffusion::StableDiffusionConfig,
    device: &Device,
    dtype: DType,
    use_guide_scale: bool,
    first_clip: bool,
) -> Result<Tensor, CandleError> {
    let tokenizer =
        Tokenizer::from_file(tokenizer).map_err(|e| CandleError::Tokenizer(e.to_string()))?;
    let pad_id = match &config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };

    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| CandleError::Tokenizer(e.to_string()))?
        .get_ids()
        .to_vec();
    if tokens.len() > config.clip.max_position_embeddings {
        return Err(CandleError::PromptTooLong {
            len: tokens.len(),
            max: config.clip.max_position_embeddings,
        });
    }
    while tokens.len() < config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    let clip_config = if first_clip {
        &config.clip
    } else {
        config.clip2.as_ref().ok_or(CandleError::Clip2Unavailable)?
    };
    let text_model =
        stable_diffusion::build_clip_transformer(clip_config, clip_weights, device, DType::F32)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let text_embeddings = if use_guide_scale {
        let mut uncond_tokens = tokenizer
            .encode(uncond_prompt, true)
            .map_err(|e| CandleError::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec();
        if uncond_tokens.len() > config.clip.max_position_embeddings {
            return Err(CandleError::PromptTooLong {
                len: uncond_tokens.len(),
                max: config.clip.max_position_embeddings,
            });
        }
        while uncond_tokens.len() < config.clip.max_position_embeddings {
            uncond_tokens.push(pad_id)
        }

        let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;
        let uncond_embeddings = text_model.forward(&uncond_tokens)?;

        Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
    } else {
        text_embeddings.to_dtype(dtype)?
    };
    Ok(text_embeddings)
}

fn sd_to_bitmap(
    vae: &AutoEncoderKL,
    latents: &Tensor,
    vae_scale: f64,
    bsize: usize,
) -> Result<Vec<Vec<u8>>, CandleError> {
    let images = vae.decode(&(latents / vae_scale)?)?;
    let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
    let mut res = vec![];
    res.reserve(bsize);
    for batch in 0..bsize {
        let image = images.i(batch)?;
        let (channel, height, width) = image.dims3()?;
        if channel != 3 {
            return Err(CandleError::BadDims {
                dims: channel,
                expected: 3,
            });
        }
        let img = image.permute((1, 2, 0))?.flatten_all()?;
        let pixels = img.to_vec1::<u8>()?;
        let buf = ImageBuffer::<Rgb<u8>, _>::from_vec(width as u32, height as u32, pixels)
            .ok_or(CandleError::BadOutput)?;
        let mut encoded = BufWriter::new(Cursor::new(Vec::new()));
        buf.write_to(&mut encoded, ImageFormat::Png)?;
        res.push(encoded.into_inner()?.into_inner());
    }
    Ok(res)
}

fn sd_generate_image(
    model: ModelFiles,
    args: ImageGenerationArgs,
    device: Device,
) -> Result<Vec<Vec<u8>>, CandleError> {
    let _span = info_span!("sd_gen_image", images = args.images, steps = args.steps).entered();
    let config = stable_diffusion::StableDiffusionConfig::v2_1(None, args.height, args.width);
    let scheduler = config.build_scheduler(args.steps)?;
    let use_guide_scale = args.guidance_scale > 1.0;
    let dtype = DType::F16;
    let bsize = 1;

    device.set_seed(args.seed.unwrap_or(random::<u64>()))?;

    let which = if model.clip2_weights.is_some() {
        vec![true, false]
    } else {
        vec![true]
    };
    let text_embeddings = which
        .iter()
        .map(|first| {
            let clip = if *first {
                &model.clip_weights
            } else {
                model.clip2_weights.as_ref().unwrap()
            };
            sd_text_embeddings(
                &args.prompt,
                &args.uncond_prompt,
                &model.tokenizer,
                clip,
                &config,
                &device,
                dtype,
                use_guide_scale,
                *first,
            )
        })
        .collect::<Result<Vec<_>, CandleError>>()?;

    let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;
    let text_embeddings = text_embeddings.repeat((bsize, 1, 1))?;

    let vae = config.build_vae(model.vae_weights, &device, dtype)?;
    let unet = config.build_unet(model.unet_weights, &device, 4, false, dtype)?;

    // This would be used in image to image scenarios
    let t_start = 0;

    let mut images = vec![];
    images.reserve(args.images as usize);
    for idx in 0..args.images {
        let _span = info_span!("image", image_index = idx).entered();
        info!("Generating image");
        let timesteps = scheduler.timesteps();
        let latents = Tensor::randn(
            0f32,
            1f32,
            (bsize, 4, config.height / 8, config.width / 8),
            &device,
        )? * scheduler.init_noise_sigma();
        let mut latents = latents?.to_dtype(dtype)?;

        for (timestep_index, &timestep) in timesteps.iter().enumerate() {
            debug!("Image generation step {timestep_index}");
            if timestep_index < t_start {
                continue;
            }
            let latent_model_input = if use_guide_scale {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;
            let noise_pred =
                unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

            let noise_pred = if use_guide_scale {
                let noise_pred = noise_pred.chunk(2, 0)?;
                let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

                (noise_pred_uncond
                    + ((noise_pred_text - noise_pred_uncond)? * args.guidance_scale)?)?
            } else {
                noise_pred
            };

            latents = scheduler.step(&noise_pred, timestep, &latents)?;
        }

        images.extend(sd_to_bitmap(&vae, &latents, args.vae_scale, bsize)?)
    }

    Ok(images)
}

const PRIOR_GUIDANCE_SCALE: f64 = 4.0;
const RESOLUTION_MULTIPLE: f64 = 42.67;
const LATENT_DIM_SCALE: f64 = 10.67;
const PRIOR_CIN: usize = 16;
const DECODER_CIN: usize = 4;

fn ws_encode_prompt(
    prompt: &str,
    uncond_prompt: Option<&str>,
    tokenizer: impl AsRef<Path>,
    clip_weights: impl AsRef<Path>,
    clip_config: stable_diffusion::clip::Config,
    device: &Device,
) -> Result<Tensor, CandleError> {
    let tokenizer =
        Tokenizer::from_file(tokenizer).map_err(|e| CandleError::Tokenizer(e.to_string()))?;
    let pad_id = match &clip_config.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| CandleError::Tokenizer(e.to_string()))?
        .get_ids()
        .to_vec();
    let tokens_len = tokens.len();
    while tokens.len() < clip_config.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    let text_model =
        stable_diffusion::build_clip_transformer(&clip_config, clip_weights, device, DType::F32)?;
    let text_embeddings = text_model.forward_with_mask(&tokens, tokens_len - 1)?;
    match uncond_prompt {
        None => Ok(text_embeddings),
        Some(uncond_prompt) => {
            let mut uncond_tokens = tokenizer
                .encode(uncond_prompt, true)
                .map_err(|e| CandleError::Tokenizer(e.to_string()))?
                .get_ids()
                .to_vec();
            let uncond_tokens_len = uncond_tokens.len();
            while uncond_tokens.len() < clip_config.max_position_embeddings {
                uncond_tokens.push(pad_id)
            }
            let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;

            let uncond_embeddings =
                text_model.forward_with_mask(&uncond_tokens, uncond_tokens_len - 1)?;
            let text_embeddings = Tensor::cat(&[text_embeddings, uncond_embeddings], 0)?;
            Ok(text_embeddings)
        }
    }
}

struct Wuerstchen {
    tokenizer: std::path::PathBuf,
    prior_tokenizer: std::path::PathBuf,
    clip: std::path::PathBuf,
    prior_clip: std::path::PathBuf,
    decoder: std::path::PathBuf,
    vq_gan: std::path::PathBuf,
    prior: std::path::PathBuf,
}

#[allow(dead_code)]
fn ws_generate_image(
    paths: Wuerstchen,
    args: ImageGenerationArgs,
    device: Device,
) -> Result<Vec<Vec<u8>>, CandleError> {
    let _span = info_span!("ws_gen_image", images = args.images, steps = args.steps).entered();
    let height = args.height.unwrap_or(1024);
    let width = args.width.unwrap_or(1024);

    let prior_text_embeddings = ws_encode_prompt(
        &args.prompt,
        Some(&args.uncond_prompt),
        &paths.prior_tokenizer,
        &paths.prior_clip,
        stable_diffusion::clip::Config::wuerstchen_prior(),
        &device,
    )?;

    let text_embeddings = ws_encode_prompt(
        &args.prompt,
        None,
        &paths.tokenizer,
        &paths.clip,
        stable_diffusion::clip::Config::wuerstchen(),
        &device,
    )?;

    let b_size = 1;
    let image_embeddings = {
        // https://huggingface.co/warp-ai/wuerstchen-prior/blob/main/prior/config.json
        let latent_height = (height as f64 / RESOLUTION_MULTIPLE).ceil() as usize;
        let latent_width = (width as f64 / RESOLUTION_MULTIPLE).ceil() as usize;
        let mut latents = Tensor::randn(
            0f32,
            1f32,
            (b_size, PRIOR_CIN, latent_height, latent_width),
            &device,
        )?;

        let prior = {
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[&paths.prior],
                    DType::F32,
                    &device,
                )?
            };
            wuerstchen::prior::WPrior::new(PRIOR_CIN, 1536, 1280, 64, 32, 24, false, vb)?
        };
        let prior_scheduler = wuerstchen::ddpm::DDPMWScheduler::new(60, Default::default())?;
        let timesteps = prior_scheduler.timesteps();
        let timesteps = &timesteps[..timesteps.len() - 1];
        for (index, &t) in timesteps.iter().enumerate() {
            debug!("Prior de-noising step {index}");
            let latent_model_input = Tensor::cat(&[&latents, &latents], 0)?;
            let ratio = (Tensor::ones(2, DType::F32, &device)? * t)?;
            let noise_pred = prior.forward(&latent_model_input, &ratio, &prior_text_embeddings)?;
            let noise_pred = noise_pred.chunk(2, 0)?;
            let (noise_pred_text, noise_pred_uncond) = (&noise_pred[0], &noise_pred[1]);
            let noise_pred = (noise_pred_uncond
                + ((noise_pred_text - noise_pred_uncond)? * PRIOR_GUIDANCE_SCALE)?)?;
            latents = prior_scheduler.step(&noise_pred, t, &latents)?;
        }
        ((latents * 42.)? - 1.)?
    };

    let vqgan = {
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&paths.vq_gan], DType::F32, &device)?
        };
        wuerstchen::paella_vq::PaellaVQ::new(vb)?
    };

    // https://huggingface.co/warp-ai/wuerstchen/blob/main/decoder/config.json
    let decoder = {
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&paths.decoder], DType::F32, &device)?
        };
        wuerstchen::diffnext::WDiffNeXt::new(
            DECODER_CIN,
            DECODER_CIN,
            64,
            1024,
            1024,
            2,
            false,
            vb,
        )?
    };

    let mut res = vec![];
    res.reserve(args.images as usize);
    for idx in 0..args.images {
        let _span = info_span!("image", image_index = idx).entered();
        info!("Generating image");
        // https://huggingface.co/warp-ai/wuerstchen/blob/main/model_index.json
        let latent_height = (image_embeddings.dim(2)? as f64 * LATENT_DIM_SCALE) as usize;
        let latent_width = (image_embeddings.dim(3)? as f64 * LATENT_DIM_SCALE) as usize;

        let mut latents = Tensor::randn(
            0f32,
            1f32,
            (b_size, DECODER_CIN, latent_height, latent_width),
            &device,
        )?;

        let scheduler = wuerstchen::ddpm::DDPMWScheduler::new(12, Default::default())?;
        let timesteps = scheduler.timesteps();
        let timesteps = &timesteps[..timesteps.len() - 1];
        for (index, &t) in timesteps.iter().enumerate() {
            debug!("Image generation step {index}");
            let ratio = (Tensor::ones(1, DType::F32, &device)? * t)?;
            let noise_pred =
                decoder.forward(&latents, &ratio, &image_embeddings, Some(&text_embeddings))?;
            latents = scheduler.step(&noise_pred, t, &latents)?;
        }
        let image = vqgan.decode(&(&latents * 0.3764)?)?;
        let image = (image.clamp(0f32, 1f32)? * 255.)?
            .to_dtype(DType::U8)?
            .i(0)?;
        let (channel, height, width) = image.dims3()?;
        if channel != 3 {
            return Err(CandleError::BadDims {
                dims: channel,
                expected: 3,
            });
        }
        let img = image.permute((1, 2, 0))?.flatten_all()?;
        let pixels = img.to_vec1::<u8>()?;
        let buf = ImageBuffer::<Rgb<u8>, _>::from_vec(width as u32, height as u32, pixels)
            .ok_or(CandleError::BadOutput)?;
        let mut encoded = BufWriter::new(Cursor::new(Vec::new()));
        buf.write_to(&mut encoded, ImageFormat::Png)?;
        res.push(encoded.into_inner()?.into_inner());
    }
    Ok(res)
}

pub struct CandleImageGenerationEndpoint {}

#[async_trait::async_trait]
impl ImageGenerationEndpoint for CandleImageGenerationEndpoint {
    async fn generate_image(
        &self,
        model: ModelFiles,
        args: ImageGenerationArgs,
    ) -> Result<Vec<Vec<u8>>, ImageGenerationEndpointError> {
        let device = match SETTINGS.read().await.read().await.gpu_policy {
            DevicePolicy::AlwaysCpu { .. } => Device::Cpu,
            DevicePolicy::AlwaysDevice { .. } => {
                Device::Cuda(CudaDevice::new(0).map_err(|e| CandleError::Candle(e))?)
            }
            _ => {
                warn!("Unknown device policy, executing on CPU");
                Device::Cpu
            }
        };

        Ok(sd_generate_image(model, args, device)?)
    }
}

impl From<CandleError> for ImageGenerationEndpointError {
    fn from(value: CandleError) -> Self {
        match value {
            CandleError::PromptTooLong { .. } => {
                ImageGenerationEndpointError::Decoding(value.to_string())
            }
            CandleError::Clip2Unavailable => ImageGenerationEndpointError::Load(value.to_string()),
            CandleError::BadDims { .. } => {
                ImageGenerationEndpointError::Decoding(value.to_string())
            }
            CandleError::Candle(_) => ImageGenerationEndpointError::Generation(value.to_string()),
            CandleError::Tokenizer(_) => ImageGenerationEndpointError::Decoding(value.to_string()),
            CandleError::BadOutput => ImageGenerationEndpointError::Encoding(value.to_string()),
            CandleError::EncodeProcessFailed(_) => {
                ImageGenerationEndpointError::Encoding(value.to_string())
            }
            CandleError::EncodeWriteFailed(_) => {
                ImageGenerationEndpointError::Encoding(value.to_string())
            }
        }
    }
}
