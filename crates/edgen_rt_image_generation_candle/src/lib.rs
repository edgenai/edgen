use std::io::BufWriter;
use std::io::{Cursor, IntoInnerError};
use std::path::Path;

use candle_core::backend::BackendDevice;
use candle_core::{CudaDevice, DType, Device, IndexOp, Module, Tensor, D};
use candle_transformers::models::stable_diffusion;
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use image::{ImageBuffer, ImageError, ImageFormat, Rgb};
use thiserror::Error;
use tokenizers::Tokenizer;

use edgen_core::image_generation::{
    ImageGenerationArgs, ImageGenerationEndpoint, ImageGenerationEndpointError, ModelFiles,
};
use edgen_core::settings::{DevicePolicy, SETTINGS};
use rand::random;
use tracing::{debug, info, info_span, warn};

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

fn text_embeddings(
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

fn to_bitmap(
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

fn generate_image(
    model: ModelFiles,
    args: ImageGenerationArgs,
    device_policy: &DevicePolicy,
) -> Result<Vec<Vec<u8>>, CandleError> {
    let _span = info_span!("gen_image", images = args.images, steps = args.steps).entered();
    let config = stable_diffusion::StableDiffusionConfig::v2_1(None, args.height, args.width);
    let scheduler = config.build_scheduler(args.steps)?;
    let device = match device_policy {
        DevicePolicy::AlwaysCpu { .. } => Device::Cpu,
        DevicePolicy::AlwaysDevice { .. } => Device::Cuda(CudaDevice::new(0)?),
        _ => {
            warn!("Unknown device policy, executing on CPU");
            Device::Cpu
        }
    };
    let use_guide_scale = args.guidance_scale > 1.0;
    let dtype = DType::F16;
    let bsize = 1;

    device.set_seed(args.seed.unwrap_or(random::<u64>()))?;

    // let which = match sd_version {
    //     StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => vec![true, false],
    //     _ => vec![true],
    // };

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
            text_embeddings(
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

        images.extend(to_bitmap(&vae, &latents, args.vae_scale, bsize)?)
    }

    Ok(images)
}

pub struct CandleImageGenerationEndpoint {}

#[async_trait::async_trait]
impl ImageGenerationEndpoint for CandleImageGenerationEndpoint {
    async fn generate_image(
        &self,
        model: ModelFiles,
        args: ImageGenerationArgs,
    ) -> Result<Vec<Vec<u8>>, ImageGenerationEndpointError> {
        let policy = SETTINGS.read().await.read().await.gpu_policy.clone();
        Ok(generate_image(model, args, &policy)?)
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
