use crate::model::{Model, ModelError, ModelKind};
use crate::types::Endpoint;
use dashmap::DashMap;
use edgen_core::settings;
use once_cell::sync::Lazy;
use serde_derive::Serialize;
use std::path::PathBuf;
use thiserror::Error;

static MODELS: Lazy<DashMap<String, ModelDescriptor>> = Lazy::new(Default::default);

#[derive(Debug, Error, Serialize)]
pub enum ModelDescriptorError {
    #[error("The specified quantization level is not available for the model")]
    QuantizationUnavailable,
    #[error(transparent)]
    Preload(#[from] ModelError),
    #[error("The specified model was not found")]
    NotFound,
}

/// The descriptor of an artificial intelligence model, containing every bit of data required to
/// execute the model.
pub enum ModelDescriptor {
    /// A stable diffusion model.
    StableDiffusion {
        /// The files that make up the model, indexed by quantization.
        files: DashMap<Quantization, StableDiffusionFiles>,

        /// The default number of diffusion steps for this model.
        steps: usize,

        /// The default Variational Auto-Encoder scale for this model.
        vae_scale: f64,
    },
}

pub struct StableDiffusionFiles {
    unet_weights_repo: String,
    unet_weights_file: String,
    vae_weights_repo: String,
    vae_weights_file: String,
    clip_weights_repo: String,
    clip_weights_file: String,
    clip2_weights_repo: Option<String>,
    clip2_weights_file: Option<String>,
    tokenizer_repo: String,
    tokenizer_file: String,
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub enum Quantization {
    Default,
    F16,
}

pub enum ModelPaths {
    StableDiffusion {
        unet_weights: PathBuf,
        vae_weights: PathBuf,
        clip_weights: PathBuf,
        clip2_weights: Option<PathBuf>,
        tokenizer: PathBuf,
    },
}

impl ModelDescriptor {
    pub async fn preload_files(
        &self,
        quantization: Quantization,
    ) -> Result<ModelPaths, ModelDescriptorError> {
        let res = match self {
            ModelDescriptor::StableDiffusion { files, .. } => {
                let files = files.get(&quantization);
                if files.is_none() {
                    return Err(ModelDescriptorError::QuantizationUnavailable);
                }
                let files = files.unwrap();
                let dir = PathBuf::from(settings::image_generation_dir().await);
                let unet = {
                    let mut unet = Model::new(
                        ModelKind::ImageDiffusion,
                        &files.unet_weights_file,
                        &files.unet_weights_repo,
                        &dir,
                    );
                    unet.preload(Endpoint::ImageGeneration).await?;
                    unet.file_path()?
                };
                let vae = {
                    let mut vae = Model::new(
                        ModelKind::ImageDiffusion,
                        &files.vae_weights_file,
                        &files.vae_weights_repo,
                        &dir,
                    );
                    vae.preload(Endpoint::ImageGeneration).await?;
                    vae.file_path()?
                };
                let clip = {
                    let mut clip = Model::new(
                        ModelKind::ImageDiffusion,
                        &files.clip_weights_file,
                        &files.clip_weights_repo,
                        &dir,
                    );
                    clip.preload(Endpoint::ImageGeneration).await?;
                    clip.file_path()?
                };
                let clip2 = if files.clip2_weights_file.is_some() {
                    let mut clip2 = Model::new(
                        ModelKind::ImageDiffusion,
                        files.clip2_weights_file.as_ref().unwrap(),
                        files.clip2_weights_repo.as_ref().unwrap(),
                        &dir,
                    );
                    clip2.preload(Endpoint::ImageGeneration).await?;
                    Some(clip2.file_path()?)
                } else {
                    None
                };
                let tokenizer = {
                    let mut tokenizer = Model::new(
                        ModelKind::ImageDiffusion,
                        &files.tokenizer_file,
                        &files.tokenizer_repo,
                        &dir,
                    );
                    tokenizer.preload(Endpoint::ImageGeneration).await?;
                    tokenizer.file_path()?
                };

                ModelPaths::StableDiffusion {
                    unet_weights: unet,
                    vae_weights: vae,
                    clip_weights: clip,
                    clip2_weights: clip2,
                    tokenizer,
                }
            }
        };
        Ok(res)
    }
}

// This should pull its data for a config file or database, but for now this is fine
pub fn init() {
    let model_files = DashMap::new();
    model_files.insert(
        Quantization::Default,
        StableDiffusionFiles {
            tokenizer_repo: "openai/clip-vit-base-patch32".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            clip_weights_repo: "stabilityai/stable-diffusion-2-1".to_string(),
            clip_weights_file: "text_encoder/model.safetensors".to_string(),
            clip2_weights_repo: None,
            clip2_weights_file: None,
            vae_weights_repo: "stabilityai/stable-diffusion-2-1".to_string(),
            vae_weights_file: "vae/diffusion_pytorch_model.safetensors".to_string(),
            unet_weights_repo: "stabilityai/stable-diffusion-2-1".to_string(),
            unet_weights_file: "unet/diffusion_pytorch_model.safetensors".to_string(),
        },
    );
    model_files.insert(
        Quantization::F16,
        StableDiffusionFiles {
            tokenizer_repo: "openai/clip-vit-base-patch32".to_string(),
            tokenizer_file: "tokenizer.json".to_string(),
            clip_weights_repo: "stabilityai/stable-diffusion-2-1".to_string(),
            clip_weights_file: "text_encoder/model.fp16.safetensors".to_string(),
            clip2_weights_repo: None,
            clip2_weights_file: None,
            vae_weights_repo: "stabilityai/stable-diffusion-2-1".to_string(),
            vae_weights_file: "vae/diffusion_pytorch_model.fp16.safetensors".to_string(),
            unet_weights_repo: "stabilityai/stable-diffusion-2-1".to_string(),
            unet_weights_file: "unet/diffusion_pytorch_model.fp16.safetensors".to_string(),
        },
    );
    let model = ModelDescriptor::StableDiffusion {
        files: model_files,
        steps: 30,
        vae_scale: 0.18215,
    };
    MODELS.insert("stable-diffusion-2-1".to_string(), model);
}

pub fn get(
    model: &str,
) -> Result<dashmap::mapref::one::Ref<String, ModelDescriptor>, ModelDescriptorError> {
    MODELS.get(model).ok_or(ModelDescriptorError::NotFound)
}
