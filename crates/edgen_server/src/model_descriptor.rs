use crate::model::{Model, ModelError, ModelKind};
use crate::openai_shim::{parse_model_param, ParseError};
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
    #[error(transparent)]
    Parse(#[from] ParseError),
}

/// The descriptor of an artificial intelligence model, containing every bit of data required to
/// execute the model.
#[derive(Clone)]
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

#[derive(Clone)]
pub struct StableDiffusionFiles {
    pub unet_weights: String,
    pub vae_weights: String,
    pub clip_weights: String,
    pub clip2_weights: Option<String>,
    pub tokenizer: String,
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
    async fn get_file(&self, file_link: &str) -> Result<PathBuf, ModelDescriptorError> {
        let (dir, kind) = match self {
            ModelDescriptor::StableDiffusion { .. } => (
                PathBuf::from(settings::image_generation_dir().await),
                ModelKind::StableDiffusion,
            ),
        };

        let path = PathBuf::from(file_link);
        if path.is_file() {
            return Ok(path);
        }

        let path = dir.join(file_link);
        if path.is_file() {
            return Ok(path);
        }

        let (owner, repo, name) = parse_model_param(file_link)?;
        let mut file = Model::new(kind, &name, &format!("{owner}/{repo}"), &dir);
        file.preload(Endpoint::ImageGeneration).await?;
        Ok(file.file_path()?)
    }

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
                let unet = self.get_file(&files.unet_weights).await?;
                let vae = self.get_file(&files.vae_weights).await?;
                let clip = self.get_file(&files.clip_weights).await?;
                let clip2 = if let Some(clip2) = &files.clip2_weights {
                    Some(self.get_file(&clip2).await?)
                } else {
                    None
                };
                let tokenizer = self.get_file(&files.tokenizer).await?;

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
            tokenizer: "openai/clip-vit-base-patch32/tokenizer.json".to_string(),
            clip_weights: "stabilityai/stable-diffusion-2-1/text_encoder/model.safetensors"
                .to_string(),
            clip2_weights: None,
            vae_weights: "stabilityai/stable-diffusion-2-1/vae/diffusion_pytorch_model.safetensors"
                .to_string(),
            unet_weights:
                "stabilityai/stable-diffusion-2-1/unet/diffusion_pytorch_model.safetensors"
                    .to_string(),
        },
    );
    model_files.insert(
        Quantization::F16,
        StableDiffusionFiles {
            tokenizer: "openai/clip-vit-base-patch32/tokenizer.json".to_string(),
            clip_weights: "stabilityai/stable-diffusion-2-1/text_encoder/model.fp16.safetensors"
                .to_string(),
            clip2_weights: None,
            vae_weights:
                "stabilityai/stable-diffusion-2-1/vae/diffusion_pytorch_model.fp16.safetensors"
                    .to_string(),
            unet_weights:
                "stabilityai/stable-diffusion-2-1/unet/diffusion_pytorch_model.fp16.safetensors"
                    .to_string(),
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
