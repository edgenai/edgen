use serde::Serialize;
use std::path::PathBuf;
use thiserror::Error;

pub struct ImageGenerationArgs {
    pub prompt: String,
    pub uncond_prompt: String,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub steps: usize,
    pub images: u32,
    pub seed: Option<u64>,
    pub guidance_scale: f64,
    pub vae_scale: f64,
}

pub struct ModelFiles {
    pub tokenizer: PathBuf,
    pub clip_weights: PathBuf,
    pub clip2_weights: Option<PathBuf>,
    pub vae_weights: PathBuf,
    pub unet_weights: PathBuf,
}

#[derive(Serialize, Error, Debug)]
pub enum ImageGenerationEndpointError {
    #[error("Could not load model: {0}")]
    Load(String),
    #[error("Failed to tokenize prompts: {0}")]
    Decoding(String),
    #[error("Failed to generate image: {0}")]
    Generation(String),
    #[error("Could not convert the output tensor into an encoded image")]
    Encoding(String),
}

#[async_trait::async_trait]
pub trait ImageGenerationEndpoint {
    async fn generate_image(
        &self,
        model: ModelFiles,
        args: ImageGenerationArgs,
    ) -> Result<Vec<Vec<u8>>, ImageGenerationEndpointError>;
}
