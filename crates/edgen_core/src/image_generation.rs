use serde::Serialize;
use std::path::Path;
use thiserror::Error;

pub struct ImageGenerationArgs {
    pub prompt: String,
    pub uncond_prompt: String,
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub guidance_scale: f64,
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
        tokenizer: impl AsRef<Path> + Send + Sync,
        clip_weights: impl AsRef<Path> + Send + Sync,
        vae_weights: impl AsRef<Path> + Send + Sync,
        unet_weights: impl AsRef<Path> + Send + Sync,
        args: ImageGenerationArgs,
    ) -> Result<Vec<Vec<u8>>, ImageGenerationEndpointError>;
}
