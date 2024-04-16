use crate::model::{Model, ModelError, ModelKind};
use crate::types::Endpoint;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use edgen_core::image_generation::{
    ImageGenerationArgs, ImageGenerationEndpoint, ImageGenerationEndpointError,
};
use edgen_core::settings;
use edgen_rt_image_generation_candle::CandleImageGenerationEndpoint;
use serde_derive::{Deserialize, Serialize};
use std::borrow::Cow;
use std::path::PathBuf;
use thiserror::Error;
use utoipa::ToSchema;

/// A request to generate images for the provided context.
///
/// An `axum` handler, [`image_generation`][image_generation], is provided to handle this request.
///
/// [image_generation]: fn.image_generation.html
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct CreateImageGenerationRequest<'a> {
    /// The prompt to be used to generate the image.
    pub prompt: Cow<'a, str>,

    /// The model to use for generating completions.
    pub model: Cow<'a, str>,

    /// The width of the generated image.
    pub width: u32,

    /// The height of the generated image.
    pub height: u32,

    /// The optional unconditional prompt.
    pub uncond_prompt: Option<Cow<'a, str>>,

    /// The number of steps to be used in the diffusion process.
    pub steps: Option<u32>,

    /// The number of samples to generate.
    pub samples: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ImageGenerationResponse {
    pub images: Vec<Vec<u8>>,
}

/// An error condition raised by the image generation API.
#[derive(Serialize, Error, ToSchema, Debug)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "error")]
pub enum ImageGenerationError {
    /// The provided model could not be loaded.
    #[error("failed to load model: {0}")]
    Model(#[from] ModelError),
    /// Some error has occured inside the endpoint.
    #[error("endpoint error: {0}")]
    Endpoint(#[from] ImageGenerationEndpointError),
}

impl IntoResponse for ImageGenerationError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(self)).into_response()
    }
}

#[utoipa::path(
post,
path = "/image/generations",
request_body = CreateImageGenerationRequest,
responses(
(status = 200, description = "OK", body = ImageGenerationResponse),
(status = 500, description = "unexpected internal server error", body = ImageGenerationError)
),
)]
pub async fn generate_image(
    Json(req): Json<CreateImageGenerationRequest<'_>>,
) -> Result<impl IntoResponse, ImageGenerationError> {
    let mut unet = Model::new(
        ModelKind::ImageDiffusion,
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "stabilityai/stable-diffusion-2-1",
        &PathBuf::from(settings::image_generation_dir().await),
    );
    unet.preload(Endpoint::ImageGeneration).await?;

    let mut vae = Model::new(
        ModelKind::ImageDiffusion,
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "stabilityai/stable-diffusion-2-1",
        &PathBuf::from(settings::image_generation_dir().await),
    );
    vae.preload(Endpoint::ImageGeneration).await?;

    let mut tokenizer = Model::new(
        ModelKind::ImageDiffusion,
        "tokenizer.json",
        "openai/clip-vit-base-patch32",
        &PathBuf::from(settings::image_generation_dir().await),
    );
    tokenizer.preload(Endpoint::ImageGeneration).await?;

    let mut clip = Model::new(
        ModelKind::ImageDiffusion,
        "text_encoder/model.fp16.safetensors",
        "stabilityai/stable-diffusion-2-1",
        &PathBuf::from(settings::image_generation_dir().await),
    );
    clip.preload(Endpoint::ImageGeneration).await?;

    let endpoint = CandleImageGenerationEndpoint {};
    let images = endpoint
        .generate_image(
            tokenizer.file_path()?,
            clip.file_path()?,
            vae.file_path()?,
            unet.file_path()?,
            ImageGenerationArgs {
                prompt: req.prompt.to_string(),
                uncond_prompt: req.uncond_prompt.unwrap_or(Cow::from("")).to_string(),
                width: req.width,
                height: req.height,
                samples: 1,
                guidance_scale: 7.5,
            },
        )
        .await?;

    Ok(Json(ImageGenerationResponse { images }))
}
