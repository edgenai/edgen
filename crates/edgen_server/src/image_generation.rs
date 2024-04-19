use crate::audio::ChatCompletionError;
use crate::model_descriptor::{ModelDescriptor, ModelDescriptorError, ModelPaths, Quantization};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use edgen_core::image_generation::{
    ImageGenerationArgs, ImageGenerationEndpoint, ImageGenerationEndpointError, ModelFiles,
};
use edgen_rt_image_generation_candle::CandleImageGenerationEndpoint;
use serde_derive::{Deserialize, Serialize};
use std::borrow::Cow;
use thiserror::Error;
use utoipa::ToSchema;

/// A request to generate images for the provided context.
/// This request is not at all conformant with OpenAI's API, as that one is very bare-bones, lacking
/// in many parameters that we need.
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
    pub width: Option<usize>,

    /// The height of the generated image.
    pub height: Option<usize>,

    /// The optional unconditional prompt.
    pub uncond_prompt: Option<Cow<'a, str>>,

    /// The number of steps to be used in the diffusion process.
    pub steps: Option<usize>,

    /// The number of images to generate.
    ///
    /// Default: 1
    pub images: Option<u32>,

    /// The random number generator seed to used for the generation.
    ///
    /// By default, a random seed is used.
    pub seed: Option<u64>,

    /// The guidance scale to use for generation, that is, how much should the model follow the
    /// prompt.
    ///
    /// Values below 1 disable guidance. (the prompt is ignored)
    pub guidance_scale: Option<f64>,

    /// The Variational Auto-Encoder scale to use for generation.
    ///
    /// This value should probably not be set.
    pub vae_scale: Option<f64>,
}

/// This request is not at all conformant with OpenAI's API, as that one returns a URL to the
/// generated image, and that is not possible for Edgen.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ImageGenerationResponse {
    /// A vector containing the byte data of the generated images.
    pub images: Vec<Vec<u8>>,
}

/// An error condition raised by the image generation API.
#[derive(Serialize, Error, ToSchema, Debug)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "error")]
pub enum ImageGenerationError {
    /// The provided model could not be loaded.
    #[error("failed to load model: {0}")]
    Model(#[from] ModelDescriptorError),
    /// Some error has occured inside the endpoint.
    #[error("endpoint error: {0}")]
    Endpoint(#[from] ImageGenerationEndpointError),
    /// This error should be unreachable.
    #[error("Something went wrong")]
    Unreachable,
}

impl IntoResponse for ImageGenerationError {
    fn into_response(self) -> Response {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(self)).into_response()
    }
}

/// POST `/v1/image/generations`: generate image for the provided parameters
///
/// The API of this endpoint is not at all conformant with OpenAI's API, as that one is very
/// bare-bones, lacking  in many parameters that we need, and also returns an URL, which Edgen
/// cannot do.
///
/// On failure, may raise a `500 Internal Server Error` with a JSON-encoded [`ImageGenerationError`]
/// to the peer..
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
    let descriptor = crate::model_descriptor::get(req.model.as_ref())?;
    let model_files;
    let default_steps;
    let default_vae_scale;
    if let ModelDescriptor::StableDiffusion {
        steps, vae_scale, ..
    } = descriptor.value()
    {
        if let ModelPaths::StableDiffusion {
            unet_weights,
            vae_weights,
            clip_weights,
            clip2_weights,
            tokenizer,
        } = descriptor.preload_files(Quantization::F16).await?
        {
            model_files = ModelFiles {
                tokenizer,
                clip_weights,
                clip2_weights,
                vae_weights,
                unet_weights,
            };
        } else {
            return Err(ImageGenerationError::Unreachable);
        }
        default_steps = steps;
        default_vae_scale = vae_scale;
    } else {
        return Err(ImageGenerationError::Unreachable);
    };

    let endpoint = CandleImageGenerationEndpoint {};
    let images = endpoint
        .generate_image(
            model_files,
            ImageGenerationArgs {
                prompt: req.prompt.to_string(),
                uncond_prompt: req.uncond_prompt.unwrap_or(Cow::from("")).to_string(),
                width: req.width,
                height: req.height,
                steps: req.steps.unwrap_or(*default_steps),
                images: req.images.unwrap_or(1),
                seed: req.seed,
                guidance_scale: req.guidance_scale.unwrap_or(7.5),
                vae_scale: req.vae_scale.unwrap_or(*default_vae_scale),
            },
        )
        .await?;

    Ok(Json(ImageGenerationResponse { images }))
}
