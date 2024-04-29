use crate::model_descriptor::{
    ModelDescriptor, ModelDescriptorError, ModelPaths, Quantization, StableDiffusionFiles,
};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use dashmap::DashMap;
use edgen_core::image_generation::{
    ImageGenerationArgs, ImageGenerationEndpoint, ImageGenerationEndpointError, ModelFiles,
};
use edgen_rt_image_generation_candle::CandleImageGenerationEndpoint;
use either::Either;
use serde_derive::{Deserialize, Serialize};
use std::borrow::Cow;
use thiserror::Error;
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Model<'a> {
    unet_weights: Cow<'a, str>,
    vae_weights: Cow<'a, str>,
    clip_weights: Cow<'a, str>,
    /// Beware that not all models support clip2.
    clip2_weights: Option<Cow<'a, str>>,
    tokenizer: Cow<'a, str>,
}

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
    pub model: Either<Cow<'a, str>, Model<'a>>,

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

    /// The random number generator seed to use for the generation.
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
    /// Required if `model` is not a pre-made descriptor name.
    ///
    /// This value should probably not be set, if `model` is a pre-made descriptor name.
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
    #[error(transparent)]
    Model(#[from] ModelDescriptorError),
    /// Some error has occurred inside the endpoint.
    #[error(transparent)]
    Endpoint(#[from] ImageGenerationEndpointError),
    /// This error should be unreachable.
    #[error("Something went wrong")]
    Unreachable,
    /// Some parameter was missing from the request.
    #[error("A parameter was missing from the request: {0}")]
    MissingParam(String),
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
/// to the peer.
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
    let quantization;
    let descriptor = match req.model {
        Either::Left(template) => {
            quantization = Quantization::F16;
            crate::model_descriptor::get(template.as_ref())?
                .value()
                .clone() // Not ideal to clone, but otherwise the code complexity will greatly increase
        }
        Either::Right(custom) => {
            if req.vae_scale.is_none() {
                return Err(ImageGenerationError::MissingParam(
                    "VAE scale must be provided when manually specifying model files".to_string(),
                ));
            }
            quantization = Quantization::Default;
            let files = DashMap::new();
            files.insert(
                quantization,
                StableDiffusionFiles {
                    tokenizer: custom.tokenizer.to_string(),
                    clip_weights: custom.clip_weights.to_string(),
                    clip2_weights: custom.clip2_weights.map(|c| c.to_string()),
                    vae_weights: custom.vae_weights.to_string(),
                    unet_weights: custom.unet_weights.to_string(),
                },
            );
            ModelDescriptor::StableDiffusion {
                files,
                steps: 30,
                vae_scale: req.vae_scale.unwrap(),
            }
        }
    };
    let model_files;
    let default_steps;
    let default_vae_scale;
    if let ModelDescriptor::StableDiffusion {
        steps, vae_scale, ..
    } = descriptor
    {
        if let ModelPaths::StableDiffusion {
            unet_weights,
            vae_weights,
            clip_weights,
            clip2_weights,
            tokenizer,
        } = descriptor.preload_files(quantization).await?
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
                steps: req.steps.unwrap_or(default_steps),
                images: req.images.unwrap_or(1),
                seed: req.seed,
                guidance_scale: req.guidance_scale.unwrap_or(7.5),
                vae_scale: req.vae_scale.unwrap_or(default_vae_scale),
            },
        )
        .await?;

    Ok(Json(ImageGenerationResponse { images }))
}
