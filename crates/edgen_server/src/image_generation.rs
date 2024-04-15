use axum::response::IntoResponse;
use axum::Json;
use serde_derive::{Deserialize, Serialize};
use std::borrow::Cow;
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

/// An error condition raised by the image generation API.
#[derive(Serialize, Error, ToSchema, Debug)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "error")]
pub enum ImageGenerationError {
    /// The provided model could not be found on the local system.
    #[error("no such model: {model_name}")]
    NoSuchModel {
        /// The name of the model.
        model_name: String,
    },

    /// The provided model could not be found on the local system.
    #[error("unknown model kind: {model_name}, {reason}")]
    UnknownModelKind {
        /// The name of the model.
        model_name: String,

        /// A human-readable error message.
        reason: Cow<'static, str>,
    },

    /// The provided model name contains prohibited characters.
    #[error("model {model_name} could not be fetched from the system: {reason}")]
    ProhibitedName {
        /// The name of the model provided.
        model_name: String,

        /// A human-readable error message.
        reason: Cow<'static, str>,
    },
}

pub async fn generate_image(
    Json(req): Json<CreateImageGenerationRequest<'_>>,
) -> Result<impl IntoResponse, ImageGenerationError> {
}
