use serde::Serialize;
use thiserror::Error;

pub struct ImageGenerationArgs {}

#[derive(Serialize, Error, Debug)]
pub enum ImageGenerationEndpointError {}

#[async_trait::async_trait]
pub trait ImageGenerationEndpoint {
    async fn generate_image(
        &self,
        args: ImageGenerationArgs,
    ) -> Result<(), ImageGenerationEndpointError>;
}
