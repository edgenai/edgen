use edgen_core::image_generation::{
    ImageGenerationArgs, ImageGenerationEndpoint, ImageGenerationEndpointError,
};

pub struct CandleImageGenerationEndpoint {}

#[async_trait::async_trait]
impl ImageGenerationEndpoint for CandleImageGenerationEndpoint {
    async fn generate_image(
        &self,
        args: ImageGenerationArgs,
    ) -> Result<(), ImageGenerationEndpointError> {
        todo!()
    }
}
