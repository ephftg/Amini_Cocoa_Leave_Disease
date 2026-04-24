from torch import Tensor
from PIL import Image

from src.utils.data_processing import fix_image_orientation
from src.inference.inference_processor.inference_processor_registry import (
    InferenceProcessorRegistry,
)
from src.inference.inference_processor.base_inference_processor import (
    BaseInferenceProcessor,
)


# register the processor
@InferenceProcessorRegistry.register("simple")
class SimpleInferenceProcessor(BaseInferenceProcessor):
    """
    Simple inference processor where the input image just needs to be
    oriented to the correct orientation and scaled between [0,1] before converting to Tensor.

    No extra processing required for the output, it returns back the model output.

    This is when the model output is already in the original scale of the input image.
    """

    def pre_process(self, images: list[Image.Image]) -> list[Tensor]:
        """
        For each image, fix the orientation and scale image values to [0,1].
        """
        tensor_images = [self.to_tensor(fix_image_orientation(img)) for img in images]
        return tensor_images

    def post_process(
        self,
        model_output: list[dict[str, Tensor]],
    ) -> list[dict[str, Tensor]]:
        """
        Return unmodified output.
        """
        return model_output
