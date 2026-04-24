import abc
from torch import Tensor
from PIL import Image
from torchvision import transforms


class BaseInferenceProcessor(abc.ABC):
    """
    Abstract base class for model-specific inference processors.
    Each model type should implement its own pre and post processing logic.

    Include the toTensor() transform since most image processing model requires it.
    """

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    @abc.abstractmethod
    def pre_process(self, images: list[Image.Image]) -> list[Tensor]:
        """
        Implement the necessary pre-processing depending on the model used
        to convert a list of PIL Images into a list of tensor ready for
        model inference.

        Args:
            images: List of PIL Images.

        Returns:
            A list of Tensor for each image.
        """

    @abc.abstractmethod
    def post_process(
        self,
        model_output: list[dict[str, Tensor]],
    ) -> list[dict[str, Tensor]]:
        """
        Implement the necessary post-processing depending on the model used
        to convert the model output to the original image resolution.

        Args:
            model_output: The output of object detection model, which is a list of dictionary
                        "boxes" is the bounding box, "labels" is the class, 'scores" is the probability.

        Returns:
            List of dictionary of the same information as the model_output but with the
            "boxes" transformed to the original image resolution. Each dictionary is
            for each image.

        """
