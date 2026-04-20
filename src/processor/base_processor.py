import abc
from typing import Any, Dict, Tuple
import pandas as pd
from PIL import Image
import torch


class BaseProcessor(abc.ABC):
    """
    A class that process the images and bounding boxes
    into the correct format for which the model expects.

    Contain the "label_offset" attribute which is the offset to the label index.
    Different models automatically assign a value to the background class.
    The offset ensures the labels of the class object are correct.
    """

    label_offset: int = 0

    @abc.abstractmethod
    def process(
        self,
        image: Image.Image,
        image_rows: pd.DataFrame,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process an input image and the corresponding bounding boxes data and
        output the processed image in tensor form and processed bounding boxes
        data in dictionary form.

        Args:
            image: PIL image to be processed.
            image_rows: DataFrame consisting of bounding boxes coordinates of objects on the image.

        Returns:
            A tuple:
                - Processed image in Tensor form.
                - Dictionary with a key value "boxes" for bounding boxes coordinates of all objects in image.

        """
