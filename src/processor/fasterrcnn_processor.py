import torch
from PIL import Image
from typing import Any, Dict, Tuple
import pandas as pd
from torchvision import transforms

from src.processor.base_processor import BaseProcessor
from src.processor.processor_registry import ProcessorRegistry


# register processor to processor registry using the same name as model
@ProcessorRegistry.register("FasterRCNN_ConvNeXtV2")
class FasterRCNNProcessor(BaseProcessor):
    """
    Processor to process the image and bounding boxes data
    to be passed into the FasterRCNN model.

    Attribute "label_offset" is set as 1 since FasterRCNN reserves index 0 for background.
    """

    label_offset = 1

    def process(
        self,
        image: Image.Image,
        image_rows: pd.DataFrame,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Abstract method for the Processor class.

        Simple processing as torchvision's FasterRCNN model already has internal processing steps
        that include resizing and normalization of image and corresponding bounding boxes.

        Scale image values across all channels to [0,1] and extract bounding boxes coordinates from DataFrame.
        """

        # scale image value to [0,1], do not need to rescale image
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(image)

        # no need to normalize bounding box, keep original value
        boxes = torch.tensor(
            image_rows[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32
        )

        # only the bounding boxes coordinates in target, the labels will be
        # add later when the label onset is applied.
        target = {"boxes": boxes}

        return img_tensor, target
