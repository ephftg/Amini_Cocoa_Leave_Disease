import torch
from PIL import Image
from typing import Any, Dict, Tuple
import pandas as pd
from torchvision import transforms

from src.processor.base_processor import BaseProcessor
from src.processor.processor_registry import ProcessorRegistry


# register to processor registry
# same name as model
@ProcessorRegistry.register("FasterRCNN_ConvNeXtV2")
class FasterRCNNProcessor(BaseProcessor):
    """
    Processor for FasterRCNN model.
    It has a label_offset of 1 since FasterRCNN reserve index 0 for background.
    """

    label_offset = 1

    def process(
        self,
        image: Image.Image,
        image_rows: pd.DataFrame,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Simple processing before input into FasterRCNN as FasterRCNN already has internal processing steps
        that include resizing and normalization of image and corresponding bounding boxes.

        Scale image values across all channels to [0,1] and extract bounding boxes coordinates.
        """

        # scale image value to [0,1], do not need to rescale image
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(image)

        # no need to normalize bounding box, keep original value
        boxes = torch.tensor(
            image_rows[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32
        )
        target = {"boxes": boxes}

        return img_tensor, target
