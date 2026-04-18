import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from typing import Any, Dict, Tuple

from src.utils.data_processing import letterbox_image, normalize_bbox
from src.processor.base_processor import BaseProcessor
from src.processor.processor_registry import ProcessorRegistry


@ProcessorRegistry.register("Detr_DinoV2")
class DINOv2Processor(BaseProcessor):
    """
    Process images and bounding box before input into the Detr_DinoV2 model.
    """

    # DINOv2 uses 0-indexed classes; last logit index is background
    label_offset = 0

    def process(
        self,
        image: Image.Image,
        image_rows: pd.DataFrame,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # letterboxed height and width for DinoV2 image
        lb_h = 224
        lb_w = 224

        lb_image, scale, pad = letterbox_image(image, (lb_w, lb_h))

        # ImageNet normalization for RGB, require scale to [0,1] first
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        img_tensor = transform(lb_image)

        # normalize bounding box to letterbox scale
        norm_boxes = []

        for _, row in image_rows.iterrows():
            norm_xmin, norm_ymin, norm_xmax, norm_ymax = normalize_bbox(
                xmin=row["xmin"],
                ymin=row["ymin"],
                xmax=row["xmax"],
                ymax=row["ymax"],
                width=lb_w,  # letterboxed width
                height=lb_h,  # letterboxed height
                scale=scale,
                pad=pad,
            )
            norm_boxes.append([norm_xmin, norm_ymin, norm_xmax, norm_ymax])

        boxes = torch.tensor(norm_boxes, dtype=torch.float32)

        target = {"boxes": boxes}

        return img_tensor, target
