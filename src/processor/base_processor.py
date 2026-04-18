import abc
from typing import Any, Dict, Tuple
import pandas as pd
from PIL import Image
import torch


class BaseProcessor(abc.ABC):
    """
    Owns all model-specific dataset logic.

    Attributes
    ----------
    label_offset : int
        0 for models where the last index is background (DINOv2),
        1 for models where index 0 is reserved for background (FasterRCNN).
    """

    label_offset: int = 0  # subclasses override as a class variable

    @abc.abstractmethod
    def process(
        self,
        image: Image.Image,
        image_rows: pd.DataFrame,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Returm image in tensor form and targets dictionary with bounding box"""
