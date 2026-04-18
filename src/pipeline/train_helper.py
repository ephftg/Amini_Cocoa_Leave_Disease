from dataclasses import dataclass, asdict
from typing import Any, Dict
import math
import logging
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
import torch

from src.utils.data_processing import fix_image_orientation
from src.utils.logging import setup_logging
from src.processor.processor_registry import ProcessorRegistry

setup_logging()
logger = logging.getLogger("pipeline")


class DetectionDataset(Dataset):
    """
    Custom Dataset that will provide the Dataset to Dataloader depending on model used.
    A model dependent processor will process the image to the correct format to be loaded to the model.
    """

    def __init__(self, df: pd.DataFrame, image_dir: Path, model_name: str) -> None:

        self.df = df  # bounding box df
        self.image_dir = image_dir  # path of original image
        self.model_name = model_name  # the model name

        # set the processor
        processor_cls = ProcessorRegistry.get(model_name)
        self.processor = processor_cls()

        # 'Image_ID' is the column with image name
        self.image_names = sorted(df["Image_ID"].unique().tolist())

        # label to integr mapping which is model dependent
        self.label_map = {
            lbl: i + self.processor.label_offset
            for i, lbl in enumerate(sorted(df["class"].unique()))
        }
        # label offset for FasterRCNN is 1, for DinoV2 is 0

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]

        # filter dataframe for the bounding box of the image
        img_rows = self.df[self.df["Image_ID"] == img_name]

        # fix orientation of image
        img = Image.open(self.image_dir / img_name).convert("RGB")
        img = fix_image_orientation(img)

        # use processor to process the image and bounding boxes, target is dictionary
        img_tensor, target = self.processor.process(img, img_rows)

        # add the labels using the label map
        labels = torch.tensor(
            [self.label_map[img_cls] for img_cls in img_rows["class"]],
            dtype=torch.int64,
        )
        target["labels"] = labels

        # target is dictionary with "boxes" and "labels"
        return img_tensor, target


@dataclass
class Params:
    """Standardised container for parameters logged to MLflow."""

    learning_rate: float
    weight_decay: float
    batch_size: int

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert dataclass to dictionary with key value pair to upload to mlflow.
        """
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class EpochMetrics:
    """Standardised container for per-epoch metrics logged to MLflow."""

    epoch: int
    train_loss: float
    val_loss: float
    val_map: float  # mAP@0.5:0.95 only for validation set
    val_map_50: float  # mAP@0.5

    # convert to dictionary key value pair to upload to mlflow
    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class EarlyStopping:
    """Stop training when validation loss has not improved for *patience* epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0001) -> None:
        self.patience = patience
        self.min_delta = min_delta  # threshold to be considered as improvement
        self.best_loss = math.inf
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            logger.info("Counter reset to zero for early stopping.")
        else:
            self.counter += 1
            logger.info(f"Early stopping count is {self.counter}")
            if self.counter >= self.patience:
                self.stop = True
                logger.info("Early stopping patience reached, stop training.")

        return self.stop
