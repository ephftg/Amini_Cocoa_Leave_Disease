from dataclasses import dataclass, asdict
from typing import Any
import math
import logging
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch import Tensor

from src.utils.data_processing import fix_image_orientation
from src.utils.logging import setup_logging
from src.processor.processor_registry import ProcessorRegistry

setup_logging()
logger = logging.getLogger("pipeline")


class DetectionDataset(Dataset):
    """
    Customized Dataset that will provide the Dataset to Dataloader depending on model used.

    For each item in Dataset, apply a model dependent processor to process the image,
    bounding box coordinates and class labels to the expected input format for the model.
    """

    def __init__(self, df: pd.DataFrame, image_dir: Path, model_name: str) -> None:
        """
        Create the Dataset from the input DataFrame and model name.

        Set the processor to use to process the data to correct input format for model.

        Create "label_map" that accounts for the model's default label offset.

        Args:
            df: DataFrame containing the bounding boxes coordinates and class labels
                of all objects for each image.
            image_dir: Path where the images are stored locally.
            model_name: Name of model which corresponds to names in the model/processor
                        registry that will be using this dataset.
        """
        self.df = df
        self.image_dir = image_dir
        self.model_name = model_name

        # set the processor using the model name
        processor_cls = ProcessorRegistry.get(model_name)
        self.processor = processor_cls()

        # 'Image_ID' is the column with image name
        # sort it to be consistent across runs
        self.image_names = sorted(df["Image_ID"].unique().tolist())

        # label to integer mapping which is model dependent as there is label offset
        # "class" is the column with the class classification
        self.label_map = {
            lbl: i + self.processor.label_offset
            for i, lbl in enumerate(sorted(df["class"].unique()))
        }

    def __len__(self) -> int:
        """Returns total number of images in dataset."""
        return len(self.image_names)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Any]]:
        """
        Return each item in the Dataset using the index

        First read the image from the "image_dir" and fix the image's orientation
        so that image is read correctly.

        Use the model dependent processor to process the image and dataframe containing
        bounding box coordinates of all objects detected in image so that
        they will be in the correct input format for the model.

        Use the "label_map" which account for label offset to assign the class labels of objects.

        Args:
            idx: Index of image

        Returns:
            A tuple of :
                - Image in tensor format.
                - Dictionary with keys "boxes" for the bounding box coordinates
                  and "labels" for the class label of all objects detected in the image.
        """
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
    """Dataclass that groups all parameters to be logged to MLflow."""

    learning_rate: float
    weight_decay: float
    batch_size: int

    def as_dict(self) -> dict[str, Any]:
        """
        Convert dataclass to dictionary which is required format
        to log multiple parameters to Mlflow using log_params().

        Returns:
            Dictionary form of dataclass with each element being the
            name and value of each parameter.
        """
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class EpochMetrics:
    """
    Dataclass that groups all the metric (loss and evaluation score)
    to be logged to MLflow for each epoch.

    Exclude the train evaluation since is equivalent to passing the data
    for second which is compuationally expensive for limited resources.

    Two mean average precision scores
        val_map: mAP@0.5:0.95, standard and more robust score for object detection
        val_map_50: mAP@0.5, more lenient score, rough object detection acceptable
    """

    epoch: int
    train_loss: float
    val_loss: float
    val_map: float  # mAP@0.5:0.95
    val_map_50: float  # mAP@0.5

    def as_dict(self) -> dict[str, Any]:
        """
        Convert dataclass to dictionary which is required format
        to log multiple metrics to Mlflow using log_metrics().

        Returns:
            Dictionary form of dataclass with each element being the
            name and value of each metric.
        """
        return {k: v for k, v in asdict(self).items() if v is not None}


class EarlyStopping:
    """
    Class to stop the training when validation loss
    has not improved by a certain threshold for several consecutive epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
    ) -> None:
        """
        Create the early stopping object with attributes.

        Args:
            patience: Number of consecutive epochs for which there is no improvement
                      and early stopping is applied.
            min_delta: Threshold to be considered as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = math.inf  # start with worst loss
        self.counter = 0  #  track how many consecutive epochs with no improvement
        self.stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Callable instance to check if the early stopping criteria is met for current epoch.

        Args:
            val_loss: Validation loss for the current epoch.

        Returns:
            Boolean value on whether to apply early stopping for this epoch.

        """
        # if validation loss has improve above a certain threshold, reset the count to zero,
        # else increase the count
        # if count exceed the patience value, apply early stop.
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
