from typing import Any
import mlflow
import torch
import torch.nn as nn
from torch import Tensor
import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import gc
import logging
import base64
import io
from fastapi import HTTPException, UploadFile
import math

from src.utils.logging import setup_logging
from src.utils.data_processing import fix_image_orientation
from src.inference.inference_processor.inference_processor_registry import (
    InferenceProcessorRegistry,
)
from src.inference.inference_processor.base_inference_processor import (
    BaseInferenceProcessor,
)

setup_logging()
logger = logging.getLogger("inference")


###############
# Loader helpers
def load_config(config_path: Path) -> dict[str, Any]:
    """
    Read the config yaml file as a dictionary and return it.

    Args:
        config_path: Path of the inference config yaml.

    Returns:
        Dictionary format of the inference configuration.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_model_from_mlflow(config: dict[str, Any]) -> nn.Module:
    """
    Load model from Mlflow and set it to inference mode for use.
    Omit artifact download since not required for persistent.

    Args:
        config: Dictionary of inference config file.

    Returns:
        Trained inference model.

    Raises:
        Error when model fails to load.
    """
    mlflow_uri = config["mlflow_uri"]
    mlflow_artifact_uri = config["mlflow_artifact_uri"]

    try:
        mlflow.set_tracking_uri(mlflow_uri)
        # use pytorch since all model is nn.Module
        model = mlflow.pytorch.load_model(mlflow_artifact_uri)

        return model
    except Exception as e:
        logger.error(
            f"Error loading model from mlflow uri: {mlflow_uri} with artifact uri: {mlflow_artifact_uri}: {str(e)}"
        )
        raise


def load_inference_processor(config: dict[str, Any]) -> BaseInferenceProcessor:
    """
    Load the inference processsor that can process the input/output of inference model.

    Args:
        config: Dictionary of inference config file.

    Returns:
        Inference processor for the inference model.

    Raises:
        KeyError if inference processor name is not registered in registry.
    """
    # import all inference processor
    InferenceProcessorRegistry.load_plugins(config["inference_processor_plugins"])

    inference_processor_name = config["inference_processor_name"]

    try:
        # get the inference processor
        # raise KeyError if name not found
        infer_proc = InferenceProcessorRegistry.get(inference_processor_name)
        return infer_proc()

    except KeyError:
        logger.error(f"Error loading inference processor: {inference_processor_name}")
        raise


###############
# Object detection helpers for fastapi
async def read_upload_files(
    files: list[UploadFile],
) -> tuple[list[Image.Image], list[str]]:
    """
    Get PIL image and image file names from UploadFile objects
    received from POST request from Streamlit interface and
    return them as a tuple of lists.

    Args:
        files: List of UploadFile to be processed.

    Returns:
        A tuple of two list:
            - PIL images.
            - Filenames of corresponding PIL images.

    Raises:
        Bad request error if any file cannot be decoded as an image.
    """
    pil_images = []
    filenames = []

    for file in files:
        try:
            # async read UploadFile
            raw = await file.read()
            filename = file.filename
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            pil_images.append(img)
            filenames.append(filename)
        except Exception as e:
            logger.error(f"Failed to read uploaded files: {str(e)}")

            raise HTTPException(
                status_code=400,
                detail=f"Could not decode '{filename}' as an image: {e}",
            )

    return pil_images, filenames


def convert_pil_to_base64_string(image: Image.Image) -> str:
    """
    Convert PIL image to jpeg Base64 encoded string since
    PIL image cannot be transported in JSON.
    Jpeg has smaller size than png.
    Set quality to 85 to reduce size while maintaining quality.

    Args:
        image: PIL image to encode.

    Returns:
        Base64 encoded string of input PIL image.
    """

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)  # reduce size while maintain quality
    buffer.seek(0)

    # convert to base64 bytes and then to string to transfer in JSON
    return base64.b64encode(buffer.read()).decode("utf-8")


def map_label_to_color(label_idx: int) -> str:
    """
    Map an index to a color.
    Use modulo operator on label index in case label range exceed provided color range.

    Args:
        label_idx: Output label of the object detected.

    Returns:
        String representing color to use for that label.

    Raise:
        KeyError if label is negative.
    """
    # color available in PIL, which is extensible
    # Choice of color should be distinct from one another and from colors related to trees for clarity
    COLORS = ["red", "blue", "purple", "orange", "black", "pink"]

    if label_idx < 0:
        raise ValueError(f"Label index must be non-negative, got {label_idx}")

    # in case label exceed range of color provided
    return COLORS[label_idx % len(COLORS)]


def map_label_to_class_name(class_dict: dict, label_idx: int) -> str:
    """
    Map output label index to class name using the provided label-class name dictionary.

    Args:
        class_dict: Dictionary mapping between label index and class.
                    Provided in inference config as it is model dependant, since background object
                    is automatically assigned different label.
                    Class names should be sorted in ascending alphabetical order except for background.
        label_idx: Output label of the object detected.

    Returns:
        Class name of the label following the class_dict.

    Raise:
        KeyError if output label is not in keys of class_dict.
    """

    if label_idx not in class_dict:
        raise KeyError(
            f"Label index '{label_idx}' not found. Available keys: {list(class_dict.keys())}"
        )

    return class_dict[label_idx]


def annotate_object_on_image(
    pil_img: Image.Image,
    detection: dict[str, Tensor],
    class_dict: dict[int, str],
) -> Image.Image:
    """
    Use the object detection results of the image to draw bounding boxes
    and label each of them with the class label and confidence score on the input image.

    Fix the orientation of the image first before doing the annotation as the detection
    results are based on the correct orientation of the image.

    Apply a score threshold of 0.5 to omit cluttering due to low confidence boxes.
    Scale text font size according to diagonal of image to ensure visibility.
    Box label with class label and confidence score at top left of box.

    Args:
        pil_image: Image to annnotate the bounding boxes on.
        detection: Dictionary of object detection results of the image in Tensor form,
                with keys "boxes" for the bounding boxes, "labels" for the class label,
                and "scores" for the confidence of the identified object.
        class_dict: Dictionary mapping between label index and class that is model dependant.

    Returns:
        Annotated PIL image with the bounding boxes and detection results.
    """
    oriented_pil_img = fix_image_orientation(pil_img)
    draw = ImageDraw.Draw(oriented_pil_img)

    # move to cpu to reduce GPU storage
    boxes = detection["boxes"].cpu()  # xyxy format
    labels = detection["labels"].cpu()
    scores = detection["scores"].cpu()

    border_pixel_thickeness = 10

    # filter for boxes with confidence score above 0.5
    score_threshold = 0.5

    # text font size scales with image while accounting for aspect ratio
    img_width, img_height = oriented_pil_img.size
    diagonal = math.sqrt(img_width**2 + img_height**2)
    font_size = max(12, int(diagonal * 0.02))
    font = ImageFont.load_default(size=font_size)

    for box, label, score in zip(boxes, labels, scores):
        # only draw boxes above score threshold
        if score > score_threshold:
            score = round(float(score), 2)
            x1, y1, x2, y2 = box
            int_label = int(label)
            class_name = map_label_to_class_name(class_dict, int_label)
            color = map_label_to_color(int_label)
            class_text = f"Class: {class_name}"
            score_text = f"Score: {score}"

            # get height of first line
            bbox = font.getbbox(class_text)
            class_text_height = bbox[3] - bbox[1] + 5

            # x1, y1 top left corner, x2,y2 is bottom right corner
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=color,
                width=border_pixel_thickeness,
            )

            # label at top left corner, inside box so that label is always on image
            draw.text(
                (x1 + border_pixel_thickeness, y1 + border_pixel_thickeness),
                class_text,
                fill=color,
                font=font,
            )

            draw.text(
                (
                    x1 + border_pixel_thickeness,
                    y1 + border_pixel_thickeness + class_text_height,
                ),
                score_text,
                fill=color,
                font=font,
            )

            logger.info(
                f"Added box: {box} with label: {int_label} and score: {score} on PIL image"
            )

    return oriented_pil_img


def object_detection_images(
    pil_images: list[Image.Image],
    model: nn.Module,
    processor: BaseInferenceProcessor,
    class_dict: dict[int, str],
    device: torch.device,
    batch_size: int = 8,
) -> list[Image.Image]:
    """
    Apply object detection on provided images using provided model
    and return annotated images with bounding boxes, class labels and confidence score.

    Apply batch processing and clear up after each batch to prevent OOM/runtime error.

    Depending on model, corresponding processor to pre-process input image
    to correct format before passing to model and post-process the bounding boxes
    from output to original scale so as to be drawn on original image.

    Args:
        pil_images: List of PIL images to do object detection for.
        model: Object detection model obtained from training.
        processor: Corresponfing processor for the object detection model that is used, since
                    different model require different processing.
        class_dict: Dictionary mapping between label index and class that is model dependant.
        device: CPU/GPU to do inference on.
        batch_size: Number of PIL image to process at one go, default is 8.

    Returns:
        list of annotated images with bounding boxes, class labels and confidence score.

    Raise:
        OOM/Runtime error during inference and other exceptions.
    """

    annotated_images = []

    # do inference in batches to prevent OOM/Runtime error if image is large or many
    for batch_start in range(0, len(pil_images), batch_size):
        batch = pil_images[batch_start : batch_start + batch_size]
        image_tensors = None
        detection_outputs = None

        try:
            # pre process input and put all to device
            image_tensors = processor.pre_process(batch)
            image_tensors = [t.to(device) for t in image_tensors]

            with torch.no_grad():
                detection_outputs = model(image_tensors)

            # post process output
            detection_outputs = processor.post_process(detection_outputs)

            # annotate image after inference
            for pil_img, detection in zip(batch, detection_outputs):
                annotated_pil_img = annotate_object_on_image(
                    pil_img, detection, class_dict
                )
                annotated_images.append(annotated_pil_img)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.error(
                f"OOM/Runtime error in doing object detection on batched images of batch size={batch_size}: {e}",
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(f"{e}", exc_info=True)
            raise

        # remove large tensor to clear memory to prevent OOM/runtime error
        finally:
            del batch, image_tensors, detection_outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    logger.info(
        "Completed object detection on batched images and returned annotated images."
    )
    return annotated_images
