import logging
from typing import Tuple, Optional
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger("pipeline")


def fix_image_orientation(image: Image.Image) -> Image.Image:
    """
    Correct the orientation of a PIL Image using its EXIF metadata.
    If EXIF metadata absent, return original image.

    Args:
        image (Image.Image): Image to correct orientation.

    Returns:
        Image.Image: Corrected image, or original image if EXIF absent.
    """
    try:
        return ImageOps.exif_transpose(image)
    except Exception:
        logger.info("Image has no EXIF data; skipping orientation fix.")
        return image


def get_image_dimensions(image: Image.Image) -> Tuple[int, int]:
    """
    Get width, height of image.

    Args:
        image (Image.Image): Input image.

    Returns:
        Tuple[int, int]: (width, height) in pixels.
    """
    width, height = image.size
    return width, height


def letterbox_image(
    image: Image.Image,
    target_shape: Tuple[int, int],
    fill_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, float, Tuple[int, int]]:
    """
    Resize an image to *target_shape* while preserving its aspect ratio and
    padding the remaining area with *fill_color* (letterboxing).

    Parameters
    ----------
    image : PIL.Image.Image
        Input image (any size).
    target_shape : Tuple[int, int]
        Desired output ``(width, height)`` in pixels.
    fill_color : Tuple[int, int, int]
        RGB fill colour for the padding regions.  Defaults to mid-grey
        ``(114, 114, 114)`` - the standard YOLO padding colour.

    Returns
    -------
    lb_image : PIL.Image.Image
        Letterboxed image of size *target_shape*.
    scale : float
        The scale factor applied to the original image.
    pad : Tuple[int, int]
        ``(pad_x, pad_y)`` -  the number of padding pixels added on
        *each side* of the shorter dimension.
    """
    src_w, src_h = image.size
    tgt_w, tgt_h = target_shape

    scale = min(tgt_w / src_w, tgt_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    resized = image.resize(
        (new_w, new_h), Image.BILINEAR
    )  # scale image to new shape using bilinear interpolation, better quality image

    lb_image = Image.new(
        "RGB", (tgt_w, tgt_h), fill_color
    )  # create new image with padding color
    pad_x = (tgt_w - new_w) // 2
    pad_y = (tgt_h - new_h) // 2
    lb_image.paste(resized, (pad_x, pad_y))  # paste image with the padding offset

    return lb_image, scale, (pad_x, pad_y)


###############
# Normalize bounding box forward and backward transform
def normalize_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
    scale: float,
    pad: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """
    Convert original image bounding box coordinates to normalized coordinates
    in the letterboxed image space.

    Parameters
    ----------
    xmin, ymin, xmax, ymax : float
        Bounding box coordinates in the original image (pixels).
    width, height : int
        Dimensions of the letterboxed (target) image.
    scale : float
        Scale factor returned by letterbox_image.
    pad : Tuple[int, int]
        (pad_x, pad_y) returned by letterbox_image.

    Returns
    -------
    Tuple[float, float, float, float]
        Normalized (xmin, ymin, xmax, ymax) in [0, 1] relative to the
        letterboxed image dimensions.
    """
    pad_x, pad_y = pad

    # Scale and shift into letterboxed pixel space, then normalize
    norm_xmin = (xmin * scale + pad_x) / width
    norm_ymin = (ymin * scale + pad_y) / height
    norm_xmax = (xmax * scale + pad_x) / width
    norm_ymax = (ymax * scale + pad_y) / height

    return norm_xmin, norm_ymin, norm_xmax, norm_ymax


def denormalize_bbox(
    norm_xmin: float,
    norm_ymin: float,
    norm_xmax: float,
    norm_ymax: float,
    width: int,
    height: int,
    scale: float,
    pad: Tuple[int, int],
    orig_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, int, int]:
    """
    Convert normalized bounding box coordinates from letterboxed image space
    back to the original image pixel coordinates.

    Parameters
    ----------
    norm_xmin, norm_ymin, norm_xmax, norm_ymax : float
        Normalized bounding box coordinates in [0, 1] relative to the
        letterboxed image dimensions.
    width, height : int
        Dimensions of the letterboxed (target) image.
    scale : float
        Scale factor returned by letterbox_image.
    pad : Tuple[int, int]
        (pad_x, pad_y) returned by letterbox_image.
    orig_size : Tuple[int, int], optional
        Original image ``(width, height)`` before letterboxing. When provided,
        coordinates are clipped to ``[0, orig_w]`` and ``[0, orig_h]``.

    Returns
    -------
    Tuple[int, int, int, int]
        Bounding box coordinates (xmin, ymin, xmax, ymax) in the original
        image pixel space, clipped and rounded to integers.
    """
    pad_x, pad_y = pad

    # Denormalize to letterboxed pixel space, then remove padding and unscale
    xmin = (norm_xmin * width - pad_x) / scale
    ymin = (norm_ymin * height - pad_y) / scale
    xmax = (norm_xmax * width - pad_x) / scale
    ymax = (norm_ymax * height - pad_y) / scale

    # Clip to original image bounds before rounding to avoid out-of-bounds
    # indices caused by floating-point drift at the edges
    if orig_size is not None:
        orig_w, orig_h = orig_size
        xmin = max(0, min(xmin, orig_w))
        ymin = max(0, min(ymin, orig_h))
        xmax = max(0, min(xmax, orig_w))
        ymax = max(0, min(ymax, orig_h))

    return round(xmin), round(ymin), round(xmax), round(ymax)


def train_test_split(
    df: pd.DataFrame, test_size: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the provided dataframe into train and test sets while preserving class
    distribution across the split using MultilabelStratifiedShuffleSplit.

    Since each Image_ID can contain multiple objects (rows), the split is
    performed at the Image_ID level. A multilabel binary matrix is built
    per image (one column per class_id) so that the stratification accounts
    for all classes present in each image simultaneously.

    Returns
    -------
    train_df : pd.DataFrame
        Training subset with reset index.
    test_df : pd.DataFrame
        Test subset with reset index.
    """

    # Image_ID is filename of image in dataframe
    image_ids = df["Image_ID"].unique()
    all_classes = sorted(df["class_id"].unique())

    # Binary matrix: rows = images, cols = class_ids
    # Entry [i, j] = 1 if image i contains at least one object of class j
    label_matrix = np.zeros((len(image_ids), len(all_classes)), dtype=int)
    image_id_to_idx = {img_id: i for i, img_id in enumerate(image_ids)}
    class_to_col = {cls: j for j, cls in enumerate(all_classes)}

    for _, row in df.iterrows():
        i = image_id_to_idx[row["Image_ID"]]
        j = class_to_col[row["class_id"]]
        label_matrix[i, j] = 1

    # --- Stratified split at Image_ID level ---
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,  # produce 1 random split
        test_size=test_size,
        random_state=seed,  # set the seed for reproducibility
    )

    # get the indices of the split
    train_indices, test_indices = next(splitter.split(image_ids, label_matrix))

    train_image_ids = set(image_ids[train_indices])
    test_image_ids = set(image_ids[test_indices])

    train_df = df[df["Image_ID"].isin(train_image_ids)].reset_index(drop=True)
    test_df = df[df["Image_ID"].isin(test_image_ids)].reset_index(drop=True)

    logger.info(
        f"Split complete — "
        f"train: {len(train_image_ids)} images / {len(train_df)} rows | "
        f"test: {len(test_image_ids)} images / {len(test_df)} rows"
    )

    return train_df, test_df
