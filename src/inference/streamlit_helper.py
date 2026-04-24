from pathlib import Path
from PIL import Image
import logging
import pandas as pd
import torch
from torch import Tensor
import base64
import io
import requests
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.utils.logging import setup_logging
from src.utils.data_processing import fix_image_orientation
from src.inference.fastapi_helper import annotate_object_on_image

setup_logging()
logger = logging.getLogger("inference")


def get_local_data(
    data_dir_path: Path,
    df_file_path: Path,
) -> tuple[list[Path], pd.DataFrame] | None:
    """
    Get the list of image file names from the input data directory path.
    Get the dataframe containing bounding boxes information of image files
    from the input dataframe file path.

    Return the image file names and dataframe if available else None.

    Args:
        data_dir_path: Path of the local image data directory.
        df_file_path: Path of the csv file with the bounding boxes information of image files.

    Returns:
        A tuple of:
            - List of image file names from the image data directory.
            - DataFrame containing bounding boxes information of image files.

        Or None if either information is not available.
    """

    if not data_dir_path.is_dir() or not df_file_path.is_file():
        return None
    try:
        df = pd.read_csv(df_file_path)
    except Exception:
        logger.info("Error reading data file provided.")
        return None

    # Get all filenames in directory
    filepaths = list(data_dir_path.iterdir())
    if not filepaths:
        return None

    filenames = [f.name for f in filepaths]

    return filenames, df


def get_pil_img_detection_from_df(
    input_df: pd.DataFrame,
    image_dir_path: Path,
    class_dict: dict[int, str],
) -> tuple[Image.Image, dict[str, Tensor]]:
    """
    Convert the dataframe containing bounding boxes information of one image,
    to the format required to use the function "annotate_object_on_image" to
    annotate bounding boxes on image which is a PIL image and a dictionary
    of the bounding boxes information.

    Args:
        input_df: Dataframe containing bounding boxes information of object detected on one image.
        image_dir_path: Path of the local image data directory.
        class_dict: Dictionary mapping between label index and class that is model dependant.

    Returns:
        A tuple of:
            - PIL image of the the image to annotate the bounding boxes information.
            - Dictionary of the bounding boxes information of the image in Tensor form,
                with keys "boxes" for the bounding boxes, "labels" for the class label,
                and "scores" for the confidence of the identified object.
    """
    # get the unique image name from the df, since it could have multiple rows
    img_fn = input_df["Image_ID"].unique().tolist()[0]

    # read image and correct the orientation
    img_path = image_dir_path / img_fn
    img = Image.open(img_path)
    img = fix_image_orientation(img)

    # create detection result to match the format required
    # for the function "annotate_object_on_image"
    detection_tensor = {}

    # convert the label - class dictionary to class - label dictionary
    label_map = {v: k for k, v in class_dict.items()}

    # use the "class" column to get the label index
    detection_tensor["labels"] = torch.tensor(
        input_df["class"].map(label_map).values, dtype=torch.long
    )

    # use the "confidence" column to get the confidence scores
    detection_tensor["scores"] = torch.tensor(input_df["confidence"].values)

    # use the "xmin, ymin, xmax, ymax" column to get the boxes information in xyxy format
    detection_tensor["boxes"] = torch.tensor(
        input_df[["xmin", "ymin", "xmax", "ymax"]].values, dtype=torch.float32
    )

    logger.info(f"For image: {img_fn}, detection tensor is {detection_tensor}")
    return img, detection_tensor


def annotate_ground_truth_on_image(
    filenames: list[str],
    image_dir_path: Path,
    ground_truth_df: pd.DataFrame,
    class_dict: dict[int, str],
) -> list[tuple[Image.Image, str]]:
    """
    Draw bounding boxes with bounding boxes information on each of the image file names provided,
    using the provided dataframe containing bounding boxes information on all images.

    Return the annotated images and corresponding file names.

    Args:
        filenames: List of image file names for which to annotate the bounding boxes.
        image_dir_path: Path of the local image data directory.
        ground_truth_df: Dataframe containing bounding boxes information of object detected
                        for all images in local image data directory.
        class_dict: Dictionary mapping between label index and class that is model dependant.

    Returns:
        A list of tuple, where each tuple has:
            - Annotated PIL image.
            - File name of original image.

    """
    # sort to be consistent
    sorted_filenames = sorted(filenames)
    results = []

    for fn in sorted_filenames:
        # filter using image name and extract the required columns
        df_fn = ground_truth_df[ground_truth_df["Image_ID"] == fn][
            ["Image_ID", "class", "confidence", "ymin", "xmin", "ymax", "xmax"]
        ]

        # convert the dataframe to the required input format
        pil_img, detection = get_pil_img_detection_from_df(
            df_fn, image_dir_path, class_dict
        )

        annotated_img = annotate_object_on_image(pil_img, detection, class_dict)
        results.append((annotated_img, fn))

    logger.info("Annotated all ground truths")
    return results


def analyze_local_images(
    inference_url: str,
    filenames: list[str],
) -> list[tuple[Image.Image, str]] | str:
    """
    Send a list of filenames to the FastAPI endpoint to get object detection results from it.

    Args:
        inference_url: The FastAPI URL endpoint that will apply object detection model on the input images.
        filenames: List of image file names from local data directory for object detection to be done on.

    Returns:
        A list of tuple, where each tuple has:
            - PIL image with bounding boxes of object detected annotated.
            - File name of original image.
        Or error message from inference.
    """
    # sort it for consist ordering
    sorted_filenames = sorted(filenames)

    # send images via POST request to FastAPI endpoint
    resp = requests.post(inference_url, data={"filenames": sorted_filenames})

    # return error string
    if resp.status_code == 500:
        error_detail = resp.json().get("detail", "Unknown inference error")
        return error_detail

    return [
        (Image.open(io.BytesIO(base64.b64decode(item["data"]))), item["filename"])
        for item in resp.json()
    ]


def analyze_upload_images(
    inference_url: str,
    file_objects: list[UploadedFile],
) -> list[tuple[Image.Image, str]] | str:
    """
    Convert the list of UploadedFile obtained from Streamlit to a list of UploadFile,
    and send them to the FastAPI endpoint to get object detection results from it.

    Args:
        inference_url: The FastAPI URL endpoint that will apply object detection model on the input images.
        file_objects: List of UploadedFile from Streamlit file uploader.

    Returns:
        A list of tuple, where each tuple has:
            - PIL image with bounding boxes of object detected annotated.
            - File name of original image.
        Or error message from inference.
    """

    # convert UploadedFile to required format to send multiple files to FastAPI endpoint
    # accepts List[UploadFile] under the argument "files"
    # Expected format is tuple (filename, image bytes, mimetype)
    files = [("files", (f.name, f.getvalue(), f.type)) for f in file_objects]

    # send image via POST request
    resp = requests.post(inference_url, files=files)

    # return error string
    if resp.status_code == 500:
        error_detail = resp.json().get("detail", "Unknown inference error")
        return error_detail

    return [
        (Image.open(io.BytesIO(base64.b64decode(item["data"]))), item["filename"])
        for item in resp.json()
    ]
