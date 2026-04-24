import base64
import logging
from contextlib import asynccontextmanager
import torch
from pathlib import Path
from fastapi.responses import JSONResponse
import asyncio
from pydantic import BaseModel, field_validator
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image

from src.utils.logging import setup_logging
from src.inference.fastapi_helper import (
    load_config,
    load_inference_processor,
    load_model_from_mlflow,
    object_detection_images,
    read_upload_files,
    convert_pil_to_base64_string,
)

setup_logging()
logger = logging.getLogger("inference")


class AnnotatedImage(BaseModel):
    """
    Annotated image to be returned from the inference endpoint.

    Attributes:
        filename: Filename of the original image before anotation.
        data: Base64-encoded string of the annotated image.
    """

    filename: str
    data: str

    @field_validator("data")
    @classmethod
    def verify_base64(cls, v: str) -> str:
        """
        Verify that the data is for the annotated image is encoded in base64 string.
        PIL image cannot be in JSON response.
        """
        try:
            base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("Image data must be a valid Base64 string")
        return v

    @field_validator("filename")
    @classmethod
    def verify_non_empty_filename(cls, v: str) -> str:
        """
        Verify that annnotated image has filename.
        """
        if not v.strip():
            raise ValueError("Filename must not be empty")
        return v.strip()


# need async that is compatible with FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the startup of the FastAPI application.

    On startup, it loads the inference configuration file,
    the inference processor, and the inference mode.

    It store them in app.state together with the device (cpu/cuda)
    available, which allows for reuse across all requests.

    It sets an asyncio lock to ensure only one inference runs at a time,
    preventing concurrent requests from running the inference model simultaneously.
    This prevent conflicts from concurrent request.

    No actions to perform after shutdown.

    Args:
        app: The FastAPI application instance.

    Raise:
        Exception if any errow when loading config, model or processor.

    """
    # set defaults
    app.state.model = None
    app.state.processor = None
    app.state.config = None
    app.state.device = None
    app.state.inference_lock = asyncio.Lock()  # set the lock

    try:
        config_path = Path("config/inference_config.yaml")
        config = load_config(config_path)

        processor = load_inference_processor(config)
        model = load_model_from_mlflow(config)
        # use cuda if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # set to inference mode
        model.eval()

        app.state.model = model
        app.state.processor = processor
        app.state.config = config
        app.state.device = device

        logger.info(f"Model loaded successfully to {device}")

    except Exception as e:
        logger.error(f"Failed to load model or processor: {e}", exc_info=True)
        raise

    # no action after shutdown
    yield


app = FastAPI(title="Object Detection Inference API", lifespan=lifespan)


@app.get("/health")
def health():
    """
    Check if model and processor are available, so inference can start.

    Returns:
        JSONResponse 200 if both are available, else 503.
    """
    model_ready = app.state.model is not None
    processor_ready = app.state.processor is not None

    if model_ready and processor_ready:
        return JSONResponse(status_code=200, content={"status": "ok"})
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "Unavailable",
                "details": "Model or processor not available.",
            },
        )


@app.post("/infer", response_model=list[AnnotatedImage], status_code=201)
async def run_inference(
    files: list[UploadFile] | None = File(None),
    filenames: list[str] | None = Form(None),
):
    """
    Endpoint which run inference on the images it received using the
    inference model and output the annotated images with the objects detected.

    Run inference with asyncio lock to prevent concurrent inference execution.
    Run inference in another thread to allow handling of other request while doing inference.

    Accept two kinds of input to handle the case where user upload images to endpoint
    or when user select images from local data directory.

    Args:
        files: List of UploadFile which are image files from Streamlit.
        filenames: List of filenames relative to the local data directory.

    Returns:
        List of annotated image of type AnnotatedImage, containing the annotated image
        encoded in Base64 string and the original image filename.

    Raise:
        Fail request validation when neither files nor filenames provided.
        OOM/Runtime error if it occurs when doing inference.
    """

    # depending on input, get the PIL images and filenames
    if files:
        logger.info("Inference input is uploaded files.")
        pil_images, image_names = await read_upload_files(files)
    elif filenames:
        logger.info("Inference input is filenames of local data.")
        img_dir = Path(app.state.config["data_dir"])
        pil_images = [Image.open(img_dir / img_fn) for img_fn in filenames]
        image_names = filenames
    else:
        raise HTTPException(status_code=422, detail="Provide either files or filenames")

    # apply the lock to ensure only one inference job is running at one time
    async with app.state.inference_lock:
        try:
            # get current async loop
            event_loop = asyncio.get_running_loop()

            # run object detection in another thread so inference does not block
            # other requests
            annotated_images = await event_loop.run_in_executor(
                None,
                lambda: object_detection_images(
                    pil_images=pil_images,
                    model=app.state.model,
                    processor=app.state.processor,
                    class_dict=app.state.config["class_map"],
                    device=app.state.device,
                    batch_size=app.state.config["inference_batch_size"],
                ),
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.error(f"OOM/Runtime error during inference: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed due to runtime/OOM error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Unexpected error during inference: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Unexpected inference error: {str(e)}"
            )

    # convert annotated image to Base64 string to be returned in JSON
    images = [
        AnnotatedImage(
            filename=name,
            data=convert_pil_to_base64_string(img),
        )
        for name, img in zip(image_names, annotated_images)
    ]
    # FastAPI automatically serialize pydantic class
    return images
