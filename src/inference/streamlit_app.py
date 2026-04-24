import streamlit as st
from pathlib import Path
import logging

from src.utils.logging import setup_logging
from src.inference.fastapi_helper import load_config
from src.inference.streamlit_helper import (
    get_local_data,
    annotate_ground_truth_on_image,
    analyze_local_images,
    analyze_upload_images,
)

setup_logging()
logger = logging.getLogger("inference")

# get config
config = load_config(Path("config/inference_config.yaml"))
data_dir = Path(config["data_dir"])
df_file = Path(config["df_file"])
class_map = config["class_map"]
inference_fastapi_url = config["inference_url"]

st.header("Cocoa Leaves Contamination Detection")

# Two possible modes
mode1 = "Files from local directory"
mode2 = "Upload files"

# select mode
st.subheader("Select Mode")
mode = st.radio(
    "Select Mode", [mode1, mode2], horizontal=True, label_visibility="collapsed"
)

if mode == mode1:
    st.subheader(mode1)

    local_data = get_local_data(data_dir, df_file)

    if local_data is None:
        st.info("Sorry, this mode is not available.")
    else:
        image_files, bbox_df = local_data

        # allow multiselect files
        selected_filenames = st.multiselect(
            "Select one or more images:",
            options=image_files,
            default=[],
            placeholder="Choose images...",
        )

        # two columns
        col1, col2 = st.columns(2)

        if selected_filenames and st.button("Analyze"):
            with st.spinner("Running model ..."):
                ground_truth_images = annotate_ground_truth_on_image(
                    selected_filenames, data_dir, bbox_df, class_map
                )
                model_predicted_images = analyze_local_images(
                    inference_fastapi_url, selected_filenames
                )

                with col1:
                    st.subheader("Ground Truth")
                    for img, caption in ground_truth_images:
                        st.image(img, caption=caption)

                with col2:
                    # if POST request fails
                    st.subheader("Model Prediction")
                    if isinstance(model_predicted_images, str):
                        st.error(model_predicted_images)

                    else:
                        for img, caption in model_predicted_images:
                            st.image(img, caption=caption)

elif mode == mode2:
    st.subheader(mode2)

    # get list of UploadedFile
    uploaded = st.file_uploader(
        "Upload images to analyze", accept_multiple_files=True, type=["jpg", "png"]
    )

    if uploaded and st.button("Analyze"):
        with st.spinner("Running model ..."):
            results = analyze_upload_images(inference_fastapi_url, uploaded)

        # if POST request fails
        if isinstance(results, str):
            st.error(results)
        else:
            for image, filename in results:
                st.image(image, caption=filename)
