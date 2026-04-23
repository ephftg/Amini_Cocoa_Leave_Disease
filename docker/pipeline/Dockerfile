FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04
# compatible with pyproject.toml, need runtime for model training
# contains CUDA libraries

# copy uv binary from official image, faster than install
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# install python using uv
COPY .python-version .
RUN UV_PYTHON_DOWNLOADS=auto uv python install

# create environment, cache to speed up but not written in
COPY pyproject.toml uv.lock .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

RUN mkdir -p /app/logs /app/data

# Copy project files
COPY main.py .
COPY src/utils ./src/utils
COPY src/model ./src/model
COPY src/processor ./src/processor
COPY src/pipeline ./src/pipeline
COPY config/train_config.yaml ./config/train_config.yaml
