FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04
# compatible with pyproject.toml, need runtime for training
# contains CUDA libraries

# copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# install python using uv
COPY .python-version .
RUN UV_PYTHON_DOWNLOADS=auto uv python install

# Set working directory
WORKDIR /app

# create environment, cache to speed up
COPY pyproject.toml uv.lock .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy project files
COPY main.py .
COPY src/ ./src/
COPY config/ ./config/

RUN mkdir -p /app/logs /app/data
