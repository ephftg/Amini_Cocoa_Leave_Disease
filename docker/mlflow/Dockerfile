FROM python:3.12-slim-bookworm
# pinned Debian version

# remove cache for smaller image
RUN pip install --no-cache-dir mlflow==2.22.4

RUN mkdir -p /mlflow/db /mlflow/artifacts

WORKDIR /mlflow

EXPOSE 5000
