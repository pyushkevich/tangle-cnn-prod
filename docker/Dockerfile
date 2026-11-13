# Runtime image for tangle-cnn jobs on GKE GPU nodes.
# Build from the repository root (with submodules checked out):
#   docker buildx build --platform linux/amd64 -f docker/Dockerfile -t pyushkevich/tangle-cnn-prod:<version> .
# CUDA 12.6 runs on the NVIDIA drivers GKE installs by default.
FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Tools used by the job scripts: vips (tiffsave) and the Google Cloud CLI (gcloud, gsutil)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg libvips-tools \
 && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends google-cloud-cli \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The base image installs PyTorch into the system Python (Ubuntu 24.04), so allow pip to
# install there too; there is no OS Python to protect inside this container
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install dependencies first, so that code changes don't invalidate this layer.
# PyTorch and torchvision come from the base image.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the package itself
COPY pyproject.toml README.md LICENSE ./
COPY tangle_cnn ./tangle_cnn
RUN pip install --no-cache-dir --no-deps .

# Job scripts. Existing GKE job specs call them relative to /app (e.g.
# "scripts/wildcat_analyze_wsi.sh"), so /app must stay the working directory;
# they are also on PATH so they can be called by name.
COPY --chmod=755 scripts ./scripts
ENV PATH=/app/scripts:$PATH

# Run as a non-root user. Some scripts write ./data into the working directory,
# so /app must be writable by that user.
RUN useradd --create-home --shell /bin/bash user \
 && chown -R user:user /app
USER user
