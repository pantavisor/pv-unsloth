# Multi-arch container for Unsloth under Pantavisor
# Supports NVIDIA Tegra (Jetson Orin) and NVIDIA DGX (Blackwell) / CUDA

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies and pvcontrol client
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    pciutils \
    && rm -rf /var/lib/apt/lists/*

# Install Unsloth, FastAPI, and serving dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    transformers \
    accelerate \
    bitsandbytes \
    triton

# Install Unsloth
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" || \
    pip install --no-cache-dir unsloth

COPY server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Storage volume for model weights
VOLUME ["/models"]

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
