# Pantavisor AI Box gateway for pv-unsloth: OpenAI-compatible API with a
# minimal chat page, stdlib only (the Nano's CUDA 10.2 cannot run PyTorch
# 2.x; the GPU is probed through the CUDA driver API).
# Deployed as a Pantavisor app container: the device's nvidia driver
# injection provides libcuda.so.1 (see pvpkg.json PV_DRIVERS_REQUIRED).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app
COPY server.py /app/server.py

# Model files land here; /models is served by /v1/models.
RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE 8000

ENTRYPOINT ["python3", "/app/server.py"]
