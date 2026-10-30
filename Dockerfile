# Pantavisor AI Box gateway for pv-unsloth: OpenAI-compatible API with a
# minimal chat page, stdlib only (the Nano's CUDA 10.2 cannot run PyTorch
# 2.x; the GPU is probed through the CUDA driver API).
# Deployed as a Pantavisor app container: the device's nvidia driver only
# loads nvgpu, so the image carries libcuda.so.1 and the Tegra libraries it
# needs, from the L4T release the BSP's nvgpu comes from (R32.7.6, t210).

FROM --platform=$BUILDPLATFORM debian:bookworm-slim AS l4t
ARG L4T=https://repo.download.nvidia.com/jetson/t210/pool/main/n
ARG L4T_VERSION=32.7.6-20241104234540
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl binutils xz-utils zstd \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /l4t
RUN set -e; \
    for p in "nvidia-l4t-core 61ada8308bd38fe6fe1793b658c83c3ef3dcc057a0d9e080dc0336e61be683c2" \
             "nvidia-l4t-cuda 0042650294496d632f5d7695a23632411ec3c1d3b2fdd664181338031bb45f86" \
             "nvidia-l4t-3d-core 44df4b465417916ac1bdb1c801892c7af948114cadf5a3a1eafb780984600a94"; do \
      set -- $p; deb="$1_${L4T_VERSION}_arm64.deb"; \
      curl -fsSL -o "$deb" "$L4T/$1/$deb"; echo "$2  $deb" | sha256sum -c -; \
      mkdir -p x && (cd x && ar x "../$deb" && tar -xf data.tar.*); rm -f x/*.tar.* x/debian-binary; \
    done; \
    mkdir -p /out/usr/lib/aarch64-linux-gnu/tegra; \
    cd x/usr/lib/aarch64-linux-gnu/tegra; \
    cp -a libcuda.so.1.1 libnvrm_gpu.so libnvrm.so libnvrm_graphics.so libnvos.so \
          libnvidia-fatbinaryloader.so.440.18 libnvidia-ptxjitcompiler.so.440.18 \
          /out/usr/lib/aarch64-linux-gnu/tegra/; \
    cd /out/usr/lib/aarch64-linux-gnu/tegra; \
    ln -s libcuda.so.1.1 libcuda.so.1; ln -s libcuda.so.1 libcuda.so; \
    ln -s libnvidia-ptxjitcompiler.so.440.18 libnvidia-ptxjitcompiler.so.1; \
    mkdir -p /out/etc/ld.so.conf.d; \
    echo /usr/lib/aarch64-linux-gnu/tegra > /out/etc/ld.so.conf.d/nvidia-tegra.conf

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=8001

COPY --from=l4t /out/ /
RUN ldconfig

WORKDIR /app
COPY server.py /app/server.py

# Model files land here; /models is served by /v1/models.
RUN mkdir -p /models
VOLUME ["/models"]

EXPOSE 8001

ENTRYPOINT ["python3", "/app/server.py"]
