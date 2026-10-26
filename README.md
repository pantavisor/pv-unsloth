# pv-unsloth

Unsloth LLM fine-tuning and inference engine packaged as a secure, signed container for **Pantavisor**.

## Features

- **Hardware Acceleration:** Native GPU passthrough (`/dev/nvhost*`, `/dev/nvmap`, `/dev/nvidia*`, `/dev/dri/*`) via Pantavisor `mdev.json`.
- **Memory Optimized:** Leverages Unsloth Triton custom attention kernels to reduce VRAM consumption by up to 70%.
- **OpenAI Compatible API:** Exposes `/v1/chat/completions`, `/v1/models`, and `/health` on port 8000.
- **Persistent Weights:** Models are mounted to `/models` mapped to `/storage/disks/unsloth-models`.
- **Zero-Touch Watchdog:** Signals `pvcontrol signal ready` to the Pantavisor supervisor upon successful CUDA initialization.

## Architecture

- **`pvpkg.json`**: Pantavisor packaging metadata and runtime flags.
- **`mdev.json`**: Character device whitelist rules.
- **`lxc.container.conf`**: CGroup permissions and shared memory (`/dev/shm` 8GB `tmpfs`).
- **`server.py`**: FastAPI serving gateway.
- **`entrypoint.sh`**: Hardware verification and startup daemon.
- **`Dockerfile`**: Multi-arch build container (arm64 / amd64).

## License

Apache-2.0
