#!/bin/sh
set -e

echo "=========================================="
echo " Starting Pantavisor Unsloth Container    "
echo "=========================================="

# Check GPU node presence
if [ -e /dev/nvhost-ctrl ] || [ -e /dev/nvidia0 ] || [ -e /dev/nvmap ]; then
    echo "[GPU Check] Detected NVIDIA hardware device nodes:"
    ls -l /dev/nv* 2>/dev/null || true
else
    echo "[GPU Check] Notice: Standard GPU device nodes not yet populated in /dev."
fi

# Signal READY to Pantavisor watchdog
if command -v pvcontrol >/dev/null 2>&1; then
    echo "[Pantavisor] Signalling container READY state..."
    pvcontrol signal ready || true
fi

# Launch the FastAPI model server
echo "[Service] Starting Unsloth inference endpoint on port 8000..."
exec python3 /app/server.py
