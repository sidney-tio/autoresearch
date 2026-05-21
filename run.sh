#!/bin/bash
#SBATCH --job-name=evade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/evade_%j.out
#SBATCH --error=logs/evade_%j.err
#
# Prompt-evasion self-optimization loop.
# Starts a vLLM server for Gemma, then runs loop.py against it.
#
# Submit with:  sbatch run.sh                 # loop until the time limit
#          or:  sbatch run.sh --iters 100     # bounded run
# (any extra args are forwarded to loop.py)

set -e

MODEL="google/gemma-4-31B-it"
GPU_MEM_UTIL=0.5
PORT="${VLLM_PORT:-8000}"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# ── Environment ────────────────────────────────────────────────────────────────
export VLLM_NO_USAGE_STATS=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_BASE_URL="http://localhost:${PORT}/v1"

# uv venv shared with the historian project — must already have
# vllm, transformers and openai installed.
source "$HOME/historian/.venv/bin/activate"

# ── Download the model ─────────────────────────────────────────────────────────
echo "Downloading $MODEL..."
hf download "$MODEL"
echo "Download complete."

# ── Wait helper ────────────────────────────────────────────────────────────────
wait_for_vllm() {
    local port="$1"
    echo "Waiting for vLLM on port $port..."
    for i in $(seq 1 120); do
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "vLLM ready after $((i * 5))s"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: vLLM did not start within 600s" >&2
    return 1
}

# ── Start the vLLM server ──────────────────────────────────────────────────────
echo "Starting vLLM: $MODEL on port $PORT (gpu-memory-utilization=$GPU_MEM_UTIL)"
vllm serve "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

wait_for_vllm "$PORT"

# ── Run the optimization loop ──────────────────────────────────────────────────
echo "Starting prompt-evasion loop..."
python loop.py "$@"
echo "Loop finished."
