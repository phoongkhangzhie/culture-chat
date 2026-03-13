#!/bin/bash
# Annotate a WildChat sample using a local vLLM server.
#
# Usage:
#   sbatch run.sh <input-jsonl> <output-jsonl> [tensor-parallel-size]
#
# Examples:
#   # From a pre-sampled local file
#   sbatch run.sh output/sample.jsonl output/annotations_vllm.jsonl
#
#   # With tensor parallelism across 4 GPUs
#   sbatch run.sh output/sample.jsonl output/annotations_vllm.jsonl 4
#
# Environment variables (override defaults):
#   VLLM_MODEL    — path or HuggingFace ID of the model to serve
#   VLLM_PORT     — port for the vLLM server (default: 8000)
#   RPM           — requests per minute throttle (default: 60)

set -e

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input-jsonl> <output-jsonl> [tensor-parallel-size]"
    exit 1
fi

INPUT_JSONL="$1"
OUTPUT_JSONL="$2"
TENSOR_PARALLEL="${3:-1}"

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
VLLM_LOG="logs/vllm-$$.log"
WAIT_TIMEOUT=2700   # 45 min — covers large model loads
WAIT_INTERVAL=10
READY_MARKER="Application startup complete."
RPM="${RPM:-120}"

mkdir -p logs

# ---------------------------------------------------------------------------
# 1. Start vLLM server in background
# ---------------------------------------------------------------------------
echo "[$(date)] Starting vLLM (model: ${VLLM_MODEL}, port: ${VLLM_PORT}, tp: ${TENSOR_PARALLEL})"
echo "[$(date)] vLLM log: ${VLLM_LOG}"

vllm serve "${VLLM_MODEL}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --dtype bfloat16 \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

# ---------------------------------------------------------------------------
# 2. Wait for model to finish loading
# ---------------------------------------------------------------------------
echo "[$(date)] Waiting for vLLM to finish loading model..."

elapsed=0
while true; do
    if grep -q "${READY_MARKER}" "${VLLM_LOG}" 2>/dev/null; then
        break
    fi

    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[$(date)] ERROR: vLLM process (PID ${VLLM_PID}) exited unexpectedly."
        echo "[$(date)] Last 20 lines of vLLM log:"
        tail -20 "${VLLM_LOG}" || true
        exit 1
    fi

    if [ "${elapsed}" -ge "${WAIT_TIMEOUT}" ]; then
        echo "[$(date)] ERROR: vLLM did not become ready within ${WAIT_TIMEOUT}s."
        kill "${VLLM_PID}" 2>/dev/null
        exit 1
    fi

    sleep "${WAIT_INTERVAL}"
    elapsed=$((elapsed + WAIT_INTERVAL))
done

echo "[$(date)] vLLM is ready (waited ${elapsed}s)."

# ---------------------------------------------------------------------------
# 3. Run annotation pipeline
# ---------------------------------------------------------------------------
python main.py annotate \
    --backend vllm \
    --model "${VLLM_MODEL}" \
    --base-url "${VLLM_BASE_URL}" \
    --input "${INPUT_JSONL}" \
    --output "${OUTPUT_JSONL}" \
    --requests-per-minute "${RPM}" \
    --verbose

# ---------------------------------------------------------------------------
# 4. Shut down vLLM cleanly
# ---------------------------------------------------------------------------
echo "[$(date)] Pipeline done. Stopping vLLM (PID ${VLLM_PID})."
kill "${VLLM_PID}"
wait "${VLLM_PID}" 2>/dev/null || true
echo "[$(date)] Done."
