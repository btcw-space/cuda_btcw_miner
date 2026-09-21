#!/usr/bin/env bash
set -euo pipefail
mkdir -p release
ARCH="${CUDA_ARCH:-sm_89}"
MAX_REGS="${1:-${CUDA_MAX_REGS:-160}}"
SIGN_BATCH="${CUDA_SIGN_BATCH:-128}"
if [[ "$SIGN_BATCH" != "64" && "$SIGN_BATCH" != "128" ]]; then
  echo "CUDA_SIGN_BATCH must be 64 or 128" >&2
  exit 2
fi
OUTPUT="release/btcw_cuda_miner_batch${SIGN_BATCH}"
nvcc -O3 -std=c++17 -arch="$ARCH" --maxrregcount "$MAX_REGS" -DBTCW_SIGN_BATCH="$SIGN_BATCH" -Xptxas=-v,-warn-spills btcw_cuda_miner.cu -o "$OUTPUT" -lrt -lpthread
cp "$OUTPUT" release/btcw_cuda_miner
echo "Built $OUTPUT for $ARCH maxrregcount=$MAX_REGS sign_batch=$SIGN_BATCH"
