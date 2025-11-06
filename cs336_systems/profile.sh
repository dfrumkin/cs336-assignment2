#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON:-"$(which python)"}
TRACE_DIR=${TRACE_DIR:-traces}
mkdir -p "$TRACE_DIR"

models=(small medium large xl "2.7B")
ctx=(128 256 512 1024)
warmups=(0 1 2 5)

for m in "${models[@]}"; do
  for c in "${ctx[@]}"; do
    for w in "${warmups[@]}"; do
      out="${TRACE_DIR}/trace_m=${m}_ctx=${c}_w=${w}_$(date +%s%N)"
      sudo nsys profile \
        --capture-range=nvtx --capture-range-end=stop \
        --sample=none --backtrace=none --gpu-metrics-devices=none \
        -o "$out" -- \
        "$PY" benchmark.py \
        model="$m" model.context_length="$c" num_warmup_steps="$w"
    done
  done
done    