#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON:-"$(which python)"}
TRACE_DIR=${TRACE_DIR:-traces}
mkdir -p "$TRACE_DIR"

forward_only=(false true)
models=(small medium large xl "2.7B")
ctx=(128 256 512 1024)

for fwd in "${forward_only[@]}"; do
  for m in "${models[@]}"; do
    for c in "${ctx[@]}"; do
      out="${TRACE_DIR}/trace_fwd=${fwd}_m=${m}_ctx=${c}"
      nsys profile \
        -o "$out" -- \
        "$PY" benchmark.py \
        forward_only="$fwd" model="$m" model.context_length="$c"
    done
  done
done