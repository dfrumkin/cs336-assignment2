#!/usr/bin/env bash

set -euo pipefail

sudo nvidia-smi -pm 0   # turn persistence mode OFF

PY=${PYTHON:-"$(which python)"}

forward_only=(false true)
models=(small medium large xl "2.7B")
ctx=(128 256 512 1024)
warmup=(0 1 2 5)

for fwd in "${forward_only[@]}"; do
    for m in "${models[@]}"; do
        for c in "${ctx[@]}"; do
            for w in "${warmup{@}}"; do
                nvidia-smi --gpu-reset
                "$PY" benchmark.py \
                    forward_only="$fwd" model="$m" model.context_length="$c" num_warmup_steps="$w"
            done
        done
    done
done

# sudo nvidia-smi -pm 1   # turn persistence mode ON - if it was on