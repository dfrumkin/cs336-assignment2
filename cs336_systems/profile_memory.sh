#!/usr/bin/env bash

sudo nvidia-smi -pm 0   # turn persistence mode OFF

PY=${PYTHON:-"$(which python)"}

mixed_precision=(false true)
forward_only=(false true)
ctx=(128 256 512)

for mx in "${mixed_precision[@]}"; do
    for fwd in "${forward_only[@]}"; do
        for c in "${ctx[@]}"; do
            sudo nvidia-smi --gpu-reset
            "$PY" benchmark.py \
                mixed_precision="$mx" forward_only="$fwd" model.context_length="$c" model="2.7B" num_warmup_steps=5 mem_profile=true use_optimizer=true
        done
    done
done

# sudo nvidia-smi -pm 1   # turn persistence mode ON - if it was on to begin with
