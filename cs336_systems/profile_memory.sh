#!/usr/bin/env bash

PY=${PYTHON:-"$(which python)"}

mixed_precision=(false true)
forward_only=(false true)
ctx=(128 256 512)

for mx in "${mixed_precision[@]}"; do
    for fwd in "${forward_only[@]}"; do
        for c in "${ctx[@]}"; do
            "$PY" benchmark.py \
                mixed_precision="$mx" forward_only="$fwd" model.context_length="$c" model="2.7B" num_warmup_steps=5 mem_profile=true
        done
    done
done