#!/usr/bin/env bash

PY=${PYTHON:-"$(which python)"}

forward_only=(false true)
ctx=(128 256 512)

for fwd in "${forward_only[@]}"; do
    for c in "${ctx[@]}"; do
        "$PY" benchmark.py \
            forward_only="$fwd" model.context_length="$c" model="2.7B" num_warmup_steps=5 mem_profile=true
    done
done