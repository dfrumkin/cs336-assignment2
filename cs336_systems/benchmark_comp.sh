#!/usr/bin/env bash

PY=${PYTHON:-"$(which python)"}

compile=(false true)
forward_only=(false true)
models=(small medium large xl "2.7B")
ctx=(128 256 512 1024)

for comp in "${compile[@]}"; do
    for fwd in "${forward_only[@]}"; do
        for m in "${models[@]}"; do
            for c in "${ctx[@]}"; do
                "$PY" benchmark.py \
                    compile="$comp" forward_only="$fwd" model="$m" model.context_length="$c" num_warmup_steps=5 use_optimizer=true
            done
        done
    done
done
