PY=${PYTHON:-"$(which python)"}

compile=(false true)
forward_only=(false true)
d_model=(16 32 64 128)
ctx=(256 1024 4096 8192 16384)

for cmpl in "${compile[@]}"; do
    for fwd in "${forward_only[@]}"; do
        for d in "${d_model[@]}"; do
            for c in "${ctx[@]}"; do
                "$PY" benchmark_attention.py compile="$cmpl" forward_only="$fwd" d_model="$d" context_length="$c"
            done
        done
    done
done