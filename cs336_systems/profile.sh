sudo nsys profile \
    --capture-range=nvtx --capture-range-end=stop --capture-child-processes=true \
    --sample=none --backtrace=none --gpu-metrics-device=none -o trace -- \
    "$(which python)" benchmark.py \
    -m model="small,medium,large,xl,'2.7B'" model.context_length="128,256,512,1024" \
    num_warmup_steps="0,1,2,5"