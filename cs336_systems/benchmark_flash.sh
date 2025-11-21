export TRITON_PRINT_AUTOTUNE=1

python benchmark_flash.py -m \
    attention=pytorch,flash_torch_bwd,flash_triton_bwd \
    context_length=128,256,512,1024,2048,4096,8192,16384,32768,65536 \
    d_model=16,32,64,128 \
    dtype=bfloat16,float32 \
    hydra.output_subdir=null \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled
