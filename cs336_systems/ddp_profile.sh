nsys profile -o naive -- python ddp_benchmark.py sync_type=individual batch_size=32 sharded_optimizer=false
nsys profile -o overlap_ind -- python ddp_benchmark.py sync_type=overlap_individual batch_size=32 sharded_optimizer=false
