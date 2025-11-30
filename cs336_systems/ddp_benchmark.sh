python ddp_benchmark.sh -m sync_type=individual,batch,overlap_individual batch_size=2,4,8,16,32 bucket_size_mb=None sharded_optimizer=false
python ddp_benchmark.sh -m sync_type=overlap_bucketed batch_size=2,4,8,16,32 bucket_size_mb=1,10,100,1000 sharded_optimizer=false
