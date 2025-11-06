# sudo nvidia-smi -pm 0   # turn persistence mode OFF
sudo "$(which python)" benchmark.py -m model="small,medium,large,xl,'2.7B'" model.context_length="128,256,512,1024" num_warmup_steps="0,1,2,5"
# sudo nvidia-smi -pm 1   # turn persistence mode ON