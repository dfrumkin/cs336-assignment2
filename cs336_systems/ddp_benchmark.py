import csv
import os
from pathlib import Path
from timeit import default_timer

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.nn_utils import cross_entropy  # type: ignore
from hydra import main
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from jaxtyping import Int
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import AdamW

from cs336_systems.ddp_overlap_bucketed import DDPOverlapBucketed
from cs336_systems.ddp_overlap_individual import DDPOverlapIndividual
from cs336_systems.sharded_optimizer import ShardedOptimizer

MB = 1024**2


def get_rand_tokens(cfg: DictConfig, device: torch.device) -> Int[Tensor, "batch_length sequence_length"]:
    return torch.randint(
        low=0, high=cfg.model.vocab_size, size=(cfg.batch_size, cfg.model.context_length), device=device
    )


def get_sweep_params() -> dict[str, str]:
    """Get overridden hydra parameters to describe the sweep job

    Returns:
        dict[str, str]: Parameter names and values as strings
    """
    task_overrides = HydraConfig.get().overrides.task
    out: dict[str, str] = {}
    for s in task_overrides:
        if "=" in s:
            k, v = s.split("=", 1)
            out[k] = v
    return dict(sorted(out.items()))


def ddp_benchmark(rank, cfg, results: dict[str, str | float]) -> None:
    # Get overridden parameters
    if rank == 0:
        print(f"Starting: {results}")

    # Initialize the process
    # Note: we are using just one node => rank == local_rank
    world_size = cfg.world_size
    assert cfg.batch_size % world_size == 0, "batch_size must be divisible by world_size"
    local_batch_size = cfg.batch_size // world_size

    assert cfg.sync_type in ("individual", "batch", "overlap_individual", "overlap_bucketed")
    overlap_individual = cfg.sync_type == "overlap_individual"
    overlap_bucketed = cfg.sync_type == "overlap_bucketed"
    sync_batch = cfg.sync_type == "batch"

    if cfg.backend == "nccl":
        assert cfg.world_size <= torch.cuda.device_count()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        sync = torch.cuda.synchronize

        def reset_peak_memory():
            torch.cuda.reset_peak_memory_stats()

        def get_peak_memory():  # type: ignore
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            return peak_memory
    else:
        assert cfg.backend == "gloo"
        device = torch.device("cpu")

        def sync():
            return None

        def reset_peak_memory():
            return None

        def get_peak_memory():
            return 0

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(cfg.backend, rank=rank, world_size=world_size)

    # Initialize for training
    reset_peak_memory()
    torch.manual_seed(rank)
    model = instantiate(cfg.model).to(device)

    if overlap_individual:
        model = DDPOverlapIndividual(model)
    elif overlap_bucketed:
        model = DDPOverlapBucketed(model, bucket_size_mb=cfg.bucket_size_mb)
    else:
        with torch.no_grad():
            for p in model.parameters():
                dist.broadcast(p, src=0)

    optim = ShardedOptimizer(model.parameters(), AdamW) if cfg.sharded_optimizer else AdamW(model.parameters())
    step_times = []
    comm_times = []

    mem_peak_init = get_peak_memory()
    mem_peak_before_optim = 0
    mem_peak_after_optim = 0

    for step in range(cfg.num_warmup_steps + cfg.num_measurement_steps):
        reset_peak_memory()

        # Create data - shard a randomly generated batch
        torch.manual_seed(step)  # Same input data in all workers
        all_x = get_rand_tokens(cfg, device)
        all_y = get_rand_tokens(cfg, device)
        start = rank * local_batch_size
        end = start + local_batch_size
        x = all_x[start:end]
        y = all_y[start:end]

        # Training step
        t0 = default_timer()
        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        sync()

        if overlap_individual or overlap_bucketed:
            t1 = t2 = 0.0  # No way to separate the communication overhead; using Nsight instead.
            model.finish_gradient_synchronization()
        else:
            t1 = default_timer()
            if sync_batch:
                # Here, we assume that gradients are dense.
                # In our implementation, embedding gradients are dense, though they could be sparse.
                grads = [p.grad for p in model.parameters() if p.grad is not None]

                if grads:
                    flat = torch._utils._flatten_dense_tensors([g.contiguous() for g in grads])  # type: ignore
                    dist.all_reduce(flat)
                    flat.div_(world_size)
                    for g, g_flat in zip(
                        grads,
                        torch._utils._unflatten_dense_tensors(flat, grads),  # type: ignore
                        strict=True,
                    ):
                        g.copy_(g_flat)
            else:
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                        p.grad.div_(world_size)

            sync()
            t2 = default_timer()

        if step >= cfg.num_warmup_steps:
            mem_peak_before_optim = max(mem_peak_before_optim, get_peak_memory())
        optim.step()
        if step >= cfg.num_warmup_steps:
            mem_peak_after_optim = max(mem_peak_after_optim, get_peak_memory())
        sync()
        t3 = default_timer()

        # Collect timing data
        if step >= cfg.num_warmup_steps:
            step_times.append(t3 - t0)
            comm_times.append(t2 - t1)

    # Now we want per-iteration MAX over ranks for time and memory
    step_times_tensor = torch.tensor(step_times, device=device)
    comm_times_tensor = torch.tensor(comm_times, device=device)
    dist.reduce(step_times_tensor, dst=0, op=dist.ReduceOp.MAX)
    dist.reduce(comm_times_tensor, dst=0, op=dist.ReduceOp.MAX)

    mem = torch.tensor([mem_peak_init, mem_peak_before_optim, mem_peak_after_optim], device=device) / MB
    dist.reduce(mem, dst=0, op=dist.ReduceOp.MAX)

    # Compute and print statistics
    if rank == 0:
        step_time_mean = step_times_tensor.mean().item() * 1000
        step_time_std = step_times_tensor.std().item() * 1000
        comm_time_mean = comm_times_tensor.mean().item() * 1000
        comm_time_std = comm_times_tensor.std().item() * 1000
        comm_time_frac = (comm_times_tensor.sum() / step_times_tensor.sum()).item()

        results.update(
            {
                "step_time_mean": step_time_mean,
                "step_time_std": step_time_std,
                "comm_time_mean": comm_time_mean,
                "comm_time_std": comm_time_std,
                "comm_time_frac": comm_time_frac,
            }
        )
        results = dict(sorted(results.items()))

        # Write statistics
        csv_path = Path(cfg.out_path)
        header = not csv_path.exists()

        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if header:
                writer.writeheader()
            writer.writerow(results)

        # Memory:
        mem_peak_init, mem_peak_before_optim, mem_peak_after_optim = mem.cpu().tolist()
        print(
            f"Memory (MB): init {mem_peak_init:.1f}, before optim {mem_peak_before_optim:.1f}, "
            f"after optim {mem_peak_after_optim:.1f}"
        )
        print(f"Finished: {results}")

    # Not strictly necessary for a simple script
    dist.barrier()
    dist.destroy_process_group()


@main(config_path="conf", config_name="ddp_benchmark", version_base=None)
def run(cfg: DictConfig) -> None:
    results = get_sweep_params()
    mp.spawn(  # type: ignore
        ddp_benchmark,
        args=(cfg, results),
        nprocs=cfg.world_size,
        join=True,
    )


if __name__ == "__main__":
    run()
