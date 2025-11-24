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
    local_batch_size = cfg.batch_size // world_size
    if cfg.backend == "nccl":
        assert cfg.world_size <= torch.cuda.device_count()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        sync = torch.cuda.synchronize
    else:
        assert cfg.backend == "gloo"
        device = torch.device("cpu")

        def sync():
            return None

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(cfg.backend, rank=rank, world_size=world_size)

    # Initialize for training
    torch.manual_seed(rank)
    model = instantiate(cfg.model).to(device)

    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p, src=0)

    optim = torch.optim.AdamW(model.parameters())
    step_times = []
    comm_times = []

    for step in range(cfg.num_warmup_steps + cfg.num_measurement_steps):
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

        t1 = default_timer()
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad.div_(world_size)

        sync()
        t2 = default_timer()

        optim.step()
        sync()
        t3 = default_timer()

        # Collect timing data
        if step >= cfg.num_warmup_steps:
            step_times.append(t3 - t0)
            comm_times.append(t2 - t1)

    # Now we want per-iteration MAX over ranks
    step_times_tensor = torch.tensor(step_times, device=device)
    comm_times_tensor = torch.tensor(comm_times, device=device)
    dist.all_reduce(step_times_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(comm_times_tensor, op=dist.ReduceOp.MAX)

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
        results = dict(sorted(results.items()))

        header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            if header:
                writer.writeheader()
            writer.writerow(results)

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
