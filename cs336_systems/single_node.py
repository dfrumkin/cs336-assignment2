import csv
import os
from pathlib import Path
from timeit import default_timer

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from humanfriendly import parse_size
from hydra import main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


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


def ddp_single_node(rank, cfg, results: dict[str, str | float]) -> None:
    if rank == 0:
        print(f"Starting: {results}")

    backend = cfg.backend
    world_size = cfg.world_size
    tensor_size = parse_size(cfg.tensor_size) // 4

    if backend == "nccl":
        assert world_size <= torch.cuda.device_count()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        sync = torch.cuda.synchronize
    else:
        assert backend == "gloo"
        device = torch.device("cpu")

        def sync():
            pass

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Allocate tensor
    x = torch.randn(tensor_size, device=device)

    for _ in range(cfg.num_warmup_steps):
        dist.all_reduce(x, async_op=False)
    sync()

    # Measurements
    times = []
    for _ in range(cfg.num_measurement_steps):
        t0 = default_timer()
        dist.all_reduce(x, async_op=False)
        sync()
        t1 = default_timer()
        times.append(t1 - t0)

    # Now we want per-iteration MAX over ranks
    times_tensor = torch.tensor(times, device=device)
    dist.all_reduce(times_tensor, op=dist.ReduceOp.MAX)  # in-place max across ranks

    # Compute and write statistics
    if rank == 0:
        times_host = times_tensor.cpu().numpy()
        results["mean"] = times_host.mean()
        results["std"] = times_host.std(ddof=1)

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


@main(config_path="conf", config_name="ddp_single_node", version_base=None)
def run(cfg: DictConfig) -> None:
    results = get_sweep_params()
    mp.spawn(  # type: ignore
        fn=ddp_single_node,
        args=(cfg, results),
        nprocs=cfg.world_size,
        join=True,
    )


if __name__ == "__main__":
    run()
