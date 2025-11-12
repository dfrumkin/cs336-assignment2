import csv
import statistics
from pathlib import Path
from timeit import default_timer

import torch
from cs336_basics.model import scaled_dot_product_attention
from hydra import main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    sync = torch.cuda.synchronize
    memory_fn = torch.cuda.memory_allocated
elif torch.mps.is_available():
    device = torch.device("mps")
    sync = torch.mps.synchronize
    memory_fn = torch.mps.current_allocated_memory
else:
    device = torch.device("cpu")

    def memory_fn():
        return 0.0

    def sync():
        return None


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


def calc_statistics(data: list[float], name: str) -> dict[str, float]:
    if data:
        mean = statistics.mean(data)
        std = statistics.stdev(data)
    else:
        mean = std = 0.0
    return {f"{name}_mean": mean, f"{name}_std": std}


@main(config_path="conf", config_name="attention", version_base=None)
def run(cfg: DictConfig) -> None:
    # Get overridden parameters
    results: dict[str, str | float] = get_sweep_params()  # type: ignore
    print(f"Starting: {results}")

    # Instantiate model and inputs
    q = torch.randn(cfg.batch_size, cfg.context_length, cfg.d_model, device=device, requires_grad=True)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Timing data
    forw_times = []
    back_times = []
    mem_after_forw = []

    attention = scaled_dot_product_attention
    if cfg.compile:
        attention = torch.compile(attention)

    if cfg.forward_only:
        # Timing forward pass for inference (no gradients)
        with torch.inference_mode():
            # Warmup
            for _ in range(cfg.num_warmup_steps):
                attention(q, k, v)
            sync()

            # Measurements
            for _ in range(cfg.num_measurement_steps):
                # Time: forward pass & sync
                t0 = default_timer()
                attention(q, k, v)
                sync()
                t1 = default_timer()

                # Collect timing data
                forw_times.append(t1 - t0)
    else:
        # Timing forward (including loss) and backward passes for training
        forw_times = []
        back_times = []

        # Warmup
        for _ in range(cfg.num_warmup_steps):
            q.grad = k.grad = v.grad = None
            out = attention(q, k, v)
            loss = out.sum()
            loss.backward()

        # Measurements
        for _ in range(cfg.num_measurement_steps):
            # zero_grad and sync
            q.grad = k.grad = v.grad = None
            sync()

            t0 = default_timer()

            # Forward pass
            out = attention(q, k, v)
            loss = out.mean()
            sync()

            t1 = default_timer()
            memory = memory_fn()  # This is super-fast

            # Backward pass
            loss.backward()
            sync()

            t2 = default_timer()

            # Collect data
            forw_times.append(t1 - t0)
            back_times.append(t2 - t1)
            mem_after_forw.append(memory)

    # Compute statistics
    results.update(calc_statistics(forw_times, "forward"))
    results.update(calc_statistics(back_times, "backward"))
    results.update(calc_statistics(mem_after_forw, "memory"))

    # Write statistics
    csv_path = Path("benchmark_attention_results.csv")
    results = dict(sorted(results.items()))

    header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if header:
            writer.writeheader()
        writer.writerow(results)

    print(f"Finished: {results}")


if __name__ == "__main__":
    run()
