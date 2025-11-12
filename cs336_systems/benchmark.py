import csv
import statistics
from contextlib import contextmanager, nullcontext
from pathlib import Path
from timeit import default_timer

import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.nn_utils import cross_entropy  # type: ignore
from hydra import main
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from jaxtyping import Int
from omegaconf import DictConfig
from torch import Tensor

# Monkey patch for annotated scaled dot-product attention - only if profiling self-attention!
# import cs336_basics
# from model_patch import annotated_scaled_dot_product_attention
# cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

# Configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    sync = torch.cuda.synchronize
elif torch.mps.is_available():
    device = torch.device("mps")
    sync = torch.mps.synchronize
else:
    device = torch.device("cpu")

    def sync():
        return None


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


def calc_statistics(times: list[float], name: str) -> dict[str, float]:
    if times:
        mean = statistics.mean(times)
        std = statistics.stdev(times)
    else:
        mean = std = 0.0
    return {f"{name}_mean": mean, f"{name}_std": std}


@contextmanager
def profile_memory(filename: str):
    root = Path("memory_dump")
    root.mkdir(exist_ok=True)
    dump_path = str(root / (filename + ".pickle"))

    torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    try:
        yield
    finally:
        torch.cuda.memory._dump_snapshot(dump_path)
        torch.cuda.memory._record_memory_history(enabled=None)


def maybe_profile_memory(enabled: bool, filename: str):
    return profile_memory(filename) if enabled else nullcontext()


@main(config_path="conf", config_name="benchmark", version_base=None)
def run(cfg: DictConfig) -> None:
    # Get overridden parameters
    results: dict[str, str | float] = get_sweep_params()  # type: ignore
    print(f"Starting: {results}")

    # Instantiate model and inputs
    model = instantiate(cfg.model).to(device)
    inputs = get_rand_tokens(cfg, device)

    # Mixed precision
    mp_context = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if cfg.mixed_precision else nullcontext()

    # Instantiate optimizer
    if cfg.use_optimizer:
        embedding = [model.token_embeddings.weight]  # type: ignore
        others = [p for p in model.parameters() if p is not model.token_embeddings.weight]  # type: ignore
        params = [
            {"params": others, "weight_decay": 0.1},
            {"params": embedding, "weight_decay": 0.0},
        ]
        optimizer = instantiate(cfg.optimizer, params=params)
    else:
        optimizer = None

    forw_times = []
    back_times = []
    suffix = "_".join(f"{k}_{v}" for k, v in results.items())
    forward_name = "forward_" + suffix
    backward_name = "backward_" + suffix
    optimizer_name = "optimizer_" + suffix
    train_step_name = "train_" + suffix
    num_mesurement_steps = 1 if cfg.mem_profile else cfg.num_measurement_steps

    if cfg.forward_only:
        # Timing forward pass for inference (no gradients)
        with mp_context, torch.inference_mode():
            # Warmup
            for _ in range(cfg.num_warmup_steps):
                model(inputs)
            sync()

            # Measurements
            for _ in range(num_mesurement_steps):
                # Time: forward pass & sync
                t0 = default_timer()
                with nvtx.range(forward_name), maybe_profile_memory(cfg.mem_profile, forward_name):
                    model(inputs)
                    sync()
                t1 = default_timer()

                # Collect timing data
                forw_times.append(t1 - t0)
    else:
        # Timing forward (including loss) and backward passes for training
        targets = get_rand_tokens(cfg, device)
        forw_times = []
        back_times = []

        # Warmup
        for _ in range(cfg.num_warmup_steps):
            model.zero_grad(set_to_none=True)
            with mp_context:
                logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()
            if optimizer is not None:
                optimizer.step()

        # Measurements
        for _ in range(num_mesurement_steps):
            with maybe_profile_memory(cfg.mem_profile, train_step_name):
                # zero_grad and sync
                model.zero_grad(set_to_none=True)
                sync()

                t0 = default_timer()

                # Forward pass
                with nvtx.range(forward_name), mp_context:
                    logits = model(inputs)
                    loss = cross_entropy(logits, targets)
                    sync()

                t1 = default_timer()

                # Backward pass
                with nvtx.range(backward_name):
                    loss.backward()
                    sync()

                t2 = default_timer()

                # Optimizer step
                if optimizer is not None:
                    with nvtx.range(optimizer_name):
                        optimizer.step()
                        sync()

            # Collect timing data
            forw_times.append(t1 - t0)
            back_times.append(t2 - t1)

    if not cfg.mem_profile:
        # Compute statistics
        results.update(calc_statistics(forw_times, "forward"))
        results.update(calc_statistics(back_times, "backward"))

        # Write statistics
        csv_path = Path("benchmark_results.csv")
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
