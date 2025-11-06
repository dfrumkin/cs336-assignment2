import csv
import statistics
import subprocess
from pathlib import Path
from timeit import default_timer

import torch
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


def calc_statistics(times: list[float], name: str) -> dict[str, float]:
    mean = statistics.mean(times)
    std = statistics.stdev(times)
    return {f"{name}_mean": mean, f"{name}_std": std}


def reset_gpu():
    print("Trying GPU reset...")
    try:
        subprocess.run(["nvidia-smi", "--gpu-reset"], capture_output=True, text=True, check=True)
        print("GPU reset successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Skipping GPU reset ({type(e).__name__}: {e})")


@main(config_path="conf", config_name="benchmark", version_base=None)
def run(cfg: DictConfig) -> None:
    # Start from a cold state
    reset_gpu()

    # Configuration - must be done after CUDA reset
    if torch.cuda.is_available():
        device = torch.device("cuda")
        sync = torch.cuda.synchronize
        backend = "inductor"
    elif torch.mps.is_available():
        device = torch.device("mps")
        sync = torch.mps.synchronize
        backend = "aot_eager"
    else:
        device = torch.device("cpu")
        backend = "eager"

        def sync():
            return None

    # Get overridden parameters (normally, from a hydra sweep run)
    results: dict[str, str | float] = get_sweep_params()  # type: ignore
    print(f"Starting: {results}")

    # Instantiate model and inputs
    model = instantiate(cfg.model).to(device)
    if cfg.compile:
        model = torch.compile(model, backend=backend)
    inputs = get_rand_tokens(cfg, device)

    if cfg.forward_only:
        times = []
        # Timing forward pass for inference (no gradients)
        with torch.inference_mode():
            # Warmup
            for _ in range(cfg.num_warmup_steps):
                model(inputs)
            sync()

            # Measurements
            for _ in range(cfg.num_measurement_steps):
                # Time: forward pass & sync
                t0 = default_timer()
                model(inputs)
                sync()
                t1 = default_timer()

                # Collect timing data
                times.append(t1 - t0)

        # Compute statistics
        results.update(calc_statistics(times, "forward_only"))
    else:
        # Timing forward (including loss) and backward passes for training
        targets = get_rand_tokens(cfg, device)
        forw_times = []
        back_times = []

        # Warmup
        for _ in range(cfg.num_warmup_steps):
            model.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            loss.backward()

        # Measurements
        for _ in range(cfg.num_measurement_steps):
            # zero_grad and sync outside of the timing loop
            model.zero_grad(set_to_none=True)
            sync()

            # Time: forward pass, loss, backward pass & sync
            t0 = default_timer()
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            sync()
            t1 = default_timer()
            loss.backward()
            sync()
            t2 = default_timer()

            # Collect timing data
            forw_times.append(t1 - t0)
            back_times.append(t2 - t1)

        # Compute statistics
        results.update(calc_statistics(forw_times, "forward"))
        results.update(calc_statistics(back_times, "backward"))

    # Write statistics into a CSV file shared among hydra sweep jobs (run sequentially)
    job_dir = Path(HydraConfig.get().runtime.output_dir)
    sweep_root = job_dir.parent
    csv_path = sweep_root / "results.csv"
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
