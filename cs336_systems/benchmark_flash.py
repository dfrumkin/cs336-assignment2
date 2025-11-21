import csv
import warnings
from pathlib import Path

import torch
from cs336_basics.model import scaled_dot_product_attention
from hydra import main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.cuda import OutOfMemoryError as CudaOOM
from triton.runtime.errors import OutOfResources as TritonOOR
from triton.testing import do_bench

from cs336_systems.flash_triton import FlashTorchBwd, FlashTritonBwd

warnings.filterwarnings("ignore", message="Skipping serialization of skipfiles_inline_module_allowlist")

assert torch.cuda.is_available()
device = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch._dynamo.reset()
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
KERNEL_RESOURCE_ERRORS = (CudaOOM, TritonOOR)


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


@main(config_path="conf", config_name="flash", version_base=None)
def run(cfg: DictConfig) -> None:
    # Get overridden parameters
    results: dict[str, str | float] = get_sweep_params()  # type: ignore
    print(f"Starting: {results}")

    # Instantiate inputs
    dtype = DTYPE_MAP[cfg.dtype]
    q = torch.randn(cfg.batch_size, cfg.context_length, cfg.d_model, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Instantiate attention
    match cfg.attention:
        case "pytorch":
            mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool, device=device))

            @torch.compile
            def forward():  # type: ignore
                return scaled_dot_product_attention(q, k, v, mask)
        case "flash_torch_bwd":

            def forward():
                return FlashTorchBwd.apply(q, k, v, True)
        case "flash_triton_bwd":

            def forward():
                return FlashTritonBwd.apply(q, k, v, True)

    # Forward inference
    try:
        with torch.inference_mode():
            fw = do_bench(forward, warmup=cfg.warmup, rep=cfg.reps)
    except KERNEL_RESOURCE_ERRORS:
        fw = ""

    # Backward
    try:
        out = forward()
        loss = out.mean()  # type: ignore

        @torch.compile
        def backward():
            q.grad = k.grad = v.grad = None
            loss.backward(retain_graph=True)

        bk = do_bench(backward, warmup=cfg.warmup, rep=cfg.reps)
    except KERNEL_RESOURCE_ERRORS:
        bk = ""

    # Forward-backward
    try:

        @torch.compile
        def forward_backward():
            out = forward()
            loss = out.mean()  # type: ignore
            q.grad = k.grad = v.grad = None
            loss.backward()

        fw_bk = do_bench(forward_backward, warmup=cfg.warmup, rep=cfg.reps)
    except KERNEL_RESOURCE_ERRORS:
        fw_bk = ""

    # Write statistics
    results.update({"fw": fw, "bk": bk, "fw_bk": fw_bk})
    csv_path = Path("benchmark_flash.csv")

    header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if header:
            writer.writeheader()
        writer.writerow(results)

    print(f"Finished: {results}")


if __name__ == "__main__":
    run()
