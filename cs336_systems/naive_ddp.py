import copy
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

WORLD_SIZE = 2
LOCAL_BATCH_SIZE = 4
STEPS = 2
BACKEND = "gloo"


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8, bias=False), nn.ReLU(), nn.Linear(8, 6, bias=False))

    def forward(self, x):
        return self.net(x)


def test_naive_ddp(rank, world_size, local_batch_size, steps, backend):
    # Initialize the process
    # Note: we are using just one node => rank == local_rank
    if backend == "nccl":
        assert world_size <= torch.cuda.device_count()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        assert backend == "gloo"
        device = torch.device("cpu")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Initialize for training
    torch.manual_seed(rank)
    dist_model = ToyModel().to(device)

    with torch.no_grad():
        for p in dist_model.parameters():
            dist.broadcast(p, src=0)

    dist_optim = torch.optim.AdamW(dist_model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    batch_size = world_size * local_batch_size

    if rank == 0:
        loc_model = copy.deepcopy(dist_model)
        loc_optim = torch.optim.AdamW(loc_model.parameters())
        print("Training...")

    for step in range(steps):
        # Create data - shard a randomly generated batch
        torch.manual_seed(step)  # Same input data in all workers
        all_x = torch.randn(batch_size, 4, device=device)
        all_y = torch.randint(0, 6, (batch_size,), device=device)
        start = rank * local_batch_size
        end = start + local_batch_size
        x = all_x[start:end]
        y = all_y[start:end]

        # Training step - DDP
        dist_optim.zero_grad(set_to_none=True)
        logits = dist_model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        for p in dist_model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad.div_(world_size)

        dist_optim.step()

        # Training step - local
        if rank == 0:
            loc_optim.zero_grad(set_to_none=True)
            logits = loc_model(all_x)
            loss = loss_fn(logits, all_y)
            loss.backward()
            loc_optim.step()

    # Now, compare the two models
    if rank == 0:
        print("Comparing parameters...")
        for (dist_name, dist_param), (loc_name, loc_param) in zip(
            dist_model.named_parameters(), loc_model.named_parameters(), strict=True
        ):
            try:
                torch.testing.assert_close(dist_param, loc_param)
            except AssertionError as e:
                print(f"Mismatch: distributed {dist_name} != local {loc_name}")
                raise e
        print("Success!")

    # Not strictly necessary for a simple script
    dist.barrier()
    dist.destroy_process_group()


def run():
    mp.spawn(
        test_naive_ddp,
        args=(WORLD_SIZE, LOCAL_BATCH_SIZE, STEPS, BACKEND),
        nprocs=WORLD_SIZE,
        join=True,
    )


if __name__ == "__main__":
    run()
