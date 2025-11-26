import torch
import torch.distributed as dist
from torch import nn


class DDPOverlapIndividual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        self.world_size = dist.get_world_size()

        # Broadcast parameters
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p, src=0)

        # Register parameter hooks
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(
                    lambda p: self.handles.append((dist.all_reduce(p.grad, async_op=True), p))
                )

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for handle, param in self.handles:
            handle.wait()
            param.grad.div_(self.world_size)
        self.handles.clear()
