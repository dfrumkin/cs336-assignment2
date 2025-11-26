import torch
import torch.distributed as dist
from torch import nn


class Bucket:
    def __init__(self, grad: torch.Tensor, world_size: int):
        self.world_size = world_size
        self.grads = [grad]
        self.flat = None
        self.handle = None

    def add(self, grad: torch.Tensor):
        self.grads.append(grad)

    def reduce(self):
        self.flat = torch._utils._flatten_dense_tensors([g.contiguous() for g in self.grads])  # type: ignore
        self.handle = dist.all_reduce(self.flat, async_op=True)

    def sync(self):
        self.handle.wait()  # type: ignore
        for g, g_flat in zip(
            self.grads,
            torch._utils._unflatten_dense_tensors(self.flat, self.grads),  # type: ignore
            strict=True,
        ):
            g.copy_(g_flat)
            g.div_(self.world_size)


class DDPOverlapBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.world_size = dist.get_world_size()
        self.buckets = []
        self.last_bucket = None
        self.last_bucket_size_mb = 0.0

        # Broadcast parameters
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p, src=0)

        # Register parameter hooks
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, p: nn.Parameter):
        # Normally, we could figure out the order of gradients and do pre-bucketing after the first pass,
        # but let us keep it simple and just do online bucketing.
        grad: torch.Tensor = p.grad  # type: ignore
        assert not grad.is_sparse
        grad_size_mb = grad.numel() * grad.element_size() / (1024 * 1024)
        if self.last_bucket is not None and self.last_bucket_size_mb + grad_size_mb <= self.bucket_size_mb:
            self.last_bucket.add(grad)  # type: ignore
            self.last_bucket_size_mb += grad_size_mb
        else:
            if self.last_bucket is not None:
                self.last_bucket.reduce()
            self.last_bucket = Bucket(grad, self.world_size)
            self.buckets.append(self.last_bucket)
            self.last_bucket_size_mb = grad_size_mb
        # A micro-optimization for the case when we have exactly filled or overfilled the bucket
        if self.last_bucket_size_mb >= self.bucket_size_mb:
            self.last_bucket.reduce()  # type: ignore
            self.last_bucket = None
            self.last_bucket_size_mb = 0.0

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        if self.last_bucket is not None:
            self.last_bucket.reduce()
        for bucket in self.buckets:
            bucket.sync()
        self.buckets.clear()
        self.last_bucket = None
        self.last_bucket_size_mb = 0.0
