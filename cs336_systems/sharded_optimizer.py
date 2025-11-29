from typing import Any

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs: Any):
        # We get an exception if initializing an optimizer with an empty param list => lazy init
        self.optimizer = None
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs

        # Our rank
        self.rank = dist.get_rank()

        # Current parameter count per rank - for load balancing
        self.rank_param_count = [0] * dist.get_world_size()

        # Parameter -> rank
        self.param_rank = {}

        # Now let the parent call our add_param_group
        super().__init__(params, defaults={})

    def step(self, closure=None, **kwargs):
        assert closure is None
        self.optimizer.step(closure, **kwargs)  # type: ignore

        # Synchronize all parameters across ranks after the step.
        # We could could bucket by dtype and flatten/unflatten to reduce the number of broadcasts.
        with torch.no_grad():
            for p in self.param_rank:
                dist.broadcast(p, src=self.param_rank[p])

    def add_param_group(self, param_group: dict[str, Any]):
        local_params = []

        # Load-balance parameters
        for p in param_group["params"]:
            target_rank = min(range(len(self.rank_param_count)), key=self.rank_param_count.__getitem__)
            self.param_rank[p] = target_rank
            self.rank_param_count[target_rank] += p.numel()

            if self.rank == target_rank:
                local_params.append(p)

        # Create a local parameter group if necessary
        if local_params:
            local_param_group = dict(param_group, params=local_params)
            if self.optimizer is None:
                self.optimizer = self.optimizer_cls(local_params, **self.kwargs)
            else:
                self.optimizer.add_param_group(local_param_group)
            super().add_param_group(local_param_group)
