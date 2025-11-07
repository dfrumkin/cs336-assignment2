import torch
import torch.nn as nn
import torch.optim as optim


# Your ToyModel as given
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


device = torch.device("cuda")
model = ToyModel(in_features=4, out_features=4).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# --- print parameter dtypes ---
print("=== Parameter dtypes ===")
for name, param in model.named_parameters():
    print(f"{name:20s}: {param.dtype}")


# --- hook to capture activations ---
def print_activation_dtype(name):
    def hook(_, __, output):
        if isinstance(output, torch.Tensor):
            print(f"Activation {name:20s}: {output.dtype}")
        elif isinstance(output, (list, tuple)):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    print(f"Activation {name}[{i}]: {o.dtype}")

    return hook


# register hooks
hooks = []
for name, module in model.named_modules():
    if name:  # skip the top-level module
        hooks.append(module.register_forward_hook(print_activation_dtype(name)))

# Dummy data
batch_size = 4
x = torch.randn(batch_size, 4, device=device)
target = torch.randn(batch_size, 4, device=device)

criterion = nn.MSELoss()

# ----- forward + backward in BF16 autocast on CUDA -----
optimizer.zero_grad(set_to_none=True)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(x)  # many ops run in BF16 under autocast
    print("logits dtype:", logits.dtype)  # likely bfloat16
    loss = criterion(logits, target)  # numerically sensitive parts may stay FP32 automatically
    print("loss dtype:", loss.dtype)  # typically float32

loss.backward()  # grads computed for FP32 params
