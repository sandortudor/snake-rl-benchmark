from __future__ import annotations


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def choose_device() -> str:
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(kind: str, input_size: int, action_count: int):
    import torch
    from torch import nn

    class QNetwork(nn.Module):
        def __init__(self, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_count),
            )

        def forward(self, x):
            return self.net(x)

    class DuelingNetwork(nn.Module):
        def __init__(self, hidden: int = 128):
            super().__init__()
            self.features = nn.Sequential(nn.Linear(input_size, hidden), nn.ReLU())
            self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.advantage = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_count),
            )

        def forward(self, x):
            features = self.features(x)
            value = self.value(features)
            advantage = self.advantage(features)
            return value + advantage - advantage.mean(dim=1, keepdim=True)

    if kind == "lightweight_snake_ddqn":
        return DuelingNetwork(hidden=64)
    if kind == "dueling_dqn":
        return DuelingNetwork(hidden=128)
    return QNetwork(hidden=128)

