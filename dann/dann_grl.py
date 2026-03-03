from __future__ import annotations

import torch
from torch import nn


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    """Gradient Reversal Layer used by DANN."""

    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def set_lambda(self, lambd: float) -> None:
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.lambd)
