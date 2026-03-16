"""Optimizer utilities, including a lightweight Yogi implementation."""

from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import Optimizer


class Yogi(Optimizer):
    """Yogi optimizer from Zaheer et al., implemented for dense gradients."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Invalid learning rate.")
        if eps <= 0.0:
            raise ValueError("Invalid epsilon value.")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1 value.")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2 value.")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Yogi does not support sparse gradients.")
                if weight_decay != 0.0:
                    grad = grad.add(param, alpha=weight_decay)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                grad_sq = grad.pow(2)
                direction = torch.sign(exp_avg_sq - grad_sq)
                exp_avg_sq.addcmul_(direction, grad_sq, value=-(1.0 - beta2))
                exp_avg_sq.clamp_(min=0.0)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denom = exp_avg_sq.sqrt() / bias_correction2**0.5
                denom.add_(eps)
                step_size = lr / bias_correction1
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
