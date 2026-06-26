from typing import Iterable, Optional

import torch


class EqPropMomentum(torch.optim.Optimizer):
    """Momentum update for an EqProp-style physics loss.

    This optimizer is exposed for PCN ablations. In this connector experiment the
    main fair comparison should still report answer cross entropy for all models.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        beta: float = 0.1,
        lambda_settle: float = 0.0,
        momentum: float = 0.9,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        defaults = dict(
            lr=lr,
            beta=beta,
            lambda_settle=lambda_settle,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    def build_physics_loss(
        self,
        free_energy: torch.Tensor,
        clamped_energy: torch.Tensor,
        settle_penalty: Optional[torch.Tensor],
        beta: float,
        lambda_settle: float,
    ) -> torch.Tensor:
        loss = (clamped_energy - free_energy) / beta
        if settle_penalty is not None and lambda_settle > 0:
            loss = loss + lambda_settle * settle_penalty
        return loss

    def step(self, free_energy=None, clamped_energy=None, settle_penalty=None, closure=None):
        if closure is not None:
            with torch.enable_grad():
                free_energy, clamped_energy, settle_penalty = closure()
        if free_energy is None or clamped_energy is None:
            raise ValueError("EqPropMomentum.step requires free_energy and clamped_energy tensors.")

        group0 = self.param_groups[0]
        physics_loss = self.build_physics_loss(
            free_energy,
            clamped_energy,
            settle_penalty,
            group0["beta"],
            group0["lambda_settle"],
        )
        self.zero_grad()
        physics_loss.backward()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                weight_decay = group["weight_decay"]
                nesterov = group["nesterov"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if weight_decay > 0:
                        grad = grad.add(p, alpha=weight_decay)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = grad.detach().clone()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)
                    update = grad.add(buf, alpha=momentum) if nesterov else buf
                    p.add_(update, alpha=-lr)

        return float(physics_loss.detach().item())


def build_optimizer(name: str, params, cfg: dict, eqprop_cfg: dict | None = None):
    name = name.lower()
    lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "eqprop_momentum":
        eq = eqprop_cfg or {}
        return EqPropMomentum(
            params,
            lr=lr,
            beta=float(eq.get("beta", 0.1)),
            lambda_settle=float(eq.get("lambda_settle", 0.0)),
            momentum=float(eq.get("momentum", 0.9)),
            dampening=float(eq.get("dampening", 0.0)),
            weight_decay=weight_decay,
            nesterov=bool(eq.get("nesterov", False)),
        )
    raise ValueError(f"Unknown optimizer: {name}")
