import torch
import torch.nn as nn

from typing import Callable, Tuple
from torchdiffeq import odeint

from nmode.types import SolverType


class ODESolver(nn.Module):
    """ODE Solver.
    Attributes:
        solver (SolverType): Solver type.
        t_span (torch.Tensor): Time span.
    """

    solver: SolverType
    t_span: torch.Tensor

    def __init__(
        self,
        t_span: torch.Tensor = torch.linspace(0, 1, 10),
        solver: SolverType = "rk4",
    ) -> None:
        super().__init__()
        self.register_buffer("t_span", t_span)
        self.solver = solver

    @torch.compile
    def forward(self, func: Callable, y0: torch.Tensor, inverse: bool = False):
        t_span = self.t_span.flip(0) if inverse else self.t_span
        return odeint(func, y0, t_span, method=self.solver)[-1]


class NeuralODEFunc(torch.autograd.Function):
    saved_tensors: Tuple[torch.Tensor, torch.Tensor]  # type: ignore
    ode_solver: ODESolver

    @staticmethod
    def forward(
        ctx: "NeuralODEFunc",
        y0: torch.Tensor,
        gamma: torch.Tensor,
        ode_solver: ODESolver,
    ) -> torch.Tensor:
        ctx.ode_solver = ode_solver

        @torch.jit.script
        def forward_func(_, y: torch.Tensor) -> torch.Tensor:
            return (
                -y
                + torch.pow(torch.sin(y + gamma), 2)
                + torch.pow(torch.e, -y) * torch.pow(torch.sin(y + gamma), 2)
            )

        y_end = ode_solver(forward_func, y0)
        ctx.save_for_backward(y_end, gamma)
        return y_end

    @staticmethod
    def backward(
        ctx: "NeuralODEFunc", *grad_output: torch.Tensor
    ) -> Tuple[None, torch.Tensor, None]:
        y_end, gamma = ctx.saved_tensors
        ode_solver = ctx.ode_solver

        @torch.jit.script
        def inverse_func(_, state: torch.Tensor):
            p, lam, _ = torch.unbind(state)
            p_gamma = p + gamma
            return torch.stack(
                [
                    -p
                    + torch.pow(torch.sin(p_gamma), 2)
                    + torch.pow(torch.e, -p) * torch.pow(torch.sin(p_gamma), 2),
                    lam
                    * (
                        1
                        - torch.pow(torch.e, -p)
                        * (
                            -torch.pow(torch.sin(p_gamma), 2)
                            + 2
                            * (torch.pow(torch.e, p) + 1)
                            * torch.sin(p_gamma)
                            * torch.cos(p_gamma)
                            - torch.pow(torch.e, p)
                        )
                    ),
                    -lam
                    * (
                        2
                        * torch.pow(torch.e, -p)
                        * (torch.pow(torch.e, p) + 1)
                        * torch.sin(p_gamma)
                        * torch.cos(p_gamma)
                    ),
                ]
            )

        initial_state = torch.stack([y_end, grad_output[0], torch.zeros_like(gamma)])

        final_state = ode_solver(inverse_func, initial_state, inverse=True)
        _, _, eta = torch.unbind(final_state)

        return None, eta, None


class NeuralMemoryODE(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, *, encoder: nn.Sequential):
        super().__init__()
        self.encoder = encoder
        self.ode_solver = ODESolver()
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.ode_func = NeuralODEFunc.apply

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.encoder(x)
        y0 = torch.zeros_like(gamma)
        features = self.ode_func(y0, gamma, self.ode_solver)
        return self.classifier(features)
