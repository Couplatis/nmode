from typing import Literal


SolverType = Literal[
    "dopri8",
    "dopri5",
    "bosh3",
    "fehlberg2",
    "adaptive_heun",
    "euler",
    "midpoint",
    "heun2",
    "heun3",
    "rk4",
    "explicit_adams",
    "implicit_adams",
    "fixed_adams",
    "scipy_solver",
]
