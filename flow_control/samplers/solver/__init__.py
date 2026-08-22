from typing import Annotated

from flow_control.utils.registry import RegistryUnion

from .base import BaseSolver, solver_registry
from .cps import CPSSolver
from .dance import DanceSolver
from .ddim import DDIMSolver
from .dpm import DPMSolver
from .flash import FlashSolver
from .flow import FlowSolver
from .sa import SASolver
from .unipc import FlowUniPCSolver

Solver = Annotated[BaseSolver, RegistryUnion(solver_registry, "type")]

__all__ = [
    "BaseSolver",
    "CPSSolver",
    "DDIMSolver",
    "DPMSolver",
    "DanceSolver",
    "FlashSolver",
    "FlowSolver",
    "FlowUniPCSolver",
    "SASolver",
    "Solver",
    "solver_registry",
]
