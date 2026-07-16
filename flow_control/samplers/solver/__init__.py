from typing import Annotated

from flow_control.utils.registry import RegistryUnion

from .base import BaseSolver, SolverState, StepResult, solver_registry
from .cps import CPSSolver
from .dance import DanceSolver
from .ddim import DDIMSolver
from .dpm import DPMSolver, DPMSolverState
from .flash import FlashSolver, FlashSolverState
from .flow import FlowSolver
from .unipc import FlowUniPCSolver, UniPCSolverState

Solver = Annotated[BaseSolver, RegistryUnion(solver_registry, "type")]

__all__ = [
    "BaseSolver",
    "CPSSolver",
    "DDIMSolver",
    "DPMSolver",
    "DPMSolverState",
    "DanceSolver",
    "FlashSolver",
    "FlashSolverState",
    "FlowSolver",
    "FlowUniPCSolver",
    "Solver",
    "SolverState",
    "StepResult",
    "UniPCSolverState",
    "solver_registry",
]
