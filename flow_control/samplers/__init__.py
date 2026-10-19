import hashlib

from .calls import Calls, ModelCall, gather
from .executor import Executor
from .multistep import DpmGrid, SaGrid, UniPcGrid
from .node import (
    BaseNode,
    Frame,
    Observer,
    RootFrame,
    Row,
    caller,
    children,
    data,
    derived,
    follow,
    randn,
    walk,
)
from .prediction import (
    BasePrediction,
    Cfg,
    CfgPlusPlus,
    Model,
    Prediction,
    Tiled,
    prediction_registry,
    variants,
)
from .shift import BaseShift, Shift, shift_registry
from .trajectory import (
    BaseGrid,
    BaseTrajectory,
    Grid,
    GridFrame,
    InitLatent,
    Renoise,
    SdeWindowGrid,
    Sequential,
    Trajectory,
    select_sde_window,
    trajectory_registry,
)
from .transition import (
    BaseTransition,
    Cps,
    Dance,
    Ddim,
    DifferentialDiffusion,
    Euler,
    Flash,
    GaussianTransition,
    StepFrame,
    StepLogProb,
    Transition,
    normal_log_prob,
    transition_registry,
)


def derive_seed(base_seed: int, key: str) -> int:
    """A deterministic per-sample seed from a base seed and a sample key."""
    h = hashlib.sha256(f"{base_seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**63)


__all__ = [
    "BaseGrid",
    "BaseNode",
    "BasePrediction",
    "BaseShift",
    "BaseTrajectory",
    "BaseTransition",
    "Calls",
    "Cfg",
    "CfgPlusPlus",
    "Cps",
    "Dance",
    "Ddim",
    "DifferentialDiffusion",
    "DpmGrid",
    "Euler",
    "Executor",
    "Flash",
    "Frame",
    "GaussianTransition",
    "Grid",
    "GridFrame",
    "InitLatent",
    "Model",
    "ModelCall",
    "Observer",
    "Prediction",
    "Renoise",
    "RootFrame",
    "Row",
    "SaGrid",
    "SdeWindowGrid",
    "Sequential",
    "Shift",
    "StepFrame",
    "StepLogProb",
    "Tiled",
    "Trajectory",
    "Transition",
    "UniPcGrid",
    "caller",
    "children",
    "data",
    "derive_seed",
    "derived",
    "follow",
    "gather",
    "normal_log_prob",
    "prediction_registry",
    "randn",
    "select_sde_window",
    "shift_registry",
    "trajectory_registry",
    "transition_registry",
    "variants",
    "walk",
]
