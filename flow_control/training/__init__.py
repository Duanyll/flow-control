"""Training package.

Kept torch-free *at import time*: importing any submodule (e.g.
``flow_control.training.launch_config``, which the torch-free launch parent
imports) runs this ``__init__``, so it must not pull torch. ``import_builtin_trainers``
therefore only *defines* the imports; they run when it is called, in the child /
command processes.
"""


def import_builtin_trainers() -> None:
    """Import the built-in trainer and objective modules so they register.

    Trainers land in ``trainer_registry``, objectives in ``objective_registry``
    (``weighted_fm`` has no trainer of its own).

    Called by the training entry points (``launch._run_child``, ``seed``,
    ``export``, ``schema``) before resolving a trainer; plugin trainers register
    via a config's ``imports``. Importing the handful of built-ins on demand is
    cheap — heavy backends (e.g. vLLM) are imported lazily inside the trainers.
    """
    import importlib

    for name in ("sft", "grpo", "nft", "awm", "ram", "weighted_fm", "inference"):
        importlib.import_module(f"flow_control.training.{name}")
