#
"""Verification: DCP save/load with multiple optimizers on the same parameters.

Goal: Validate that torch.distributed.checkpoint can handle multiple optimizers
that all track the same model parameters. If this works, we can decouple EMA
from the base optimizer (no more mixin via make_ema_optimizer), and support
multiple sets of EMA weights simultaneously (useful for certain RL strategies).

Usage:
    uv run python examples/fsdp/multi_optim.py verify         # CPU: correctness
    uv run python examples/fsdp/multi_optim.py verify_dcp     # CPU: DCP save/load
    uv run python examples/fsdp/multi_optim.py fsdp_verify    # FSDP multi-GPU
"""

import os
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import fully_shard

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------------------------------
# Mock EMA optimizer: a standalone optimizer that just maintains EMA buffers
# without mixin. It wraps the same parameters but does NOT call a base
# optimizer's step -- it only reads the current param values and updates its
# internal EMA state.
# ---------------------------------------------------------------------------


class StandaloneEMAOptimizer(torch.optim.Optimizer):
    """A standalone EMA optimizer that maintains EMA buffers for tracked parameters.

    Unlike the mixin-based EMAOptimizer, this is a fully independent optimizer.
    It shares parameters with the main optimizer but only reads their values
    to update its own EMA state. It does NOT modify the parameters.

    This lets DCP treat it as a separate optimizer with its own state_dict.
    """

    def __init__(self, params, decay: float = 0.999):
        defaults = {"decay": decay}
        super().__init__(params, defaults)
        self._ema_step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Read current param values and update EMA buffers."""
        self._ema_step_count += 1
        for group in self.param_groups:
            decay = group["decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "ema_buffer" not in state:
                    state["ema_buffer"] = p.clone().to(torch.float32)
                else:
                    state["ema_buffer"].lerp_(p.to(torch.float32), 1 - decay)
        return None

    @torch.no_grad()
    def apply_shadow(self):
        """Replace parameters with their EMA values, saving current params as backup."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "ema_buffer" in state:
                    state["ema_backup"] = p.clone()
                    p.copy_(state["ema_buffer"].to(p.dtype))

    @torch.no_grad()
    def restore(self):
        """Restore parameters from backup after apply_shadow."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "ema_backup" in state:
                    p.copy_(state["ema_backup"])
                    del state["ema_backup"]


class DummyBaseOptimizer(torch.optim.Optimizer):
    """A trivial optimizer that sets params to fill_value * step_count.

    Makes expected values analytically computable.
    Writes a dummy entry to self.state so that DCP's _init_optim_state
    recognizes the optimizer as already initialized (real optimizers like
    AdamW always populate self.state on the first step).
    """

    def __init__(self, params, fill_value: float = 1.0):
        defaults = {"fill_value": fill_value}
        super().__init__(params, defaults)
        self._step_count_val = 0

    @torch.no_grad()
    def step(self, closure=None):
        self._step_count_val += 1
        for group in self.param_groups:
            fill = group["fill_value"] * self._step_count_val
            for p in group["params"]:
                if p.grad is not None:
                    p.fill_(fill)
                    # Write to self.state like real optimizers do,
                    # so _init_optim_state() sees non-empty state.
                    if "step" not in self.state[p]:
                        self.state[p]["step"] = torch.tensor(0.0)
                    self.state[p]["step"] += 1
        return None


def compute_expected_ema(fill_value: float, decay: float, num_steps: int) -> float:
    """Compute expected EMA value analytically.

    Step 1: param = v*1, ema initialized = v*1
    Step k (k>=2): param = v*k, ema = decay * prev_ema + (1-decay) * v*k
    """
    ema = fill_value * 1.0
    for k in range(2, num_steps + 1):
        param_val = fill_value * k
        ema = decay * ema + (1 - decay) * param_val
    return ema


# ---------------------------------------------------------------------------
# Stateful wrapper for DCP with multiple optimizers
# ---------------------------------------------------------------------------


class MultiOptimTrainingState(Stateful):
    """Wraps model + multiple optimizers for DCP save/load.

    Each optimizer gets its own key in the state dict.
    Uses get_optimizer_state_dict / set_optimizer_state_dict per optimizer
    to avoid FQN key collisions when multiple optimizers share the same params.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizers: dict[str, torch.optim.Optimizer],
    ):
        self.model = model
        self.optimizers = optimizers

    def state_dict(self):
        opts = StateDictOptions(strict=False)
        result: dict[str, dict] = {}
        # Save model state once
        result["model"] = get_model_state_dict(self.model, options=opts)
        # Save each optimizer state separately
        for name, opt in self.optimizers.items():
            result[f"optim_{name}"] = get_optimizer_state_dict(
                self.model, opt, options=opts
            )
        return result

    def load_state_dict(self, state_dict):
        opts = StateDictOptions(strict=False)
        set_model_state_dict(self.model, state_dict["model"], options=opts)
        for name, opt in self.optimizers.items():
            set_optimizer_state_dict(
                self.model, opt, state_dict[f"optim_{name}"], options=opts
            )


# ---------------------------------------------------------------------------
# CPU tests
# ---------------------------------------------------------------------------


def run_local_verify():
    """Verify that multiple optimizers on the same params work correctly."""
    from rich import print

    print("[bold]Running multi-optimizer correctness verification...[/bold]")

    fill_value = 1.0
    decay_fast = 0.9
    decay_slow = 0.99
    num_steps = 10

    model = nn.Linear(4, 4, bias=False)
    model.weight.data.fill_(0.0)

    # Two optimizers sharing the same parameters:
    # 1. A base optimizer that modifies parameters
    # 2. Two EMA optimizers with different decay rates
    base_opt = DummyBaseOptimizer(model.parameters(), fill_value=fill_value)
    ema_fast = StandaloneEMAOptimizer(model.parameters(), decay=decay_fast)
    ema_slow = StandaloneEMAOptimizer(model.parameters(), decay=decay_slow)

    for step in range(1, num_steps + 1):
        model.weight.grad = torch.ones_like(model.weight)

        # Base optimizer updates params
        base_opt.step()
        # EMA optimizers read updated params and update their buffers
        ema_fast.step()
        ema_slow.step()

        base_opt.zero_grad()

        expected_param = fill_value * step
        actual_param = model.weight.data.mean().item()
        assert abs(actual_param - expected_param) < 1e-5

        expected_ema_fast = compute_expected_ema(fill_value, decay_fast, step)
        actual_ema_fast = ema_fast.state[model.weight]["ema_buffer"].mean().item()
        assert abs(actual_ema_fast - expected_ema_fast) < 1e-5

        expected_ema_slow = compute_expected_ema(fill_value, decay_slow, step)
        actual_ema_slow = ema_slow.state[model.weight]["ema_buffer"].mean().item()
        assert abs(actual_ema_slow - expected_ema_slow) < 1e-5

        if step % 3 == 0 or step == num_steps:
            print(
                f"  Step {step}: param={actual_param:.4f}, "
                f"ema_fast={actual_ema_fast:.4f} (exp {expected_ema_fast:.4f}), "
                f"ema_slow={actual_ema_slow:.4f} (exp {expected_ema_slow:.4f})"
            )

    # Verify apply_shadow / restore for each EMA
    param_before = model.weight.data.clone()

    ema_fast.apply_shadow()
    ema_val_fast = model.weight.data.mean().item()
    expected_fast = compute_expected_ema(fill_value, decay_fast, num_steps)
    assert abs(ema_val_fast - expected_fast) < 1e-5
    ema_fast.restore()
    assert torch.allclose(param_before, model.weight.data)

    ema_slow.apply_shadow()
    ema_val_slow = model.weight.data.mean().item()
    expected_slow = compute_expected_ema(fill_value, decay_slow, num_steps)
    assert abs(ema_val_slow - expected_slow) < 1e-5
    ema_slow.restore()
    assert torch.allclose(param_before, model.weight.data)

    print("  apply_shadow / restore round-trip OK for both EMA optimizers")
    print("[green]Multi-optimizer correctness verification passed.[/green]")


def _dcp_save_worker(ckpt_path: str, result_queue: mp.Queue):
    """Subprocess: train save_at_step steps, save checkpoint, send expected values back."""
    fill_value = 1.0
    decay_fast = 0.9
    decay_slow = 0.99
    save_at_step = 5

    model = nn.Linear(4, 4, bias=False)
    model.weight.data.fill_(0.0)
    base_opt = DummyBaseOptimizer(model.parameters(), fill_value=fill_value)
    ema_fast = StandaloneEMAOptimizer(model.parameters(), decay=decay_fast)
    ema_slow = StandaloneEMAOptimizer(model.parameters(), decay=decay_slow)

    for _step in range(1, save_at_step + 1):
        model.weight.grad = torch.ones_like(model.weight)
        base_opt.step()
        ema_fast.step()
        ema_slow.step()
        base_opt.zero_grad()

    opts = StateDictOptions(strict=False)
    state = {
        "model": get_model_state_dict(model, options=opts),
        "optim_base": get_optimizer_state_dict(model, base_opt, options=opts),
        "optim_ema_fast": get_optimizer_state_dict(model, ema_fast, options=opts),
        "optim_ema_slow": get_optimizer_state_dict(model, ema_slow, options=opts),
        "base_step_count": base_opt._step_count_val,
        "ema_fast_step_count": ema_fast._ema_step_count,
        "ema_slow_step_count": ema_slow._ema_step_count,
    }
    dcp.save(state, checkpoint_id=ckpt_path)

    # Send expected values to the parent process
    result_queue.put(
        {
            "expected_param": fill_value * save_at_step,
            "expected_ema_fast": compute_expected_ema(
                fill_value, decay_fast, save_at_step
            ),
            "expected_ema_slow": compute_expected_ema(
                fill_value, decay_slow, save_at_step
            ),
        }
    )


def _dcp_load_worker(ckpt_path: str, expected: dict, result_queue: mp.Queue):
    """Subprocess: load checkpoint from disk, continue training, verify correctness."""
    fill_value = 1.0
    decay_fast = 0.9
    decay_slow = 0.99
    save_at_step = 5
    total_steps = 10

    model = nn.Linear(4, 4, bias=False)
    model.weight.data.fill_(999.0)
    base_opt = DummyBaseOptimizer(model.parameters(), fill_value=fill_value)
    ema_fast = StandaloneEMAOptimizer(model.parameters(), decay=decay_fast)
    ema_slow = StandaloneEMAOptimizer(model.parameters(), decay=decay_slow)

    # Initialize optimizer states by running one step
    model.weight.grad = torch.ones_like(model.weight)
    base_opt.step()
    ema_fast.step()
    ema_slow.step()
    base_opt.zero_grad()

    opts = StateDictOptions(strict=False)
    load_state = {
        "model": get_model_state_dict(model, options=opts),
        "optim_base": get_optimizer_state_dict(model, base_opt, options=opts),
        "optim_ema_fast": get_optimizer_state_dict(model, ema_fast, options=opts),
        "optim_ema_slow": get_optimizer_state_dict(model, ema_slow, options=opts),
        "base_step_count": 0,
        "ema_fast_step_count": 0,
        "ema_slow_step_count": 0,
    }
    dcp.load(load_state, checkpoint_id=ckpt_path)

    set_model_state_dict(model, load_state["model"], options=opts)
    set_optimizer_state_dict(model, base_opt, load_state["optim_base"], options=opts)
    set_optimizer_state_dict(
        model, ema_fast, load_state["optim_ema_fast"], options=opts
    )
    set_optimizer_state_dict(
        model, ema_slow, load_state["optim_ema_slow"], options=opts
    )
    base_opt._step_count_val = load_state["base_step_count"]
    ema_fast._ema_step_count = load_state["ema_fast_step_count"]
    ema_slow._ema_step_count = load_state["ema_slow_step_count"]

    # Verify loaded state matches expected values from the save process
    loaded_param = model.weight.data.mean().item()
    loaded_ema_fast = ema_fast.state[model.weight]["ema_buffer"].mean().item()
    loaded_ema_slow = ema_slow.state[model.weight]["ema_buffer"].mean().item()

    errors: list[str] = []
    if abs(loaded_param - expected["expected_param"]) >= 1e-5:
        errors.append(f"Param mismatch: {loaded_param} vs {expected['expected_param']}")
    if abs(loaded_ema_fast - expected["expected_ema_fast"]) >= 1e-5:
        errors.append(
            f"EMA fast mismatch: {loaded_ema_fast} vs {expected['expected_ema_fast']}"
        )
    if abs(loaded_ema_slow - expected["expected_ema_slow"]) >= 1e-5:
        errors.append(
            f"EMA slow mismatch: {loaded_ema_slow} vs {expected['expected_ema_slow']}"
        )

    # Continue training
    for _step in range(save_at_step + 1, total_steps + 1):
        model.weight.grad = torch.ones_like(model.weight)
        base_opt.step()
        ema_fast.step()
        ema_slow.step()
        base_opt.zero_grad()

    expected_param_final = fill_value * total_steps
    expected_ema_fast_final = compute_expected_ema(fill_value, decay_fast, total_steps)
    expected_ema_slow_final = compute_expected_ema(fill_value, decay_slow, total_steps)
    actual_param = model.weight.data.mean().item()
    actual_ema_fast = ema_fast.state[model.weight]["ema_buffer"].mean().item()
    actual_ema_slow = ema_slow.state[model.weight]["ema_buffer"].mean().item()

    if abs(actual_param - expected_param_final) >= 1e-5:
        errors.append(
            f"Resume param mismatch: {actual_param} vs {expected_param_final}"
        )
    if abs(actual_ema_fast - expected_ema_fast_final) >= 1e-5:
        errors.append(
            f"Resume EMA fast mismatch: {actual_ema_fast} vs {expected_ema_fast_final}"
        )
    if abs(actual_ema_slow - expected_ema_slow_final) >= 1e-5:
        errors.append(
            f"Resume EMA slow mismatch: {actual_ema_slow} vs {expected_ema_slow_final}"
        )

    result_queue.put(
        {
            "errors": errors,
            "loaded_param": loaded_param,
            "loaded_ema_fast": loaded_ema_fast,
            "loaded_ema_slow": loaded_ema_slow,
            "final_param": actual_param,
            "final_ema_fast": actual_ema_fast,
            "final_ema_slow": actual_ema_slow,
            "expected_param_final": expected_param_final,
            "expected_ema_fast_final": expected_ema_fast_final,
            "expected_ema_slow_final": expected_ema_slow_final,
        }
    )


def run_local_dcp_verify():
    """Verify DCP save/load across separate processes preserves multi-optimizer state."""
    from rich import print

    print(
        "[bold]Running multi-optimizer DCP cross-process save/load verification...[/bold]"
    )

    tmpdir = tempfile.mkdtemp(prefix="multi_optim_dcp_test_")
    ctx = mp.get_context("spawn")

    try:
        ckpt_path = os.path.join(tmpdir, "ckpt")

        # Phase 1: save in a child process
        save_queue: mp.Queue = ctx.Queue()
        save_proc = ctx.Process(target=_dcp_save_worker, args=(ckpt_path, save_queue))
        save_proc.start()
        save_proc.join()
        assert save_proc.exitcode == 0, (
            f"Save process failed with exit code {save_proc.exitcode}"
        )

        expected = save_queue.get()
        print(
            f"  [save process] step 5: param={expected['expected_param']:.4f}, "
            f"ema_fast={expected['expected_ema_fast']:.4f}, "
            f"ema_slow={expected['expected_ema_slow']:.4f}"
        )

        # Phase 2: load in a different child process
        load_queue: mp.Queue = ctx.Queue()
        load_proc = ctx.Process(
            target=_dcp_load_worker, args=(ckpt_path, expected, load_queue)
        )
        load_proc.start()
        load_proc.join()
        assert load_proc.exitcode == 0, (
            f"Load process failed with exit code {load_proc.exitcode}"
        )

        result = load_queue.get()
        if result["errors"]:
            for err in result["errors"]:
                print(f"  [red]ERROR: {err}[/red]")
            raise AssertionError(
                "Cross-process DCP verification failed: " + "; ".join(result["errors"])
            )

        print(
            f"  [load process] after load: param={result['loaded_param']:.4f}, "
            f"ema_fast={result['loaded_ema_fast']:.4f}, "
            f"ema_slow={result['loaded_ema_slow']:.4f}"
        )
        print(
            f"  [load process] after resume (step 10): "
            f"param={result['final_param']:.4f} (exp {result['expected_param_final']:.4f}), "
            f"ema_fast={result['final_ema_fast']:.4f} (exp {result['expected_ema_fast_final']:.4f}), "
            f"ema_slow={result['final_ema_slow']:.4f} (exp {result['expected_ema_slow_final']:.4f})"
        )
        print(
            "[green]Multi-optimizer DCP cross-process save/load verification passed.[/green]"
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# FSDP multi-GPU tests
# ---------------------------------------------------------------------------


def setup_process(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _assert_close_local(t: torch.Tensor, expected_val: float, atol: float = 1e-5):
    local = t.to_local() if hasattr(t, "to_local") else t
    expected = torch.full_like(local, expected_val)
    if not torch.allclose(local, expected, atol=atol):
        raise AssertionError(
            f"Expected all elements ~ {expected_val}, got mean={local.mean().item():.6f}, "
            f"max_diff={(local - expected).abs().max().item():.6f}"
        )


def _fsdp_verify_worker(rank, world_size, tmpdir):  # noqa: C901
    """FSDP verification: multiple optimizers on same params + DCP save/load."""
    setup_process(rank, world_size)

    fill_value = 1.0
    decay_fast = 0.9
    decay_slow = 0.99
    save_at_step = 5
    total_steps = 10

    def _log(msg):
        if rank == 0:
            print(msg)  # noqa: T201

    # --- Phase 1: create model + multiple optimizers, train, save ---
    model = ToyModel().cuda(rank)
    for p in model.parameters():
        p.data.zero_()
    fully_shard(model)
    trainable = list(model.parameters())

    base_opt = DummyBaseOptimizer(trainable, fill_value=fill_value)
    ema_fast = StandaloneEMAOptimizer(trainable, decay=decay_fast)
    ema_slow = StandaloneEMAOptimizer(trainable, decay=decay_slow)

    for _step in range(1, save_at_step + 1):
        for p in trainable:
            p.grad = torch.ones_like(p)
        base_opt.step()
        ema_fast.step()
        ema_slow.step()
        base_opt.zero_grad()

    expected_param = fill_value * save_at_step
    expected_ema_fast = compute_expected_ema(fill_value, decay_fast, save_at_step)
    expected_ema_slow = compute_expected_ema(fill_value, decay_slow, save_at_step)

    for p in trainable:
        _assert_close_local(p.data, expected_param)
        _assert_close_local(ema_fast.state[p]["ema_buffer"], expected_ema_fast)
        _assert_close_local(ema_slow.state[p]["ema_buffer"], expected_ema_slow)
    _log(
        f"  Phase 1 OK: step {save_at_step}, param={expected_param}, "
        f"ema_fast={expected_ema_fast:.4f}, ema_slow={expected_ema_slow:.4f}"
    )

    # Save via DCP using MultiOptimTrainingState
    ckpt_path = os.path.join(tmpdir, "fsdp_ckpt")
    training_state = MultiOptimTrainingState(
        model, {"base": base_opt, "ema_fast": ema_fast, "ema_slow": ema_slow}
    )
    save_sd = {
        "training": training_state.state_dict(),
        "base_step_count": base_opt._step_count_val,
    }
    dcp.save(save_sd, checkpoint_id=ckpt_path)
    _log(f"  Saved FSDP DCP checkpoint to {ckpt_path}")
    dist.barrier()

    # --- Phase 2: fresh model, load, resume, verify ---
    model2 = ToyModel().cuda(rank)
    for p in model2.parameters():
        p.data.fill_(999.0)
    fully_shard(model2)
    trainable2 = list(model2.parameters())

    base_opt2 = DummyBaseOptimizer(trainable2, fill_value=fill_value)
    ema_fast2 = StandaloneEMAOptimizer(trainable2, decay=decay_fast)
    ema_slow2 = StandaloneEMAOptimizer(trainable2, decay=decay_slow)

    # Initialize optimizer states
    for p in trainable2:
        p.grad = torch.ones_like(p)
    base_opt2.step()
    ema_fast2.step()
    ema_slow2.step()
    base_opt2.zero_grad()

    training_state2 = MultiOptimTrainingState(
        model2, {"base": base_opt2, "ema_fast": ema_fast2, "ema_slow": ema_slow2}
    )
    load_sd = {
        "training": training_state2.state_dict(),
        "base_step_count": 0,
    }
    dcp.load(load_sd, checkpoint_id=ckpt_path)
    training_state2.load_state_dict(load_sd["training"])
    base_opt2._step_count_val = load_sd["base_step_count"]

    for p in trainable2:
        _assert_close_local(p.data, expected_param)
        _assert_close_local(ema_fast2.state[p]["ema_buffer"], expected_ema_fast)
        _assert_close_local(ema_slow2.state[p]["ema_buffer"], expected_ema_slow)
    _log("  Phase 2: checkpoint loaded and verified OK")

    # Continue training
    for _step in range(save_at_step + 1, total_steps + 1):
        for p in trainable2:
            p.grad = torch.ones_like(p)
        base_opt2.step()
        ema_fast2.step()
        ema_slow2.step()
        base_opt2.zero_grad()

    expected_param_final = fill_value * total_steps
    expected_ema_fast_final = compute_expected_ema(fill_value, decay_fast, total_steps)
    expected_ema_slow_final = compute_expected_ema(fill_value, decay_slow, total_steps)

    for p in trainable2:
        _assert_close_local(p.data, expected_param_final)
        _assert_close_local(ema_fast2.state[p]["ema_buffer"], expected_ema_fast_final)
        _assert_close_local(ema_slow2.state[p]["ema_buffer"], expected_ema_slow_final)

    _log(
        f"  Phase 2 OK: step {total_steps}, param={expected_param_final}, "
        f"ema_fast={expected_ema_fast_final:.4f}, ema_slow={expected_ema_slow_final:.4f}"
    )
    _log("[PASS] FSDP multi-optimizer + DCP verification passed.")
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["verify", "verify_dcp", "fsdp_verify"],
    )
    args = parser.parse_args()

    if args.mode == "verify":
        run_local_verify()
    elif args.mode == "verify_dcp":
        run_local_dcp_verify()
    elif args.mode == "fsdp_verify":
        tmpdir = tempfile.mkdtemp(prefix="multi_optim_fsdp_test_")
        try:
            world_size = torch.cuda.device_count()
            mp.spawn(_fsdp_verify_worker, args=(world_size, tmpdir), nprocs=world_size)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
