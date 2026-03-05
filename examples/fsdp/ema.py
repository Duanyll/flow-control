"""Verification tests for EMA optimizer: correctness, DCP save/load, and FSDP multi-GPU.

Usage:
    uv run python examples/fsdp/ema.py verify              # CPU: EMA correctness with known values
    uv run python examples/fsdp/ema.py verify_init_backup   # CPU: init_backup correctness
    uv run python examples/fsdp/ema.py verify_dcp           # CPU: DCP save/load preserves EMA state
    uv run python examples/fsdp/ema.py fsdp_verify          # FSDP 4-GPU: EMA + DCP + init_backup
    uv run python examples/fsdp/ema.py local                # CPU: smoke test with real AdamW
    uv run python examples/fsdp/ema.py fsdp                 # FSDP: smoke test with real AdamW
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
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import fully_shard

from flow_control.utils.ema import (
    apply_ema_maybe,
    apply_init_maybe,
    make_ema_optimizer,
)

# ---------------------------------------------------------------------------
# Shared model / LoRA / helpers
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


class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=4):
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(x))


def install_lora_layers(model, r=4):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lora_module = LoRALinear(module, r=r)
            parent_module = model
            name_parts = name.split(".")
            for part in name_parts[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name_parts[-1], lora_module)


class DummyOptimizer(torch.optim.Optimizer):
    """A trivial optimizer that sets each parameter to a known value on each step.

    After step ``k`` (1-indexed), every trainable parameter is set to ``fill_value * k``.
    This makes the expected EMA values analytically computable.
    """

    def __init__(self, params, fill_value: float = 1.0, **kwargs):
        defaults = {"fill_value": fill_value}
        defaults.update(kwargs)
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        self._step_count += 1
        for group in self.param_groups:
            fill = group["fill_value"] * self._step_count
            for p in group["params"]:
                if p.grad is not None:
                    p.fill_(fill)
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


class EMATrainingState(Stateful):
    """Wraps model + EMA optimizer for DCP save/load, following the pattern in dcp.py."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        model_sd, optim_sd = get_state_dict(
            self.model,
            [self.optimizer],
            options=StateDictOptions(strict=False),
        )
        return {"model": model_sd, "optim": optim_sd}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            [self.optimizer],
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
            options=StateDictOptions(strict=False),
        )


# ---------------------------------------------------------------------------
# CPU tests
# ---------------------------------------------------------------------------


def run_local_ema_test():
    """Smoke test: EMA with real AdamW on CPU."""
    from rich import print

    print("[bold]Running local EMA smoke test...[/bold]")
    model = ToyModel()
    model.requires_grad_(False)
    install_lora_layers(model, r=4)
    optimizer_cls = make_ema_optimizer(torch.optim.AdamW)
    optimizer = optimizer_cls(model.parameters(), lr=1e-3, ema_decay=0.999)

    for _ in range(10):
        inputs = torch.randn(16, 128)
        outputs = model(inputs)
        loss = outputs.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with apply_ema_maybe(optimizer):
        model(torch.randn(16, 128))

    print("[green]Local EMA smoke test passed.[/green]")


def run_local_ema_verify():
    """Verify EMA computation correctness with known values using DummyOptimizer."""
    from rich import print

    print("[bold]Running local EMA correctness verification...[/bold]")

    fill_value = 1.0
    decay = 0.9
    num_steps = 5

    model = nn.Linear(4, 4, bias=False)
    model.weight.data.fill_(0.0)

    EMADummy = make_ema_optimizer(DummyOptimizer)
    optimizer = EMADummy(model.parameters(), fill_value=fill_value, ema_decay=decay)

    for step in range(1, num_steps + 1):
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        optimizer.zero_grad()

        expected_param = fill_value * step
        actual_param = model.weight.data.mean().item()
        assert abs(actual_param - expected_param) < 1e-5, (
            f"Step {step}: param mismatch: expected {expected_param}, got {actual_param}"
        )

        expected_ema = compute_expected_ema(fill_value, decay, step)
        actual_ema = optimizer.state[model.weight]["ema_buffer"].mean().item()
        assert abs(actual_ema - expected_ema) < 1e-5, (
            f"Step {step}: EMA mismatch: expected {expected_ema}, got {actual_ema}"
        )
        print(
            f"  Step {step}: param={actual_param:.4f} (exp {expected_param:.4f}), "
            f"ema={actual_ema:.4f} (exp {expected_ema:.4f})"
        )

    # Verify apply_shadow / restore round-trip
    param_before = model.weight.data.clone()
    with apply_ema_maybe(optimizer):
        ema_val = model.weight.data.mean().item()
        expected_ema = compute_expected_ema(fill_value, decay, num_steps)
        assert abs(ema_val - expected_ema) < 1e-5
    assert torch.allclose(param_before, model.weight.data), (
        "restore failed after apply_shadow"
    )

    print("[green]Local EMA correctness verification passed.[/green]")


def run_local_init_backup_verify():
    """Verify init_backup (reference model restore) correctness."""
    from rich import print

    print("[bold]Running local init_backup correctness verification...[/bold]")

    fill_value = 1.0
    decay = 0.9
    num_steps = 5

    model = nn.Linear(4, 4, bias=False)
    model.weight.data.fill_(42.0)

    EMADummy = make_ema_optimizer(DummyOptimizer)
    optimizer = EMADummy(
        model.parameters(),
        fill_value=fill_value,
        ema_decay=decay,
        enable_init_backup=True,
    )

    # First step triggers init backup (saved BEFORE the base optimizer step)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()

    init_backup = optimizer.state[model.weight]["init_backup"]
    assert abs(init_backup.mean().item() - 42.0) < 1e-5
    print(f"  Init backup value: {init_backup.mean().item():.4f} (expected 42.0)")

    for _step in range(2, num_steps + 1):
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step()
        optimizer.zero_grad()

    expected_current = fill_value * num_steps
    print(f"  Current param: {model.weight.data.mean().item():.4f}")

    with apply_init_maybe(optimizer):
        restored = model.weight.data.mean().item()
        print(f"  Restored param (init backup): {restored:.4f}")
        assert abs(restored - 42.0) < 1e-5

    after_restore = model.weight.data.mean().item()
    assert abs(after_restore - expected_current) < 1e-5
    print(f"  Param after restore: {after_restore:.4f}")

    print("[green]Local init_backup correctness verification passed.[/green]")


def run_local_dcp_verify():
    """Verify that DCP save/load preserves EMA state and continued training is correct."""
    from rich import print

    print("[bold]Running local DCP save/load verification...[/bold]")

    fill_value = 1.0
    decay = 0.9
    save_at_step = 3
    total_steps = 6
    tmpdir = tempfile.mkdtemp(prefix="ema_dcp_test_")

    try:
        # --- Phase 1: train save_at_step steps, then save ---
        model = nn.Linear(4, 4, bias=False)
        model.weight.data.fill_(0.0)
        EMADummy = make_ema_optimizer(DummyOptimizer)
        optimizer = EMADummy(
            model.parameters(),
            fill_value=fill_value,
            ema_decay=decay,
            enable_init_backup=True,
        )

        for _step in range(1, save_at_step + 1):
            model.weight.grad = torch.ones_like(model.weight)
            optimizer.step()
            optimizer.zero_grad()

        ema_at_save = optimizer.state[model.weight]["ema_buffer"].clone()
        init_at_save = optimizer.state[model.weight]["init_backup"].clone()
        expected_ema_at_save = compute_expected_ema(fill_value, decay, save_at_step)
        assert abs(ema_at_save.mean().item() - expected_ema_at_save) < 1e-5
        print(
            f"  Before save (step {save_at_step}): ema={ema_at_save.mean().item():.4f}"
        )

        # Save via DCP (non-distributed, single-process)
        ckpt_path = os.path.join(tmpdir, "ckpt")
        state = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "step_count": optimizer._step_count,
        }
        dcp.save(state, checkpoint_id=ckpt_path)
        print(f"  Saved checkpoint to {ckpt_path}")

        # --- Phase 2: create fresh model+optimizer, load, continue training ---
        model2 = nn.Linear(4, 4, bias=False)
        model2.weight.data.fill_(999.0)  # garbage value to prove load works
        optimizer2 = EMADummy(
            model2.parameters(),
            fill_value=fill_value,
            ema_decay=decay,
            enable_init_backup=True,
        )
        # Need to initialize optimizer state structure before loading
        # (DCP loads into existing state dict structure)
        model2.weight.grad = torch.ones_like(model2.weight)
        optimizer2.step()
        optimizer2.zero_grad()

        load_state = {
            "model": model2.state_dict(),
            "optim": optimizer2.state_dict(),
            "step_count": 0,
        }
        dcp.load(load_state, checkpoint_id=ckpt_path)
        model2.load_state_dict(load_state["model"])
        optimizer2.load_state_dict(load_state["optim"])
        optimizer2._step_count = load_state["step_count"]
        # Already had first step, so disable first-step init backup
        optimizer2._first_step = False

        # Verify loaded state matches
        loaded_ema = optimizer2.state[model2.weight]["ema_buffer"]
        assert torch.allclose(loaded_ema, ema_at_save), (
            f"EMA mismatch after load: {loaded_ema.mean().item()} vs {ema_at_save.mean().item()}"
        )
        loaded_init = optimizer2.state[model2.weight]["init_backup"]
        assert torch.allclose(loaded_init, init_at_save), (
            f"init_backup mismatch after load: {loaded_init.mean().item()} vs {init_at_save.mean().item()}"
        )
        loaded_param = model2.weight.data.mean().item()
        expected_param_at_save = fill_value * save_at_step
        assert abs(loaded_param - expected_param_at_save) < 1e-5
        print(
            f"  After load: param={loaded_param:.4f}, "
            f"ema={loaded_ema.mean().item():.4f}, init={loaded_init.mean().item():.4f}"
        )

        # Continue training from step save_at_step+1 to total_steps
        for _step in range(save_at_step + 1, total_steps + 1):
            model2.weight.grad = torch.ones_like(model2.weight)
            optimizer2.step()
            optimizer2.zero_grad()

        actual_ema = optimizer2.state[model2.weight]["ema_buffer"].mean().item()
        expected_ema = compute_expected_ema(fill_value, decay, total_steps)
        actual_param = model2.weight.data.mean().item()
        expected_param = fill_value * total_steps

        print(
            f"  After resume (step {total_steps}): param={actual_param:.4f} (exp {expected_param:.4f}), "
            f"ema={actual_ema:.4f} (exp {expected_ema:.4f})"
        )
        assert abs(actual_param - expected_param) < 1e-5, (
            f"Param mismatch: {actual_param} vs {expected_param}"
        )
        assert abs(actual_ema - expected_ema) < 1e-5, (
            f"EMA mismatch: {actual_ema} vs {expected_ema}"
        )

        # Verify init_backup survived the save/load cycle
        with apply_init_maybe(optimizer2):
            restored = model2.weight.data.mean().item()
            assert abs(restored - 0.0) < 1e-5, (
                f"init_backup should be 0.0 (original value), got {restored}"
            )

        print("[green]Local DCP save/load verification passed.[/green]")

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


def run_fsdp_ema_test(rank, world_size):
    """Smoke test: FSDP + EMA with real AdamW."""
    setup_process(rank, world_size)
    model = ToyModel().cuda(rank)
    model.requires_grad_(False)
    install_lora_layers(model, r=4)
    model: nn.Module = fully_shard(model)  # type: ignore

    optimizer_cls = make_ema_optimizer(torch.optim.AdamW)
    optimizer = optimizer_cls(model.parameters(), lr=1e-3, ema_decay=0.999)

    for _ in range(10):
        inputs = torch.randn(16, 128).cuda(rank)
        outputs = model(inputs)
        loss = outputs.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with apply_ema_maybe(optimizer):
        model(torch.randn(16, 128).cuda(rank))

    dist.destroy_process_group()


def _assert_close_local(t: torch.Tensor, expected_val: float, atol: float = 1e-5):
    """Assert that all elements of a (possibly sharded) tensor are close to expected_val.

    DTensors from FSDP don't support `torch.allclose`, so we extract the local
    shard first with `to_local()`.
    """
    local = t.to_local() if hasattr(t, "to_local") else t
    expected = torch.full_like(local, expected_val)
    if not torch.allclose(local, expected, atol=atol):
        raise AssertionError(
            f"Expected all elements ≈ {expected_val}, got mean={local.mean().item():.6f}, "
            f"max_diff={(local - expected).abs().max().item():.6f}"
        )


def _clone_local(t: torch.Tensor) -> torch.Tensor:
    """Clone a (possibly sharded) tensor to a plain local tensor."""
    local = t.to_local() if hasattr(t, "to_local") else t
    return local.clone()


def _assert_tensors_equal(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5):
    """Assert two (possibly sharded) tensors are element-wise close."""
    a_local = a.to_local() if hasattr(a, "to_local") else a
    b_local = b.to_local() if hasattr(b, "to_local") else b
    if not torch.allclose(a_local, b_local, atol=atol):
        raise AssertionError(
            f"Tensors not close: max_diff={(a_local - b_local).abs().max().item():.6f}"
        )


def _make_fsdp_model_and_optimizer(rank, fill_value, decay, enable_init_backup=True):
    """Create a sharded ToyModel with LoRA and an EMA DummyOptimizer."""
    model = ToyModel().cuda(rank)
    model.requires_grad_(False)
    install_lora_layers(model, r=4)
    # Fill all trainable (LoRA) params with zeros for deterministic verification
    for p in model.parameters():
        if p.requires_grad:
            p.data.zero_()
    fully_shard(model)
    EMADummy = make_ema_optimizer(DummyOptimizer)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = EMADummy(
        trainable,
        fill_value=fill_value,
        ema_decay=decay,
        enable_init_backup=enable_init_backup,
    )
    return model, optimizer, trainable


def _fsdp_verify_phase1(
    rank, trainable, optimizer, save_at_step, fill_value, decay, tmpdir, _log
):
    """Train save_at_step steps, verify EMA values and round-trips, save DCP."""
    for _step in range(1, save_at_step + 1):
        for p in trainable:
            p.grad = torch.ones_like(p)
        optimizer.step()
        optimizer.zero_grad()

    expected_ema = compute_expected_ema(fill_value, decay, save_at_step)
    expected_param = fill_value * save_at_step
    for p in trainable:
        state = optimizer.state[p]
        _assert_close_local(state["ema_buffer"], expected_ema)
        _assert_close_local(p.data, expected_param)
        _assert_close_local(state["init_backup"], 0.0)
    _log(
        f"  Phase 1 OK: step {save_at_step}, param={expected_param}, ema={expected_ema:.4f}"
    )

    # Verify apply_shadow round-trip
    param_snapshots = [_clone_local(p.data) for p in trainable]
    with apply_ema_maybe(optimizer):
        for p in trainable:
            _assert_close_local(p.data, expected_ema)
    for p, snap in zip(trainable, param_snapshots, strict=True):
        _assert_tensors_equal(p.data, snap)
    _log("  apply_shadow / restore round-trip OK")

    # Verify init_backup round-trip
    with apply_init_maybe(optimizer):
        for p in trainable:
            _assert_close_local(p.data, 0.0)
    for p, snap in zip(trainable, param_snapshots, strict=True):
        _assert_tensors_equal(p.data, snap)
    _log("  apply_init / restore round-trip OK")

    return expected_ema, expected_param


def _fsdp_verify_phase2(
    rank,
    model2,
    optimizer2,
    trainable2,
    save_at_step,
    total_steps,
    fill_value,
    decay,
    expected_ema,
    expected_param,
    ckpt_path,
    _log,
):
    """Load DCP into fresh model, continue training, verify EMA is still correct."""
    # Initialize optimizer state by running one dummy step (so state structure exists)
    for p in trainable2:
        p.grad = torch.ones_like(p)
    optimizer2.step()
    optimizer2.zero_grad()

    # Load checkpoint
    ema_state2 = EMATrainingState(model2, optimizer2)
    load_sd = {"training": ema_state2.state_dict(), "step_count": 0}
    dcp.load(load_sd, checkpoint_id=ckpt_path)
    ema_state2.load_state_dict(load_sd["training"])
    optimizer2._step_count = load_sd["step_count"]
    optimizer2._first_step = False

    for p in trainable2:
        state = optimizer2.state[p]
        _assert_close_local(state["ema_buffer"], expected_ema)
        _assert_close_local(p.data, expected_param)
        _assert_close_local(state["init_backup"], 0.0)
    _log("  Phase 2: checkpoint loaded and verified OK")

    # Continue training
    for _step in range(save_at_step + 1, total_steps + 1):
        for p in trainable2:
            p.grad = torch.ones_like(p)
        optimizer2.step()
        optimizer2.zero_grad()

    expected_ema_final = compute_expected_ema(fill_value, decay, total_steps)
    expected_param_final = fill_value * total_steps
    for p in trainable2:
        state = optimizer2.state[p]
        _assert_close_local(state["ema_buffer"], expected_ema_final)
        _assert_close_local(p.data, expected_param_final)
    _log(
        f"  Phase 2 OK: step {total_steps}, param={expected_param_final}, "
        f"ema={expected_ema_final:.4f}"
    )

    with apply_init_maybe(optimizer2):
        for p in trainable2:
            _assert_close_local(p.data, 0.0)
    _log("  init_backup after resume OK")


def _fsdp_verify_worker(rank, world_size, tmpdir):
    """Worker for FSDP verification: EMA correctness + DCP save/load + init_backup."""
    setup_process(rank, world_size)

    fill_value = 1.0
    decay = 0.9
    save_at_step = 3
    total_steps = 6

    def _log(msg):
        if rank == 0:
            print(msg)  # noqa: T201

    # Phase 1: train, verify, save
    model, optimizer, trainable = _make_fsdp_model_and_optimizer(
        rank, fill_value, decay
    )
    expected_ema, expected_param = _fsdp_verify_phase1(
        rank, trainable, optimizer, save_at_step, fill_value, decay, tmpdir, _log
    )

    ckpt_path = os.path.join(tmpdir, "fsdp_ckpt")
    ema_state = EMATrainingState(model, optimizer)
    dcp.save(
        {"training": ema_state.state_dict(), "step_count": optimizer._step_count},
        checkpoint_id=ckpt_path,
    )
    _log(f"  Saved FSDP DCP checkpoint to {ckpt_path}")
    dist.barrier()

    # Phase 2: fresh model, load, resume, verify
    model2, optimizer2, trainable2 = _make_fsdp_model_and_optimizer(
        rank, fill_value, decay
    )
    _fsdp_verify_phase2(
        rank,
        model2,
        optimizer2,
        trainable2,
        save_at_step,
        total_steps,
        fill_value,
        decay,
        expected_ema,
        expected_param,
        ckpt_path,
        _log,
    )

    _log("[PASS] FSDP EMA + DCP + init_backup verification passed.")
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "local",
            "fsdp",
            "verify",
            "verify_init_backup",
            "verify_dcp",
            "fsdp_verify",
        ],
    )
    args = parser.parse_args()

    if args.mode == "local":
        run_local_ema_test()
    elif args.mode == "fsdp":
        world_size = torch.cuda.device_count()
        mp.spawn(run_fsdp_ema_test, args=(world_size,), nprocs=world_size)
    elif args.mode == "verify":
        run_local_ema_verify()
    elif args.mode == "verify_init_backup":
        run_local_init_backup_verify()
    elif args.mode == "verify_dcp":
        run_local_dcp_verify()
    elif args.mode == "fsdp_verify":
        tmpdir = tempfile.mkdtemp(prefix="fsdp_ema_test_")
        try:
            world_size = torch.cuda.device_count()
            mp.spawn(_fsdp_verify_worker, args=(world_size, tmpdir), nprocs=world_size)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
