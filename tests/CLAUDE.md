# Instructions for agents working under `tests/`

## Test budget

1. Never add a test without first asking the user how many new tests you may
   add, and never exceed that number. Say what each proposed test covers and
   which real failure it locks down.
2. Counting: one test = one `def test_*` function or `unittest` method. Extra
   cases inside one function (loops, subTest, table-driven inputs) are free.
3. Trading deletions for additions is welcome — use `/prune-tests` to find the
   deadweight.

## What earns a slot

1. Regressions we actually hit. The docstring or a comment must name the
   problem it prevents: the symptom, and the commit/PR or config that exposed it.
2. Behaviour spanning several modules, processes, or devices: sampler
   plan/executor semantics, microbatch parity, distributed/FSDP workers,
   config/recipe round-trips.

## What does not

1. Specification tests for shallow behaviour — getters, Pydantic field
   validation, enum members.
2. Shotgun coverage for hypothetical failures. Nail down the bugs that bit us.
3. Anything `uv run pyright` already proves.
4. Single-module checks — put those in an `if __name__ == "__main__":` block
   next to the code, as the root `CLAUDE.md` requires. Promote to `tests/` only
   once the behaviour proves fragile across modules or processes.

## Layout and running

1. `test_*.py` — collected tests (`unittest`-style). Run with
   `uv run pytest tests/test_foo.py` or `uv run pytest tests/`.
2. `*_worker.py`, `*_smoke.py` — harnesses launched by hand. Deliberately not
   named `test_*` so pytest ignores them; keep it that way.
3. A test that skips must be able to un-skip. The suite reports 0 skips today.
   If a `skipUnless` guards data that lives on one machine only, check the data
   in or delete the test.

## Numeric baselines

`tests/fixtures/sampler_baselines/` holds solver outputs replayed bitwise by
`BaselineParityTest`, so numeric drift in any solver fails loudly.

1. Regenerate with `uv run python tests/sampler_baseline_capture.py` only when
   algorithm behaviour changes on purpose, and commit the new fixtures in the
   same change. A fixture diff with no intended behaviour change is a bug.
2. Do not add a fixture the capture script cannot regenerate.
3. Fixtures load under `weights_only=True`; keep them free of pickled objects.
