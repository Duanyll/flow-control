"""Shuffle and grouping plans (design §6) and the RL rollout split (§9.3).

Pure functions over an ``Index``. A plan is a list of groups; every group holds
exactly ``n`` planned rows with similar ``cost`` so that the ranks, which consume
groups in lock-step via ``rank_rows``, forward similarly sized samples at any
moment (FSDP balance). Padding is a plan-time concept: a short tail group is
filled by repeating rows of the same block, marked ``is_padding=True``.
"""

import itertools

import torch

from flow_control.data.index import Index

PlannedRow = tuple[int, bool]
"""``(row_id, is_padding)``."""
Group = list[PlannedRow]


def sort_key(index: Index, i: int) -> tuple[int, str]:
    e = index.entries[i]
    return e.cost, e.sig


def _generator(seed: int, epoch: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed * 1_000_003 + epoch)


def _chunk_with_padding(rows: list[int], n: int) -> list[Group]:
    """Cut ``rows`` into groups of ``n``. A short last group (including the case
    ``len(rows) < n``) is filled by cycling through that group's own rows, marked
    padding, so the padding cost matches the rows it sits next to."""
    if n < 1:
        raise ValueError(f"group size must be >= 1, got {n}")
    groups: list[Group] = []
    for start in range(0, len(rows), n):
        chunk = rows[start : start + n]
        group: Group = [(row_id, False) for row_id in chunk]
        group.extend((chunk[j % len(chunk)], True) for j in range(n - len(chunk)))
        groups.append(group)
    return groups


def groups_shuffled(index: Index, n: int, k: int, seed: int, epoch: int) -> list[Group]:
    """random-cache training: global permutation -> megabatches of ``n * k`` ->
    sort each by ``(cost, sig)`` -> cut into ``n`` -> shuffle the group order."""
    if k < 1:
        raise ValueError(f"megabatch_groups must be >= 1, got {k}")
    g = _generator(seed, epoch)
    perm = torch.randperm(len(index), generator=g).tolist()
    groups: list[Group] = []
    for start in range(0, len(perm), n * k):
        mega = sorted(perm[start : start + n * k], key=lambda i: sort_key(index, i))
        groups.extend(_chunk_with_padding(mega, n))
    order = torch.randperm(len(groups), generator=g).tolist()
    return [groups[i] for i in order]


def groups_sorted(index: Index, n: int) -> list[Group]:
    """inference / validation: whole index sorted by ``(cost, sig)``, cut into ``n``."""
    return _chunk_with_padding(
        sorted(range(len(index)), key=lambda i: sort_key(index, i)), n
    )


def _shard_of(index: Index, i: int) -> int:
    loc = index.entries[i].loc
    if not isinstance(loc, tuple):
        raise TypeError(
            f"groups_packed needs a packed cache index (loc = (shard, offset, size)); "
            f"entry {i} has loc={loc!r}. Use groups_shuffled for random caches."
        )
    return loc[0]


def groups_packed(
    index: Index,
    n: int,
    seed: int,
    epoch: int,
    shuffle_within_shard: bool = False,
) -> list[Group]:
    """packed cache: shard order permuted; inside a shard the rows are already
    ``(cost, sig)``-sorted, so consecutive ``n`` form a group and no group crosses
    a shard boundary. ``shuffle_within_shard`` also permutes the groups of each
    shard (more seeks inside one tar, still one shard at a time)."""
    g = _generator(seed, epoch)
    shards = [
        _chunk_with_padding(list(run), n)
        for _, run in itertools.groupby(
            range(len(index)), key=lambda i: _shard_of(index, i)
        )
    ]
    order = torch.randperm(len(shards), generator=g).tolist()
    groups: list[Group] = []
    for shard_id in order:
        shard_groups = shards[shard_id]
        if shuffle_within_shard:
            inner = torch.randperm(len(shard_groups), generator=g).tolist()
            shard_groups = [shard_groups[j] for j in inner]
        groups.extend(shard_groups)
    return groups


def groups_plain(
    num_rows: int, n: int, seed: int, epoch: int, shuffle: bool
) -> list[Group]:
    """no cache: no cost to sort by, only an optional permutation cut into ``n``
    (groups exist solely for the rank split)."""
    if shuffle:
        rows = torch.randperm(num_rows, generator=_generator(seed, epoch)).tolist()
    else:
        rows = list(range(num_rows))
    return _chunk_with_padding(rows, n)


def rank_rows(groups: list[Group], rank: int, world_size: int) -> list[PlannedRow]:
    """Strided slice of every group, concatenated: rank ``r`` takes elements
    ``r, r + R, r + 2R, ...`` so all ranks advance through the same group at the
    same time and each gets ``len(groups) * n / world_size`` rows."""
    if groups and len(groups[0]) % world_size != 0:
        raise ValueError(
            f"group_size={len(groups[0])} must be a multiple of world_size={world_size}; "
            "otherwise the ranks get unequal row counts and FSDP collectives desync."
        )
    return [row for g in groups for row in g[rank::world_size]]


def expand_rollouts(
    prompt_ids: list[int],
    K: int,
    rank: int,
    world_size: int,
    *,
    whole_prompts: bool,
) -> list[tuple[int, int]]:
    """Expand ``M`` prompts into ``(prompt_id, k)`` pairs and take this rank's
    **strided** share.

    Strided, not contiguous: the ``i``-th rollout of every rank then maps to
    prompts adjacent in the group sequence, so at any moment all ranks forward
    similar costs (needs ``group_size % world_size == 0``, §6.1). A contiguous
    split cannot guarantee that once a rank's share is smaller than one group.

    ``whole_prompts`` (pairwise rewards): rotate at prompt level so all ``K``
    rollouts of a prompt land on the same rank.
    """
    if whole_prompts:
        return [(p, k) for p in prompt_ids[rank::world_size] for k in range(K)]
    flat = [(p, k) for p in prompt_ids for k in range(K)]
    return flat[rank::world_size]


if __name__ == "__main__":
    from rich import print

    from flow_control.data.index import IndexEntry

    g = torch.Generator().manual_seed(0)
    entries = [
        IndexEntry(f"r{i}", int(torch.randint(1, 20, (1,), generator=g)), "", None, "")
        for i in range(21)
    ]
    index = Index(entries, {})

    groups = groups_shuffled(index, n=4, k=2, seed=1, epoch=0)
    print(groups)
    assert all(len(grp) == 4 for grp in groups)
    real = [r for grp in groups for r, pad in grp if not pad]
    assert sorted(real) == list(range(21))
    assert groups == groups_shuffled(index, n=4, k=2, seed=1, epoch=0)
    assert groups != groups_shuffled(index, n=4, k=2, seed=1, epoch=1)

    r0, r1 = rank_rows(groups, 0, 2), rank_rows(groups, 1, 2)
    assert len(r0) == len(r1) == 2 * len(groups)
    # Real rows are exclusive per rank; cycled padding may repeat a row on both.
    assert not {r for r, pad in r0 if not pad} & {r for r, pad in r1 if not pad}

    print(_chunk_with_padding([7, 8], 5))
    assert _chunk_with_padding([7, 8], 5) == [
        [(7, False), (8, False), (7, True), (8, True), (7, True)]
    ]
    assert groups_plain(5, 2, seed=0, epoch=0, shuffle=False) == [
        [(0, False), (1, False)],
        [(2, False), (3, False)],
        [(4, False), (4, True)],
    ]

    rollouts = expand_rollouts(
        [10, 11, 12], K=2, rank=1, world_size=2, whole_prompts=False
    )
    print(rollouts)
    assert rollouts == [(10, 1), (11, 1), (12, 1)]
    whole = expand_rollouts([10, 11, 12], K=2, rank=1, world_size=2, whole_prompts=True)
    assert whole == [(11, 0), (11, 1)]
