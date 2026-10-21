# Data formats on disk

Directory layouts written by `flow-control preprocess` (random cache), `flow-control pack` (packed cache) and `ReportWriter` (inference report). Both caches, and a report's `records/`, open with `{"type": "cache", "path": ...}`; `meta.json` tells them apart. How they are read is in [data.md](data.md).

## Shared: `index.jsonl` and Row Files

`index.jsonl` holds one `IndexEntry` (`data/index.py`) per line; a row's `row_id` is its 0-based line number.

| Field | Type | Meaning |
|---|---|---|
| `key` | `str` | Row key, unique in the cache, `[A-Za-z0-9._-]+` |
| `cost` | `int` | `processor.get_cost` of the row |
| `sig` | `str` | `shape_signature(row)`: `name:AxBxC` per top-level tensor and `name:[len]` per list, sorted by name, comma-joined. A sort tie-break after `cost`, never a bucket |
| `image_size` | `[h, w]` or `null` | Pixel size from the row |
| `loc` | `str` or `[shard, offset, size]` | Directory backend: path relative to the cache root. LMDB backend: the LMDB key. Packed: shard number, byte offset of the member data, byte size |

One FLUX.1 training row at 1024², in a directory cache and after packing:

```jsonl
{"key": "000123", "cost": 4608, "sig": "clean_latents:2x4096x64,pooled_prompt_embeds:1x768,prompt_embeds:1x512x4096", "image_size": [1024, 1024], "loc": "rows/3f/000123.pt"}
{"key": "000123", "cost": 4608, "sig": "clean_latents:2x4096x64,pooled_prompt_embeds:1x768,prompt_embeds:1x512x4096", "image_size": [1024, 1024], "loc": [0, 1536, 5245377]}
```

A row payload (a `.pt` file, an LMDB value or a tar member) is `torch.save(row)` of the whole row dict, reserved keys included. Readers use `torch.load(weights_only=True)`, so values are tensors, numbers, strings, `None`, and lists, tuples or dicts of them. `padding` is never written.

A directory without `meta.json` is not a cache: writers create `meta.json` last, after merging their per-worker parts.

## Random Cache

```
<cache>/
├── meta.json
├── index.jsonl          # sorted by key
├── rows/<hh>/<key>.pt   # backend "directory": hh = sha1(key).hexdigest()[:2], 256 sub-directories
├── data.mdb             # backend "lmdb": key (utf-8) -> torch.save bytes
└── lock.mdb             # backend "lmdb"
```

| `meta.json` field | Value |
|---|---|
| `format` | `"random"` |
| `version` | `1` (`CACHE_FORMAT_VERSION`) |
| `backend` | `"directory"` or `"lmdb"` |
| `mode` | The preprocess `processing_mode`: `"training"` or `"inference"` |
| `processor` | The preprocess `processor` config |
| `rows` | Number of entries |

A report's `records/` carries only `format`, `version`, `backend` and `rows`.

While preprocess runs, each sink worker writes `index-<worker>.jsonl`; `finalize_cache` merges them sorted by key, rejects duplicate keys, writes `index.jsonl` and `meta.json`, and deletes the parts.

## Packed Cache

```
<packed>/
├── meta.json
├── index.jsonl          # shard order, then member order inside the shard
└── shards/
    ├── 000000.tar       # members <key>.pt in (cost, sig) order
    ├── 000001.tar
    └── ...
```

`meta.json` keeps the source's fields other than `format`, `backend`, `version` and `rows` (so `mode` and `processor`), and adds:

| Field | Value |
|---|---|
| `format` | `"packed"` |
| `version` | `1` |
| `shard_size` | `PackConfig.shard_size`: rows per shard; the last shard may be shorter |
| `group_size` | `PackConfig.group_size`, recorded only; the reading trainer's `group_size` forms the groups |
| `seed` | `PackConfig.seed` |
| `source` | Absolute path of the input random cache |
| `shards` | Number of shards |
| `rows` | Number of entries |

A shard is an uncompressed tar written by `tarfile` (pax headers allowed). Shard `s` holds rows `s * shard_size` to `(s + 1) * shard_size - 1` of a `randperm(N)` seeded by `PackConfig.seed`, sorted by `(cost, sig)`. Each member is `<key>.pt` with the row's `torch.save` bytes. `loc` points past the member's header blocks at its data, so reading a row is one `seek` and one `read(size)`. A short shard is not padded: padding is added at read time by the plan.

## Inference Report

```
<report>/
├── meta.json
├── metrics.jsonl
├── previews/<key>.png   # "previews": true
└── records/             # "records": true; a random cache, backend "directory"
    ├── meta.json
    ├── index.jsonl
    └── rows/<hh>/<key>.pt
```

| `meta.json` field | Value |
|---|---|
| Caller's `meta` | Inference passes `{"config": <the inference config, model_dump(mode="json")>}` |
| `rows` | Lines in `metrics.jsonl` |
| `ranks` | Number of ranks |

`metrics.jsonl` has one line per non-padding sample, rank 0's lines first, then rank 1's, and so on:

| Field | Value |
|---|---|
| `key` | Row key |
| `rank` | Rank that sampled the row |
| `image_size` | `[h, w]` or `null` |
| `cost` | `int` or `null` |
| Metrics | The `metrics` passed to `ReportWriter.write`: nested dicts flattened with `.`, tensors as numbers or lists. Inference with `reward`: `reward` (weighted sum of the normalized components), `reward_raw.<label>`, `reward_normalized.<label>` |

`previews/<key>.png` is `row["clean_image"]`, or `processor.annotate_output(...)` with `annotate_output_image: true`.

A `records/` row holds `key`, `cost`, `image_size` and the fields `decode_output` wrote (`clean_image` and any task outputs). With `save_extra: true` it holds the whole working row instead: source fields, conditions, `clean_latents` set to the sampled endpoint, and the decoded outputs. With `reward`, it also holds `reward` and the `reward_raw` / `reward_normalized` dicts keyed by label.

While inference runs, each rank writes `metrics-<rank>.jsonl` and `records/index-<rank>.jsonl`. `ReportWriter.finalize` merges them on rank 0 and deletes them. A new `ReportWriter` on rank 0 deletes any parts left in the directory by an earlier run before writing.
