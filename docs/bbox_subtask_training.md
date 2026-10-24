# Bbox subtask training v1

This opt-in extension consumes `dataset_goal_bbox_v1`. Existing configurations without
`bbox` retain the original prompts, objective and deployment format. No detector head,
new embedding rows or extra model parameters are introduced.

## Configuration

`configs/subtask/bbox_common.json` sets:

```json
"bbox": {
  "contract": "fabric-goal-bbox-v1",
  "weight": 5.0,
  "type_weight": 1.0,
  "coordinate_weight": 1.0,
  "format_weight": 0.1
}
```

The complete objective is the existing text/action objective plus
`5 * (bbox_type_ce + bbox_coordinate_ce + 0.1 * bbox_format_ce)`.
The additional term is disabled in expert-only/flow-only optimization, but its
diagnostics can still be evaluated. Original loss weights and reductions are preserved.

Use `bbox_hybrid.json`, `bbox_hybrid_insulated.json`, or `bbox_two_stage.json`.
They have distinct outputs and a shared bbox-specific cache and stage-one root.
The default flow weight inherited from common.json is unchanged (1.0); this is
not the separate existing `pro6000_*_la10` recipe. To compare with another recipe,
explicitly carry its non-bbox settings into a derived configuration.

## Data and sequence contract

- Read original `subtask` and structured `grounding_json`. Require exact agreement
  with `subtask_grounded`, valid flags and primitive type during dataset scan.
- Placement emits `Target: cam_front xyxy [x1, y1, x2, y2]; Subtask: ...`.
- Retrieve/flip emit `Target: none; Subtask: ...`. No coordinate loss or coordinate
  placeholder is produced for these goals.
- Parallel goals are serialized in dataset order, separated by ` || `; every goal
  has its own target. Coordinate groups are identified by goal and coordinate.
- Joint training rejects missing/invalid grounding on otherwise valid frames.
  This v1 implementation deliberately does not silently turn missing boxes into none
  or execute placement with unknown spatial conditions.
- Only original unboxed images are inputs. Four q1000 integer coordinates refer to
  continuous front-camera boundaries, in xyxy order. Existing resize preserves these
  normalized coordinates; geometric crop/flip augmentation is not supported.
- The high-level context ends before `Target:`. All target/type/coordinate/format,
  subtask and FAST output tokens are causal. The bbox cannot see later ground-truth
  subtask tokens. The low-level context is rebuilt from state and grounded goals,
  excluding overall task and FAST tokens.
- Segment encoding is explicit and identical for high-level target and rebuilt
  low-level goal text. It reuses the existing tokenizer and permits decoder whitespace
  around protocol punctuation, but rejects missing fields, wrong camera, noninteger,
  out-of-range or reversed coordinates. No coordinate repair/clipping is performed.

## Loss reduction

Each coordinate's token CE is averaged separately. The four coordinate averages
are equally weighted, then averaged over valid bbox goals. Type and format losses
are averaged over tokens within each goal and then over valid goals, including none.
Bbox tokens are excluded from legacy subtask/FAST loss denominators and from the
subtask key/template partition. Template fitting uses train-split primitive texts only.

The trainer counts bbox goals over the entire optimizer update before backward,
including every microbatch and every DDP rank. It applies world-size compensation
for DDP gradient averaging, without another microbatch divisor. Padded sentinel
samples do not contribute. A rank or update containing zero bbox goals has a
finite, graph-connected zero coordinate loss. Existing non-bbox objectives retain
their original microbatch reduction.

Training/validation log `bbox_{type,coordinate,format}_ce` and corresponding goal
counts. Training also logs `optimization_weighted_bbox_term`; it is an additional
term, not included in the old subtask-vs-FAST fractions.

## Commands

Run from the repository root after setting environment-specific pretrained/tokenizer
paths in the inherited runtime configuration and provisioning the expected GPUs:

```sh
python -m fluxvla.training.subtask.prepare --config configs/subtask/bbox_hybrid.json scan
python -m fluxvla.training.subtask.prepare --config configs/subtask/bbox_hybrid.json prepare
python -m fluxvla.training.subtask.run --config configs/subtask/bbox_hybrid.json
```

The cache is rebuilt into `cache/goal_bbox_v1`; an existing destination is refused.
The text budget is 512. Real tokenizer length is checked during cache preparation;
overflow fails rather than truncating labels or FAST tokens. The local checkout
does not contain the production PaliGemma tokenizer/base weights, so the production
token budget and GPU memory usage must be measured when preparing the real cache.

Teacher-forced CE plus free-running localization evaluation:

```sh
python -m fluxvla.training.subtask.evaluate --config configs/subtask/bbox_hybrid.json --bundle PATH_TO_BUNDLE --bbox-generation-limit 200 --bbox-max-new-tokens 256
```

The generation evaluator takes observations and overall task only. It reports type
and format accuracy, end-to-end IoU (malformed/missing boxes score zero), valid-box
IoU, missed-box rate and center error in q1000 units conditioned on valid boxes.
The limit selects the first N validation records; use the full validation count for
aggregate evaluation, since a small ordered prefix is not representative.

Deployment bundles carry the bbox config and prompt transform. Loading rejects
model/contract mismatch. Default bundle inference uses a 256-token budget for bbox
models. CLI callers passing an explicit `InferenceConfig` should set an adequate
budget; `scripts/evaluate_subtask.py` supports `--max-new-tokens 256`.
Malformed grounding is rejected before action sampling. Fresh coordinates for the
same semantic goals update immediately without a task-switch vote; held predictions
retain both text and coordinates. Semantic transitions preserve existing debounce.

## Validation scope

Tests cover dataset/cache/FAST integration, masks, no-box and parallel goals,
grouped loss and gradients, optimizer-update normalization, causal isolation,
deployment contract checks and inference conditioning. A real two-process CPU Gloo
test checks a DDP rank with no bbox against the global gradient reference.
No robot commands or production training are part of these checks.

The existing checkpoint tests use POSIX statvfs and fsync behavior. Windows cannot
validate POSIX durability; a test-only compatibility harness can verify serialization
without claiming those durability guarantees. The pre-existing native-bundle fixture
also lacks an explicit unified-token objective; do not weaken production validation
to accept that outdated fixture.
