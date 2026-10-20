"""Validate the FAST prompt layout end to end on real tokenizer + precomputed FAST ids.

Reads ``fast_tokens.npz`` (from ``prepare_fast_tokens.py``) plus a tokenizer dir, packs
every sample with :class:`PrepareSubtaskFastPrompts`, and writes a layout report.

    python scripts/architecture/check_fast_prompt_layout.py \
        --tokenizer <dir with tokenizer.model> --fast fixtures/fast_tokens.npz \
        --labels fixtures/labels.json --output fixtures/layout.json --max-len 256
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fluxvla import register_training_components
register_training_components()
from fluxvla.engines import build_tokenizer_from_cfg
from fluxvla.transforms.subtask_fast_prompts import PrepareSubtaskFastPrompts


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--fast', required=True, help='npz with fast_ids + fast_lengths')
    ap.add_argument('--labels', required=True, help='json list of {overall_task, subtask, states}')
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--state-dim', type=int, default=32)
    ap.add_argument('--model-vocab', type=int, default=257152)
    args = ap.parse_args()

    tokenizer = build_tokenizer_from_cfg(dict(type='PretrainedTokenizer', model_path=args.tokenizer))
    prompts = PrepareSubtaskFastPrompts(tokenizer, max_len=args.max_len, state_dim=args.state_dim,
                                        model_vocab_size=args.model_vocab)
    ids = np.load(args.fast, allow_pickle=False)
    flat, lengths = ids['fast_ids'], ids['fast_lengths']
    labels = json.loads(Path(args.labels).read_text())

    rows, offset, failures = [], 0, []
    for index, label in enumerate(labels):
        n = int(lengths[index])
        seq = flat[offset:offset + n]
        offset += n
        try:
            packed = prompts.high_level(label['overall_task'], np.asarray(label['states']),
                                        label['subtask'], seq)
        except ValueError as exc:
            failures.append(dict(sample=index, error=str(exc)))
            continue
        valid = int(packed['valid_mask'].sum())
        rows.append(dict(
            sample=index, overall_task=label['overall_task'], subtask=label['subtask'],
            fast_tokens=n, valid_tokens=valid, max_len=args.max_len,
            headroom=args.max_len - valid,
            prefix=int((~packed['loss_mask'][:valid]).sum()),
            subtask_region=int(packed['subtask_region_mask'].sum()),
            action_region=int(packed['action_region_mask'].sum()),
            eos_is_last=bool(packed['input_ids'][valid - 1] == prompts.eos_id),
            ar_causal_only_on_targets=bool(
                not packed['ar_mask'][:valid][~packed['loss_mask'][:valid]].any()),
        ))
    if offset != len(flat):
        raise SystemExit(f'fast token buffer mismatch: consumed {offset} of {len(flat)}')
    if not rows:
        raise SystemExit(f'no sample packed; failures={failures}')

    report = dict(
        contract='pi05-subtask-fast-v1',
        tokenizer_dir=str(Path(args.tokenizer).resolve()),
        fast_file=str(Path(args.fast).resolve()), fast_sha256=sha256(args.fast),
        max_len=args.max_len, state_dim=args.state_dim, model_vocab=args.model_vocab,
        samples=len(rows), failures=failures,
        token_budget=dict(min_valid=min(r['valid_tokens'] for r in rows),
                          max_valid=max(r['valid_tokens'] for r in rows),
                          max_fast=max(r['fast_tokens'] for r in rows),
                          max_headroom=min(r['headroom'] for r in rows)),
        rows=rows,
    )
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(dict(samples=report['samples'], failures=len(failures),
                          token_budget=report['token_budget']), indent=2))
    for row in rows:
        print(f"  #{row['sample']} fast={row['fast_tokens']:3d} valid={row['valid_tokens']:3d} "
              f"headroom={row['headroom']:3d} prefix={row['prefix']:3d} "
              f"sub={row['subtask_region']:3d} act={row['action_region']:3d} "
              f"eos_last={row['eos_is_last']}")


if __name__ == '__main__':
    main()
