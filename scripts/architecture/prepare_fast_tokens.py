"""Offline FAST action-token preparation (requires scipy + HF remote code).

Run this in an environment that has ``scipy`` and can load
``physical-intelligence/fast`` (see ``docs/plans/pi05_subtask_fast_reproduction.md``
T1.1-T1.3). Training only consumes the resulting ``fast_tokens`` arrays, so the
training environment needs neither scipy nor remote-code transformers.

Example:
    python scripts/architecture/prepare_fast_tokens.py \
        --tokenizer assets/fast \
        --fixtures experiments/pi05_subtask_architecture/shoes_expansion_20261017/fixtures/samples.npz \
        --output experiments/pi05_subtask_architecture/shoes_expansion_20261017/fixtures/fast_tokens.npz \
        --max-tokens 160
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', required=True, help='local dir of physical-intelligence/fast')
    parser.add_argument('--fixtures', required=True, help='npz containing an `actions` array')
    parser.add_argument('--output', required=True)
    parser.add_argument('--actions-key', default='actions')
    parser.add_argument('--max-tokens', type=int, default=160,
                        help='hard budget for the FAST segment; offsets are not truncated')
    parser.add_argument('--scale', type=float, default=None, help='override DCT scale')
    parser.add_argument('--chunk', type=int, default=512,
                        help='encode this many action windows per batch (bounded memory)')
    parser.add_argument('--uncompressed', action='store_true',
                        help='write fast_tokens.npz uncompressed so training can memory-map it')
    args = parser.parse_args()

    from transformers import AutoProcessor

    tokenizer_dir = Path(args.tokenizer)
    processor = AutoProcessor.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
    if args.scale is not None:
        processor.scale = args.scale
    print(f'processor={type(processor).__name__} scale={processor.scale} '
          f'vocab={processor.vocab_size} min_token={processor.min_token}')

    data = np.load(args.fixtures, allow_pickle=False, mmap_mode='r')
    actions = data[args.actions_key]
    if actions.ndim != 3:
        raise SystemExit(f'expected actions (N,T,D), got {actions.shape}')
    n, horizon, action_dim = actions.shape

    sequences, lengths = [], []
    for start in range(0, n, max(1, args.chunk)):
        chunk = np.asarray(actions[start:start + max(1, args.chunk)])
        encoded = processor(chunk)
        sequences.extend(encoded)
        lengths.extend(len(seq) for seq in encoded)
        print(f'encoded {min(start + args.chunk, n)}/{n}', flush=True)
    lengths = np.asarray(lengths)
    if lengths.max() > args.max_tokens:
        worst = int(lengths.argmax())
        raise SystemExit(
            f'FAST segment exceeds budget: max={lengths.max()} > {args.max_tokens} '
            f'(sample {worst}, horizon={horizon}, dim={action_dim}). No truncation is applied; '
            f'raise --max-tokens or shorten the action horizon.')
    for seq in sequences:
        if min(seq) < 0 or max(seq) >= processor.vocab_size:
            raise SystemExit('FAST id outside the tokenizer vocabulary')

    offset = int(processor.min_token)
    reconstructed = processor.decode(sequences, time_horizon=horizon, action_dim=action_dim)
    error = np.abs(reconstructed - actions)
    real_dim = min(action_dim, 14)
    report = dict(
        tokenizer_dir=str(tokenizer_dir.resolve()),
        tokenizer_files={name: sha256(tokenizer_dir / name)
                         for name in sorted(p.name for p in tokenizer_dir.iterdir() if p.is_file())},
        scale=processor.scale, vocab_size=int(processor.vocab_size), min_token=offset,
        fixtures=str(Path(args.fixtures).resolve()), fixtures_sha256=sha256(Path(args.fixtures)),
        actions_key=args.actions_key, samples=int(n), horizon=int(horizon), action_dim=int(action_dim),
        token_lengths=dict(min=int(lengths.min()), max=int(lengths.max()),
                           mean=float(lengths.mean()), p95=float(np.percentile(lengths, 95))),
        max_tokens_budget=int(args.max_tokens), uncompressed=bool(args.uncompressed),
        chunk=int(args.chunk),
        roundtrip=dict(mse_all_dims=float(error.mean()), mse_first_14=float(error[..., :real_dim].mean()),
                       max_first_14=float(error[..., :real_dim].max()),
                       note='FAST is a lossy representation for training supervision; '
                            'this error is not the flow-matching action error.'),
    )
    if not np.isfinite(reconstructed).all():
        raise SystemExit('non-finite actions after FAST roundtrip')

    flat = np.concatenate([np.asarray(seq, dtype=np.int64) for seq in sequences])
    lengths_out = lengths.astype(np.int64)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.uncompressed:
        np.savez(out, fast_ids=flat, fast_lengths=lengths_out)
    else:
        np.savez_compressed(out, fast_ids=flat, fast_lengths=lengths_out)
    report['output'] = str(out.resolve())
    report['output_sha256'] = sha256(out)
    report['flat_ids_sha256'] = hashlib.sha256(flat.tobytes()).hexdigest()
    (out.parent / (out.stem + '.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('samples', 'token_lengths', 'roundtrip')}, indent=2))


if __name__ == '__main__':
    main()
