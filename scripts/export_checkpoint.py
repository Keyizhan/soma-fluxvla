"""Export a verified full training snapshot; never requires or loads robot drivers."""
import argparse
import copy
import json
from pathlib import Path
from fluxvla.training.subtask.checkpoints import export_bundle, verify_full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='full .pt with .pt.ready.json')
    parser.add_argument('--output', required=True, help='new .deploy directory')
    parser.add_argument('--cache', help='relocated original verified training cache')
    parser.add_argument('--tokenizer', help='relocated original tokenizer directory')
    args = parser.parse_args()
    state = verify_full(args.checkpoint)
    metadata = {k: v for k, v in state.items() if k not in ('model_state', 'optimizer', 'rng')}
    cfg = copy.deepcopy(state['config'])
    if args.cache:
        cfg['cache'] = args.cache
    if args.tokenizer:
        cfg['tokenizer'] = args.tokenizer
    # Verify the cache and its identity before deriving normalization from it.
    from fluxvla.training.subtask.prepare import verify
    from fluxvla.training.subtask.common import digest
    verify(dict(state['config'], cache=cfg['cache']))
    for name, expected in state['config']['tokenizer_sha256'].items():
        if digest(Path(cfg['tokenizer']) / name) != expected:
            raise ValueError(f'tokenizer differs from training: {name}')
    if digest(Path(cfg['cache']) / 'READY.json') != state['cache_identity']:
        raise ValueError('cache differs from the training snapshot')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('choose a new output directory; existing exports are immutable')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = export_bundle(state['model_state'], output, cfg, state['model_config'], metadata)
    print(json.dumps({'deployment': str(result), 'stage': state['stage'], 'update': state['update']}))


if __name__ == '__main__':
    main()
