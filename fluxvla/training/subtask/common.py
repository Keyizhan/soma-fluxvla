"""Lightweight pipeline contracts; no torch import during data preparation."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

JOINTS = (0,1,2,3,4,5,7,8,9,10,11,12)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def lr_at(update, total, peak, warmup_fraction=.05, floor_ratio=0.):
    """LR used by update N, matching ordinary PI05's post-step scheduler."""
    if not 1 <= update <= total:
        raise ValueError('update outside schedule')
    if floor_ratio != 0:
        raise ValueError('ordinary PI05 cosine schedule requires zero floor')
    if update == 1:
        return 0.
    step = update - 1
    warm = int(total * warmup_fraction)
    if step < warm:
        return peak * step / max(1, warm)
    progress = (step - warm) / max(1, total - warm)
    return peak * max(0., .5 * (1 + math.cos(math.pi * progress)))


def epoch_batches(indices, seed, epoch, global_batch, micro, world, rank):
    if global_batch % (micro * world):
        raise ValueError('global batch must be divisible by micro batch * world size')
    order = np.random.default_rng(seed + epoch).permutation(indices).tolist()
    block = micro * world
    for start in range(0, len(order), global_batch):
        batch = order[start:start+global_batch]
        batch += [-1] * (global_batch - len(batch))
        yield [batch[offset + rank*micro:offset + (rank+1)*micro]
               for offset in range(0, global_batch, block)]

def anchors(valid, horizon):
    """Use every valid observation once; labels never cut action windows."""
    return [(t, min(horizon, len(valid)-t-1)) for t, ok in enumerate(valid) if ok]

def target_window(states, actions, frame, length):
    # Ordinary PI05 starts at t+1; masks separately exclude repeated episode padding.
    indices = np.minimum(np.arange(frame+1, frame+1+length), len(actions)-1)
    out = actions[indices].copy()
    out[:, JOINTS] -= states[frame, JOINTS]
    return out

def stats(values):
    # Dimension-by-dimension quantiles avoid a full copy of a large target memmap.
    return {name: [float(np.quantile(values[:, d], q)) for d in range(14)]
            for name, q in [('q01', .01), ('q99', .99)]}

def normalize(values, st):
    lo, hi = np.array(st['q01']), np.array(st['q99'])
    return (2*(values-lo)/np.where(hi-lo < 1e-6, 1., hi-lo)-1).astype('float32')

def validate_training_method(cfg):
    """Keep method names and stage recipes unambiguous; accept legacy custom configs."""
    method = cfg.get('training_method')
    if method is None:
        return
    second = {'hybrid': 'hybrid', 'hybrid_insulated': 'hybrid_insulated',
              'two_stage': 'action_expert'}
    if method not in second:
        raise ValueError(f'unknown training_method: {method}')
    expected=[('stage2',second[method])]
    if method in ('hybrid_insulated','two_stage'):expected.insert(0,('stage1','stage1_fast'))
    if [(s['name'],s['preset']) for s in cfg['stages']]!=expected:
        raise ValueError('training_method and stage recipes disagree')
