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

def lr_at(update, total, peak, warmup_fraction=.05, floor_ratio=.1):
    if not 1 <= update <= total:
        raise ValueError('update outside schedule')
    warm = max(1, math.ceil(total * warmup_fraction))
    if update <= warm:
        return peak * update / warm
    progress = (update - warm) / max(1, total - warm)
    return peak * (floor_ratio + (1-floor_ratio) * .5 * (1 + math.cos(math.pi*progress)))

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

def runs(labels, valid):
    start = None
    for i, (label, ok) in enumerate(zip(labels, valid)):
        if start is not None and (not ok or label != labels[start]):
            yield start, i
            start = None
        if ok and start is None:
            start = i
    if start is not None:
        yield start, len(labels)

def anchors(labels, valid, stride, horizon):
    return [(t, min(horizon, end-t-1))
            for start, end in runs(labels, valid) for t in range(start, end, stride)]

def target_window(states, actions, frame, length):
    out = actions[frame+1:frame+1+length].copy()
    out[:, JOINTS] -= states[frame, JOINTS]
    return out

def stats(values):
    # Dimension-by-dimension quantiles avoid a full copy of a large target memmap.
    return {name: [float(np.quantile(values[:, d], q)) for d in range(14)]
            for name, q in [('q01', .01), ('q99', .99)]}

def normalize(values, st):
    lo, hi = np.array(st['q01']), np.array(st['q99'])
    return (2*(values-lo)/np.where(hi-lo < 1e-6, 1., hi-lo)-1).astype('float32')

def split_groups(episodes, count, seed):
    groups = {}
    for i, ep in enumerate(episodes):
        groups.setdefault(ep['trajectory_sha256'], []).append(i)
    keys = list(groups)
    np.random.default_rng(seed).shuffle(keys)
    possible = {0: []}
    for key in keys:
        for n, selected in list(possible.items()):
            target = n + len(groups[key])
            if target <= count and target not in possible:
                possible[target] = selected + groups[key]
    if count not in possible:
        raise ValueError('cannot split requested episode count without duplicate trajectory leakage')
    val = set(possible[count])
    if not val or len(val) >= len(episodes):
        raise ValueError('empty train or validation split')
    return val
