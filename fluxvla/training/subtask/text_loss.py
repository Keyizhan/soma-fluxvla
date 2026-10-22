"""Explicit text objectives. Schedules use one-based optimizer updates within each stage."""
import copy
import math


def resolve_text_loss(value):
    if not isinstance(value, dict):
        raise ValueError('text_loss must be an object')
    c = copy.deepcopy(value)
    allowed = {'mode', 'weighting', 'subtask_weight', 'fast_weight', 'end_subtask_weight',
               'end_fast_weight', 'start_update', 'end_update', 'split_subtask',
               'key_weight', 'template_weight', 'normalize_weights'}
    if set(c) - allowed:
        raise ValueError(f'unknown text_loss keys: {sorted(set(c)-allowed)}')
    for k, v in dict(mode='unified_token', weighting='fixed', subtask_weight=1., fast_weight=1.,
                     split_subtask=False, key_weight=1., template_weight=.1).items():
        c.setdefault(k, v)
    if c['mode'] not in ('unified_token', 'separate') or c['weighting'] not in ('fixed', 'linear'):
        raise ValueError('invalid text_loss mode or weighting')
    if not isinstance(c['split_subtask'], bool):
        raise ValueError('split_subtask must be boolean')
    if 'normalize_weights' in c and not isinstance(c['normalize_weights'], bool):
        raise ValueError('normalize_weights must be boolean')
    for k in ('subtask_weight', 'fast_weight', 'key_weight', 'template_weight',
              'end_subtask_weight', 'end_fast_weight'):
        if k in c and (isinstance(c[k], bool) or not isinstance(c[k], (int, float)) or not math.isfinite(c[k]) or c[k] < 0):
            raise ValueError(f'{k} must be finite and nonnegative')
    if c['subtask_weight'] + c['fast_weight'] == 0 or c['key_weight'] + c['template_weight'] == 0:
        raise ValueError('loss weight pairs cannot both be zero')
    if c['mode'] == 'unified_token' and (c['weighting'] != 'fixed' or c['subtask_weight'] != 1 or c['fast_weight'] != 1):
        raise ValueError('task weights/schedules require separate mode')
    if c['weighting'] == 'linear':
        c.setdefault('end_subtask_weight', c['subtask_weight'])
        c.setdefault('end_fast_weight', 10.)
        c.setdefault('start_update', 1)
        for k in ('start_update', 'end_update'):
            if k in c and (isinstance(c[k], bool) or not isinstance(c[k], int) or c[k] < 1):
                raise ValueError(f'{k} must be a positive integer')
        if 'end_update' in c and c['end_update'] < c['start_update']:
            raise ValueError('end_update precedes start_update')
        if c['end_subtask_weight'] + c['end_fast_weight'] == 0:
            raise ValueError('end weights cannot both be zero')
    elif any(k in c for k in ('end_subtask_weight', 'end_fast_weight', 'start_update', 'end_update')):
        raise ValueError('schedule fields require linear weighting')
    return c


def weights_at(c, update, total_updates):
    c = resolve_text_loss(c)
    p = 0.
    if c['weighting'] == 'linear':
        start, end = c['start_update'], c.get('end_update', total_updates)
        if end < start:
            raise ValueError('schedule start exceeds stage total updates')
        p = float(update >= end) if start == end else min(1., max(0., (update-start)/(end-start)))
    weights = tuple(c[k] + p*(c.get('end_'+k, c[k])-c[k]) for k in ('subtask_weight', 'fast_weight'))
    return normalize_pair(*weights) if c.get('normalize_weights', False) else weights


def normalize_pair(a, b):
    # Scale first to avoid overflow when finite weights are individually large.
    scale = max(a, b)
    a, b = a / scale, b / scale
    return a / (a+b), b / (a+b)


def model_loss_settings(c, preset):
    c = resolve_text_loss(c)
    enabled = preset not in ('action_expert', 'flow_only')
    sw, fw = c['subtask_weight'], c['fast_weight']
    kw, tw = c['key_weight'], c['template_weight']
    if c.get('normalize_weights', False) and c['mode']=='separate':
        sw, fw = normalize_pair(sw, fw)
        kw, tw = normalize_pair(kw, tw)
    result = dict(text_loss_mode=c['mode'], token_loss_weight=float(enabled and c['mode']=='unified_token'),
                hl_loss_weight=sw if enabled and c['mode']=='separate' else 0.,
                fast_loss_weight=fw if enabled and c['mode']=='separate' else 0.,
                split_subtask_loss=c['split_subtask'], subtask_key_weight=kw,
                subtask_template_weight=tw)
    if c.get('normalize_weights', False):
        result['normalize_text_weights'] = True
    return result


def apply_text_schedule(model, c, preset, update, total_updates):
    if c is not None and c['mode'] == 'separate' and preset not in ('action_expert', 'flow_only'):
        model.hl_loss_weight, model.fast_loss_weight = weights_at(c, update, total_updates)
