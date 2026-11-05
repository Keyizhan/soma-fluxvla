#!/usr/bin/env python3
"""Check soma-fluxvla PI0.5 ``predict_action`` against the deployed upstream FluxVLA.

Two processes, because both packages are named ``fluxvla``:

  capture  (upstream FluxVLA + piper-vla-deploy on PYTHONPATH): run the real
           ``Pi05FluxVLABackend`` on prepared samples, record the exact
           ``predict_action`` inputs, the sampled noise and the output.
  compare  (soma-fluxvla on PYTHONPATH): rebuild the model from the release's
           ``resolved_config.json`` + weights, feed the recorded inputs and noise,
           and report the difference.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def capture(a):
    from soma_deploy.models.pi05_fluxvla.server import build_backend

    with np.load(a.samples, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    meta = json.loads(str(arrays['meta']))
    backend = build_backend({'release': str(a.release), 'inference_backend': 'eager', 'seed': 7,
                             'reseed_each_request': True, 'warmup_runs': 0})
    vla, records = backend._vla, []
    original_predict, original_noise = vla.predict_action, vla.sample_noise

    def sample_noise(shape, device):
        noise = original_noise(shape, device)
        records[-1]['noise'] = noise.detach().cpu()
        return noise

    def predict_action(**inputs):
        records.append({'inputs': {k: (v.detach().cpu() if torch.is_tensor(v) else
                                       [x.detach().cpu() for x in v] if isinstance(v, list) else v)
                                   for k, v in inputs.items()}})
        out = original_predict(**inputs)
        records[-1]['output'] = out.detach().float().cpu()
        return out

    vla.sample_noise, vla.predict_action = sample_noise, predict_action
    for i in range(min(a.count, len(arrays['state']))):
        backend.infer({'state': arrays['state'][i].astype(np.float32),
                       'images': {c: arrays[f'image_{c}'][i] for c in backend.metadata.camera_keys},
                       'instruction': meta['tasks'][i]})
    torch.save({'records': records, 'mixed_precision': backend._mixed_precision,
                'dtype': str(backend._dtype)}, a.capture)
    print(f'captured {len(records)} predict_action calls -> {a.capture}')


def compare(a):
    from fluxvla import register_training_components
    from safetensors.torch import load_file

    register_training_components()
    from fluxvla.engines import build_vla_from_cfg

    data = torch.load(a.capture, weights_only=False)
    resolved = json.loads((a.release / 'payload' / 'resolved_config.json').read_text())
    model_cfg = dict(resolved['model'], type='PI05FlowMatching', pretrained_name_or_path=None)
    vla = build_vla_from_cfg(model_cfg)
    vla.load_state_dict(load_file(str(a.release / 'weights' / 'model.safetensors'), device='cpu'), strict=True)
    vla.to(device='cuda', dtype=torch.bfloat16).eval()

    def cuda(v):
        if torch.is_tensor(v):
            return v.cuda()
        return [x.cuda() for x in v] if isinstance(v, list) else v

    worst, rows = 0.0, []
    for rec in data['records']:
        inputs = {k: cuda(v) for k, v in rec['inputs'].items()}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=data['mixed_precision']):
            out = vla.predict_action(**inputs, noise=rec['noise'].cuda()).detach().float().cpu()
        diff = (out - rec['output']).abs()
        worst = max(worst, float(diff.max()))
        rows.append(dict(max=float(diff.max()), mean=float(diff.mean()), bitwise=bool(torch.equal(out, rec['output']))))
    print(json.dumps(dict(samples=len(rows), bitwise_equal=sum(r['bitwise'] for r in rows),
                          max_abs_diff=worst, mean_abs_diff=float(np.mean([r['mean'] for r in rows]))), indent=2))
    return 0 if worst <= a.tolerance else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--release', type=Path, required=True)
    c.add_argument('--samples', type=Path, required=True)
    c.add_argument('--capture', type=Path, required=True)
    c.add_argument('--count', type=int, default=8)
    m = sub.add_parser('compare')
    m.add_argument('--release', type=Path, required=True)
    m.add_argument('--capture', type=Path, required=True)
    m.add_argument('--tolerance', type=float, default=1e-2, help='max |diff| on normalized actions')
    a = p.parse_args()
    raise SystemExit(capture(a) if a.mode == 'capture' else compare(a))


if __name__ == '__main__':
    main()
