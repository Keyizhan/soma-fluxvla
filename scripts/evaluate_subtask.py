"""Free-running subtask/action evaluation through the same policy used by serving.

NPZ: states [N,14] physical rad/meters; images [N,9,H,W] uint8 RGB.
Optional actions [N,T,14] are ABSOLUTE physical GT, action_masks have that shape.
JSON: {records: [{episode, frame, overall_task, subtask (optional)}, ...]}.
No ground-truth subtask is supplied to inference. Input must be episode/frame ordered.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from fluxvla.inference.subtask import InferenceConfig, SubtaskPolicy, JOINTS, GRIPPERS
from fluxvla.inference.text import subtask_for_action_prompt


def evaluate(policy, arrays, records, output, *, seed=1702, limit=None):
    states, images = arrays['states'], arrays['images']
    if len(states) != len(records) or len(images) != len(records):
        raise ValueError('record and observation counts differ')
    if states.shape[1:] != (14,) or images.ndim != 4 or images.shape[1] != 9 or images.dtype != np.uint8:
        raise ValueError('expected physical states [N,14], uint8 images [N,9,H,W]')
    total = len(records) if limit is None else min(limit, len(records))
    if total < 1:
        raise ValueError('no observations selected')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    predictions, previous, closed = [], None, set()
    correct = labelled = switches = eligible = 0
    joint_se = grip_ae = 0.0
    joint_n = grip_n = 0
    started = time.perf_counter()
    with (output/'predictions.jsonl').open('w') as log:
        for i, r in enumerate(records[:total]):
            ep = (r.get('source', ''), r['episode'])
            if previous is None or previous[0] != ep:
                if ep in closed:
                    raise ValueError('episodes must be contiguous')
                if previous is not None:
                    closed.add(previous[0])
                policy.reset()
            elif r['frame'] <= previous[1]:
                raise ValueError('frames must increase within an episode')
            generator = torch.Generator(device=policy.device).manual_seed(seed+i)
            noise = torch.randn((1, policy.model.n_action_steps, policy.model.max_action_dim),
                                generator=generator, device=policy.device)
            result = policy.infer(dict(state=states[i], instruction=r['overall_task'], images={
                key: images[i, c*3:(c+1)*3].transpose(1, 2, 0)
                for c, key in enumerate(policy.camera_keys)}), noise=noise)
            predicted = subtask_for_action_prompt(result['subtask'])
            gt = subtask_for_action_prompt(r['subtask']) if r.get('subtask') else None
            if gt is not None:
                labelled += 1
                correct += predicted == gt
                if previous is not None and previous[0] == ep and previous[2] == gt:
                    eligible += 1
                    switches += predicted != previous[3]
            previous = (ep, r['frame'], gt, predicted)
            actions = result['actions']
            predictions.append(actions)
            if 'actions' in arrays:
                target = arrays['actions'][i]
                mask = arrays['action_masks'][i].astype(bool) if 'action_masks' in arrays else np.ones_like(target, bool)
                if target.shape != actions.shape or mask.shape != actions.shape:
                    raise ValueError('GT action/mask horizon must match the bundle')
                if not np.isfinite(target[mask]).all():
                    raise ValueError('non-finite GT actions')
                error = actions-target
                jmask, gmask = mask[:, JOINTS], mask[:, GRIPPERS]
                joint_se += float(np.square(error[:, JOINTS][jmask]).sum())
                grip_ae += float(np.abs(error[:, GRIPPERS][gmask]).sum())
                joint_n += int(jmask.sum()); grip_n += int(gmask.sum())
            log.write(json.dumps(dict(index=i, episode=r['episode'], frame=r['frame'],
                ground_truth=gt, subtask=result['subtask'], trace=result['trace']), ensure_ascii=False)+'\n')
            log.flush()
    summary = dict(samples=total, seed=seed, prefix_source=policy.config.prefix_source,
        confirmations=policy.config.subtask_switch_confirmations,
        exact_subtask_accuracy=correct/labelled if labelled else None,
        labelled_samples=labelled, within_gt_interval_switches=switches,
        within_gt_interval_comparisons=eligible,
        joint_rmse_rad=(joint_se/joint_n)**.5 if joint_n else None,
        gripper_mae_m=grip_ae/grip_n if grip_n else None,
        elapsed_s=time.perf_counter()-started,
        note='sampled actions/free-running text; not teacher-forced CE or flow loss')
    np.savez_compressed(output/'predictions.npz', actions=np.asarray(predictions))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--observations', required=True)
    p.add_argument('--records', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=1702)
    p.add_argument('--limit', type=int)
    p.add_argument('--confirmations', type=int, default=1, help='1 measures raw predictions; 2 measures deployment filtering')
    p.add_argument('--allow-legacy-contract', action='store_true')
    a = p.parse_args()
    policy = SubtaskPolicy.from_bundle(a.bundle, device=a.device,
        config=InferenceConfig(subtask_switch_confirmations=a.confirmations),
        allow_legacy_contract=a.allow_legacy_contract)
    records = json.loads(Path(a.records).read_text())['records']
    with np.load(a.observations, allow_pickle=False) as arrays:
        print(json.dumps(evaluate(policy, arrays, records, a.output, seed=a.seed, limit=a.limit), indent=2))


if __name__ == '__main__':
    main()
