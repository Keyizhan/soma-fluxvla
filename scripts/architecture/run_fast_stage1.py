# Historical architecture/EMA experiments. Canonical two-stage trainer: fluxvla.training.subtask.
"""Stage-1 / hybrid FAST training over a fixture set (subtask CE + FAST CE + flow).

Stage 1 (`--preset stage1_fast`) trains the VLM on discrete actions; `--preset hybrid`
adds the flow branch; `hybrid_insulated` additionally stops flow gradients at the prefix.

Memory: fixtures are read through a memory map when `samples.npz` is uncompressed, so
images are fetched per batch instead of being held in RAM. Use
`scripts/architecture/prepare_fast_tokens.py --uncompressed` when preparing big data.

    python scripts/architecture/run_fast_stage1.py --fixtures <dir> --tokenizer <dir> \
        --pretrained <pi05_base> --preset stage1_fast --steps 600 --batch-size 4 \
        --device cuda --output <dir>
"""
import argparse
import hashlib
import json
import os
import runpy
import time
from pathlib import Path

import numpy as np
import torch
from fluxvla.training.ema import EMA, ema_weights, trainable_parameters
import torch.distributed as dist
import torch.nn.functional as F

from fluxvla import register_training_components
register_training_components()
from fluxvla.engines import build_tokenizer_from_cfg, build_vla_from_cfg
from fluxvla.transforms.subtask_fast_prompts import (PrepareSubtaskFastPrompts,
                                                    clean_reference)

REPO = Path(__file__).resolve().parents[2]


def ddp_setup(device_arg):
    """Return (rank, world_size, device). No-op outside torchrun."""
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return 0, 1, torch.device(device_arg)
    backend = 'nccl' if str(device_arg).startswith('cuda') else 'gloo'
    if not (dist.is_available() and dist.is_initialized()):
        dist.init_process_group(backend=backend)
    local = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local}') if backend == 'nccl' else torch.device('cpu')
    return dist.get_rank(), dist.get_world_size(), device


def ddp_cleanup(rank, world_size):
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


class HighLevelOnly(torch.nn.Module):
    """DDP needs the called method to be `forward`; stage 1 only runs the text branch."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model.forward_high_level(*args, **kwargs)


def high_level_of(train_model):
    """Return a callable for the text branch that keeps DDP's hooks in the loop."""
    wrapped = isinstance(train_model, torch.nn.parallel.DistributedDataParallel)
    inner = train_model.module if wrapped else train_model
    if isinstance(inner, HighLevelOnly):
        return train_model                      # DDP.forward -> HighLevelOnly.forward
    return inner.forward_high_level


def clause_set(text):
    """Normalised set of clauses; parallel labels join two arms with '||'."""
    parts = str(text).lower().split('||')
    return tuple(sorted(part.strip().strip('.').strip() for part in parts if part.strip()))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Fixtures:
    """Fixture reader: prompt data in RAM, images through a memory map."""

    def __init__(self, root):
        self.root = Path(root)
        manifest = json.loads((self.root / 'samples.json').read_text())
        self.records = manifest['records']
        try:
            data = np.load(self.root / 'samples.npz', allow_pickle=False, mmap_mode='r')
            self.mapped = True
        except ValueError:
            data = np.load(self.root / 'samples.npz', allow_pickle=False)
            self.mapped = False
        self._data = data
        self.states = np.asarray(data['states'])
        self.img_masks = np.asarray(data['img_masks'])
        fast = np.load(self.root / 'fast_tokens.npz', allow_pickle=False)
        flat, lengths = fast['fast_ids'], fast['fast_lengths']
        if not (len(self.records) == len(self.states) == len(lengths)):
            raise SystemExit('fixture arrays disagree on sample count')
        self.fast_ids, offset = [], 0
        for n in lengths:
            self.fast_ids.append(flat[offset:offset + int(n)])
            offset += int(n)
        if offset != len(flat):
            raise SystemExit('FAST buffer not fully consumed')
        self.actions = data['actions'] if 'actions' in data else None
        self.action_masks = data['action_masks'] if 'action_masks' in data else None

    def __len__(self):
        return len(self.records)

    def texts(self, index):
        record = self.records[index]
        return record['overall_task'], record['subtask']

    def image(self, index, image_size):
        tensor = torch.from_numpy(np.asarray(self._data['images'][index]).astype(np.float32)
                                  / 255.0 * 2 - 1)
        if tensor.shape[-1] != image_size:
            tensor = F.interpolate(tensor[None], size=(image_size, image_size),
                                   mode='bilinear', align_corners=False)[0]
        return tensor

    def actions_for(self, index, n_action_steps=None):
        if self.actions is None:
            return None, None
        actions = torch.from_numpy(np.asarray(self.actions[index]))
        masks = torch.from_numpy(np.asarray(self.action_masks[index]))
        if n_action_steps is not None:
            # embed_suffix derives its attention schedule from n_action_steps, so the
            # window length must match the model config exactly.
            actions, masks = actions[:n_action_steps], masks[:n_action_steps]
        return actions, masks

    def subset(self, split):
        return [i for i, r in enumerate(self.records) if r['split'] == split]


def prepare(fx, prompts, state_dim):
    packed = []
    for i in range(len(fx)):
        overall, subtask = fx.texts(i)
        states = np.asarray(fx.states[i], dtype=np.float64)[:state_dim]
        packed.append(dict(
            high=prompts.high_level(overall, states, subtask, fx.fast_ids[i]),
            low=prompts.low_level(subtask, states, overall),
            img_masks=torch.as_tensor(fx.img_masks[i])))
    return packed


def collate(items, indices, fx, image_size, device, with_actions, n_action_steps=None):
    batch = {}
    keys = [k for k in items[0]['high'] if isinstance(items[0]['high'][k], np.ndarray)]
    for key in keys:
        batch[f'high_{key}'] = torch.as_tensor(np.stack([it['high'][key] for it in items])).to(device)
    low_keys = [k for k in items[0]['low'] if isinstance(items[0]['low'][k], np.ndarray)]
    for key in low_keys:
        batch[f'low_{key}'] = torch.as_tensor(np.stack([it['low'][key] for it in items])).to(device)
    batch['images'] = torch.stack([fx.image(i, image_size) for i in indices]).to(device)
    batch['img_masks'] = torch.stack([it['img_masks'] for it in items]).to(device)
    if with_actions:
        actions, masks = zip(*(fx.actions_for(i, n_action_steps) for i in indices))
        if actions[0] is None:
            raise SystemExit('preset needs actions but fixtures have none')
        batch['actions'] = torch.stack(actions).to(device)
        batch['action_masks'] = torch.stack(masks).to(device)
    return batch


def build_model(preset, smoke, device, pretrained=None, prompt_options=None):
    ns = runpy.run_path(str(REPO / 'configs/pi05_subtask_fast.py'))
    cfg = dict(ns['PRESETS'][preset]['model'])
    if prompt_options is not None:
        cfg['low_level_text_attention'] = 'causal' if prompt_options['all_causal'] else 'bidirectional'
        cfg['low_level_task_independent'] = not prompt_options['low_level_sees_task']
    cfg['pretrained_name_or_path'] = None if smoke else pretrained
    if smoke:
        cfg.update(proj_width=16, n_action_steps=4)
        for key in ('llm_backbone', 'llm_expert'):
            cfg[key].update(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8)
        cfg['llm_expert']['adarms_cond_dim'] = 16
        cfg['projector'].update(in_dim=16, out_dim=16)
        cfg['action_in_proj'].update(out_dim=16)
        cfg['action_out_proj'].update(in_dim=16)
        for key in ('time_mlp_in', 'time_mlp_out'):
            cfg[key].update(in_dim=16, out_dim=16)
        cfg['vision_backbone']['vision_config'].update(hidden_size=16, intermediate_size=32,
                                                       num_hidden_layers=1, image_size=16,
                                                       patch_size=8)
    model = build_vla_from_cfg(cfg)
    if cfg.get('pretrained_name_or_path'):
        model.from_pretrained()
        print(f'model: loaded pretrained weights from {cfg["pretrained_name_or_path"]}', flush=True)
    else:
        print('model: random init (no pretrained weights)', flush=True)
    return model.to(device), cfg


@torch.no_grad()
def evaluate(model, packed, prompts, indices, device, fx, state_dim, image_size,
             max_new_tokens=24, limit=None):
    model.eval()
    rows = []
    for i in (indices[:limit] if limit else indices):
        batch = collate([packed[i]], [i], fx, image_size, device, with_actions=False)
        out = model.forward_high_level(
            batch['images'], batch['high_input_ids'], batch['high_valid_mask'],
            batch['high_ar_mask'], batch['high_labels'], batch['img_masks'],
            hl_subtask_region_mask=batch['high_subtask_region_mask'],
            hl_action_region_mask=batch['high_action_region_mask'])
        overall, subtask = fx.texts(i)
        states = np.asarray(fx.states[i], dtype=np.float64)[:state_dim]
        context = prompts.context(overall, states)
        gen = model.generate_subtask(
            batch['images'], torch.as_tensor(context['input_ids'])[None].to(device),
            torch.as_tensor(context['valid_mask'])[None].to(device), batch['img_masks'],
            eos_token_id=prompts.eos_id, pad_token_id=prompts.pad_id,
            max_new_tokens=max_new_tokens,
            ar_mask=torch.as_tensor(context['ar_mask'])[None].to(device))
        text = prompts.decode(gen['token_ids'][0].tolist())
        target = clean_reference(subtask).rstrip('.')
        rows.append(dict(index=int(i), loss_token=out['loss_token'].item(),
                         loss_subtask=out['loss_subtask'].item(),
                         loss_fast=float(out['loss_action_ce'].detach()),
                         target=target, predicted=text, eos=bool(gen['finished'][0]),
                         exact=(bool(gen['finished'][0]) and text == target),
                         # Long / parallel labels make exact match brittle; also score the
                         # clause sets so a swapped clause order is not counted as an error.
                         clauses_exact=clause_set(text) == clause_set(target),
                         is_parallel=len(clause_set(target)) > 1))
    model.train()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixtures', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--preset', default='stage1_fast')
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--accum', type=int, default=1, help='gradient accumulation steps')
    ap.add_argument('--eval-every', type=int, default=1, help='run eval every N eval_steps')
    ap.add_argument('--eval-steps', type=int, default=0, help='0 = evaluate only at the end')
    ap.add_argument('--eval-limit', type=int, default=0, help='0 = all validation samples')
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--pretrained', default=None)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--resume-step', action='store_true',
                    help='continue the previous step counter (default: restart it at 0)')
    ap.add_argument('--resume-optimizer', action='store_true',
                    help='also restore AdamW state (off by default: stage boundaries reset it)')
    ap.add_argument('--state-dim', type=int, default=14)
    ap.add_argument('--all-causal', dest='all_causal', action='store_true', default=None,
                    help='force legacy all-causal text (default: current preset full-prefix contract)')
    ap.add_argument('--no-all-causal', dest='all_causal', action='store_false',
                    help='bidirectional conditioning prefix')
    ap.add_argument('--low-level-sees-task', dest='low_level_sees_task', action='store_true',
                    default=None,
                    help='force legacy Task+Subtask action context (default: current preset subtask-only)')
    ap.add_argument('--low-level-subtask-only', dest='low_level_sees_task',
                    action='store_false',
                    help='p(a|o,s) without the overall task')
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--seed', type=int, default=1702)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    rank, world_size, device = ddp_setup(args.device)
    out = Path(args.output)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed + rank)
    rng = np.random.default_rng(args.seed + rank)

    fx = Fixtures(args.fixtures)
    if not args.smoke and not args.pretrained:
        raise SystemExit('a real run requires --pretrained <pi05_base dir>')
    tokenizer = build_tokenizer_from_cfg(dict(type='PretrainedTokenizer', model_path=args.tokenizer))
    ns = runpy.run_path(str(REPO / 'configs/pi05_subtask_fast.py'))
    preset = ns['PRESETS'][args.preset]['model']
    contract = ns['PRESETS'][args.preset]['contract']
    prompt_options = {key: contract[key] for key in ('all_causal', 'low_level_sees_task',
        'low_level_prompt_format', 'low_level_action_tail', 'low_level_eos', 'low_level_segmented')}
    for key in ('all_causal', 'low_level_sees_task'):
        if getattr(args, key) is not None:
            prompt_options[key] = getattr(args, key)
    if prompt_options['low_level_sees_task']:
        prompt_options['low_level_prompt_format'] = 'legacy'
    prompts = PrepareSubtaskFastPrompts(
        tokenizer, max_len=args.max_len, state_dim=args.state_dim,
        model_vocab_size=int(preset['llm_backbone']['vocab_size']),
        fast_vocab_size=preset['fast_vocab_size'],
        fast_skip_tokens=preset['fast_skip_tokens'],
        **prompt_options)
    model, cfg = build_model(args.preset, args.smoke, device, args.pretrained, prompt_options)
    model_for_eval = model
    train_model = model
    if world_size > 1:
        if rank == 0:
            print(f'DDP: world_size={world_size}, device={device}', flush=True)
        inner = model if model.action_loss_weight > 0 else HighLevelOnly(model)
        train_model = torch.nn.parallel.DistributedDataParallel(
            inner, device_ids=[device.index] if device.type == 'cuda' else None,
            find_unused_parameters=True)
    image_size = int(cfg['vision_backbone']['vision_config']['image_size'])

    packed = prepare(fx, prompts, args.state_dim)
    train = fx.subset('train')
    validation = fx.subset('validation')
    if not train:
        raise SystemExit('no training split in fixtures')
    with_actions = model.action_loss_weight > 0
    weights = dict(token=model.token_loss_weight, subtask=model.hl_loss_weight, fast=model.fast_loss_weight,
                   flow=model.action_loss_weight)
    loss_contract = dict(text_loss_mode=model.text_loss_mode, weights=weights)
    lengths = [len(r) for r in fx.fast_ids]
    header = dict(contract=prompts.variant, preset=args.preset, smoke=bool(args.smoke),
                  all_causal=bool(prompts.all_causal),
                  low_level_sees_task=bool(prompts.low_level_sees_task),
                  weights=weights, text_loss_mode=model.text_loss_mode,
                  train=len(train), validation=len(validation),
                  batch_size=args.batch_size, accum=args.accum,
                  mapped_images=bool(fx.mapped),
                  fast_tokens=dict(min=int(min(lengths)), max=int(max(lengths)),
                                   mean=float(np.mean(lengths))),
                  prompt_len=dict(min=int(min(p['high']['valid_mask'].sum() for p in packed)),
                                  max=int(max(p['high']['valid_mask'].sum() for p in packed))),
                  device=str(device), world_size=world_size, steps=args.steps, lr=args.lr,
                  seed=args.seed, per_gpu_batch=args.batch_size,
                  exact_beta_time=bool(model.exact_beta_time))
    print(json.dumps(header), flush=True)

    if not with_actions:
        # Stage 1 never runs the flow branch, so its parameters get no gradient. Freezing
        # them matches that semantics and drops their optimizer state (~20 GB), which is
        # what lets the per-GPU batch grow. DDP still needs find_unused_parameters=True.
        expert_prefixes = ('llm_expert', 'action_in_proj', 'action_out_proj',
                           'time_mlp_in', 'time_mlp_out')
        frozen = 0
        for name, param in model.named_parameters():
            if name.startswith(expert_prefixes) and param.requires_grad:
                param.requires_grad_(False)
                frozen += param.numel()
        if rank == 0:
            print(f'stage-1: froze the action branch ({frozen/1e6:.1f} M params)', flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        if (args.resume_optimizer or args.resume_step) and state.get('loss_contract') != loss_contract:
            raise ValueError('resume objective differs or is missing; use weight-only warm start')
        missing, unexpected = model.load_state_dict(state['model_delta'], strict=False)
        if unexpected:
            raise SystemExit(f'unexpected keys in resume checkpoint: {unexpected[:5]}')
        if args.resume_optimizer and state.get('optimizer') is not None:
            optimizer.load_state_dict(state['optimizer'])
            print('resumed optimizer state', flush=True)
        # Stage 2 starts a fresh step counter: the saved step belongs to the previous stage.
        start_step = int(state.get('step', 0)) if args.resume_step else 0
        print(f'resumed from {args.resume} at step {start_step}', flush=True)

    model.train()
    history, evaluations = [], []
    start = time.perf_counter()
    effective = max(1, args.accum)
    # Current presets use rank/microbatch-local valid means; accumulation and DDP
    # average these objectives equally, not as one globally concatenated token batch.
    shard = train[rank::world_size] if world_size > 1 else train
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        totals = dict(loss=0.0, loss_token=0.0, loss_subtask=0.0, loss_fast=0.0, loss_flow=None)
        for _ in range(effective):
            picked = rng.integers(0, len(shard), size=args.batch_size).tolist()
            sample_indices = [shard[i] for i in picked]
            batch = collate([packed[i] for i in sample_indices], sample_indices, fx,
                            image_size, device, with_actions,
                            int(cfg['n_action_steps']) if with_actions else None)
            if with_actions:
                result = train_model(
                    images=batch['images'], states=torch.zeros(len(sample_indices), 32, device=device),
                    img_masks=batch['img_masks'],
                    hl_input_ids=batch['high_input_ids'],
                    hl_valid_mask=batch['high_valid_mask'],
                    hl_ar_mask=batch['high_ar_mask'], hl_labels=batch['high_labels'],
                    hl_subtask_region_mask=batch['high_subtask_region_mask'],
                    hl_action_region_mask=batch['high_action_region_mask'],
                    ll_input_ids=batch['low_input_ids'],
                    ll_valid_mask=batch['low_valid_mask'],
                    actions=batch['actions'], action_masks=batch['action_masks'])
                loss = result['loss']
                token_loss = result['loss_token'].detach()
                subtask_loss, fast_loss, flow_loss = (result['loss_subtask'],
                                                      result['loss_fast'], result['loss_action'])
            else:
                high = high_level_of(train_model)(
                    batch['images'], batch['high_input_ids'], batch['high_valid_mask'],
                    batch['high_ar_mask'], batch['high_labels'], batch['img_masks'],
                    hl_subtask_region_mask=batch['high_subtask_region_mask'],
                    hl_action_region_mask=batch['high_action_region_mask'])
                token_loss = high['loss_token'].detach()
                loss = (weights['token'] * high['loss_token'] if model.text_loss_mode == 'unified_token'
                        else weights['subtask'] * high['loss_subtask']
                        + weights['fast'] * high['loss_action_ce'])
                subtask_loss, fast_loss, flow_loss = (high['loss_subtask'].detach(),
                                                      high['loss_action_ce'].detach(), None)
            if not torch.isfinite(loss):
                raise SystemExit(f'non-finite loss at step {step} on rank {rank}')
            (loss / effective).backward()
            totals['loss'] += float(loss.detach()) / effective
            totals['loss_token'] += float(token_loss) / effective
            totals['loss_subtask'] += float(subtask_loss) / effective
            totals['loss_fast'] += float(fast_loss) / effective
            if flow_loss is not None:
                totals['loss_flow'] = (totals['loss_flow'] or 0.0) + float(flow_loss) / effective
            del loss, subtask_loss, fast_loss, flow_loss
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(grad_norm):
            raise SystemExit(f'non-finite gradient at step {step} on rank {rank}')
        optimizer.step()
        row = dict(step=step, grad_norm=round(float(grad_norm), 3),
                   loss=round(totals['loss'], 6), loss_token=round(totals['loss_token'], 6),
                   loss_subtask=round(totals['loss_subtask'], 6),
                   loss_fast=round(totals['loss_fast'], 6),
                   loss_flow=(None if totals['loss_flow'] is None
                              else round(totals['loss_flow'], 6)))
        history.append(row)
        if rank == 0 and (step == 1 or step % max(1, args.steps // 60) == 0 or step == args.steps):
            row['t'] = round(time.perf_counter() - start, 1)
            print(json.dumps(row), flush=True)
        if (args.eval_steps and step % args.eval_steps == 0 and validation and rank == 0
                and (step % (args.eval_steps * args.eval_every) == 0 if args.eval_every > 1 else True)):
            evaluations.append(dict(step=step, validation=evaluate(
                model_for_eval, packed, prompts, validation, device, fx,
                args.state_dim, image_size, limit=args.eval_limit or None)))
    seconds = time.perf_counter() - start

    final = (evaluate(model, packed, prompts, validation, torch.device(args.device), fx,
                      args.state_dim, image_size, limit=args.eval_limit or None)
             if validation else [])
    report = dict(header=header, history=history, evaluations=evaluations,
                  final_validation=final, seconds=seconds,
                  fixtures=dict(path=str(Path(args.fixtures).resolve()),
                                samples_json_sha256=sha256(Path(args.fixtures) / 'samples.json'),
                                fast_tokens_sha256=sha256(Path(args.fixtures) / 'fast_tokens.npz')),
                  peak_allocated_bytes=(torch.cuda.max_memory_allocated()
                                        if torch.cuda.is_available() else None))
    if rank == 0:
        torch.save(dict(model_delta={n: p.detach().cpu() for n, p in model.named_parameters()
                                 if p.requires_grad},
                    optimizer=optimizer.state_dict(), step=args.steps,
                    contract=prompts.version, loss_contract=loss_contract,
                    preset=args.preset, smoke=bool(args.smoke)),
                   out / 'model_delta.pt')
    if rank != 0:
        ddp_cleanup(rank, world_size)
        return
    report['checkpoint'] = str(out / 'model_delta.pt')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    exact = sum(r['exact'] for r in final)
    clauses = sum(r['clauses_exact'] for r in final)
    parallel = [r for r in final if r['is_parallel']]
    print(json.dumps(dict(
        complete=str(out / 'report.json'), seconds=round(seconds, 2), world_size=world_size,
        loss_first=history[0]['loss'], loss_last=history[-1]['loss'],
        val_exact=f'{exact}/{len(final)}' if final else 'n/a',
        val_clauses=f'{clauses}/{len(final)}' if final else 'n/a',
        val_parallel_clauses=(f"{sum(r['clauses_exact'] for r in parallel)}/{len(parallel)}"
                              if parallel else 'n/a')), indent=2))


if __name__ == '__main__':
    main()
