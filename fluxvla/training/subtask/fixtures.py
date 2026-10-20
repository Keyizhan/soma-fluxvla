"""Shared fixture loading, batching and model construction for subtask training."""
import copy
import json
import os
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from fluxvla import register_training_components
from fluxvla.engines import build_vla_from_cfg
register_training_components()
def ddp_setup(device_arg):
    """Return (rank, world_size, device). No-op outside torchrun."""
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return 0, 1, torch.device(device_arg)
    backend = 'nccl' if str(device_arg).startswith('cuda') else 'gloo'
    if not (dist.is_available() and dist.is_initialized()):
        # The default 10-minute NCCL watchdog kills the job whenever one rank does
        # non-collective work (validation, writing a 5.8 GB delta to the shared disk).
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=60))
    local = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local}') if backend == 'nccl' else torch.device('cpu')
    return dist.get_rank(), dist.get_world_size(), device

def ddp_cleanup(rank, world_size):
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

class Fixtures:
    """Fixture reader: prompt data in RAM, images through a memory map."""

    def __init__(self, root):
        self.root = Path(root)
        manifest = json.loads((self.root / 'samples.json').read_text())
        self.records = manifest['records']
        if (self.root / 'arrays').is_dir():
            data = {p.stem: np.load(p, allow_pickle=False, mmap_mode='r')
                    for p in (self.root / 'arrays').glob('*.npy')}
            self.mapped = True
        else:
            data = np.load(self.root / 'samples.npz', allow_pickle=False)
            self.mapped = False  # NPZ members are never memory mapped.
        self._data = data
        # NpzFile.__getitem__ re-reads the whole member on every access. Binding the array
        # once turns "1.89 GB per image" into "1.89 GB per Fixtures()". Measured: 64 images
        # per optimizer step cost 121 GB of reads, which is what pinned the CPU and left
        # both GPUs at 0% utilisation.
        self._images = data['images']
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
        # Keep uint8 on the host: the H2D copy is 4x smaller and the normalisation plus the
        # numpy->float32 conversion used to cost ~116 MB of CPU work per optimizer step.
        tensor = torch.from_numpy(np.array(self._images[index], copy=True))
        if tensor.shape[-1] != image_size:
            tensor = F.interpolate(tensor[None].float(), size=(image_size, image_size),
                                   mode='bilinear', align_corners=False)[0]
        return tensor

    def actions_for(self, index, n_action_steps=None):
        if self.actions is None:
            return None, None
        actions = torch.from_numpy(np.array(self.actions[index], copy=True))
        masks = torch.from_numpy(np.array(self.action_masks[index], copy=True))
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
            high=prompts.high_level(overall, states, subtask,
                                    fx.fast_ids[i] if len(fx.fast_ids[i]) else None),
            low=prompts.low_level(subtask, states, overall),
            img_masks=torch.as_tensor(np.array(fx.img_masks[i], copy=True))))
    return packed

def collate(items, indices, fx, image_size, device, with_actions, n_action_steps=None):
    batch = {}
    keys = [k for k in items[0]['high'] if isinstance(items[0]['high'][k], np.ndarray)]
    for key in keys:
        batch[f'high_{key}'] = torch.as_tensor(np.stack([it['high'][key] for it in items])).to(device)
    low_keys = [k for k in items[0]['low'] if isinstance(items[0]['low'][k], np.ndarray)]
    for key in low_keys:
        batch[f'low_{key}'] = torch.as_tensor(np.stack([it['low'][key] for it in items])).to(device)
    images = torch.stack([fx.image(i, image_size) for i in indices]).to(device)
    if images.dtype == torch.uint8:
        images = images.float().div(255.0).mul(2.0).sub(1.0)
    batch['images'] = images
    batch['img_masks'] = torch.stack([it['img_masks'] for it in items]).to(device)
    if with_actions:
        actions, masks = zip(*(fx.actions_for(i, n_action_steps) for i in indices))
        if actions[0] is None:
            raise SystemExit('preset needs actions but fixtures have none')
        batch['actions'] = torch.stack(actions).to(device)
        batch['action_masks'] = torch.stack(masks).to(device)
    return batch

def build_model(preset, smoke, device, pretrained=None, dtype=torch.float32,
                attn_math='bf16'):
    from fluxvla.training.presets.pi05_subtask_fast import PRESETS
    cfg = copy.deepcopy(PRESETS[preset]['model'])
    cfg['pretrained_name_or_path'] = None if smoke else pretrained
    cfg['attn_math'] = attn_math
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
    if device.type != 'cuda':
        dtype = torch.float32          # CPU smoke runs in fp32
    # Original fluxvla policy (ddp_train_runner._move_vla_to_device): cast the whole model
    # to bf16 after loading, then run the step under bf16 autocast.
    return model.to(device=device, dtype=dtype), cfg
