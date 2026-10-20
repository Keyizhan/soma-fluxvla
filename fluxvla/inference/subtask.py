"""One inference path for offline evaluation and policy serving.

Inputs use physical robot state and RGB uint8 images. ``actions`` are absolute
14-dimensional joint/gripper targets; this module never commands hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from .preprocessing import normalize_state, resize_rgb
from .subtask_hold import SubtaskHold
from .subtask_stability import SubtaskStability
from .text import cut_at_action_sentinel, subtask_for_action_prompt

CAMERAS = ("images.cam_front", "images.cam_left_wrist", "images.cam_right_wrist")
JOINTS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPERS = (6, 13)


@dataclass(frozen=True)
class InferenceConfig:
    prefix_source: str = "text"
    max_new_tokens: int = 64
    num_denoise_steps: int = 10
    use_text_cache: bool = True
    subtask_switch_confirmations: int = 2
    subtask_update_interval_s: float = 0.0
    subtask_switch_reset_after_s: float = 5.0

    def __post_init__(self):
        if self.prefix_source not in {"text", "generation"}:
            raise ValueError("prefix_source must be text or generation")
        for name in ("max_new_tokens", "num_denoise_steps", "subtask_switch_confirmations"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.prefix_source == "generation" and (
            not self.use_text_cache or self.subtask_switch_confirmations != 1
            or self.subtask_update_interval_s != 0
        ):
            raise ValueError("generation requires fresh cached decode and confirmations=1")
        SubtaskHold(self.subtask_update_interval_s)
        SubtaskStability(self.subtask_switch_confirmations, self.subtask_switch_reset_after_s)


def quantiles(stats, name):
    entry = stats.get("private", stats)[name]
    low, high = (np.asarray(entry[k], dtype=np.float32) for k in ("q01", "q99"))
    if (low.shape != (14,) or high.shape != (14,) or not np.isfinite(low).all()
            or not np.isfinite(high).all() or np.any(high < low)):
        raise ValueError(f"invalid {name} quantiles")
    return {"q01": low, "q99": high}


class SubtaskPolicy:
    """Stateful single-stream policy. Use a separate instance per robot/session."""

    def __init__(self, model, prompts, statistics, *, config=None, camera_keys=CAMERAS,
                 image_size=224, contract=None):
        self.model = model.eval()
        self.prompts = prompts
        self.config = config or InferenceConfig()
        self.camera_keys = tuple(camera_keys)
        if len(self.camera_keys) != 3 or len(set(self.camera_keys)) != 3:
            raise ValueError("three unique camera keys are required")
        self.image_size = int(image_size)
        self.state_stats = quantiles(statistics, "state")
        self.action_stats = quantiles(statistics, "action")
        self.contract = contract or {}
        arithmetic = self.contract.get("state_normalization", "float32_quantiles")
        if arithmetic not in {"float32_quantiles", "float64_quantiles_then_float32"}:
            raise ValueError("unsupported state normalization arithmetic")
        self.quantile_dtype = np.float64 if arithmetic == "float64_quantiles_then_float32" else np.float32
        self.state_stats = {
            k: np.asarray(statistics.get("private", statistics)["state"][k], dtype=self.quantile_dtype)
            for k in ("q01", "q99")
        }
        self.reset()

    def reset(self):
        """Start an episode/session without reusing labels from its predecessor."""
        c = self.config
        self.hold = SubtaskHold(c.subtask_update_interval_s)
        self.stability = SubtaskStability(c.subtask_switch_confirmations,
                                          c.subtask_switch_reset_after_s)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @classmethod
    def from_bundle(cls, path, *, config=None, device="cuda", allow_legacy_contract=False):
        from fluxvla import register_training_components
        from fluxvla.engines import build_tokenizer_from_cfg, build_vla_from_cfg
        from fluxvla.transforms.subtask_fast_prompts import PrepareSubtaskFastPrompts
        from .bundle import verify_bundle_files, validate_conditioning
        from safetensors.torch import load_file

        root = Path(path).expanduser().resolve()
        manifest = verify_bundle_files(root)
        contract = json.loads((root / "inference_contract.json").read_text())
        cfg = config or InferenceConfig()
        validate_conditioning(contract, cfg, allow_legacy_contract=allow_legacy_contract)
        native = json.loads((root / "model_config.json").read_text())
        if native["type"] != "PI05Subtask" or native.get("pretrained_name_or_path") is not None:
            raise ValueError("inference requires a self-contained PI05Subtask full-state bundle")
        if manifest.get("metadata", {}).get("stage") != "stage2":
            raise ValueError("robot policy requires a stage2 bundle with trained flow actions")
        if (native["n_action_steps"] != contract["horizon"]
                or native["max_action_dim"] != contract["padded_action_dim"]
                or native["vision_backbone"]["vision_config"]["image_size"] != contract["image_size"]):
            raise ValueError("model geometry conflicts with the inference contract")
        register_training_components()
        from accelerate import init_empty_weights
        with init_empty_weights(include_buffers=False):
            model = build_vla_from_cfg(native)
        model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True, assign=True)
        dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
        model.to(device=device, dtype=dtype).eval()
        tokenizer = build_tokenizer_from_cfg(
            dict(type="PretrainedTokenizer", model_path=str(root / "tokenizer")))
        prompts = PrepareSubtaskFastPrompts(
            tokenizer, max_len=contract["max_prompt_length"], state_dim=14,
            model_vocab_size=native["llm_backbone"]["vocab_size"],
            fast_vocab_size=native["fast_vocab_size"], fast_skip_tokens=native["fast_skip_tokens"],
            all_causal=contract["all_causal"], low_level_sees_task=contract["low_level_sees_task"])
        aliases = contract["camera_aliases"]
        policy = cls(model, prompts, json.loads((root / "normalization.json").read_text()),
                     config=cfg, camera_keys=[a[0] for a in aliases],
                     image_size=contract["image_size"], contract=contract)
        policy.bundle_path = str(root)
        policy.bundle_metadata = manifest["metadata"]
        policy.legacy_contract_override = (
            "conditioning" not in contract and allow_legacy_contract)
        return policy

    def _images(self, images):
        frames = []
        for key in self.camera_keys:
            if key not in images:
                raise ValueError(f"missing camera {key}")
            frame = resize_rgb(images[key], (self.image_size, self.image_size))
            frames.append(np.ascontiguousarray(frame.transpose(2, 0, 1)))
        pixels = np.concatenate(frames, axis=0)
        return torch.from_numpy(pixels[None]).float().div_(255).mul_(2).sub_(1).to(self.device)

    @torch.inference_mode()
    def infer(self, payload, *, noise=None):
        c, device = self.config, self.device
        state = np.asarray(payload["state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("state must contain 14 finite physical values")
        overall = str(payload["instruction"]).strip()
        if not overall:
            raise ValueError("overall task must not be empty")
        images = self._images(payload["images"])
        masks = torch.ones(1, 3, device=device, dtype=torch.bool)
        bins = normalize_state(state, self.state_stats,
                               quantile_dtype=self.quantile_dtype).astype(np.float64)
        context = self.prompts.context(overall, bins)

        def tensor(value):
            return torch.as_tensor(value)[None].to(device)

        ids, valid, ar = (tensor(context[k]) for k in ("input_ids", "valid_mask", "ar_mask"))
        started = time.perf_counter()
        held = self.hold.get(overall, started)
        refreshed = held is None
        generated = None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if refreshed:
                generated = self.model.generate_subtask(
                    images, ids, valid, masks, eos_token_id=self.prompts.eos_id,
                    pad_token_id=self.prompts.pad_id, max_new_tokens=c.max_new_tokens,
                    use_cache=c.use_text_cache, ar_mask=ar,
                    return_cache=c.prefix_source == "generation", stop_token_ids=None)
                token_ids = generated["token_ids"][0, :int(generated["lengths"][0])].tolist()
                raw_text = self.prompts.decode(token_ids)
                candidate = cut_at_action_sentinel(raw_text)
                if not candidate:
                    raise ValueError("empty generated subtask")
                record = dict(subtask=candidate, raw_text=raw_text, token_ids=token_ids,
                              finished=bool(generated["finished"][0]))
            else:
                record = held["value"]
                candidate, raw_text, token_ids = (record[k] for k in ("subtask", "raw_text", "token_ids"))
            text_seconds = time.perf_counter() - started
            decision = self.stability.propose(overall, candidate, started, fresh=refreshed)
            action_text, prefix_cache = None, None
            if c.prefix_source == "text":
                action_text = subtask_for_action_prompt(decision.active)
                low = self.prompts.low_level(action_text, bins, overall)
                low_ids, low_valid = tensor(low["input_ids"]), tensor(low["valid_mask"])
            else:
                low_ids, low_valid = ids, valid
                prefix_cache = generated.get("cache")
                if prefix_cache is None:
                    raise ValueError("generation cache is missing")
                boundary = self.prompts._encode("\nAction:")
                end = next((i + len(boundary) for i in range(len(token_ids) - len(boundary) + 1)
                            if token_ids[i:i+len(boundary)] == boundary), None)
                if end is None:
                    raise ValueError("generated output did not reach the Action boundary")
                prefix_cache = prefix_cache.prefix(prefix_cache.length - len(token_ids) + end)
            sampled = self.model.sample_actions_given_subtask(
                images, torch.zeros(1, self.model.max_action_dim, device=device),
                low_ids, low_valid, masks, num_steps=c.num_denoise_steps, noise=noise,
                use_cache=c.prefix_source == "text", prefix_cache=prefix_cache)
        rows = sampled.detach().float().cpu().numpy()[0, :, :14]
        if rows.shape != (self.model.n_action_steps, 14) or not np.isfinite(rows).all():
            raise ValueError("invalid sampled action chunk")
        lo, hi = (self.action_stats[k] for k in ("q01", "q99"))
        # Same quantile inverse as the deployed backend; no output clipping.
        semantic = ((rows + 1) * .5 * (hi - lo) + lo).astype(np.float32)
        absolute = semantic.copy()
        absolute[:, JOINTS] += state[list(JOINTS)]
        if not np.isfinite(absolute).all():
            raise ValueError("invalid decoded action chunk")
        if refreshed:
            held = self.hold.commit(overall, started, record)
        self.stability.commit(decision)
        return dict(
            actions=absolute, semantic_actions=semantic, normalized_actions=rows,
            state_used=state.copy(), subtask=decision.active,
            trace=dict(
                low_level_prefix_source=c.prefix_source, action_subtask_text=action_text,
                subtask_candidate=candidate, subtask_active=decision.active,
                subtask_pending=decision.pending, subtask_pending_count=decision.count,
                subtask_switch_reason=decision.reason, subtask_switched=decision.switched,
                subtask_switch_confirmations=c.subtask_switch_confirmations,
                subtask_refreshed=refreshed, subtask_generation_id=held["generation_id"],
                subtask_age_s=started-held["started"],
                subtask_update_interval_s=c.subtask_update_interval_s,
                generated_token_ids=token_ids, raw_generation=raw_text,
                generation_tokens=len(token_ids) if refreshed else 0,
                source_generation_tokens=len(token_ids), max_new_tokens=c.max_new_tokens,
                stop_at_action_sentinel=False, text_cache=c.use_text_cache,
                num_denoise_steps=c.num_denoise_steps, attention_math=self.model.attn_math,
                text_seconds=text_seconds, total_seconds=time.perf_counter()-started,
                legacy_contract_override=getattr(self, "legacy_contract_override", False),
            ))
