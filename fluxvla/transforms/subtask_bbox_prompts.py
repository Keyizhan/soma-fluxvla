"""Opt-in grounding prompts; legacy prompt implementation remains unchanged."""

import numpy as np

from fluxvla.engines import TRANSFORMS
from .bbox_grounding import CONTRACT, parse_grounded, validate_goals
from .subtask_fast_prompts import PrepareSubtaskFastPrompts, clean_reference
from .template_masks import token_key_mask


@TRANSFORMS.register_module()
class PrepareSubtaskBboxPrompts(PrepareSubtaskFastPrompts):
    version = "pi05-subtask-bbox-v1"
    grounding_contract = CONTRACT

    def _context_text(self, overall, states):
        return f"Task: {clean_reference(overall)}; State: {self.state_text(states)};\n"

    def context(self, overall_task, states):
        ids = [self.bos_id] + self._encode(self._context_text(overall_task, states))
        if len(ids) > self.max_len:
            raise ValueError("bbox context exceeds token budget")
        n = self.max_len - len(ids)
        return dict(
            input_ids=np.array(ids + [self.pad_id] * n),
            valid_mask=np.array([True] * len(ids) + [False] * n),
            ar_mask=np.array([False] * self.max_len),
        )

    def _goal_segments(self, goals):
        # Explicit segmented encoding is part of this versioned contract, on both
        # the teacher-forced high-level output and rebuilt low-level prefix.
        segments = []
        for i, g in enumerate(validate_goals(goals)):
            if i:
                segments.append((" || ", "format", i, -1))
            segments.append(("Target: ", "format", i, -1))
            if g["target_type"] == "bbox":
                segments.extend([("cam_front xyxy", "type", i, -1), (" [", "format", i, -1)])
                for k, v in enumerate(g["bbox_xyxy_q1000"]):
                    if k:
                        segments.append((", ", "format", i, -1))
                    segments.append((str(v), "coordinate", i, k))
                segments.append(("]", "format", i, -1))
            else:
                segments.append(("none", "type", i, -1))
            segments.append(("; Subtask: ", "format", i, -1))
            segments.append((g["subtask"], "subtask", i, -1))
        return segments

    def high_level(self, overall_task, states, subtask, fast_token_ids=None, grounding=None):
        goals = validate_goals(grounding)
        if " || ".join(g["subtask"] for g in goals) != subtask:
            raise ValueError("grounding subtask disagrees with original text")
        prefix = [self.bos_id] + self._encode(self._context_text(overall_task, states))
        tokens = list(prefix)
        regions = {name: [-1] * len(prefix) for name in ("type", "coordinate", "format")}
        sub, act, key, codes = ([False] * len(prefix) for _ in range(4))
        for text, region, goal, coord in self._goal_segments(goals):
            ids = self._encode(text)
            if not ids:
                raise ValueError("empty grounding token segment")
            tokens.extend(ids)
            for name in regions:
                group = goal * 4 + coord if name == "coordinate" else goal
                regions[name].extend([group if region == name else -1] * len(ids))
            sub.extend([region == "subtask"] * len(ids))
            act.extend([False] * len(ids))
            codes.extend([False] * len(ids))
            key.extend(
                token_key_mask(self.tokenizer, text, ids, self.subtask_mask_spec).tolist()
                if region == "subtask"
                else [False] * len(ids)
            )
        tail = self._encode("\nAction: ")
        fast = [] if fast_token_ids is None else self.fast_to_model(fast_token_ids).tolist()
        ending = ([] if fast_token_ids is None else self._encode("|")) + [self.eos_id]
        tokens.extend(tail + fast + ending)
        length = len(tail + fast + ending)
        for values in regions.values():
            values.extend([-1] * length)
        sub.extend([fast_token_ids is None] * length)
        act.extend([fast_token_ids is not None] * length)
        key.extend([False] * length)
        codes.extend([False] * len(tail) + [True] * len(fast) + [False] * len(ending))
        size = len(tokens)
        if size > self.max_len:
            raise ValueError(f"Grounded prompt needs {size} tokens > {self.max_len}; no truncation")
        pad = self.max_len - size

        def boolean(values):
            return np.array(values + [False] * pad, dtype=bool)

        ids = np.array(tokens + [self.pad_id] * pad, dtype=np.int64)
        loss = np.array([False] * len(prefix) + [True] * (size - len(prefix)) + [False] * pad)
        packed = dict(
            input_ids=ids,
            labels=np.where(loss, ids, -100),
            loss_mask=loss,
            valid_mask=boolean([True] * size),
            ar_mask=loss.copy(),
            subtask_region_mask=boolean(sub),
            action_region_mask=boolean(act),
            subtask_key_mask=boolean(key),
            action_code_mask=boolean(codes),
        )
        packed["subtask_template_mask"] = (
            packed["subtask_region_mask"] & ~packed["subtask_key_mask"]
        )
        for name, values in regions.items():
            packed["bbox_" + name + "_groups"] = np.array(values + [-1] * pad, dtype=np.int64)
        self.check(packed)
        return packed

    def low_level(self, subtask, states, overall_task=None, grounding=None):
        goals = parse_grounded(subtask) if grounding is None else validate_goals(grounding)
        ids = [self.bos_id] + self._encode(f"State: {self.state_text(states)};\n")
        for text, _, _, _ in self._goal_segments(goals):
            ids.extend(self._encode(text))
        if len(ids) > self.max_len:
            raise ValueError("grounded low-level prefix exceeds token budget")
        n = self.max_len - len(ids)
        return dict(
            input_ids=np.array(ids + [self.pad_id] * n, dtype=np.int64),
            valid_mask=np.array([True] * len(ids) + [False] * n),
        )

    def decode_subtask(self, ids):
        return self.decode(ids).split("\nAction:", 1)[0].strip()

    def __call__(self, inputs):
        high = self.high_level(
            inputs["overall_task"],
            inputs["states"],
            inputs["subtask"],
            inputs.get("fast_token_ids"),
            grounding=inputs["grounding_json"],
        )
        low = self.low_level(
            inputs["subtask"], inputs["states"], grounding=inputs["grounding_json"]
        )
        out = dict(inputs)
        out.update(
            {f"hl_{key}": value for key, value in high.items() if not key.startswith("bbox_")}
        )
        out["hl_bbox_groups"] = {
            name: high["bbox_" + name + "_groups"] for name in ("type", "coordinate", "format")
        }
        out.update(
            ll_input_ids=low["input_ids"],
            ll_valid_mask=low["valid_mask"],
            has_hl=np.array(True),
            has_action=np.array(True),
        )
        return out
