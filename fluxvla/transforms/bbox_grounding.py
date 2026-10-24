"""Versioned, parameter-free grounding format shared by data and inference."""

import json
import math
import re

CONTRACT = "fabric-goal-bbox-v1"
PARTS = ("type", "coordinate", "format")


def resolve_bbox(value):
    if not isinstance(value, dict):
        raise ValueError("bbox must be an object; omit it to disable grounding")
    result = dict(
        contract=CONTRACT, weight=5.0, type_weight=1.0, coordinate_weight=1.0, format_weight=0.1
    )
    if set(value) - set(result):
        raise ValueError("unknown bbox configuration key")
    result.update(value)
    if result["contract"] != CONTRACT:
        raise ValueError("unsupported bbox contract")
    for key in ("weight", "type_weight", "coordinate_weight", "format_weight"):
        v = result[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"bbox {key} must be finite and nonnegative")
    return result


def validate_goals(value):
    goals = json.loads(value) if isinstance(value, str) else value
    if not isinstance(goals, list) or not goals:
        raise ValueError("grounding requires a nonempty goal list")
    for goal in goals:
        if not isinstance(goal.get("subtask"), str) or not goal["subtask"].strip():
            raise ValueError("grounding goal requires subtask")
        kind = goal.get("target_type")
        if kind not in ("bbox", "none"):
            raise ValueError("missing grounding is unsupported for joint bbox/action training")
        if not goal.get("type_loss_valid", True):
            raise ValueError("invalid grounding type supervision")
        operation = goal.get("operation")
        if operation is not None:
            expected = "bbox" if operation == "place_fabric_in_basket" else "none"
            if (
                operation
                not in (
                    "place_fabric_in_basket",
                    "retrieve_fabric_to_staging",
                    "flip_fabric_face_up_on_staging",
                )
                or kind != expected
            ):
                raise ValueError("grounding type disagrees with primitive")
        box = goal.get("bbox_xyxy_q1000")
        if kind == "bbox":
            if goal.get("camera") != "images.cam_front" or not goal.get(
                "coordinate_loss_valid", True
            ):
                raise ValueError("bbox requires valid front camera coordinates")
            if not isinstance(box, list) or len(box) != 4 or any(type(x) is not int for x in box):
                raise ValueError("bbox requires four integer coordinates")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                raise ValueError("bbox outside coordinate domain or unordered")
        elif box is not None or goal.get("coordinate_loss_valid", False):
            raise ValueError("none must not contain coordinate supervision")
    return goals


def render_goals(goals):
    rendered = []
    for g in validate_goals(goals):
        target = (
            "cam_front xyxy [" + ", ".join(map(str, g["bbox_xyxy_q1000"])) + "]"
            if g["target_type"] == "bbox"
            else "none"
        )
        rendered.append(f"Target: {target}; Subtask: {g['subtask']}")
    return " || ".join(rendered)


def parse_grounded(text):
    """Strict complete-output parser. Never repairs or clips invalid predictions."""
    goals = []
    for clause in re.split(r"\s*\|\|\s*", text.strip()):
        match = re.fullmatch(
            r"Target\s*:\s*(none|cam_front\s+xyxy\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\])\s*;\s*Subtask\s*:\s*(.+)",
            clause.strip(),
        )
        if not match:
            raise ValueError("invalid grounded generation format")
        box = None if match[1] == "none" else [int(match[i]) for i in range(2, 6)]
        goals.append(
            dict(
                target_type="none" if box is None else "bbox",
                bbox_xyxy_q1000=box,
                camera=None if box is None else "images.cam_front",
                subtask=match[6],
            )
        )
    return validate_goals(goals)


def semantic_key(text):
    return tuple(g["subtask"].strip().lower().rstrip(".; ") for g in parse_grounded(text))


def localization_metrics(prediction, reference):
    """End-to-end metrics: malformed/missing predictions count as zero IoU."""
    reference = validate_goals(reference)
    try:
        predicted = parse_grounded(prediction)
        legal = len(predicted) == len(reference)
    except ValueError:
        predicted, legal = [], False
    result = dict(
        valid_format=int(legal),
        examples=1,
        type_correct=0,
        goals=len(reference),
        bbox_count=0,
        iou_sum=0.0,
        center_error_sum=0.0,
        valid_bbox_count=0,
        missed_bbox=0,
    )
    for i, truth in enumerate(reference):
        pred = predicted[i] if legal else None
        result["type_correct"] += int(
            pred is not None and pred["target_type"] == truth["target_type"]
        )
        if truth["target_type"] != "bbox":
            continue
        result["bbox_count"] += 1
        if pred is None or pred["target_type"] != "bbox":
            result["missed_bbox"] += 1
            continue
        a, b = truth["bbox_xyxy_q1000"], pred["bbox_xyxy_q1000"]
        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0, min(a[3], b[3]) - max(a[1], b[1])
        )
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
        result["iou_sum"] += intersection / union
        result["center_error_sum"] += math.hypot(
            (a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2
        )
        result["valid_bbox_count"] += 1
    return result
