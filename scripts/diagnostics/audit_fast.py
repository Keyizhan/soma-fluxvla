"""Read-only FAST audit: logs, batch coverage, codec reconstruction and teacher metrics."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def audit_logs(source, output, full=None):
    from fluxvla.training.subtask.common import epoch_batches, lr_at

    result = {}
    for stage in ["stage1", "stage2"]:
        rows = [json.loads(x) for x in (source / (stage + ".jsonl")).read_text().splitlines()]
        vals = [json.loads(x) for x in (source / (stage + "_val.jsonl")).read_text().splitlines()]
        cfg = vals[0]["config"]
        recipe = next(s for s in cfg["stages"] if s["name"] == stage)
        steps = vals[0]["steps_per_epoch"]
        updates = [r["update"] for r in rows]
        entry = dict(
            updates=len(rows),
            consecutive_updates=updates == list(range(1, len(rows) + 1)),
            epoch_ids_correct=all(r["epoch"] == (r["update"] - 1) // steps + 1 for r in rows),
            max_lr_schedule_error=max(
                abs(
                    r["lr"]
                    - lr_at(
                        r["update"],
                        r["total_updates"],
                        recipe["peak_lr"],
                        cfg["warmup_fraction"],
                        cfg["min_lr_ratio"],
                    )
                )
                for r in rows
            ),
            boundaries=[],
            epochs=[],
        )
        for epoch in range(1, recipe["epochs"] + 1):
            group = [r for r in rows if r["epoch"] == epoch]
            entry["epochs"].append(
                dict(
                    epoch=epoch,
                    train_fast_mean=float(np.mean([r["fast_ce"] for r in group])),
                    train_fast_first20=float(np.mean([r["fast_ce"] for r in group[:20]])),
                    train_fast_last20=float(np.mean([r["fast_ce"] for r in group[-20:]])),
                    validation=vals[epoch - 1]["validation"],
                )
            )
            if epoch > 1:
                at = (epoch - 1) * steps
                entry["boundaries"].append(
                    dict(
                        update=at + 1,
                        fast_pre20=float(np.mean([r["fast_ce"] for r in rows[at - 20 : at]])),
                        fast_post20=float(np.mean([r["fast_ce"] for r in rows[at : at + 20]])),
                        lr_before=rows[at - 1]["lr"],
                        lr_after=rows[at]["lr"],
                    )
                )
        if full:
            records = json.loads((full / "samples.json").read_text())["records"]
            indices = [i for i, r in enumerate(records) if r["split"] == "train"]
            lengths = np.load(full / "valid_lengths.npy")
            orders = []
            epochs = []
            for epoch in range(1, recipe["epochs"] + 1):
                order = []
                for step in zip(
                    *(
                        epoch_batches(
                            indices,
                            cfg["seed"],
                            epoch,
                            cfg["global_batch"],
                            cfg["micro_batch"],
                            8,
                            rank,
                        )
                        for rank in range(8)
                    )
                ):
                    order.extend(
                        i for rank_batch in step for micro in rank_batch for i in micro if i >= 0
                    )
                counts = Counter(order)
                epochs.append(
                    dict(
                        epoch=epoch,
                        count=len(order),
                        unique=len(counts),
                        exact_once=set(counts) == set(indices) and max(counts.values()) == 1,
                        order_sha256=hashlib.sha256(
                            np.asarray(order, dtype=np.int64).tobytes()
                        ).hexdigest(),
                        first1280_length_mean=float(np.mean(lengths[order[:1280]])),
                        last1280_length_mean=float(np.mean(lengths[order[-1280:]])),
                    )
                )
                orders.append(order)
            entry["replayed_schedule"] = epochs
            entry["adjacent_same_position_fraction"] = [
                float(np.mean(np.asarray(a) == np.asarray(b))) for a, b in zip(orders, orders[1:])
            ]
            entry["schedule_note"] = (
                "Replayed from archived source/config; no original per-step sample-ID trace exists."
            )
        result[stage] = entry
    save(output, result)
    return result


def processor():
    from transformers import PreTrainedTokenizerFast

    root = Path(__file__).resolve().parents[2] / "assets/fast"
    spec = importlib.util.spec_from_file_location(
        "local_fast_processor", root / "processing_action_tokenizer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = json.loads((root / "processor_config.json").read_text())
    return module.UniversalActionProcessor(
        PreTrainedTokenizerFast(tokenizer_file=str(root / "tokenizer.json")),
        **{k: cfg[k] for k in ["scale", "vocab_size", "min_token"]},
    )


def audit_codec(subset, full, output):
    from scipy.fft import dct
    from fluxvla.training.subtask.fixtures import Fixtures

    fx = Fixtures(subset)
    m = json.loads((subset / "samples.json").read_text())
    proc = processor()
    st = m["normalization"]["action"]
    span = np.asarray(st["q99"]) - np.asarray(st["q01"])
    joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    rows = []
    for i, r in enumerate(fx.records):
        n = r["valid_length"]
        action = np.asarray(fx.actions[i, :n, :14])
        ids = fx.fast_ids[i]
        if n == 0:
            continue
        encoded = proc(action)[0]
        decoded = proc.bpe_tokenizer.decode(ids.tolist())
        if len(decoded) != n * 14:
            raise ValueError("cached FAST coefficient count differs from valid horizon")
        restored = proc.decode([ids.tolist()], time_horizon=n, action_dim=14)[0]
        error = restored - action
        physical = error * span * 0.5
        coeff = np.around(dct(action, axis=0, norm="ortho") * proc.scale)
        rows.append(
            dict(
                source_index=r["source_index"],
                split=r["split"],
                episode=r["episode"],
                length=n,
                fast_tokens=len(ids),
                exact_reencoding=np.array_equal(encoded, ids),
                normalized_mse=float(np.square(error).mean()),
                normalized_max_abs=float(abs(error).max()),
                joint_rmse_rad=float(np.sqrt(np.square(physical[:, joints]).mean())),
                gripper_mae_mm=float(abs(physical[:, [6, 13]]).mean() * 1000),
                clipped_coefficients=int((coeff < proc.min_token).sum()),
                coefficient_count=coeff.size,
            )
        )
    lens = np.load(full / "valid_lengths.npy")
    ff = np.load(full / "fast_tokens.npz")
    result = dict(
        sample_count=len(rows),
        all_cached_tokens_reencode_exactly=all(r["exact_reencoding"] for r in rows),
        full_length_counts={
            str(int(n)): int(c) for n, c in zip(*np.unique(lens, return_counts=True))
        },
        all_fast_token_lengths=dict(
            mean=float(ff["fast_lengths"].mean()), max=int(ff["fast_lengths"].max())
        ),
        groups={},
        rows=rows,
    )
    for split in ["train", "validation"]:
        for kind in ["short", "full"]:
            group = [
                r
                for r in rows
                if r["split"] == split
                and ((r["length"] == 50) if kind == "full" else r["length"] < 50)
            ]
            result["groups"][split + "_" + kind] = {
                k: float(np.mean([r[k] for r in group]))
                for k in [
                    "normalized_mse",
                    "normalized_max_abs",
                    "joint_rmse_rad",
                    "gripper_mae_mm",
                    "fast_tokens",
                ]
            }
            result["groups"][split + "_" + kind]["samples"] = len(group)
    save(output, result)
    return result


def token_metrics(model, b):
    import torch
    import torch.nn.functional as F

    high = model.forward_high_level(
        b["images"],
        b["high_input_ids"],
        b["high_valid_mask"],
        b["high_ar_mask"],
        b["high_labels"],
        b["img_masks"],
        hl_subtask_region_mask=b["high_subtask_region_mask"],
        hl_action_region_mask=b["high_action_region_mask"],
        return_logits=True,
    )
    logits = high["logits"][:, :-1].float()
    labels = b["high_labels"][:, 1:]
    valid = labels != -100
    top = model.llm_backbone.config.vocab_size - 1 - model.fast_skip_tokens
    bottom = top - model.fast_vocab_size + 1
    fast = valid & (labels >= bottom) & (labels <= top)
    struct = valid & b["high_action_region_mask"][:, 1:] & ~fast
    sub = valid & b["high_subtask_region_mask"][:, 1:]
    results = {}
    for name, mask in [("fast", fast), ("struct", struct), ("subtask", sub)]:
        scores = logits[mask]
        targets = labels[mask]
        results[name + "_count"] = int(mask.sum())
        results[name + "_nll_sum"] = (
            float(F.cross_entropy(scores, targets, reduction="sum")) if len(targets) else 0.0
        )
        results[name + "_correct"] = (
            int((scores.argmax(-1) == targets).sum()) if len(targets) else 0
        )
    if fast.any():
        f = logits[fast]
        target = labels[fast]
        results["fast_restricted_nll_sum"] = float(
            F.cross_entropy(f[:, bottom : top + 1], target - bottom, reduction="sum")
        )
        results["fast_vocab_mass_sum"] = float(
            (torch.logsumexp(f[:, bottom : top + 1], -1) - torch.logsumexp(f, -1)).exp().sum()
        )
        first = fast & (fast.long().cumsum(-1) == 1)
        results["first_fast_count"] = int(first.sum())
        results["first_fast_nll_sum"] = float(
            F.cross_entropy(logits[first], labels[first], reduction="sum")
        )
        results["first_fast_correct"] = int((logits[first].argmax(-1) == labels[first]).sum())
    results["reported_fast_ce"] = float(high["loss_action_ce"])
    return results


def aggregate(rows):
    result = dict(samples=len(rows))
    for kind in ["fast", "struct", "subtask"]:
        count = sum(r[kind + "_count"] for r in rows)
        result[kind + "_count"] = count
        result[kind + "_ce"] = sum(r[kind + "_nll_sum"] for r in rows) / max(1, count)
        result[kind + "_accuracy"] = sum(r[kind + "_correct"] for r in rows) / max(1, count)
    count = result["fast_count"]
    result["fast_restricted_ce"] = sum(r.get("fast_restricted_nll_sum", 0) for r in rows) / max(
        1, count
    )
    result["fast_vocab_mass"] = sum(r.get("fast_vocab_mass_sum", 0) for r in rows) / max(1, count)
    first_count = sum(r.get("first_fast_count", 0) for r in rows)
    result["first_fast_ce"] = sum(r.get("first_fast_nll_sum", 0) for r in rows) / max(
        1, first_count
    )
    result["first_fast_accuracy"] = sum(r.get("first_fast_correct", 0) for r in rows) / max(
        1, first_count
    )
    return result


def score_bundle(bundle, subset, output, limit=None):
    import torch
    from fluxvla.inference import SubtaskPolicy
    from fluxvla.training.subtask.fixtures import Fixtures, prepare, collate

    policy = SubtaskPolicy.from_bundle(bundle, allow_legacy_contract=True)
    m = policy.model
    fx = Fixtures(subset)
    packed = prepare(fx, policy.prompts, 14)
    rows = []
    probes = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, r in enumerate(fx.records[:limit]):
            b = collate([packed[i]], [i], fx, 224, policy.device, False)
            row = dict(
                source_index=r["source_index"],
                split=r["split"],
                episode=r["episode"],
                length=r["valid_length"],
                **token_metrics(m, b),
            )
            rows.append(row)
            if i % 8 == 0:
                m.train()
                training_mode = token_metrics(m, b)
                m.eval()
                changed = dict(b)
                other = (i + len(fx) // 2) % len(fx)
                changed["images"] = collate(
                    [packed[other]], [other], fx, 224, policy.device, False
                )["images"]
                shuffled = token_metrics(m, changed)
                probes.append(
                    dict(
                        source_index=r["source_index"],
                        split=r["split"],
                        original=aggregate([row]),
                        shuffled_images=aggregate([shuffled]),
                        training_mode=aggregate([training_mode]),
                    )
                )
            if (i + 1) % 16 == 0:
                print("SCORED", i + 1, flush=True)
    report = dict(
        bundle=Path(bundle).name, teacher_forced=True, rows=rows, groups={}, probes=probes
    )
    for split in ["train", "validation"]:
        for kind in ["short", "full", "all"]:
            selected = [
                r
                for r in rows
                if r["split"] == split
                and (kind == "all" or (r["length"] == 50 if kind == "full" else r["length"] < 50))
            ]
            report["groups"][split + "_" + kind] = aggregate(selected)
    save(output, report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["logs", "codec", "score"])
    p.add_argument("--source", type=Path)
    p.add_argument("--full", type=Path)
    p.add_argument("--subset", type=Path)
    p.add_argument("--bundle")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    required = {"logs": ["source"], "codec": ["subset", "full"], "score": ["bundle", "subset"]}
    for field in required[a.mode]:
        if getattr(a, field) is None:
            p.error(f"{a.mode} requires --{field}")
    if a.limit is not None and a.limit < 1:
        p.error("--limit must be positive")
    if a.mode == "logs":
        audit_logs(a.source, a.output, a.full)
    elif a.mode == "codec":
        audit_codec(a.subset, a.full, a.output)
    else:
        score_bundle(a.bundle, a.subset, a.output, a.limit)
    print("AUDIT_SAVED", a.output, flush=True)


if __name__ == "__main__":
    main()
