"""Small, matched offline objective ablations. Never exports deployment weights.

Every arm starts from the same full bundle, sample order and RNG seeds. Results
measure small-subset adaptation, not a reproduction of the original Stage1 run.
"""

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from fluxvla.inference import SubtaskPolicy
from fluxvla.training.subtask.fixtures import Fixtures, prepare, collate
from audit_fast import token_metrics, aggregate, save


def seed(value):
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def select(fx, split, per_group):
    rng = np.random.default_rng(1702)
    result = []
    for full in [False, True]:
        ids = [
            i
            for i, r in enumerate(fx.records)
            if r["split"] == split and (r["valid_length"] == 50) == full
        ]
        result.extend(rng.choice(ids, min(per_group, len(ids)), replace=False).tolist())
    return result


@torch.no_grad()
def evaluate(m, fx, packed, indices):
    previous = m.training
    m.eval()
    rows = []
    flow_num = flow_den = 0.0
    for i in indices:
        b = collate([packed[i]], [i], fx, 224, torch.device("cuda"), True, m.n_action_steps)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rows.append(token_metrics(m, b))
            seed(99100 + i)
            flow = m.forward_low_level(
                b["images"],
                torch.zeros(1, 32, device="cuda"),
                b["low_input_ids"],
                b["low_valid_mask"],
                b["actions"],
                b["img_masks"],
                action_masks=b["action_masks"],
            )
        n = int(b["action_masks"].sum())
        flow_num += float(flow["loss"]) * n
        flow_den += n
    m.train(previous)
    return dict(aggregate(rows), flow_mse=flow_num / max(1, flow_den))


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    fx = Fixtures(args.subset)
    train = select(fx, "train", 8)
    val = select(fx, "validation", 8)
    if len(train) != 16 or len(val) != 16:
        raise ValueError("Need at least 8 short and 8 full-horizon examples per split")
    train_episodes = {(fx.records[i]["source"], fx.records[i]["episode"]) for i in train}
    val_episodes = {(fx.records[i]["source"], fx.records[i]["episode"]) for i in val}
    if train_episodes & val_episodes:
        raise ValueError("Train and validation episodes overlap")
    weights = {
        "fast_only": (0.0, 1.0, 0.0),
        "stage1_objective": (10.0, 1.0, 0.0),
        "hybrid": (0.15, 0.15, 1.0),
        "hybrid_insulated": (0.15, 0.15, 1.0),
    }
    seed(1702)
    p = SubtaskPolicy.from_bundle(args.bundle, allow_legacy_contract=True)
    m = p.model
    packed = prepare(fx, p.prompts, 14)
    m.stop_gradient_flow_to_prefix = args.arm == "hybrid_insulated"
    recipe = weights[args.arm]
    for name, param in m.named_parameters():
        param.requires_grad_(
            not (
                name == "llm_expert.embed_tokens.weight"
                or (
                    recipe[2] == 0
                    and name.startswith(
                        (
                            "llm_expert",
                            "action_in_proj",
                            "action_out_proj",
                            "time_mlp_in",
                            "time_mlp_out",
                        )
                    )
                )
            )
        )
    parameters = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    metadata = dict(
        arm=args.arm,
        start_bundle=Path(args.bundle).name,
        updates=args.updates,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=1702,
        loss_weights=recipe,
        stop_gradient_flow_to_prefix=m.stop_gradient_flow_to_prefix,
        train_source_indices=[fx.records[i]["source_index"] for i in train],
        validation_source_indices=[fx.records[i]["source_index"] for i in val],
        note="Matched-start small-subset adaptation; stage1_objective is NOT original Stage1 continuation; BF16 parameters/Adam states match source trainer.",
    )
    save(output / "config.json", metadata)
    checkpoints = []
    for update in range(args.updates + 1):
        if update in {0, args.updates // 2, args.updates}:
            point = dict(
                update=update,
                train=evaluate(m, fx, packed, train),
                validation=evaluate(m, fx, packed, val),
            )
            checkpoints.append(point)
            save(output / "evaluation.json", checkpoints)
            print("EVAL", json.dumps(point), flush=True)
        if update == args.updates:
            break
        start = time.perf_counter()
        m.train()
        opt.zero_grad(set_to_none=True)
        # Deterministic permutation of the same 16 examples for every arm.
        indices = []
        for j in range(args.batch_size):
            position = update * args.batch_size + j
            order = np.random.default_rng(1702 + position // len(train)).permutation(train)
            indices.append(int(order[position % len(train)]))
        # Weight CE and flow by valid token/element counts over the accumulated batch.
        counts = np.array(
            [
                [
                    np.count_nonzero(packed[i]["high"][kind][1:])
                    for kind in ["subtask_region_mask", "action_region_mask"]
                ]
                + [int(fx.action_masks[i].sum())]
                for i in indices
            ]
        )
        totals = counts.sum(0).clip(min=1)
        losses = np.zeros(3)
        for j, i in enumerate(indices):
            seed(1702 + update * args.batch_size + j)
            b = collate(
                [packed[i]], [i], fx, 224, torch.device("cuda"), bool(recipe[2]), m.n_action_steps
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = m.forward_high_level(
                    b["images"],
                    b["high_input_ids"],
                    b["high_valid_mask"],
                    b["high_ar_mask"],
                    b["high_labels"],
                    b["img_masks"],
                    hl_subtask_region_mask=b["high_subtask_region_mask"],
                    hl_action_region_mask=b["high_action_region_mask"],
                )
                values = [h["loss_subtask"], h["loss_action_ce"]]
                if recipe[2]:
                    f = m.forward_low_level(
                        b["images"],
                        torch.zeros(1, 32, device="cuda"),
                        b["low_input_ids"],
                        b["low_valid_mask"],
                        b["actions"],
                        b["img_masks"],
                        action_masks=b["action_masks"],
                    )
                    values.append(f["loss"])
                else:
                    values.append(values[0] * 0)
                factors = counts[j] / totals
                loss = sum(
                    w * v * float(factors[k]) for k, (w, v) in enumerate(zip(recipe, values))
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite objective")
            loss.backward()
            losses += np.array([float(v.detach()) for v in values]) * factors
            del loss, values, h, b
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(grad):
            raise FloatingPointError("nonfinite gradient")
        opt.step()
        row = dict(
            update=update + 1,
            subtask_ce=float(losses[0]),
            fast_ce=float(losses[1]),
            flow_mse=float(losses[2]),
            grad_norm=float(grad),
            seconds=time.perf_counter() - start,
            peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3,
        )
        with (output / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if update < 2 or (update + 1) % 8 == 0:
            print("UPDATE", json.dumps(row), flush=True)
    save(
        output / "COMPLETE.json",
        dict(status="complete", hardware_commands=False, weights_saved=False, points=checkpoints),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--subset", required=True)
    p.add_argument(
        "--arm",
        required=True,
        choices=["fast_only", "stage1_objective", "hybrid", "hybrid_insulated"],
    )
    p.add_argument("--output", required=True)
    p.add_argument("--updates", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    args = p.parse_args()
    if args.updates < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        p.error("updates, batch-size and lr must be positive; lr must be finite")
    run(args)


if __name__ == "__main__":
    main()
