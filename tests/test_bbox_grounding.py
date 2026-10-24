"""Grounding supervision, causal isolation, reduction and inference regressions."""

import copy
import json

import numpy as np
import pytest
import torch

from fluxvla.transforms.bbox_grounding import (
    localization_metrics,
    parse_grounded,
    render_goals,
    resolve_bbox,
    semantic_key,
)
from fluxvla.transforms.subtask_bbox_prompts import PrepareSubtaskBboxPrompts
from fluxvla.models.vlas.bbox_loss import bbox_statistics
from fluxvla.inference.subtask_stability import SubtaskStability
from test_subtask_fast import CharTokenizer
from test_subtask_fast_model import tiny_config, image_batch
from fluxvla.engines import build_vla_from_cfg


def goals():
    return [
        dict(
            target_type="bbox",
            camera="images.cam_front",
            bbox_xyxy_q1000=[0, 483, 305, 698],
            subtask="Place fabric B in the middle-left basket using the left arm.",
        ),
        dict(target_type="none", bbox_xyxy_q1000=None, subtask="Move fabric to staging."),
    ]


def prompts():
    return PrepareSubtaskBboxPrompts(
        CharTokenizer(),
        max_len=512,
        state_dim=2,
        model_vocab_size=512,
        fast_vocab_size=16,
        fast_skip_tokens=0,
        all_causal=False,
        low_level_sees_task=False,
        low_level_prompt_format="state_subtask",
        low_level_segmented=True,
        low_level_action_tail="",
    )


def pack(gs=None):
    gs = goals() if gs is None else gs
    return prompts().high_level(
        "sort", np.zeros(2), " || ".join(g["subtask"] for g in gs), [1, 2], grounding=gs
    )


def test_parallel_masks_partition_and_none_has_no_coordinate():
    p = pack()
    masks = [p["subtask_region_mask"], p["action_region_mask"]]
    masks += [p["bbox_" + name + "_groups"] >= 0 for name in ("type", "coordinate", "format")]
    assert np.array_equal(np.sum(masks, axis=0), p["loss_mask"].astype(int))
    ids = p["bbox_coordinate_groups"]
    assert set(ids[ids >= 0]) == {0, 1, 2, 3}
    assert set(p["bbox_type_groups"]) == {-1, 0, 1}
    assert np.array_equal(p["ar_mask"], p["loss_mask"])
    assert not np.any(p["subtask_key_mask"] & (ids >= 0))
    assert pack([goals()[1]])["bbox_coordinate_groups"].max() == -1


def test_roundtrip_low_level_and_strict_coordinates():
    gs = goals()
    p = prompts()
    text = render_goals(gs)
    assert [g["bbox_xyxy_q1000"] for g in parse_grounded(text)] == [[0, 483, 305, 698], None]
    a = p.low_level("", np.zeros(2), grounding=gs)
    b = p.low_level(text, np.zeros(2))
    assert np.array_equal(a["input_ids"], b["input_ids"])
    for bad in [
        "Target: missing; Subtask: x",
        text.replace("305", "1001"),
        text.replace("305", "0"),
        text + " || ",
    ]:
        with pytest.raises(ValueError):
            parse_grounded(bad)
    with pytest.raises(ValueError):
        p.high_level("sort", np.zeros(2), "incorrect text", grounding=gs)


def test_coordinate_equal_weight_and_accumulation():
    # Unequal token lengths: coordinate 0 has two tokens, others one.
    groups = {n: torch.full((1, 6), -1, dtype=torch.long) for n in ("type", "coordinate", "format")}
    groups["coordinate"][0, 1:] = torch.tensor([0, 0, 1, 2, 3])
    mask = torch.ones(1, 5, dtype=torch.bool)
    values = torch.tensor([2.0, 4.0, 8.0, 10.0, 12.0], requires_grad=True)
    out = bbox_statistics(values, mask, groups)
    assert out["bbox_coordinate_ce"].item() == pytest.approx((3 + 8 + 10 + 12) / 4)
    out["bbox_coordinate_ce"].backward()
    assert torch.allclose(values.grad, torch.tensor([0.125, 0.125, 0.25, 0.25, 0.25]))
    counts = torch.tensor([0.0, 2.0, 0.0])
    a = bbox_statistics(values, mask, groups, denominators=counts)["bbox_coordinate_ce"]
    b = bbox_statistics(values * 2, mask, groups, denominators=counts)["bbox_coordinate_ce"]
    assert (a + b).item() == pytest.approx(out["bbox_coordinate_ce"].item() * 1.5)


def test_zero_supervision_is_finite_and_connected():
    values = torch.empty(0, requires_grad=True)
    groups = {n: torch.full((1, 6), -1, dtype=torch.long) for n in ("type", "coordinate", "format")}
    out = bbox_statistics(values, torch.zeros(1, 5, dtype=torch.bool), groups)
    loss = sum(out["bbox_" + n + "_ce"] for n in ("type", "coordinate", "format"))
    assert loss.item() == 0
    loss.backward()


def test_metrics_penalize_invalid_and_missing_boxes():
    gs = goals()
    assert localization_metrics(render_goals(gs), gs)["iou_sum"] == 1
    result = localization_metrics("broken", gs)
    assert result["iou_sum"] == 0 and result["missed_bbox"] == 1
    assert result["bbox_count"] == 1 and result["valid_bbox_count"] == 0


def test_stability_updates_coordinates_without_switching_primitive():
    stability = SubtaskStability(2, key_fn=semantic_key, refresh_same=True)
    a = render_goals(goals())
    b = a.replace("305", "306")
    stability.commit(stability.propose("sort", a, 1.0))
    update = stability.propose("sort", b, 2.0)
    assert not update.switched and update.active == b and update.reason == "same_task"


def test_config_opt_in_and_weights():
    from fluxvla.training.subtask.config import load_config, canonical_prompts
    from pathlib import Path

    root = Path(__file__).parents[1]
    cfg = load_config(root / "configs/subtask/bbox_hybrid.json")
    assert cfg["bbox"]["weight"] == 5
    assert isinstance(canonical_prompts(CharTokenizer(), cfg), PrepareSubtaskBboxPrompts)
    assert "bbox" not in load_config(root / "configs/subtask/fabric106.json")
    for bad in [float("nan"), -1, True]:
        with pytest.raises(ValueError):
            resolve_bbox({"weight": bad})


def test_scan_and_cache_preserve_grounding_and_reject_legacy_cache(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path
    from test_subtask_pipeline import synthetic, CODE
    from fluxvla.training.subtask.prepare import prepare as cache, verify, validate_cache_config
    from fluxvla.training.subtask.fixtures import Fixtures, prepare
    from fluxvla.training.subtask.config import canonical_prompts

    cfg = synthetic(tmp_path)
    cfg.update(bbox=resolve_bbox({}), max_prompt_length=512, cache_workers=1)
    gs = goals()
    for path in Path(cfg["incoming"]).rglob("*.parquet"):
        t = pq.read_table(path)
        t = t.set_column(
            t.schema.get_field_index("subtask"),
            "subtask",
            pa.array([" || ".join(g["subtask"] for g in gs)] * len(t)),
        )
        for key, value in [
            ("grounding_json", json.dumps(gs)),
            ("subtask_grounded", render_goals(gs)),
            ("grounding_valid", 1),
        ]:
            t = t.append_column(key, pa.array([value] * len(t)))
        pq.write_table(t, path)
    root = cache(cfg, CODE)
    verify(cfg)
    fx = Fixtures(root)
    assert len(fx.records) == 24 and all("grounding_json" in r for r in fx.records)
    packed = prepare(fx, canonical_prompts(CharTokenizer(), cfg), 14)
    assert packed[0]["high"]["bbox_coordinate_groups"].max() == 3
    old = dict(cfg)
    old.pop("bbox")
    with pytest.raises(ValueError, match="bbox"):
        validate_cache_config(json.loads((root / "READY.json").read_text()), old)


def test_bundle_records_bbox_and_rejects_weight_change(tmp_path):
    from fluxvla.training.subtask.config import resolve_config
    from fluxvla.training.subtask.checkpoints import bundle_assets

    cfg = resolve_config(dict(bbox=resolve_bbox({})))
    cfg.update(cache=str(tmp_path), max_prompt_length=512)
    (tmp_path / "samples.json").write_text(json.dumps(dict(normalization={})))
    native = dict(
        type="PI05Subtask",
        n_action_steps=50,
        vision_backbone=dict(vision_config=dict(image_size=224)),
        low_level_text_attention="bidirectional",
        low_level_task_independent=True,
        text_loss_mode="unified_token",
        token_loss_weight=1.0,
        hl_loss_weight=0.0,
        fast_loss_weight=0.0,
        action_loss_weight=1.0,
        bbox=cfg["bbox"],
    )
    metadata = dict(
        preset="hybrid",
        cache_identity="fixture",
        stage="stage2",
        update=1,
        config=copy.deepcopy(cfg),
    )
    _, _, contract = bundle_assets(cfg, native, metadata)
    assert (
        contract["bbox"]["weight"] == 5
        and contract["prompt_transform"] == "PrepareSubtaskBboxPrompts"
    )
    metadata["config"]["bbox"]["weight"] = 1.0
    with pytest.raises(ValueError, match="bbox"):
        bundle_assets(cfg, native, metadata)


def test_bbox_bundle_serialization_and_policy_reload(tmp_path, monkeypatch):
    # Windows-only shims test serialization, not POSIX crash durability.
    import os
    import shutil
    from types import SimpleNamespace
    from fluxvla.training.subtask import checkpoints as cp
    from fluxvla.training.subtask.config import resolve_config
    from fluxvla.inference.subtask import SubtaskPolicy, CAMERAS
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    if os.name == "nt":
        monkeypatch.setattr(
            os,
            "statvfs",
            lambda p: SimpleNamespace(f_bavail=shutil.disk_usage(p).free, f_frsize=1),
            raising=False,
        )
        monkeypatch.setattr(os, "fsync", lambda fd: None)
        monkeypatch.setattr(cp, "sync_dir", lambda p: None)
    tokenizer = Tokenizer(
        models.WordLevel({"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}, unk_token="<unk>")
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
    ).save_pretrained(tmp_path / "tokenizer")
    stats = dict(state=dict(q01=[-1] * 14, q99=[1] * 14), action=dict(q01=[0] * 14, q99=[1] * 14))
    (tmp_path / "samples.json").write_text(json.dumps(dict(normalization=stats)))
    cfg = resolve_config(dict(bbox=resolve_bbox({})))
    cfg.update(
        cache=str(tmp_path),
        tokenizer=str(tmp_path / "tokenizer"),
        max_prompt_length=512,
        camera_aliases=[[k] for k in CAMERAS],
    )
    native = tiny_config(
        bbox=cfg["bbox"],
        text_loss_mode="unified_token",
        hl_loss_weight=0.0,
        fast_loss_weight=0.0,
        token_loss_weight=1.0,
        action_loss_weight=1.0,
        low_level_task_independent=True,
        low_level_text_attention="bidirectional",
        low_level_prefix_source="text",
        fast_vocab_size=16,
        fast_skip_tokens=0,
    )
    for key in ("llm_backbone", "llm_expert"):
        native[key]["vocab_size"] = 512
    model = build_vla_from_cfg(native).eval()
    metadata = dict(
        preset="hybrid",
        cache_identity="fixture",
        stage="stage2",
        update=1,
        config=cfg,
        model_config=native,
    )
    bundle = cp.export_bundle(model.state_dict(), tmp_path / "bundle", cfg, native, metadata)
    policy = SubtaskPolicy.from_bundle(bundle, device="cpu")
    assert isinstance(policy.prompts, PrepareSubtaskBboxPrompts)
    assert policy.model.bbox["weight"] == 5 and policy.config.max_new_tokens == 256
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, policy.model.state_dict()[name])


def model_input(p):
    result = {
        f"hl_{k}": torch.as_tensor(v)[None]
        for k, v in p.items()
        if k
        in (
            "input_ids",
            "valid_mask",
            "ar_mask",
            "labels",
            "subtask_region_mask",
            "action_region_mask",
            "subtask_key_mask",
            "subtask_template_mask",
            "action_code_mask",
        )
    }
    result["hl_bbox_groups"] = {
        n: torch.as_tensor(p["bbox_" + n + "_groups"])[None]
        for n in ("type", "coordinate", "format")
    }
    return result


def test_model_bbox_causal_isolation_and_loss_partition():
    torch.manual_seed(7)
    cfg = tiny_config(
        bbox=resolve_bbox({}),
        text_loss_mode="unified_token",
        hl_loss_weight=0.0,
        fast_loss_weight=0.0,
        token_loss_weight=1.0,
        attn_math="fp32",
    )
    for key in ("llm_backbone", "llm_expert"):
        cfg[key]["vocab_size"] = 512
    model = build_vla_from_cfg(cfg).eval()
    p = pack([goals()[0]])
    a = model_input(p)
    images, masks = image_batch(batch=1)
    out = model.forward_high_level(images, img_masks=masks, **a)
    assert out["bbox_coordinate_count"].item() == 1
    expected = 5 * (out["bbox_type_ce"] + out["bbox_coordinate_ce"] + 0.1 * out["bbox_format_ce"])
    assert torch.allclose(out["loss_bbox"], expected)
    b = copy.deepcopy(a)
    # Alter subtask input tokens only; all bbox logits precede those tokens.
    b["hl_input_ids"][b["hl_subtask_region_mask"]] = 42
    out2 = model.forward_high_level(images, img_masks=masks, **b)
    assert torch.equal(out["bbox_coordinate_sum"], out2["bbox_coordinate_sum"])
    assert out["token_count"].item() == int(
        (a["hl_subtask_region_mask"] | a["hl_action_region_mask"]).sum()
    )
    out["loss_bbox"].backward()
    assert model.llm_backbone.embed_tokens.weight.grad.abs().sum() > 0
    bad = copy.deepcopy(a)
    bad["hl_bbox_groups"]["type"][0, 0] = 0
    with pytest.raises(ValueError):
        model.forward_high_level(images, img_masks=masks, **bad)

    from fluxvla.training.subtask.train import Loss, training_metrics

    batch = {f"high_{k}": torch.as_tensor(v)[None] for k, v in p.items()}
    batch.update(images=images, img_masks=masks, real=torch.ones(1, dtype=torch.bool))
    loss, _, statistics = Loss(model, False)(batch, torch.ones(3), return_statistics=True)
    assert torch.allclose(loss, out["loss_token_objective"] + out["loss_bbox"])
    assert training_metrics(statistics)["bbox_coordinate_count"] == 1


def test_inference_passes_predicted_coordinates_and_rejects_invalid_before_actions():
    from test_inference_policy import Model
    from fluxvla.inference.subtask import SubtaskPolicy, InferenceConfig

    class GroundedModel(Model):
        bbox = resolve_bbox({})
        low_level_task_independent = True
        low_level_text_attention = "bidirectional"
        low_level_prefix_source = "text"
        output = render_goals(goals())

        def generate_subtask(self, *args, **kwargs):
            ids = CharTokenizer().encode(self.output + "\nAction: ")
            return dict(
                token_ids=torch.tensor([ids]), lengths=torch.tensor([len(ids)]), finished=[True]
            )

        def sample_actions_given_subtask(self, *args, **kwargs):
            self.action_ids = args[2]
            return super().sample_actions_given_subtask(*args, **kwargs)

    p = prompts()
    p.state_dim = 14
    model = GroundedModel()
    stats = dict(state=dict(q01=[-1] * 14, q99=[1] * 14), action=dict(q01=[0] * 14, q99=[1] * 14))
    policy = SubtaskPolicy(
        model, p, stats, config=InferenceConfig(max_new_tokens=512), image_size=16
    )
    payload = dict(
        state=np.zeros(14),
        instruction="sort",
        images={k: np.zeros((16, 16, 3), np.uint8) for k in policy.camera_keys},
    )
    result = policy.infer(payload)
    assert result["trace"]["grounding"][0]["bbox_xyxy_q1000"] == [0, 483, 305, 698]
    assert np.array_equal(
        model.action_ids[0].numpy(), p.low_level(model.output, np.zeros(14))["input_ids"]
    )
    model.output = "Target: cam_front xyxy [0, 4, 1001, 8]; Subtask: Place."
    with pytest.raises(ValueError):
        policy.infer(payload)


def _ddp_worker(rank, rendezvous, output):
    import torch.distributed as dist
    from datetime import timedelta
    from pathlib import Path

    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            model.module.weight.fill_(1.0)
        value = model(torch.ones(1, 1)).sum()
        groups = {
            n: torch.full((1, 6), -1, dtype=torch.long) for n in ("type", "coordinate", "format")
        }
        mask = torch.zeros(1, 5, dtype=torch.bool)
        if rank == 0:
            groups["coordinate"][0, 1:] = torch.tensor([0, 0, 1, 2, 3])
            mask[:] = True
        nll = value * torch.tensor([2.0, 4.0, 8.0, 10.0, 12.0] if rank == 0 else [])
        loss = bbox_statistics(nll, mask, groups)["bbox_coordinate_ce"]
        loss.backward()
        Path(output, f"{rank}.json").write_text(
            json.dumps(dict(loss=loss.item(), gradient=model.module.weight.grad.item()))
        )
    finally:
        dist.destroy_process_group()


def test_two_rank_ddp_zero_bbox_rank_matches_global_mean(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _ddp_worker, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)), nprocs=2, join=True
    )
    for rank in range(2):
        result = json.loads((tmp_path / f"{rank}.json").read_text())
        assert result["loss"] == pytest.approx(8.25)
        assert result["gradient"] == pytest.approx(8.25)
