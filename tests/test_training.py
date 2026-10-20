import itertools
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluxvla import register_training_components

register_training_components()


def test_configuration_registration_and_cache_hook(tmp_path):
    from fluxvla.engines import DATASETS, RUNNERS, TRANSFORMS, VLAS
    from fluxvla.transforms.transform_inputs import ProcessParquetInputs
    from test_video_cache import make_cache

    config = runpy.run_path(str(Path(__file__).parents[1] / "configs/pi05_piper.py"))
    assert VLAS.get(config["model"]["type"]) is not None
    assert RUNNERS.get(config["runner"]["type"]) is not None
    dataset = config["train_dataloader"]["dataset"]
    assert DATASETS.get(dataset["type"]) is not None
    for transform in dataset["datasets"][0]["transforms"]:
        assert TRANSFORMS.get(transform["type"]) is not None
    source, directory, frames = make_cache(tmp_path)
    transform = ProcessParquetInputs(
        [], [], video_cache={"directory": str(directory), "data_root": str(source.parents[2])}
    )
    out = transform.decode_video_frames_torchvision(source, [0.04], 0.01)
    np.testing.assert_array_equal(out.numpy()[0], frames[1].transpose(2, 0, 1))
    transform._video_cache.close()


def test_rank_worker_shards_are_disjoint_and_cover_epoch(monkeypatch):
    from fluxvla.datasets.dataset_wrapper import DistributedRepeatingDataset

    shards = []
    for rank in range(2):
        for worker in range(3):
            dataset = DistributedRepeatingDataset.__new__(DistributedRepeatingDataset)
            dataset.rank, dataset.world_size = rank, 2
            dataset.total_len, dataset.seed = 101, 42
            dataset._epoch, dataset.shuffle, dataset.reshuffle_each_epoch = 0, True, True
            dataset._get_item_from_global_idx = lambda i: i
            monkeypatch.setattr(
                torch.utils.data,
                "get_worker_info",
                lambda worker=worker: SimpleNamespace(id=worker, num_workers=3),
            )
            expected = len(range(rank * 3 + worker, 101, 6))
            shards.append(list(itertools.islice(iter(dataset), expected)))
    flat = list(itertools.chain.from_iterable(shards))
    assert len(flat) == len(set(flat)) == 101
    assert set(flat) == set(range(101))


def test_ddp_checkpoint_restores_optimizer_scheduler_and_next_update(tmp_path):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from fluxvla.engines.runners.ddp_train_runner import DDPTrainRunner

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    try:

        def runner():
            r = DDPTrainRunner.__new__(DDPTrainRunner)
            r.vla = DDP(
                torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
            )
            r.optimizer = torch.optim.AdamW(r.vla.parameters(), lr=0.001)
            r.lr_scheduler = torch.optim.lr_scheduler.StepLR(r.optimizer, step_size=2, gamma=0.5)
            r.tokenizer, r.max_keep_ckpts = None, 2
            r.metric, r.current_epoch = SimpleNamespace(global_step=0), 0
            return r

        inputs, targets = torch.randn(5, 3), torch.randn(5, 2)

        def step(r):
            r.optimizer.zero_grad()
            torch.nn.functional.mse_loss(r.vla(inputs), targets).backward()
            r.optimizer.step()
            r.lr_scheduler.step()

        original = runner()
        step(original)
        step(original)
        original.save_checkpoint(tmp_path, global_step=4052, epoch=2, train_loss=0.015)
        resumed = runner()
        resumed.resume_from = str(tmp_path / "checkpoints/latest-checkpoint.pt")
        resumed.resume()
        assert resumed.metric.global_step == 4052 and resumed.current_epoch == 2
        assert resumed.lr_scheduler.state_dict() == original.lr_scheduler.state_dict()
        assert len(resumed.optimizer.state) == len(original.optimizer.state)
        step(original)
        step(resumed)
        for a, b in zip(original.vla.parameters(), resumed.vla.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="Missing key"):
            resumed._load_model_state({})
    finally:
        dist.destroy_process_group()


def test_tiny_pi05_forward_and_backward():
    from fluxvla.engines import build_vla_from_cfg

    config = runpy.run_path(str(Path(__file__).parents[1] / "configs/pi05_piper.py"))["model"]
    config["pretrained_name_or_path"] = None
    for key in ("llm_backbone", "llm_expert"):
        config[key].update(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=64,
            max_position_embeddings=128,
        )
    config["llm_expert"]["adarms_cond_dim"] = 16
    config["vision_backbone"]["vision_config"].update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=8,
    )
    config["projector"].update(in_dim=16, out_dim=16)
    config["action_in_proj"].update(out_dim=16)
    config["action_out_proj"].update(in_dim=16)
    config["time_mlp_in"].update(in_dim=16, out_dim=16)
    config["time_mlp_out"].update(in_dim=16, out_dim=16)
    config.update(proj_width=16, n_action_steps=4)
    config["rtc_training_config"]["max_delay"] = 2
    model = build_vla_from_cfg(config)
    output = model(
        images=torch.randn(2, 9, 16, 16),
        lang_tokens=torch.randint(0, 64, (2, 5)),
        states=torch.randn(2, 32),
        actions=torch.randn(2, 4, 32),
        img_masks=torch.ones(2, 3, dtype=torch.bool),
        lang_masks=torch.ones(2, 5, dtype=torch.bool),
    )
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_resume_entry_skips_initial_weights_and_checks_path(tmp_path, monkeypatch):
    import importlib.util
    import sys

    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("soma_train_entry", root / "scripts/train.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    checkpoint = tmp_path / "resume.pt"
    checkpoint.write_bytes(b"Preflight only: runner checkpoint contents are tested separately")
    seen = []
    monkeypatch.setattr(entry, "train", lambda args, cfg: seen.append(cfg))
    argv = [
        "train.py",
        "--config",
        str(root / "configs/pi05_piper.py"),
        "--work-dir",
        str(tmp_path / "run"),
        "--resume-from",
        str(checkpoint),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    entry.main()
    assert seen[0].model.pretrained_name_or_path is None
    checkpoint.unlink()
    with pytest.raises(SystemExit) as exc:
        entry.main()
    assert exc.value.code == 2
