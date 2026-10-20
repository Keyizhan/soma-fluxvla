# Derived from FluxVLA (Limx Dynamics); see LICENSE and NOTICE.
import argparse
import json
import os
import random
import draccus
import numpy as np
import torch
import yaml
from mmengine import Config, DictAction
from fluxvla.datasets.utils import save_dataset_statistics, save_grouped_dataset_statistics
from fluxvla.engines import build_dataset_from_cfg, build_runner_from_cfg, initialize_overwatch
from fluxvla.engines.utils.torch_utils import set_global_seed

overwatch = initialize_overwatch(__name__)


def _save_resolved_config(cfg, work_dir):
    config_yaml_path = os.path.join(work_dir, "config.yaml")
    config_json_path = os.path.join(work_dir, "config.json")
    with open(config_yaml_path, "w") as f_yaml:
        draccus.dump(cfg.to_dict(), f_yaml)
    with open(config_yaml_path, "r") as f_yaml, open(config_json_path, "w") as f_json:
        yaml_cfg = yaml.safe_load(f_yaml)
        json.dump(yaml_cfg, f_json, indent=2)


def _get_nested_value(obj, path):
    for key in path:
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            obj = getattr(obj, key, None)
    return obj


def _resolve_train_seed(cfg):
    """Resolve an explicit training seed from existing config fields."""
    for path in (("runner", "seed"), ("train_dataloader", "seed"), ("seed",)):
        seed = _get_nested_value(cfg, path)
        if seed is not None:
            return int(seed)
    return None


def _set_rank_training_seed(seed):
    rank_seed = int(seed) + overwatch.rank()
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    return rank_seed


def train(args, cfg):
    """Train the model with the given configuration.

    Args:
        cfg (Config): The configuration object containing training settings.
    """
    seed = _resolve_train_seed(cfg)
    if seed is not None:
        set_global_seed(seed)
        if overwatch.is_rank_zero():
            overwatch.info(f"Training seed set to {seed}.")
    os.makedirs(args.work_dir, exist_ok=True)
    dataset = build_dataset_from_cfg(cfg.train_dataloader.dataset)
    if overwatch.is_rank_zero() and hasattr(dataset, "dataset_statistics"):
        save_dataset_statistics(dataset.dataset_statistics, args.work_dir)
    elif overwatch.is_rank_zero() and hasattr(dataset, "grouped_dataset_statistics"):
        save_grouped_dataset_statistics(dataset.grouped_dataset_statistics, args.work_dir)
    if overwatch.is_rank_zero():
        _save_resolved_config(cfg, args.work_dir)
    if (
        overwatch.is_rank_zero()
        and hasattr(dataset, "batch_transform")
        and hasattr(dataset.batch_transform, "base_tokenizer")
    ):
        tokenizer = dataset.batch_transform.base_tokenizer
        tokenizer.save_pretrained(args.work_dir)
        overwatch.info(f"Saved tokenizer to {args.work_dir}")
    if hasattr(cfg.runner, "metric"):
        cfg.runner.metric.run_dir = args.work_dir
    cfg.runner.cfg = cfg
    cfg.runner.args = args
    if args.resume_from is not None:
        cfg.runner.resume_from = args.resume_from
    runner = build_runner_from_cfg(cfg.runner)
    runner.run_setup(n_train_examples=len(dataset))
    if seed is not None:
        rank_seed = _set_rank_training_seed(seed)
        if overwatch.is_rank_zero():
            overwatch.info(
                f"Training RNG reset after model setup; rank-local base seed is {rank_seed}."
            )
    runner.run(dataset)


def main():
    parser = argparse.ArgumentParser(description="PI0.5 Piper training and checkpoint resume")
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()
    from fluxvla import register_training_components

    register_training_components()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.resume_from is not None:
        if not os.path.isfile(args.resume_from):
            parser.error(f"Resume checkpoint does not exist: {args.resume_from}")
        if not args.resume_from.endswith(".pt"):
            parser.error(
                "Full training resume requires a .pt checkpoint, including optimizer state"
            )
        # A full resume checkpoint already contains every model tensor. Avoid
        # reading the initial 14 GB pretrained weights again before overwriting them.
        cfg.model.pretrained_name_or_path = None
    train(args, cfg)


if __name__ == "__main__":
    main()
