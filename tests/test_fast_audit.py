"""Ensure FAST diagnostics separate action tokens from structural text."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import math
import torch


def test_fast_diagnostic_uniform_distribution_has_known_entropy_and_mass():
    spec = importlib.util.spec_from_file_location(
        "fast_audit", Path(__file__).parents[1] / "scripts/diagnostics/audit_fast.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # V=16, skip=2 -> FAST ids 10..13; two FAST targets, two delimiters, one subtask.
    ids = torch.tensor([[1, 2, 10, 11, 3, 4]])

    class Model:
        fast_skip_tokens = 2
        fast_vocab_size = 4
        llm_backbone = SimpleNamespace(config=SimpleNamespace(vocab_size=16))

        def forward_high_level(self, *args, **kwargs):
            return dict(logits=torch.zeros(1, 6, 16), loss_action_ce=torch.tensor(math.log(16)))

    b = dict(
        images=None,
        img_masks=None,
        high_input_ids=ids,
        high_valid_mask=torch.ones_like(ids, dtype=torch.bool),
        high_ar_mask=torch.ones_like(ids, dtype=torch.bool),
        high_labels=torch.tensor([[-100, 2, 10, 11, 3, 4]]),
        high_subtask_region_mask=torch.tensor([[0, 1, 0, 0, 0, 0]], dtype=torch.bool),
        high_action_region_mask=torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.bool),
    )
    result = module.aggregate([module.token_metrics(Model(), b)])
    assert (
        result["fast_count"] == 2 and result["struct_count"] == 2 and result["subtask_count"] == 1
    )
    assert abs(result["fast_ce"] - math.log(16)) < 1e-6
    assert abs(result["fast_restricted_ce"] - math.log(4)) < 1e-6
    assert abs(result["fast_vocab_mass"] - 0.25) < 1e-6
