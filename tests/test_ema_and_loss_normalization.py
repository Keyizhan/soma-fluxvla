"""EMA on the stage-1 entry point, and the two loss normalisations.

Only plain CPU tensors are used: `EMA` / `ema_weights` are imported straight from
`scripts/architecture/run_fast_stage1.py` and `global_weighted_mean` from
`fluxvla.models.vlas.pi05_subtask`; no real model is built (the module-level imports of
the script are the only cost).

Loss-normalisation conclusion (checked below, not assumed):

    ours   = global_weighted_mean((nll * w).sum(), w.sum()) = sum(nll*w) / sum(w)
    theirs = mean_s( sum_i(nll_si*w_si) / sum_i(w_si) )

With one count per sample `c_s = sum_i(w_si)` and per-sample mean `m_s`,
`ours = sum_s c_s*m_s / sum_s c_s` (a *count-weighted* average of per-sample means) while
`theirs = mean_s m_s` (an *unweighted* average). The two agree iff every `c_s` is equal
(or, degenerate, iff all `m_s` are equal), so:
  * equal token counts per sample  -> identical up to fp error (<= 1e-6 here);
  * unequal token counts           -> different in general, by
                                      `sum_s (c_s/sum_c - 1/B) * m_s`.
Worked example used in the unequal test: counts [2, 1], per-sample nll [1, 3] and [5]
give m = [2, 5], so ours = (2*2 + 1*5)/3 = 3.0, theirs = (2 + 5)/2 = 3.5, gap = -0.5.
This is why a per-sample-weighted batch mean is *not* a drop-in replacement for the
global reduction used in the model: they only coincide when the FAST/subtask token
counts per sample are equal.
"""
import importlib.util
from pathlib import Path

import torch

from fluxvla.models.vlas.pi05_subtask import global_weighted_mean

SCRIPT = Path(__file__).parents[1] / 'scripts/architecture/run_fast_stage1.py'


def load_entry_point():
    spec = importlib.util.spec_from_file_location('run_fast_stage1', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_fast_stage1 = load_entry_point()
EMA = run_fast_stage1.EMA
ema_weights = run_fast_stage1.ema_weights
trainable_parameters = run_fast_stage1.trainable_parameters


def make_model(fill, dtype=torch.float32):
    """One-parameter module; `weight` stays a leaf with requires_grad=True."""
    model = torch.nn.Linear(1, 1, bias=False).to(dtype)
    with torch.no_grad():
        model.weight.fill_(fill)
    return model


def set_weight(model, value):
    with torch.no_grad():
        model.weight.fill_(value)


# --- EMA maths ------------------------------------------------------------

def test_ema_shadow_matches_a_hand_computed_sequence():
    """shadow_0 = w_0, then shadow_k = 0.5*shadow_{k-1} + 0.5*w_k (decay 0.5).

    w = [1 (init), 2, 4, 8] -> shadows 1, 1.5, 2.75, 5.375.
    """
    model = make_model(1.0)
    ema = EMA(trainable_parameters(model), decay=0.5)
    torch.testing.assert_close(ema.shadow['weight'], torch.tensor([[1.0]]))

    expected = 1.0
    for weight, hand in ((2.0, 1.5), (4.0, 2.75), (8.0, 5.375)):
        set_weight(model, weight)
        ema.update(trainable_parameters(model))
        expected = 0.5 * expected + 0.5 * weight
        assert abs(expected - hand) < 1e-12
        torch.testing.assert_close(ema.shadow['weight'], torch.tensor([[hand]]))


def test_ema_decay_zero_equals_the_current_parameter():
    """decay=0 -> shadow = 0*shadow + 1*param, i.e. a copy of the live weights."""
    model = make_model(0.25)
    ema = EMA(trainable_parameters(model), decay=0.0)
    assert ema.enabled is False
    for weight in (-3.0, 7.5):
        set_weight(model, weight)
        ema.update(trainable_parameters(model))
        torch.testing.assert_close(ema.shadow['weight'], model.weight.detach())


def test_ema_shadow_is_fp32_for_a_bf16_parameter():
    model = make_model(1.0, dtype=torch.bfloat16)
    ema = EMA(trainable_parameters(model), decay=0.9)
    assert ema.shadow['weight'].dtype == torch.float32
    set_weight(model, 2.0)
    ema.update(trainable_parameters(model))
    # 0.9*1 + 0.1*2 = 1.1, in fp32 even though the parameter is bf16.
    torch.testing.assert_close(ema.shadow['weight'], torch.tensor([[1.1]]),
                               rtol=0, atol=1e-6)


def test_ema_state_dict_round_trip_is_a_copy():
    model = make_model(1.0)
    ema = EMA(trainable_parameters(model), decay=0.9)
    set_weight(model, 3.0)
    ema.update(trainable_parameters(model))
    state = ema.state_dict()

    fresh = EMA(trainable_parameters(make_model(0.0)), decay=0.0)
    fresh.load_state_dict(state)
    torch.testing.assert_close(fresh.shadow['weight'], ema.shadow['weight'])
    assert fresh.decay == 0.9
    # The checkpoint must not alias the live shadow.
    fresh.shadow['weight'].add_(10.0)
    torch.testing.assert_close(state['shadow']['weight'], ema.shadow['weight'])


# --- evaluation / restore --------------------------------------------------

def test_ema_evaluation_restores_the_live_weights_exactly():
    """A fake evaluation runs on the shadow and leaves the parameters bit-identical."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(),
                                torch.nn.Linear(3, 2))
    with torch.no_grad():  # a second, non-trainable buffer-like parameter set
        model[2].weight.add_(0.5)
    frozen = model[2].weight.detach().clone()
    model[2].weight.requires_grad_(False)

    ema = EMA(trainable_parameters(model), decay=0.75)
    for step in range(3):  # drift the live weights away from the shadow
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.add_(0.1 * (step + 1))
        ema.update(trainable_parameters(model))

    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    assert not torch.equal(before['0.weight'], ema.shadow['0.weight'])

    seen = {}
    with ema_weights(model, ema):
        for name, param in model.named_parameters():
            seen[name] = param.detach().clone()
        # Pretend the evaluation scribbles over the EMA-managed weights: the restore comes
        # from the clone backup, so even this cannot leak into the checkpoint.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ema.shadow:
                    param.mul_(0.0)

    for name, param in model.named_parameters():
        if name in ema.shadow:
            torch.testing.assert_close(seen[name], ema.shadow[name],
                                       rtol=0, atol=0, check_dtype=False)
        else:  # untouched (requires_grad=False) parameter keeps its original value
            torch.testing.assert_close(seen[name], before[name])
        # Lossless restore: element-wise identical, same dtype and device.
        assert param.dtype == before[name].dtype
        assert param.device == before[name].device
        assert torch.equal(param.detach(), before[name]), f'{name} was not restored'
    torch.testing.assert_close(model[2].weight.detach(), frozen, rtol=0, atol=0)


# --- loss normalisation ----------------------------------------------------

def per_sample_mean_then_batch_mean(nll, counts):
    """mean_s(sum_i(nll_si) / c_s): the 'average of per-sample means' reduction."""
    assert int(sum(counts)) == nll.numel()
    means, start = [], 0
    for count in counts:
        means.append(nll[start:start + count].mean())
        start += count
    return torch.stack(means).mean()


def test_loss_normalisations_agree_when_counts_are_equal():
    torch.manual_seed(7)
    counts = [3, 3, 3, 3]
    nll = torch.rand(sum(counts), dtype=torch.float64) * 4.0
    weights = torch.ones_like(nll)  # one count per valid token

    ours = global_weighted_mean((nll * weights).sum(), weights.sum())
    theirs = per_sample_mean_then_batch_mean(nll, counts)
    torch.testing.assert_close(ours, theirs, rtol=0, atol=1e-6)
    # Both are just the global mean of the per-token nll in this case.
    torch.testing.assert_close(ours, nll.mean(), rtol=0, atol=1e-6)


def test_loss_normalisations_differ_when_counts_are_unequal():
    counts = [2, 1]
    nll = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
    weights = torch.ones_like(nll)

    ours = global_weighted_mean((nll * weights).sum(), weights.sum())
    theirs = per_sample_mean_then_batch_mean(nll, counts)

    means = [torch.tensor(2.0), torch.tensor(5.0)]  # (1+3)/2 and 5/1
    total = float(sum(counts))
    analytic = sum(float(count) / total * float(mean)
                   for count, mean in zip(counts, means))       # ours: count-weighted
    analytic_theirs = sum(float(mean) for mean in means) / len(means)  # theirs: unweighted
    assert abs(float(ours) - analytic) < 1e-12
    assert abs(float(theirs) - analytic_theirs) < 1e-12
    # The gap is exactly sum_s (c_s/sum_c - 1/B) * m_s = (2/3-1/2)*2 + (1/3-1/2)*5 = -0.5.
    gap = float(ours) - float(theirs)
    assert abs(gap - (-0.5)) < 1e-12
    assert abs(gap) > 1e-3, 'unequal token counts must not reduce to the same value'

    # Same conclusion with a non-trivial per-sample weight: the weight enters the
    # numerator and the denominator, i.e. it re-weights the count, it does not equalise it.
    scaled = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)  # sample 2 weight 0
    ours_scaled = global_weighted_mean((nll * scaled).sum(), scaled.sum())
    torch.testing.assert_close(ours_scaled, torch.tensor(2.0, dtype=torch.float64),
                               rtol=0, atol=1e-12)
