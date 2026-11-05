"""soma-fluxvla owns plain PI0.5 sampling (plan B); pin what a refactor can silently break.

``scripts/check_predict_parity.py`` proves numerical agreement with the deployed
upstream FluxVLA on real weights and a GPU. These tests cover the parts that need
no weights at all: the Euler loop bounds, the prefix-RTC overwrite rule and the
fact that ``PI05FlowMatching`` actually exposes sampling.
"""
import inspect

import pytest
import torch

from fluxvla.models.vlas.pi0_flowmatching import PI0FlowMatching
from fluxvla.models.vlas.pi05_flowmatching import PI05FlowMatching


class _Stub:
    """Only the attribute the prefix-RTC integrator reads."""

    n_action_steps = 4


def test_plain_integrator_runs_the_upstream_number_of_steps():
    times = []

    def denoise(x, t):
        times.append(float(t[0]))
        return torch.ones_like(x)

    x = torch.zeros(2, 5, 7)
    out = PI0FlowMatching._predict_action_plain(
        _Stub(), x, denoise, 2, torch.tensor(-0.1), 'cpu')
    # Upstream starts at t=1.0 and steps while t >= -dt/2, so ten steps at dt=-0.1.
    assert len(times) == 10
    assert times[0] == pytest.approx(1.0)
    assert times[-1] == pytest.approx(0.1)
    assert torch.allclose(out, torch.full_like(x, -1.0), atol=1e-6)


def test_prefix_rtc_reapplies_the_prefix_and_zeroes_its_time():
    seen = []

    def denoise(x, t):
        seen.append(t.clone())
        return torch.zeros_like(x)

    prev = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3) / 10
    out = PI0FlowMatching._predict_action_prefix_rtc(
        _Stub(), torch.zeros(2, 4, 3), denoise, 2, torch.tensor(-0.5), 'cpu', prev, 2)
    assert torch.equal(out[:, :2], prev[:, :2])
    assert torch.equal(out[:, 2:], torch.zeros(2, 2, 3))
    assert len(seen) == 2  # t = 1.0, 0.5 before t < -dt/2
    for t_pos in seen:
        assert torch.count_nonzero(t_pos[:, :2]) == 0
        assert torch.count_nonzero(t_pos[:, 2:]) == t_pos[:, 2:].numel()


def test_pi05_flowmatching_inherits_the_ported_sampling():
    assert PI05FlowMatching.predict_action is PI0FlowMatching.predict_action
    assert PI05FlowMatching.denoise_step is PI0FlowMatching.denoise_step
    params = inspect.signature(PI0FlowMatching.predict_action).parameters
    # scripts/check_predict_parity.py feeds the recorded noise back in.
    assert {'noise', 'prev_actions', 'prefix_len', 'rtc_config'} <= set(params)
