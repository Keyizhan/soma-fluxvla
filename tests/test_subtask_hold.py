import pytest

from fluxvla.inference.subtask_hold import SubtaskHold

def test_half_hz_text_with_two_hz_action_requests():
    hold = SubtaskHold(2.0)
    refreshes = []
    ids = []
    for now in [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
        record = hold.get("sort", now)
        if record is None:
            refreshes.append(now)
            record = hold.commit("sort", now, {"subtask": str(now)})
        ids.append(record["generation_id"])
    assert refreshes == [0, 2, 4]
    assert ids == [1, 1, 1, 1, 2, 2, 2, 2, 3]

def test_instruction_change_expiry_and_clock_rewind_invalidate():
    hold = SubtaskHold(2)
    hold.commit("sort", 10, {"subtask": "left arm"})
    assert hold.get("sort", 11) is not None
    assert hold.get("different task", 11) is None
    assert hold.get("sort", 12) is None
    assert hold.get("sort", 9) is None

def test_default_never_reuses_text():
    hold = SubtaskHold()
    hold.commit("sort", 0, {"subtask": "left arm"})
    assert hold.get("sort", 0) is None

@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_interval(value):
    with pytest.raises(ValueError):
        SubtaskHold(value)
