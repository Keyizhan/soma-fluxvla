import pytest

from fluxvla.inference.subtask_stability import SubtaskStability

def tick(gate, text, t, **kw):
    d = gate.propose("sort", text, t, **kw)
    gate.commit(d)
    return d

def test_single_frame_flip_rejected_but_persistent_transition_accepted():
    gate = SubtaskStability(2)
    results = [tick(gate, x, i * .5) for i, x in enumerate(["A", "B", "A", "B", "B"])]
    assert [d.active for d in results] == ["A", "A", "A", "A", "B"]
    assert [d.switched for d in results] == [False, False, False, False, True]

def test_equivalent_text_does_not_change_action_condition():
    gate = SubtaskStability(2)
    label = "Left arm: pick. || Right arm: place.;"
    tick(gate, label, 0)
    d = tick(gate, "right arm: place || LEFT_arm: pick", .5)
    assert d.active == label and d.reason == "same_task"

def test_held_prediction_is_not_an_extra_vote():
    gate = SubtaskStability(2)
    tick(gate, "A", 0)
    tick(gate, "B", .5)
    d = tick(gate, "B", 1, fresh=False)
    assert d.active == "A" and d.count == 1 and d.observed_at == .5
    assert tick(gate, "B", 1.5).active == "B"

def test_failed_inference_does_not_commit_vote():
    gate = SubtaskStability(2)
    tick(gate, "A", 0)
    gate.propose("sort", "B", .5)
    assert tick(gate, "B", 1).active == "A"

def test_task_change_idle_and_clock_rewind_reset():
    for instruction, now in [("new task", 1), ("sort", 5), ("sort", -1)]:
        gate = SubtaskStability(2)
        tick(gate, "A", 0)
        d = gate.propose(instruction, "B", now)
        assert d.active == "B" and d.reason == "initialize"

def test_default_preserves_every_candidate_exactly():
    gate = SubtaskStability()
    for i, label in enumerate(["A.;", "a", "B", "A"]):
        assert tick(gate, label, i).active == label

@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_confirmations(k):
    with pytest.raises(ValueError):
        SubtaskStability(k)
