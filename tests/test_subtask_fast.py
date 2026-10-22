"""Contract tests for the FAST-token prompt layout (Ke-Wang reference parity)."""
import numpy as np
import pytest

from fluxvla import register_training_components
register_training_components()
from fluxvla.transforms.subtask_fast_prompts import (
    PrepareSubtaskFastPrompts, clean_reference)

MODEL_VOCAB = 257152
FAST_VOCAB = 2048
SKIP = 128
TOP = MODEL_VOCAB - 1 - SKIP


class CharTokenizer:
    """Deterministic character-level stand-in (ids 3.., BOS=2, EOS=1, PAD=0)."""
    bos_token_id, eos_token_id, pad_token_id = 2, 1, 0

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(ch) % 251 for ch in text]

    def decode(self, ids):
        return ''.join(chr(int(i) - 3) for i in ids)


def prompts(**kwargs):
    base = dict(max_len=256, state_dim=4, model_vocab_size=MODEL_VOCAB,
                fast_vocab_size=FAST_VOCAB, fast_skip_tokens=SKIP)
    base.update(kwargs)
    return PrepareSubtaskFastPrompts(CharTokenizer(), **base)


def fast_ids(n=5):
    return np.arange(7, 7 + n, dtype=np.int64)


def valid_slice(packed):
    return int(packed['valid_mask'].sum())


def segments(packed):
    """Split the valid region into (name, [positions]) using the masks."""
    n = valid_slice(packed)
    sub = packed['subtask_region_mask'][:n]
    act = packed['action_region_mask'][:n]
    loss = packed['loss_mask'][:n]
    idx = np.arange(n)
    return dict(prefix=idx[~loss & ~sub & ~act], subtask=idx[sub], action=idx[act])


def test_reference_cleaning():
    # Reference cleaning does not collapse repeated whitespace; parity is intentional.
    assert clean_reference('  Pick_Up  the CUP. ') == 'pick up  the cup.'
    assert clean_reference('pick up the cup.') == 'pick up the cup.'
    assert clean_reference('抓住布片!') == '抓住布片.'
    assert clean_reference('') == '.'


def test_fast_layout_masks_and_sentinel():
    p = prompts()
    ids = fast_ids()
    packed = p.high_level('Pick up the cup', np.array([0.5, -0.25, 1.0, 0.0]), 'move the arm', ids)
    seg = segments(packed)
    mapped = p.fast_to_model(ids)
    n = valid_slice(packed)

    # EOS is the last valid token and lives inside the action region (reference layout).
    assert packed['input_ids'][n - 1] == p.eos_id
    assert packed['action_region_mask'][n - 1]
    # FAST ids appear contiguously, exactly at action-region positions, in order.
    positions = [i for i in seg['action'] if packed['input_ids'][i] in set(mapped.tolist())]
    assert [int(packed['input_ids'][i]) for i in positions] == mapped.tolist()
    assert np.diff(positions).tolist() == [1] * (len(positions) - 1)
    # The `|` sentinel sits between the FAST ids and EOS.
    assert packed['input_ids'][positions[-1] + 2] == p.eos_id
    # Default layout follows the reference: the whole real sequence is causal.
    n_valid = int(packed['valid_mask'].sum())
    assert packed['ar_mask'][:n_valid].all()
    # The bidirectional prefix is the opt-in deviation switch.
    open_prefix = prompts(all_causal=False).high_level('Pick up the cup',
                                                       np.array([0.5, -0.25, 1.0, 0.0]),
                                                       'move the arm', ids)
    assert not open_prefix['ar_mask'][:len(seg['prefix'])].any()
    assert packed['ar_mask'][n:].sum() == 0
    # With the reference (all-causal) default every real token is causal, not only targets.
    assert packed['ar_mask'][:len(packed['valid_mask'])].sum() == int(packed['valid_mask'].sum())
    # Region masks partition the supervised region.
    assert np.array_equal(packed['loss_mask'][:n],
                          packed['subtask_region_mask'][:n] | packed['action_region_mask'][:n])
    assert packed['input_ids'][0] == p.bos_id
    assert p.check(packed) is None


def test_shifted_targets_stay_inside_their_region():
    p = prompts()
    ids = fast_ids(3)
    packed = p.high_level('g', np.zeros(4), 's', ids)
    n = valid_slice(packed)
    tokens = packed['input_ids']
    for name, mask_key in (('subtask', 'subtask_region_mask'), ('action', 'action_region_mask')):
        shifted = packed[mask_key][1:]  # next-token shift: position i predicts token i+1
        targets = np.where(shifted, tokens[1:], -100)[:n - 1]
        assert (targets != -100).any(), name
    # Supervised targets must never include padding.
    assert all(int(t) != p.pad_id for t in packed['input_ids'][:n])


def test_labels_are_the_supervised_tokens():
    p = prompts()
    packed = p.high_level('g', np.zeros(4), 's', fast_ids(2))
    if 'labels' not in packed:
        pytest.skip('labels key not emitted')
    expected = np.where(packed['loss_mask'], packed['input_ids'], -100)
    assert np.array_equal(packed['labels'], expected)
    assert set(np.unique(packed['labels'][packed['loss_mask']])).issubset(set(packed['input_ids'].tolist()))
    # Region masks must be recoverable from the labels and the FAST range.
    assert (packed['labels'] != -100).sum() == packed['loss_mask'].sum()


def test_flow_mode_has_no_action_region():
    p = prompts()
    packed = p.high_level('g', np.zeros(4), 'place the shoe')
    n = valid_slice(packed)
    assert not packed['action_region_mask'].any()
    assert packed['subtask_region_mask'][:n][packed['loss_mask'][:n]].all()
    assert packed['input_ids'][n - 1] == p.eos_id


def test_fast_to_model_mapping_bounds_and_inverse():
    p = prompts()
    ids = np.array([0, 1, FAST_VOCAB - 1], dtype=np.int64)
    mapped = p.fast_to_model(ids)
    assert mapped[0] == TOP and mapped[-1] == TOP - (FAST_VOCAB - 1)
    assert np.array_equal(p.model_to_fast(mapped), ids)
    with pytest.raises(ValueError, match='out of range'):
        p.fast_to_model(np.array([FAST_VOCAB]))
    with pytest.raises(ValueError, match='not in the relocated FAST range'):
        p.model_to_fast(np.array([5]))


def test_overflow_fails_fast_with_breakdown():
    p = prompts(max_len=24)
    with pytest.raises(ValueError, match='breakdown'):
        p.high_level('a very long overall instruction ' * 4, np.zeros(4), 'subtask', fast_ids(2))
    with pytest.raises(ValueError, match='no silent'):
        p.high_level('a very long overall instruction ' * 4, np.zeros(4), 'subtask', fast_ids(2))


def test_check_rejects_inconsistent_masks():
    p = prompts()
    packed = p.high_level('g', np.zeros(4), 's', fast_ids(2))
    broken = {k: np.array(v, copy=True) for k, v in packed.items()}
    broken['action_region_mask'][broken['subtask_region_mask']] = True
    with pytest.raises(ValueError, match='disjoint'):
        p.check(broken)
    broken = {k: np.array(v, copy=True) for k, v in packed.items()}
    broken['loss_mask'][-1] = True
    with pytest.raises(ValueError, match='padding must carry no masks'):
        p.check(broken)
    broken = {k: np.array(v, copy=True) for k, v in packed.items()}
    broken['loss_mask'][:] = False
    broken['subtask_region_mask'][:] = False
    broken['action_region_mask'][:] = False
    with pytest.raises(ValueError, match='No supervised token'):
        p.check(broken)


def test_all_causal_variant_matches_the_reference_attention():
    p = prompts()                              # Ke-Wang default
    packed = p.high_level('g', np.zeros(4), 's', fast_ids(2))
    n = valid_slice(packed)
    assert packed['ar_mask'][:n].all(), 'condition and targets are all causal'
    assert not packed['ar_mask'][n:].any(), 'padding is never causal'
    assert p.variant.endswith('+allcausal')
    assert p.check(packed) is None
    # Bidirectional variant keeps the condition open (deviation switch).
    q = prompts(all_causal=False)
    ref = q.high_level('g', np.zeros(4), 's', fast_ids(2))
    m = valid_slice(ref)
    assert not ref['ar_mask'][:m][~ref['loss_mask'][:m]].any()
    assert q.variant == q.version


def test_low_level_task_switch():
    sub, overall = 'move the arm', 'sort the fabric'
    off = prompts(low_level_sees_task=False).low_level(sub, np.zeros(4))
    on = prompts().low_level(sub, np.zeros(4), overall)      # Ke-Wang default
    assert len(on['valid_mask'].nonzero()[0]) > len(off['valid_mask'].nonzero()[0])
    with pytest.raises(ValueError, match='requires overall_task'):
        prompts().low_level(sub, np.zeros(4))


def test_context_returns_an_ar_mask():
    off = prompts(all_causal=False).context('g', np.zeros(4))
    on = prompts().context('g', np.zeros(4))
    assert off['ar_mask'].shape == off['input_ids'].shape
    assert not off['ar_mask'].any()
    assert on['ar_mask'][:int(on['valid_mask'].sum())].all()


def test_call_wires_high_and_low_level():
    p = prompts()
    out = p(dict(states=np.zeros(4), overall_task='g', subtask='move the arm',
                 fast_token_ids=fast_ids(4)))
    assert out['hl_action_region_mask'].any()
    assert out['ll_input_ids'].shape == (p.max_len,)
    assert out['ll_valid_mask'].any() and not out['ll_valid_mask'].all()
    assert out['has_hl'] and out['has_action']


def test_defaults_follow_the_reference_implementation():
    p = prompts()
    assert p.all_causal is True
    assert p.low_level_sees_task is True
    assert p.variant.endswith('+allcausal')


def test_reference_hybrid_low_prefix_matches_high_without_action_region():
    p = prompts(low_level_action_tail='', low_level_segmented=True)
    high = p.high_level('sort', np.zeros(4), 'pick', fast_ids())
    low = p.low_level('pick', np.zeros(4), 'sort')
    expected = high['input_ids'][high['valid_mask'] & ~high['action_region_mask']]
    np.testing.assert_array_equal(expected,low['input_ids'][low['valid_mask']])


def test_reference_expert_prefix_includes_action_delimiter_and_eos():
    p = prompts(low_level_action_tail='\nAction: ', low_level_eos=True, low_level_segmented=True)
    high = p.high_level('sort', np.zeros(4), 'pick', None)
    low = p.low_level('pick', np.zeros(4), 'sort')
    np.testing.assert_array_equal(high['input_ids'],low['input_ids'])


@pytest.mark.parametrize('segmented', [False, True])
def test_task_free_prefix_is_identical_for_changed_or_missing_overall_task(segmented):
    p = prompts(low_level_sees_task=False, low_level_prompt_format='state_subtask',
                low_level_action_tail='', low_level_eos=False,
                low_level_segmented=segmented, all_causal=False)
    state = np.array([.1, -.2, .3, .4])
    variants = [p.low_level('move the arm', state, task)
                for task in ('sort red fabric', 'discard blue fabric', None)]
    for other in variants[1:]:
        for key in variants[0]:
            np.testing.assert_array_equal(variants[0][key], other[key])
    valid = variants[0]['input_ids'][variants[0]['valid_mask']]
    text = p.decode(valid)
    assert text.startswith('State: ') and '; Subtask: move the arm.;' in text
    assert 'Task:' not in text and 'Action:' not in text
    assert p.eos_id not in valid
    assert not np.any((valid > TOP - FAST_VOCAB) & (valid <= TOP))
    # Overall task remains an input to the high-level planner.
    left = p.context('sort red fabric', state)['input_ids']
    right = p.context('discard blue fabric', state)['input_ids']
    assert not np.array_equal(left, right)


@pytest.mark.parametrize('bad', [dict(low_level_sees_task=True),
                                  dict(low_level_action_tail='Action:'),
                                  dict(low_level_eos=True)])
def test_task_free_format_rejects_incompatible_tail_or_condition(bad):
    config = dict(low_level_sees_task=False, low_level_prompt_format='state_subtask',
                  low_level_action_tail='', low_level_eos=False)
    config.update(bad)
    with pytest.raises(ValueError, match='state_subtask requires'):
        prompts(**config)
