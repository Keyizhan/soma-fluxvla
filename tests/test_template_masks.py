"""Generic corpus masks and exact tokenizer alignment, without model weights."""
import numpy as np
import pytest
from fluxvla.transforms.template_masks import TemplateMaskSpec, normalize, token_key_mask
from fluxvla.transforms.subtask_fast_prompts import PrepareSubtaskFastPrompts


class OffsetTokenizer:
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 100 for c in text]

    def __call__(self, text, **kwargs):
        return dict(input_ids=self.encode(text), offset_mapping=[(i, i + 1) for i in range(len(text))])


def test_induction_distinct_labels_family_verbs_and_roundtrip():
    labels = ['Move red cloth to basket', 'Place blue cloth to basket']
    spec = TemplateMaskSpec.fit(labels * 7)
    assert spec.fingerprint == TemplateMaskSpec.fit(reversed(labels)).fingerprint
    assert spec.fingerprint == TemplateMaskSpec.from_dict(spec.to_dict()).fingerprint
    text = normalize(labels[0])
    mask = spec.key_char_mask(text)
    assert mask[:4].all() and mask[5:8].all()
    assert not mask[text.index('cloth'):text.index('cloth') + 5].any()
    assert spec.key_char_mask(normalize('entirely unseen sentence')).all()
    assert TemplateMaskSpec.fit(['same'] * 10).key_char_mask(normalize('same')).all()


def test_multilingual_explicit_slots_and_straddling_subwords():
    spec = TemplateMaskSpec(templates=['将{object}放到{destination}'])
    text = '将红布放到左箱.;'
    assert spec.key_char_mask(text).tolist() == [False, True, True, False, False, True, True, False, False]

    class SubwordTokenizer:
        def __call__(self, text, **kwargs):
            return dict(input_ids=[7, 8, 9], offset_mapping=[(0, 2), (2, 5), (5, 9)])

    # All three cross a keyword span; a mixed token must never lose supervision.
    assert token_key_mask(SubwordTokenizer(), text, [7, 8, 9], spec).all()
    with pytest.warns(RuntimeWarning, match='input_ids_mismatch'):
        assert token_key_mask(SubwordTokenizer(), text, [8, 8, 9], spec).all()
    with pytest.warns(RuntimeWarning, match='offsets_unavailable'):
        assert token_key_mask(object(), text, [7, 8, 9], spec).all()
    assert spec.diagnostics['input_ids_mismatch'] == 1
    with pytest.raises(ValueError, match='slot'):
        TemplateMaskSpec(templates=['a constant whole sentence'])


def test_real_local_tokenizer_offsets():
    import os
    from pathlib import Path
    path = Path(os.environ.get('TEMPLATE_TEST_TOKENIZER', 'D:/dl_models/Qwen3-embedding4B'))
    if not (path / 'tokenizer.json').exists():
        pytest.skip('Set TEMPLATE_TEST_TOKENIZER to a local fast tokenizer directory')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    spec = TemplateMaskSpec(templates=['move {object} to basket', '将{object}放到{destination}'])
    for text in ['move red fabric to basket.;', '将红布放到左箱.;']:
        ids = tokenizer.encode(text, add_special_tokens=False)
        mask = token_key_mask(tokenizer, text, ids, spec)
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        chars = spec.key_char_mask(text)
        assert mask.tolist() == [bool(chars[a:b].any()) for a, b in offsets]
        assert mask.any() and not mask.all()
    assert spec.diagnostics == {'aligned_segments': 2}


def test_prompt_masks_cover_region_without_changing_tokenization():
    tok = OffsetTokenizer()
    common = dict(tokenizer=tok, max_len=512, state_dim=2, model_vocab_size=4096)
    plain = PrepareSubtaskFastPrompts(**common)
    split = PrepareSubtaskFastPrompts(**common, subtask_templates=['move {object} to basket'])
    for task, state, label in [('longer overall task', [0, 1], 'Move red cloth to basket'),
                               ('short', [-1], 'Move blue cloth to basket')]:
        for fast in ([1, 2, 3], None):
            a = plain.high_level(task, state, label, fast)
            b = split.high_level(task, state, label, fast)
            assert np.array_equal(a['input_ids'], b['input_ids'])
            assert not (b['subtask_key_mask'] & b['subtask_template_mask']).any()
            assert np.array_equal(b['subtask_key_mask'] | b['subtask_template_mask'], b['subtask_region_mask'])
            assert b['subtask_key_mask'].sum() == len(label.split(' ', 1)[1].rsplit(' to ', 1)[0])
            assert b['action_code_mask'].sum() == (3 if fast is not None else 0)
            assert not (b['action_code_mask'] & ~b['action_region_mask']).any()
