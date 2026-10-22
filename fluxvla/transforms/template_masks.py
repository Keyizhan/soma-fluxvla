"""Train-only, deterministic lexical template masks (not semantic annotation).

Corpus induction retains a common subsequence of lexical units across distinct
training labels. Thus cross-family differences remain keys. Repeated examples
do not bias the skeleton; a singleton corpus and unseen labels are all-key.
Constants can still be semantically important: explicit {slot} templates are
preferred when semantic supervision is available. No validation-time fitting.
"""
import hashlib
import json
import re
import string
import warnings
from collections import Counter
from difflib import SequenceMatcher

import numpy as np


def normalize(text):
    value = str(text).lower().strip().replace('_', ' ').replace('\n', ' ')
    if value and value[-1] in string.punctuation:
        value = value[:-1]
    return value + '.'


def units(text):
    # CJK characters remain individual units; other scripts use word units.
    return list(re.finditer(r'[\u3400-\u9fff]|[^\W_]+|[^\w\s]', text, re.UNICODE))


class TemplateMaskSpec:
    VERSION = 1

    def __init__(self, records=None, templates=()):
        self.records = records or {}
        self.templates = list(templates)
        self.diagnostics = Counter()
        for template in self.templates:
            if not re.search(r'\{[^{}]+\}', template):
                raise ValueError('Explicit templates must contain at least one {slot}')
        for text, spans in self.records.items():
            if not isinstance(text, str) or any(
                    len(span) != 2 or not all(isinstance(x, int) for x in span)
                    or not 0 <= span[0] < span[1] <= len(text) for span in spans):
                raise ValueError('Invalid template character spans')

    @classmethod
    def fit(cls, train_texts):
        texts = sorted(set(normalize(t) for t in train_texts))
        records = {text: [] for text in texts}
        if len(texts) < 2:
            return cls(records)
        skeleton = [m.group() for m in units(texts[0])]
        for text in texts[1:]:
            other = [m.group() for m in units(text)]
            matcher = SequenceMatcher(None, skeleton, other, autojunk=False)
            skeleton = [skeleton[i] for block in matcher.get_matching_blocks()
                        for i in range(block.a, block.a + block.size)]
        for text in texts:
            matches = units(text)
            matcher = SequenceMatcher(None, skeleton, [m.group() for m in matches], autojunk=False)
            records[text] = [[matches[i].start(), matches[i].end()]
                             for block in matcher.get_matching_blocks()
                             for i in range(block.b, block.b + block.size)]
        return cls(records)

    def to_dict(self):
        return dict(version=self.VERSION, records=self.records, templates=self.templates)

    @classmethod
    def from_dict(cls, value):
        if value.get('version') != cls.VERSION:
            raise ValueError('Unsupported template mask specification version')
        return cls(value.get('records'), value.get('templates', ()))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                        ensure_ascii=False).encode('utf-8')).hexdigest()

    def key_char_mask(self, text):
        """Text must be the exact normalized segment, including optional ';'."""
        base = text[:-1] if text.endswith(';') else text
        for template in self.templates:
            # Protect slot markers from the normalizer's underscore replacement.
            parts = re.split(r'(\{[^{}]+\})', template)
            pattern = ''.join('(.+?)' if p.startswith('{') and p.endswith('}')
                              else re.escape(p.lower().replace('_', ' ').replace('\n', ' '))
                              for p in parts)
            match = re.fullmatch(pattern, base.rstrip('.')) or re.fullmatch(pattern, base)
            if match:
                mask = np.zeros(len(text), dtype=bool)
                for a, b in (match.span(i) for i in range(1, len(match.groups()) + 1)):
                    mask[a:b] = True
                return mask
        mask = np.ones(len(text), dtype=bool)
        if base not in self.records or not self.records[base]:
            return mask
        for a, b in self.records[base]:
            mask[a:b] = False
        # Whitespace and appended protocol punctuation carry no semantic field.
        for i, char in enumerate(text):
            if char.isspace() or i >= len(base):
                mask[i] = False
        return mask


def token_key_mask(tokenizer, text, token_ids, spec):
    """Use exact offset mapping, never retokenize segments separately.

    Offset-less tokenizers or mismatched encodings fall back to all-key rather
    than silently misaligning loss weights. A token touching a key is a key.
    """
    key = spec.key_char_mask(text)
    inner = getattr(tokenizer, 'tokenizer', tokenizer)
    def fallback(reason):
        spec.diagnostics[reason] += 1
        if spec.diagnostics[reason] == 1 and (spec.records or spec.templates):
            warnings.warn('Template mask alignment fallback: ' + reason +
                          '; all tokens in this segment are treated as keys.',
                          RuntimeWarning, stacklevel=2)
        return np.ones(len(token_ids), dtype=bool)
    try:
        encoded = inner(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded['input_ids']) != list(token_ids):
            return fallback('input_ids_mismatch')
        offsets = encoded['offset_mapping']
        if len(offsets) != len(token_ids):
            return fallback('offset_length_mismatch')
        if any(b <= a or a < 0 or b > len(text) for a, b in offsets):
            return fallback('invalid_offsets')
        spec.diagnostics['aligned_segments'] += 1
        return np.asarray([True if b <= a or a < 0 or b > len(text)
                           else key[a:b].any() for a, b in offsets], dtype=bool)
    except (TypeError, AttributeError, KeyError, NotImplementedError, ValueError):
        return fallback('offsets_unavailable')
