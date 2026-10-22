"""FAST-token prompt contract for pi0.5 subtask + FAST supervision.

Mirrors the reference implementation
``Ke-Wang1017/openpi_subtask@385d0e8330397ecfa29e9b675042b424c4d5386c``
(``src/openpi/models/tokenizer.py::PaligemmaTokenizer.tokenize_high_low_prompt``).

Deliberate departures from the reference (documented, tested):
  * Text defaults to causal attention, matching the reference.
  * Overflow is a hard error with a diagnostic breakdown instead of a silent
    truncation that would cut FAST tokens and the ``|`` / EOS sentinel.

FAST action tokens are *precomputed offline* (see
``scripts/architecture/prepare_fast_tokens.py``) and passed in as raw FAST ids in
``[0, fast_vocab_size)``; this module only performs the vocabulary relocation and
the mask bookkeeping, so it needs neither scipy nor remote-code transformers.
"""
import numpy as np

from fluxvla.engines import TRANSFORMS, build_tokenizer_from_cfg

REFERENCE = 'Ke-Wang1017/openpi_subtask@385d0e8330397ecfa29e9b675042b424c4d5386c'
PUNCTUATION = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
STATE_BIN_EDGES = np.linspace(-1, 1, 257)[:-1]


def clean_reference(text):
    """Lowercase, normalise separators, strip trailing punctuation, append '.'."""
    value = str(text).lower().strip().replace('_', ' ').replace('\n', ' ')
    if value and value[-1] in PUNCTUATION:
        value = value[:-1]
    return value + '.'


@TRANSFORMS.register_module()
class PrepareSubtaskFastPrompts:
    """Build the six parallel arrays required by FAST-token training.

    Layout (fast mode), segments are contiguous and in this order::

        [BOS] Task: <g>. State: <bins>; Subtask: <s>;  \\nAction: <fast>|<EOS>

    ``subtask_region_mask`` covers ``<s>;``; ``action_region_mask`` covers
    ``\\nAction: <fast>|`` + EOS. Both are subsets of ``loss_mask``. All masks are
    expressed on *input* positions; cross-entropy targets are shifted by one.
    """

    version = 'pi05-subtask-fast-v1'

    def __init__(self, tokenizer, max_len=256, state_dim=32, state_bins=256,
                 model_vocab_size=None, fast_vocab_size=2048, fast_skip_tokens=128,
                 low_level_sees_task=True, all_causal=True, low_level_action_tail='\nAction: ', low_level_eos=False,
                 low_level_segmented=False, low_level_prompt_format="legacy"):
        if low_level_prompt_format not in ('legacy', 'state_subtask'):
            raise ValueError('unsupported low_level_prompt_format')
        if low_level_prompt_format == 'state_subtask' and (
                low_level_sees_task or low_level_action_tail or low_level_eos):
            raise ValueError('state_subtask requires no overall task, Action tail, or EOS')
        self.low_level_prompt_format = low_level_prompt_format
        self.low_level_segmented = low_level_segmented
        self.low_level_eos = low_level_eos
        self.low_level_action_tail = low_level_action_tail
        self.tokenizer = (build_tokenizer_from_cfg(tokenizer)
                          if isinstance(tokenizer, dict) else tokenizer)
        self.max_len = int(max_len)
        self.state_dim = int(state_dim)
        self.state_bins = int(state_bins)
        self.model_vocab_size = model_vocab_size
        self.fast_vocab_size = int(fast_vocab_size)
        self.fast_skip_tokens = int(fast_skip_tokens)
        # T3.4 switch: the reference implementation lets the action context see the
        # overall task; p(a|o,s) does not. The training default follows the reference.
        self.low_level_sees_task = bool(low_level_sees_task)
        # Defaults follow the Ke-Wang reference (all-causal text, action context sees the
        # overall task). Set these False to get the paper-form deviations:
        #   all_causal=False            -> bidirectional conditioning prefix
        #   low_level_sees_task=False   -> p(a|o,s) without the overall task
        self.all_causal = bool(all_causal)
        self.variant = self.version + ('+allcausal' if self.all_causal else '')
        if self.low_level_prompt_format == 'state_subtask':
            self.variant += '+state-subtask'
        self.pad_id = self.tokenizer.pad_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        if any(x is None for x in (self.pad_id, self.bos_id, self.eos_id)):
            raise ValueError('BOS, EOS and PAD IDs must be defined')
        if not self.state_bins == len(STATE_BIN_EDGES):
            raise ValueError('state_bins must match the discretisation table')
        if self.model_vocab_size is not None:
            highest = self.model_vocab_size - 1 - self.fast_skip_tokens
            if highest - self.fast_vocab_size + 1 < 0:
                raise ValueError('FAST vocabulary does not fit below the model vocabulary')

    # ------------------------------------------------------------------ tokens
    def state_text(self, states):
        values = np.asarray(states, dtype=np.float64).reshape(-1)
        if values.size > self.state_dim or not np.isfinite(values).all():
            raise ValueError('Expected finite, already-normalized state within state_dim')
        padded = np.zeros(self.state_dim)
        padded[:values.size] = values
        bins = np.digitize(padded, bins=STATE_BIN_EDGES) - 1
        return ' '.join(map(str, bins.tolist()))

    def _encode(self, text):
        ids = [int(x) for x in self.tokenizer.encode(text, add_special_tokens=False)]
        if self.model_vocab_size is not None:
            if any(x < 0 or x >= self.model_vocab_size for x in ids):
                raise ValueError('Text contains a token outside the model vocabulary')
        return ids

    def fast_to_model(self, fast_ids):
        """Relocate raw FAST ids into the tail of the PaliGemma vocabulary."""
        ids = np.asarray(fast_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError('Empty FAST token sequence')
        if ids.min() < 0 or ids.max() >= self.fast_vocab_size:
            raise ValueError(f'FAST ids out of range [0,{self.fast_vocab_size}): '
                             f'min={ids.min()} max={ids.max()}')
        if self.model_vocab_size is None:
            raise ValueError('model_vocab_size is required to relocate FAST ids')
        mapped = self.model_vocab_size - 1 - self.fast_skip_tokens - ids
        if mapped.min() < 0 or mapped.max() >= self.model_vocab_size:
            raise ValueError('Relocated FAST ids fall outside the model vocabulary')
        return mapped

    def model_to_fast(self, model_ids):
        ids = np.asarray(model_ids, dtype=np.int64).reshape(-1)
        top = self.model_vocab_size - 1 - self.fast_skip_tokens
        if ids.size and (ids.min() <= top - self.fast_vocab_size or ids.max() > top):
            raise ValueError('Token ids are not in the relocated FAST range')
        return top - ids

    # ----------------------------------------------------------------- packing
    def _finalise(self, segments, breakdown):
        tokens, valid, ar, loss, subtask_region, action_region = ([], [], [], [], [], [])
        for ids, is_target, region in segments:
            n = len(ids)
            tokens += ids
            valid += [True] * n
            ar += [True if self.all_causal else bool(is_target)] * n
            loss += [bool(is_target)] * n
            subtask_region += [region == 'subtask'] * n
            action_region += [region == 'action'] * n
        total = len(tokens)
        if total > self.max_len:
            raise ValueError(
                f'Prompt needs {total} tokens but max_len={self.max_len}; no silent '
                f'truncation (would cut FAST tokens / sentinel). breakdown={breakdown}')
        pad = self.max_len - total
        ids = np.asarray(tokens + [self.pad_id] * pad, dtype=np.int64)
        out = dict(
            input_ids=ids,
            labels=np.where(np.asarray(loss + [False] * pad, dtype=bool), ids, -100),
            valid_mask=np.asarray(valid + [False] * pad, dtype=bool),
            ar_mask=np.asarray(ar + [False] * pad, dtype=bool),
            loss_mask=np.asarray(loss + [False] * pad, dtype=bool),
            subtask_region_mask=np.asarray(subtask_region + [False] * pad, dtype=bool),
            action_region_mask=np.asarray(action_region + [False] * pad, dtype=bool),
        )
        self.check(out)
        return out

    @staticmethod
    def check(packed):
        valid = packed['valid_mask']
        ar = packed['ar_mask']
        loss = packed['loss_mask']
        sub = packed['subtask_region_mask']
        act = packed['action_region_mask']
        if np.any(~valid & (ar | loss | sub | act)):
            raise ValueError('padding must carry no masks')
        if np.any(sub & act):
            raise ValueError('subtask and action regions must be disjoint')
        if np.any((sub | act) & ~loss):
            raise ValueError('region masks must be subsets of loss_mask')
        if np.any(loss & (~valid | ~ar)):
            raise ValueError('supervised positions must be valid causal targets')
        if not loss.any():
            raise ValueError('No supervised token in prompt')
        if not packed['input_ids'][:int(valid.sum())].size:
            raise ValueError('Empty prompt')

    def high_level(self, overall_task, states, subtask, fast_token_ids=None):
        """Fast mode when ``fast_token_ids`` is given, subtask-only mode otherwise."""
        condition = (f'Task: {clean_reference(overall_task)}; '
                     f'State: {self.state_text(states)}; Subtask: ')
        prefix = [self.bos_id] + self._encode(condition)
        subtask_segment = self._encode(clean_reference(subtask) + ';')
        segments = [(prefix, False, 'none'), (subtask_segment, True, 'subtask')]
        breakdown = dict(prefix=len(prefix), subtask=len(subtask_segment), action=0)
        if fast_token_ids is not None:
            action = (self._encode('\nAction: ')
                      + self.fast_to_model(fast_token_ids).tolist()
                      + self._encode('|') + [self.eos_id])
            segments.append((action, True, 'action'))
            breakdown['action'] = len(action)
        else:
            # Reference flow mode: subtask, then the Action sentinel and EOS.
            tail = self._encode('\nAction: ') + [self.eos_id]
            segments.append((tail, True, 'subtask'))
            breakdown['subtask'] += len(tail)
        breakdown['total'] = sum(breakdown[k] for k in ('prefix', 'subtask', 'action'))
        return self._finalise(segments, breakdown)

    def stop_token_ids(self):
        # Ids of the "\nAction: " delimiter that follows the subtask in fast mode.
        # Fast mode is "<subtask>;\nAction: <FAST>|<EOS>", so the model never emits an EOS
        # right after the subtask; decoding must stop at this delimiter, otherwise the FAST
        # tokens are scored as text and the character error rate is meaningless.
        return tuple(self._encode('\nAction: ')) or (None,)

    def decode_subtask(self, ids):
        # Decode only the subtask text: stop at EOS *or* at the Action delimiter.
        stop = self.stop_token_ids()
        kept, window = [], []
        for token in ids:
            token = int(token)
            if token == self.eos_id:
                break
            window.append(token)
            if len(window) > len(stop):
                kept.append(window.pop(0))
            if len(window) == len(stop) and tuple(window) == stop:
                break
        values = [t for t in kept if t not in (self.pad_id, self.bos_id)]
        inner = getattr(self.tokenizer, 'tokenizer', self.tokenizer)
        return inner.decode(values).strip()

    def decode(self, ids):
        """Decode generated text, stopping at EOS and dropping PAD/BOS."""
        values = []
        for token in ids:
            token = int(token)
            if token == self.eos_id:
                break
            if token not in (self.pad_id, self.bos_id):
                values.append(token)
        inner = getattr(self.tokenizer, 'tokenizer', self.tokenizer)
        return inner.decode(values).strip()

    def context(self, overall_task, states):
        """Conditioning-only prompt used for generation: everything before the target."""
        text = (f'Task: {clean_reference(overall_task)}; '
                f'State: {self.state_text(states)}; Subtask: ')
        ids = [self.bos_id] + self._encode(text)
        if len(ids) > self.max_len:
            raise ValueError(f'Context needs {len(ids)} tokens > max_len={self.max_len}')
        pad = self.max_len - len(ids)
        return dict(input_ids=np.asarray(ids + [self.pad_id] * pad, dtype=np.int64),
                    valid_mask=np.asarray([True] * len(ids) + [False] * pad, dtype=bool),
                    ar_mask=np.asarray([self.all_causal] * len(ids) + [False] * pad,
                                       dtype=bool))

    def low_level(self, subtask, states, overall_task=None):
        """Conditioning-only prefix for the action expert (no supervised tokens)."""
        if self.low_level_prompt_format == 'state_subtask':
            # A separate context is required: generated high-level KV still contains
            # the overall task even when the visible low-level text omits it.
            condition = f'State: {self.state_text(states)}; Subtask: '
            answer = clean_reference(subtask) + ';'
            text = condition + answer
        elif self.low_level_sees_task:
            if overall_task is None:
                raise ValueError('low_level_sees_task=True requires overall_task')
            text = (f'Task: {clean_reference(overall_task)}; '
                    f'State: {self.state_text(states)}; '
                    f'Subtask: {clean_reference(subtask)};' + self.low_level_action_tail)
        else:
            text = f'Task: {clean_reference(subtask)}; State: {self.state_text(states)};\nAction: '
        ids = [self.bos_id] + self._encode(text)
        if self.low_level_segmented and self.low_level_prompt_format == 'state_subtask':
            ids = [self.bos_id] + self._encode(condition) + self._encode(answer)
        elif self.low_level_segmented and self.low_level_sees_task:
            condition = (f'Task: {clean_reference(overall_task)}; '
                         f'State: {self.state_text(states)}; Subtask: ')
            ids = ([self.bos_id] + self._encode(condition)
                   + self._encode(clean_reference(subtask) + ';')
                   + (self._encode(self.low_level_action_tail) if self.low_level_action_tail else []))
        if self.low_level_eos:
            ids.append(self.eos_id)
        if len(ids) > self.max_len:
            raise ValueError(f'Low-level prompt needs {len(ids)} tokens > max_len={self.max_len}')
        pad = self.max_len - len(ids)
        return dict(input_ids=np.asarray(ids + [self.pad_id] * pad, dtype=np.int64),
                    valid_mask=np.asarray([True] * len(ids) + [False] * pad, dtype=bool))

    def __call__(self, inputs):
        fast_ids = inputs.get('fast_token_ids')
        high = self.high_level(inputs['overall_task'], inputs['states'],
                               inputs['subtask'], fast_ids)
        low = self.low_level(inputs['subtask'], inputs['states'], inputs.get('overall_task'))
        out = dict(inputs)
        out.update({f'hl_{key}': value for key, value in high.items()})
        out.update(ll_input_ids=low['input_ids'], ll_valid_mask=low['valid_mask'])
        out['has_hl'] = np.array(True)
        out['has_action'] = np.array(True)
        return out
