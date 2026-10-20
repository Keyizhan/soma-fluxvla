"""Architecture v1 prompt contract; an AI implementation, not an official template."""
import numpy as np

from fluxvla.engines import TRANSFORMS, build_tokenizer_from_cfg


@TRANSFORMS.register_module()
class PrepareSubtaskPrompts:
    version = 'pi05-subtask-v1'

    def __init__(self, tokenizer, max_len=200, state_dim=32, model_vocab_size=None):
        self.tokenizer = build_tokenizer_from_cfg(tokenizer) if isinstance(tokenizer, dict) else tokenizer
        self.max_len = max_len
        self.state_dim = state_dim
        self.model_vocab_size = model_vocab_size
        self.pad_id = self.tokenizer.pad_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        if any(x is None for x in (self.pad_id, self.bos_id, self.eos_id)):
            raise ValueError('BOS, EOS and PAD IDs must be defined')

    @staticmethod
    def clean(text):
        return str(text).strip().replace('_', ' ').replace('\n', ' ')

    def condition(self, task, states, high_level=False):
        states = np.asarray(states)
        if states.ndim != 1 or len(states) > self.state_dim or not np.isfinite(states).all():
            raise ValueError('Expected finite, already-normalized 1D state within state_dim')
        padded = np.zeros(self.state_dim)
        padded[:len(states)] = states
        bins = np.digitize(padded, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        state_text = ' '.join(map(str, bins))
        ending = 'Subtask: ' if high_level else 'Action: '
        return f'Task: {self.clean(task)}, State: {state_text};\n{ending}'

    def _encode(self, text):
        ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        if self.model_vocab_size is not None and any(token < 0 or token >= self.model_vocab_size for token in ids):
            raise ValueError('Text contains a token outside the model vocabulary')
        return ids

    def pack(self, condition, target=None):
        # Explicit segment encoding is the v1 contract: one BOS and one target EOS.
        prefix = [self.bos_id] + self._encode(condition)
        suffix = [] if target is None else self._encode(self.clean(target)) + [self.eos_id]
        if target is not None and not self.clean(target):
            raise ValueError('Empty subtask target')
        length = len(prefix) + len(suffix)
        if length > self.max_len:
            raise ValueError(f'Prompt needs {length} tokens, max_len={self.max_len}; no truncation')
        pad = self.max_len - length
        return dict(
            input_ids=np.array(prefix + suffix + [self.pad_id] * pad, dtype=np.int64),
            valid_mask=np.array([True] * length + [False] * pad),
            ar_mask=np.array([False] * len(prefix) + [True] * len(suffix) + [False] * pad),
            labels=np.array([-100] * len(prefix) + suffix + [-100] * pad, dtype=np.int64),
        )

    def high_level(self, overall_task, states, subtask=None):
        return self.pack(self.condition(overall_task, states, high_level=True), subtask)

    def low_level(self, subtask, states):
        return self.pack(self.condition(subtask, states))

    def decode(self, ids):
        values = []
        for token in ids:
            token = int(token)
            if token == self.eos_id:
                break
            if token not in (self.pad_id, self.bos_id):
                values.append(token)
        inner = getattr(self.tokenizer, 'tokenizer', self.tokenizer)
        return inner.decode(values).strip()

    def __call__(self, inputs):
        inputs = dict(inputs)
        if not self.clean(inputs['subtask']):
            raise ValueError('A real subtask target is required; no implicit identity fallback')
        high = self.high_level(inputs['overall_task'], inputs['states'], inputs['subtask'])
        low = self.low_level(inputs['subtask'], inputs['states'])
        inputs.update({f'hl_{key}': value for key, value in high.items()})
        inputs.update(ll_input_ids=low['input_ids'], ll_valid_mask=low['valid_mask'])
        inputs['has_hl'] = np.array(True)
        inputs['has_action'] = np.array(True)
        return inputs
