# Derived from FluxVLA (Limx Dynamics); see LICENSE and NOTICE.
import os
from typing import Dict
import numpy as np
from fluxvla.engines import TRANSFORMS

@TRANSFORMS.register_module()
class ProcessPrompts:
    """Process and tokenize prompts for language models.
    This class handles the tokenization of prompts using a specified
    tokenizer, ensuring that the tokenized output adheres to a maximum
    length. It can also optionally prepare labels for language modeling
    tasks by creating a copy of the tokenized input.

    Args:
        tokenizer (Dict): Configuration for building the tokenizer.
        max_len (int, optional): Maximum length for tokenized prompts.
            Defaults to 180.
        with_labels (bool, optional): Whether to include labels in
            the output. Defaults to False.
    """

    def __init__(self, tokenizer: Dict, model_path: str=None, max_len: int=180, with_labels: bool=False, with_state: bool=False, ignore_index: int=-100, negative_prompt=None):
        from fluxvla.engines import build_tokenizer_from_cfg
        if model_path is not None:
            tokenizer['model_path'] = os.path.join(model_path, 'tokenizer')
        self.tokenizer = build_tokenizer_from_cfg(tokenizer)
        self.max_len = max_len
        self.with_labels = with_labels
        self.with_state = with_state
        self.ignore_index = ignore_index
        self.negative_prompt = negative_prompt

    def _tokenize_single_prompt(self, prompt: str, state: np.ndarray | None=None):
        if state is not None:
            tokens = self.tokenizer(prompt, state=state, add_special_tokens=True)['input_ids']
        else:
            tokens = self.tokenizer(prompt, add_special_tokens=True)['input_ids']
        token_mask = [True] * len(tokens)
        tokens_len = len(tokens)
        if self.max_len is not None:
            if tokens_len < self.max_len:
                padding = [False] * (self.max_len - tokens_len)
                tokens = tokens + padding
                token_mask = token_mask + padding
            else:
                tokens = tokens[:self.max_len]
                token_mask = token_mask[:self.max_len]
        return (tokens, token_mask)

    def __call__(self, inputs):
        """Tokenize and process the prompt in the input data.
        The method tokenizes the 'prompt' field in the input dictionary,
        applies padding or truncation to meet the maximum length requirement,
        and optionally creates a 'labels' field for language modeling tasks.

        Args:
            inputs (Dict): Input data containing a 'prompt' key.
        """
        assert 'prompt' in inputs, "Data must contain 'prompt' key."
        if self.with_state:
            assert 'state' in inputs, "Data must contain 'state' key."
            state = inputs['state']
        else:
            state = None
        tokens, token_mask = self._tokenize_single_prompt(inputs['prompt'], state)
        lang_tokens = [tokens]
        lang_masks = [token_mask]
        if self.negative_prompt is not None:
            negative_tokens, negative_token_mask = self._tokenize_single_prompt(self.negative_prompt, state)
            lang_tokens.append(negative_tokens)
            lang_masks.append(negative_token_mask)
        labels = list(tokens)
        inputs['lang_tokens'] = np.array(tokens)
        inputs['lang_masks'] = np.array(token_mask)
        if self.with_labels:
            assert 'actions' in inputs, "Data must contain 'actions' key."
            actions = inputs['actions']
            inputs['labels'] = np.array(labels)
            inputs['labels'][:-(len(actions[0]) + 1)] = self.ignore_index
        return inputs
