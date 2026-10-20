# Derived from FluxVLA (Limx Dynamics); see LICENSE and NOTICE.
"""
This module defines an abstract base class `PromptBuilder` and
a concrete implementation `PurePrompter` for building multi-turn
prompts for chat-based language models (LLMs). The goal is to
ensure consistent formatting across turns and models, with special
handling for different roles (e.g., human vs. LLM).
"""
from typing import Dict
import numpy as np
from fluxvla.engines import TRANSFORMS

@TRANSFORMS.register_module()
class PreparePromptWithState:
    """Prepare prompt with state for PI05.
    This transform prepares the prompt with state for PI05.
    It formats the prompt to include the task description and state.

    Args:
        max_state_dim (int): The maximum dimension of the state.
        task_key (str): The key of the task description in the input data.
        lowercase_task_description (bool): Whether to lowercase task text.
        add_action_prefix (bool): Whether to append ``Action:`` to the prompt.
    """

    def __init__(self, max_state_dim: int=32, task_key: str='task_description', lowercase_task_description: bool=False, add_action_prefix: bool=True, *args, **kwargs):
        self.max_state_dim = max_state_dim
        self.task_key = task_key
        self.lowercase_task_description = lowercase_task_description
        self.add_action_prefix = add_action_prefix

    def __call__(self, inputs: Dict) -> Dict:
        state = inputs['states']
        if state is None:
            raise ValueError('State is required for PI05')
        assert self.task_key in inputs, f"Data must contain '{self.task_key}' key."
        task_description = inputs[self.task_key]
        state_padded = np.zeros(self.max_state_dim)
        state_padded[:state.shape[0]] = state
        discretized_states = np.digitize(state_padded, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        cleaned_text = task_description.strip().replace('_', ' ').replace('\n', ' ')
        if self.lowercase_task_description:
            cleaned_text = cleaned_text.lower()
        state_str = ' '.join(map(str, discretized_states))
        suffix = ';\nAction: ' if self.add_action_prefix else ';\n'
        full_prompt = f'Task: {cleaned_text}, State: {state_str}{suffix}'
        inputs['prompt'] = full_prompt
        return inputs
