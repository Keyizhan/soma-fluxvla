"""Legacy normalized-tensor architecture replay.

Production and physical-unit offline evaluation use fluxvla.inference.SubtaskPolicy.
This research harness does not implement the deployment I/O contract."""
import time

import numpy as np
import torch


class SubtaskRuntime:
    def __init__(self, model, prompts, max_new_tokens=32, num_steps=10):
        self.model = model
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.num_steps = num_steps
        self.version = 0

    def _decode_subtask(self, row):
        decoder = getattr(self.prompts, 'decode_subtask', None) or self.prompts.decode
        return decoder(row)

    def _low_level(self, subtask, state, overall_task):
        try:
            return self.prompts.low_level(subtask, state, overall_task)
        except TypeError:
            return self.prompts.low_level(subtask, state)

    @torch.no_grad()
    def infer(self, images, states, overall_tasks, img_masks, observation_version, noise=None):
        if self.model.training:
            raise ValueError('Call eval() before inference')
        if len(overall_tasks) != states.shape[0]:
            raise ValueError('One overall task per sample is required')
        self.version += 1
        request_version = self.version
        state_rows = states.detach().cpu().numpy()

        def stack(rows, key):
            return torch.from_numpy(np.stack([row[key] for row in rows])).to(states.device)

        def sync():
            if states.is_cuda:
                torch.cuda.synchronize(states.device)

        high = [self.prompts.high_level(g, s, None)
                for g, s in zip(overall_tasks, state_rows)]
        sync()
        start = time.perf_counter()
        source = getattr(self.model, 'low_level_prefix_source', 'text')
        generated = self.model.generate_subtask(
            images, stack(high, 'input_ids'), stack(high, 'valid_mask'), img_masks,
            eos_token_id=self.prompts.eos_id, pad_token_id=self.prompts.pad_id,
            max_new_tokens=self.max_new_tokens,
            ar_mask=torch.as_tensor(np.stack([row['ar_mask'] for row in high])).to(states.device),
            return_cache=(source == 'generation'))
        sync()
        text_end = time.perf_counter()
        texts = [self._decode_subtask(row) for row in generated['token_ids'].cpu().tolist()]
        if any(not text for text in texts):
            raise ValueError('Subtask is empty or lacks EOS; no action was sampled')
        source = getattr(self.model, 'low_level_prefix_source', 'text')
        if source == 'generation':
            # Ke-Wang: the action expert reads the KV produced while generating the subtask,
            # so the overall task and the generated tokens are part of the action context.
            if generated.get('cache') is None:
                raise ValueError('generation prefix source requires generate_subtask(return_cache=True)')
            low = high
            actions = self.model.sample_actions_given_subtask(
                images, states, stack(high, 'input_ids'), stack(high, 'valid_mask'),
                img_masks, noise=noise, num_steps=self.num_steps,
                prefix_cache=generated['cache'])
        else:
            # Deviation switch: rebuild the low-level prompt from the decoded text only,
            # which is what training does (annotated text, re-encoded).
            low = [self._low_level(s, state, task)
                   for s, state, task in zip(texts, state_rows, overall_tasks)]
            actions = self.model.sample_actions_given_subtask(
                images, states, stack(low, 'input_ids'), stack(low, 'valid_mask'),
                img_masks, noise=noise, num_steps=self.num_steps)
        sync()
        end = time.perf_counter()
        return dict(actions=actions, subtasks=texts, generation=generated,
                    ll_input_ids=stack(low, 'input_ids'),
                    trace=dict(observation_version=observation_version,
                               subtask_version=request_version, action_version=request_version,
                               generation_calls=1, text_seconds=text_end-start,
                               action_seconds=end-text_end, total_seconds=end-start,
                               template_version=getattr(self.prompts, 'variant',
                                                        getattr(self.prompts, 'version', 'unknown')),
                               low_level_prefix_source=source,
                               finished=bool(generated['finished'].all())))
