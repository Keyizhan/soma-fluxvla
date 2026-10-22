"""Shared-weight, separate-context pi0.5 subtask extension (architecture v1).

Sources and departures are documented in docs/research/pi05_subtask.md.
No new LM head parameters: logits use the verified input embedding transpose.
"""
import warnings
import math

from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fluxvla.engines import VLAS
from fluxvla.engines.utils.model_utils import make_att_2d_masks
from .pi05_flowmatching import PI05FlowMatching
from .subtask_kv import run_kv_layers


def global_weighted_mean(total, count):
    """Global valid-element mean with DDP-averaged gradients and global display value."""
    if dist.is_available() and dist.is_initialized():
        counts = count.detach().clone()
        displayed = total.detach().clone()
        dist.all_reduce(counts)
        dist.all_reduce(displayed)
        denominator = counts.clamp_min(1e-8)
        gradient_loss = total * dist.get_world_size() / denominator
        return gradient_loss + (displayed / denominator - gradient_loss.detach())
    return total / count.clamp_min(1e-8)


@VLAS.register_module()
class PI05Subtask(PI05FlowMatching):
    def __init__(self, hl_loss_weight=1.0, action_loss_weight=1.0, fast_loss_weight=1.0,
                 text_loss_mode='separate', token_loss_weight=0.0,
                 fast_vocab_size=None, fast_skip_tokens=128, exact_beta_time=True,
                 rope_precision='jax', attn_math='fp32', warn_on_bf16_cache=True,
                 low_level_prefix_source='generation', low_level_task_independent=False,
                 stop_gradient_flow_to_prefix=False, text_use_cache=False,
                 action_prefix_cache=False, flow_prefix_cache=False,
                 text_decode_fp32=False, low_level_text_attention='bidirectional', loss_reduction='global', **kwargs):
        super().__init__(**kwargs)
        if loss_reduction not in ("local", "global"):
            raise ValueError("unsupported loss reduction")
        self.loss_reduction = loss_reduction
        if text_loss_mode not in ('separate', 'unified_token'):
            raise ValueError('unsupported text_loss_mode')
        for name, value in (('hl_loss_weight', hl_loss_weight),
                            ('action_loss_weight', action_loss_weight),
                            ('fast_loss_weight', fast_loss_weight),
                            ('token_loss_weight', token_loss_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if text_loss_mode == 'unified_token' and (hl_loss_weight != 0 or fast_loss_weight != 0):
            raise ValueError('unified_token requires zero separate hl/fast loss weights')
        if text_loss_mode == 'separate' and token_loss_weight != 0:
            raise ValueError('separate text loss cannot use token_loss_weight')
        self.text_loss_mode = text_loss_mode
        self.token_loss_weight = token_loss_weight
        # hl = subtask CE, fast = FAST action-token CE, action = flow-matching MSE.
        self.hl_loss_weight = hl_loss_weight
        self.fast_loss_weight = fast_loss_weight
        self.action_loss_weight = action_loss_weight
        self.fast_vocab_size = fast_vocab_size
        self.fast_skip_tokens = fast_skip_tokens
        if fast_vocab_size is not None:
            vocab = self.llm_backbone.config.vocab_size
            if vocab - 1 - fast_skip_tokens - fast_vocab_size + 1 < 0:
                raise ValueError('FAST vocabulary does not fit below the model vocabulary')
        # Old bundles may contain exact_beta_time=True; False must never restore the bug.
        if exact_beta_time is not True:
            raise ValueError("exact_beta_time=False is unsupported; time_sampler must be 'beta'")
        if rope_precision not in ('jax', 'hf'):
            raise ValueError("rope_precision must be 'jax' or 'hf'")
        # 'jax' = openpi/big_vision (f32 rotate, downcast result); 'hf' = legacy behaviour.
        if attn_math not in ('fp32', 'bf16'):
            raise ValueError("attn_math must be 'fp32' or 'bf16'")
        # Attention logits in float32 (the openpi policy). One setting is honoured by the
        # training CE path, the cached decode and the joint action path, so all three agree.
        if low_level_prefix_source not in ('generation', 'text'):
            raise ValueError("low_level_prefix_source must be 'generation' or 'text'")
        # 'generation' = Ke-Wang: reuse the KV produced while generating the subtask.
        # 'text'       = rebuild the low-level prefix from the decoded text (train-consistent).
        if low_level_task_independent and low_level_prefix_source != 'text':
            raise ValueError('task-independent low-level conditioning requires rebuilt text prefix')
        self.low_level_task_independent = bool(low_level_task_independent)
        self.low_level_prefix_source = low_level_prefix_source
        self.attn_math = attn_math
        self.rope_precision = rope_precision
        self.warn_on_bf16_cache = warn_on_bf16_cache
        self.exact_beta_time = exact_beta_time
        self.stop_gradient_flow_to_prefix = stop_gradient_flow_to_prefix
        self.text_use_cache = text_use_cache
        self.action_prefix_cache = action_prefix_cache
        self.flow_prefix_cache = flow_prefix_cache
        self.text_decode_fp32 = text_decode_fp32
        if low_level_text_attention not in ('causal', 'bidirectional'):
            raise ValueError('unsupported low_level_text_attention')
        self.low_level_text_attention = low_level_text_attention

    def _loss_mean(self, total, count):
        if self.loss_reduction == 'local':
            return total / (count + 1e-8)
        return global_weighted_mean(total, count)

    def embed_prefix(self, images, lang_tokens, img_masks, lang_masks, *args, **kwargs):
        prefix, valid, ar = super().embed_prefix(
            images, lang_tokens, img_masks, lang_masks, *args, **kwargs)
        if self.low_level_text_attention == 'causal':
            ar = ar.clone()
            ar[:, -lang_tokens.shape[1]:] = True
        return prefix, valid, ar

    def _assert_no_fast_tokens(self, token_ids, where):
        """The low-level prefix must never carry FAST action tokens (leak guard)."""
        if self.fast_vocab_size is None:
            return
        vocab = self.llm_backbone.config.vocab_size
        top = vocab - 1 - self.fast_skip_tokens
        bottom = top - self.fast_vocab_size + 1
        if torch.any((token_ids >= bottom) & (token_ids <= top)):
            raise ValueError(
                f'{where} must not contain FAST action tokens '
                f'(relocated range [{bottom},{top}])')

    def _prefill_low(self, images, ids, valid_mask, img_masks):
        prefix, valid, ar = self.embed_prefix(images, ids, img_masks, valid_mask)
        _, cache = run_kv_layers(
            self, self.llm_backbone, prefix, (valid.long().cumsum(-1)-1).clamp_min(0),
            make_att_2d_masks(valid, ar), valid=valid)
        return cache

    def _cached_action_velocity(self, states, x, time, cache):
        suffix, valid_suffix, ar_suffix, cond = self.embed_suffix(states, x, time)
        prefix_valid = cache.valid[:, :cache.length]
        allowed = torch.cat([
            valid_suffix[:, :, None] & prefix_valid[:, None, :],
            make_att_2d_masks(valid_suffix, ar_suffix)], dim=-1)
        positions = prefix_valid.long().sum(-1, keepdim=True) + valid_suffix.long().cumsum(-1)-1
        hidden, _ = run_kv_layers(self, self.llm_expert, suffix, positions,
                                  allowed, cond=cond, cache=cache)
        return self.action_out_proj(hidden[:, -self.n_action_steps:].float())

    def project_text(self, hidden):
        weight = self.llm_backbone.embed_tokens.weight
        return F.linear(hidden.to(weight.dtype), weight)

    def high_level_hidden(self, images, input_ids, valid_mask, ar_mask, img_masks):
        prefix, valid, _ = self.embed_prefix(images, input_ids, img_masks, valid_mask)
        n_images = prefix.shape[1] - input_ids.shape[1]
        if ar_mask.shape != input_ids.shape or valid_mask.shape != input_ids.shape:
            raise ValueError('Text masks must match token IDs')
        ar = torch.cat([torch.zeros_like(valid[:, :n_images]), ar_mask.bool()], dim=1)
        allowed = make_att_2d_masks(valid, ar)
        # Single implementation shared with the cached decode. This is only safe together
        # with attn_math='fp32': with bf16 attention the single-query cached step disagreed
        # with the full recompute by up to ~32 logits (see parity_20261017/ROPE_PARITY.md).
        hidden, _ = run_kv_layers(
            self, self.llm_backbone, prefix,
            (valid.long().cumsum(-1) - 1).clamp_min(0), allowed, valid=valid)
        return hidden, valid, n_images

    def forward_high_level(self, images, hl_input_ids, hl_valid_mask, hl_ar_mask,
                           hl_labels, img_masks, has_hl=None, sample_weight=None,
                           return_logits=False, hl_subtask_region_mask=None,
                           hl_action_region_mask=None):
        hidden, _, offset = self.high_level_hidden(
            images, hl_input_ids, hl_valid_mask, hl_ar_mask, img_masks)
        if hl_labels.shape != hl_input_ids.shape:
            raise ValueError('Labels must be aligned to text positions')
        supervised = hl_labels != -100
        if torch.any(supervised & (~hl_valid_mask.bool() | ~hl_ar_mask.bool())):
            raise ValueError('Targets must be valid causal text tokens')
        if supervised[:, 0].any():
            raise ValueError('First text token must be conditioning context')
        if (hl_subtask_region_mask is None) != (hl_action_region_mask is None):
            raise ValueError('Provide both region masks or neither')
        if hl_subtask_region_mask is not None:
            for name, mask in (('subtask', hl_subtask_region_mask),
                               ('action', hl_action_region_mask)):
                if mask.shape != hl_input_ids.shape:
                    raise ValueError(f'{name} region mask must align to token positions')
                if torch.any(mask.bool() & (~hl_valid_mask.bool() | ~hl_ar_mask.bool())):
                    raise ValueError(f'{name} region must be valid causal targets')
            if torch.any(hl_subtask_region_mask.bool() & hl_action_region_mask.bool()):
                raise ValueError('Subtask and action regions must be disjoint')
            if self.text_loss_mode == 'unified_token' and torch.any(
                    (hl_subtask_region_mask.bool() | hl_action_region_mask.bool()) != supervised):
                raise ValueError('unified_token regions must cover exactly all supervised targets')
        targets = hl_labels[:, 1:]
        target_mask = targets != -100
        if has_hl is not None:
            target_mask = target_mask & has_hl.bool()[:, None]
        pred_hidden = hidden[:, offset:offset + hl_input_ids.shape[1] - 1]
        # Project only selected prediction positions, never image/PAD positions.
        logits_selected = self.project_text(pred_hidden[target_mask])
        nll = F.cross_entropy(logits_selected.float(), targets[target_mask], reduction='none')
        weights = torch.ones_like(nll)
        if sample_weight is not None:
            weights = sample_weight[:, None].expand_as(targets)[target_mask].to(nll)
        # Region masks describe input positions; the CE target at position i is token i+1.
        if hl_subtask_region_mask is None:
            subtask_nll, subtask_weights = nll, weights
            action_nll = action_weights = None
        else:
            flat_subtask = hl_subtask_region_mask.bool()[:, 1:][target_mask]
            flat_action = hl_action_region_mask.bool()[:, 1:][target_mask]
            subtask_nll, subtask_weights = nll[flat_subtask], weights[flat_subtask]
            action_nll, action_weights = nll[flat_action], weights[flat_action]
        # Keep a zero-supervision graph without all-ignored mean-CE NaNs.
        subtask_sum = (subtask_nll * subtask_weights).sum()
        subtask_count = subtask_weights.sum()
        fast_sum = nll.sum() * 0 if action_nll is None else (action_nll * action_weights).sum()
        fast_count = subtask_count.new_zeros(()) if action_weights is None else action_weights.sum()
        token_count = subtask_count + fast_count
        loss_token = self._loss_mean(subtask_sum + fast_sum, token_count)
        loss_subtask = self._loss_mean(subtask_sum, subtask_count)
        output = dict(loss_subtask=loss_subtask, loss=loss_subtask,
                      loss_token=loss_token, token_count=token_count.detach(),
                      subtask_nll_sum=subtask_sum, fast_nll_sum=fast_sum,
                      token_subtask_contribution=self._loss_mean(subtask_sum, token_count),
                      token_fast_contribution=self._loss_mean(fast_sum, token_count),
                      target_count=subtask_count.detach(),
                      loss_action_ce=None,
                      action_target_count=torch.zeros((), device=loss_subtask.device))
        if return_logits:
            output['logits'] = self.project_text(hidden[:, offset:]).float()
        if action_nll is not None:
            loss_action_ce = self._loss_mean(fast_sum, fast_count)
            output.update(loss_action_ce=loss_action_ce, loss=loss_subtask + loss_action_ce,
                          action_target_count=fast_count.detach())
        if self.text_loss_mode == 'unified_token':
            output['loss'] = loss_token
        return output

    def forward_low_level(self, images, states, ll_input_ids, ll_valid_mask, actions,
                          img_masks, action_masks=None, has_action=None,
                          sample_weight=None, noise=None, time=None, use_prefix_cache=None, rtc_delays=None):
        self._assert_no_fast_tokens(ll_input_ids, 'Low-level prefix')
        noise = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        time = self.sample_time(actions.shape[0], actions.device) if time is None else time
        mask = torch.ones_like(actions) if action_masks is None else action_masks
        while mask.ndim < actions.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(actions).to(actions)
        if has_action is not None:
            mask = mask * has_action[:, None, None].to(mask)
        t = time
        if self.rtc_training_config and self.rtc_training_config.get('enabled', False):
            from fluxvla.engines.utils.rtc_training import apply_rtc_time_conditioning, sample_training_delay
            if rtc_delays is None:
                rtc_delays = sample_training_delay(
                    actions.shape[0], self.rtc_training_config.get('max_delay', 5),
                    self.rtc_training_config.get('distribution', 'exponential'),
                    self.rtc_training_config.get('temperature', 1.), actions.device)
            t, mask = apply_rtc_time_conditioning(time, mask, rtc_delays, actions.shape[1], clean_time=0.)
        mixing_time = t[..., None] if t.ndim == 2 else t[:, None, None]
        x = mixing_time * noise + (1-mixing_time) * actions
        cached = self.flow_prefix_cache if use_prefix_cache is None else use_prefix_cache
        if cached or self.stop_gradient_flow_to_prefix:
            cache = self._prefill_low(images, ll_input_ids, ll_valid_mask, img_masks)
            if self.stop_gradient_flow_to_prefix:
                cache = cache.detached()
            prediction = self._cached_action_velocity(states, x, t, cache)
        else:
            prefix, valid_prefix, ar_prefix = self.embed_prefix(images, ll_input_ids, img_masks, ll_valid_mask)
            suffix, valid_suffix, ar_suffix, cond = self.embed_suffix(states, x, t)
            valid_tokens = torch.cat([valid_prefix, valid_suffix], dim=1)
            ar = torch.cat([ar_prefix, ar_suffix], dim=1)
            hidden, _ = self.forward_model(
                [prefix, suffix], self._prepare_attention_masks_4d(make_att_2d_masks(valid_tokens, ar)),
                (valid_tokens.long().cumsum(-1)-1).clamp_min(0),
                use_cache=False, adarms_cond=[None, cond])
            prediction = self.action_out_proj(hidden[:, -self.n_action_steps:].float())
        if self.ori_action_dim is not None:
            prediction = prediction[..., :self.ori_action_dim]
        result = dict(predictions=prediction)
        target = (noise - actions)[..., :result['predictions'].shape[-1]]
        valid = mask[..., :target.shape[-1]]
        if sample_weight is not None:
            valid = valid * sample_weight[:, None, None].to(valid)
        result['action_squared_error_sum'] = ((result['predictions'] - target).square() * valid).sum()
        result['loss'] = self._loss_mean(result['action_squared_error_sum'], valid.sum())
        result['valid_action_count'] = valid.sum().detach()
        return result

    def forward(self, images, states, hl_input_ids, hl_valid_mask, hl_ar_mask, hl_labels,
                ll_input_ids, ll_valid_mask, actions, img_masks, action_masks=None,
                has_hl=None, has_action=None, sample_weight=None, noise=None, time=None,
                hl_subtask_region_mask=None, hl_action_region_mask=None, **kwargs):
        high = self.forward_high_level(
            images, hl_input_ids, hl_valid_mask, hl_ar_mask, hl_labels, img_masks,
            has_hl=has_hl, sample_weight=sample_weight,
            hl_subtask_region_mask=hl_subtask_region_mask,
            hl_action_region_mask=hl_action_region_mask)
        low = self.forward_low_level(
            images, states, ll_input_ids, ll_valid_mask, actions, img_masks,
            action_masks=action_masks, has_action=has_action, sample_weight=sample_weight,
            noise=noise, time=time)
        loss_fast = high['loss_action_ce']
        if self.text_loss_mode == 'unified_token':
            loss = self.token_loss_weight * high['loss_token'] + self.action_loss_weight * low['loss']
            subtask_metric = high['loss_subtask']
            fast_metric = loss_fast
        else:
            loss = self.hl_loss_weight * high['loss_subtask'] + self.action_loss_weight * low['loss']
            if loss_fast is not None:
                loss = loss + self.fast_loss_weight * loss_fast
            subtask_metric = self.hl_loss_weight * high['loss_subtask']
            fast_metric = None if loss_fast is None else self.fast_loss_weight * loss_fast
        return dict(loss=loss,
                    loss_subtask=subtask_metric.detach(), loss_hl=subtask_metric.detach(),
                    loss_fast=None if fast_metric is None else fast_metric.detach(),
                    loss_token=high['loss_token'],
                    subtask_ce=high['loss_subtask'].detach(),
                    fast_ce=None if loss_fast is None else loss_fast.detach(),
                    token_count=high['token_count'],
                    subtask_nll_sum=high['subtask_nll_sum'], fast_nll_sum=high['fast_nll_sum'],
                    token_subtask_contribution=high['token_subtask_contribution'],
                    token_fast_contribution=high['token_fast_contribution'],
                    loss_action=(self.action_loss_weight * low['loss']).detach(),
                    target_count=high['target_count'],
                    action_target_count=high['action_target_count'],
                    predictions=low['predictions'])

    @torch.no_grad()
    def generate_subtask(self, images, hl_input_ids, hl_valid_mask, img_masks,
                         eos_token_id, pad_token_id, max_new_tokens=32,
                         use_cache=None, return_logits=False, profile=False,
                         ar_mask=None, return_cache=False, stop_token_ids=None):
        # Stable architecture comparison mode. This also keeps vision prefill
        # in FP32; it is explicit and never described as BF16 text decoding.
        context = torch.autocast(images.device.type, enabled=False) if self.text_decode_fp32 else nullcontext()
        with context:
            return self._generate_subtask_impl(
                images, hl_input_ids, hl_valid_mask, img_masks, eos_token_id, pad_token_id,
                max_new_tokens, use_cache, return_logits, profile, ar_mask, return_cache,
                stop_token_ids)

    def _generate_subtask_impl(self, images, hl_input_ids, hl_valid_mask, img_masks,
                              eos_token_id, pad_token_id, max_new_tokens,
                              use_cache, return_logits, profile, ar_mask=None,
                              return_cache=False, stop_token_ids=None):
        if self.training:
            raise ValueError('Call eval() before generation')
        if max_new_tokens < 1:
            raise ValueError('max_new_tokens must be positive')
        ids, valid = hl_input_ids.clone(), hl_valid_mask.bool().clone()
        if not valid.any(-1).all():
            raise ValueError('Each sample requires nonempty context')
        if ar_mask is None:
            ar = torch.zeros_like(valid)
        else:
            if ar_mask.shape != hl_input_ids.shape:
                raise ValueError('ar_mask must match the prompt shape')
            ar = ar_mask.bool().clone() & valid
        generated = torch.full((ids.shape[0], max_new_tokens), pad_token_id,
                               dtype=ids.dtype, device=ids.device)
        lengths = torch.zeros(ids.shape[0], dtype=torch.long, device=ids.device)
        finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        cached = self.text_use_cache if use_cache is None else use_cache
        if (cached and self.attn_math != 'fp32' and self.warn_on_bf16_cache
                and not self.text_decode_fp32):
            warnings.warn(
                'Cached subtask decoding in bf16 is not numerically faithful to the '
                'recompute path: up to ~32 logits difference and 32% greedy flips on an '
                'under-trained model (~1.3 logits / 0 flips once converged). '
                'Use use_cache=False (the evaluation default) or text_decode_fp32=True.',
                RuntimeWarning, stacklevel=2)
        if return_cache and not cached:
            raise ValueError('return_cache requires cached text generation')
        stop_tokens = tuple(int(token) for token in (stop_token_ids or ()))
        stopped_at_boundary = torch.zeros_like(finished)
        cache = None
        logit_rows = []
        timings = []
        for step in range(max_new_tokens):
            if profile:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            if cached and step == 0:
                prefix, full_valid, _ = self.embed_prefix(images, ids, img_masks, valid)
                n_images = prefix.shape[1] - ar.shape[1]
                if n_images < 0:
                    raise ValueError('Prompt longer than the embedded prefix')
                ar_full = torch.cat([torch.zeros(ar.shape[0], n_images, dtype=torch.bool,
                                                 device=ar.device), ar], dim=1)
                hidden, cache = run_kv_layers(
                    self, self.llm_backbone, prefix,
                    (full_valid.long().cumsum(-1)-1).clamp_min(0),
                    make_att_2d_masks(full_valid, ar_full),
                    valid=full_valid, capacity=prefix.shape[1]+max_new_tokens-1+int(return_cache),
                    text_precision=True)
            elif cached:
                # Only the previous generated token is new. Cache never crosses requests.
                token_valid = valid[:, -1:]
                combined_valid = torch.cat([cache.valid[:, :cache.length], token_valid], dim=-1)
                allowed = token_valid[:, :, None] & combined_valid[:, None, :]
                token = self.llm_backbone.embed_tokens(ids[:, -1:]) * (self.llm_backbone.config.hidden_size**.5)
                hidden, cache = run_kv_layers(
                    self, self.llm_backbone, token,
                    (combined_valid.long().sum(-1, keepdim=True)-1).clamp_min(0),
                    allowed, cache=cache, append=True, valid=token_valid, text_precision=True)
            else:
                hidden, full_valid, _ = self.high_level_hidden(images, ids, valid, ar, img_masks)
            if cached and step > 0:
                last_hidden = hidden[:, -1]
            else:
                positions = torch.arange(full_valid.shape[1], device=ids.device)[None]
                last_index = torch.where(full_valid, positions, -1).amax(-1)
                last_hidden = hidden[torch.arange(ids.shape[0], device=ids.device), last_index]
            logits = self.project_text(last_hidden)
            if return_logits:
                logit_rows.append(logits.float())
            next_ids = logits.argmax(-1)
            if profile:
                end_event.record()
                timings.append((start_event, end_event))
            active = ~finished
            generated[:, step] = torch.where(active, next_ids, pad_token_id)
            lengths += active.long()
            finished |= active & (next_ids == eos_token_id)
            if stop_tokens and step + 1 >= len(stop_tokens):
                stop = torch.as_tensor(stop_tokens, device=ids.device, dtype=ids.dtype)
                at_boundary = active & (generated[:, step+1-len(stop_tokens):step+1] == stop).all(-1)
                stopped_at_boundary |= at_boundary
                finished |= at_boundary
            if finished.all():
                break
            ids = torch.cat([ids, torch.where(active, next_ids, pad_token_id)[:, None]], dim=1)
            valid = torch.cat([valid, active[:, None]], dim=1)
            ar = torch.cat([ar, active[:, None]], dim=1)
        if return_cache:
            # Match Wangke's decode loop: every emitted token, including the
            # final boundary/EOS token, must already be represented in KV.
            # The normal loop consumes the previous token before predicting;
            # flush the last emitted token once, without re-encoding a prompt.
            token_valid = active[:, None]
            combined_valid = torch.cat([cache.valid[:, :cache.length], token_valid], dim=-1)
            allowed = token_valid[:, :, None] & combined_valid[:, None, :]
            token = self.llm_backbone.embed_tokens(generated[:, step:step+1]) * (self.llm_backbone.config.hidden_size**.5)
            _, cache = run_kv_layers(
                self, self.llm_backbone, token,
                (combined_valid.long().sum(-1, keepdim=True)-1).clamp_min(0),
                allowed, cache=cache, append=True, valid=token_valid, text_precision=True)
        result = dict(token_ids=generated, lengths=lengths, finished=finished,
                      stopped_at_boundary=stopped_at_boundary,
                      cache_info=cache.info() if cached else None)
        result['cache'] = cache if (return_cache and cached) else None
        if return_logits:
            result['logits'] = torch.stack(logit_rows, dim=1)
        if profile:
            torch.cuda.synchronize(images.device)
            result['step_ms'] = [start.elapsed_time(end) for start, end in timings]
        return result

    @torch.no_grad()
    def sample_actions_given_subtask(self, images, states, ll_input_ids, ll_valid_mask,
                                     img_masks, noise=None, num_steps=10,
                                     use_cache=None, return_trace=False, profile=False,
                                     prefix_cache=None):
        if self.training or num_steps < 1:
            raise ValueError('Sampling requires eval() and positive num_steps')
        self._assert_no_fast_tokens(ll_input_ids, 'Low-level prefix')
        batch = states.shape[0]
        x = self.sample_noise((batch, self.n_action_steps, self.max_action_dim), states.device) if noise is None else noise.clone()
        cached = self.action_prefix_cache if use_cache is None else use_cache
        if prefix_cache is not None:
            if self.low_level_task_independent:
                raise ValueError('task-independent low-level conditioning rejects external prefix_cache')
            # Ke-Wang path: the expert reads the KV produced during subtask generation,
            # which already contains the overall task and the generated subtask tokens.
            cached = True
        if profile:
            start_event = torch.cuda.Event(enable_timing=True)
            prefix_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        if prefix_cache is not None:
            cache = prefix_cache
        elif cached:
            cache = self._prefill_low(images, ll_input_ids, ll_valid_mask, img_masks)
        else:
            prefix, valid_prefix, ar_prefix = self.embed_prefix(images, ll_input_ids, img_masks, ll_valid_mask)
        if profile:
            prefix_event.record()
        cache_lengths = []
        for step in range(num_steps):
            time = torch.full((batch,), 1 - step / num_steps, device=x.device, dtype=x.dtype)
            if cached:
                velocity = self._cached_action_velocity(states, x, time, cache)
                cache_lengths.append(cache.length)
            else:
                suffix, valid_suffix, ar_suffix, cond = self.embed_suffix(states, x, time)
                valid = torch.cat([valid_prefix, valid_suffix], dim=1)
                ar = torch.cat([ar_prefix, ar_suffix], dim=1)
                suffix_out, _ = self.forward_model(
                    [prefix, suffix], self._prepare_attention_masks_4d(make_att_2d_masks(valid, ar)),
                    (valid.long().cumsum(-1) - 1).clamp_min(0),
                    use_cache=False, adarms_cond=[None, cond])
                velocity = self.action_out_proj(suffix_out[:, -self.n_action_steps:].float())
            x = x - velocity / num_steps
        actions = x[..., :self.ori_action_dim] if self.ori_action_dim is not None else x
        if profile:
            end_event.record()
            torch.cuda.synchronize(images.device)
        if return_trace:
            result = dict(actions=actions, cache_info=cache.info() if cached else None,
                          cache_lengths=cache_lengths)
            if profile:
                result.update(prefix_ms=start_event.elapsed_time(prefix_event),
                              denoise_ms=prefix_event.elapsed_time(end_event))
            return result
        return actions
