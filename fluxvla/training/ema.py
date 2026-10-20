"""Optional EMA experiment utilities; canonical two-stage training does not use EMA."""
from contextlib import contextmanager
import math
import torch


def trainable_parameters(model):
    return {name: value for name, value in model.named_parameters() if value.requires_grad}


class EMA:
    def __init__(self, parameters, decay):
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError('EMA decay must be finite in [0, 1)')
        self.decay = float(decay)
        self.shadow = {k: v.detach().float().clone() for k, v in parameters.items()}

    @property
    def enabled(self):
        return self.decay > 0

    @torch.no_grad()
    def update(self, parameters):
        if set(parameters) != set(self.shadow):
            raise ValueError('EMA parameter set changed')
        for name, param in parameters.items():
            self.shadow[name].mul_(self.decay).add_(param.detach().float(), alpha=1-self.decay)

    def state_dict(self):
        return dict(decay=self.decay, shadow={k: v.clone() for k, v in self.shadow.items()})

    def load_state_dict(self, state):
        if set(state['shadow']) != set(self.shadow):
            raise ValueError('EMA checkpoint parameter set differs')
        replacement = EMA(state['shadow'], state['decay'])
        for name, value in replacement.shadow.items():
            if value.shape != self.shadow[name].shape:
                raise ValueError('EMA checkpoint shape differs')
            replacement.shadow[name] = value.to(self.shadow[name].device)
        self.decay, self.shadow = replacement.decay, replacement.shadow


@contextmanager
def ema_weights(model, ema):
    params = dict(model.named_parameters())
    backup = {k: params[k].detach().clone() for k in ema.shadow}
    try:
        with torch.no_grad():
            for k, value in ema.shadow.items():
                params[k].copy_(value)
        yield
    finally:
        with torch.no_grad():
            for k, value in backup.items():
                params[k].copy_(value)
