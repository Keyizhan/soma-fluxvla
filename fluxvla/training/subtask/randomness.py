"""Stateless sample keyed flow draws; validation never consumes training RNG."""
import hashlib
import torch
from fluxvla.engines.utils.model_utils import sample_beta

def flow_random(ids, actions, seed, stream, update=0, time_sampler='beta',
                time_beta_alpha=1.5, time_beta_beta=1.0):
    noise=[];times=[]
    for sample in ids:
        key=f'{seed}:{stream}:{update}:{sample}'
        def generator(component):
            value=int.from_bytes(hashlib.sha256(f'{key}:{component}'.encode()).digest()[:8],'little')%(2**63-1)
            return torch.Generator(device='cpu').manual_seed(value)
        noise.append(torch.randn(actions.shape[1:],generator=generator('noise')))
        # Configured Beta law (default 1.5, 1), scaled away from endpoints.
        # MUST use the same correct Beta law; retain independent per-sample RNG.
        times.append(sample_beta(time_beta_alpha, time_beta_beta, 1, 'cpu', sampler=time_sampler,
                                 generator=generator('time'))[0]*.999+.001)
    return torch.stack(noise).to(actions),torch.stack(times).to(device=actions.device)


def rtc_random(ids, seed, stream, update, config, device):
    """Use the ordinary RTC sampler with isolated, sample-keyed RNG."""
    from fluxvla.engines.utils.rtc_training import sample_training_delay
    values=[]
    for sample in ids:
        key=f'{seed}:{stream}:{update}:{sample}:rtc'
        value=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63-1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(value)
            values.append(sample_training_delay(1,config['max_delay'],config['distribution'],config['temperature'],'cpu')[0])
    return torch.stack(values).to(device)
