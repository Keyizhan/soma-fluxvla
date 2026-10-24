"""Loss-weight independent evaluation with token CE counts and contributions."""
import argparse
import json
from pathlib import Path
import torch
from .config import load_config, canonical_prompts, stage1_identity
from .common import digest
from .checkpoints import verify_bundle, load_bundle
from .fixtures import Fixtures, build_model, prepare
from .train import Loss, validate
from .prepare import verify
from fluxvla.engines import build_tokenizer_from_cfg

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--bundle',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--bbox-generation-limit',type=int,default=0,help='free-running validation samples; 0 disables')
    p.add_argument('--bbox-max-new-tokens',type=int,default=256)
    a=p.parse_args();cfg=load_config(a.config);verify(cfg)
    meta=verify_bundle(a.bundle)['metadata'];device=torch.device(a.device)
    if meta.get('cache_identity')!=digest(Path(cfg['cache'])/'READY.json') or meta.get('stage1_identity')!=stage1_identity(cfg):
        raise ValueError('evaluation cache/common configuration differs from bundle')
    model,model_cfg=build_model(meta['preset'],meta.get('smoke',False),device,None,
        torch.bfloat16 if cfg['dtype']=='bf16' else torch.float32,cfg['attn_math'],
        time_sampler=cfg['time_sampler'],time_beta_alpha=cfg['time_beta_alpha'],time_beta_beta=cfg['time_beta_beta'],
        loss_objective=cfg['loss_objective'],joint_flow_weight=cfg['joint_flow_weight'],
        text_loss=cfg.get('text_loss'),bbox=cfg.get('bbox'))
    load_bundle(model,Path(a.bundle))
    fx=Fixtures(cfg['cache'])
    tok=build_tokenizer_from_cfg(dict(type='PretrainedTokenizer',model_path=cfg['tokenizer']))
    prompts=canonical_prompts(tok,cfg)
    packed=prepare(fx,prompts,14)
    if meta.get('subtask_mask_spec') is not None and prompts.subtask_mask_spec.to_dict()!=meta['subtask_mask_spec']:
        raise ValueError('evaluation template masks differ from checkpoint training masks')
    from .text_loss import apply_text_schedule
    apply_text_schedule(model,cfg.get('text_loss'),meta['preset'],meta['update'],meta['total_updates'])
    result=validate(Loss(model,True,metrics=True),fx.subset('validation'),fx,packed,cfg,
        model_cfg['vision_backbone']['vision_config']['image_size'],device,model.n_action_steps,1,0)
    if a.bbox_generation_limit:
        if cfg.get('bbox') is None or a.bbox_generation_limit < 0:
            raise ValueError('bbox generation requires bbox config and positive limit')
        from .bbox_evaluate import evaluate_grounding
        result['bbox_generation'] = evaluate_grounding(model,prompts,fx,fx.subset('validation')[:a.bbox_generation_limit],
            model_cfg['vision_backbone']['vision_config']['image_size'],device,a.bbox_max_new_tokens)
    print(json.dumps(dict(metrics=result,samples=len(fx.subset('validation')),split='validation',flow_seed=cfg['seed'])))

if __name__=='__main__':main()
