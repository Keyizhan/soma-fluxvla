"""True-epoch DDP training: weight-only snapshots; optimizer saved only at Stage2 final."""
from .config import load_config, canonical_prompts
import argparse
import hashlib
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
import torch.distributed as dist
from .common import digest, epoch_batches, lr_at, write_json
from .checkpoints import (GIB, atomic_full, checkpoint_policy, durable_json,
    export_bundle, load_bundle, verify_bundle, verify_full)


from .fixtures import Fixtures, build_model, collate, ddp_setup, ddp_cleanup, prepare
from fluxvla.engines import build_tokenizer_from_cfg


def cleanup_stage1_candidates(candidates, keep_epoch_checkpoints=False):
    """Remove transient selection snapshots only when retention is disabled."""
    deleted=[]
    if not keep_epoch_checkpoints:
        for _,epoch,_,path in candidates:
            if path.exists():
                shutil.rmtree(path)
                deleted.append(epoch)
    return deleted


def configure_trainable_parameters(model, preset):
    """Freeze inactive branches explicitly, including vision in expert-only mode."""
    with_actions = model.action_loss_weight > 0
    for name, param in model.named_parameters():
        frozen = name == 'llm_expert.embed_tokens.weight'
        if preset == 'action_expert':
            frozen |= name.startswith(('llm_backbone', 'vision_backbone', 'projector'))
        if not with_actions:
            frozen |= name.startswith(('llm_expert', 'action_in_proj', 'action_out_proj',
                                       'time_mlp_in', 'time_mlp_out'))
        param.requires_grad_(not frozen)


def build_optimizer(model, peak_lr, weight_decay):
    from types import SimpleNamespace
    from fluxvla.optimizers.lr_scheduler_policies import BaseLRSchedulerPolicy
    groups=BaseLRSchedulerPolicy().build_param_groups(
        SimpleNamespace(vla=model,optimizer_cfg=dict(lr=peak_lr,weight_decay=weight_decay)))
    return torch.optim.AdamW(groups,lr=peak_lr)


def barrier(world):
    if world>1:dist.barrier()


def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                python=random.getstate(),numpy=json.dumps(np.random.get_state(),default=lambda x:x.tolist()))


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python'])
    s=json.loads(state['numpy']);np.random.set_state((s[0],np.asarray(s[1],dtype='uint32'),s[2],s[3],s[4]))


def make_batch(indices, fx, packed, cfg, image_size, device, horizon, actions):
    safe=[max(0,i) for i in indices]
    b=collate([packed[i] for i in safe],safe,fx,image_size,device,actions,horizon)
    from .randomness import flow_random
    b['sample_ids']=[fx.records[i]['sample_id'] for i in safe]
    if actions:
        b['flow_noise'],b['flow_time']=flow_random(b['sample_ids'],b['actions'],cfg['seed'],cfg.get('_flow_stream','train'),cfg.get('_flow_update',0),time_sampler=cfg['time_sampler'],
            time_beta_alpha=cfg['time_beta_alpha'],time_beta_beta=cfg['time_beta_beta'])
    b.update(flow_seed=cfg['seed'],flow_stream=cfg.get('_flow_stream','train'),flow_update=cfg.get('_flow_update',0))
    b['real']=torch.tensor([i>=0 for i in indices],device=device)
    return b


class Loss(torch.nn.Module):
    def __init__(self,model,with_actions,metrics=False):
        super().__init__();self.model=model;self.with_actions=with_actions;self.metrics=metrics

    def forward(self,b,factors,return_counts=False,return_statistics=False):
        m=self.model
        counts=torch.zeros(3,device=b['images'].device)
        unified=getattr(m,'text_loss_mode','separate')=='unified_token'
        text_enabled=(m.token_loss_weight>0 if unified else m.hl_loss_weight>0 or m.fast_loss_weight>0)
        high=low=None
        if not self.metrics and not text_enabled:
            raw=[torch.zeros((),device=b['images'].device)]*2
        else:
            bbox_args = {}
            if getattr(m, 'bbox', None) is not None:
                bbox_args = dict(hl_bbox_groups={name: b['high_bbox_'+name+'_groups']
                    for name in ('type','coordinate','format')}, bbox_denominators=b.get('bbox_denominators'))
            high=m.forward_high_level(b['images'],b['high_input_ids'],b['high_valid_mask'],b['high_ar_mask'],
                b['high_labels'],b['img_masks'],has_hl=b['real'],
                hl_subtask_region_mask=b['high_subtask_region_mask'],hl_action_region_mask=b['high_action_region_mask'],
                **bbox_args,
                **{k: b['high_'+k[3:]] for k in ('hl_subtask_key_mask','hl_subtask_template_mask','hl_action_code_mask') if 'high_'+k[3:] in b})
            raw=[high['loss_subtask'],high['loss_action_ce']]
            counts[0]=high['target_count'];counts[1]=high['action_target_count']
        if self.with_actions:
            rtc_delays=None
            if m.rtc_training_config and m.rtc_training_config.get('enabled',False) and 'sample_ids' in b:
                from .randomness import rtc_random
                rtc_delays=rtc_random(b['sample_ids'],b['flow_seed'],b['flow_stream'],b['flow_update'],m.rtc_training_config,b['images'].device)
            low=m.forward_low_level(b['images'],torch.zeros(len(b['real']),32,device=b['images'].device),
                b['low_input_ids'],b['low_valid_mask'],b['actions'],b['img_masks'],
                action_masks=b['action_masks'],has_action=b['real'],noise=b.get('flow_noise'),time=b.get('flow_time'),rtc_delays=rtc_delays)
            raw.append(low['loss'])
            counts[2]=low['valid_action_count']
        else:raw.append(raw[0]*0.)
        if high is not None:
            sums=[high.get('subtask_nll_sum',raw[0]*counts[0]),
                  high.get('fast_nll_sum',raw[1]*counts[1])]
            count=counts[:2].sum()
            contributions=[high.get('token_subtask_contribution',sums[0]/count.clamp_min(1)),
                           high.get('token_fast_contribution',sums[1]/count.clamp_min(1))]
            token=high.get('loss_token',sum(contributions))
        else:
            sums=[raw[0],raw[1]]
            contributions=[raw[0],raw[1]]
            token=raw[0]
        sums.append(low.get('action_squared_error_sum',raw[2]*counts[2]) if low is not None else raw[2])
        flow_term=raw[2]*m.action_loss_weight*factors[2]
        if unified:
            if not torch.equal(factors[0],factors[1]):
                raise ValueError('unified token CE requires equal accumulation factors for both token regions')
            token_term=(high.get('loss_token_objective',token) if high is not None else token)*m.token_loss_weight*factors[0]
        else:
            token_term=(high.get('loss_subtask_objective',raw[0]) if high is not None else raw[0])*m.hl_loss_weight*factors[0]+raw[1]*m.fast_loss_weight*factors[1]
        weighted_fast = (high.get('token_fast_objective', contributions[1]) if high is not None else contributions[1])*m.token_loss_weight*factors[0] if unified else raw[1]*m.fast_loss_weight*factors[1]
        weighted_subtask = token_term-weighted_fast
        bbox_term = token_term*0
        if high is not None and text_enabled and getattr(m,'bbox',None) is not None:
            bbox_term = high['loss_bbox'] * (1. if 'bbox_denominators' in b else factors[0])
        total=token_term+flow_term+bbox_term
        result=(total,torch.stack([v.detach() for v in raw]))
        if return_statistics:
            # Sufficient statistics are pooled by count; optimization entries already
            # include the accumulation factor and are averaged over ranks only.
            optimization=torch.stack([total,token*factors[0],
                contributions[0]*factors[0],contributions[1]*factors[0],token_term,flow_term])
            statistics=torch.cat([torch.stack(sums).detach().double(),counts.detach().double(),
                optimization.detach().double(),counts.new_tensor([high is not None]).double()])
            extra = []
            for name in ('subtask_key', 'subtask_template', 'fast_code'):
                for suffix in ('_nll_sum', '_count'):
                    extra.append(high.get(name+suffix, counts.new_zeros(())) if high is not None else counts.new_zeros(()))
            statistics = torch.cat([statistics, torch.stack(extra+[weighted_subtask,weighted_fast]).detach().double()])
            if getattr(m, 'bbox', None) is not None:
                bbox_stats = [high['bbox_'+name+suffix] if high is not None else counts.new_zeros(())
                              for name in ('type','coordinate','format') for suffix in ('_sum','_count')]
                statistics = torch.cat([statistics, torch.stack(bbox_stats+[bbox_term]).detach().double()])
            return (*result,statistics)
        return (*result,counts) if return_counts else result


def diagnostic_metrics(sums,counts):
    """Count-pooled diagnostics, distinct from microbatch-averaged optimization."""
    if isinstance(sums,torch.Tensor):sums=sums.detach().cpu().tolist()
    if isinstance(counts,torch.Tensor):counts=counts.detach().cpu().tolist()
    sums=[float(v) for v in sums];counts=[float(v) for v in counts]
    def ratio(numerator,denominator):
        return numerator/denominator if denominator>0 else None
    token_count=counts[0]+counts[1]
    nll=sums[0]+sums[1]
    return dict(metric_scope='global_valid_element_pool',
        subtask_ce=ratio(sums[0],counts[0]),fast_ce=ratio(sums[1],counts[1]),
        flow_mse=ratio(sums[2],counts[2]),token_ce=ratio(nll,token_count),
        subtask_token_count=counts[0],fast_token_count=counts[1],token_count=token_count,
        flow_element_count=counts[2],subtask_nll_sum=sums[0],fast_nll_sum=sums[1],
        flow_squared_error_sum=sums[2],
        subtask_token_fraction=ratio(counts[0],token_count),fast_token_fraction=ratio(counts[1],token_count),
        token_subtask_contribution=ratio(sums[0],token_count),token_fast_contribution=ratio(sums[1],token_count),
        subtask_loss_fraction=ratio(sums[0],nll),fast_loss_fraction=ratio(sums[1],nll))


def training_metrics(statistics,world=1):
    """Format already-summed rank statistics; all-reduce must happen first."""
    if isinstance(statistics,torch.Tensor):statistics=statistics.detach().cpu().tolist()
    row=diagnostic_metrics(statistics[:3],statistics[3:6])
    if len(statistics) >= 19:
        for index, name in enumerate(('subtask_key', 'subtask_template', 'fast_code')):
            numerator, count = statistics[13+2*index:15+2*index]
            row[name+'_ce'] = numerator/count if count > 0 else None
            row[name+'_token_count'] = count
    if len(statistics) >= 21:
        ws,wf = (float(v)/world for v in statistics[19:21])
        row.update(optimization_weighted_subtask_term=ws,optimization_weighted_fast_term=wf,
            optimization_weighted_subtask_fraction=ws/(ws+wf) if ws+wf>0 else None,
            optimization_weighted_fast_fraction=wf/(ws+wf) if ws+wf>0 else None)
    if len(statistics) >= 28:
        for i, name in enumerate(('type','coordinate','format')):
            total, count = statistics[21+2*i:23+2*i]
            row['bbox_'+name+'_ce'] = total/count if count else None
            row['bbox_'+name+'_count'] = count
        row['optimization_weighted_bbox_term'] = statistics[27]/world
        row['bbox_optimization_scope'] = 'global_valid_goal_mean_over_optimizer_update'
    row['optimization_loss_fraction_scope']='raw_unified_token_ce_not_weighted_objective'
    objective,token,subtask,fast,token_term,flow_term=[float(v)/world for v in statistics[6:12]]
    text_computed=bool(statistics[12]>0)
    row.update(optimization_scope=('legacy_microbatch_mean_plus_global_update_bbox_mean' if len(statistics)>=28 else 'equal_microbatch_mean_then_ddp_rank_mean'),
        optimization_loss=objective,
        optimization_token_ce=token if text_computed else None,
        optimization_token_subtask_contribution=subtask if text_computed else None,
        optimization_token_fast_contribution=fast if text_computed else None,
        optimization_subtask_loss_fraction=subtask/(subtask+fast) if text_computed and subtask+fast>0 else None,
        optimization_fast_loss_fraction=fast/(subtask+fast) if text_computed and subtask+fast>0 else None,
        optimization_weighted_token_term=token_term,optimization_weighted_flow_term=flow_term,
        text_metrics_computed=text_computed)
    return row


def save_deployment(model,out,metadata,world,rank,cfg,model_cfg):
    result=[None]
    if rank==0:
        try:
            path=out/f"step-{metadata['update']:08d}.deploy"
            export_bundle(model.state_dict(),path,cfg,model_cfg,metadata,
                          int(cfg['checkpoint'].get('safety_reserve_gib',20)*GIB))
            durable_json(out/'latest.json',dict(kind='deployment',path=str(path),
                epoch_index=metadata['epoch_index'],batch_cursor=metadata['batch_cursor'],update=metadata['update'],
                optimizer_present=False,exact_resume=False))
            result[0]=dict(ok=True)
            print(json.dumps(dict(deployment=str(path),update=metadata['update'],optimizer_saved=False)),flush=True)
        except Exception as e:result[0]=dict(ok=False,error=f'{type(e).__name__}: {e}')
    if world>1:dist.broadcast_object_list(result,src=0)
    if not result[0]['ok']:raise RuntimeError('deployment transaction failed: '+result[0]['error'])
    barrier(world)


def collect_rng(world):
    state=rng_state();states=[None]*world
    if world>1:dist.all_gather_object(states,state)
    else:states=[state]
    return states


@torch.no_grad()
def validate(loss_model,indices,fx,packed,cfg,image_size,device,horizon,world,rank):
    loss_model=Loss(loss_model.model,True,metrics=True)
    loss_model.eval()
    cfg=dict(cfg,_flow_stream='validation',_flow_update=0)
    micro=cfg['micro_batch'];block=micro*world
    bbox_enabled = getattr(loss_model.model, 'bbox', None) is not None
    combined=torch.zeros(18 if bbox_enabled else 12,device=device,dtype=torch.float64)
    for start in range(0,len(indices),block):
        full=indices[start:start+block];full+= [-1]*(block-len(full))
        chunk=full[rank*micro:(rank+1)*micro]
        b=make_batch(chunk,fx,packed,cfg,image_size,device,horizon,True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
            _,_,statistics=loss_model(b,torch.ones(3,device=device),return_statistics=True)
        combined[:6]+=statistics[:6]
        combined[6:12]+=statistics[13:19]
        if bbox_enabled:
            combined[12:18]+=statistics[21:27]
    if world>1:dist.all_reduce(combined)
    loss_model.train()
    result = diagnostic_metrics(combined[:3],combined[3:6])
    for index, name in enumerate(('subtask_key', 'subtask_template', 'fast_code')):
        numerator, count = combined[6+2*index:8+2*index].tolist()
        result[name+'_ce'] = numerator/count if count > 0 else None
        result[name+'_token_count'] = count
    if bbox_enabled:
        for i, name in enumerate(('type','coordinate','format')):
            total, count = combined[12+2*i:14+2*i].tolist()
            result['bbox_'+name+'_ce'] = total/count if count else None
            result['bbox_'+name+'_count'] = count
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--stage',required=True)
    ap.add_argument('--initialize');ap.add_argument('--smoke',action='store_true');ap.add_argument('--device',default='cuda')
    ap.add_argument('--stop-after-epoch',type=int,help='controlled recovery drill; does not change the LR schedule')
    ap.add_argument('--stop-after-update',type=int,help='controlled stop; never creates an extra snapshot')
    ap.add_argument('--warm-start',action='store_true',help='explicitly continue latest weight-only snapshot with a fresh optimizer; not exact resume')
    args=ap.parse_args();cfg=load_config(args.config)
    from .common import validate_training_method
    validate_training_method(cfg)
    stage=next(s for s in cfg['stages'] if s['name']==args.stage)
    if args.stage=='stage1' and Path(cfg['output']).resolve()!=(Path(cfg['shared_stage1_root'])/f"seed-{cfg['seed']}").resolve():
        raise ValueError('stage1 must use shared per-seed output; use the comparison/run orchestrator')
    policy=checkpoint_policy(cfg,args.stage)
    interval=policy.get('save_every_steps')
    rank,world,device=ddp_setup(args.device)
    if not args.smoke and (device.type!='cuda' or world!=cfg['expected_gpus']):
        raise ValueError(f'require {cfg["expected_gpus"]} GPUs, got world={world}, device={device}')
    if device.type=='cuda':torch.cuda.set_device(device)
    seed=cfg['seed']+rank
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    from .prepare import validate_cache_config
    ready=Path(cfg['cache'])/'READY.json'
    validate_cache_config(json.loads(ready.read_text()),cfg)
    fx=Fixtures(cfg['cache'])
    identity=digest(ready)
    tok=build_tokenizer_from_cfg(dict(type='PretrainedTokenizer',model_path=cfg['tokenizer']))
    prompts=canonical_prompts(tok,cfg)
    packed=prepare(fx,prompts,14)
    output=Path(cfg['output']);out=output/args.stage
    if rank==0:out.mkdir(parents=True,exist_ok=True)
    barrier(world)
    model,model_cfg=build_model(stage['preset'],args.smoke,device,cfg['pretrained'],
        torch.bfloat16 if cfg['dtype']=='bf16' else torch.float32,cfg['attn_math'],
        time_sampler=cfg['time_sampler'],time_beta_alpha=cfg['time_beta_alpha'],time_beta_beta=cfg['time_beta_beta'],
        loss_objective=cfg['loss_objective'],joint_flow_weight=cfg['joint_flow_weight'],text_loss=cfg.get('text_loss'),bbox=cfg.get('bbox'))
    with_actions=model.action_loss_weight>0
    configure_trainable_parameters(model, stage['preset'])
    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=build_optimizer(model,stage['peak_lr'],cfg['weight_decay'])
    train=fx.subset('train');validation=fx.subset('validation')
    if not train or not validation:raise ValueError('both train and validation anchors required')
    steps=math.ceil(len(train)/cfg['global_batch']);total=steps*stage['epochs']
    from .config import stage1_identity
    metadata=dict(stage1_identity=stage1_identity(cfg),stage=args.stage,preset=stage['preset'],cache_identity=identity,config=cfg,world_size=world,
        steps_per_epoch=steps,total_updates=total,smoke=args.smoke,contract=prompts.variant,
        loss_objective=cfg['loss_objective'],joint_flow_weight=cfg['joint_flow_weight'])
    # Include model/normalization identity in the recovery contract, not just trainable tensors.
    metadata['model_config']=model_cfg
    if 'text_loss' in cfg and prompts.subtask_mask_spec is not None:
        metadata['subtask_mask_spec']=prompts.subtask_mask_spec.to_dict()
        metadata['subtask_mask_fingerprint']=prompts.subtask_mask_spec.fingerprint
    metadata['normalization_sha256']=hashlib.sha256(json.dumps(json.loads((Path(cfg['cache'])/'samples.json').read_text())['normalization'],sort_keys=True).encode()).hexdigest()
    if args.stage=='stage2' and cfg['training_method'] in ('hybrid_insulated','two_stage'):
        receipt_path=Path(cfg['shared_stage1_root'])/f"seed-{cfg['seed']}"/'selected-stage1.json'
        selected=json.loads(receipt_path.read_text())
        shared_manifest=verify_bundle(selected['deployment'])
        if selected['weight_sha256']!=digest(Path(selected['deployment'])/'model.safetensors') or selected['stage1_identity']!=stage1_identity(cfg) or selected['cache_identity']!=identity:
            raise ValueError('selected shared stage1 identity changed')
        if shared_manifest['metadata'].get('stage1_identity')!=stage1_identity(cfg):raise ValueError('shared bundle configuration changed')
        metadata.update(stage1_weight_sha256=selected['weight_sha256'],optimizer_reset=True,schedule_reset=True)
    metadata['base_weight_sha256']=cfg['base_sha256']
    metadata['base_weight_identity_source']='configured SHA256; orchestrator preflight verifies actual base file'
    metadata.setdefault('optimizer_reset',True)
    metadata.setdefault('schedule_reset',True)
    metadata['initialization_source']='selected_stage1' if cfg['training_method'] in ('hybrid_insulated','two_stage') and args.stage=='stage2' else 'base'
    metadata['optimizer_updates_budget']=steps*sum(s['epochs'] for s in cfg['stages'])
    metadata['extra_stage1_updates']=steps*next((s['epochs'] for s in cfg['stages'] if s['name']=='stage1'),0)
    start_epoch=0;batch_cursor=0;update=0;restored_rng=None;reuse_final_full=False
    complete=out/'COMPLETE.json'
    if complete.exists():
        done=json.loads(complete.read_text())
        done_manifest=verify_bundle(done['deployment'])
        for key,value in metadata.items():
            if done_manifest['metadata'].get(key)!=value:raise ValueError(f'completed training contract changed: {key}')
        ddp_cleanup(rank,world)
        return
    latest=out/'latest.json'
    record=json.loads(latest.read_text()) if latest.is_file() else None
    if record:
        load=Path(record['path'])
        if record.get('kind')=='deployment':
            if not args.warm_start:
                raise ValueError('latest checkpoint is weight-only; pass --warm-start explicitly to continue '
                                 'with optimizer reinitialized (not bitwise recovery)')
            manifest=load_bundle(model,load)
            state=manifest['metadata']
            for key,value in metadata.items():
                if state.get(key)!=value:raise ValueError(f'warm-start contract changed: {key}')
            if rank==0:print(json.dumps(dict(status='WEIGHT_ONLY_WARM_START',update=state['update'],
                optimizer_reinitialized=True,rng_restored=False,bitwise_guarantee=False)),flush=True)
        elif record.get('kind')=='full':
            state=verify_full(load)
            for key,value in metadata.items():
                if state.get(key)!=value:raise ValueError(f'finalization contract changed: {key}')
            if args.stage!='stage2' or state['update']!=total:
                raise ValueError('final_only policy rejects a non-final optimizer checkpoint')
            model.load_state_dict(state['model_state'],strict=True)
            opt.load_state_dict(state['optimizer'])
            restored_rng=state['rng'][rank]
            reuse_final_full=True
        else:raise ValueError('unknown checkpoint kind')
        start_epoch=state['epoch_index'];batch_cursor=state['batch_cursor'];update=state['update']
        if not 0<=start_epoch<=stage['epochs'] or not 0<=batch_cursor<steps or update!=start_epoch*steps+batch_cursor:
            raise ValueError('invalid checkpoint epoch/batch cursor')
        del state
    elif args.initialize:
        if cfg['training_method']=='hybrid':raise ValueError('hybrid must initialize directly from base')
        receipt_path=Path(cfg['shared_stage1_root'])/f"seed-{cfg['seed']}"/'selected-stage1.json'
        receipt=json.loads(receipt_path.read_text())
        if Path(receipt['deployment']).resolve()!=Path(args.initialize).resolve() or receipt['weight_sha256']!=digest(Path(args.initialize)/'model.safetensors'):
            raise ValueError('stage2 must load the selected shared stage1 weight hash')
        initialized=load_bundle(model,Path(args.initialize))
        init=initialized['metadata']
        if init.get('stage1_identity')!=stage1_identity(cfg):raise ValueError('shared stage1 configuration mismatch')
        metadata['stage1_weight_sha256']=digest(Path(args.initialize)/'model.safetensors')
        metadata['optimizer_reset']=True
        metadata['schedule_reset']=True
        if init['stage']!='stage1' or init['update']!=init['total_updates'] or init['cache_identity']!=identity:
            raise ValueError('stage2 initialization must be the completed stage1 deployment for this cache')
    elif args.stage=='stage2' and cfg['training_method'] in ('hybrid_insulated','two_stage'):
        raise ValueError('Stage2 requires a completed Stage1 deployment bundle or full Stage2 resume')
    loss_model=Loss(model,with_actions)
    wrapped=torch.nn.parallel.DistributedDataParallel(loss_model,
        device_ids=[device.index] if device.type=='cuda' else None,find_unused_parameters=with_actions) if world>1 else loss_model
    image_size=model_cfg['vision_backbone']['vision_config']['image_size'];horizon=model_cfg['n_action_steps']
    if rank==0:
        write_json(out/'contract.json',metadata)
        print(json.dumps(dict(stage=stage,world=world,train=len(train),validation=len(validation),
            steps_per_epoch=steps,total_updates=total,micro_batch=cfg['micro_batch'],global_batch=cfg['global_batch'],
            accum=cfg['global_batch']//(cfg['micro_batch']*world),resume_epoch=start_epoch)),flush=True)
    model.train()
    # Restore after DDP construction: setup must not consume the saved per-rank stream.
    if restored_rng is not None:restore_rng(restored_rng)
    stopped=False
    for epoch_index in range(start_epoch,stage['epochs']):
        epoch=epoch_index+1
        began=time.monotonic()
        for cursor,batches in enumerate(epoch_batches(train,cfg['seed'],epoch,cfg['global_batch'],cfg['micro_batch'],world,rank)):
            if epoch_index==start_epoch and cursor<batch_cursor:continue
            opt.zero_grad(set_to_none=True)
            update+=1
            from .text_loss import apply_text_schedule
            apply_text_schedule(model,cfg.get('text_loss'),stage['preset'],update,total)
            lr=lr_at(update,total,stage['peak_lr'],cfg['warmup_fraction'],cfg['min_lr_ratio'])
            for group in opt.param_groups:group['lr']=lr
            bbox_enabled = getattr(model, 'bbox', None) is not None
            bbox_denominators = None
            if bbox_enabled:
                from fluxvla.models.vlas.bbox_loss import update_denominators
                bbox_denominators = update_denominators(packed, batches, device)
            statistics=torch.zeros(28 if bbox_enabled else 21,device=device,dtype=torch.float64)
            for micro_index,chunk in enumerate(batches):
                b=make_batch(chunk,fx,packed,dict(cfg,_flow_update=update),image_size,device,horizon,with_actions)
                if bbox_denominators is not None:
                    b['bbox_denominators'] = bbox_denominators
                factors=torch.full((3,),1./len(batches),device=device)
                sync=wrapped.no_sync() if world>1 and micro_index<len(batches)-1 else nullcontext()
                with sync:
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda' and cfg['dtype']=='bf16'):
                        loss,_,micro_statistics=wrapped(b,factors,return_statistics=True)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                    loss.backward()
                statistics+=micro_statistics
            grad=torch.nn.utils.clip_grad_norm_(trainable,cfg['grad_clip'])
            if not torch.isfinite(grad):raise FloatingPointError('nonfinite gradient norm')
            opt.step()
            # Logging uses sums/counts across every rank, never just rank zero.
            # This reduction is detached and does not alter DDP optimization.
            if world>1:dist.all_reduce(statistics)
            if rank==0:
                row=dict(stage=args.stage,epoch=epoch,update=update,total_updates=total,lr=lr,
                    grad_norm=float(grad),text_loss_mode=model.text_loss_mode,
                    token_loss_weight=model.token_loss_weight,flow_loss_weight=model.action_loss_weight,
                    subtask_loss_weight=model.hl_loss_weight,fast_loss_weight=model.fast_loss_weight,
                    **training_metrics(statistics,world))
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if update%10==0 or update%steps==0 or update==1:print(json.dumps(row),flush=True)
            stop_now=args.stop_after_update is not None and update>=args.stop_after_update
            if args.stage=='stage2' and update%interval==0 and update<total:
                next_epoch=epoch_index+(cursor+1==steps)
                next_cursor=0 if cursor+1==steps else cursor+1
                meta=dict(metadata,epoch=epoch,epoch_index=next_epoch,batch_cursor=next_cursor,update=update)
                save_deployment(model,out,meta,world,rank,cfg,model_cfg)
            if stop_now and cursor+1<steps:
                stopped=True
                break
        if stopped:break
        # Validation is deterministic and consumes no training RNG; restore it defensively.
        saved_rng=rng_state()
        val=validate(loss_model,validation,fx,packed,cfg,image_size,device,horizon,world,rank)
        restore_rng(saved_rng)
        meta=dict(metadata,epoch=epoch,epoch_index=epoch,batch_cursor=0,update=update,
                  validation=val,epoch_seconds=time.monotonic()-began)
        if rank==0:
            with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(meta)+'\n')
        eval_epochs=set(int(x) for x in (stage.get('eval_at_epochs') or []))
        if args.stage=='stage1' and (epoch in eval_epochs or stage.get('keep_epoch_checkpoints',False)):
            save_deployment(model,out,meta,world,rank,cfg,model_cfg)
            if rank==0:
                print(json.dumps(dict(status='STAGE1_EVAL',epoch=epoch,update=update,validation=val)),flush=True)
        if (args.stop_after_epoch and epoch>=args.stop_after_epoch) or (args.stop_after_update and update>=args.stop_after_update):break
    result=[None]
    if update==total:
        # Only the Stage2 final checkpoint ever serializes optimizer or per-rank RNG.
        states=collect_rng(world) if args.stage=='stage2' else None
        final_meta=dict(metadata,epoch=stage['epochs'],epoch_index=stage['epochs'],batch_cursor=0,update=update)
        if rank==0:
            try:
                reserve=int(policy.get('safety_reserve_gib',20)*GIB)
                final_path=out/f'step-{update:08d}.pt' if args.stage=='stage2' else None
                if final_path is not None:
                    if not (reuse_final_full and Path(record['path'])==final_path):
                        atomic_full(model,opt,final_path,final_meta,states,reserve)
                    durable_json(latest,dict(kind='full',path=str(final_path),epoch_index=stage['epochs'],
                        batch_cursor=0,update=update,optimizer_present=True,exact_resume=True))
                if args.stage=='stage1' and stage.get('eval_at_epochs'):
                    metric=stage.get('pick_best_metric','subtask_ce')
                    by_epoch={}
                    val_path=out/'validation.jsonl'
                    if val_path.is_file():
                        for line in val_path.read_text().splitlines():
                            if not line.strip():
                                continue
                            row=json.loads(line)
                            ep=row.get('epoch'); v=row.get('validation') or {}
                            if ep is not None and metric in v:
                                by_epoch[int(ep)]=(float(v[metric]), int(row.get('update') or 0))
                    cand=[]
                    for ep in stage['eval_at_epochs']:
                        ep=int(ep)
                        if ep not in by_epoch:
                            continue
                        loss,u=by_epoch[ep]
                        p=out/f'step-{u:08d}.deploy'
                        if p.exists():
                            cand.append((loss, ep, u, p))
                    if not cand:
                        raise RuntimeError('stage1 eval_at_epochs produced no deploy dirs')
                    cand.sort()
                    best_loss, best_ep, best_u, best_p = cand[0]
                    load_bundle(model, best_p)
                    final_meta=dict(final_meta, picked_eval_epoch=best_ep, picked_eval_update=best_u,
                        picked_val={metric: best_loss})
                    final_meta['update']=update
                    deployment=export_bundle(model.state_dict(), out/'final.deploy', cfg, model_cfg, final_meta, reserve)
                    deleted=cleanup_stage1_candidates(cand,stage.get('keep_epoch_checkpoints',False))
                    print(json.dumps(dict(status='STAGE1_PICK_BEST', metric=metric, best_epoch=best_ep,
                        best_update=best_u, best_val=best_loss, kept=str(deployment),
                        deleted=deleted,keep_epoch_checkpoints=stage.get('keep_epoch_checkpoints',False))), flush=True)
                else:
                    deployment=export_bundle(model.state_dict(),(out/'final.deploy' if args.stage=='stage1'
                        else out/f'step-{update:08d}.deploy'),cfg,model_cfg,final_meta,reserve)
                durable_json(complete,dict(epoch=stage['epochs'],update=update,deployment=str(deployment),
                    latest_full=str(final_path) if final_path is not None else None,optimizer_save='final_only'))
                if args.stage=='stage1':
                    durable_json(latest,dict(kind='deployment',path=str(deployment),update=update,
                        optimizer_present=False,exact_resume=False))
                result[0]=dict(ok=True)
            except Exception as e:result[0]=dict(ok=False,error=f'{type(e).__name__}: {e}')
        if world>1:dist.broadcast_object_list(result,src=0)
        if not result[0]['ok']:raise RuntimeError('final deployment failed: '+result[0]['error'])
    ddp_cleanup(rank,world)

if __name__=='__main__':main()
