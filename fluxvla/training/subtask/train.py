"""True-epoch DDP training: weight-only snapshots; optimizer saved only at Stage2 final."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import sys
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
from fluxvla.transforms.subtask_fast_prompts import PrepareSubtaskFastPrompts


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
    b['real']=torch.tensor([i>=0 for i in indices],device=device)
    return b


def local_counts(indices, packed, fx, horizon, with_actions):
    counts=np.zeros(3,dtype='float64')
    for i in indices:
        if i<0:continue
        p=packed[i]['high']
        counts[0] += np.count_nonzero(p['subtask_region_mask'][1:] & (p['labels'][1:]!=-100))
        counts[1] += np.count_nonzero(p['action_region_mask'][1:] & (p['labels'][1:]!=-100))
        if with_actions:counts[2]+=np.count_nonzero(fx.action_masks[i,:horizon])
    return counts


def global_counts(batches,packed,fx,horizon,with_actions,device,world):
    result=torch.as_tensor(np.stack([local_counts(b,packed,fx,horizon,with_actions) for b in batches]),device=device)
    if world>1:dist.all_reduce(result)
    return result


class Loss(torch.nn.Module):
    def __init__(self,model,with_actions):
        super().__init__();self.model=model;self.with_actions=with_actions

    def forward(self,b,factors):
        m=self.model
        high=m.forward_high_level(b['images'],b['high_input_ids'],b['high_valid_mask'],b['high_ar_mask'],
            b['high_labels'],b['img_masks'],has_hl=b['real'],
            hl_subtask_region_mask=b['high_subtask_region_mask'],hl_action_region_mask=b['high_action_region_mask'])
        raw=[high['loss_subtask'],high['loss_action_ce']]
        if self.with_actions:
            low=m.forward_low_level(b['images'],torch.zeros(len(b['real']),32,device=b['images'].device),
                b['low_input_ids'],b['low_valid_mask'],b['actions'],b['img_masks'],
                action_masks=b['action_masks'],has_action=b['real'])
            raw.append(low['loss'])
        else:raw.append(raw[0]*0.)
        weights=[m.hl_loss_weight,m.fast_loss_weight,m.action_loss_weight]
        total=sum(v*w*factors[i] for i,(v,w) in enumerate(zip(raw,weights)))
        return total,torch.stack([v.detach() for v in raw])


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
    loss_model.eval()
    micro=cfg['micro_batch'];block=micro*world
    totals=np.zeros(3);denoms=np.zeros(3)
    for start in range(0,len(indices),block):
        full=indices[start:start+block];full+= [-1]*(block-len(full))
        chunk=full[rank*micro:(rank+1)*micro]
        c=global_counts([chunk],packed,fx,horizon,loss_model.with_actions,device,world)[0]
        b=make_batch(chunk,fx,packed,cfg,image_size,device,horizon,loss_model.with_actions)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
            _,raw=loss_model(b,torch.ones(3,device=device))
        totals+=raw.float().cpu().numpy()*c.cpu().numpy();denoms+=c.cpu().numpy()
    loss_model.train()
    return dict(zip(['subtask_ce','fast_ce','flow_mse'],(totals/np.maximum(denoms,1)).tolist()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--stage',required=True)
    ap.add_argument('--initialize');ap.add_argument('--smoke',action='store_true');ap.add_argument('--device',default='cuda')
    ap.add_argument('--stop-after-epoch',type=int,help='controlled recovery drill; does not change the LR schedule')
    ap.add_argument('--stop-after-update',type=int,help='controlled stop; never creates an extra snapshot')
    ap.add_argument('--warm-start',action='store_true',help='explicitly continue latest weight-only snapshot with a fresh optimizer; not exact resume')
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text())
    stage=next(s for s in cfg['stages'] if s['name']==args.stage)
    policy=checkpoint_policy(cfg,args.stage)
    interval=policy.get('save_every_steps')
    rank,world,device=ddp_setup(args.device)
    if not args.smoke and (device.type!='cuda' or world!=cfg['expected_gpus']):
        raise ValueError(f'require {cfg["expected_gpus"]} GPUs, got world={world}, device={device}')
    if device.type=='cuda':torch.cuda.set_device(device)
    seed=cfg['seed']+rank
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    fx=Fixtures(cfg['cache']);ready=Path(cfg['cache'])/'READY.json'
    identity=digest(ready)
    tok=build_tokenizer_from_cfg(dict(type='PretrainedTokenizer',model_path=cfg['tokenizer']))
    prompts=PrepareSubtaskFastPrompts(tok,state_dim=14,max_len=cfg['max_prompt_length'],model_vocab_size=257152)
    packed=prepare(fx,prompts,14)
    output=Path(cfg['output']);out=output/args.stage
    if rank==0:out.mkdir(parents=True,exist_ok=True)
    barrier(world)
    model,model_cfg=build_model(stage['preset'],args.smoke,device,cfg['pretrained'],
        torch.bfloat16 if cfg['dtype']=='bf16' else torch.float32,cfg['attn_math'])
    with_actions=model.action_loss_weight>0
    for name,param in model.named_parameters():
        if name=='llm_expert.embed_tokens.weight' or (not with_actions and name.startswith(('llm_expert','action_in_proj','action_out_proj','time_mlp_in','time_mlp_out'))):
            param.requires_grad_(False)
    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=stage['peak_lr'],weight_decay=cfg['weight_decay'])
    train=fx.subset('train');validation=fx.subset('validation')
    if not train or not validation:raise ValueError('both train and validation anchors required')
    steps=math.ceil(len(train)/cfg['global_batch']);total=steps*stage['epochs']
    metadata=dict(stage=args.stage,preset=stage['preset'],cache_identity=identity,config=cfg,world_size=world,
        steps_per_epoch=steps,total_updates=total,smoke=args.smoke,contract=prompts.variant)
    # Include model/normalization identity in the recovery contract, not just trainable tensors.
    metadata['model_config']=model_cfg
    metadata['normalization_sha256']=digest(Path(cfg['cache'])/'samples.json')
    start_epoch=0;batch_cursor=0;update=0;restored_rng=None;reuse_final_full=False
    complete=out/'COMPLETE.json'
    if complete.exists():
        done=json.loads(complete.read_text())
        verify_bundle(done['deployment'])
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
        initialized=load_bundle(model,Path(args.initialize))
        init=initialized['metadata']
        if init['stage']!='stage1' or init['update']!=init['total_updates'] or init['cache_identity']!=identity:
            raise ValueError('stage2 initialization must be the completed stage1 deployment for this cache')
    elif args.stage=='stage2':
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
            counts=global_counts(batches,packed,fx,horizon,with_actions,device,world)
            denominators=counts.sum(0).clamp_min(1)
            opt.zero_grad(set_to_none=True)
            update+=1
            lr=lr_at(update,total,stage['peak_lr'],cfg['warmup_fraction'],cfg['min_lr_ratio'])
            for group in opt.param_groups:group['lr']=lr
            values=torch.zeros(3,device=device)
            for micro_index,chunk in enumerate(batches):
                b=make_batch(chunk,fx,packed,cfg,image_size,device,horizon,with_actions)
                factors=counts[micro_index]/denominators
                sync=wrapped.no_sync() if world>1 and micro_index<len(batches)-1 else nullcontext()
                with sync:
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda' and cfg['dtype']=='bf16'):
                        loss,raw=wrapped(b,factors)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
                    loss.backward()
                values+=raw*factors
            grad=torch.nn.utils.clip_grad_norm_(trainable,cfg['grad_clip'])
            if not torch.isfinite(grad):raise FloatingPointError('nonfinite gradient norm')
            opt.step()
            if rank==0:
                row=dict(stage=args.stage,epoch=epoch,update=update,total_updates=total,lr=lr,
                    subtask_ce=float(values[0]),fast_ce=float(values[1]),flow_mse=float(values[2]),grad_norm=float(grad))
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
        if args.stage=='stage1' and epoch in eval_epochs:
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
                    for _, ep, u, p in cand:
                        if p.exists():
                            shutil.rmtree(p)
                    print(json.dumps(dict(status='STAGE1_PICK_BEST', metric=metric, best_epoch=best_ep,
                        best_update=best_u, best_val=best_loss, kept=str(deployment),
                        deleted=[ep for _,ep,_,_ in cand if ep!=best_ep])), flush=True)
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
