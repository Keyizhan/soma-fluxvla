"""Small, explicitly AI-labelled real-data architecture experiment; no robot control."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time

import numpy as np
import torch

from fluxvla import register_training_components
register_training_components()
from fluxvla.engines import build_vla_from_cfg
from fluxvla.transforms.subtask_prompts import PrepareSubtaskPrompts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets',required=True)
    parser.add_argument('--fixtures',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--delta',help='Reload a model-only delta for evaluation')
    parser.add_argument('--checkpoint-fixtures',help='For evaluation only, verify prior checkpoint fixtures when evaluating expanded data')
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    fixtures=Path(args.fixtures);assets=Path(args.assets)
    torch.set_num_threads(4);torch.manual_seed(1702);torch.cuda.manual_seed_all(1702)
    rng=np.random.default_rng(1702)
    manifest=json.loads((fixtures/'samples.json').read_text())
    records=manifest['records'];data=np.load(fixtures/'samples.npz',allow_pickle=False)
    arrays={k:torch.from_numpy(data[k]).cuda() for k in data.files}
    arrays['images']=arrays['images'].float()/255*2-1
    data.close()
    train=[i for i,r in enumerate(records) if r['split']=='train']
    validation=[i for i,r in enumerate(records) if r['split']=='validation']
    if set(records[i]['episode'] for i in train)&set(records[i]['episode'] for i in validation):
        raise ValueError('Episode leakage')
    if len({records[i]['overall_task'] for i in train+validation})!=1:
        raise ValueError('This diagnostic requires the same overall goal for all stages')
    config=runpy.run_path(str(Path(__file__).resolve().parents[2]/'configs/pi05_subtask_architecture.py'))['model']
    # Preserve the historical shoes experiment contract when architecture defaults evolve.
    config.update(pretrained_name_or_path=str(assets/'model.safetensors'),ori_action_dim=14,
                  text_use_cache=False, action_prefix_cache=False, flow_prefix_cache=False,
                  stop_gradient_flow_to_prefix=False, text_decode_fp32=False)
    report=dict(seed=1702,source_category='AI experiment design and local measurement',
        annotation_source='AI_visual_annotation; not independently verified',
        fixture_manifest_sha256=hashlib.sha256((fixtures/'samples.json').read_bytes()).hexdigest(),
        train_indices=train,validation_indices=validation,records=records,torch=torch.__version__,
        contract=dict(base='public pi05_base; no identity checkpoint',template='pi05-subtask-v1',
            steps=args.steps,objective='CE + flow MSE, weights1:1',microbatch=1,accumulation=3,
            selection='one random training anchor per class per optimizer step; no validation selection',
            lr=1e-5,optimizer='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_grad_norm=1.,
            trainable='all except vision encoder and unused action-expert input embedding',
            precision='FP32 params + BF16 autocast',rtc=False,use_cache=False,max_new_tokens=16,
            time_sampling='explicit torch.distributions.Beta(1.5,1.0)*.999+.001; existing approximate helper bypassed',
            eval_time=.5,eval_noise='fixed torch CUDA generator seed1703; shared across condition comparisons',
            semantic_metric='exact free-generated text AND EOS; no class-constrained decoding'))
    print('load_start',flush=True)
    model=build_vla_from_cfg(config);model.from_pretrained();model=model.cuda().eval()
    if args.delta:
        state=torch.load(args.delta,map_location='cpu',weights_only=True)
        if state['fixture_manifest_sha256']!=report['fixture_manifest_sha256']:
            if args.steps or not args.checkpoint_fixtures:
                raise ValueError('Fixture mismatch; changed fixtures permitted only for explicit verified evaluation')
            prior_path=Path(args.checkpoint_fixtures)/'samples.json'
            prior=json.loads(prior_path.read_text())
            if hashlib.sha256(prior_path.read_bytes()).hexdigest()!=state['fixture_manifest_sha256']:
                raise ValueError('Checkpoint reference fixture hash mismatch')
            if prior['normalization']!=manifest['normalization'] or prior['preprocessing']!=manifest['preprocessing']:
                raise ValueError('Changed preprocessing/normalization would invalidate this comparison')
            for key in ['images','states','actions','action_masks','img_masks']:
                if prior['arrays'][key]['shape'][1:]!=manifest['arrays'][key]['shape'][1:]:
                    raise ValueError('Changed tensor geometry')
            if {r['subtask'] for r in prior['records']}!={r['subtask'] for r in records}:
                raise ValueError('Changed label contract')
            report['verified_evaluation_fixture_change']=dict(prior_manifest_sha256=state['fixture_manifest_sha256'],
                new_manifest_sha256=report['fixture_manifest_sha256'],normalization_identical=True)
        missing,unexpected=model.load_state_dict(state['model_delta'],strict=False)
        if unexpected or any(not n.startswith(('vision_backbone.','llm_expert.embed_tokens.')) for n in missing):
            raise ValueError(f'Unexpected partial checkpoint mapping: {missing}, {unexpected}')
        report['loaded_delta']=args.delta
        del state
    model.requires_grad_(True);model.vision_backbone.requires_grad_(False)
    model.llm_expert.embed_tokens.requires_grad_(False)
    parameters=[p for p in model.parameters() if p.requires_grad]
    report['trainable_parameters']=sum(p.numel() for p in parameters)
    prompts=PrepareSubtaskPrompts(dict(type='PretrainedTokenizer',model_path=str(assets)),
                                 max_len=200,model_vocab_size=257152)
    import sentencepiece as spm
    sp=spm.SentencePieceProcessor(model_file=str(assets/'tokenizer.model'))
    def row_tensor(x):return torch.from_numpy(np.asarray(x)).unsqueeze(0).cuda()
    def prompt_row(i,condition='normal'):
        states=arrays['states'][i:i+1]
        if condition in ('zero_state','text_only'):states=torch.zeros_like(states)
        state_np=states[0].cpu().numpy()
        high=prompts.high_level(records[i]['overall_task'],state_np,records[i]['subtask'])
        context=prompts.high_level(records[i]['overall_task'],state_np)
        image=arrays['images'][i:i+1];img_masks=arrays['img_masks'][i:i+1]
        if condition in ('no_images','text_only'):
            image=torch.zeros_like(image);img_masks=torch.zeros_like(img_masks)
        elif condition=='shuffled_images':
            j=validation[(validation.index(i)+3)%len(validation)]
            image=arrays['images'][j:j+1]
        return states,image,img_masks,high,context
    # Extend the prior fixture's tokenizer check to every real condition and label.
    for i in train+validation:
        state=arrays['states'][i].cpu().numpy()
        for segment in [prompts.condition(records[i]['overall_task'],state,True),
                        prompts.condition(records[i]['subtask'],state),records[i]['subtask']]:
            if prompts._encode(segment)!=sp.encode(segment):raise ValueError('Real-data tokenizer mismatch')
    report['all_real_prompt_segments_match_sentencepiece']=True
    noise_generator=torch.Generator(device='cuda').manual_seed(1703)
    eval_noise=torch.randn((len(records),50,32),device='cuda',generator=noise_generator)
    torch.save(eval_noise.cpu(),out/'eval_noise.pt')

    @torch.no_grad()
    def evaluate(indices,condition='normal',with_flow=False):
        model.eval();results=[]
        for i in indices:
            states,images,img_masks,high,context=prompt_row(i,condition)
            start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                ce=model.forward_high_level(images,row_tensor(high['input_ids']),row_tensor(high['valid_mask']),
                    row_tensor(high['ar_mask']),row_tensor(high['labels']),img_masks)['loss'].item()
                generated=model.generate_subtask(images,row_tensor(context['input_ids']),
                    row_tensor(context['valid_mask']),img_masks,eos_token_id=prompts.eos_id,
                    pad_token_id=prompts.pad_id,max_new_tokens=16)
                text=prompts.decode(generated['token_ids'][0].cpu().tolist())
                row=dict(sample_id=records[i]['sample_id'],stage=records[i]['stage'],target=records[i]['subtask'],
                    predicted=text,eos=bool(generated['finished'][0]),ce=ce,
                    token_ids=generated['token_ids'][0].cpu().tolist())
                row['exact']=row['eos'] and text==row['target']
                if with_flow:
                    low=prompts.low_level(row['target'],states[0].cpu().numpy())
                    common=dict(images=images,states=states,actions=arrays['actions'][i:i+1],img_masks=img_masks,
                        action_masks=arrays['action_masks'][i:i+1],noise=eval_noise[i:i+1],time=torch.full((1,),.5,device='cuda'))
                    row['gt_flow_loss']=model.forward_low_level(ll_input_ids=row_tensor(low['input_ids']),
                        ll_valid_mask=row_tensor(low['valid_mask']),**common)['loss'].item()
            torch.cuda.synchronize();row['seconds']=time.perf_counter()-start;results.append(row)
        by_episode={}
        for episode in sorted({records[i]['episode'] for i in indices}):
            ids={records[i]['sample_id'] for i in indices if records[i]['episode']==episode}
            selected=[r for r in results if r['sample_id'] in ids]
            by_episode[str(episode)]=dict(count=len(selected),exact=sum(r['exact'] for r in selected),
                                          eos=sum(r['eos'] for r in selected))
        return dict(condition=condition,count=len(results),exact=sum(r['exact'] for r in results),
                    eos=sum(r['eos'] for r in results),mean_ce=float(np.mean([r['ce'] for r in results])),
                    per_episode=by_episode,rows=results)

    report['before']={split:evaluate(indices,with_flow=True) for split,indices in [('train',train),('validation',validation)]}
    print(json.dumps({'before':{k:{x:v[x] for x in ['count','exact','eos','mean_ce']} for k,v in report['before'].items()}}),flush=True)
    (out/'report.partial.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    report['training']=[]
    if args.steps:
        optimizer=torch.optim.AdamW(parameters,lr=1e-5,weight_decay=0.)
        classes=sorted({records[i]['stage'] for i in train})
        pools=[[i for i in train if records[i]['stage']==stage] for stage in classes]
        time_dist=torch.distributions.Beta(torch.tensor(1.5,device='cuda'),torch.tensor(1.,device='cuda'))
        train_start=time.perf_counter()
        for step in range(1,args.steps+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            totals=dict(loss=0.,loss_hl=0.,loss_action=0.)
            selected=[]
            for pool in pools:
                i=int(rng.choice(pool));selected.append(records[i]['sample_id'])
                states,images,img_masks,high,_=prompt_row(i)
                low=prompts.low_level(records[i]['subtask'],states[0].cpu().numpy())
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    result=model(images=images,states=states,img_masks=img_masks,
                        hl_input_ids=row_tensor(high['input_ids']),hl_valid_mask=row_tensor(high['valid_mask']),
                        hl_ar_mask=row_tensor(high['ar_mask']),hl_labels=row_tensor(high['labels']),
                        ll_input_ids=row_tensor(low['input_ids']),ll_valid_mask=row_tensor(low['valid_mask']),
                        actions=arrays['actions'][i:i+1],action_masks=arrays['action_masks'][i:i+1],
                        noise=torch.randn_like(arrays['actions'][i:i+1]),time=time_dist.sample((1,))*.999+.001)
                    loss=result['loss']/len(pools)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward()
                for key in totals:totals[key]+=result[key].item()/len(pools)
                del result,loss
            grad_norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(grad_norm):raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            if step==1 or step%10==0 or step==args.steps:
                row=dict(step=step,**totals,grad_norm=grad_norm.item(),samples=selected)
                report['training'].append(row);print(json.dumps(row),flush=True)
                (out/'report.partial.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        torch.cuda.synchronize();report['training_seconds']=time.perf_counter()-train_start
        # No claim of optimizer/RNG resume: this small-run artifact is a model delta.
        del optimizer
        model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    report['after']={split:evaluate(indices,with_flow=True) for split,indices in [('train',train),('validation',validation)]}
    print(json.dumps({'after':{k:{x:v[x] for x in ['count','exact','eos','mean_ce']} for k,v in report['after'].items()}}),flush=True)
    report['ablations']={mode:evaluate(validation,condition=mode) for mode in ['no_images','shuffled_images','zero_state','text_only']}

    @torch.no_grad()
    def action_comparisons():
        results=[];model.eval()
        for i,predicted in zip(validation,report['after']['validation']['rows']):
            states=arrays['states'][i:i+1];images=arrays['images'][i:i+1];img_masks=arrays['img_masks'][i:i+1]
            conditions={'gt':records[i]['subtask'],'overall_goal_baseline':records[i]['overall_task']}
            if predicted['eos'] and predicted['predicted']:
                conditions['predicted']=predicted['predicted']
            row=dict(sample_id=records[i]['sample_id'],target=records[i]['subtask'],predicted=predicted['predicted'],
                     eos=predicted['eos'],conditions={},comparison_claim='offline normalized MSE only; not robot success')
            sampled={}
            for name,text in conditions.items():
                prompt=prompts.low_level(text,states[0].cpu().numpy())
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=model.forward_low_level(images,states,row_tensor(prompt['input_ids']),row_tensor(prompt['valid_mask']),
                        arrays['actions'][i:i+1],img_masks,action_masks=arrays['action_masks'][i:i+1],
                        noise=eval_noise[i:i+1],time=torch.full((1,),.5,device='cuda'))['loss'].item()
                    generated_actions=model.sample_actions_given_subtask(images,states,row_tensor(prompt['input_ids']),
                        row_tensor(prompt['valid_mask']),img_masks,noise=eval_noise[i:i+1],num_steps=10)
                if not torch.isfinite(generated_actions).all():raise RuntimeError('Nonfinite action')
                sampled[name]=generated_actions
                mse=(generated_actions-arrays['actions'][i:i+1,:,:14]).square().mean().item()
                row['conditions'][name]=dict(actual_condition=text,flow_loss=loss,sampled_normalized_mse=mse,
                                             shape=list(generated_actions.shape))
            if 'predicted' in sampled:
                row['predicted_vs_gt_max_action_difference']=(sampled['predicted']-sampled['gt']).abs().max().item()
            results.append(row)
        return results

    report['action_comparisons']=action_comparisons()
    # Persist evaluation before a potentially slow NFS checkpoint write.
    (out/'report.partial.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if args.steps:
        save_start=time.perf_counter()
        delta={name:p.detach().cpu() for name,p in model.named_parameters() if p.requires_grad}
        checkpoint=out/'model_delta.pt'
        torch.save(dict(model_delta=delta,fixture_manifest_sha256=report['fixture_manifest_sha256'],
                        contract=report['contract'],base_asset=str(assets)),checkpoint)
        del delta
        report['checkpoint']=dict(path=str(checkpoint),bytes=checkpoint.stat().st_size,
            save_seconds=time.perf_counter()-save_start,scope='all changed model parameters, requires exact base; no optimizer/RNG state')
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'complete':str(out/'report.json'),'ablations':{k:(v['exact'],v['count']) for k,v in report['ablations'].items()}}),flush=True)


if __name__=='__main__':main()
