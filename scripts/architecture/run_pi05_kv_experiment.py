"""Controlled architecture experiments, not a robot task or a success benchmark."""
import argparse
from contextlib import nullcontext
import hashlib
import itertools
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


def fixtures():
    """All three binary inputs affect text; state and subtask affect rule actions."""
    rows = []
    for image, state, goal in itertools.product(range(2), repeat=3):
        label = image ^ state ^ goal
        pixels = np.zeros((9,224,224), dtype=np.float32)
        pixels.fill(-.7)
        for camera in range(3):
            pixels[3*camera+image] = .7
            pixels[3*camera+2] = np.linspace(-.3,.3,224)[None,:]
        states = np.zeros(32,dtype=np.float32)
        states[0] = -.5 if state == 0 else .5
        states[1] = -states[0]
        target = 'move left' if label == 0 else 'move right'
        actions = rule_actions(states, label)
        rows.append(dict(id=f'i{image}s{state}g{goal}',image_bit=image,state_bit=state,
                         goal_bit=goal,subtask=target,
                         goal='apply the normal rule' if goal == 0 else 'apply the reversed rule',
                         split='validation' if (image,state,goal) in [(0,0,0),(1,1,1)] else 'train',
                         images=pixels,states=states,actions=actions))
    return rows


def rule_actions(states, label):
    temporal = np.linspace(1,.5,50,dtype=np.float32)[:,None]
    dims = np.linspace(.5,1,32,dtype=np.float32)[None,:]
    return .15*states[None,:] + (-1 if label == 0 else 1)*.25*temporal*dims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['correctness','baseline','cached','insulated'],required=True)
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--save-delta',action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(1704)
    torch.cuda.manual_seed_all(1704)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = fixtures()
    metadata = [{k:v for k,v in row.items() if k not in ['images','states','actions']} for row in rows]
    arrays = {k:np.stack([row[k] for row in rows]) for k in ['images','states','actions']}
    np.savez_compressed(out/'fixtures.npz',**arrays)
    report = dict(mode=args.mode,source='AI controlled architecture design and local measurements',
                  reference_commit='Ke-Wang1017/openpi_subtask@385d0e8330397ecfa29e9b675042b424c4d5386c',
                  limitation='Procedural color/state/XOR prompts and deterministic vectors; no robot semantics',
                  fixtures=metadata,fixture_sha256=hashlib.sha256((out/'fixtures.npz').read_bytes()).hexdigest(),
                  array_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},
                  contract=dict(seed=1704,steps=args.steps,microbatch=1,accumulation=2,
                                objective='CE + flow MSE, weights 1:1; 32 real architecture dimensions',
                                optimizer='AdamW lr1e-5 betas .9/.999 eps1e-8 WD0 clip1',
                                normalization='procedural values already normalized; no dataset fitting',
                                precision='FP32 main params + inherited norm dtypes; BF16 training/flow, FP32 text generation including vision; TF32 disabled',
                                training_time='NumPy true Beta(1.5,1)*.999+.001; seed1706',
                                noise_seed=1705,eval_noise_seed=1707,rtc=False,
                                low_context='observation + subtask only; fresh cache per call'),
                  tolerances=dict(fp32=dict(rtol=2e-4,atol=5e-5),
                                  bf16=dict(rtol=.03,atol=.125),gradient_sample_relative_l2=.03))
    def save():
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    save()
    cached = args.mode in ['cached','insulated']
    cfg = runpy.run_path(str(Path(__file__).resolve().parents[2]/'configs/pi05_subtask_architecture.py'))['model']
    cfg.update(pretrained_name_or_path=str(Path(args.assets)/'model.safetensors'),
               text_use_cache=cached,action_prefix_cache=cached,flow_prefix_cache=cached,
               stop_gradient_flow_to_prefix=args.mode=='insulated')
    print('load_start',args.mode,flush=True)
    model = build_vla_from_cfg(cfg)
    model.from_pretrained()
    model = model.cuda().eval()
    prompts = PrepareSubtaskPrompts(dict(type='PretrainedTokenizer',model_path=args.assets),
                                   max_len=200,model_vocab_size=257152)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(Path(args.assets)/'tokenizer.model'))
    for row in rows:
        for text in [prompts.condition(row['goal'],row['states'],True),
                     prompts.condition(row['subtask'],row['states']),row['subtask']]:
            assert prompts._encode(text)==sp.encode(text)
    report['sentencepiece_segments_identical'] = True
    report['parameters'] = sum(p.numel() for p in model.parameters())
    report['model_options'] = {name:getattr(model,name) for name in [
        'text_use_cache','action_prefix_cache','flow_prefix_cache',
        'stop_gradient_flow_to_prefix','text_decode_fp32']}
    report['parameter_dtype_numel'] = {}
    for p in model.parameters():
        dtype = str(p.dtype)
        report['parameter_dtype_numel'][dtype] = report['parameter_dtype_numel'].get(dtype,0)+p.numel()
    data = {k:torch.from_numpy(v).cuda() for k,v in arrays.items()}
    eval_noise = torch.randn(8,50,32,device='cuda',generator=torch.Generator(device='cuda').manual_seed(1707))
    img_masks = torch.tensor([[True,False,True]],device='cuda')
    def tensor(x):
        return torch.from_numpy(np.asarray(x)).unsqueeze(0).cuda()
    def inputs(i):
        row = rows[i]
        high = prompts.high_level(row['goal'],row['states'],row['subtask'])
        context = prompts.high_level(row['goal'],row['states'])
        low = prompts.low_level(row['subtask'],row['states'])
        common = dict(images=data['images'][i:i+1],states=data['states'][i:i+1],
                      ll_input_ids=tensor(low['input_ids']),ll_valid_mask=tensor(low['valid_mask']),
                      actions=data['actions'][i:i+1],img_masks=img_masks,
                      noise=eval_noise[i:i+1],time=torch.tensor([.5],device='cuda'))
        return high,context,common
    def high_args(high, common):
        return dict(images=common['images'],hl_input_ids=tensor(high['input_ids']),
                    hl_valid_mask=tensor(high['valid_mask']),hl_ar_mask=tensor(high['ar_mask']),
                    hl_labels=tensor(high['labels']),img_masks=img_masks)
    if args.mode=='correctness':
        report['checks'] = []
        high,context,common = inputs(0)
        for precision in ['fp32','bf16']:
            autocast = torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
            with torch.no_grad(),autocast:
                gen_args = dict(images=common['images'],hl_input_ids=tensor(context['input_ids']),
                                hl_valid_mask=tensor(context['valid_mask']),img_masks=img_masks,
                                eos_token_id=-1,pad_token_id=prompts.pad_id,max_new_tokens=8,return_logits=True)
                a = model.generate_subtask(**gen_args,use_cache=False)
                b = model.generate_subtask(**gen_args,use_cache=True)
                tol = report['tolerances'][precision]
                torch.testing.assert_close(a['logits'],b['logits'],**tol)
                assert torch.equal(a['token_ids'],b['token_ids'])
                flow_a = model.forward_low_level(**common,use_prefix_cache=False)
                flow_b = model.forward_low_level(**common,use_prefix_cache=True)
                torch.testing.assert_close(flow_a['predictions'],flow_b['predictions'],**tol)
                action_args = {k:v for k,v in common.items() if k not in ['actions','time']}
                sa = model.sample_actions_given_subtask(**action_args,use_cache=False)
                sb = model.sample_actions_given_subtask(**action_args,use_cache=True,return_trace=True)
                torch.testing.assert_close(sa,sb['actions'],**tol)
                assert len(set(sb['cache_lengths']))==1
                report['checks'].append(dict(precision=precision,
                    text_actual_precision='FP32; BF16 autocast explicitly disabled for text generation',
                    text_logits_max_difference=(a['logits']-b['logits']).abs().max().item(),tokens_equal=True,
                    flow_prediction_max_difference=(flow_a['predictions']-flow_b['predictions']).abs().max().item(),
                    sampled_actions_max_difference=(sa-sb['actions']).abs().max().item(),
                    text_cache=b['cache_info'],action_cache=sb['cache_info'],action_cache_lengths=sb['cache_lengths']))
            save()
        # Full model gradient coverage, with up to 1024 deterministic elements per parameter for numerical comparison.
        model.requires_grad_(True)
        model.llm_expert.embed_tokens.requires_grad_(False)
        def gradients():
            stats, samples = {}, {}
            for name,p in model.named_parameters():
                g = p.grad
                if g is None:
                    stats[name] = dict(present=False,norm=0.,finite=True)
                else:
                    flat = g.detach().float().reshape(-1)
                    stats[name] = dict(present=True,norm=flat.norm().item(),finite=bool(torch.isfinite(flat).all()))
                    samples[name] = flat[::max(1,(flat.numel()+1023)//1024)].cpu()
            assert all(x['finite'] for x in stats.values())
            return stats,samples
        report['gradient_checks'] = {}
        snapshots = {}
        for mode in ['baseline','cached','insulated','ce_after_insulation']:
            model.zero_grad(set_to_none=True)
            model.stop_gradient_flow_to_prefix=mode in ['insulated','ce_after_insulation']
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result = (model.forward_high_level(**high_args(high,common)) if mode=='ce_after_insulation'
                          else model.forward_low_level(**common,use_prefix_cache=mode!='baseline'))
            result['loss'].backward()
            stats,samples = gradients()
            report['gradient_checks'][mode] = dict(loss=result['loss'].item(),parameters=stats)
            snapshots[mode] = samples
            del result
            save()
        reference = snapshots['baseline']
        squared_error, squared_reference = 0.,0.
        for name,g in reference.items():
            other = snapshots['cached'].get(name,torch.zeros_like(g))
            squared_error += (g-other).square().sum().item()
            squared_reference += g.square().sum().item()
        relative = (squared_error/max(squared_reference,1e-30))**.5
        report['cached_vs_baseline_gradient_sample_relative_l2'] = relative
        assert relative<=report['tolerances']['gradient_sample_relative_l2']
        isolated = report['gradient_checks']['insulated']['parameters']
        assert all(v['norm']==0 for n,v in isolated.items() if n.startswith(('vision_backbone.','projector.','llm_backbone.')))
        assert any(v['norm']>0 for n,v in isolated.items() if n.startswith('llm_expert.'))
        ce = report['gradient_checks']['ce_after_insulation']['parameters']
        assert any(v['norm']>0 for n,v in ce.items() if n.startswith('llm_backbone.'))
        assert any(v['norm']>0 for n,v in ce.items() if n.startswith('vision_backbone.'))
        report['passed'] = True
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        save()
        print('CORRECTNESS_PASS',flush=True)
        return

    @torch.no_grad()
    def evaluate():
        model.eval()
        results = []
        for i,row in enumerate(rows):
            high,context,common = inputs(i)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                ce = model.forward_high_level(**high_args(high,common))['loss'].item()
                gen = model.generate_subtask(common['images'],tensor(context['input_ids']),
                    tensor(context['valid_mask']),img_masks,prompts.eos_id,prompts.pad_id,max_new_tokens=8)
                text = prompts.decode(gen['token_ids'][0].cpu().tolist())
                flow = model.forward_low_level(**common)['loss'].item()
                action_args = {k:v for k,v in common.items() if k not in ['actions','time']}
                actions = model.sample_actions_given_subtask(**action_args)
                flipped = prompts.low_level('move right' if row['subtask']=='move left' else 'move left',row['states'])
                other = model.sample_actions_given_subtask(**{**action_args,'ll_input_ids':tensor(flipped['input_ids']),
                    'll_valid_mask':tensor(flipped['valid_mask'])})
            results.append(dict(id=row['id'],split=row['split'],target=row['subtask'],predicted=text,
                                tokens=gen['token_ids'][0].cpu().tolist(),eos=bool(gen['finished'][0]),
                                exact=bool(gen['finished'][0]) and text==row['subtask'],ce=ce,flow=flow,
                                sampled_mse=(actions-common['actions']).square().mean().item(),
                                switched_subtask_action_rms=(actions-other).square().mean().sqrt().item(),
                                switched_subtask_target_mse=(other-common['actions']).square().mean().item()))
        return results
    report['before'] = evaluate()
    save()
    model.requires_grad_(True)
    model.vision_backbone.requires_grad_(False)
    model.llm_expert.embed_tokens.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    report['trainable_parameters'] = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(params,lr=1e-5,weight_decay=0.)
    pools = [[i for i,row in enumerate(rows) if row['split']=='train' and row['subtask']==label]
             for label in ['move left','move right']]
    sample_rng = np.random.default_rng(1704)
    time_samples = np.random.default_rng(1706).beta(1.5,1,(args.steps,2))*.999+.001
    report['training_time_sha256'] = hashlib.sha256(time_samples.tobytes()).hexdigest()
    noise_rng = torch.Generator(device='cuda').manual_seed(1705)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    report['training'] = []
    for step in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = dict(loss=0.,loss_hl=0.,loss_action=0.)
        selected = []
        for category,pool in enumerate(pools):
            i = int(sample_rng.choice(pool))
            selected.append(rows[i]['id'])
            high,_,common = inputs(i)
            common.update(noise=torch.randn(common['actions'].shape,device='cuda',generator=noise_rng),
                          time=torch.tensor([time_samples[step,category]],device='cuda',dtype=torch.float32))
            high_kw = high_args(high,common)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result = model(**{**common,**high_kw})
                loss = result['loss']/2
            assert torch.isfinite(loss)
            loss.backward()
            for key in totals:
                totals[key] += result[key].item()/2
            del result,loss
        norm = torch.nn.utils.clip_grad_norm_(params,1.)
        assert torch.isfinite(norm)
        optimizer.step()
        report['training'].append(dict(step=step+1,**totals,grad_norm=norm.item(),samples=selected))
        if step==0 or (step+1)%10==0:
            print(json.dumps(report['training'][-1]),flush=True)
            save()
    torch.cuda.synchronize()
    report['training_seconds'] = time.perf_counter()-started
    report['training_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['training_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    del optimizer
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    report['after'] = evaluate()
    save()
    # Warm-up separately, then profile a fixed eight-token budget (EOS stopping disabled only here).
    _,context,common = inputs(0)
    model.eval()
    report['latency'] = []
    for iteration in range(8):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            torch.cuda.synchronize()
            start = time.perf_counter()
            generated = model.generate_subtask(common['images'],tensor(context['input_ids']),
                tensor(context['valid_mask']),img_masks,-1,prompts.pad_id,max_new_tokens=8,profile=True)
            text_seconds = time.perf_counter()-start
            action = model.sample_actions_given_subtask(**{k:v for k,v in common.items() if k not in ['actions','time']},
                                                       return_trace=True,profile=True)
        report['latency'].append(dict(warmup=iteration<2,text_seconds=text_seconds,text_step_ms=generated['step_ms'],
                                     text_cache=generated['cache_info'],action_prefix_ms=action['prefix_ms'],
                                     action_denoise_ms=action['denoise_ms'],action_cache=action['cache_info'],
                                     action_cache_lengths=action['cache_lengths']))
    if args.save_delta:
        save_started = time.perf_counter()
        delta = {n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
        torch.save(dict(model_delta=delta,contract=report['contract'],mode=args.mode,
                        base_asset=str(Path(args.assets)),array_sha256=report['array_sha256']),out/'model_delta.pt')
        del delta
        report['checkpoint_save_seconds'] = time.perf_counter()-save_started
        report['checkpoint_bytes'] = (out/'model_delta.pt').stat().st_size
    report['complete'] = True
    report['checkpoint_saved'] = args.save_delta
    report['checkpoint_note'] = 'Model-only delta when requested; no optimizer/RNG resume claim. Frozen source and inputs saved.'
    save()
    print('COMPLETE',args.mode,flush=True)


if __name__=='__main__':
    main()
