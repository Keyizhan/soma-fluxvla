"""Fresh-process reload, trained-cache alignment, and generated-condition replay."""
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
from fluxvla.models.vlas.subtask_runtime import SubtaskRuntime
from fluxvla.transforms.subtask_prompts import PrepareSubtaskPrompts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--assets',required=True)
    args=parser.parse_args()
    root=Path(args.root)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    config=runpy.run_path(str(Path(__file__).resolve().parents[2]/'configs/pi05_subtask_architecture.py'))['model']
    config['pretrained_name_or_path']=str(Path(args.assets)/'model.safetensors')
    model=build_vla_from_cfg(config)
    model.from_pretrained()
    model=model.cuda().eval().requires_grad_(False)
    prompts=PrepareSubtaskPrompts(dict(type='PretrainedTokenizer',model_path=args.assets),
                                 max_len=200,model_vocab_size=257152)
    rule_actions=runpy.run_path(str(Path(__file__).with_name('run_pi05_kv_experiment.py')))['rule_actions']
    result=dict(source='AI fresh-process measurements; procedural architecture inputs only',
                raw_training_action_tensors_saved=False,groups={})
    def save():
        (root/'reload_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    def tensor(x):return torch.from_numpy(np.asarray(x)).unsqueeze(0).cuda()
    for mode in ['baseline','cached','insulated']:
        run=root/'runs'/mode
        expected=json.loads((run/'report.json').read_text())
        assert expected['complete'] and expected['checkpoint_saved']
        digest=hashlib.sha256()
        with (run/'model_delta.pt').open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
        state=torch.load(run/'model_delta.pt',map_location='cpu',weights_only=True)
        assert state['array_sha256']==expected['array_sha256']
        missing,unexpected=model.load_state_dict(state['model_delta'],strict=False)
        assert not unexpected
        assert all(n.startswith(('vision_backbone.','llm_expert.embed_tokens.')) for n in missing)
        del state
        for name,value in expected['model_options'].items():setattr(model,name,value)
        data=np.load(run/'fixtures.npz',allow_pickle=False)
        for key in data.files:
            assert hashlib.sha256(data[key].tobytes()).hexdigest()==expected['array_sha256'][key]
        noise=torch.randn(8,50,32,device='cuda',generator=torch.Generator(device='cuda').manual_seed(1707))
        runtime=SubtaskRuntime(model,prompts,max_new_tokens=8,num_steps=10)
        group=dict(checkpoint_sha256=digest.hexdigest(),rows=[])
        masks=torch.tensor([[True,False,True]],device='cuda')
        with torch.no_grad():
            for i,(row,prior) in enumerate(zip(expected['fixtures'],expected['after'])):
                images=tensor(data['images'][i]);states=tensor(data['states'][i])
                context=prompts.high_level(row['goal'],data['states'][i])
                common=dict(images=images,hl_input_ids=tensor(context['input_ids']),
                            hl_valid_mask=tensor(context['valid_mask']),img_masks=masks,
                            eos_token_id=prompts.eos_id,pad_token_id=prompts.pad_id,max_new_tokens=8,return_logits=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    a=model.generate_subtask(**common,use_cache=False)
                    b=model.generate_subtask(**common,use_cache=True)
                    trained=b if model.text_use_cache else a
                    assert trained['token_ids'][0].cpu().tolist()==prior['tokens']
                    assert bool(trained['finished'][0])==prior['eos']
                    assert torch.equal(a['token_ids'],b['token_ids'])
                    torch.testing.assert_close(a['logits'],b['logits'],rtol=2e-4,atol=5e-5)
                    low=prompts.low_level(row['subtask'],data['states'][i])
                    action_args=dict(images=images,states=states,ll_input_ids=tensor(low['input_ids']),
                                     ll_valid_mask=tensor(low['valid_mask']),img_masks=masks,noise=noise[i:i+1])
                    ua=model.sample_actions_given_subtask(**action_args,use_cache=False)
                    ca=model.sample_actions_given_subtask(**action_args,use_cache=True)
                    torch.testing.assert_close(ua,ca,rtol=.03,atol=.125)
                    mse=((ca if model.action_prefix_cache else ua)-tensor(data['actions'][i])).square().mean().item()
                    assert abs(mse-prior['sampled_mse'])<1e-6
                    flip_text='move right' if row['subtask']=='move left' else 'move left'
                    flip=prompts.low_level(flip_text,data['states'][i])
                    switched=model.sample_actions_given_subtask(**{**action_args,
                        'll_input_ids':tensor(flip['input_ids']),'ll_valid_mask':tensor(flip['valid_mask'])},use_cache=True)
                    flip_target=tensor(rule_actions(data['states'][i],int(flip_text=='move right')))
                    replay=None
                    if prior['eos'] and prior['predicted']:
                        replay=runtime.infer(images,states,[row['goal']],masks,observation_version=i,
                                             noise=noise[i:i+1])
                        assert replay['trace']['generation_calls']==1
                        assert replay['subtasks'][0]==prior['predicted']
                group['rows'].append(dict(id=row['id'],split=row['split'],reload_tokens_identical=True,
                    cache_tokens_identical=True,text_logits_max_difference=(a['logits']-b['logits']).abs().max().item(),
                    cached_uncached_action_max_difference=(ua-ca).abs().max().item(),
                    reloaded_gt_action_mse=mse,training_report_mse_difference=abs(mse-prior['sampled_mse']),
                    switched_condition_correct_target_mse=(switched-flip_target).square().mean().item(),
                    switched_condition_original_target_mse=(switched-tensor(data['actions'][i])).square().mean().item(),
                    switched_action_rms=(ca-switched).square().mean().sqrt().item(),
                    generated_condition_action_mse=None if replay is None else (replay['actions']-tensor(data['actions'][i])).square().mean().item(),
                    generated_subtask=prior['predicted'],trace=None if replay is None else replay['trace']))
        # Training jobs have all exited before this verifier starts. Compare
        # inference sequentially on the same GPU, with no competing GPU job.
        first=expected['fixtures'][0]
        images=tensor(data['images'][0]);states=tensor(data['states'][0])
        context=prompts.high_level(first['goal'],data['states'][0])
        low=prompts.low_level(first['subtask'],data['states'][0])
        group['isolated_latency']=[]
        torch.cuda.reset_peak_memory_stats()
        for iteration in range(8):
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                torch.cuda.synchronize()
                started=time.perf_counter()
                generated=model.generate_subtask(images,tensor(context['input_ids']),tensor(context['valid_mask']),
                    masks,-1,prompts.pad_id,max_new_tokens=8,profile=True)
                seconds=time.perf_counter()-started
                actions=model.sample_actions_given_subtask(images,states,tensor(low['input_ids']),
                    tensor(low['valid_mask']),masks,noise=noise[0:1],return_trace=True,profile=True)
            group['isolated_latency'].append(dict(warmup=iteration<2,text_seconds=seconds,
                text_step_ms=generated['step_ms'],text_cache=generated['cache_info'],
                action_prefix_ms=actions['prefix_ms'],action_denoise_ms=actions['denoise_ms'],
                action_cache=actions['cache_info'],action_cache_lengths=actions['cache_lengths']))
        group['inference_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        data.close()
        result['groups'][mode]=group
        save()
        print('RELOAD_PASS',mode,flush=True)
    result['passed']=True
    save()


if __name__=='__main__':main()
