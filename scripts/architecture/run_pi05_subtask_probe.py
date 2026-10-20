"""E0/E2 architecture probe. Fixed procedural images; no robotics-performance claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import time

import numpy as np
import torch
from safetensors import safe_open

from fluxvla import register_training_components
register_training_components()
from fluxvla.engines import build_vla_from_cfg
from fluxvla.models.vlas.pi05_flowmatching import PI05FlowMatching
from fluxvla.transforms.subtask_prompts import PrepareSubtaskPrompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--language-delta', help='Reload a prior E2 language checkpoint into a fresh base model')
    parser.add_argument('--replay', action='store_true', help='Run synchronous generation-to-action observation replay')
    args = parser.parse_args()
    out = Path(args.output);out.mkdir(parents=True, exist_ok=True)
    assets = Path(args.assets)
    torch.set_num_threads(4)
    torch.manual_seed(1701);torch.cuda.manual_seed_all(1701)
    config = runpy.run_path(str(Path(__file__).resolve().parents[2]/'configs/pi05_subtask_architecture.py'))['model']
    config.update(type='PI05Subtask', pretrained_name_or_path=str(assets/'model.safetensors'),
                  rtc_training_config=None, ori_action_dim=32,
                  hl_loss_weight=1.0, action_loss_weight=1.0,
                  text_use_cache=False, action_prefix_cache=False, flow_prefix_cache=False,
                  stop_gradient_flow_to_prefix=False, text_decode_fp32=False)
    # Predeclare numerical tolerances before running.
    report = dict(seed=1701, fixture='procedural-rgb-v1; identity text; no real robot data',
                  torch=torch.__version__, tolerances={'rtol':1e-4,'atol':1e-5},
                  rtc='disabled in both baseline and extension', assets=str(assets),
                  optimization={'objective':'CE only for E2', 'lr':1e-5, 'optimizer':'AdamW',
                                'weight_decay':0.0,'clip_grad_norm':1.0,
                                'trainable':'language backbone only','steps_requested':args.steps})
    print('load_start', flush=True)
    model = build_vla_from_cfg(config)
    model.from_pretrained()
    report['parameter_count'] = sum(p.numel() for p in model.parameters())
    with safe_open(assets/'model.safetensors',framework='pt',device='cpu') as weights:
        key='paligemma_with_expert.paligemma.lm_head.weight'
        source=weights.get_tensor(key)
        report['head_shape']=list(source.shape)
        report['head_loaded_exact']=torch.equal(source,model.llm_backbone.embed_tokens.weight.detach())
        if not report['head_loaded_exact']:raise RuntimeError('LM table differs from source')
        del source
    report['small_asset_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in assets.iterdir() if p.is_file() and p.stat().st_size<25_000_000}
    prompts=PrepareSubtaskPrompts(dict(type='PretrainedTokenizer',model_path=str(assets)),
                                  max_len=200,model_vocab_size=report['head_shape'][0])
    report['special_ids']={'bos':prompts.bos_id,'eos':prompts.eos_id,'pad':prompts.pad_id}
    report['tokenizer_vocab']=len(prompts.tokenizer)
    vocab=prompts.tokenizer.tokenizer.get_vocab()
    outside={token:idx for token,idx in vocab.items() if idx>=report['head_shape'][0]}
    report['tokenizer_added_outside_model']=outside
    if prompts.tokenizer.tokenizer.vocab_size != report['head_shape'][0]:
        raise RuntimeError('Base vocabulary mismatch')
    if outside != {'<image>':257152}:
        raise RuntimeError('Unexpected added tokens require explicit investigation')
    import sentencepiece as spm
    sp=spm.SentencePieceProcessor(model_file=str(assets/'tokenizer.model'))
    mismatches=[dict(id=i,hf=prompts.tokenizer.tokenizer.convert_ids_to_tokens(i),sp=sp.id_to_piece(i))
                for i in range(sp.vocab_size())
                if prompts.tokenizer.tokenizer.convert_ids_to_tokens(i)!=sp.id_to_piece(i)]
    report['sentencepiece_vocab_size']=sp.vocab_size()
    report['sentencepiece_id_piece_mismatches']=mismatches
    # This asset spells the tab byte as a literal tab in HF, and <0x09> in SP.
    if mismatches != [dict(id=226,hf='\t',sp='<0x09>')]:
        raise RuntimeError('Unexplained vocabulary order difference from SentencePiece asset')
    task='move the object'
    state=np.zeros(32,dtype=np.float32)
    high=prompts.high_level(task,state,task)
    low=prompts.low_level(task,state)
    context=prompts.high_level(task,state)
    segments=[prompts.condition(task,state,True),prompts.condition(task,state),task]
    report['segment_tokenizer_equivalence']=[prompts._encode(s)==sp.encode(s) for s in segments]
    if not all(report['segment_tokenizer_equivalence']):raise RuntimeError('Tokenizer segment mismatch')
    report['prompt_contract']={'version':prompts.version,'high_ids':high['input_ids'].tolist(),
        'high_valid':high['valid_mask'].tolist(),'high_labels':high['labels'].tolist(),
        'low_ids':low['input_ids'].tolist(),'target':task}
    def tensor(x):return torch.from_numpy(np.asarray(x)).unsqueeze(0).cuda()
    row=torch.linspace(-1,1,224)
    image=torch.stack([row[None,:].expand(224,224),row[:,None].expand(224,224),torch.zeros(224,224)])
    images=torch.cat([image,image.flip(-1),image.flip(-2)])[None].cuda()
    img_masks=torch.tensor([[True,False,True]],device='cuda')
    states=tensor(state)
    model=model.cuda().eval()
    action=torch.randn(1,50,32,device='cuda')
    noise=torch.randn_like(action);time_value=torch.tensor([.5],device='cuda')
    ids,valid=tensor(low['input_ids']),tensor(low['valid_mask'])
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        legacy=PI05FlowMatching.forward(model,images=images,states=states,lang_tokens=ids,
            lang_masks=valid,actions=action,img_masks=img_masks,noise=noise,time=time_value)
        extension=model.forward_low_level(images,states,ids,valid,action,img_masks,noise=noise,time=time_value)
        delta=(legacy['predictions']-extension['predictions']).abs().max().item()
        report['E0']={'max_action_prediction_difference':delta,
                      'legacy_loss':legacy['loss'].item(),'extension_loss':extension['loss'].item()}
        torch.testing.assert_close(legacy['predictions'],extension['predictions'],rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(legacy['loss'],extension['loss'],rtol=1e-4,atol=1e-5)
    del legacy,extension
    if args.language_delta:
        language_delta=torch.load(args.language_delta,map_location='cpu',weights_only=True)
        model.llm_backbone.load_state_dict(language_delta,strict=True)
        del language_delta
        report['loaded_language_delta']=args.language_delta
    batch=dict(images=images,hl_input_ids=tensor(high['input_ids']),
               hl_valid_mask=tensor(high['valid_mask']),hl_ar_mask=tensor(high['ar_mask']),
               hl_labels=tensor(high['labels']),img_masks=img_masks)
    def evaluate():
        model.eval()
        start=time_module()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            ce=model.forward_high_level(**batch)['loss'].item()
            generated=model.generate_subtask(images,tensor(context['input_ids']),tensor(context['valid_mask']),
                img_masks,eos_token_id=prompts.eos_id,pad_token_id=prompts.pad_id,max_new_tokens=8)
        torch.cuda.synchronize()
        text=prompts.decode(generated['token_ids'][0].cpu().tolist())
        return dict(ce=ce,text=text,finished=bool(generated['finished'][0]),
                    token_ids=generated['token_ids'][0].cpu().tolist(),seconds=time_module()-start)
    def time_module():return __import__('time').perf_counter()
    report['initial']=evaluate()
    print(json.dumps({'initial':report['initial'],'E0':report['E0']},ensure_ascii=False),flush=True)
    report['training']=[]
    if args.steps:
        model.requires_grad_(False)
        model.llm_backbone.requires_grad_(True)
        optimizer=torch.optim.AdamW(model.llm_backbone.parameters(),lr=1e-5,weight_decay=0.0)
        train_start=time_module()
        for step in range(1,args.steps+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model.forward_high_level(**batch)['loss']
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite CE')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.llm_backbone.parameters(),1.0)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            if step==1 or step%10==0 or step==args.steps:
                row=dict(step=step,train_ce=loss.item(),grad_norm=norm.item())
                row['evaluation']=evaluate();report['training'].append(row)
                print(json.dumps(row,ensure_ascii=False),flush=True)
                (out/'report.partial.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
                if row['evaluation']['text']==task and row['evaluation']['finished']:
                    report['stop_reason']='identity sequence and EOS reproduced';break
        report['training_seconds']=time_module()-train_start
        # Model-only language delta, explicitly not full optimizer resume.
        checkpoint=out/'language_delta.pt'
        torch.save({k:v.detach().cpu() for k,v in model.llm_backbone.state_dict().items()},checkpoint)
        before=evaluate()
        delta=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.llm_backbone.load_state_dict(delta,strict=True);del delta
        after=evaluate()
        report['reload']={'same_tokens':before['token_ids']==after['token_ids'],
                          'ce_difference':abs(before['ce']-after['ce']),
                          'checkpoint':str(checkpoint),'scope':'language model only; not full optimizer resume'}
        if not report['reload']['same_tokens']:raise RuntimeError('Reload changed generation')
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    report['final']=evaluate()
    if args.replay:
        from fluxvla.models.vlas.subtask_runtime import SubtaskRuntime
        runtime=SubtaskRuntime(model,prompts,max_new_tokens=8,num_steps=10)
        report['replay']=[]
        for observation_version,observed_images in enumerate((images,images.flip(-1)),start=1):
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                result=runtime.infer(observed_images,states,[task],img_masks,
                                     observation_version=observation_version,noise=noise)
                direct=model.sample_actions_given_subtask(observed_images,states,
                    tensor(low['input_ids']),tensor(low['valid_mask']),img_masks,noise=noise,num_steps=10)
            torch.testing.assert_close(result['actions'],direct,rtol=1e-4,atol=1e-5)
            if not torch.isfinite(result['actions']).all():raise RuntimeError('Nonfinite action')
            row=dict(subtasks=result['subtasks'],trace=result['trace'],
                     action_shape=list(result['actions'].shape),
                     actions=result['actions'].cpu().tolist(),
                     actual_low_prompt_equals_gt=bool(torch.equal(result['ll_input_ids'],tensor(low['input_ids']))),
                     direct_action_max_difference=(direct-result['actions']).abs().max().item())
            report['replay'].append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='actions'},ensure_ascii=False),flush=True)
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('COMPLETE',str(out/'report.json'),flush=True)


if __name__=='__main__':main()
