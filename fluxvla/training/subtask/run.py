"""Validate data and run joint training or the two-stage recipe."""
from .config import load_config
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .common import digest, write_json

HERE=Path(__file__).resolve().parent

def preflight(cfg, require_gpu, verify_weights=False):
    checks={}
    for package in ['torch','transformers','numpy','pyarrow','av','scipy','tokenizers','safetensors']:
        checks[package]=importlib.metadata.version(package)
    if checks['transformers']!='5.3.0':raise ValueError('this snapshot requires transformers==5.3.0')
    base=Path(cfg['pretrained'])
    for name in ['model.safetensors','tokenizer.json','tokenizer.model','tokenizer_config.json']:
        if not (base/name).is_file():raise FileNotFoundError(base/name)
    checks['base_weight_bytes']=(base/'model.safetensors').stat().st_size
    for name,expected in cfg['tokenizer_sha256'].items():
        if digest(base/name)!=expected:raise ValueError(f'tokenizer SHA256 mismatch: {name}')
    checks['base_sha256_verified']=verify_weights
    checks['free_GiB']=round(shutil.disk_usage(base).free/1024**3,2)
    checks['incoming_exists']=Path(cfg['incoming']).exists()
    limit_path=Path('/sys/fs/cgroup/memory.max')
    limit=limit_path.read_text().strip() if limit_path.exists() else 'max'
    checks['memory_limit']=limit
    if require_gpu:
        if cfg.get('checkpoint',{}).get('optimizer_save')!='final_only':
            raise RuntimeError('checkpoint optimizer policy must be final_only')
        interval=cfg.get('checkpoint',{}).get('save_every_steps')
        if not isinstance(interval,int) or isinstance(interval,bool) or interval<1:
            raise RuntimeError('Stage2 checkpoint interval N is not configured; set checkpoint.save_every_steps before training')
        if limit!='max' and int(limit)<128*1024**3:
            raise RuntimeError('training needs >=128 GiB cgroup host memory; current instance is still memory-limited')
        result=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True,capture_output=True,check=True)
        checks['gpus']=result.stdout.strip().splitlines()
        if len(checks['gpus'])!=cfg['expected_gpus']:raise RuntimeError(f'require {cfg["expected_gpus"]} GPUs: {checks["gpus"]}')
        if cfg['global_batch']%(cfg['micro_batch']*cfg['expected_gpus']):raise ValueError('invalid global batch')
    if verify_weights:
        print('Verifying original PI0.5 base SHA256...',flush=True)
        if digest(base/'model.safetensors')!=cfg['base_sha256']:raise ValueError('original base SHA256 mismatch')
    print(json.dumps(checks,ensure_ascii=False,indent=2),flush=True)
    return checks


def checkpoint_storage_plan(cfg, stop_after_stage=None):
    samples=json.loads((Path(cfg['cache'])/'samples.json').read_text())
    train=sum(r['split']=='train' for r in samples['records'])
    steps=math.ceil(train/cfg['global_batch'])
    stages=cfg['stages']
    if stop_after_stage is not None:
        names=[s['name'] for s in stages]
        if stop_after_stage not in names:raise ValueError('stop stage is absent from configuration')
        stages=stages[:names.index(stop_after_stage)+1]
    stage1=next((s for s in stages if s['name']=='stage1'),None)
    stage2=next((s for s in stages if s['name']=='stage2'),None)
    interval=cfg['checkpoint']['save_every_steps']
    total=steps*stage2['epochs'] if stage2 else 0
    points=list(range(interval,total+1,interval))
    if total and (not points or points[-1]!=total):points.append(total)
    gib=1024**3
    model=int(7*gib)  # 6.737 GiB bf16 weights + tokenizer/contracts and margin
    full=int(21*gib)
    safety=int(cfg['checkpoint'].get('safety_reserve_gib',20)*gib)
    epoch_bundles=stage1['epochs'] if stage1 and stage1.get('keep_epoch_checkpoints',False) else 0
    peak=(full if stage2 else 0)+(1+epoch_bundles+len(points))*model+safety
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    available=shutil.disk_usage(root).free
    fresh=not any(root.glob('stage*/latest.json'))
    plan=dict(train_anchors=train,optimizer_steps_per_epoch=steps,stage2_total_updates=total,
        total_optimizer_updates=steps*sum(s['epochs'] for s in stages),
        extra_stage1_updates=steps*(stage1['epochs'] if stage1 else 0),
        stop_after_stage=stop_after_stage,stage1_retained_epoch_bundles=epoch_bundles,
        stage2_save_steps=points,model_bundle_bound_bytes=model,full_resume_bound_bytes=full,
        peak_reserved_bytes=peak,available_bytes=available,fresh_run=fresh,
        description='active stages: final full snapshot (Stage2 only) + Stage1 final and retained epochs + Stage2 saved models + safety')
    write_json(root/'checkpoint-storage-plan.json',plan)
    if fresh and available<peak:
        raise OSError(f'training checkpoint peak needs {peak/gib:.2f} GiB, available {available/gib:.2f} GiB')
    print(json.dumps(plan),flush=True)
    return plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--check',action='store_true');p.add_argument('--warm-start-stage2',action='store_true',help='explicit fresh-optimizer continuation from latest model-only Stage2 snapshot')
    p.add_argument('--stop-after-stage',choices=['stage1','stage2'],help='finish the selected stage and its receipt without starting later stages')
    a=p.parse_args();cfg=load_config(a.config)
    if a.stop_after_stage and a.stop_after_stage not in [s['name'] for s in cfg['stages']]:
        p.error('--stop-after-stage is absent from configured stages')
    from .common import validate_training_method
    validate_training_method(cfg)
    preflight(cfg,not a.check,verify_weights=not a.check)
    if a.check:return
    env=dict(os.environ,PYTHONPATH=str(HERE.parents[2]),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    def run(*args):subprocess.run(list(args),check=True,env=env,cwd=Path.cwd())
    # Verify count and labels before decoding videos or creating output.
    run(sys.executable,'-m','fluxvla.training.subtask.prepare','scan','--config',a.config)
    cache=Path(cfg['cache'])
    if not cache.exists():run(sys.executable,'-m','fluxvla.training.subtask.prepare','prepare','--config',a.config)
    run(sys.executable,'-m','fluxvla.training.subtask.prepare','verify','--config',a.config)
    checkpoint_storage_plan(cfg,a.stop_after_stage)
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    contract=out/'run-config.json'
    if contract.exists() and json.loads(contract.read_text())!=cfg:
        raise ValueError('run configuration changed; use a new output directory')
    write_json(contract,cfg)
    from .config import stage1_identity
    from .checkpoints import verify_bundle
    shared=Path(cfg['shared_stage1_root'])/f"seed-{cfg['seed']}"
    shared_file=shared/'resolved-config.json'
    if cfg['training_method'] in ('hybrid_insulated','two_stage'):
        shared.mkdir(parents=True,exist_ok=True)
        shared_cfg=dict(cfg,output=str(shared))
        if shared_file.exists():
            if stage1_identity(load_config(shared_file))!=stage1_identity(cfg):
                raise ValueError('shared stage1 configuration changed; use a new shared root')
        else:write_json(shared_file,shared_cfg)
    previous=None
    for stage in cfg['stages']:
        directory=(shared if stage['name']=='stage1' else out)/stage['name']
        if not (directory/'COMPLETE.json').is_file():
            command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={cfg["expected_gpus"]}','--module','fluxvla.training.subtask.train','--config',str(shared_file) if stage['name']=='stage1' else a.config,'--stage',stage['name']]
            if previous:command+=['--initialize',str(previous)]
            if stage['name']=='stage2' and a.warm_start_stage2:command+=['--warm-start']
            if stage['name']=='stage1':
                lock=shared/'stage1.lock'
                try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
                except FileExistsError:raise RuntimeError('shared stage1 is already running or interrupted; inspect stage1.lock')
                try:run(*command)
                finally:
                    os.close(fd);lock.unlink()
            else:run(*command)
        completed=json.loads((directory/'COMPLETE.json').read_text())
        previous=Path(completed['deployment'])
        verified=verify_bundle(previous)
        meta=verified['metadata']
        if meta.get('cache_identity')!=digest(cache/'READY.json') or meta.get('stage1_identity')!=stage1_identity(cfg):
            raise ValueError('completed bundle has wrong cache or common stage1 identity')
        if stage['name']=='stage2':
            if meta.get('preset')!=stage['preset'] or meta.get('config')!=cfg or (cfg['training_method'] in ('hybrid_insulated','two_stage') and meta.get('stage1_weight_sha256')!=selected['weight_sha256']):
                raise ValueError('completed stage2 method/config/initialization identity mismatch')
        if stage['name']=='stage1':
            selected=dict(deployment=str(previous),weight_sha256=digest(previous/'model.safetensors'),cache_identity=meta['cache_identity'],stage1_identity=stage1_identity(cfg))
            receipt=shared/'selected-stage1.json'
            if receipt.exists() and json.loads(receipt.read_text())!=selected:raise ValueError('shared stage1 selected weights changed')
            write_json(receipt,selected)
            write_json(out/'stage1-initialization.json',selected)
        if stage['name']==a.stop_after_stage:
            write_json(out/'STOPPED_AFTER_STAGE.json',dict(status='STAGE_COMPLETE',stage=stage['name'],deployment=str(previous),config_sha256=digest(a.config)))
            print(json.dumps(dict(status='STOPPED_AFTER_STAGE',stage=stage['name'],deployment=str(previous))),flush=True)
            return

    write_json(out/'COMPLETE.json',dict(status='COMPLETE',config_sha256=digest(a.config)))

if __name__=='__main__':main()
