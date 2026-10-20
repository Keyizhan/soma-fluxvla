"""Preflight -> cache -> stage1 -> stage2. No automatic GPU provisioning."""
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


def checkpoint_storage_plan(cfg):
    samples=json.loads((Path(cfg['cache'])/'samples.json').read_text())
    train=sum(r['split']=='train' for r in samples['records'])
    steps=math.ceil(train/cfg['global_batch'])
    stage2=next(s for s in cfg['stages'] if s['name']=='stage2')
    interval=cfg['checkpoint']['save_every_steps']
    total=steps*stage2['epochs']
    points=list(range(interval,total+1,interval))
    if not points or points[-1]!=total:points.append(total)
    gib=1024**3
    model=int(7*gib)  # 6.737 GiB bf16 weights + tokenizer/contracts and margin
    full=int(21*gib)
    safety=int(cfg['checkpoint'].get('safety_reserve_gib',20)*gib)
    peak=full+(1+len(points))*model+safety
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    fs=os.statvfs(root);available=fs.f_bavail*fs.f_frsize
    fresh=not any(root.glob('stage*/latest.json'))
    plan=dict(train_anchors=train,optimizer_steps_per_epoch=steps,stage2_total_updates=total,
        stage2_save_steps=points,model_bundle_bound_bytes=model,full_resume_bound_bytes=full,
        peak_reserved_bytes=peak,available_bytes=available,fresh_run=fresh,
        description='one final full snapshot + Stage1 final model + Stage2 saved models + safety')
    write_json(root/'checkpoint-storage-plan.json',plan)
    if fresh and available<peak:
        raise OSError(f'training checkpoint peak needs {peak/gib:.2f} GiB, available {available/gib:.2f} GiB')
    print(json.dumps(plan),flush=True)
    return plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--check',action='store_true');p.add_argument('--warm-start-stage2',action='store_true',help='explicit fresh-optimizer continuation from latest model-only Stage2 snapshot')
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    preflight(cfg,not a.check,verify_weights=not a.check)
    if a.check:return
    env=dict(os.environ,PYTHONPATH=str(HERE.parents[2]),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    def run(*args):subprocess.run(list(args),check=True,env=env,cwd=Path.cwd())
    # Verify count and labels before decoding videos or creating output.
    run(sys.executable,'-m','fluxvla.training.subtask.prepare','scan','--config',a.config)
    cache=Path(cfg['cache'])
    if not cache.exists():run(sys.executable,'-m','fluxvla.training.subtask.prepare','prepare','--config',a.config)
    run(sys.executable,'-m','fluxvla.training.subtask.prepare','verify','--config',a.config)
    checkpoint_storage_plan(cfg)
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    contract=out/'run-config.json'
    if contract.exists() and json.loads(contract.read_text())!=cfg:
        raise ValueError('run configuration changed; use a new output directory')
    write_json(contract,cfg)
    previous=None
    for stage in cfg['stages']:
        directory=out/stage['name']
        if not (directory/'COMPLETE.json').is_file():
            command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={cfg["expected_gpus"]}','--module','fluxvla.training.subtask.train','--config',a.config,'--stage',stage['name']]
            if previous:command+=['--initialize',str(previous)]
            if stage['name']=='stage2' and a.warm_start_stage2:command+=['--warm-start']
            run(*command)
        completed=json.loads((directory/'COMPLETE.json').read_text())
        previous=Path(completed['deployment'])
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',config_sha256=digest(a.config)))

if __name__=='__main__':main()
