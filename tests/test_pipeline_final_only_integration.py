"""Opt-in CPU DDP policy smoke using an existing prepared tiny fixture cache.

PIPELINE_SMOKE_CONFIG=/path/to/smoke/config.json pytest -q -s this_file.py
"""
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys

import pytest
import torch

CODE=Path(__file__).parents[1]
from fluxvla.training.subtask.checkpoints import verify_bundle, verify_full


@pytest.mark.skipif(not os.environ.get('PIPELINE_SMOKE_CONFIG'),reason='requires prepared tiny PI05 fixture cache')
def test_final_optimizer_only_cpu_ddp(tmp_path):
    source=json.loads(Path(os.environ['PIPELINE_SMOKE_CONFIG']).read_text())
    root=Path(os.environ.get('PIPELINE_SMOKE_OUTPUT',str(tmp_path)))
    root.mkdir(parents=True,exist_ok=True)
    configs={}
    for name,interval in [('n1',1),('n2',2)]:
        cfg=json.loads(json.dumps(source))
        cfg['output']=str(root/name)
        cfg['checkpoint']=dict(format='full-state-deploy-v2',optimizer_save='final_only',
                               save_every_steps=interval,safety_reserve_gib=0)
        for stage in cfg['stages']:stage['epochs']=2
        path=root/(name+'.json');path.write_text(json.dumps(cfg));configs[name]=path
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')

    def run(name,stage,*extra,success=True):
        log=root/(name+'-'+stage+'-'+str(len(list(root.glob('*.log'))))+'.log')
        with log.open('w') as f:
            result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',
                '--nproc_per_node=2','--module','fluxvla.training.subtask.train','--config',str(configs[name]),
                '--stage',stage,'--smoke','--device','cpu',*map(str,extra)],env=env,cwd=CODE,
                stdout=f,stderr=subprocess.STDOUT,timeout=180)
        if success:assert result.returncode==0,log.read_text()[-10000:]
        else:assert result.returncode!=0
        return log

    # Stage1 never writes an optimizer or even a weight snapshot before final.
    run('n1','stage1','--stop-after-epoch',1)
    assert not list((root/'n1/stage1').glob('*.pt'))
    assert not list((root/'n1/stage1').glob('*.deploy'))
    run('n1','stage1')
    first=root/'n1/stage1/final.deploy'
    verify_bundle(first)
    assert not list((root/'n1/stage1').glob('*.pt'))

    # Mid-epoch stop at a real N boundary: weight-only, no optimizer/RNG serialization.
    run('n1','stage2','--initialize',first,'--stop-after-update',1)
    intermediate=root/'n1/stage2/step-00000001.deploy'
    m=verify_bundle(intermediate)
    assert m['metadata']['epoch_index']==0 and m['metadata']['batch_cursor']==1
    assert 'rng' not in m['metadata'] and 'optimizer' not in m['metadata']
    assert not list((root/'n1/stage2').rglob('*.pt'))
    rejected=run('n1','stage2',success=False)
    assert '--warm-start explicitly' in rejected.read_text()
    continued=run('n1','stage2','--warm-start')
    assert 'WEIGHT_ONLY_WARM_START' in continued.read_text()
    assert '"optimizer_reinitialized": true' in continued.read_text()

    # Stopping off-boundary does not manufacture an extra checkpoint.
    run('n2','stage2','--initialize',first,'--stop-after-update',1)
    assert not list((root/'n2/stage2').glob('*.deploy'))
    assert not list((root/'n2/stage2').glob('*.pt'))
    run('n2','stage2','--initialize',first,'--stop-after-update',2)
    assert len(list((root/'n2/stage2').glob('*.deploy')))==1
    assert not list((root/'n2/stage2').glob('*.pt'))
    run('n2','stage2','--warm-start')

    for name,count in [('n1',4),('n2',2)]:
        stage=root/name/'stage2'
        fulls=list(stage.glob('*.pt'))
        assert len(fulls)==1 and fulls[0].name=='step-00000004.pt'
        state=verify_full(fulls[0]);assert state['optimizer']['state'] and len(state['rng'])==2
        assert state['update']==state['total_updates']==4
        bundles=list(stage.glob('*.deploy'));assert len(bundles)==count
        for bundle in bundles:verify_bundle(bundle)
        done=json.loads((stage/'COMPLETE.json').read_text())
        assert done['optimizer_save']=='final_only' and done['latest_full']==str(fulls[0])
        final=Path(done['deployment'])
        from safetensors.torch import load_file
        deployed=load_file(str(final/'model.safetensors'))
        assert all(torch.equal(t,deployed[k]) for k,t in state['model_state'].items())
    # Retry after final full succeeded but deployment/COMPLETE publication failed.
    # Reuse the validated full file, without another optimizer serialization or peak copy.
    stage=root/'n2/stage2'
    done=json.loads((stage/'COMPLETE.json').read_text())
    final_full=Path(done['latest_full'])
    before=(final_full.stat().st_ino,final_full.stat().st_mtime_ns,final_full.stat().st_size)
    (stage/'COMPLETE.json').unlink()
    shutil.rmtree(done['deployment'])
    run('n2','stage2')
    assert before==(final_full.stat().st_ino,final_full.stat().st_mtime_ns,final_full.stat().st_size)
    verify_bundle(json.loads((stage/'COMPLETE.json').read_text())['deployment'])
    receipt=dict(status='PASSED',policy='strict_final_optimizer_only',world_size=2,device='cpu',
        stage1_intermediate_artifacts=0,stage1_final_optimizer_files=0,
        stage2_intermediate_optimizer_files=0,stage2_final_optimizer_files_each=1,
        stage2_n1_deployments=4,stage2_n2_deployments=2,off_boundary_stop_creates_no_snapshot=True,
        default_weight_only_resume_rejected=True,explicit_warm_start_reinitializes_optimizer=True,
        final_model_bundle_equal=True,bitwise_mid_epoch_optimizer_recovery='not applicable; deliberately no intermediate optimizer/RNG')
    receipt['finalization_retry_reuses_existing_full_without_rewrite']=True
    (root/'verification.json').write_text(json.dumps(receipt,indent=2))
