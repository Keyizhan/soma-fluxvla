"""New pipeline contracts: valid prefixes, leakage, exact batches and real cache encoding."""
import io
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

CODE=Path(__file__).resolve().parents[1]
from fluxvla.training.subtask.common import anchors, epoch_batches, lr_at, split_groups, target_window
from fluxvla.training.subtask.prepare import scan, fit_stats, prepare as prepare_cache, verify


def test_short_runs_never_cross_boundary():
    labels=['a']*3+['b']*2+['invalid']+['c']
    result=anchors(labels,[1,1,1,1,1,0,1],2,50)
    assert result==[(0,2),(2,0),(3,1),(6,0)]
    states=np.arange(7*14,dtype='float32').reshape(7,14)
    actions=states+5
    out=target_window(states,actions,0,2)
    assert np.array_equal(out[:,0],[19,33])
    assert np.array_equal(out[:,6],actions[1:3,6])


def test_epoch_visits_once_across_ranks_and_accum():
    all_real=[]
    for rank in range(8):
        batches=list(epoch_batches(list(range(131)),1702,1,64,4,8,rank))
        assert len(batches)==3 and all(len(b)==2 for b in batches)
        all_real.extend(i for b in batches for micro in b for i in micro if i>=0)
    assert sorted(all_real)==list(range(131))
    assert list(epoch_batches(list(range(131)),1702,1,64,4,8,0))==list(epoch_batches(list(range(131)),1702,1,64,4,8,0))
    with pytest.raises(ValueError):list(epoch_batches([1],1,1,64,3,8,0))


def test_schedule_boundaries_and_resume():
    assert lr_at(1,100,7.5e-5)==pytest.approx(1.5e-5)
    assert lr_at(5,100,7.5e-5)==7.5e-5
    assert lr_at(100,100,7.5e-5)==pytest.approx(7.5e-6)
    resumed=[lr_at(i,100,7.5e-5) for i in range(48,101)]
    assert resumed==[lr_at(i,100,7.5e-5) for i in range(1,101)][47:]
    assert lr_at(1,200,7.5e-5)==pytest.approx(7.5e-6)  # fresh Stage2 warmup


def test_grouped_split_no_leakage():
    eps=[dict(trajectory_sha256=x) for x in ['a','a','b','c','d']]
    val=split_groups(eps,2,1702)
    assert len(val)==2
    assert {eps[i]['trajectory_sha256'] for i in val}.isdisjoint({eps[i]['trajectory_sha256'] for i in range(5) if i not in val})
    with pytest.raises(ValueError):split_groups([dict(trajectory_sha256='a')]*2,1,1)


def synthetic(tmp_path):
    root=tmp_path/'incoming'/'lerobot_v2_1';(root/'meta').mkdir(parents=True);(root/'data').mkdir()
    img=io.BytesIO();Image.new('RGB',(20,20),(20,40,60)).save(img,format='PNG')
    image=dict(bytes=img.getvalue(),path=None)
    keys=['images.cam_front','images.cam_left_wrist','images.cam_right_wrist']
    features={key:dict(dtype='image') for key in keys}
    info=dict(total_episodes=2,total_frames=24,chunks_size=1000,data_path='data/episode_{episode_index:06d}.parquet',features=features)
    (root/'meta/info.json').write_text(json.dumps(info))
    entries=[]
    for ep in range(2):
        n=12
        state=np.arange(n*14,dtype='float32').reshape(n,14)*.01+ep*100
        action=state+.1
        table=dict(state=state.tolist(),actions=action.tolist(),timestamp=(np.arange(n)/30).tolist(),
            subtask=['grasp']*3+['move']*8+['end'],subtask_valid=[True]*12,overall_task=['fold']*12)
        for key in keys:table[key]=[image]*n
        pq.write_table(pa.table(table),root/f'data/episode_{ep:06d}.parquet')
        entries.append(dict(episode_index=ep,length=n))
    (root/'meta/episodes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in entries))
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tokenizer = Tokenizer(models.WordLevel({'<pad>':0,'<eos>':1,'<bos>':2,'<unk>':3}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='<pad>', eos_token='<eos>',
                           bos_token='<bos>', unk_token='<unk>').save_pretrained(tmp_path/'tokenizer')
    cfg=json.loads((CODE/'configs/subtask/fabric106.json').read_text())
    cfg.update(incoming=str(root.parent),cache=str(tmp_path/'cache'),expected_episodes=2,validation_episodes=1,
        image_size=16,horizon=4,anchors_every=2,tokenizer=str(tmp_path/'tokenizer'),
        output=str(tmp_path/'runs'),global_batch=4,micro_batch=1,expected_gpus=1,
        stages=[dict(name='stage1',preset='stage1_fast',epochs=1,peak_lr=1e-4),dict(name='stage2',preset='hybrid',epochs=1,peak_lr=7.5e-5)])
    from fluxvla.training.subtask.common import digest
    cfg['tokenizer_sha256'] = {'tokenizer.json': digest(tmp_path/'tokenizer/tokenizer.json')}
    return cfg


def test_delta_statistics_use_train_only(tmp_path):
    cfg=synthetic(tmp_path);eps=scan(cfg)
    train=next(e for e in eps if e['split']=='train')
    table=pq.read_table(train['path']).to_pydict()
    s=np.asarray(table['state'],dtype='float32');a=np.asarray(table['actions'],dtype='float32')
    expected=np.concatenate([target_window(s,a,f,n) for f,n in train['anchors'] if n])
    st=fit_stats(eps,tmp_path)
    np.testing.assert_allclose(st['action']['q01'],np.quantile(expected,.01,axis=0))
    np.testing.assert_allclose(st['action']['q99'],np.quantile(expected,.99,axis=0))
    assert st['action']['q99'][0]<2  # not the absolute joint angle of the 100-offset episode


def test_real_end_to_end_cache_and_fast(tmp_path):
    cfg=synthetic(tmp_path)
    cfg['cache_workers']=2
    root=prepare_cache(cfg,CODE)
    verify(cfg)
    actions=np.load(root/'arrays/actions.npy',mmap_mode='r')
    lengths=np.load(root/'arrays/valid_lengths.npy')
    masks=np.load(root/'arrays/action_masks.npy')
    assert isinstance(actions,np.memmap)
    fast=np.load(root/'fast_tokens.npz')
    for i,n in enumerate(lengths):
        assert masks[i,:n,:14].all()
        assert not masks[i,n:].any() and not masks[i,:,:][...,14:].any()
        assert np.all(actions[i,n:]==0)
        assert (fast['fast_lengths'][i]==0)==(n==0)
    manifest=json.loads((root/'samples.json').read_text())
    assert manifest['split_counts']['train']==manifest['split_counts']['validation']
    # Persist a smoke fixture outside pytest's ephemeral directory when requested.
    import os,shutil
    if os.environ.get('SUBTASK_SMOKE_DIR'):
        target=Path(os.environ['SUBTASK_SMOKE_DIR']);target.mkdir(parents=True,exist_ok=True)
        shutil.copytree(root,target/'cache',dirs_exist_ok=True)
        shutil.copytree(cfg['tokenizer'],target/'tokenizer',dirs_exist_ok=True)
        smoke_cfg=dict(cfg,cache=str(target/'cache'),output=str(target/'runs'),
                       tokenizer=str(target/'tokenizer'))
        ready_path=target/'cache/READY.json'
        ready=json.loads(ready_path.read_text())
        ready['config']['tokenizer']=smoke_cfg['tokenizer']
        ready_path.write_text(json.dumps(ready,indent=2))
        (target/'config.json').write_text(json.dumps(smoke_cfg))
    # Tampered cache must fail its receipt check.
    (root/'prompt_report.json').write_text('{}')
    with pytest.raises(ValueError,match='verification failed'):verify(cfg)


def test_video_frames_count_gate(tmp_path):
    import av
    from fluxvla.training.subtask.prepare import cache_camera
    path=tmp_path/'camera.mp4'
    with av.open(str(path),'w') as c:
        stream=c.add_stream('mpeg4',rate=30);stream.width=16;stream.height=16;stream.pix_fmt='yuv420p'
        for i in range(3):
            frame=av.VideoFrame.from_ndarray(np.full((16,16,3),i*40,dtype='uint8'),format='rgb24')
            for packet in stream.encode(frame):c.mux(packet)
        for packet in stream.encode():c.mux(packet)
    target=np.zeros((2,9,16,16),dtype='uint8')
    cache_camera(dict(n=3),dict(kind='video',path=str(path)),target,[(0,0),(2,1)],0,16)
    assert target[1,:3].mean()>target[0,:3].mean()+40
    with pytest.raises(ValueError,match='refusing guessed alignment'):
        cache_camera(dict(n=4),dict(kind='video',path=str(path)),target,[(0,0)],0,16)




def test_fixed_400_step_storage_plan(tmp_path,monkeypatch):
    import importlib.util
    from types import SimpleNamespace
    from fluxvla.training.subtask import run as module
    cfg=json.loads((CODE/'configs/subtask/fabric106.json').read_text())
    cache=tmp_path/'cache';cache.mkdir()
    (cache/'samples.json').write_text(json.dumps(dict(records=[dict(split='train')]*10878+[dict(split='validation')]*1614)))
    cfg.update(cache=str(cache),output=str(tmp_path/'runs'))
    cfg['checkpoint']['save_every_steps']=400
    monkeypatch.setattr(module.os,'statvfs',lambda p:SimpleNamespace(f_bavail=120,f_frsize=1024**3))
    plan=module.checkpoint_storage_plan(cfg)
    assert plan['optimizer_steps_per_epoch']==170
    assert plan['stage2_total_updates']==1360
    assert plan['stage2_save_steps']==[400,800,1200,1360]
    assert plan['peak_reserved_bytes']==76*1024**3
    monkeypatch.setattr(module.os,'statvfs',lambda p:SimpleNamespace(f_bavail=75,f_frsize=1024**3))
    with pytest.raises(OSError,match='checkpoint peak needs'):module.checkpoint_storage_plan(cfg)
