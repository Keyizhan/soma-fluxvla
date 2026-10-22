"""Streaming LeRobot v2.1 -> real .npy mmap cache, delta stats and fixed-horizon FAST."""
from .config import load_config, canonical_prompts, PROMPT_CONTRACT, CACHE_VERSION
import argparse
import io
import json
import shutil
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
WINDOW_CONTRACT = 'next-frame-fixed-horizon-masked-tail-v3'
SAMPLING_CONTRACT = 'all_valid_frames'

from .common import anchors, digest, normalize, stats, target_window, write_json


def choose(names, aliases):
    matched = [key for key in aliases if key in names]
    if len(matched) != 1:
        raise ValueError(f'expected exactly one of {aliases}; found {matched}')
    return matched[0]


def flag(value):
    if value not in (True, False, 0, 1) or value is None:
        raise ValueError(f'invalid subtask_valid {value!r}; require bool/0/1')
    return bool(value)


def scan(cfg):
    incoming = Path(cfg['incoming']).resolve()
    roots = sorted({p.parent.parent for p in incoming.rglob('meta/info.json')})
    if (incoming / 'meta/info.json').is_file():
        roots = sorted(set(roots) | {incoming})
    episodes = []
    for root in roots:
        info = json.loads((root/'meta/info.json').read_text())
        entries = [json.loads(line) for line in (root/'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
        if len(entries) != int(info['total_episodes']):
            raise ValueError(f'{root}: incomplete episode metadata')
        seen = set()
        rows_total = 0
        for entry in entries:
            idx = int(entry['episode_index'])
            if idx in seen:
                raise ValueError(f'duplicate episode {root}:{idx}')
            seen.add(idx)
            fmt = dict(episode_index=idx, episode_chunk=idx//int(info.get('chunks_size', 1000)))
            path = root/info['data_path'].format(**fmt)
            names = pq.ParquetFile(path).schema_arrow.names
            state_key = choose(names, ['state', 'observation.state'])
            action_key = choose(names, ['actions', 'action'])
            required = [state_key, action_key, 'timestamp', 'subtask', 'subtask_valid', 'overall_task']
            table = pq.read_table(path, columns=required).to_pydict()
            state = np.asarray(table[state_key], dtype='float32')
            action = np.asarray(table[action_key], dtype='float32')
            n = len(state)
            if state.shape != (n,14) or action.shape != (n,14) or n < 1:
                raise ValueError(f'{path}: require nonempty 14 dimensional state/action')
            stamps = np.asarray(table['timestamp'], dtype='float64')
            if not all(np.isfinite(x).all() for x in (state, action, stamps)) or np.any(np.diff(stamps) <= 0):
                raise ValueError(f'{path}: non-finite values or non-increasing timestamps')
            if n != int(entry['length']):
                raise ValueError(f'{path}: expected {entry["length"]} rows, got {n}')
            rows_total += n
            valid = [flag(x) for x in table['subtask_valid']]
            labels = table['subtask']
            if any(ok and (not isinstance(label,str) or not label.strip()) for label,ok in zip(labels,valid)):
                raise ValueError(f'{path}: valid frames have missing subtask text')
            overall = table['overall_task']
            if any(not isinstance(t,str) or not t.strip() for t in overall) or len(set(overall)) != 1:
                raise ValueError(f'{path}: require one nonempty overall_task')
            picked = anchors(valid, cfg['horizon'])
            if not picked:
                raise ValueError(f'{path}: no valid annotated anchors')
            # Fingerprint trajectories, ignoring labels and root to catch duplicate exports.
            import hashlib
            fingerprint = hashlib.sha256(state.tobytes()+action.tobytes()+stamps.tobytes()).hexdigest()
            camera_keys = [choose(info['features'], aliases) for aliases in cfg['camera_aliases']]
            sources = []
            for key in camera_keys:
                kind = info['features'][key]['dtype']
                if kind == 'video':
                    template = info.get('video_path', 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
                    video = root/template.format(**fmt, video_key=key)
                    if not video.is_file():
                        raise FileNotFoundError(video)
                    sources.append(dict(kind=kind, path=str(video), key=key))
                elif kind == 'image' and key in names:
                    sources.append(dict(kind=kind, path=str(path), key=key))
                else:
                    raise ValueError(f'unsupported camera {key}: {kind}')
            episodes.append(dict(root=str(root), episode=idx, path=str(path), state_key=state_key,
                action_key=action_key, n=n, anchors=picked, trajectory_sha256=fingerprint,
                parquet_sha256=digest(path), sources=sources))
        if rows_total != int(info['total_frames']):
            raise ValueError(f'{root}: metadata frame count does not match actual frames')
    if len(episodes) != cfg['expected_episodes']:
        raise ValueError(f'waiting for complete data: expected {cfg["expected_episodes"]} episodes, found {len(episodes)}')
    manifest=json.loads(Path(cfg['split_manifest']).read_text())
    assignments={int(k):v for k,v in manifest['episodes'].items()}
    if set(assignments)!={ep['episode'] for ep in episodes}:
        raise ValueError('split manifest must cover every episode exactly once')
    groups={}
    for ep in episodes:
        ep['split']=assignments[ep['episode']]
        if ep['split'] not in ('train','validation','test'):raise ValueError('invalid split')
        groups.setdefault(ep['trajectory_sha256'],set()).add(ep['split'])
    if any(len(v)>1 for v in groups.values()):
        raise ValueError('duplicate trajectory crosses fixed episode splits')
    return episodes


def read_numeric(ep):
    t = pq.read_table(ep['path'], columns=[ep['state_key'], ep['action_key'], 'timestamp', 'subtask', 'overall_task']).to_pydict()
    return t, np.asarray(t[ep['state_key']], dtype='float32'), np.asarray(t[ep['action_key']], dtype='float32')


def fit_stats(episodes, temp):
    train = [ep for ep in episodes if ep['split']=='train']
    counts = [sum(ep['n'] for ep in train), sum(n for ep in train for _,n in ep['anchors'])]
    if not counts[1]:
        raise ValueError('train has no action targets')
    paths = [temp/'raw_states.npy', temp/'raw_targets.npy']
    maps = [np.lib.format.open_memmap(p, mode='w+', dtype='float32', shape=(n,14)) for p,n in zip(paths,counts)]
    offsets = [0,0]
    for ep in train:
        _, states, actions = read_numeric(ep)
        maps[0][offsets[0]:offsets[0]+len(states)] = states
        offsets[0] += len(states)
        for frame, length in ep['anchors']:
            maps[1][offsets[1]:offsets[1]+length] = target_window(states,actions,frame,length)
            offsets[1] += length
    result = dict(state=stats(maps[0]), action=stats(maps[1]), state_rows=counts[0], valid_action_rows=counts[1])
    for m in maps:
        m.flush()
        m._mmap.close()
    del maps
    for p in paths:
        p.unlink()
    return result


def resized(frame, size):
    return np.asarray(Image.fromarray(frame).convert('RGB').resize((size,size), Image.Resampling.BILINEAR)).transpose(2,0,1)


def cache_camera(ep, source, target, indices, camera, size):
    wanted = {frame: row for frame,row in indices}
    if source['kind'] == 'video':
        import av
        with av.open(source['path']) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'NONE'
            stream.codec_context.thread_count = 1
            count = 0
            for count, frame in enumerate(container.decode(stream), start=1):
                if count-1 in wanted:
                    target[wanted[count-1],camera*3:(camera+1)*3] = resized(frame.to_ndarray(format='rgb24'),size)
            if count != ep['n']:
                raise ValueError(f'{source["path"]}: {count} video frames != {ep["n"]} parquet rows; refusing guessed alignment')
    else:
        offset = 0
        for batch in pq.ParquetFile(source['path']).iter_batches(batch_size=8, columns=[source['key']]):
            for j, cell in enumerate(batch.column(0).to_pylist()):
                if offset+j in wanted:
                    payload = cell.get('bytes') if isinstance(cell,dict) else cell
                    if not isinstance(payload,bytes):
                        raise ValueError('inline image requires embedded bytes')
                    target[wanted[offset+j],camera*3:(camera+1)*3] = resized(np.asarray(Image.open(io.BytesIO(payload)).convert('RGB')),size)
            offset += batch.num_rows
        if offset != ep['n']:
            raise ValueError('inline camera frame count mismatch')


def encode_fast(root, cfg, processor):
    actions = np.load(root/'arrays/actions.npy', mmap_mode='r')
    # FAST retains the complete padded window; flow masks alone exclude the tail.
    lengths = np.full(len(actions), cfg['horizon'], dtype='int32')
    result = [[] for _ in lengths]
    squared, absolute, count, worst = 0.,0.,0,0.
    for length in sorted(set(lengths.tolist())):
        if length == 0:
            continue
        indices = np.flatnonzero(lengths == length)
        for start in range(0,len(indices),64):
            ids = indices[start:start+64]
            batch = np.asarray(actions[ids,:int(length),:14])
            tokens = processor(batch)
            for i, seq in zip(ids,tokens):
                seq = list(map(int,seq))
                if not seq or len(seq)>cfg['max_fast_tokens'] or min(seq)<0 or max(seq)>=processor.vocab_size:
                    raise ValueError(f'sample {i}: invalid/over-budget FAST length {len(seq)}; no truncation')
                result[i] = seq
            for seq in tokens:
                if len(processor.bpe_tokenizer.decode(seq))!=int(length)*14:
                    raise ValueError('FAST roundtrip coefficient shape mismatch; refusing decoder zero fallback')
            reconstruction = np.asarray(processor.decode(tokens,time_horizon=int(length),action_dim=14))
            error = reconstruction-batch
            if not np.isfinite(error).all():
                raise ValueError('nonfinite FAST roundtrip')
            squared += float(np.square(error).sum()); absolute += float(np.abs(error).sum())
            worst = max(worst,float(np.abs(error).max())); count += error.size
    flat = np.asarray([x for seq in result for x in seq],dtype='int64')
    np.savez(root/'fast_tokens.npz', fast_ids=flat, fast_lengths=np.array(list(map(len,result)),dtype='int64'))
    write_json(root/'fast_report.json',dict(samples=len(result),text_only=int(sum(lengths==0)),
        max_tokens=max(map(len,result)),mse=squared/max(1,count),mae=absolute/max(1,count),max_abs=worst,
        encoded_dimensions=14, padding_encoded=False))


def cache_camera_job(args):
    ep,source,filename,indices,camera,size=args
    target=np.load(filename,mmap_mode='r+')
    cache_camera(ep,source,target,indices,camera,size)
    target.flush()
    return source['path']


def prepare(cfg, code_root, processor=None):
    if cfg.get('version') != CACHE_VERSION or cfg.get('prompt_contract') != PROMPT_CONTRACT:
        raise ValueError('cache preparation requires the paper prompt contract and cache version')
    output = Path(cfg['cache'])
    if output.exists():
        raise FileExistsError(f'{output} already exists; use verify or a new cache path')
    all_episodes = scan(cfg)
    episodes = [ep for ep in all_episodes if ep['split'] != 'test']
    n = sum(len(ep['anchors']) for ep in episodes)
    size,h = cfg['image_size'],cfg['horizon']
    estimate = n*(9*size*size+4*h*32+h*32+14*4)+sum(ep['n'] for ep in episodes)*14*4
    reserve=cfg.get('checkpoint_reserve_gib',26)*1024**3
    if shutil.disk_usage(output.parent).free < estimate*1.15+reserve:
        raise OSError(f'cache needs {estimate/1024**3:.2f} GiB plus 15% margin and {reserve/1024**3:.0f} GiB checkpoint reserve; free space before caching')
    temp = output.with_name(output.name+'.building')
    if temp.exists():
        raise FileExistsError(f'incomplete prior cache: {temp}; inspect/remove explicitly before retry')
    (temp/'arrays').mkdir(parents=True)
    write_json(temp/'split_manifest.json',dict(episodes=all_episodes,source_sha256=digest(cfg['split_manifest'])))
    normalization = fit_stats(episodes,temp)
    specifications = dict(images=((n,9,size,size),'uint8'), states=((n,14),'float32'),
        actions=((n,h,32),'float32'),action_masks=((n,h,32),'bool'),img_masks=((n,3),'bool'),valid_lengths=((n,),'int32'))
    arrays = {k:np.lib.format.open_memmap(temp/f'arrays/{k}.npy',mode='w+',dtype=dtype,shape=shape)
              for k,(shape,dtype) in specifications.items()}
    records = []
    offset = 0
    workers=int(cfg.get('cache_workers',1))
    pool=None;pending=[]
    if workers>1:
        import concurrent.futures,multiprocessing
        pool=concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))
    for ep in episodes:
        table,state,action = read_numeric(ep)
        mapping = [(frame,offset+j) for j,(frame,_) in enumerate(ep['anchors'])]
        for camera,source in enumerate(ep['sources']):
            if pool is None:
                cache_camera(ep,source,arrays['images'],mapping,camera,size)
            else:
                pending.append(pool.submit(cache_camera_job,(ep,source,str(temp/'arrays/images.npy'),mapping,camera,size)))
        for frame,length in ep['anchors']:
            i = offset
            arrays['states'][i] = normalize(state[frame],normalization['state'])
            arrays['actions'][i] = 0
            arrays['action_masks'][i] = False
            window = target_window(state,action,frame,h)
            arrays['actions'][i,:,:14] = normalize(window,normalization['action'])
            arrays['action_masks'][i,:length,:14] = True
            arrays['img_masks'][i] = True
            arrays['valid_lengths'][i] = length
            records.append(dict(sample_id=f'{len(records):08d}',source=ep['root'],episode=ep['episode'],frame=frame,
                split=ep['split'],valid_length=length,timestamp=float(table['timestamp'][frame]),
                overall_task=table['overall_task'][frame],subtask=table['subtask'][frame]))
            offset += 1
        print(json.dumps(dict(episode=ep['episode'],source=ep['root'],cached=offset,total=n)),flush=True)
    if pool is not None:
        try:
            for future in pending:future.result()
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
    for a in arrays.values():
        a.flush()
        a._mmap.close()
    del arrays
    write_json(temp/'samples.json',dict(records=records,episodes=episodes,normalization=normalization,
        conventions=dict(action_window=WINDOW_CONTRACT,joint_delta_dims=[0,1,2,3,4,5,7,8,9,10,11,12],grippers='absolute meters',
            horizon=h,sampling=SAMPLING_CONTRACT,invalid_padding='repeat last action, masked from flow loss',normalization='train valid delta windows q01/q99'),
        split_counts={s:sum(r['split']==s for r in records) for s in ('train','validation')}))
    if processor is None:
        # Explicit local BPE construction is compatible with transformers 5.3;
        # AutoProcessor's generic legacy FAST config loses tokenizer_file in 5.x.
        import importlib.util
        from transformers import PreTrainedTokenizerFast
        fast_dir=code_root/'assets/fast'
        spec=importlib.util.spec_from_file_location('subtask_local_fast',fast_dir/'processing_action_tokenizer.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        settings=json.loads((fast_dir/'processor_config.json').read_text())
        processor=module.UniversalActionProcessor(
            PreTrainedTokenizerFast(tokenizer_file=str(fast_dir/'tokenizer.json')),
            **{k:settings[k] for k in ['scale','vocab_size','min_token']})
    encode_fast(temp,cfg,processor)
    # Check the complete text+FAST length before producing READY (no silent truncation).
    from .fixtures import Fixtures, prepare as pack
    from fluxvla.engines import build_tokenizer_from_cfg
    tok = build_tokenizer_from_cfg(dict(type='PretrainedTokenizer',model_path=cfg['tokenizer']))
    prompts = canonical_prompts(tok,cfg)
    packed = pack(Fixtures(temp),prompts,14)
    write_json(temp/'prompt_report.json',dict(max_tokens=max(int(p['high']['valid_mask'].sum()) for p in packed),samples=n))
    del packed
    files = {str(p.relative_to(temp)):dict(size=p.stat().st_size,sha256=digest(p)) for p in sorted(temp.rglob('*')) if p.is_file()}
    write_json(temp/'READY.json',dict(prompt_contract=PROMPT_CONTRACT,window_contract=WINDOW_CONTRACT,sampling=SAMPLING_CONTRACT,version=cfg['version'],config=cfg,files=files,samples=n,
        data_identity=[dict(episode=e['episode'],split=e['split'],parquet_sha256=e['parquet_sha256'],trajectory_sha256=e['trajectory_sha256']) for e in all_episodes],split_manifest_sha256=digest(cfg['split_manifest']),normalization=normalization,train_episodes=sum(e['split']=='train' for e in episodes),validation_episodes=sum(e['split']=='validation' for e in episodes)))
    temp.rename(output)
    return output


def validate_cache_contract(ready):
    if ready.get('window_contract') != WINDOW_CONTRACT:
        raise ValueError('old action-window cache; rebuild in a new cache directory')
    if ready.get('sampling') != SAMPLING_CONTRACT:
        raise ValueError('unsupported cache sampling; rebuild with all_valid_frames in a new cache directory')
    if ready.get('version') != CACHE_VERSION or ready.get('prompt_contract') != PROMPT_CONTRACT:
        raise ValueError('cache predates paper prompt/attention contract; rebuild cache')


def validate_cache_config(ready,cfg):
    validate_cache_contract(ready)
    # Fixed episode manifests make this cache independent of the training seed.
    for key in ('version','prompt_contract','expected_episodes','validation_episodes','horizon','image_size','max_prompt_length','max_fast_tokens','camera_aliases','tokenizer','split_manifest'):
        if cfg[key]!=ready['config'][key]:
            raise ValueError(f'cache config changed: {key}')
    if ready.get('split_manifest_sha256')!=digest(cfg['split_manifest']):
        raise ValueError('fixed split manifest changed')


def verify(cfg):
    root=Path(cfg['cache'])
    ready=json.loads((root/'READY.json').read_text())
    validate_cache_contract(ready)
    for name,meta in ready['files'].items():
        p=root/name
        if not p.is_file() or p.stat().st_size!=meta['size'] or digest(p)!=meta['sha256']:
            raise ValueError(f'cache verification failed: {p}')
    validate_cache_config(ready,cfg)
    current=scan(cfg)
    actual=[dict(episode=e['episode'],split=e['split'],parquet_sha256=e['parquet_sha256'],trajectory_sha256=e['trajectory_sha256']) for e in current]
    if ready.get('data_identity')!=actual or ready.get('split_manifest_sha256')!=digest(cfg['split_manifest']):
        raise ValueError('dataset or fixed split identity changed; rebuild cache')
    print(json.dumps(dict(status='CACHE_VERIFIED',samples=ready['samples'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('command',choices=['scan','prepare','verify']);a=p.parse_args()
    cfg=load_config(a.config)
    if a.command=='scan':
        eps=scan(cfg);print(json.dumps(dict(episodes=len(eps),anchors=sum(len(e['anchors']) for e in eps))))
    elif a.command=='prepare': prepare(cfg,Path(__file__).resolve().parents[3])
    else:verify(cfg)

if __name__=='__main__':main()
