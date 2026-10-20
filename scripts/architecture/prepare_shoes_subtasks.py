"""Build only inspected anchors; train-only normalization and timestamp checks."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import decord
import numpy as np
import pyarrow.parquet as pq


JOINT_DELTA_MASK = np.array([True]*6+[False]+[True]*6+[False])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--annotations',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--normalization-from',help='Freeze statistics from an earlier train-only fixture manifest')
    args=parser.parse_args()
    annotations=json.loads(Path(args.annotations).read_text())
    root=Path(annotations['dataset_root']);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    tables={ep['episode']:pq.read_table(root/f'data/chunk-000/episode_{ep["episode"]:06d}.parquet').to_pydict()
            for ep in annotations['episodes']}
    def array(table,key):return np.asarray(table[key],dtype=np.float32)
    states_for_stats=[];actions_for_stats=[]
    for ep in annotations['episodes']:
        if ep['split']!='train':continue
        table=tables[ep['episode']]
        end=ep['segments'][-1]['frame_range'][1]
        states=array(table,'observation.state')[:end];actions=array(table,'action')[:end]
        states_for_stats.append(states)
        # Match base dataset's t+1 ... t+50 window, then current-state joint delta.
        current=states[:-50]
        for offset in range(1,51):
            window=actions[offset:offset+len(current)].copy()
            window[:,JOINT_DELTA_MASK]-=current[:,JOINT_DELTA_MASK]
            actions_for_stats.append(window)
    norm={}
    for key,parts in [('state',states_for_stats),('action',actions_for_stats)]:
        values=np.concatenate(parts)
        norm[key]=dict(q01=np.quantile(values,.01,axis=0).tolist(),q99=np.quantile(values,.99,axis=0).tolist(),count=len(values))
    norm.update(source='Computed from train episodes 0/1 only, before post-completion exclusion',
                method='quantile, epsilon=1e-6, no clipping, normalized 14 dimensions then zero-pad to32',
                action_window='t+1...t+50',joint_delta_mask=JOINT_DELTA_MASK.tolist())
    if args.normalization_from:
        norm=json.loads(Path(args.normalization_from).read_text())['normalization']
    def normalize(values,key):
        stats=norm[key];low=np.array(stats['q01'],dtype=np.float32);high=np.array(stats['q99'],dtype=np.float32)
        result=(values-low)/(high-low+1e-6)*2-1
        return np.pad(result,[(0,0)]*(result.ndim-1)+[(0,18)]).astype(np.float32)
    records=[];images=[];states=[];actions=[];masks=[];alignment=[]
    for ep in annotations['episodes']:
        episode=ep['episode'];table=tables[episode]
        anchors=[(segment,frame) for segment in ep['segments'] for frame in segment['sample_frames']]
        indices=[frame for _,frame in anchors]
        camera_images=[]
        for camera in annotations['cameras']:
            path=root/f'videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4'
            reader=decord.VideoReader(str(path),num_threads=1)
            pts=reader.get_frame_timestamp(indices)[:,0]
            expected=np.array(table['timestamp'])[indices]
            error=float(np.max(np.abs(pts-expected)))
            if error>1/60:raise ValueError(f'Video/parquet alignment failed: {path}: {error}')
            frames=reader.get_batch(indices).asnumpy()
            camera_images.append(np.stack([cv2.resize(frame,(224,224),interpolation=cv2.INTER_LINEAR).transpose(2,0,1) for frame in frames]))
            alignment.append(dict(episode=episode,camera=camera,max_timestamp_error_s=error,
                                  frames_in_video=len(reader),parquet_frames=len(table['timestamp'])))
            if len(reader)!=len(table['timestamp']):raise ValueError('Video length mismatch')
        all_states=array(table,'observation.state');all_actions=array(table,'action')
        for i,(segment,index) in enumerate(anchors):
            if int(table['frame_index'][index])!=index or int(table['episode_index'][index])!=episode:
                raise ValueError('Parquet row/frame/episode index mismatch')
            if index+50>=segment['frame_range'][1]:raise ValueError('Action window crosses uncertain segment boundary')
            window=all_actions[index+1:index+51].copy()
            window[:,JOINT_DELTA_MASK]-=all_states[index,JOINT_DELTA_MASK]
            images.append(np.concatenate([camera[i] for camera in camera_images],axis=0))
            states.append(normalize(all_states[index],'state'))
            actions.append(normalize(window,'action'))
            masks.append(np.broadcast_to(np.arange(32)<14,(50,32)).copy())
            records.append(dict(sample_id=f'ep{episode:03d}_f{index:04d}',episode=episode,frame=index,
                timestamp=float(table['timestamp'][index]),split=ep['split'],stage=segment['stage'],
                subtask=segment['label'],overall_task=annotations['overall_task'],annotation_source='AI_visual_annotation'))
    payload=dict(images=np.stack(images),states=np.stack(states),actions=np.stack(actions),
                 action_masks=np.stack(masks),img_masks=np.ones((len(images),3),dtype=bool))
    if not all(np.isfinite(x).all() for x in payload.values()):raise ValueError('Nonfinite data')
    np.savez_compressed(out/'samples.npz',**payload)
    manifest=dict(records=records,normalization=norm,alignment=alignment,
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in payload.items()},
        input_annotation_sha256=hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest(),
        parquet_sha256={str(ep):hashlib.sha256((root/f'data/chunk-000/episode_{ep:06d}.parquet').read_bytes()).hexdigest() for ep in tables},
        preprocessing='decord RGB; OpenCV INTER_LINEAR224; image norm uint8/255*2-1 applied by training script; base future-action/delta conventions',
        norm_stats_warning='Tiny train-only fitted statistics differ from production shoes statistics; no comparison to the existing task model is claimed.')
    if args.normalization_from:
        manifest['normalization_frozen_from']=str(Path(args.normalization_from).resolve())
    (out/'samples.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(samples=len(records),train=sum(r['split']=='train' for r in records),
        validation=sum(r['split']=='validation' for r in records),max_pts_error=max(x['max_timestamp_error_s'] for x in alignment)),indent=2))


if __name__=='__main__':main()
