"""Extract timestamped evidence for human/AI visual annotation; never invent labels."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', nargs='+', type=int, default=[0, 1, 49])
    parser.add_argument('--frames', nargs='*', type=int)
    parser.add_argument('--cameras', nargs='+', default=['cam_front', 'cam_left_wrist', 'cam_right_wrist'])
    parser.add_argument('--thumb-width', type=int, default=400)
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((data/'meta/info.json').read_text())
    cameras = args.cameras
    width=args.thumb_width; height=width*3//4; panel_height=height+26
    metadata = dict(data_root=str(data), source_info=info, episodes=[],
                    note='Contact sheets are visual evidence only; labels require inspection.')
    for episode in args.episodes:
        table = pq.read_table(data/f'data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet')
        times = np.asarray(table['timestamp'].to_pylist())
        indices = args.frames or np.linspace(0, len(times)-1, 24, dtype=int).tolist()
        if any(i < 0 or i >= len(times) for i in indices):
            raise ValueError('Frame outside episode')
        record = dict(episode=episode, length=len(times), columns=table.column_names,
                      timestamp_monotonic=bool(np.all(np.diff(times)>0)),
                      timestamp_range=[float(times[0]), float(times[-1])], cameras=[])
        for camera in cameras:
            video = data/f'videos/chunk-{episode//1000:03d}/observation.images.{camera}/episode_{episode:06d}.mp4'
            reader = cv2.VideoCapture(str(video))
            if not reader.isOpened():raise RuntimeError(f'Cannot open {video}')
            fps = reader.get(cv2.CAP_PROP_FPS)
            frames_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            images=[]
            for index in indices:
                reader.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = reader.read()
                if not ok:raise RuntimeError(f'Cannot decode {video}:{index}')
                image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                image.thumbnail((width,height))
                panel=Image.new('RGB',(width,panel_height),'white');panel.paste(image,(0,22))
                ImageDraw.Draw(panel).text((5,3),f'ep{episode} {camera} f{index} {times[index]:.3f}s',fill='black')
                images.append(panel)
            reader.release()
            cols=4;rows=(len(images)+cols-1)//cols
            sheet=Image.new('RGB',(cols*width,rows*panel_height),'#cccccc')
            for i,image in enumerate(images):sheet.paste(image,((i%cols)*width,(i//cols)*panel_height))
            path=out/f'ep{episode:03d}_{camera}_{indices[0]}_{indices[-1]}.jpg'
            sheet.save(path,quality=92)
            record['cameras'].append(dict(camera=camera,path=str(video),fps=fps,
                video_frames=frames_count,frames=indices,sheet=path.name,
                sha256=hashlib.sha256(video.read_bytes()).hexdigest()))
        metadata['episodes'].append(record)
    (out/'inspection.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':main()
