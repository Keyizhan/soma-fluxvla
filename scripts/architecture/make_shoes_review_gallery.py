"""Compact wrist-view evidence for explicitly selected visual anchors."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inspection',required=True)
    parser.add_argument('--choices',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    source=Path(args.inspection);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((source/'inspection.json').read_text())
    choices=json.loads(Path(args.choices).read_text())
    for episode,choice in choices.items():
        ep=next(x for x in metadata['episodes'] if x['episode']==int(episode))
        frames=[f for group in choice['frames'] for f in group]
        result=Image.new('RGB',(400,170*len(frames)),'white')
        for col,camera in enumerate(['cam_left_wrist','cam_right_wrist']):
            record=next(x for x in ep['cameras'] if x['camera']==camera)
            sheet=Image.open(source/record['sheet'])
            width=sheet.width//4;panel_height=width*3//4+26
            for row,frame in enumerate(frames):
                index=record['frames'].index(frame)
                x,y=(index%4)*width,(index//4)*panel_height
                tile=sheet.crop((x,y+22,x+width,y+22+width*3//4)).resize((200,150))
                result.paste(tile,(col*200,row*170+20))
                ImageDraw.Draw(result).text((col*200+3,row*170+3),f'ep{episode} f{frame} {"L" if col==0 else "R"}',fill='black')
        result.save(out/f'ep{int(episode):03d}_wrists.jpg',quality=60)


if __name__=='__main__':main()
