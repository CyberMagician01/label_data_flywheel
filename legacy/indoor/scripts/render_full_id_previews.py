"""从全量结果读取连续300帧，输出全局左右对比与同ID拖尾。"""
import argparse,json
from pathlib import Path
from collections import defaultdict,deque
import cv2,numpy as np
from dedup_overlap_tracks import color,IMAGES

ROOT=Path('/root/autodl-tmp/indoor_ids_full_latest_20260906')
SOURCE=Path('/root/autodl-tmp/indoor_yolo_v3_pose_raw_allframes_20260905/pose')
STARTS={'B-5-1':2418,'B-5-2':2418,'B-5-3':6283,'B-5-4':6275}

def render(video):
    cv2.setNumThreads(1);out=ROOT/'previews';out.mkdir(exist_ok=True)
    writer=cv2.VideoWriter(str(out/f'{video}_global_compare_source.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(3840,1140));assert writer.isOpened()
    history=[defaultdict(lambda:deque(maxlen=30)),defaultdict(lambda:deque(maxlen=30))]
    for n,f in enumerate(range(STARTS[video],STARTS[video]+300)):
        filename=f'frame_{f:08d}';rows=[json.loads((r/video/(filename+'.json')).read_text()) for r in [SOURCE,ROOT/'final/frames']]
        im=cv2.imread(str(IMAGES/video/(filename+'.jpg')));assert im is not None
        canvas=np.zeros((1140,3840,3),np.uint8)
        for side,row in enumerate(rows):
            panel=im.copy()
            for d in row['detections']:
                if side==0:
                    b=np.round(d['bbox_xyxy']).astype(int);cv2.rectangle(panel,tuple(b[:2]),tuple(b[2:]),(255,210,70),2)
                    continue
                tid=d['track_id'];b=np.round(d['bbox_xyxy']).astype(int);c=color(tid);pt=tuple(((b[:2]+b[2:])/2).astype(int));h=history[side][tid]
                if h and f-h[-1][0]>1:h.clear()
                h.append((f,pt))
                for a,z in zip(h,list(h)[1:]):cv2.line(panel,a[1],z[1],c,1,cv2.LINE_AA)
                cv2.rectangle(panel,tuple(b[:2]),tuple(b[2:]),c,2);cv2.putText(panel,str(tid),(max(0,b[0]),max(15,b[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,.48,c,1,cv2.LINE_AA)
            canvas[:1080,side*1920:(side+1)*1920]=panel
            label='BEFORE: raw model output' if side==0 else 'AFTER: complete cumulative pipeline'
            cv2.putText(canvas,f'{video} | Frame {f} | {label} | Boxes {len(row["detections"])}',(side*1920+14,1108),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            footer='Original model boxes | Raw output has no tracking IDs' if side==0 else '5s ID memory | Same-ID gaps <=3s filled | No post-fill deletion | 1s trail'
            cv2.putText(canvas,footer,(side*1920+14,1137),cv2.FONT_HERSHEY_SIMPLEX,.6,(230,230,230),1)
        writer.write(canvas)
        if n==150:cv2.imwrite(str(out/f'{video}_global_preview.jpg'),canvas)
    writer.release();cap=cv2.VideoCapture(str(out/f'{video}_global_compare_source.mp4'))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==300 and cap.get(cv2.CAP_PROP_FPS)==30
    for f in [0,150,299]:cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,_=cap.read();assert ok
    cap.release();print(json.dumps(dict(video=video,preview_frames=300,fps=30,start=STARTS[video],width=3840,height=1140)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);a=p.parse_args();render(a.video)
