"""核对去重计数、独立框不变，并展示包含边缘案例的局部对比。"""
import json
from collections import Counter
import cv2
import numpy as np
from dedup_overlap_tracks import SOURCE, OUT, IMAGES, geometry, save, color

def main(video, method):
    paths=sorted((OUT/method/'frames'/video).glob('*.json'))
    candidates=[]; seen=set(); audit=Counter(); id743=[]
    for path in paths:
        before=json.loads((SOURCE/'frames'/video/path.name).read_text())
        after=json.loads(path.read_text()); old=before['detections']; kept={d['track_id']:d for d in after['detections']}
        inter,_,_,_=geometry(old)
        for i,d in enumerate(old):
            if not (inter[i]>0).any(): assert kept[d['track_id']]==d; audit['isolated_unchanged']+=1
        old_by_id={d['track_id']:d for d in old}
        assert all(d==old_by_id[d['track_id']] for d in after['detections'])
        audit['frames']+=1
        audit['before']+=len(old);audit['after']+=len(kept)
        if 743 in kept:id743.append(int(path.stem.split('_')[1]))
        for e in after['dedup_removed']:
            assert e['kept_track_id'] in kept
            pair=tuple(sorted([e['track_id'],e['kept_track_id']]))
            if pair in seen:continue
            seen.add(pair); candidates.append((path,e))
    audit['id743_present_frames']=len(id743)
    audit['id743_internal_missing']=id743[-1]-id743[0]+1-len(id743) if id743 else 0
    save(OUT/f'{video}_{method}_audit.json',audit)
    # 按IoU低、中、高各取一例，避免只展示最容易的重复框。
    candidates.sort(key=lambda p:p[1]['iou'])
    selected=[candidates[i] for i in sorted(set([0,len(candidates)//2,len(candidates)-1]))] if candidates else []
    canvas=np.zeros((len(selected)*430,1280,3),np.uint8)
    for k,(path,e) in enumerate(selected):
        before=json.loads((SOURCE/'frames'/video/path.name).read_text());after=json.loads(path.read_text())
        im=cv2.imread(str(IMAGES/video/(path.stem+'.jpg')))
        bb=np.array(e['detection']['bbox_xyxy']); keeper=next(d for d in before['detections'] if d['track_id']==e['kept_track_id'])
        union=np.array([np.minimum(bb[:2],keeper['bbox_xyxy'][:2]),np.maximum(bb[2:],keeper['bbox_xyxy'][2:])]).reshape(-1)
        center=(union[:2]+union[2:])/2; w=max(160.,(union[2]-union[0])*1.6,(union[3]-union[1])*1.6*16/9);h=w*9/16
        x=int(np.clip(center[0]-w/2,0,1920-w));y=int(np.clip(center[1]-h/2,0,1080-h));w=int(w);h=int(h)
        for side,row in enumerate([before,after]):
            panel=cv2.resize(im[y:y+h,x:x+w],(640,360));sx=640/w;sy=360/h
            for d in row['detections']:
                a,b,c,z=d['bbox_xyxy']
                if c<x or a>x+w or z<y or b>y+h:continue
                box=[round((a-x)*sx),round((b-y)*sy),round((c-x)*sx),round((z-y)*sy)]
                col=(60,60,255) if d['track_id']==e['track_id'] else color(d['track_id'])
                cv2.rectangle(panel,tuple(box[:2]),tuple(box[2:]),col,2)
                cv2.putText(panel,str(d['track_id']),(max(0,box[0]),max(15,box[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,.5,col,1,cv2.LINE_AA)
            canvas[k*430+70:k*430+430,side*640:(side+1)*640]=panel
            cv2.putText(canvas,'BEFORE' if side==0 else 'AFTER | Removed: 1 highlighted box',(side*640+10,k*430+58),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        text=f'{video} {path.stem} | {method} | remove ID {e["track_id"]}, retain ID {e["kept_track_id"]}'
        cv2.putText(canvas,text,(10,k*430+24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    if selected:cv2.imwrite(str(OUT/f'{video}_{method}_examples.jpg'),canvas)
    save(OUT/f'{video}_{method}_examples.json',[dict(frame=p.stem,**e) for p,e in selected])

if __name__=='__main__':
    for video in ['B-5-1','B-5-3']:
        for method in ['nms70','keypoints']:main(video,method)
