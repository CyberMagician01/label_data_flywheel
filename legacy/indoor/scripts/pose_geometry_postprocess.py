"""CPU 关键点去重和定向扩框；自动接收随后完成的密度结果。"""
import argparse
import copy
import json
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path('/root/autodl-tmp')
RAW=ROOT/'indoor_yolo_v3_pose_raw_allframes_20260905'
DENSITY=ROOT/'indoor_density128x80_pose_keeporiginal_20260905'
OUT=ROOT/'indoor_density_pose_geometry_20260905'

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
    temp.replace(path)

def transform(record):
    detections=record['detections'];n=len(detections)
    if not n:return [],[],0
    boxes=np.asarray([d['bbox_xyxy'] for d in detections],dtype=float)
    kp=np.asarray([[d['keypoints']['head'],d['keypoints']['abdomen_tip']] for d in detections],dtype=float)
    wh=boxes[:,2:]-boxes[:,:2];area=wh[:,0]*wh[:,1];diag=np.linalg.norm(wh,axis=1)
    valid=(kp[:,:,2].min(axis=1)>=.15)&(kp[:,:,2].mean(axis=1)>=.25)&(np.linalg.norm(kp[:,0,:2]-kp[:,1,:2],axis=1)>=.15*diag)&np.isfinite(kp).all(axis=(1,2))&(area>0)
    ids=np.flatnonzero(valid)
    neighbors={}
    if len(ids):
        tree=cKDTree(kp[ids,0,:2])
        for local in tree.query_pairs(.15*diag[ids].max()):
            i,j=ids[list(local)]
            limit=.15*min(diag[i],diag[j])
            if np.linalg.norm(kp[i,0,:2]-kp[j,0,:2])>limit or np.linalg.norm(kp[i,1,:2]-kp[j,1,:2])>limit:continue
            inter=np.maximum(0,np.minimum(boxes[i,2:],boxes[j,2:])-np.maximum(boxes[i,:2],boxes[j,:2])).prod()
            if inter/min(area[i],area[j])<.75:continue
            neighbors.setdefault(int(i),[]).append(int(j));neighbors.setdefault(int(j),[]).append(int(i))
    # 只删除与最终保留框直接匹配的框，避免链式合并相邻蜜蜂。
    order=sorted(range(n),key=lambda i:(-area[i],-detections[i]['det_confidence'],i))
    rank={idx:r for r,idx in enumerate(order)};deleted={}
    for i in order:
        if i in deleted:continue
        for j in neighbors.get(i,[]):
            if rank[j]>rank[i] and j not in deleted:deleted[j]=i
    kept=[];expanded=0
    for i,d in enumerate(detections):
        if i in deleted:continue
        row=copy.deepcopy(d);row['geometry_source_index']=i
        if valid[i]:
            box=boxes[i].copy();width,height=wh[i]
            low=kp[i,:,:2].min(axis=0);high=kp[i,:,:2].max(axis=0)
            limits=np.array([width,height,width,height])*.10
            requested=np.array([low[0]-.10*width,low[1]-.10*height,high[0]+.10*width,high[1]+.10*height])
            box[:2]=np.maximum(box[:2]-limits[:2],np.minimum(box[:2],requested[:2]))
            box[2:]=np.minimum(box[2:]+limits[2:],np.maximum(box[2:],requested[2:]))
            box[[0,2]]=box[[0,2]].clip(0,record['width']);box[[1,3]]=box[[1,3]].clip(0,record['height'])
            if not np.array_equal(box,boxes[i]):
                row['pre_expansion_bbox_xyxy']=boxes[i].tolist()
                row['bbox_xyxy']=box.tolist();row['expanded_edges']=[edge for j,edge in enumerate(['left','top','right','bottom']) if box[j]!=boxes[i,j]]
                expanded+=1
        kept.append(row)
    removed=[dict(source_index=i,kept_source_index=j,reason='same_head_tail_and_containment',detection=detections[i]) for i,j in sorted(deleted.items())]
    return kept,removed,expanded

def process(key):
    density=DENSITY/'frames'/key
    source=density if density.exists() else RAW/'pose'/key
    stage='density' if source==density else 'raw'
    record=json.loads(source.read_text())
    final,removed,expanded=transform(record)
    result=dict(record)
    result['density_counts']=result.pop('counts',{})
    result['detections']=final
    result['geometry']=dict(source_stage=stage,before=len(record['detections']),after=len(final),removed=len(removed),expanded=expanded)
    result['geometry_removed_detections']=removed
    result['counts']=dict(before=len(record['detections']),after=len(final),removed=len(removed),expanded=expanded,added=0)
    assert len(final)+len(removed)==len(record['detections'])
    for row in final:
        original=record['detections'][row['geometry_source_index']]
        assert row['keypoints']==original['keypoints'] and row['det_confidence']==original['det_confidence']
        a=original['bbox_xyxy'];b=row['bbox_xyxy'];w=a[2]-a[0];h=a[3]-a[1]
        assert b[0]<=a[0] and b[1]<=a[1] and b[2]>=a[2] and b[3]>=a[3]
        assert a[0]-b[0]<=.1*w+1e-6 and a[1]-b[1]<=.1*h+1e-6 and b[2]-a[2]<=.1*w+1e-6 and b[3]-a[3]<=.1*h+1e-6
    save(OUT/'frames'/key,result)
    return str(key),stage,len(removed),expanded

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    upstream=json.loads((DENSITY/'status.json').read_text())
    assert upstream.get('completed') and upstream['frames']==36010
    save(OUT/'run_manifest.json',dict(raw=str(RAW),density=str(DENSITY),pose_min=.15,pose_mean=.25,
         separation_ratio=.15,head_tail_distance_ratio=.15,containment=.75,edge_margin=.10,max_expansion_per_edge=.10,
         order=['deduplicate_original_geometry','directional_expansion'],workers=8))
    paths=[p.relative_to(RAW/'pose') for p in sorted((RAW/'pose').glob('*/*.json'))]
    if args.pilot:
        samples=json.loads((ROOT/'datasets/scene_B_123_v3_sliced_20260905/split_manifest.json').read_text())['samples']
        paths=sorted({Path(r['video_id'])/('frame_%08d.json'%int(Path(r['image']).stem.rsplit('_',1)[1])) for r in samples if r['split']!='train'})
    latest={};start=time.monotonic()
    with ProcessPoolExecutor(max_workers=8) as pool:
        def consume(todo):
            for key,stage,removed,expanded in pool.map(process,todo,chunksize=32):
                latest[key]=(stage,removed,expanded)
                if len(latest)%1000==0:status('processing')
        def status(stage):
            data=dict(stage=stage,completed=stage=='complete',frames=len(latest),total_frames=len(paths),
                 density_frames=sum(v[0]=='density' for v in latest.values()),
                 removed=sum(v[1] for v in latest.values()),expanded=sum(v[2] for v in latest.values()),elapsed_seconds=time.monotonic()-start)
            if stage=='complete' and not args.pilot:
                assert len(latest)==36010
                data.update(before=upstream['after'],after=upstream['after']-data['removed'])
            save(OUT/('pilot_status.json' if args.pilot else 'status.json'),data)
            print(json.dumps(data),flush=True)
        consume(paths)
        if args.pilot:status('complete');return
        while True:
            todo=[path for path in paths if latest[str(path)][0]!='density' and (DENSITY/'frames'/path).exists()]
            if todo:consume(todo)
            upstream=json.loads((DENSITY/'status.json').read_text())
            if upstream.get('completed') and all(v[0]=='density' for v in latest.values()):status('complete');break
            if upstream.get('stage')=='failed':status('upstream_failed');raise RuntimeError('density stage failed')
            status('geometry_ready_waiting_density');time.sleep(30)

if __name__=='__main__':main()
