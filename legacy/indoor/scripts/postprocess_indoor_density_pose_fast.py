"""复用全量框和关键点，只对动态密度候选做额外推理；原始结果不覆盖。"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT=Path('/root/autodl-tmp')
RAW=ROOT/'indoor_yolo_v3_pose_raw_allframes_20260905'
OUT=ROOT/'indoor_density128x80_pose_keeporiginal_20260905'
FRAMES=ROOT/'flywheel_indoor_scene_b_20260904/frames'
DATA=ROOT/'datasets/scene_B_123_v3_sliced_20260905'
sys.path[:0]=[str(ROOT/'flywheel_indoor_scene_b_20260904/full_eval_bundle/scripts'),str(ROOT/'flywheel_indoor_scene_b_20260904')]
from indoor_pose_fusion_full_eval_4090 import make_recheck_candidates,PoseVerifier

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
    tmp.replace(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--mode',choices=['pilot','all'],default='pilot')
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--baseline',action='store_true')
    args=p.parse_args()
    cv2.setNumThreads(2);torch.set_num_threads(4);torch.cuda.set_device(args.device)
    raw_manifest=json.loads((RAW/'run_manifest.json').read_text())
    model_path=Path(raw_manifest['detector'])
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==raw_manifest['detector_sha256']
    assert hashlib.sha256(Path(raw_manifest['pose']).read_bytes()).hexdigest()==raw_manifest['pose_sha256']
    params=SimpleNamespace(device=args.device,cell_w=128,cell_h=80,recheck_w=256,recheck_h=160,
                           neighbor_threshold=2,max_candidates=24,recheck_confidence=.05)
    if args.device==0 and not args.benchmark:
        save(OUT/'run_manifest.json',dict(source=str(RAW),detector_sha256=raw_manifest['detector_sha256'],
             pose_sha256=raw_manifest['pose_sha256'],parameters=vars(params),
             remove_original_when='disabled: pilot showed recall loss from low pose filtering',
             added_pose_min=.15,added_pose_mean=.25,added_separation_ratio=.15,
             box_expansion=False,optical_flow=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    if args.mode=='pilot':
        samples=json.loads((DATA/'split_manifest.json').read_text())['samples']
        paths=sorted({RAW/'pose'/r['video_id']/('frame_%08d.json' % int(Path(r['image']).stem.rsplit('_',1)[1])) for r in samples if r['split']!='train'})
        assert len(paths)==36
    else:
        paths=sorted((RAW/'pose').glob('*/*.json'))
        assert len(paths)==36010
    selected=paths[args.device::2]
    pending=selected if args.benchmark else [path for path in selected if not (OUT/'frames'/path.parent.name/path.name).exists()]
    model=YOLO(str(model_path))
    vit=ROOT/'flywheel_outdoor_pose_20260903/source/ViTPose'
    pose=PoseVerifier(vit,vit/'configs/bee_pose/subset_13_20260827/scene_B_only_13.py',Path(raw_manifest['pose']),args.device)
    if not args.baseline:
        from density_speed_helpers import prefetch,candidates as fast_candidates
        from mmpose.datasets.pipelines import Compose
        transforms=pose.pipeline.transforms
        loaders=[t for t in transforms if t.__class__.__name__=='LoadImageFromFile']
        assert len(loaders)==1 and loaders[0].channel_order=='rgb'
        def loaded(data):
            data['image_file']=None
            return data
        pose.pipeline=Compose([loaded if t in loaders else t for t in transforms])
    started=time.monotonic();count=added=removed=0
    def progress(completed=False):
        elapsed=time.monotonic()-started
        info=dict(mode=args.mode,device=args.device,new_frames=count,pending_at_start=len(pending),
                  shard_frames=len(selected),added=added,removed=removed,elapsed_seconds=elapsed,
                  fps=count/max(elapsed,.001),completed=completed)
        tag=f'bench_{args.baseline}_' if args.benchmark else ''
        save(OUT/f'{tag}{args.mode}_{args.device}_status.json',info)
        print(json.dumps(info),flush=True)
    iterator=pending if args.baseline else prefetch(pending,FRAMES,params)
    for item in iterator:
        if args.baseline:
            path=item;raw=json.loads(path.read_text())
            image=cv2.imread(str(FRAMES/path.parent.name/(path.stem+'.jpg')))
            assert image is not None
        else:path,raw,image,*_=item
        original=raw['detections']
        baseline=[d['bbox_xyxy']+[d['det_confidence']] for d in original]
        candidates,centers=make_recheck_candidates(model,image,baseline,params) if args.baseline else fast_candidates(model,item,params)
        points=pose(image if args.baseline else cv2.cvtColor(image,cv2.COLOR_BGR2RGB),candidates,batch_size=256)
        kept=[];deleted=[];rejected=0
        for idx,det in enumerate(original):
            # 低关键点分数不是误检的充分证据，小样本已证实会误删真蜂。
            kept.append(dict(original_id=idx,origin='baseline',**det))
        additions=[]
        for candidate,kp in zip(candidates,points):
            x1,y1,x2,y2,score=candidate
            separation=np.linalg.norm(kp[0,:2]-kp[1,:2])/max(math.hypot(x2-x1,y2-y1),1e-6)
            if kp[:,2].min()<.15 or kp[:,2].mean()<.25 or separation<.15:
                rejected+=1;continue
            additions.append(dict(origin='density',class_id=0,bbox_xyxy=candidate[:4],det_confidence=score,
                keypoints=dict(head=kp[0].tolist(),abdomen_tip=kp[1].tolist())))
        final=kept+additions
        assert len(final)==len(original)-len(deleted)+len(additions)
        record=dict(frame=raw['frame'],video=path.parent.name,width=raw['width'],height=raw['height'],
                    detections=final,removed_detections=deleted,density_centers=centers,
                    counts=dict(before=len(original),after=len(final),added=len(additions),removed=len(deleted),
                    density_candidates=len(candidates),rejected_candidates=rejected))
        folder=f'benchmark_{args.baseline}_{args.device}' if args.benchmark else 'frames'
        save(OUT/folder/path.parent.name/path.name,record)
        count+=1;added+=len(additions);removed+=len(deleted)
        if count%16==0:progress()
    progress(True)

if __name__=='__main__':main()
