"""保持密度/IoU 参数不变，提前准备裁剪并向量化候选与原框的重叠计算。"""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import cv2
import numpy as np
from indoor_adaptive_recheck_4090 import dynamic_low_density_centers,centered_crop_with_padding,iou

def prepare(path,frames,args):
    raw=json.loads(path.read_text())
    image=cv2.imread(str(frames/path.parent.name/(path.stem+'.jpg')))
    assert image is not None
    baseline=[d['bbox_xyxy']+[d['det_confidence']] for d in raw['detections']]
    h,w=image.shape[:2]
    centers=dynamic_low_density_centers(baseline,w,h,args.cell_w,args.cell_h,args.neighbor_threshold,max_candidates=args.max_candidates)
    crops=[];metadata=[]
    for cx,cy,d in centers:
        crop,ox,oy=centered_crop_with_padding(image,cx,cy,args.recheck_w,args.recheck_h)
        crops.append(crop);metadata.append((cx,cy,d,ox,oy))
    return path,raw,image,baseline,centers,crops,metadata

def prefetch(paths,frames,args):
    with ThreadPoolExecutor(max_workers=3) as pool:
        q=deque()
        for path in paths:
            q.append(pool.submit(prepare,path,frames,args))
            if len(q)>=6:yield q.popleft().result()
        while q:yield q.popleft().result()

def candidates(model,item,args):
    _,_,image,baseline,centers,crops,metadata=item
    if not crops:return [],centers
    results=model.predict(crops,imgsz=640,conf=args.recheck_confidence,iou=.45,device=args.device,half=True,batch=min(16,len(crops)),verbose=False)
    candidates=[];h,w=image.shape[:2]
    for result,(cx,cy,_,ox,oy) in zip(results,metadata):
        boxes=result.boxes.xyxy.detach().cpu().numpy().astype(np.float64)
        scores=result.boxes.conf.detach().cpu().numpy()
        if not len(boxes):continue
        valid=(boxes[:,0]>3)&(boxes[:,1]>3)&(boxes[:,2]<args.recheck_w-3)&(boxes[:,3]<args.recheck_h-3)
        full=boxes+np.array([ox,oy,ox,oy])
        bx=(full[:,0]+full[:,2])/2;by=(full[:,1]+full[:,3])/2
        valid&=(bx>=cx-args.cell_w/2)&(bx<cx+args.cell_w/2)&(by>=cy-args.cell_h/2)&(by<cy+args.cell_h/2)
        for box,score in zip(full[valid],scores[valid]):
            candidates.append([max(0.,box[0]),max(0.,box[1]),min(float(w),box[2]),min(float(h),box[3]),float(score)])
    candidates.sort(key=lambda x:x[4],reverse=True)
    # 向量化原框重叠测试；新增候选之间仍保持原来的顺序和贪心规则。
    if candidates and baseline:
        a=np.asarray(candidates)[:,:4];b=np.asarray(baseline)[:,:4]
        wh=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]))
        inter=wh[:,:,0]*wh[:,:,1]
        aa=np.maximum(0,a[:,2]-a[:,0])*np.maximum(0,a[:,3]-a[:,1])
        bb=np.maximum(0,b[:,2]-b[:,0])*np.maximum(0,b[:,3]-b[:,1])
        overlap=inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)
        blocked=(overlap>=.45).any(axis=1)
    else:blocked=np.zeros(len(candidates),dtype=bool)
    kept=[]
    for candidate,block in zip(candidates,blocked):
        if block or any(iou(candidate[:4],old[:4])>=.45 for old in kept):continue
        kept.append(candidate)
    return kept,centers
