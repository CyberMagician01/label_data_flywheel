"""双卡分片：全量检测完成后再预测头尾关键点，保留全部原始预测。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque

ROOT = Path('/root/autodl-tmp')
WORK = ROOT / 'indoor_yolo_v3_pose_raw_allframes_20260905'
FRAMES = ROOT / 'flywheel_indoor_scene_b_20260904/frames'
MODEL = ROOT / 'indoor_scene_b_yolov8x_p2_v3_123_20260905/runs/scene_b_123_v3_yolov8x_p2_seed2026/weights/best.pt'
POSE = ROOT / 'flywheel_indoor_scene_b_20260904/weights/indoor_pose_epoch_8.pth'
VIT = ROOT / 'flywheel_outdoor_pose_20260903/source/ViTPose'
SHA = 'c3c18aa0df794c3980d74aa16181e12dac4cf9f8f9f2a5113669499587c2a1d4'

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')))
    temp.replace(path)

def paths():
    result = []
    for video, count in zip(['B-5-1','B-5-2','B-5-3','B-5-4'], [8996,8996,9013,9005]):
        items = sorted((FRAMES/video).glob('frame_*.jpg'))
        assert len(items) == count
        assert [p.stem for p in items] == [f'frame_{i:08d}' for i in range(count)]
        result.extend(items)
    return result

def output(stage, path):
    return WORK/stage/path.parent.name/(path.stem+'.json')

def prefetch(fn, items, workers=4, ahead=8):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        iterator = iter(items)
        for item in iterator:
            pending.append(pool.submit(fn,item))
            if len(pending) >= ahead:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()

def worker(args):
    import cv2
    import numpy as np
    import torch
    import torchvision
    cv2.setNumThreads(1)
    torch.set_num_threads(args.cpu_threads)
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.benchmark = True
    device = f'cuda:{args.device}'
    selected = paths()[args.device::2]
    if args.benchmark:
        selected=[p for p in selected if output('pose',p).exists()]
        selected=selected[::max(1,len(selected)//32)][:32]
    elif args.limit:
        # 测速覆盖四个视频。
        selected = selected[::max(1,len(selected)//args.limit)][:args.limit]
    if not args.benchmark:
        selected = [p for p in selected if not output(args.stage,p).exists()]
    count = instances = 0
    start = time.monotonic()
    def progress(done=False):
        elapsed = time.monotonic()-start
        state = dict(stage=args.stage, device=args.device, new_frames=count,
                     scheduled_frames=len(selected), instances=instances,
                     elapsed_seconds=elapsed, fps=count/max(elapsed,1),
                     batch=args.batch, completed=done,
                     peak_gpu_mb=torch.cuda.max_memory_allocated()/1024**2)
        suffix=f'bench_{args.baseline}_' if args.benchmark else ''
        save(WORK/f'{suffix}{args.stage}_{args.device}_status.json',state)
        print(json.dumps(state), flush=True)
    if args.stage == 'detect':
        from ultralytics import YOLO
        assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == SHA
        model = YOLO(str(MODEL)).model.to(device).half().eval()
        def prepare(path):
            image = cv2.imread(str(path))
            h,w = image.shape[:2]
            assert (h,w) == (1080,1920)
            def starts(n):
                a=list(range(0,n-640+1,512))
                if a[-1] != n-640: a.append(n-640)
                return a
            coords=[(x,y) for y in starts(h) for x in starts(w)]
            crops=np.stack([cv2.cvtColor(image[y:y+640,x:x+640],cv2.COLOR_BGR2RGB) for x,y in coords])
            return path,coords,torch.from_numpy(crops).permute(0,3,1,2).contiguous()
        def infer(items):
            nonlocal count,instances
            tensor=torch.cat([i[2] for i in items]).pin_memory().to(device,non_blocking=True).half()/255
            with torch.inference_mode():
                pred=model(tensor)
                if isinstance(pred,(tuple,list)): pred=pred[0]
                pred=pred.transpose(1,2)
                offset=0
                for path,coords,_ in items:
                    boxes=[];scores=[]
                    for tile,(x,y) in zip(pred[offset:offset+len(coords)],coords):
                        tile=tile[tile[:,4]>.25]
                        box=torch.cat([tile[:,:2]-tile[:,2:4]/2,tile[:,:2]+tile[:,2:4]/2],1)
                        box[:,[0,2]]=(box[:,[0,2]]+x).clamp(0,1920)
                        box[:,[1,3]]=(box[:,[1,3]]+y).clamp(0,1080)
                        boxes.append(box);scores.append(tile[:,4])
                    offset+=len(coords)
                    boxes=torch.cat(boxes);scores=torch.cat(scores)
                    keep=torchvision.ops.nms(boxes,scores,.25)
                    values=torch.cat([boxes[keep],scores[keep,None]],1).float().cpu().numpy()
                    assert np.isfinite(values).all()
                    detections=[dict(bbox_xyxy=b[:4].tolist(),det_confidence=float(b[4]),class_id=0) for b in values]
                    label=WORK/'labels'/path.parent.name/(path.stem+'.txt')
                    label.parent.mkdir(parents=True,exist_ok=True)
                    label.write_text(''.join(f'0 {(b[0]+b[2])/3840:.8f} {(b[1]+b[3])/2160:.8f} {(b[2]-b[0])/1920:.8f} {(b[3]-b[1])/1080:.8f} {b[4]:.8f}\n' for b in values))
                    save(output('detect',path),dict(frame=path.name,video=path.parent.name,width=1920,height=1080,detections=detections))
                    count+=1;instances+=len(values)
            if count % 64 < len(items): progress()
        batch=[]
        for item in prefetch(prepare,selected):
            batch.append(item)
            if len(batch)==args.batch:
                infer(batch);batch=[]
        if batch: infer(batch)
    else:
        sys.path.insert(0,str(ROOT/'flywheel_indoor_scene_b_20260904'))
        from indoor_adaptive_recheck_4090 import PoseVerifier
        pose=PoseVerifier(VIT,VIT/'configs/bee_pose/subset_13_20260827/scene_B_only_13.py',POSE,args.device)
        if not args.baseline:
            from mmpose.datasets.pipelines import Compose
            norms=[t for t in pose.model.cfg.test_pipeline if t['type']=='NormalizeTensor']
            assert len(norms)==1 and list(norms[0]['mean'])==[.485,.456,.406] and list(norms[0]['std'])==[.229,.224,.225]
            transforms=pose.pipeline.transforms
            loaders=[t for t in transforms if t.__class__.__name__=='LoadImageFromFile']
            assert len(loaders)==1 and loaders[0].channel_order=='rgb' and loaders[0].color_type=='color'
            def uint8_tensor(data):
                data['img']=torch.from_numpy(data['img']).permute(2,0,1).contiguous()
                return data
            converted=[]
            for transform in transforms:
                name=transform.__class__.__name__
                if name in ['LoadImageFromFile','NormalizeTensor']:
                    continue
                converted.append(uint8_tensor if name=='ToTensor' else transform)
            assert sum(t.__class__.__name__=='NormalizeTensor' for t in transforms)==1
            pose.pipeline=Compose(converted)
            mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
            std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
        def prepare(path):
            record=json.loads(output('detect',path).read_text())
            image=cv2.imread(str(path)); prepared=[]
            if not args.baseline:
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            for idx,det in enumerate(record['detections']):
                x1,y1,x2,y2=det['bbox_xyxy']
                box=np.asarray([x1,y1,x2-x1+1,y2-y1+1,det['det_confidence']],dtype=np.float32)
                center,scale=pose.box2cs(pose.model.cfg,box)
                data=dict(center=center,scale=scale,bbox_score=float(box[4]),bbox_id=idx,
                    dataset=pose.dataset_info.dataset_name,joints_3d=np.zeros((2,3),np.float32),
                    joints_3d_visible=np.zeros((2,3),np.float32),rotation=0,
                    ann_info=dict(image_size=np.asarray(pose.model.cfg.data_cfg.image_size),num_joints=2,flip_pairs=pose.dataset_info.flip_pairs),img=image)
                if not args.baseline:
                    data['image_file']=None
                prepared.append(pose.pipeline(data))
            return path,record,prepared
        def infer(items):
            nonlocal count,instances
            flat=[row for _,_,prepared in items for row in prepared]
            predictions=[]
            for idx in range(0,len(flat),args.batch):
                chunk=flat[idx:idx+args.batch]
                batch=pose.collate(chunk,samples_per_gpu=len(chunk))
                batch=pose.scatter(batch,[next(pose.model.parameters()).device])[0]
                if not args.baseline:
                    batch['img']=(batch['img'].float()/255-mean)/std
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    result=pose.model(img=batch['img'],img_metas=batch['img_metas'],return_loss=False,return_heatmap=False)
                predictions.extend(np.asarray(result['preds'],dtype=np.float32))
            assert len(predictions)==len(flat)
            offset=0
            for path,record,prepared in items:
                for det,points in zip(record['detections'],predictions[offset:offset+len(prepared)]):
                    assert points.shape==(2,3) and np.isfinite(points).all()
                    det['keypoints']=dict(head=points[0].tolist(),abdomen_tip=points[1].tolist())
                offset+=len(prepared)
                if args.benchmark:
                    save(WORK/f'benchmark_{args.baseline}_{args.device}'/path.parent.name/(path.stem+'.json'),record)
                else:
                    save(output('pose',path),record)
                count+=1;instances+=len(prepared)
            if count % 32 < len(items): progress()
        block=[]
        for item in prefetch(prepare,selected,workers=4,ahead=8):
            block.append(item)
            if len(block)==4:
                infer(block);block=[]
        if block: infer(block)
    progress(True)

def supervise(args):
    files=paths()
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == SHA
    manifest=dict(frame_count=len(files),detector=str(MODEL),detector_sha256=SHA,
                  pose=str(POSE),pose_sha256=hashlib.sha256(POSE.read_bytes()).hexdigest(),
                  detection_confidence=.25,nms_iou=.25,tile_size=640,tile_stride=512,
                  postprocessing=False,keypoint_filter=False,shards=2,
                  keypoints=['head','abdomen_tip'],det_batch=args.det_batch,pose_batch=args.pose_batch)
    manifest['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['optimization']={'rgb_conversion':'once_per_frame','normalization':'gpu_float32','crop_transfer':'uint8','cpu_threads':4,'flip_test':'unchanged'}
    save(WORK/'run_manifest.json',manifest)
    for stage,batch in [('detect',args.det_batch),('pose',args.pose_batch)]:
        save(WORK/'status.json',dict(stage=stage,started_at=time.time(),completed=False))
        jobs=[]
        for gpu in [0,1]:
            log=open(WORK/f'{stage}_{gpu}.log','a')
            proc=subprocess.Popen([sys.executable,'-u',__file__,'--stage',stage,'--device',str(gpu),'--batch',str(batch)],stdout=log,stderr=subprocess.STDOUT)
            jobs.append((proc,log))
        codes=[p.wait() for p,_ in jobs]
        for _,log in jobs: log.close()
        if any(codes):
            save(WORK/'status.json',dict(stage=stage,failed=True,exit_codes=codes))
            raise RuntimeError(f'{stage}: {codes}')
        total=0
        for path in files:
            record=json.loads(output(stage,path).read_text())
            total+=len(record['detections'])
            if stage=='detect':
                label=WORK/'labels'/path.parent.name/(path.stem+'.txt')
                assert len(label.read_text().splitlines())==len(record['detections'])
            if stage=='pose':
                baseline=json.loads(output('detect',path).read_text())
                assert len(record['detections'])==len(baseline['detections'])
                for a,b in zip(record['detections'],baseline['detections']):
                    assert a['bbox_xyxy']==b['bbox_xyxy'] and a['det_confidence']==b['det_confidence']
                    assert set(a['keypoints'])=={'head','abdomen_tip'}
        save(WORK/f'{stage}_COMPLETE.json',dict(frames=len(files),instances=total,completed_at=time.time()))
    save(WORK/'status.json',dict(stage='complete',frames=len(files),instances=total,completed=True))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['detect','pose','supervise'],required=True)
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--det-batch',type=int,default=4)
    p.add_argument('--pose-batch',type=int,default=256)
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--baseline',action='store_true')
    p.add_argument('--cpu-threads',type=int,default=4)
    args=p.parse_args();WORK.mkdir(parents=True,exist_ok=True)
    if args.stage=='supervise': supervise(args)
    else: worker(args)
