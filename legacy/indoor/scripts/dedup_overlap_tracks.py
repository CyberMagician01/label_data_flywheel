"""只对相交框验证重复；独立框及低关键点证据框保留。另存两种消融结果。"""
import argparse, json, sys, time, colorsys
from pathlib import Path
from collections import Counter, defaultdict, deque
import cv2
import numpy as np

ROOT = Path('/root/autodl-tmp')
SOURCE = ROOT/'indoor_persistent_ids_filled_smoothed_20260906'
OUT = ROOT/'indoor_overlap_dedup_20260906'
IMAGES = ROOT/'flywheel_indoor_scene_b_20260904/frames'

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(',', ':')))

def geometry(rows):
    if not rows:return np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0)),np.zeros(0)
    b = np.array([d['bbox_xyxy'] for d in rows])
    wh = np.maximum(0, np.minimum(b[:, None, 2:], b[None, :, 2:])-np.maximum(b[:, None, :2], b[None, :, :2]))
    inter = wh.prod(2); np.fill_diagonal(inter, 0)
    area = (b[:, 2:]-b[:, :2]).prod(1)
    return inter, inter/np.maximum(area[:, None]+area[None, :]-inter, 1e-6), inter/np.maximum(np.minimum(area[:, None], area[None, :]), 1e-6), np.linalg.norm(b[:, 2:]-b[:, :2], axis=1)

def infer(video, device):
    import torch
    sys.path.insert(0, str(ROOT/'flywheel_indoor_scene_b_20260904'))
    from indoor_adaptive_recheck_4090 import PoseVerifier
    torch.set_num_threads(4); torch.cuda.set_device(device); cv2.setNumThreads(1)
    vit = ROOT/'flywheel_outdoor_pose_20260903/source/ViTPose'
    manifest = json.loads((ROOT/'indoor_yolo_v3_pose_raw_allframes_20260905/run_manifest.json').read_text())
    pose = PoseVerifier(vit, vit/'configs/bee_pose/subset_13_20260827/scene_B_only_13.py', Path(manifest['pose']), device)
    from mmpose.datasets.pipelines import Compose
    loaders = [t for t in pose.pipeline.transforms if t.__class__.__name__ == 'LoadImageFromFile']
    assert len(loaders) == 1 and loaders[0].channel_order == 'rgb'
    def loaded(data):
        data['image_file'] = None
        return data
    pose.pipeline = Compose([loaded if t in loaders else t for t in pose.pipeline.transforms])
    start = time.time(); total = 0
    for n, path in enumerate(sorted((SOURCE/'frames'/video).glob('*.json')), 1):
        target = OUT/'pose_cache'/video/path.name
        if target.exists(): continue
        rows = json.loads(path.read_text())['detections']
        inter, _, _, _ = geometry(rows)
        indexes = np.flatnonzero((inter > 0).any(1)).tolist()
        im = cv2.imread(str(IMAGES/video/(path.stem+'.jpg')))
        boxes = [rows[i]['bbox_xyxy']+[1.] for i in indexes]
        points = pose(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), boxes, batch_size=256) if boxes else np.empty((0, 2, 3))
        save(target, {'indexes':indexes, 'points':np.asarray(points).tolist()})
        total += len(indexes)
        if n % 25 == 0: print(json.dumps(dict(video=video, frames=n, crops=total, seconds=round(time.time()-start, 1))), flush=True)

def process(video):
    paths = sorted((SOURCE/'frames'/video).glob('*.json'))
    raw = [json.loads(p.read_text()) for p in paths]
    support = Counter(d['track_id'] for r in raw for d in r['detections'] if d.get('origin') != 'track_interpolation')
    # 整个片段固定优先级，避免置信度抖动使两个重复ID轮流胜出。
    stats = {m:Counter() for m in ['nms70', 'keypoints']}
    events = []
    for path, row in zip(paths, raw):
        ds = row['detections']; n = len(ds)
        inter, iou, ios, diag = geometry(ds)
        isolated = ~(inter > 0).any(1)
        cache = json.loads((OUT/'pose_cache'/video/path.name).read_text())
        kp = np.zeros((n, 2, 3)); kp[cache['indexes']] = cache['points']
        valid = (kp[:, :, 2].min(1) >= .15) & (kp[:, :, 2].mean(1) >= .25) & (np.linalg.norm(kp[:, 0, :2]-kp[:, 1, :2], axis=1) >= .15*diag)
        # ViTPose裁剪会扩展上下文，不能把邻蜂的框外关键点当作本框重复证据。
        boxes = np.asarray([d['bbox_xyxy'] for d in ds])
        margin = np.maximum(2., .1*(boxes[:, 2:]-boxes[:, :2]))
        inside = ((kp[:, :, :2] >= (boxes[:, :2]-margin)[:, None, :]) & (kp[:, :, :2] <= (boxes[:, 2:]+margin)[:, None, :])).all(axis=(1,2))
        valid &= inside
        distances = np.linalg.norm(kp[:, None, :, :2]-kp[None, :, :, :2], axis=3).max(2)
        kp_duplicate = (inter > 0) & (ios >= .5) & valid[:, None] & valid[None, :] & (distances <= .12*np.minimum(diag[:, None], diag[None, :]))
        rank = sorted(range(n), key=lambda i:(-support[ds[i]['track_id']], ds[i]['track_id']))
        for method, duplicate in [('nms70', iou >= .7), ('keypoints', kp_duplicate)]:
            kept = []; removed = []; suppressed = {}; decided = set()
            for i in rank:
                decided.add(i)
                if i not in suppressed:
                    kept.append(i)
                    for other in np.flatnonzero(duplicate[i]):
                        if other not in decided: suppressed.setdefault(int(other),i)
                    continue
                j = suppressed[i]
                removed.append(dict(index=i, kept_index=j, track_id=ds[i]['track_id'], kept_track_id=ds[j]['track_id'], iou=float(iou[i,j]), intersection_over_smaller=float(ios[i,j]), max_keypoint_distance=float(distances[i,j]), detection=ds[i], fresh_keypoints=kp[i].tolist(), kept_fresh_keypoints=kp[j].tolist()))
            keep_set = set(kept)
            assert all(i in keep_set for i in np.flatnonzero(isolated))
            result = dict(row)
            result['detections'] = [d for i,d in enumerate(ds) if i in keep_set]
            result['dedup_removed'] = removed
            result['deduplication'] = dict(method=method, before=n, after=len(kept), removed=len(removed), isolated=int(isolated.sum()), isolated_kept=int(isolated.sum()), nms_iou=.7, keypoint_distance_ratio=.12, keypoint_overlap_over_smaller=.5, keypoint_min=.15, keypoint_mean=.25, keypoint_separation_ratio=.15, keypoint_bbox_margin='max(2px,10pct_side)', priority='observed_frame_count_desc_then_track_id', source=str(SOURCE))
            save(OUT/method/'frames'/video/path.name, result)
            st = stats[method]; st.update(frames=1,before=n,after=len(kept),removed=len(removed),isolated=int(isolated.sum()),isolated_kept=int(isolated.sum()), removed_interpolation=sum(e['detection'].get('origin')=='track_interpolation' for e in removed))
            st['removed_original'] = st['removed']-st['removed_interpolation']
            for e in removed:
                if len(events) < 10000: events.append(dict(video=video, frame=path.stem, method=method, **e))
    save(OUT/f'{video}_summary.json', stats)
    save(OUT/f'{video}_events.json', events)
    print(json.dumps(stats), flush=True)

def color(tid):
    return tuple(round(255*x) for x in colorsys.hsv_to_rgb((tid*.61803398875)%1,.7,1)[::-1])

def render(video, method, before_label='BEFORE dedup (filled + smoothed)', footer='Same ID / same color | 1s trail | Isolated boxes preserved'):
    cv2.setNumThreads(1)
    roi = json.loads((ROOT/'indoor_stable_id_test_20260906'/f'{video}_summary.json').read_text())['roi_xywh']
    rx, ry, _, _ = roi
    writer = cv2.VideoWriter(str(OUT/f'{video}_{method}_compare_source.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),30,(3840,1140))
    assert writer.isOpened()
    history = [defaultdict(lambda:deque(maxlen=30)),defaultdict(lambda:deque(maxlen=30))]
    for k,path in enumerate(sorted((OUT/method/'frames'/video).glob('*.json'))):
        before = json.loads((SOURCE/'frames'/video/path.name).read_text()); after = json.loads(path.read_text())
        f = int(path.stem.split('_')[1]); im = cv2.imread(str(IMAGES/video/(path.stem+'.jpg')))
        canvas = np.zeros((1140,3840,3),np.uint8)
        for side,row in enumerate([before,after]):
            panel = cv2.resize(im[ry:ry+540,rx:rx+960],(1920,1080))
            for d in row['detections']:
                tid=d['track_id']; x1,y1,x2,y2=d['bbox_xyxy']
                b=[round(2*(x1-rx)),round(2*(y1-ry)),round(2*(x2-rx)),round(2*(y2-ry))]
                point=(round((b[0]+b[2])/2),round((b[1]+b[3])/2)); hist=history[side][tid]
                if hist and f-hist[-1][0]>1: hist.clear()
                hist.append((f,point))
                if not rx <= (x1+x2)/2 < rx+960 or not ry <= (y1+y2)/2 < ry+540: continue
                c=color(tid)
                for p,q in zip(hist,list(hist)[1:]): cv2.line(panel,p[1],q[1],c,1,cv2.LINE_AA)
                cv2.rectangle(panel,tuple(b[:2]),tuple(b[2:]),c,2)
                cv2.putText(panel,str(tid),(max(0,b[0]),max(15,b[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,.48,c,1,cv2.LINE_AA)
            canvas[:1080,side*1920:(side+1)*1920]=panel
            roi_removed=sum(rx <= (e['detection']['bbox_xyxy'][0]+e['detection']['bbox_xyxy'][2])/2 < rx+960 and ry <= (e['detection']['bbox_xyxy'][1]+e['detection']['bbox_xyxy'][3])/2 < ry+540 for e in after['dedup_removed'])
            title=f'{video} | Frame {f} | '+(before_label if side==0 else f'{method} | Removed: {len(after["dedup_removed"])} full / {roi_removed} view')
            cv2.putText(canvas,title,(side*1920+16,1108),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2)
            cv2.putText(canvas,footer,(side*1920+16,1137),cv2.FONT_HERSHEY_SIMPLEX,.65,(230,230,230),1)
        writer.write(canvas)
        if k==150: cv2.imwrite(str(OUT/f'{video}_{method}_preview.jpg'),canvas)
    writer.release()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--device',type=int,default=0);p.add_argument('--stage',choices=['infer','process','render'],required=True);p.add_argument('--method',default='keypoints');a=p.parse_args();OUT.mkdir(exist_ok=True)
    if a.stage=='infer':infer(a.video,a.device)
    elif a.stage=='process':process(a.video)
    else:render(a.video,a.method)
