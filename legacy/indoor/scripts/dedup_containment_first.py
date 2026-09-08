"""交集占小框80%时先留大删小，随后才用关键点判断；不推翻面积阶段胜者。"""
import argparse,json
from collections import Counter
import numpy as np
import dedup_overlap_tracks as base
import audit_overlap_dedup as audit

CACHE=base.OUT/'pose_cache'
OUT=base.ROOT/'indoor_containment80_dedup_20260906'

def suppress(rank, duplicate, candidates):
    kept=[]; removed={}; candidates=set(candidates)
    for i in rank:
        if i not in candidates or i in removed:continue
        kept.append(i)
        for j in np.flatnonzero(duplicate[i]):
            j=int(j)
            if j in candidates and j not in kept and j not in removed:removed[j]=i
    return kept,removed

def process(video,threshold):
    paths=sorted((base.SOURCE/'frames'/video).glob('*.json'))
    raw=[json.loads(p.read_text()) for p in paths]
    support=Counter(d['track_id'] for row in raw for d in row['detections'] if d.get('origin')!='track_interpolation')
    stats={m:Counter() for m in ['area80','area80_pose']}
    for path,row in zip(paths,raw):
        ds=row['detections'];n=len(ds);inter,iou,ios,diag=base.geometry(ds)
        boxes=np.asarray([d['bbox_xyxy'] for d in ds]);wh=boxes[:,2:]-boxes[:,:2];area=wh.prod(1)
        rank=sorted(range(n),key=lambda i:(-area[i],-support[ds[i]['track_id']],ds[i]['track_id']))
        isolated=~(inter>0).any(1)
        area_kept,area_removed=suppress(rank,ios>=threshold,range(n))
        cache=json.loads((CACHE/video/path.name).read_text());kp=np.zeros((n,2,3));kp[cache['indexes']]=cache['points']
        margin=np.maximum(2.,.1*wh)
        inside=((kp[:,:,:2]>=(boxes[:,:2]-margin)[:,None,:])&(kp[:,:,:2]<=(boxes[:,2:]+margin)[:,None,:])).all(axis=(1,2))
        valid=inside&(kp[:,:,2].min(1)>=.15)&(kp[:,:,2].mean(1)>=.25)&(np.linalg.norm(kp[:,0,:2]-kp[:,1,:2],axis=1)>=.15*diag)
        distances=np.linalg.norm(kp[:,None,:,:2]-kp[None,:,:,:2],axis=3).max(2)
        duplicate=(inter>0)&(ios>=.5)&valid[:,None]&valid[None,:]&(distances<=.12*np.minimum(diag[:,None],diag[None,:]))
        # 面积阶段已选中的大框不再被关键点阶段删除，防止规则反转及间接连锁误删。
        protected=set(area_removed.values())
        if protected:duplicate[:,list(protected)]=False
        final_kept,pose_removed=suppress(rank,duplicate,area_kept)
        for method,kept,extra in [('area80',area_kept,{}),('area80_pose',final_kept,pose_removed)]:
            removed=[];keep_set=set(kept)
            for stage,pairs in [('containment',area_removed),('keypoints',extra)]:
                for i,j in pairs.items():
                    assert j in keep_set and area[j]>=area[i]
                    removed.append(dict(index=i,kept_index=j,track_id=ds[i]['track_id'],kept_track_id=ds[j]['track_id'],stage=stage,iou=float(iou[i,j]),intersection_over_smaller=float(ios[i,j]),removed_area=float(area[i]),kept_area=float(area[j]),max_keypoint_distance=float(distances[i,j]),detection=ds[i],fresh_keypoints=kp[i].tolist(),kept_fresh_keypoints=kp[j].tolist()))
            assert all(i in keep_set for i in np.flatnonzero(isolated))
            result=dict(row);result['detections']=[d for i,d in enumerate(ds) if i in keep_set];result['dedup_removed']=removed
            result['deduplication']=dict(method=method,before=n,after=len(kept),removed=len(removed),area_removed=len(area_removed),pose_removed=len(extra),isolated=int(isolated.sum()),isolated_kept=int(isolated.sum()),intersection_over_smaller_threshold=threshold,priority='area_desc_observed_support_desc_track_id',area_winners_protected_from_pose=True,keypoint_min=.15,keypoint_mean=.25,keypoint_distance_ratio=.12,keypoint_overlap_over_smaller=.5,keypoint_separation_ratio=.15,keypoint_bbox_margin='max(2px,10pct_side)',source=str(base.SOURCE))
            base.save(OUT/method/'frames'/video/path.name,result)
            st=stats[method];st.update(frames=1,before=n,after=len(kept),removed=len(removed),area_removed=len(area_removed),pose_removed=len(extra),isolated=int(isolated.sum()),isolated_kept=int(isolated.sum()),removed_interpolation=sum(e['detection'].get('origin')=='track_interpolation' for e in removed),removed_thin=int(sum(max(wh[e['index']])/max(min(wh[e['index']]),1e-6)>=4 or min(wh[e['index']])<=8 for e in removed)))
            st['removed_original']=st['removed']-st['removed_interpolation']
    base.save(OUT/f'{video}_summary.json',stats);print(json.dumps(stats),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--threshold',type=float,default=.8);a=p.parse_args()
    OUT.mkdir(exist_ok=True);process(a.video,a.threshold)
    base.OUT=OUT;audit.OUT=OUT
    for method in ['area80','area80_pose']:
        audit.main(a.video,method);base.render(a.video,method)
