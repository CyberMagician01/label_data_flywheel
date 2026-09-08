"""最终框全量复核：包含优先、头尾一致、窄框借用邻蜂关键点、连续帧确认。"""
import argparse,json,sys,time
from collections import Counter,defaultdict,deque
import cv2,numpy as np
from dedup_overlap_tracks import ROOT,IMAGES,geometry,save,color
from reassociate_final_bees import metrics

SOURCE=ROOT/'indoor_interior_memory_20260906/offline_birth4_edge150_inside150_fill90/final/frames'
OUT=ROOT/'indoor_final_temporal_dedup_20260906'
CANDIDATE_IOS=.35

def infer(video,device):
    import torch
    torch.set_num_threads(4);torch.cuda.set_device(device);cv2.setNumThreads(1)
    sys.path.insert(0,str(ROOT/'flywheel_indoor_scene_b_20260904'))
    from indoor_adaptive_recheck_4090 import PoseVerifier
    vit=ROOT/'flywheel_outdoor_pose_20260903/source/ViTPose'
    manifest=json.loads((ROOT/'indoor_yolo_v3_pose_raw_allframes_20260905/run_manifest.json').read_text())
    from pathlib import Path
    pose=PoseVerifier(vit,vit/'configs/bee_pose/subset_13_20260827/scene_B_only_13.py',Path(manifest['pose']),device)
    from mmpose.datasets.pipelines import Compose
    loaders=[t for t in pose.pipeline.transforms if t.__class__.__name__=='LoadImageFromFile']
    assert len(loaders)==1 and loaders[0].channel_order=='rgb'
    def loaded(data):data['image_file']=None;return data
    pose.pipeline=Compose([loaded if t in loaders else t for t in pose.pipeline.transforms])
    stats=Counter();started=time.time()
    for n,path in enumerate(sorted((SOURCE/video).glob('*.json')),1):
        target=OUT/'pose_cache'/video/path.name
        if target.exists():continue
        ds=json.loads(path.read_text())['detections'];_,_,ios,_=geometry(ds)
        indexes=np.flatnonzero((ios>=CANDIDATE_IOS).any(1)).tolist()
        im=cv2.imread(str(IMAGES/video/(path.stem+'.jpg')))
        kp=pose(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),[ds[i]['bbox_xyxy']+[1.] for i in indexes],batch_size=256) if indexes else np.empty((0,2,3))
        save(target,dict(indexes=indexes,points=np.asarray(kp).tolist()))
        stats.update(frames=1,crops=len(indexes))
        if n%50==0:print(json.dumps(dict(stage='pose',video=video,**stats,seconds=round(time.time()-started,1))),flush=True)

def evidence(ds,cache):
    _,_,ios,diag=geometry(ds);b=np.array([d['bbox_xyxy'] for d in ds]);wh=b[:,2:]-b[:,:2];area=wh.prod(1)
    kp=np.zeros((len(ds),2,3));kp[cache['indexes']]=cache['points']
    margin=np.maximum(2.,.1*wh)
    inside=((kp[:,:,:2]>=(b[:,:2]-margin)[:,None,:])&(kp[:,:,:2]<=(b[:,2:]+margin)[:,None,:])).all(axis=(1,2))
    confident=(kp[:,:,2].min(1)>=.15)&(kp[:,:,2].mean(1)>=.25)&(np.linalg.norm(kp[:,0,:2]-kp[:,1,:2],axis=1)>=.15*diag)
    valid=confident&inside;result={}
    for i,j in zip(*np.where(np.triu(ios>=.35,1))):
        direct=np.linalg.norm(kp[i,:,:2]-kp[j,:,:2],axis=1).max()
        reverse=np.linalg.norm(kp[i,:,:2]-kp[j,::-1,:2],axis=1).max()
        distance=float(min(direct,reverse));norm=distance/min(diag[i],diag[j]);reason=None
        if valid[i] and valid[j] and norm<=.16:reason='same_body_keypoints'
        big,small=(i,j) if area[i]>=area[j] else (j,i)
        # 不把框外关键点直接当作正常关键点：只对窄小框检查是否反复借用了大框的整只蜜蜂。
        small_in_big=((kp[small,:,:2]>=(b[big,:2]-margin[big]))&(kp[small,:,:2]<=(b[big,2:]+margin[big]))).all()
        narrow=wh[small].max()/wh[small].min()>=3 and area[small]<=.65*area[big]
        if reason is None and valid[big] and confident[small] and not inside[small] and small_in_big and narrow and norm<=.20:
            reason='narrow_box_borrows_big_body'
        pair=tuple(sorted([int(ds[i]['track_id']),int(ds[j]['track_id'])]))
        result[pair]=dict(indexes=[int(i),int(j)],ios=float(ios[i,j]),pose_distance_ratio=norm,reason=reason,both_independently_valid=bool(valid[i] and valid[j]),fresh_keypoints=[kp[i].tolist(),kp[j].tolist()])
    return result,ios,area

def process(video):
    paths=sorted((SOURCE/video).glob('*.json'));raw={int(p.stem.split('_')[1]):json.loads(p.read_text()) for p in paths}
    frames={};positive=defaultdict(set);areas=defaultdict(list);observed=Counter()
    for path in paths:
        f=int(path.stem.split('_')[1]);ds=raw[f]['detections'];cache=json.loads((OUT/'pose_cache'/video/path.name).read_text())
        ev,_,area=evidence(ds,cache);frames[f]=ev
        for d,a in zip(ds,area):
            areas[d['track_id']].append(a)
            if d.get('origin')!='reassociation_interpolation':observed[d['track_id']]+=1
        for pair,e in ev.items():
            if e['reason']:positive[pair].add(f)
    # 固定轨迹优先级，不跟随逐帧置信度改变胜者。
    priority={tid:(-float(np.median(a)),-observed[tid],tid) for tid,a in areas.items()}
    stats=Counter();final={};events=[];fixed_pairs=Counter()
    for f,row in raw.items():
        ds=row['detections'];inter,_,ios,_=geometry(ds);b=np.array([d['bbox_xyxy'] for d in ds]);area=(b[:,2:]-b[:,:2]).prod(1)
        isolated=~(inter>0).any(1);removed={};protected=set();kept=[]
        # 包含关系按当前实际面积处理，不能由后续关键点阶段反过来删除胜出的大框。
        for i in sorted(range(len(ds)),key=lambda i:(-area[i],priority[ds[i]['track_id']])):
            winner=next((j for j in kept if ios[i,j]>=.8),None)
            if winner is None:kept.append(i)
            else:removed[i]=dict(kept_index=winner,reason='containment_080');protected.add(winner)
        active={}
        for pair,e in frames[f].items():
            support=sum(ff in positive[pair] for ff in range(f-2,f+3))
            # 必须至少3/5帧有一致证据；不把一次真实交叉传播成整段删除。
            if support<3:continue
            if not e['reason'] and e['both_independently_valid']:continue
            i,j=e['indexes']
            if i in removed or j in removed:continue
            fixed_loser=max([i,j],key=lambda k:priority[ds[k]['track_id']])
            if fixed_loser in protected:continue
            active[frozenset([i,j])]=dict(e,support_frames=support)
        final_kept=[]
        for i in sorted(kept,key=lambda i:(i not in protected,priority[ds[i]['track_id']])):
            winner=next((j for j in final_kept if frozenset([i,j]) in active),None)
            if winner is None or i in protected:final_kept.append(i);continue
            detail=active[frozenset([i,winner])]
            removed[i]=dict(kept_index=winner,reason=detail['reason'] or 'temporal_confirmation',evidence=detail)
            fixed_pairs[(ds[i]['track_id'],ds[winner]['track_id'])]+=1
        keep_set=set(final_kept);assert len(keep_set)+len(removed)==len(ds)
        assert all(i in keep_set for i in np.flatnonzero(isolated))
        rows=[]
        for i,e in removed.items():
            j=e['kept_index'];assert j in keep_set
            item=dict(**e,detection=ds[i],kept_track_id=ds[j]['track_id'],ios=float(ios[i,j]),removed_area=float(area[i]),kept_area=float(area[j]))
            rows.append(item);events.append(dict(frame=f,video=video,**item))
            stats[e['reason']]+=1;stats['removed_interpolation' if ds[i].get('origin')=='reassociation_interpolation' else 'removed_observed']+=1
        new=dict(row,detections=[d for i,d in enumerate(ds) if i in keep_set],final_dedup_removed=rows)
        new['final_deduplication']=dict(before=len(ds),after=len(keep_set),removed=len(removed),candidate_ios=.35,containment_ios=.8,pose_distance_ratio=.16,narrow_pose_distance_ratio=.20,temporal_support='3_of_5_frames',priority='protected_containment_winners_then_track_median_area_then_observed_support',id_retention_frames=150,interpolation_max_gap_frames=90,isolated_preserved=True)
        assert len({d['track_id'] for d in new['detections']})==len(new['detections'])
        final[f]=new;save(OUT/'final/frames'/video/f'frame_{f:08d}.json',new)
        stats.update(frames=1,before=len(ds),after=len(keep_set),removed=len(removed),isolated_preserved=int(isolated.sum()))
    reverse_pairs=[list(pair) for pair in fixed_pairs if (pair[1],pair[0]) in fixed_pairs];assert not reverse_pairs
    report=dict(stats=dict(stats),before=metrics(raw),after=metrics(final),pose_pair_winner_reversals=len(reverse_pairs),source=str(SOURCE))
    save(OUT/f'{video}_summary.json',report);save(OUT/f'{video}_events.json',events)
    print(json.dumps(dict(stage='dedup',video=video,stats=dict(stats),before={k:v for k,v in report['before'].items() if k not in ['cumulative_ids','change_examples']},after={k:v for k,v in report['after'].items() if k not in ['cumulative_ids','change_examples']})),flush=True)

def render(video):
    cv2.setNumThreads(1);writer=cv2.VideoWriter(str(OUT/f'{video}_global_compare_source.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(3840,1140));assert writer.isOpened()
    history=[defaultdict(lambda:deque(maxlen=30)),defaultdict(lambda:deque(maxlen=30))]
    for n,path in enumerate(sorted((OUT/'final/frames'/video).glob('*.json'))):
        before=json.loads((SOURCE/video/path.name).read_text());after=json.loads(path.read_text());f=int(path.stem.split('_')[1]);im=cv2.imread(str(IMAGES/video/(path.stem+'.jpg')));canvas=np.zeros((1140,3840,3),np.uint8)
        for side,row in enumerate([before,after]):
            panel=im.copy()
            for d in row['detections']:
                tid=d['track_id'];b=np.round(d['bbox_xyxy']).astype(int);pt=(int((b[0]+b[2])/2),int((b[1]+b[3])/2));h=history[side][tid]
                if h and f-h[-1][0]>1:h.clear()
                h.append((f,pt));col=color(tid)
                for a,z in zip(h,list(h)[1:]):cv2.line(panel,a[1],z[1],col,1,cv2.LINE_AA)
                cv2.rectangle(panel,tuple(b[:2]),tuple(b[2:]),col,2);cv2.putText(panel,str(tid),(max(0,b[0]),max(15,b[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,.48,col,1,cv2.LINE_AA)
            canvas[:1080,side*1920:(side+1)*1920]=panel
            title=f'{video} Frame {f} | '+('BEFORE final dedup' if side==0 else 'AFTER final dedup')+f' | Boxes: {len(row["detections"])}'
            if side:title+=f' | Removed: {len(after["final_dedup_removed"])}'
            cv2.putText(canvas,title,(side*1920+14,1108),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2)
            cv2.putText(canvas,'Same IDs and colors on both sides | 5s ID memory | Up to 3s filling | 1s trail',(side*1920+14,1137),cv2.FONT_HERSHEY_SIMPLEX,.6,(220,220,220),1)
        writer.write(canvas)
        if n==150:cv2.imwrite(str(OUT/f'{video}_global_preview.jpg'),canvas)
    writer.release()
    # 选择不同规则和不同ID对，展示实际删掉了哪些框；只改变显示，保留框本身不移动。
    events=json.loads((OUT/f'{video}_events.json').read_text());chosen=[];seen=set()
    for reason in ['narrow_box_borrows_big_body','same_body_keypoints','containment_080','temporal_confirmation']:
        for e in sorted(events,key=lambda e:-e.get('evidence',{}).get('support_frames',0)):
            pair=(e['detection']['track_id'],e['kept_track_id'])
            if e['reason']==reason and pair not in seen:chosen.append(e);seen.add(pair);break
    sheet=np.zeros((max(1,len(chosen))*260,1000,3),np.uint8)
    for n,e in enumerate(chosen):
        path=f'frame_{e["frame"]:08d}.json';row=json.loads((SOURCE/video/path).read_text());pair=[e['detection'],next(d for d in row['detections'] if d['track_id']==e['kept_track_id'])]
        b=np.array([d['bbox_xyxy'] for d in pair]);c=(b[:,:2].min(0)+b[:,2:].max(0))/2;x=int(np.clip(c[0]-125,0,1670));y=int(np.clip(c[1]-55,0,970));im=cv2.imread(str(IMAGES/video/path.replace('.json','.jpg')))
        for side,ds in enumerate([pair,pair[1:]]):
            crop=cv2.resize(im[y:y+110,x:x+250],(500,220))
            for d in ds:
                box=np.round((np.array(d['bbox_xyxy'])-[x,y,x,y])*2).astype(int);col=color(d['track_id']);cv2.rectangle(crop,tuple(box[:2]),tuple(box[2:]),col,2)
                cv2.putText(crop,str(d['track_id']),tuple(np.maximum(box[:2]+[0,-3],[0,15])),cv2.FONT_HERSHEY_SIMPLEX,.55,col,1)
            sheet[n*260:n*260+220,side*500:(side+1)*500]=crop
        cv2.putText(sheet,f'{video} Frame {e["frame"]}: {e["reason"]} | LEFT before / RIGHT after',(10,n*260+246),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    cv2.imwrite(str(OUT/f'{video}_removed_examples.jpg'),sheet)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--device',type=int,default=0);p.add_argument('--stage',choices=['all','infer','process','render'],default='all');a=p.parse_args();OUT.mkdir(exist_ok=True)
    if a.stage in ['all','infer']:infer(a.video,a.device)
    if a.stage in ['all','process']:process(a.video)
    if a.stage in ['all','render']:render(a.video)
