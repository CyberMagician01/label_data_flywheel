"""最终框重新一对一关联；旧ID只溯源。优先连续位置，休眠最多150帧。"""
import argparse,copy,json,time,colorsys
from pathlib import Path
from collections import Counter,defaultdict,deque
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
import dedup_overlap_tracks as geom

ROOT=Path('/root/autodl-tmp');SOURCE=ROOT/'indoor_thin_filter_20260906/thin/frames'
OUT=ROOT/'indoor_final_reassociation_20260906';IMAGES=geom.IMAGES

def describe(im,boxes):
    ds=[]
    for b in boxes:
        x,y,z,q=np.round(b).astype(int);x=max(0,min(im.shape[1]-1,x));y=max(0,min(im.shape[0]-1,y));z=max(x+1,min(im.shape[1],z));q=max(y+1,min(im.shape[0],q))
        p=cv2.resize(im[y:q,x:z],(8,8)).astype(float).ravel();p-=p.mean();p/=max(np.linalg.norm(p),1e-6);ds.append(p)
    return np.asarray(ds).reshape(-1,64)

def pair_geometry(a,b):
    aw=a[:,2:]-a[:,:2];bw=b[:,2:]-b[:,:2]
    dist=np.linalg.norm((a[:,:2]+a[:,2:])[:,None]/2-(b[:,:2]+b[:,2:])[None]/2,axis=2)
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    iou=inter/np.maximum(aw.prod(1)[:,None]+bw.prod(1)[None]-inter,1e-6)
    diag=np.minimum(np.linalg.norm(aw,axis=1)[:,None],np.linalg.norm(bw,axis=1)[None])
    ratio=np.maximum(aw[:,None]/np.maximum(bw[None],1),bw[None]/np.maximum(aw[:,None],1)).max(2)
    return dist,iou,diag,ratio

def metrics(data):
    seen=set();growth=[];changes=[];prev=None;tracks=defaultdict(list);counts=[]
    for f,row in sorted(data.items()):
        ds=row['detections'];b=np.array([d['bbox_xyxy'] for d in ds]).reshape(-1,4);ids=np.array([d['track_id'] for d in ds]);seen.update(ids.tolist());growth.append(len(seen));counts.append(len(ds))
        for tid in ids:tracks[int(tid)].append(f)
        if prev is not None and len(prev[0]) and len(b):
            pb,pi=prev;dist,iou,_,_=pair_geometry(pb,b);near=dist.argmin(1);back=dist.argmin(0)
            for i,j in enumerate(near):
                if back[j]==i and dist[i,j]<=5 and iou[i,j]>=.5 and pi[i]!=ids[j]:changes.append(dict(frame=f,old_id=int(pi[i]),new_id=int(ids[j]),distance=float(dist[i,j]),iou=float(iou[i,j])))
        prev=b,ids
    gaps=sum(sum(b-a>1 for a,b in zip(fs,fs[1:])) for fs in tracks.values())
    return dict(frames=len(data),box_instances=sum(counts),mean_boxes=float(np.mean(counts)),unique_ids=len(seen),first_frame_ids=counts[0],new_ids_after_first=len(seen)-counts[0],stationary_neighbor_changes=len(changes),change_examples=changes[:30],internal_gaps=gaps,cumulative_ids=growth,median_track_frames=float(np.median([len(x) for x in tracks.values()])))

def associate(video,interior_memory=False):
    started=time.time();cv2.setNumThreads(1)
    raw={int(p.stem.split('_')[1]):json.loads(p.read_text()) for p in sorted((SOURCE/video).glob('*.json'))}
    tracks=[];next_id=1;result={};stages=Counter()
    for f,row in raw.items():
        ds=row['detections'];boxes=np.array([d['bbox_xyxy'] for d in ds]);image=cv2.imread(str(IMAGES/video/f'frame_{f:08d}.jpg'),0);features=describe(image,boxes)
        def retained(t):
            center=(t['box'][:2]+t['box'][2:])/2
            inside=80<center[0]<1840 and 80<center[1]<1000
            return (inside or f-t['last']<=300) if interior_memory else f-t['last']<=150
        tracks=[t for t in tracks if retained(t)];remaining=set(range(len(ds)));assigned={};matched=set()
        for stage in ['locked','recent','lost']:
            ts=[t for t in tracks if t['id'] not in matched and (f-t['last']==1 if stage=='locked' else f-t['last']<=2 if stage=='recent' else f-t['last']>2)]
            js=sorted(remaining)
            if not ts or not js:continue
            pred=np.array([t['box'] for t in ts]);dist,iou,diag,ratio=pair_geometry(pred,boxes[js]);sim=np.asarray([t['desc'] for t in ts])@features[js].T
            if stage=='locked':
                # 双向最近、位置近且高度重叠时直接延续，不因模型置信度变化改号。
                near=dist.argmin(1);back=dist.argmin(0);pairs=[]
                for i,j in enumerate(near):
                    if back[j]!=i or dist[i,j]>6 or iou[i,j]<.5 or ratio[i,j]>2.5:continue
                    alternatives=np.delete(dist[i],j);margin=alternatives.min()-dist[i,j] if len(alternatives) else 100
                    if margin<2 and iou[i,j]<.75:continue
                    pairs.append((i,int(j)))
            else:
                if stage=='recent':valid=(dist<=np.clip(.65*diag,18,45))&(ratio<=4)&((iou>=.08)|(dist<=.35*diag))
                else:
                    age=np.array([f-t['last'] for t in ts])[:,None]
                    valid=(dist<=np.clip(.65*diag,18,45))&(ratio<=4)&(sim>=.10)&((iou>=.08)|(dist<=.35*diag))
                cost=dist/np.maximum(diag,1)+1.1*(1-iou)+.2*(1-sim)
                rr,cc=linear_sum_assignment(np.where(valid,cost,1e6));pairs=[(int(i),int(j)) for i,j in zip(rr,cc) if valid[i,j]]
            for i,jj in pairs:
                j=js[jj];t=ts[i];assigned[j]=t['id'];matched.add(t['id']);remaining.remove(j)
                desc=.7*features[j]+.3*t['desc'];desc/=max(np.linalg.norm(desc),1e-6)
                t.update(last=f,box=boxes[j].copy(),desc=desc);stages[stage]+=1
        for j in sorted(remaining):
            assigned[j]=next_id;tracks.append(dict(id=next_id,last=f,box=boxes[j].copy(),desc=features[j].copy()));next_id+=1;stages['new']+=1
        new=copy.deepcopy(row)
        for i,d in enumerate(new['detections']):
            d['pre_reassociation_track_id']=d['track_id'];d['track_id']=assigned[i]
            assert all(d[k]==v for k,v in ds[i].items() if k!='track_id')
        assert len(set(assigned.values()))==len(ds)
        new['reassociation']=dict(max_lost_frames=None if interior_memory else 150,interior_memory=interior_memory,edge_retention_frames=300 if interior_memory else 150,edge_band_px=80,old_ids_used_for_matching=False,stages=['mutual_nearest_locked','recent','lost'],source=str(SOURCE))
        result[f]=new;geom.save(OUT/'associated/frames'/video/f'frame_{f:08d}.json',new)
    report=dict(before=metrics(raw),associated=metrics(result),stages=dict(stages),seconds=time.time()-started)
    geom.save(OUT/f'{video}_association_summary.json',report);print(json.dumps({k:v for k,v in report.items() if k!='before' and k!='associated'}),flush=True)
    print(json.dumps({k:{a:b for a,b in v.items() if a not in ['cumulative_ids','change_examples']} for k,v in report.items() if k in ['before','associated']}),flush=True)

def stitch(video):
    data={int(p.stem.split('_')[1]):json.loads(p.read_text()) for p in sorted((OUT/'associated/frames'/video).glob('*.json'))};start=min(data);groups={}
    for f,row in data.items():
        im=cv2.imread(str(IMAGES/video/f'frame_{f:08d}.jpg'),0);ds=row['detections'];desc=describe(im,[d['bbox_xyxy'] for d in ds])
        for i,d in enumerate(ds):
            tid=d['track_id'];g=groups.setdefault(tid,dict(members={tid},rows=[],mask=0));g['rows'].append((f,i,np.array(d['bbox_xyxy']),desc[i]));g['mask']|=1<<(f-start)
    merges=[]
    for iteration in range(6):
        ids=list(groups);stats={}
        for tid,g in groups.items():
            b=np.median([r[2] for r in g['rows']],axis=0);desc=np.mean([r[3] for r in g['rows']],axis=0);desc/=max(np.linalg.norm(desc),1e-6);stats[tid]=b,(b[:2]+b[2:])/2,desc
        pairs=[]
        for i,j in cKDTree([stats[t][1] for t in ids]).query_pairs(25):
            a,b=ids[i],ids[j]
            if groups[a]['mask']&groups[b]['mask']:continue
            ba,ca,da=stats[a];bb,cb,db=stats[b];sim=float(da@db)
            if sim<.35:continue
            _,_,_,rat=pair_geometry(ba[None],bb[None])
            if rat[0,0]>3:continue
            pairs.append((float(np.linalg.norm(ca-cb))+10*(1-sim),a,b))
        used=set();merged_now=0
        for _,a,b in sorted(pairs):
            if a in used or b in used:continue
            events=sorted([(r[0],r[2],r[3],0) for r in groups[a]['rows']]+[(r[0],r[2],r[3],1) for r in groups[b]['rows']],key=lambda r:r[0]);safe=True;bridges=[]
            for left,right in zip(events,events[1:]):
                if left[3]==right[3]:continue
                dt=right[0]-left[0];dist=float(np.linalg.norm((right[1][:2]+right[1][2:]-left[1][:2]-left[1][2:])/2));sim=float(left[2]@right[2])
                if dt>150 or dist>min(30,6+1.5*dt) or sim<.15:safe=False;break
                bridges.append([dt,dist,sim])
            if not safe:continue
            keep,drop=min(a,b),max(a,b);groups[keep]['members']|=groups[drop]['members'];groups[keep]['rows']+=groups[drop]['rows'];groups[keep]['mask']|=groups[drop]['mask'];del groups[drop];used|={a,b};merged_now+=1;merges.append(dict(keep=keep,drop=drop,bridges=bridges))
        if not merged_now:break
    mapping={member:tid for tid,g in groups.items() for member in g['members']}
    for f,row in data.items():
        for d in row['detections']:d['pre_fragment_merge_id']=d['track_id'];d['track_id']=mapping[d['track_id']]
        assert len({d['track_id'] for d in row['detections']})==len(row['detections'])
        geom.save(OUT/'associated/frames'/video/f'frame_{f:08d}.json',row)
    report=json.loads((OUT/f'{video}_association_summary.json').read_text());report['before_stitch']=report['associated'];report['associated']=metrics(data);report['merged_fragments']=len(merges);geom.save(OUT/f'{video}_association_summary.json',report);geom.save(OUT/f'{video}_merges.json',merges)
    print(json.dumps(dict(merged=len(merges),metrics={k:v for k,v in report['associated'].items() if k not in ['cumulative_ids','change_examples']})),flush=True)

def fill(video,max_gap_frames=150,observed_only=False):
    data={int(p.stem.split('_')[1]):json.loads(p.read_text()) for p in sorted((OUT/'associated/frames'/video).glob('*.json'))}
    tracks=defaultdict(dict)
    for f,row in data.items():
        for d in row['detections']:tracks[d['track_id']][f]=d
    stats=Counter();candidates=defaultdict(list)
    # 先按新轨迹轻度平滑中心，宽高不变，不跨缺口。
    for tid,rows in tracks.items():
        fs=np.array(sorted(rows));groups=np.split(fs,np.flatnonzero(np.diff(fs)>1)+1)
        for group in groups:
            if len(group)<3:continue
            b=np.array([rows[int(f)]['bbox_xyxy'] for f in group]);wh=b[:,2:]-b[:,:2];centers=(b[:,:2]+b[:,2:])/2
            smooth=centers+np.clip(.6*(gaussian_filter1d(centers,1,axis=0,mode='nearest',truncate=2)-centers),-2,2)
            smooth=np.maximum(wh/2,np.minimum(np.array([1920,1080])-wh/2,smooth));final=np.concatenate([smooth-wh/2,smooth+wh/2],axis=1)
            for f,box in zip(group,final):
                d=rows[int(f)];d['pre_reassociation_smoothing_bbox']=d['bbox_xyxy'];d['bbox_xyxy']=box.tolist();d['reassociation_smoothing']='center_only_gaussian5_max2px';stats['smoothed']+=1
        if observed_only:continue
        fs=sorted(rows)
        for a,b in zip(fs,fs[1:]):
            gap=b-a-1
            if not gap:continue
            aa=np.array(rows[a]['bbox_xyxy']);bb=np.array(rows[b]['bbox_xyxy']);dist,iou,diag,ratio=pair_geometry(aa[None],bb[None]);distance=float(dist[0,0]);size_ratio=float(ratio[0,0])
            safe=gap<=min(30,max_gap_frames) and distance<=min(40,8+2*(gap+1)) and size_ratio<=2
            if 30<gap<=max_gap_frames:safe=distance<=min(16,.35*diag[0,0]) and size_ratio<=1.5
            if not safe:stats['unfilled_gaps']+=1;continue
            for f in range(a+1,b):
                alpha=(f-a)/(b-a);box=aa*(1-alpha)+bb*alpha;wh=box[2:]-box[:2]
                if max(wh)/min(wh)>=6:stats['thin_rejected']+=1;continue
                candidates[f].append(dict(track_id=tid,bbox_xyxy=box.tolist(),origin='reassociation_interpolation',det_confidence=None,keypoints={},class_id=0,verified=False,interpolation={'anchor_frames':[a,b],'missing_frames':gap,'method':'linear_xyxy'}))
    # 只向空缺位置补框，已有框不由插值框取代。
    for f,row in data.items():
        cs=sorted(candidates[f],key=lambda d:-(d['bbox_xyxy'][2]-d['bbox_xyxy'][0])*(d['bbox_xyxy'][3]-d['bbox_xyxy'][1]));existing=row['detections'];accepted=[];rejected=[]
        for d in cs:
            bb=np.array([e['bbox_xyxy'] for e in existing+accepted]);box=np.array(d['bbox_xyxy'])
            inter=np.maximum(0,np.minimum(bb[:,2:],box[2:])-np.maximum(bb[:,:2],box[:2])).prod(1)
            ios=inter/np.maximum(np.minimum((bb[:,2:]-bb[:,:2]).prod(1),(box[2:]-box[:2]).prod()),1e-6)
            if np.max(ios,initial=0)>=.8:rejected.append(d);stats['overlap_rejected']+=1
            else:accepted.append(d)
        row['reassociation_candidates']=accepted;row['reassociation_overlap_rejected']=rejected
        row['interpolation_max_gap_frames']=max_gap_frames
        stats['proposed']+=len(cs);stats['geometric_accepted']+=len(accepted)
        geom.save(OUT/'pending_pose/frames'/video/f'frame_{f:08d}.json',row)
    geom.save(OUT/f'{video}_fill_summary.json',dict(stats));print(json.dumps(stats),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--stage',choices=['associate','stitch','fill'],required=True);a=p.parse_args();OUT.mkdir(exist_ok=True)
    if a.stage=='associate':associate(a.video)
    elif a.stage=='stitch':stitch(a.video)
    else:fill(a.video)
