import os,sys,json,math
from pathlib import Path
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']=''
root=Path('/root/bee_tracking_pilot_20260907')
sys.path.insert(0,str(root/'repo'))
from tracking import track
video,mode=sys.argv[1:3]
base=root/'gt_eval_01_03_20260907'/video
out=base/'tracking'
out.mkdir(parents=True,exist_ok=True)
meta=json.load(open(base/'inputs.json'));records=meta['records']
x0,y0=meta['roi'][:2]
poses={}
pose_centers={}
for fr,r in enumerate(records):
    for d in r['detections']:
        x,y=d['center'];kp=d.get('keypoints',{});h=kp.get('head');t=kp.get('abdomen_tip')
        if not h or not t:continue
        delta=np.array(h[:2])-np.array(t[:2]);length=float(np.linalg.norm(delta));q=float(min(h[2],t[2]))
        key=(fr,x,y);pose_centers[key]=(np.array(h[:2])+np.array(t[:2]))/2
        poses[key]=(float(np.arctan2(delta[1],delta[0])),length,q if length>=5 else 0.)
def pose(fr,xy):
    return poses.get((int(fr),int(xy[0]),int(xy[1])),(0.,1.,0.))
def wrap(a):return np.arctan2(np.sin(a),np.cos(a))
original_dist=track.dist_m
def enhanced_dist(t,active,last_seen,lens,fr_i,preds):
    # Retain the original candidate gates and appearance distance.
    d=original_dist(t,active,last_seen,lens,fr_i,preds)
    if not len(d):return d
    now=int(fr_i[0])
    for slot in np.unique(d[:,0]).astype(int):
        ix=np.where(d[:,0]==slot)[0]; detix=d[ix,1].astype(int)
        hist=np.unique(last_seen[slot]);hist=hist[hist>=0][-7:]
        hist=hist[t[hist,slot,0]!=0]
        last=int(hist[-1]);gap=now-last;lastxy=t[last,slot,:2].astype(float)
        shift=np.zeros(2);trust=0.
        if len(hist)>=3:
            velocities=np.diff(t[hist,slot,:2],axis=0)/np.diff(hist)[:,None]
            # Recent smoothed velocity; downweight jitter with inconsistent directions.
            weights=np.exp(np.linspace(-1.5,0,len(velocities)))
            velocity=np.average(velocities,axis=0,weights=weights)
            speed=float(np.linalg.norm(velocity))
            consistency=speed/(float(np.average(np.linalg.norm(velocities,axis=1),weights=weights))+1e-6)
            velocity*=min(1.,15./max(speed,1e-6))
            shift=velocity*sum(.85**k for k in range(gap))
            trust=.75*np.clip(consistency,0,1)*(.97**max(0,gap-1))
        residual=np.linalg.norm(preds[detix,:2]-(lastxy+shift),axis=1)/max(1,np.floor(np.sqrt(gap)))
        d[ix,2]=(1-trust)*d[ix,2]+trust*residual
        if mode=='momentum_pose':
            angle,length,quality=pose(last,lastxy)
            past=[(int(f),pose(f,t[f,slot,:2])) for f in hist]
            rates=[]
            for (f1,p1),(f2,p2) in zip(past[:-1],past[1:]):
                if min(p1[2],p2[2])>=.3 and f2-f1<=3:
                    rates.append(float(wrap(p2[0]-p1[0])/(f2-f1)))
            turn=float(np.clip(np.median(rates[-4:]),-.2,.2)) if rates else 0.
            expected=angle+turn*sum(.8**k for k in range(gap))
            for row,j in zip(ix,detix):
                a2,l2,q2=pose(now,preds[j,:2])
                reliability=np.clip((min(quality,q2)-.15)/.45,0,1)*np.exp(-max(0,gap-1)/15)
                # Soft, bounded penalty: a turn alone never rejects a detection.
                angular=.5*(1-np.cos(wrap(a2-expected)))/2
                lengthpen=.12*min(abs(np.log(max(l2,1)/max(length,1))),1)
                d[row,4]+=reliability*(angular+lengthpen)
    return d

rejection_stats={'rejected_candidates':0,'considered_candidates':0}
def gated_dist(t,active,last_seen,lens,fr_i,preds):
    d=enhanced_dist(t,active,last_seen,lens,fr_i,preds)
    if not len(d):return d
    now=int(fr_i[0]);keep=np.ones(len(d),bool)
    for slot in np.unique(d[:,0]).astype(int):
        ix=np.where(d[:,0]==slot)[0]
        hist=np.unique(last_seen[slot]);hist=hist[hist>=0][-7:]
        hist=hist[t[hist,slot,0]!=0];last=int(hist[-1]);gap=now-last
        xy=t[hist,slot,:2].astype(float);velocity=np.zeros(2);scatter=0.
        if len(hist)>=3:
            delta=np.diff(hist)
            vel=np.diff(xy,axis=0)/delta[:,None]
            vel=vel[delta<=3]
            if len(vel)>=2:
                velocity=np.median(vel,axis=0)
                scatter=float(np.median(np.linalg.norm(vel-velocity,axis=1)))
        speed=float(np.linalg.norm(velocity))
        velocity*=min(1.,12./max(speed,1e-6))
        shift=velocity*sum(.90**k for k in range(gap))
        # Uncertainty grows slowly. Long gaps no longer imply a 120-160px search radius.
        radius=min(55.,16.+2.*np.sqrt(gap)+min(scatter,2.5)*np.sqrt(gap))
        for row in ix:
            j=int(d[row,1]);lastxy=xy[-1];current=preds[j,:2].astype(float)
            previous_point=lastxy;current_point=current
            residual=float(np.linalg.norm(current_point-(previous_point+shift)))
            if residual>radius:
                keep[row]=False
    rejection_stats['considered_candidates']+=len(d)
    rejection_stats['rejected_candidates']+=int((~keep).sum())
    return d[keep]

track.FR2=len(records);track.LOOK_BACK_BASE=150;track.LEN_MIN=30
track.FTS_DIR=str(base/'embeddings');track.TRACK_DIR=str(out/'tracks')
track.dist_m=gated_dist
original_get_matches=track.get_matches
track.get_matches=lambda d: original_get_matches(d) if len(d) else np.empty((0,2),dtype=int)
raw=[]
(out/'raw_tracks').mkdir(exist_ok=True)
original_save=track.save_trajectory
def audit(t,i,num,saved,discarded):
    frames=np.where(t[:,i,0]!=0)[0]
    np.savetxt(out/'raw_tracks'/f'{len(raw):06d}.txt',np.column_stack([frames,t[frames,i,:4]]),fmt='%d',delimiter=',')
    raw.append({'start':int(frames[0]),'length':len(frames)})
    return original_save(t,i,num,saved,discarded)
track.save_trajectory=audit
track.get_look_back=lambda x,y,is_butt: (75 if x<192 or x>1728 or y<108 or y>972 else 150)*(10 if is_butt else 1)
track.build_trajectories();track.sort_trajectories()
saved=[r for r in raw if r['length']>=30]
result={'video':video,'mode':mode,'frames':len(records),'fps':30,'raw_new_ids':sum(r['start']>0 for r in raw),'retained_new_ids':sum(r['start']>0 for r in saved),'raw_ids':len(raw),'retained_ids':len(saved),'retained_records':sum(r['length'] for r in saved),'input_records':sum(r['length'] for r in raw),'pose_valid_records':sum(p[2]>=.3 for p in poses.values()),'pose_total_records':len(poses)}
result['candidate_gate_stats']=rejection_stats
result['coverage']=result['retained_records']/result['input_records']
json.dump(result,open(out/'result.json','w'),indent=2)
print('RESULT',json.dumps(result),flush=True)
