"""仅依据局部实测与身体证据修正冲突身份，再从修正后的端点完整补帧。"""
import argparse, copy, json
from collections import Counter, defaultdict
import cv2
import numpy as np
import dedup_final_temporal_pose as pose_runner
import keep_interpolation_and_ownership as ownership
from dedup_overlap_tracks import ROOT, IMAGES, geometry, save
from reassociate_final_bees import metrics, pair_geometry, describe

SOURCE=ROOT/'indoor_dense_consolidation_thin_expansion_20260906/final/frames'
OUT=ROOT/'indoor_local_identity_repair_20260906'

def infer(video, device):
    pose_runner.SOURCE=SOURCE;pose_runner.OUT=OUT;pose_runner.CANDIDATE_IOS=.2
    pose_runner.infer(video,device)

def process(video):
    cv2.setNumThreads(1)
    raw={};observed=defaultdict(dict);kp={};features={};conflicts=defaultdict(list)
    for p in sorted((SOURCE/video).glob('*.json')):
        f=int(p.stem.split('_')[1]);row=json.loads(p.read_text());raw[f]=row;ds=row['detections']
        cache=json.loads((OUT/'pose_cache'/video/p.name).read_text());ev,valid,coverage,area=ownership.body_evidence(ds,cache)
        points=np.zeros((len(ds),2,3))
        if cache['indexes']:points[cache['indexes']]=cache['points']
        obs=[i for i,d in enumerate(ds) if d.get('origin')!='reassociation_interpolation']
        im=cv2.imread(str(IMAGES/video/(p.stem+'.jpg')),0);desc=describe(im,[ds[i]['bbox_xyxy'] for i in obs])
        for z,i in enumerate(obs):
            d=ds[i];tid=d['track_id'];observed[tid][f]=copy.deepcopy(d);features[tid,f]=desc[z]
            if valid[i] and coverage[i]>=.8:kp[tid,f]=points[i,:,:2]
        for pair,e in ev.items():
            i,j=e['indexes']
            has_fill=any(ds[k].get('origin')=='reassociation_interpolation' for k in [i,j])
            strong=e['reason'] in ['same_body','partial_box_borrows_body','empty_box_borrows_one_body_point'] or (e['reason']=='containment_080' and valid[i] and valid[j] and e['distance_ratio']<=.2)
            if has_fill and strong and not e['distinct']:conflicts[pair].append(f)
    frames={tid:np.array(sorted(rows)) for tid,rows in observed.items()}
    def boundary(tid,cut):
        fs=frames[tid];left=fs[fs<cut];right=fs[fs>=cut]
        return (int(left[-1]),int(right[0])) if len(left) and len(right) else None
    def box(tid,f):return np.array(observed[tid][f]['bbox_xyxy'])
    def link(a,fa,b,fb):
        aa,bb=box(a,fa),box(b,fb);dt=fb-fa
        dist,iou,diag,ratio=pair_geometry(aa[None],bb[None]);distance=float(dist[0,0]);scale=float(diag[0,0]);rat=float(ratio[0,0]);sim=float(features[a,fa]@features[b,fb])
        norm=distance/max(scale,1);pose_distance=None
        if (a,fa) in kp and (b,fb) in kp:
            pa,pb=kp[a,fa],kp[b,fb];pose_distance=float(min(np.linalg.norm(pa-pb,axis=1).mean(),np.linalg.norm(pa-pb[::-1],axis=1).mean())/max(scale,1))
        cost=norm+.2*np.log(max(rat,1))+.15*(1-sim)+(.7*pose_distance if pose_distance is not None else .12)
        valid=0<dt<=150 and distance<=min(55,max(20,.65*scale)) and rat<=4 and (sim>=.15 or pose_distance is not None and pose_distance<.3)
        return float(cost),bool(valid),dict(frames=[fa,fb],distance_px=distance,shape_ratio=rat,appearance_similarity=sim,body_distance_ratio=pose_distance)
    def path_overlap(a,b,ends,cross):
        la,ra,lb,rb=ends;lo=max(la,lb)+1;hi=min(ra,rb)
        if hi<=lo:return 0.
        total=[]
        for f in range(lo,hi):
            ar,br=(b,a) if cross else (a,b);af,bf=(rb,ra) if cross else (ra,rb)
            t=(f-la)/(af-la);x=box(a,la)*(1-t)+box(ar,af)*t
            t=(f-lb)/(bf-lb);y=box(b,lb)*(1-t)+box(br,bf)*t
            inter=np.maximum(0,np.minimum(x[2:],y[2:])-np.maximum(x[:2],y[:2])).prod()
            ios=float(inter/max(min((x[2:]-x[:2]).prod(),(y[2:]-y[:2]).prod()),1e-6));total.append(ios*ios)
        return float(np.mean(total))
    proposals=[];rejected=Counter();seen=set();diagnostics=[]
    for (a,b),fs in conflicts.items():
        support=set(fs)
        for cut in fs:
            if sum(f in support for f in range(cut-2,cut+3))<3:continue
            ba,bb=boundary(a,cut),boundary(b,cut)
            if ba is None or bb is None:continue
            ends=(*ba,*bb);key=(a,b,*ends)
            if key in seen:continue
            seen.add(key);la,ra,lb,rb=ends
            old=[link(a,la,a,ra),link(b,lb,b,rb)];new=[link(a,la,b,rb),link(b,lb,a,ra)]
            old_overlap=path_overlap(a,b,ends,False);new_overlap=path_overlap(a,b,ends,True)
            old_cost=sum(x[0] for x in old)+.6*old_overlap;new_cost=sum(x[0] for x in new)+.6*new_overlap
            reason=None
            if max(ra-la-1,rb-lb-1)<=90 and max(rb-la-1,ra-lb-1)>90:reason='would_turn_fillable_gap_into_unfilled_gap'
            elif not all(x[1] for x in new):reason='alternative_motion_or_appearance_not_supported'
            elif new_cost>old_cost*.8 or old_cost-new_cost<.15:reason='no_clear_endpoint_improvement'
            elif new_overlap>old_overlap*.85:reason='bridge_conflict_not_reduced'
            item=dict(pair=[a,b],cut=cut,anchors=list(ends),before_cost=old_cost,after_cost=new_cost,before_bridge_overlap=old_overlap,after_bridge_overlap=new_overlap,before_links=[x[2] for x in old],after_links=[x[2] for x in new],reason=reason)
            if reason:rejected[reason]+=1
            else:proposals.append(item)
            diagnostics.append(item)
    # 每条受影响身份本轮只采用一次明确改善的交换，避免多个局部提案互相覆盖。
    used=set();accepted=[]
    for item in sorted(proposals,key=lambda x:x['after_cost']-x['before_cost']):
        a,b=item['pair']
        if a in used or b in used:rejected['overlapping_repair_proposal']+=1;continue
        accepted.append(item);used.update([a,b])
    repaired=defaultdict(dict)
    exchanges={tid:(b if tid==a else a,item['cut']) for item in accepted for a,b in [item['pair']] for tid in [a,b]}
    for tid,rows in observed.items():
        for f,d in rows.items():
            assigned=tid
            exchange=exchanges.get(tid)
            if exchange is not None and f>=exchange[1]:assigned=exchange[0]
            d['pre_local_repair_track_id']=tid;d['track_id']=assigned
            if assigned!=tid:d['local_identity_repair']='endpoint_supported_tail_exchange'
            assert f not in repaired[assigned];repaired[assigned][f]=d
    output={f:dict(row,detections=[]) for f,row in raw.items()};stats=Counter();expected=[]
    for tid,rows in repaired.items():
        for f,d in rows.items():output[f]['detections'].append(d);stats['observed']+=1;stats['relabeled_observed']+=int(d['pre_local_repair_track_id']!=tid)
        fs=sorted(rows)
        for a,b in zip(fs,fs[1:]):
            if not 0<b-a-1<=90:continue
            expected.append(dict(track_id=tid,anchors=[a,b],missing_frames=b-a-1))
            for f in range(a+1,b):
                t=(f-a)/(b-a);xy=np.array(rows[a]['bbox_xyxy'])*(1-t)+np.array(rows[b]['bbox_xyxy'])*t
                output[f]['detections'].append(dict(track_id=tid,bbox_xyxy=xy.tolist(),origin='reassociation_interpolation',det_confidence=None,class_id=0,keypoints={},verified=False,interpolation_protected=True,interpolation=dict(anchor_frames=[a,b],missing_frames=b-a-1,method='linear_xyxy')));stats['interpolated']+=1
    overlaps=Counter()
    for f,row in output.items():
        assert len({d['track_id'] for d in row['detections']})==len(row['detections'])
        for name,r in [('before',raw[f]),('after',row)]:
            _,_,ios,_=geometry(r['detections']);overlaps[name]+=int(np.count_nonzero(np.triu(ios>=.5,1)))
        row['continuous_interpolation']=dict(id_retention_frames=150,max_missing_frames=90,post_fill_filtering=False,same_id_gaps_filled=True,source_anchors='locally_reassociated_observed_boxes')
        save(OUT/'final/frames'/video/f'frame_{f:08d}.json',row)
    report=dict(accepted_local_exchanges=len(accepted),candidate_pairs=len(conflicts),candidate_boundaries=len(seen),rejections=dict(rejected),stats=dict(stats),overlap_pairs_ios050=dict(overlaps),before=metrics(raw),after=metrics(output),post_fill_removed=0)
    save(OUT/f'{video}_summary.json',report);save(OUT/f'{video}_repairs.json',accepted);save(OUT/f'{video}_diagnostics.json',diagnostics);save(OUT/f'{video}_expected_gaps.json',expected)
    print(json.dumps({k:v for k,v in report.items() if k not in ['before','after']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--device',type=int,default=0);p.add_argument('--stage',choices=['infer','process','render'],required=True);a=p.parse_args();OUT.mkdir(exist_ok=True)
    if a.stage=='infer':infer(a.video,a.device)
    elif a.stage=='process':process(a.video)
    else:ownership.REFERENCE=SOURCE;ownership.OUT=OUT;ownership.render(a.video,after_title='AFTER: local ID repair + interpolation')
