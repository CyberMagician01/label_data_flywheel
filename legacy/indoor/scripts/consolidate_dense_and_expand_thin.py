"""用重叠轨迹的身体证据合并重复ID，独立窄框扩展后再生成连续插值。"""
import argparse,json
from collections import Counter,defaultdict
import cv2,numpy as np
import dedup_final_temporal_pose as pose_runner
import keep_interpolation_and_ownership as ownership
from dedup_overlap_tracks import ROOT,IMAGES,geometry,save,color
from reassociate_final_bees import metrics

SOURCE=ROOT/'indoor_keep_interpolation_20260906/final/frames'
OUT=ROOT/'indoor_dense_consolidation_thin_expansion_20260906'

def separate(a,b,positions):
    count=0
    for f in positions[a].keys()&positions[b].keys():
        aa=positions[a][f];bb=positions[b][f];inter=np.maximum(0,np.minimum(aa[2:],bb[2:])-np.maximum(aa[:2],bb[:2])).prod();ios=inter/max(min((aa[2:]-aa[:2]).prod(),(bb[2:]-bb[:2]).prod()),1e-6)
        distance=np.linalg.norm((aa[:2]+aa[2:]-bb[:2]-bb[2:])/2);diag=min(np.linalg.norm(aa[2:]-aa[:2]),np.linalg.norm(bb[2:]-bb[:2]))
        if ios<.2 and distance>.7*diag:count+=1
        if count>=3:return True
    return False

def overlap(box,others):
    if not len(others):return np.zeros(0),np.zeros(0)
    inter=np.maximum(0,np.minimum(others[:,2:],box[2:])-np.maximum(others[:,:2],box[:2])).prod(1);a=np.prod(box[2:]-box[:2]);b=(others[:,2:]-others[:,:2]).prod(1)
    return inter/np.maximum(a+b-inter,1e-6),inter/np.maximum(np.minimum(a,b),1e-6)

def expansion(box,others,points):
    wh=box[2:]-box[:2];axis=int(np.argmin(wh));short=wh[axis];long=wh[1-axis]
    if long/short<3:return box,None
    iou,ios=overlap(box,others)
    if np.max(iou,initial=0)>=.25 or np.max(ios,initial=0)>=.7 or np.count_nonzero(ios>=.2)>1:return box,None
    extra=max(2.,min(8.,.4*short,max(.25*short,long/3-short)))
    share=.5
    if points is not None and np.min(points[:,2])>=.15 and np.mean(points[:,2])>=.25:
        location=(np.mean(points[:,axis])-box[axis])/short
        if .6<location<=1.1:share=.75
        elif -.1<=location<.4:share=.25
    for factor in [1.,.5]:
        proposed=box.copy();proposed[axis]-=extra*(1-share)*factor;proposed[axis+2]+=extra*share*factor;proposed[:2]=np.maximum(proposed[:2],0);proposed[2:]=np.minimum(proposed[2:],[1920,1080])
        niou,nios=overlap(proposed,others)
        if np.max(niou,initial=0)>.32 or np.any((ios<.8)&(nios>=.8)) or np.any(nios-ios>.18):continue
        return proposed,dict(short_axis=axis,total_added_px=float((proposed[axis+2]-proposed[axis])-short),positive_side_fraction=share)
    return box,None

def process(video):
    paths=sorted((SOURCE/video).glob('*.json'));raw={};qualities={};points={};pair_info=defaultdict(lambda:dict(positive=set(),interpolated_positive=0,distinct_observed=0,seen=0));positions=defaultdict(dict)
    for p in paths:
        f=int(p.stem.split('_')[1]);row=json.loads(p.read_text());raw[f]=row;ds=row['detections'];cache=json.loads((OUT/'pose_cache'/video/p.name).read_text());ev,valid,cov,area=ownership.body_evidence(ds,cache);qualities[f]=(valid,cov,area)
        kp=np.zeros((len(ds),2,3))
        if cache['indexes']:kp[cache['indexes']]=cache['points']
        points[f]=kp
        for d in ds:
            if d.get('origin')!='reassociation_interpolation':positions[d['track_id']][f]=np.array(d['bbox_xyxy'])
        for pair,e in ev.items():
            i,j=e['indexes'];st=pair_info[pair];st['seen']+=1;both_observed=all(ds[k].get('origin')!='reassociation_interpolation' for k in [i,j])
            if e['distinct'] and both_observed:st['distinct_observed']+=1
            supported=e['reason'] in ['same_body','partial_box_borrows_body','empty_box_borrows_one_body_point'] or (e['reason']=='containment_080' and valid[i] and valid[j] and e['distance_ratio']<=.2)
            if supported:st['positive'].add(f);st['interpolated_positive']+=int(not both_observed)
    parent={tid:tid for tid in positions};members={tid:{tid} for tid in parent};merges=[];vetoes=[]
    def find(tid):
        while parent[tid]!=tid:parent[tid]=parent[parent[tid]];tid=parent[tid]
        return tid
    forbidden={pair for pair,st in pair_info.items() if st['distinct_observed']>=3}
    for pair,st in sorted(pair_info.items(),key=lambda kv:-len(kv[1]['positive'])):
        fs=st['positive']
        if not any(sum(ff in fs for ff in range(f-2,f+3))>=3 for f in fs):continue
        a,b=map(find,pair)
        if a==b:continue
        if any(tuple(sorted([i,j])) in forbidden or separate(i,j,positions) for i in members[a] for j in members[b]):vetoes.append(dict(pair=pair,reason='independent_observed_tracks'));continue
        winner=min([a,b],key=lambda tid:(min(min(positions[t]) for t in members[tid]),tid));loser=b if winner==a else a
        parent[loser]=winner;members[winner]|=members.pop(loser);merges.append(dict(pair=pair,winner=winner,loser=loser,evidence_frames=len(fs),evidence_involving_interpolation=st['interpolated_positive']))
    mapping={tid:find(tid) for tid in parent};output={};events=[];stats=Counter()
    for f,row in raw.items():
        ds=row['detections'];valid,cov,area=qualities[f];all_groups=defaultdict(list);observed_groups=defaultdict(list)
        for i,d in enumerate(ds):
            tid=mapping[d['track_id']];all_groups[tid].append(i)
            if d.get('origin')!='reassociation_interpolation':observed_groups[tid].append(i)
        # 包括预测轨迹占据的位置，用于判断窄框周围是否拥挤。
        neighborhood={tid:np.array(ds[max(ii,key=lambda i:(ds[i].get('origin')!='reassociation_interpolation',valid[i],area[i]))]['bbox_xyxy']) for tid,ii in all_groups.items()}
        retained=[]
        for tid,indexes in observed_groups.items():
            k=max(indexes,key=lambda i:(valid[i] and cov[i]>=.8,area[i],ds[i]['track_id']==tid));d=dict(ds[k]);d['pre_dense_merge_id']=d['track_id'];d['track_id']=tid
            box=np.array(d['bbox_xyxy']);others=np.array([b for t,b in neighborhood.items() if t!=tid]).reshape(-1,4);expanded,detail=expansion(box,others,points[f][k])
            if detail:
                d['pre_independent_expansion_bbox']=d['bbox_xyxy'];d['bbox_xyxy']=expanded.tolist();d['independent_thin_expansion']=detail;stats['expanded_observed']+=1;events.append(dict(frame=f,track_id=tid,original_id=ds[k]['track_id'],before=box.tolist(),after=expanded.tolist(),**detail))
            retained.append(d);stats['duplicate_observed_removed']+=len(indexes)-1
        output[f]=dict(row,detections=retained);stats.update(frames=1,observed_after=len(retained))
    tracks=defaultdict(dict)
    for f,row in output.items():
        for d in row['detections']:tracks[d['track_id']][f]=d
    expected=[]
    for tid,by_frame in tracks.items():
        fs=sorted(by_frame)
        for a,b in zip(fs,fs[1:]):
            gap=b-a-1
            if not 0<gap<=90:continue
            expected.append(dict(track_id=tid,anchors=[a,b],missing_frames=gap))
            for f in range(a+1,b):
                alpha=(f-a)/(b-a);box=np.array(by_frame[a]['bbox_xyxy'])*(1-alpha)+np.array(by_frame[b]['bbox_xyxy'])*alpha
                output[f]['detections'].append(dict(track_id=tid,bbox_xyxy=box.tolist(),origin='reassociation_interpolation',det_confidence=None,class_id=0,keypoints={},verified=False,interpolation_protected=True,interpolation=dict(anchor_frames=[a,b],missing_frames=gap,method='linear_xyxy')));stats['interpolated']+=1
    for f,row in output.items():
        assert len({d['track_id'] for d in row['detections']})==len(row['detections'])
        row['continuous_interpolation']=dict(id_retention_frames=150,max_missing_frames=90,post_fill_filtering=False,same_id_gaps_filled=True,source_anchors='identity_consolidated_expanded_observed_boxes')
        row['dense_track_consolidation']=dict(merged_pairs=len(merges),post_fill_deletion=False);save(OUT/'final/frames'/video/f'frame_{f:08d}.json',row)
    report=dict(stats=dict(stats),merged_identity_pairs=len(merges),vetoed_pairs=len(vetoes),before=metrics(raw),after=metrics(output),filtered_interpolations=0)
    save(OUT/f'{video}_summary.json',report);save(OUT/f'{video}_merges.json',dict(accepted=merges,vetoes=vetoes));save(OUT/f'{video}_aliases.json',mapping);save(OUT/f'{video}_expected_gaps.json',expected);save(OUT/f'{video}_expansion_events.json',events)
    print(json.dumps(dict(video=video,stats=dict(stats),merges=len(merges),vetoes=len(vetoes),after={k:v for k,v in report['after'].items() if k not in ['cumulative_ids','change_examples']})),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--device',type=int,default=0);p.add_argument('--stage',choices=['all','process','render'],default='all');a=p.parse_args();OUT.mkdir(exist_ok=True)
    if a.stage=='all':pose_runner.SOURCE=SOURCE;pose_runner.OUT=OUT;pose_runner.CANDIDATE_IOS=.2;pose_runner.infer(a.video,a.device)
    if a.stage in ['all','process']:process(a.video)
    if a.stage in ['all','render']:ownership.REFERENCE=SOURCE;ownership.OUT=OUT;ownership.render(a.video)
