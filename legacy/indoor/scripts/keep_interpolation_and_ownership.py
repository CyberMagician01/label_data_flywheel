"""先按身体完整性合并重复实测轨迹，再无条件保留同ID的三秒内线性补框。"""
import argparse,json
from collections import Counter,defaultdict,deque
import cv2,numpy as np
import dedup_final_temporal_pose as base
from dedup_overlap_tracks import ROOT,IMAGES,geometry,save,color
from reassociate_final_bees import metrics

RAW=ROOT/'indoor_interior_memory_20260906/offline_birth4_edge150_inside150_fill90/pending_pose/frames'
REFERENCE=ROOT/'indoor_final_temporal_dedup_20260906/final/frames'
OUT=ROOT/'indoor_keep_interpolation_20260906'

def prepare(video):
    for p in sorted((RAW/video).glob('*.json')):
        row=json.loads(p.read_text());assert all(d.get('origin')!='reassociation_interpolation' for d in row['detections'])
        save(OUT/'observed/frames'/video/p.name,row)

def body_evidence(ds,cache):
    if not ds:return {},np.zeros(0,dtype=bool),np.zeros(0),np.zeros(0)
    inter,_,ios,diag=geometry(ds);b=np.array([d['bbox_xyxy'] for d in ds]);wh=b[:,2:]-b[:,:2];area=wh.prod(1)
    kp=np.zeros((len(ds),2,3))
    if cache['indexes']:kp[cache['indexes']]=cache['points']
    xy=kp[:,:,:2]
    margin=np.maximum(2.,.1*wh)
    inside=((xy>=(b[:,:2]-margin)[:,None,:])&(xy<=(b[:,2:]+margin)[:,None,:])).all(axis=(1,2))
    length=np.linalg.norm(xy[:,0]-xy[:,1],axis=1)
    confident=(kp[:,:,2].min(1)>=.15)&(kp[:,:,2].mean(1)>=.25)&(length>=.15*diag)
    valid=inside&confident
    # 头尾线段在自身框中的覆盖比例。框外邻蜂的高分关键点不能证明这个框里有完整蜜蜂。
    t=np.linspace(0,1,11);body=xy[:,0,None,:]*(1-t[None,:,None])+xy[:,1,None,:]*t[None,:,None]
    coverage=((body>=b[:,:2,None].transpose(0,2,1))&(body<=b[:,2:,None].transpose(0,2,1))).all(2).mean(1)
    candidates={}
    for i,j in zip(*np.where(np.triu(ios>=.2,1))):
        pair=tuple(sorted([ds[i]['track_id'],ds[j]['track_id']]))
        dist=min(np.linalg.norm(xy[i]-xy[j],axis=1).max(),np.linalg.norm(xy[i]-xy[j,::-1],axis=1).max())/min(diag[i],diag[j])
        reason=None;owner=None
        if ios[i,j]>=.8:reason='containment_080';owner=i if area[i]>=area[j] else j
        elif valid[i] and valid[j] and dist<=.20:reason='same_body';owner=i if area[i]>=area[j] else j
        else:
            for good,bad in [(i,j),(j,i)]:
                if not valid[good] or coverage[good]<.8 or area[bad]>1.1*area[good]:continue
                # 空框只借用了邻蜂的一个高分点：该点在自己框外、却落在另一框完整身体线上。
                strongest=int(np.argmax(kp[bad,:,2]));point=xy[bad,strongest];axis=xy[good,1]-xy[good,0]
                projection=float((point-xy[good,0])@axis/max(axis@axis,1e-6));nearest=xy[good,0]+np.clip(projection,0,1)*axis
                outside_self=bool(((point<b[bad,:2]-2)|(point>b[bad,2:]+2)).any())
                on_body=np.linalg.norm(point-nearest)<=.15*diag[good] and -.1<=projection<=1.1
                if not valid[bad] and kp[bad,strongest,2]>=.20 and kp[bad,1-strongest,2]<.15 and outside_self and on_body:
                    reason='empty_box_borrows_one_body_point';owner=good;break
                if coverage[bad]>=.65:continue
                borrowed=((xy[bad]>=(b[good,:2]-margin[good]))&(xy[bad]<=(b[good,2:]+margin[good]))).all()
                weak_pose=kp[bad,:,2].min()>=.10 and kp[bad,:,2].mean()>=.18 and length[bad]>=.1*diag[bad]
                if borrowed and weak_pose and dist<=.30:
                    reason='partial_box_borrows_body';owner=good;break
        distinct=bool(valid[i] and valid[j] and coverage[i]>=.8 and coverage[j]>=.8 and dist>.4 and ios[i,j]<.8)
        candidates[pair]=dict(indexes=[int(i),int(j)],reason=reason,owner_id=int(ds[owner]['track_id']) if owner is not None else None,ios=float(ios[i,j]),distance_ratio=float(dist),distinct=distinct,coverage=[float(coverage[i]),float(coverage[j])])
    return candidates,valid,coverage,area

def process(video):
    paths=sorted((OUT/'observed/frames'/video).glob('*.json'));raw={};evs={};qualities={};pair_stats=defaultdict(lambda:dict(positive=set(),distinct=0,seen=0,owner=Counter(),reasons=Counter()));track_area=defaultdict(list);track_quality=defaultdict(list)
    for p in paths:
        f=int(p.stem.split('_')[1]);row=json.loads(p.read_text());raw[f]=row;cache=json.loads((OUT/'pose_cache'/video/p.name).read_text())
        ev,valid,coverage,area=body_evidence(row['detections'],cache);evs[f]=ev;qualities[f]=(valid,coverage,area)
        for i,d in enumerate(row['detections']):track_area[d['track_id']].append(area[i]);track_quality[d['track_id']].append(float(valid[i])*coverage[i])
        for pair,e in ev.items():
            st=pair_stats[pair];st['seen']+=1;st['distinct']+=int(e['distinct'])
            if e['reason']:st['positive'].add(f);st['owner'][e['owner_id']]+=1;st['reasons'][e['reason']]+=1
    positions=defaultdict(dict)
    for f,row in raw.items():
        for d in row['detections']:positions[d['track_id']][f]=np.array(d['bbox_xyxy'])
    parent={tid:tid for tid in track_area};members={tid:{tid} for tid in parent}
    def find(tid):
        while parent[tid]!=tid:parent[tid]=parent[parent[tid]];tid=parent[tid]
        return tid
    forbidden={pair for pair,st in pair_stats.items() if st['distinct']>max(2,.1*st['seen'])}
    def clearly_separate(a,b):
        separate=0
        for f in positions[a].keys()&positions[b].keys():
            aa=positions[a][f];bb=positions[b][f];inter=np.maximum(0,np.minimum(aa[2:],bb[2:])-np.maximum(aa[:2],bb[:2])).prod()
            ios=inter/max(min((aa[2:]-aa[:2]).prod(),(bb[2:]-bb[:2]).prod()),1e-6)
            distance=np.linalg.norm((aa[:2]+aa[2:]-bb[:2]-bb[2:])/2);diag=min(np.linalg.norm(aa[2:]-aa[:2]),np.linalg.norm(bb[2:]-bb[:2]))
            if ios<.2 and distance>.7*diag:separate+=1
            if separate>=3:return True
        return False
    accepted=[];rejected=[]
    for pair,st in sorted(pair_stats.items(),key=lambda item:-len(item[1]['positive'])):
        fs=st['positive'];confirmed=any(sum(ff in fs for ff in range(f-2,f+3))>=3 for f in fs)
        if not confirmed:continue
        a,b=map(find,pair)
        if a==b:continue
        if any(tuple(sorted([i,j])) in forbidden or clearly_separate(i,j) for i in members[a] for j in members[b]):rejected.append(dict(pair=pair,reason='independent_bodies_seen'));continue
        winner_id=max(st['owner'],key=lambda tid:(st['owner'][tid],np.median(track_area[tid]),-tid));winner=find(winner_id);loser=b if winner==a else a
        parent[loser]=winner;members[winner]|=members.pop(loser)
        accepted.append(dict(pair=pair,winner=winner,loser=loser,positive_frames=len(fs),candidate_frames=st['seen'],reasons=dict(st['reasons'])))
    mapping={tid:find(tid) for tid in parent};observed={};events=[];stats=Counter()
    # 一条旧ID可能混入过别的蜜蜂。只修正已证实的错误归属片段，不把整条轨迹强行合并。
    episodes=defaultdict(list)
    for item in rejected:
        pair=tuple(item['pair']);available=[f for f in raw if pair in evs[f] and not evs[f][pair]['distinct']]
        chunks=[]
        for f in available:
            if not chunks or f-chunks[-1][-1]>30:chunks.append([])
            chunks[-1].append(f)
        for chunk in chunks:
            evidence=[evs[f][pair] for f in chunk if evs[f][pair]['reason']]
            if len(evidence)<3:continue
            votes=Counter(e['owner_id'] for e in evidence);owner=votes.most_common(1)[0][0];bad=pair[1] if pair[0]==owner else pair[0]
            episodes[bad].append(dict(start=chunk[0],end=chunk[-1],owner=mapping[owner],reason='wrong_owner_episode',positive_frames=len(evidence)))
    # 错误归属的片段不能再由旧ID两端插值补回来；真正分离的剩余片段使用分开的身份。
    segment_ids={};next_id=max(parent)+1
    for tid,intervals in episodes.items():
        remaining=sorted(f for f in positions[tid] if not any(e['start']<=f<=e['end'] for e in intervals));sections=[]
        for f in remaining:
            if not sections or any(sections[-1][-1]<e['start']<=e['end']<f for e in intervals):sections.append([])
            sections[-1].append(f)
        longest=max(range(len(sections)),key=lambda i:len(sections[i])) if sections else -1
        for n,section in enumerate(sections):
            assigned=mapping[tid] if n==longest else next_id
            if n!=longest:next_id+=1
            for f in section:segment_ids[(tid,f)]=assigned
    for f,row in raw.items():
        ds=row['detections'];valid,coverage,area=qualities[f];groups=defaultdict(list);kept=[];removed=[]
        for i,d in enumerate(ds):
            tid=d['track_id'];episode=next((e for e in episodes[tid] if e['start']<=f<=e['end']),None)
            if episode:
                item=dict(detection=d,kept_track_id=episode['owner'],reason='wrong_owner_episode');removed.append(item);events.append(dict(video=video,frame=f,**item));stats['wrong_owner_episode']+=1
            else:groups[segment_ids.get((tid,f),mapping[tid])].append(i)
        for tid,indexes in groups.items():
            # 同一身份同帧只留一个框：优先完整身体，随后较完整的大框；不使用逐帧置信度争胜。
            k=max(indexes,key=lambda i:(valid[i] and coverage[i]>=.8,area[i],ds[i]['track_id']==tid))
            d=dict(ds[k]);d['pre_ownership_track_id']=d['track_id'];d['track_id']=tid;kept.append(d)
            for i in indexes:
                if i==k:continue
                reason=evs[f].get(tuple(sorted([ds[i]['track_id'],ds[k]['track_id']])),{}).get('reason') or 'confirmed_duplicate_identity'
                item=dict(detection=ds[i],kept_track_id=tid,kept_original_id=ds[k]['track_id'],reason=reason)
                removed.append(item);events.append(dict(video=video,frame=f,**item));stats[reason]+=1
        observed[f]=dict(row,detections=kept,ownership_removed=removed)
        stats.update(frames=1,observed_before=len(ds),observed_after=len(kept),observed_removed=len(removed))
    # 只以去重后的实测框作为插值端点。无位移、尺寸、重叠或关键点过滤，输出后不再删除。
    tracks=defaultdict(dict)
    for f,row in observed.items():
        for d in row['detections']:tracks[d['track_id']][f]=d
    expected=[]
    for tid,rows in tracks.items():
        fs=sorted(rows)
        for a,b in zip(fs,fs[1:]):
            gap=b-a-1
            if not 0<gap<=90:continue
            expected.append(dict(track_id=tid,anchors=[a,b],missing_frames=gap))
            aa=np.array(rows[a]['bbox_xyxy']);bb=np.array(rows[b]['bbox_xyxy'])
            for f in range(a+1,b):
                alpha=(f-a)/(b-a);box=(aa*(1-alpha)+bb*alpha).tolist()
                observed[f]['detections'].append(dict(track_id=tid,bbox_xyxy=box,origin='reassociation_interpolation',det_confidence=None,class_id=0,keypoints={},verified=False,interpolation_protected=True,interpolation=dict(anchor_frames=[a,b],missing_frames=gap,method='linear_xyxy')));stats['interpolated']+=1
    final_ids={f:{d['track_id'] for d in row['detections']} for f,row in observed.items()}
    for gap in expected:assert all(gap['track_id'] in final_ids[f] for f in range(gap['anchors'][0]+1,gap['anchors'][1]))
    for f,row in observed.items():
        assert len(final_ids[f])==len(row['detections'])
        row['continuous_interpolation']=dict(id_retention_frames=150,max_missing_frames=90,post_fill_filtering=False,same_id_gaps_filled=True,source_anchors='deduplicated_observed_boxes')
        row['final_dedup_removed']=row['ownership_removed'];save(OUT/'final/frames'/video/f'frame_{f:08d}.json',row)
    before={int(p.stem.split('_')[1]):json.loads(p.read_text()) for p in sorted((REFERENCE/video).glob('*.json'))}
    report=dict(stats=dict(stats),merged_identity_pairs=len(accepted),merge_vetoes=len(rejected),before=metrics(before),after=metrics(observed),all_eligible_gaps_complete=True,filtered_interpolations=0,track_id_253_before=[f for f,r in before.items() if any(d['track_id']==253 for d in r['detections'])],track_id_253_after=[f for f,r in observed.items() if any(d['track_id']==mapping.get(253,253) for d in r['detections'])])
    report['ownership_episode_count']=sum(len(v) for v in episodes.values());report['split_identity_segments']=next_id-max(parent)-1
    save(OUT/f'{video}_summary.json',report);save(OUT/f'{video}_aliases.json',mapping);save(OUT/f'{video}_merges.json',dict(accepted=accepted,vetoes=rejected,ownership_episodes=dict(episodes)));save(OUT/f'{video}_events.json',events);save(OUT/f'{video}_expected_gaps.json',expected)
    print(json.dumps(dict(video=video,stats=dict(stats),merged=len(accepted),vetoes=len(rejected),after={k:v for k,v in report['after'].items() if k not in ['cumulative_ids','change_examples']})),flush=True)

def render(video, after_title='AFTER: body ownership + protected filling'):
    cv2.setNumThreads(1);writer=cv2.VideoWriter(str(OUT/f'{video}_global_compare_source.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(3840,1140));assert writer.isOpened();hist=[defaultdict(lambda:deque(maxlen=30)),defaultdict(lambda:deque(maxlen=30))]
    for n,p in enumerate(sorted((OUT/'final/frames'/video).glob('*.json'))):
        f=int(p.stem.split('_')[1]);before=json.loads((REFERENCE/video/p.name).read_text());after=json.loads(p.read_text());im=cv2.imread(str(IMAGES/video/(p.stem+'.jpg')));canvas=np.zeros((1140,3840,3),np.uint8)
        for side,row in enumerate([before,after]):
            panel=im.copy()
            for d in row['detections']:
                tid=d['track_id'];b=np.round(d['bbox_xyxy']).astype(int);pt=(int((b[0]+b[2])/2),int((b[1]+b[3])/2));h=hist[side][tid]
                if h and f-h[-1][0]>1:h.clear()
                h.append((f,pt));c=color(tid)
                for a,z in zip(h,list(h)[1:]):cv2.line(panel,a[1],z[1],c,1,cv2.LINE_AA)
                cv2.rectangle(panel,tuple(b[:2]),tuple(b[2:]),c,2);cv2.putText(panel,str(tid),(max(0,b[0]),max(15,b[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,.48,c,1,cv2.LINE_AA)
            canvas[:1080,side*1920:(side+1)*1920]=panel
            title=f'{video} Frame {f} | '+('BEFORE' if side==0 else after_title)+f' | Boxes {len(row["detections"])}'
            cv2.putText(canvas,title,(side*1920+14,1108),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            cv2.putText(canvas,'5s ID memory | Same-ID gaps <=3s filled | No post-fill deletion | 1s trail',(side*1920+14,1137),cv2.FONT_HERSHEY_SIMPLEX,.6,(230,230,230),1)
        writer.write(canvas)
        if n==150:cv2.imwrite(str(OUT/f'{video}_global_preview.jpg'),canvas)
    writer.release()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--device',type=int,default=0);p.add_argument('--stage',choices=['all','process','render'],default='all');a=p.parse_args();OUT.mkdir(exist_ok=True)
    if a.stage=='all':
        prepare(a.video);base.SOURCE=OUT/'observed/frames';base.OUT=OUT;base.CANDIDATE_IOS=.2;base.infer(a.video,a.device)
    if a.stage in ['all','process']:process(a.video)
    if a.stage in ['all','render']:render(a.video)
