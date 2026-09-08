"""内部身份持续保留；仅实测框构建短轨，离线一对一接续后再分配身份与插值。"""
import argparse,json,copy,time
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
import reassociate_final_bees as common
from dedup_overlap_tracks import save

ROOT=common.ROOT;SOURCE=common.SOURCE;OUT=ROOT/'indoor_interior_memory_20260906';REFERENCE=ROOT/'indoor_final_reassociation_20260906'

def interior(box):
    c=(box[:2]+box[2:])/2
    return bool(80<c[0]<1840 and 80<c[1]<1000)

def memory_only(video):
    common.OUT=OUT/'memory_only';common.OUT.mkdir(parents=True,exist_ok=True);common.associate(video,interior_memory=True);common.stitch(video)

def offline(video,birth_cost=1.8,edge_retention_frames=30,interior_retention_frames=None,max_gap_frames=150,observed_only=False):
    started=time.time();cv2.setNumThreads(1)
    raw={int(p.stem.split('_')[1]):json.loads(p.read_text()) for p in sorted((SOURCE/video).glob('*.json'))}
    groups={};records={};next_piece=0;previous=None;stats=Counter()
    for f,row in raw.items():
        original=row['detections'];ds=[d for d in original if d.get('origin')!='track_interpolation'];records[f]=dict(row,detections=copy.deepcopy(ds));stats['observed_boxes']+=len(ds);stats['legacy_interpolation_rebuilt']+=len(original)-len(ds)
        boxes=np.asarray([d['bbox_xyxy'] for d in ds]).reshape(-1,4);im=cv2.imread(str(common.IMAGES/video/f'frame_{f:08d}.jpg'),0);features=common.describe(im,boxes);assignment={}
        # 只锁定连续帧的可靠短轨；失配先成为片段，不立即认定出现新蜜蜂。
        if previous is not None and len(boxes):
            pb,pids=previous;dist,iou,diag,ratio=common.pair_geometry(pb,boxes);near=dist.argmin(1);back=dist.argmin(0);used_i=set();used_j=set()
            for i,j in enumerate(near):
                if back[j]!=i or dist[i,j]>6 or iou[i,j]<.5 or ratio[i,j]>2.5:continue
                assignment[int(j)]=pids[i];used_i.add(i);used_j.add(int(j))
            ii=[i for i in range(len(pb)) if i not in used_i];jj=[j for j in range(len(boxes)) if j not in used_j]
            if ii and jj:
                dd=dist[np.ix_(ii,jj)];ov=iou[np.ix_(ii,jj)];ra=ratio[np.ix_(ii,jj)];dg=diag[np.ix_(ii,jj)];valid=(dd<=np.clip(.4*dg,12,24))&(ov>=.2)&(ra<=3)
                cost=dd/np.maximum(dg,1)+1.1*(1-ov);rr,cc=linear_sum_assignment(np.where(valid,cost,1e6))
                for r,c in zip(rr,cc):
                    if valid[r,c]:assignment[jj[c]]=pids[ii[r]]
        pids=[]
        for j,d in enumerate(ds):
            if j not in assignment:assignment[j]=next_piece;groups[next_piece]=[];next_piece+=1
            piece=assignment[j];groups[piece].append((f,j,boxes[j].copy(),features[j].copy()));pids.append(piece)
        previous=(boxes,pids) if len(boxes) else None
        if len(records)%500==0:print(json.dumps(dict(stage='tracklets',video=video,frames=len(records),total=len(raw),pieces=next_piece)),flush=True)
    n=len(groups);starts=np.zeros(n,int);ends=np.zeros(n,int);first=np.zeros((n,4));last=np.zeros((n,4));front=np.zeros((n,64));back=np.zeros((n,64));inside=np.zeros(n,bool)
    for i,rows in groups.items():
        starts[i]=rows[0][0];ends[i]=rows[-1][0];first[i]=rows[0][2];last[i]=rows[-1][2];inside[i]=interior(last[i])
        # 每个片段最多使用前/后150帧（5秒）的外观信息，降低单帧裁剪抖动。
        front[i]=np.mean([r[3] for r in rows if r[0]-rows[0][0]<150],axis=0);back[i]=np.mean([r[3] for r in rows if rows[-1][0]-r[0]<150],axis=0)
        front[i]/=max(np.linalg.norm(front[i]),1e-6);back[i]/=max(np.linalg.norm(back[i]),1e-6)
    # DAG路径覆盖：片段结束只能连到时间更晚的片段开始；每端最多一条连接。
    centers=(first[:,:2]+first[:,2:])/2
    finite_window=max(edge_retention_frames,interior_retention_frames) if interior_retention_frames is not None else None
    tree=cKDTree(np.column_stack([centers,starts*45/finite_window]) if finite_window else centers);rr=[];cc=[];values=[];edge_details={}
    for i in range(n):
        center=(last[i,:2]+last[i,2:])/2
        js=tree.query_ball_point(np.r_[center,ends[i]*45/finite_window],45+1e-9,p=np.inf) if finite_window else tree.query_ball_point(center,45)
        retention=interior_retention_frames if inside[i] else edge_retention_frames
        js=[j for j in js if starts[j]>ends[i] and (retention is None or starts[j]-ends[i]<=retention)]
        if not js:continue
        dist,iou,diag,ratio=common.pair_geometry(last[i:i+1],first[js]);sim=back[i]@front[js].T;gap=starts[js]-ends[i]
        valid=(dist[0]<=np.clip(.65*diag[0],18,45))&(ratio[0]<=3.5)&((iou[0]>=.08)|(dist[0]<=.3*diag[0]))&((sim>=.15)|(dist[0]<=.12*diag[0]))
        for k,j in enumerate(js):
            if not valid[k]:continue
            cost=float(dist[0,k]/max(diag[0,k],1)+.8*(1-iou[0,k])+.35*(1-sim[k])+.08*np.log1p(gap[k]/30))
            rr.append(i);cc.append(j);values.append(cost+.001);edge_details[(i,j)]=dict(gap_frames=int(gap[k]),distance=float(dist[0,k]),iou=float(iou[0,k]),appearance=float(sim[k]),interior=bool(inside[i]))
    edge_count=len(values)
    # 每条片段可不接续；内部的新身份成本略高，仍允许在证据不足时新建。
    for i in range(n):rr.append(i);cc.append(n+i);values.append(birth_cost if inside[i] else birth_cost*.8)
    graph=coo_matrix((values,(rr,cc)),shape=(n,2*n)).tocsr();left,right=min_weight_full_bipartite_matching(graph)
    successor={int(i):int(j) for i,j in zip(left,right) if j<n};predecessor={j:i for i,j in successor.items()};assert len(predecessor)==len(successor)
    roots=sorted([i for i in range(n) if i not in predecessor],key=lambda i:(starts[i],i));piece_id={};links=[]
    for tid,root in enumerate(roots,1):
        i=root
        while True:
            piece_id[i]=tid
            if i not in successor:break
            j=successor[i];assert ends[i]<starts[j];links.append(dict(from_piece=i,to_piece=j,track_id=tid,**edge_details[(i,j)]));i=j
    assert len(piece_id)==n
    for piece,rows in groups.items():
        for f,j,_,_ in rows:
            d=records[f]['detections'][j];d['pre_reassociation_track_id']=d['track_id'];d['track_id']=piece_id[piece];d['observation_tracklet_id']=piece
    mode=f'offline_birth{birth_cost:g}_edge{edge_retention_frames}'
    if interior_retention_frames is not None:mode+=f'_inside{interior_retention_frames}'
    if max_gap_frames!=150:mode+=f'_fill{max_gap_frames}'
    folder=OUT/mode;folder.mkdir(parents=True,exist_ok=True)
    for f,row in records.items():
        assert len({d['track_id'] for d in row['detections']})==len(row['detections'])
        row['reassociation']=dict(interior_memory_no_expiry=interior_retention_frames is None,interior_retention_frames=interior_retention_frames,edge_band_px=80,edge_retention_frames=edge_retention_frames,appearance_window_frames=150,interpolation_can_create_id=False,offline_one_to_one_path_cover=True,new_identity_cost=birth_cost)
        save(folder/'associated/frames'/video/f'frame_{f:08d}.json',row)
    reference_path=REFERENCE/f'{video}_summary.json'
    reference=json.loads(reference_path.read_text())['final'] if reference_path.exists() else None
    report=dict(before=reference,associated=common.metrics(records),tracklets=n,candidate_links=edge_count,selected_links=len(links),input_stats=dict(stats),seconds=time.time()-started)
    save(folder/f'{video}_association_summary.json',report);save(folder/f'{video}_links.json',links)
    print(json.dumps(dict(video=video,tracklets=n,links=len(links),stats=dict(stats),metrics={k:v for k,v in report['associated'].items() if k not in ['cumulative_ids','change_examples']})),flush=True)
    common.OUT=folder;common.fill(video,max_gap_frames=max_gap_frames,observed_only=observed_only)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--mode',choices=['memory_only','offline'],required=True);p.add_argument('--birth-cost',type=float,default=1.8);p.add_argument('--edge-retention-frames',type=int,default=30);p.add_argument('--interior-retention-frames',type=int);p.add_argument('--max-gap-frames',type=int,default=150);a=p.parse_args();OUT.mkdir(exist_ok=True)
    if a.mode=='memory_only':memory_only(a.video)
    else:offline(a.video,a.birth_cost,a.edge_retention_frames,a.interior_retention_frames,a.max_gap_frames)
