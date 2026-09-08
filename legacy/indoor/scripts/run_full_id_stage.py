"""四视频全量流程的独立阶段；复用小测试规则，整视频关联不分块重置ID。"""
import argparse,json,os,time,copy,gzip
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np
import dedup_overlap_tracks as geometry_module
import dedup_final_temporal_pose as pose_runner
import dedup_containment_first as containment
import track_interior_memory as tracking
import reassociate_final_bees as common
import keep_interpolation_and_ownership as ownership
import consolidate_dense_and_expand_thin as dense
import repair_local_identity_conflicts as repair
from full_id_storage import JsonPath,save

ROOT=JsonPath('/root/autodl-tmp/indoor_ids_full_latest_20260906')
SOURCE=Path('/root/autodl-tmp/indoor_tracking_allframes_no_pose_filter_20260906/frames')
EXPECTED={'B-5-1':8996,'B-5-2':8996,'B-5-3':9013,'B-5-4':9005}
MEMORY=ROOT/'03_memory/offline_birth4_edge150_inside150_fill90'
for module in [geometry_module,pose_runner,tracking,ownership,dense,repair]:module.save=save

def progress(stage,video,**kw):
    record=dict(stage=stage,video=video,time=time.time(),**kw)
    suffix=f".gpu{kw['device']}" if 'frames' in kw and 'device' in kw else ''
    p=Path(ROOT)/'progress'/f'{video}{suffix}.json';p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix(f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(record));os.replace(tmp,p)
    print(json.dumps(record),flush=True)

def prepare(video):
    paths=sorted((SOURCE/video).glob('*.json'));assert len(paths)==EXPECTED[video]
    stats=Counter()
    for n,p in enumerate(paths):
        assert int(p.stem.split('_')[1])==n
        row=json.loads(p.read_text());ds=[d for d in row['detections'] if d.get('origin') not in ['track_interpolation','reassociation_interpolation']]
        assert len({d['track_id'] for d in ds})==len(ds)
        result={k:row[k] for k in ['video','frame','width','height'] if k in row};result['detections']=ds
        save(ROOT/'01_input/frames'/video/p.name,result);stats.update(frames=1,observed=len(ds),old_interpolation_discarded=len(row['detections'])-len(ds))
        if (n+1)%500==0:progress('prepare',video,frames=n+1,total=len(paths))
    save(ROOT/'01_input'/f'{video}_summary.json',dict(stats))

def infer(video,device,source,destination,stage,frame_start=None,frame_end=None):
    import torch,sys
    torch.set_num_threads(4);torch.cuda.set_device(device);cv2.setNumThreads(1)
    sys.path.insert(0,'/root/autodl-tmp/flywheel_indoor_scene_b_20260904')
    from indoor_adaptive_recheck_4090 import PoseVerifier
    vit=Path('/root/autodl-tmp/flywheel_outdoor_pose_20260903/source/ViTPose')
    manifest=json.loads(Path('/root/autodl-tmp/indoor_yolo_v3_pose_raw_allframes_20260905/run_manifest.json').read_text())
    pose=PoseVerifier(vit,vit/'configs/bee_pose/subset_13_20260827/scene_B_only_13.py',Path(manifest['pose']),device)
    from mmpose.datasets.pipelines import Compose
    loaders=[t for t in pose.pipeline.transforms if t.__class__.__name__=='LoadImageFromFile'];assert len(loaders)==1 and loaders[0].channel_order=='rgb'
    def loaded(data):data['image_file']=None;return data
    pose.pipeline=Compose([loaded if t in loaders else t for t in pose.pipeline.transforms])
    from prefetched_pose import prepare as prepare_pose,predict as predict_pose
    paths=sorted((source/video).glob('*.json'));assert len(paths)==EXPECTED[video]
    if frame_start is not None:paths=paths[frame_start:frame_end]
    pending=[p for p in paths if not (destination/'pose_cache'/video/p.name).exists()];done=len(paths)-len(pending);crops=0;started=time.time()
    def load(p):
        ds=json.loads(p.read_text())['detections'];_,_,ios,_=geometry_module.geometry(ds);indexes=np.flatnonzero((ios>=.2).any(1)).tolist()
        im=cv2.imread(str(geometry_module.IMAGES/video/(p.stem+'.jpg')));assert im is not None
        prepared=prepare_pose(pose,cv2.cvtColor(im,cv2.COLOR_BGR2RGB),[ds[i]['bbox_xyxy']+[1.] for i in indexes])
        return p,indexes,prepared
    # 最多预取四帧，提前完成裁剪和归一化，让CPU准备与GPU推理重叠。
    with ThreadPoolExecutor(2) as pool:
        iterator=iter(pending);queue=[]
        for _ in range(4):
            p=next(iterator,None)
            if p is not None:queue.append(pool.submit(load,p))
        while queue:
            p,indexes,prepared=queue.pop(0).result();nxt=next(iterator,None)
            if nxt is not None:queue.append(pool.submit(load,nxt))
            points=predict_pose(pose,prepared,batch_size=256)
            save(destination/'pose_cache'/video/p.name,dict(indexes=indexes,points=np.asarray(points).tolist()));done+=1;crops+=len(indexes)
            if done%100==0:progress(stage,video,frames=done,total=len(paths),device=device,crops=crops,seconds=round(time.time()-started,1))
    assert all((destination/'pose_cache'/video/p.name).exists() for p in paths)
    progress(stage,video,frames=len(paths),total=len(paths),device=device,crops=crops,seconds=round(time.time()-started,1),frame_start=frame_start,frame_end=frame_end)

def prefilter(video):
    stats=Counter();paths=sorted((ROOT/'01_input/frames'/video).glob('*.json'))
    support=Counter(d['track_id'] for p in paths for d in json.loads(p.read_text())['detections'])
    for n,p in enumerate(paths,1):
        row=json.loads(p.read_text());ds=row['detections']
        if not ds:
            save(ROOT/'02_prefilter/frames'/video/p.name,row);stats['frames']+=1;continue
        inter,iou,ios,diag=geometry_module.geometry(ds);b=np.array([d['bbox_xyxy'] for d in ds]);wh=b[:,2:]-b[:,:2];area=wh.prod(1)
        rank=sorted(range(len(ds)),key=lambda i:(-area[i],-support[ds[i]['track_id']],ds[i]['track_id']))
        kept,removed=containment.suppress(rank,ios>=.8,range(len(ds)))
        cache=json.loads((ROOT/'01_input/pose_cache'/video/p.name).read_text());kp=np.zeros((len(ds),2,3))
        if cache['indexes']:kp[cache['indexes']]=cache['points']
        margin=np.maximum(2.,.1*wh);inside=((kp[:,:,:2]>=(b[:,:2]-margin)[:,None,:])&(kp[:,:,:2]<=(b[:,2:]+margin)[:,None,:])).all(axis=(1,2))
        valid=inside&(kp[:,:,2].min(1)>=.15)&(kp[:,:,2].mean(1)>=.25)&(np.linalg.norm(kp[:,0,:2]-kp[:,1,:2],axis=1)>=.15*diag)
        distance=np.linalg.norm(kp[:,None,:,:2]-kp[None,:,:,:2],axis=3).max(2)
        duplicate=(inter>0)&(ios>=.5)&valid[:,None]&valid[None,:]&(distance<=.12*np.minimum(diag[:,None],diag[None,:]))
        protected=list(set(removed.values()))
        if protected:duplicate[:,protected]=False
        kept,pose_removed=containment.suppress(rank,duplicate,kept)
        final=[i for i in kept if wh[i].max()/wh[i].min()<6]
        stats.update(frames=1,before=len(ds),containment_removed=len(removed),pose_removed=len(pose_removed),extreme_thin_removed=len(kept)-len(final),after=len(final))
        save(ROOT/'02_prefilter/frames'/video/p.name,dict(row,detections=[ds[i] for i in final]))
        if n%500==0:progress('prefilter',video,frames=n,total=len(paths))
    save(ROOT/'02_prefilter'/f'{video}_summary.json',dict(stats))

def configure():
    tracking.SOURCE=ROOT/'02_prefilter/frames';tracking.OUT=ROOT/'03_memory';tracking.REFERENCE=ROOT/'no_clip_reference'
    ownership.RAW=MEMORY/'pending_pose/frames';ownership.OUT=ROOT/'04_ownership';ownership.REFERENCE=ROOT/'02_prefilter/frames'
    dense.SOURCE=ownership.OUT/'final/frames';dense.OUT=ROOT/'05_dense'
    repair.SOURCE=dense.OUT/'final/frames';repair.OUT=ROOT/'06_repair'

def audit_export(video):
    stats=Counter();tracks=defaultdict(dict);before_ids=set();after_ids=set();final_dir=Path(ROOT)/'final/frames'/video;final_dir.mkdir(parents=True,exist_ok=True)
    paths=sorted((repair.SOURCE/video).glob('*.json'));assert len(paths)==EXPECTED[video]
    mot=Path(ROOT)/'final/mot'/f'{video}.txt.gz';mot.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(mot,'wt',compresslevel=1) as mf:
        for n,p in enumerate(paths):
            f=int(p.stem.split('_')[1]);assert f==n
            old=json.loads(p.read_text());new=json.loads((repair.OUT/'final/frames'/video/p.name).read_text())
            a={d['track_id']:d for d in old['detections'] if d.get('origin')!='reassociation_interpolation'}
            b={d['pre_local_repair_track_id']:d for d in new['detections'] if d.get('origin')!='reassociation_interpolation'}
            assert a.keys()==b.keys();assert len({d['track_id'] for d in new['detections']})==len(new['detections'])
            for tid,d in a.items():assert d['bbox_xyxy']==b[tid]['bbox_xyxy'];stats['observed_unchanged']+=1
            before_ids.update(d['track_id'] for d in old['detections']);after_ids.update(d['track_id'] for d in new['detections'])
            final=[]
            for d in new['detections']:
                tracks[d['track_id']][f]=(d['bbox_xyxy'],d.get('origin')=='reassociation_interpolation',d.get('interpolation_protected',False))
                keep=['track_id','bbox_xyxy','origin','det_confidence','class_id','keypoints','interpolation','interpolation_protected','pre_local_repair_track_id','local_identity_repair','independent_thin_expansion']
                item={k:d[k] for k in keep if k in d};final.append(item);x,y,z,q=d['bbox_xyxy'];confidence=d.get('det_confidence');confidence=1. if confidence is None else confidence
                mf.write(f'{f+1},{d["track_id"]},{x:.6f},{y:.6f},{z-x:.6f},{q-y:.6f},{confidence:.6f},-1,-1,-1\n')
            result=dict(video=video,frame=f,width=1920,height=1080,fps=30,detections=final,continuous_interpolation=new['continuous_interpolation'])
            target=final_dir/p.name;tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(result,separators=(',',':')));os.replace(tmp,target)
            stats.update(frames=1,boxes=len(final))
            if (n+1)%500==0:progress('audit_export',video,frames=n+1,total=len(paths))
    assert before_ids==after_ids
    for tid,rows in tracks.items():
        fs=sorted(f for f,d in rows.items() if not d[1])
        for a,b in zip(fs,fs[1:]):
            if not 0<b-a-1<=90:continue
            stats['eligible_gaps']+=1
            for f in range(a+1,b):
                assert f in rows and rows[f][1] and rows[f][2];t=(f-a)/(b-a);expected=np.array(rows[a][0])*(1-t)+np.array(rows[b][0])*t
                assert np.allclose(rows[f][0],expected);stats['protected_fill_verified']+=1
    report=dict(video=video,completed=True,stats=dict(stats),unique_ids=len(after_ids),local_repair_new_ids=0,local_repair_observed_removed=0,missing_eligible_frames=0,post_fill_removed=0,json_frame_index_base=0,mot_frame_index_base=1)
    target=Path(ROOT)/f'{video}_audit.json';target.write_text(json.dumps(report,indent=2));progress('audit_export',video,completed=True,**dict(stats))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True,choices=list(EXPECTED));p.add_argument('--stage',required=True);p.add_argument('--device',type=int,default=0);p.add_argument('--frame-start',type=int);p.add_argument('--frame-end',type=int);a=p.parse_args();configure();cv2.setNumThreads(1)
    bounds=dict(frame_start=a.frame_start,frame_end=a.frame_end)
    if a.frame_start is not None:assert a.stage.startswith('pose_') and 0<=a.frame_start<a.frame_end<=EXPECTED[a.video]
    progress(a.stage,a.video,started=True,device=a.device)
    if a.stage=='prepare':prepare(a.video)
    elif a.stage=='pose_pre':infer(a.video,a.device,ROOT/'01_input/frames',ROOT/'01_input',a.stage,**bounds)
    elif a.stage=='prefilter':prefilter(a.video)
    elif a.stage=='track':tracking.offline(a.video,birth_cost=4,edge_retention_frames=150,interior_retention_frames=150,max_gap_frames=90,observed_only=True)
    elif a.stage=='pose_owner':
        link=Path(ownership.OUT)/'observed/frames'/a.video;link.parent.mkdir(parents=True,exist_ok=True)
        try:link.symlink_to(Path(ownership.RAW)/a.video,target_is_directory=True)
        except FileExistsError:pass
        infer(a.video,a.device,ownership.OUT/'observed/frames',ownership.OUT,a.stage,**bounds)
    elif a.stage=='owner':ownership.process(a.video)
    elif a.stage=='pose_dense':infer(a.video,a.device,dense.SOURCE,dense.OUT,a.stage,**bounds)
    elif a.stage=='dense':dense.process(a.video)
    elif a.stage=='pose_repair':infer(a.video,a.device,repair.SOURCE,repair.OUT,a.stage,**bounds)
    elif a.stage=='repair':repair.process(a.video)
    elif a.stage=='audit_export':
        audit_export(a.video)
        from render_full_id_previews import render
        progress('preview',a.video,started=True);render(a.video)
    else:raise ValueError(a.stage)
    if a.frame_start is None:
        marker=Path(ROOT)/'done'/f'{a.video}_{a.stage}.json';marker.parent.mkdir(exist_ok=True,parents=True);marker.write_text(json.dumps(dict(completed=True,time=time.time())))
    progress(a.stage,a.video,completed=True,chunk=a.frame_start is not None,frame_start=a.frame_start,frame_end=a.frame_end,device=a.device)
