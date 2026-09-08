import os,json,sys,sqlite3,time,hashlib
from pathlib import Path
R=Path('/root/bee_tracking_pilot_20260907')
O=R/'full_20pct_20260907'
SOURCE=Path('/root/autodl-tmp/indoor_ids_full_latest_20260906/final/frames')
IMAGES=Path('/root/autodl-tmp/flywheel_indoor_scene_b_20260904/frames')
VIDEOS=['B-5-1','B-5-2','B-5-3','B-5-4']
def atomic_json(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)
def load_detections(video,fr):
 raw=json.load(open(SOURCE/video/f'frame_{fr:08d}.json'));ds=[];seen=set()
 for d in raw['detections']:
  if d.get('origin')=='reassociation_interpolation':continue
  a,b,c,e=d['bbox_xyxy'];xy=(int((a+c)/2),int((b+e)/2))
  if not (0<xy[0]<1920 and 0<xy[1]<1080) or xy in seen:continue
  seen.add(xy);ds.append(d)
 return ds
def prepare():
 O.mkdir(exist_ok=True)
 counts={}
 for v in VIDEOS:
  paths=sorted((SOURCE/v).glob('frame_*.json'));frames=[int(p.stem.split('_')[-1]) for p in paths]
  assert frames==list(range(len(frames))),(v,'non-contiguous frames')
  counts[v]=len(frames);(O/v/'embeddings').mkdir(parents=True,exist_ok=True)
 reuse={}
 for base in sorted((R/'gt_eval_01_03_20260907').iterdir()):
  if not (base/'inputs.json').exists() or not (base/'embedding_ready.json').exists() and not (base/'embedding_ready_0.json').exists():continue
  meta=json.load(open(base/'inputs.json'))
  if not isinstance(meta,dict) or meta.get('roi')!=[0,0,1920,1080]:continue
  v=meta['video']
  for i,rec in enumerate(meta['records']):
   p=base/'embeddings'/f'{i:06d}.npy'
   if p.exists():reuse[f"{v}/{rec['source_frame']}"]=str(p)
 atomic_json(O/'reuse_index.json',reuse)
 config={'videos':counts,'total_frames':sum(counts.values()),'fps':30,'frame_index_base':0,'image_size':[1920,1080],'detector_source':str(SOURCE),'exclude_origin':'reassociation_interpolation','appearance_checkpoint':str(next(R.glob('data/**/model_005000.ckpt.index'))),'internal_retention_frames':150,'edge_retention_frames':75,'minimum_observations':30,'maximum_endpoint_gap_frames':90,'overlap_threshold':.2,'overlap_definition':'intersection/minimum_box_area','minimum_intersection_pixels':16,'suppression_only_interpolation':True,'whole_video_association':True,'restart_ids_at_chunks':False,'reuse_candidates':len(reuse)}
 atomic_json(O/'config.json',config)
 db=sqlite3.connect(O/'embedding_jobs.sqlite');db.execute('create table if not exists jobs(video text,start integer,end integer,status text,gpu integer,updated real,primary key(video,start))')
 for start in range(0,max(counts.values()),240):
  for v in ['B-5-3','B-5-1','B-5-4','B-5-2']:
   if start<counts[v]:db.execute('insert or ignore into jobs values(?,?,?,?,?,?)',(v,start,min(start+240,counts[v]),'pending',None,time.time()))
 db.commit();db.close()
 print(json.dumps(config),flush=True)
if __name__=='__main__':prepare()
