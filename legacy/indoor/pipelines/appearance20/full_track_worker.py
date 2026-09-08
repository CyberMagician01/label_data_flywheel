import os,sys,time,json,pickle,argparse
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_CPP_MIN_LOG_LEVEL']='3';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
from full_common import *
from full_assoc_vectorized import optimized_source
sys.path.insert(0,str(R/'repo'))
from tracking import track

# 保持原候选条件和计算顺序，只把候选外观距离批量计算。
def fast_dist(t,active,last_seen,lens,fr_i,preds):
 res=[];now=int(fr_i[0]);maxlen=np.max(lens)
 for slot in np.flatnonzero(active):
  last=int(last_seen[slot,-1]);x,y=t[last,slot,:2];gap=now-last
  butt=np.all(t[last_seen[slot,5:],slot,2]==1)
  dist=np.sqrt((x-preds[:,0])**2+(y-preds[:,1])**2)
  dist=dist*3 if butt else dist/np.floor(np.sqrt(gap))
  if gap>track.get_look_back(x,y,butt):continue
  ix=np.flatnonzero(dist<=track.SQ)
  if not len(ix):continue
  vs=t[last_seen[slot],slot,4:].astype(np.float64)
  vd=np.sqrt(np.sum((preds[ix,None,4:]-vs[None,:,:])**2,axis=2)).min(axis=1)
  valid=vd<=track.VIS_CUT;ix=ix[valid];vd=vd[valid]
  block=np.empty((len(ix),5));block[:,0]=slot;block[:,1]=ix;block[:,2]=dist[ix];block[:,3]=vd;block[:,4]=1-lens[slot]/maxlen;res.append(block)
 return np.concatenate(res) if res else np.empty((0,5))

def fast_matches(d):
 if not len(d):return np.empty((0,2),int)
 order=np.argsort(d[:,3]+d[:,2]/track.DDIV+d[:,4],kind='stable')
 useda=set();usedb=set();out=[]
 for k in order:
  a,b=map(int,d[k,:2])
  if a not in useda and b not in usedb:out.append((a,b));useda.add(a);usedb.add(b)
 return np.array(out,int).reshape(-1,2)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('video');ap.add_argument('--regression');ap.add_argument('--stop-at',type=int);args=ap.parse_args()
 v=args.video
 if args.regression:
  old=R/'gt_eval_01_03_20260907'/args.regression;meta=json.load(open(old/'inputs.json'));N=len(meta['records']);base=O/'regression'/args.regression
  def detections(fr):
   return [{'bbox_xyxy':d['box'],'keypoints':d['keypoints']} for d in meta['records'][fr]['detections']]
  def embpath(fr):return old/'embeddings'/f'{fr:06d}.npy'
 else:
  N=json.load(open(O/'config.json'))['videos'][v];base=O/v
  def detections(fr):return load_detections(v,fr)
  def embpath(fr):return base/'embeddings'/f'{fr:06d}.npy'
 base.mkdir(parents=True,exist_ok=True);tracks=base/'track_fragments';tracks.mkdir(exist_ok=True)
 ready=base/'tracking_complete.json'
 if ready.exists():print('ALREADY_COMPLETE',v,flush=True);return
 ns={'np':np,'track':track,'poses':{},'pose_centers':{},'mode':'momentum_pose'}
 source=optimized_source((R/'track_gt_eval.py').read_text());exec(source[source.index('def pose('):source.index('track.FR2=')],ns);ns['original_dist']=fast_dist
 track.get_look_back=lambda x,y,butt: (75 if x<192 or x>1728 or y<108 or y>972 else 150)*(10 if butt else 1)
 def read(fr):
  p=embpath(fr);wait=time.time()
  while not p.exists():
   atomic_json(base/'tracking_progress.json',{'state':'waiting_embeddings','frame':fr,'total_frames':N,'updated':time.time()})
   if time.time()-wait>1800:raise RuntimeError('embedding wait exceeded 30 minutes')
   time.sleep(3)
  preds=np.load(p).astype(np.float64);ds=detections(fr)
  assert len(ds)==len(preds),(v,fr,len(ds),len(preds))
  for d,row in zip(ds,preds):
   bb=d['bbox_xyxy'];x,y=int((bb[0]+bb[2])/2),int((bb[1]+bb[3])/2);assert (x,y)==tuple(row[:2])
   kp=d.get('keypoints',{});h=kp.get('head');a=kp.get('abdomen_tip')
   if h and a:
    delta=np.array(h[:2])-a[:2];length=float(np.linalg.norm(delta));q=float(min(h[2],a[2]))
    ns['poses'][(fr,x,y)]=(float(np.arctan2(delta[1],delta[0])),length,q if length>=5 else 0.)
  return preds
 cp=base/'tracking_checkpoint.pkl'
 first=read(0);initial=len(first);capacity=max(initial*4,32)
 t=np.zeros((N+1,capacity,68),np.float32);active=np.zeros(capacity,bool);last_seen=np.full((capacity,10),-1,int);lens=np.full(capacity,-1,int)
 t[0,:initial]=first;active[:initial]=True;last_seen[:initial]=0;lens[:initial]=1
 tr_nb=0;saved_records=0;discarded_records=0;raw_ids=initial;start=1
 if cp.exists():
  with open(cp,'rb') as fh:state=pickle.load(fh)
  assert state['N']==N and state['video']==v
  capacity=len(state['active'])
  if capacity!=t.shape[1]:t=np.zeros((N+1,capacity,68),np.float32)
  active=state['active'];last_seen=state['last_seen'];lens=state['lens'];tr_nb=state['tr_nb'];saved_records=state['saved_records'];discarded_records=state['discarded_records'];raw_ids=state['raw_ids'];start=state['frame']+1
  t[0,:initial]=0
  ns['poses']=state['poses'];ns['rejection_stats']=state['rejection_stats']
  for slot,rows,features in state['histories']:
   t[rows[:,0].astype(int),slot,:4]=rows[:,1:]
   t[last_seen[slot],slot,:]=features
  del state
  print('RESUMED',v,start,flush=True)
 def save(slot):
  nonlocal tr_nb,saved_records,discarded_records
  fs=np.flatnonzero(t[:,slot,0]);rows=np.column_stack([fs,t[fs,slot,:4]]).astype(np.int32)
  if len(rows)>=30:
   p=tracks/f'{tr_nb:06d}.npy'
   with open(str(p)+'.tmp','wb') as fh:np.save(fh,rows)
   os.replace(str(p)+'.tmp',p);saved_records+=len(rows);tr_nb+=1
  else:discarded_records+=len(rows)
 def checkpoint(fr):
  histories=[];needed=set()
  for slot in np.flatnonzero(active):
   fs=np.flatnonzero(t[:,slot,0]);rows=np.column_stack([fs,t[fs,slot,:4]]).astype(np.int32)
   histories.append((int(slot),rows,t[last_seen[slot],slot,:].copy()))
   for f in np.unique(last_seen[slot]):needed.add((int(f),int(t[f,slot,0]),int(t[f,slot,1])))
  ns['poses']={k:p for k,p in ns['poses'].items() if k in needed}
  state={'N':N,'video':v,'frame':fr,'active':active,'last_seen':last_seen,'lens':lens,'tr_nb':tr_nb,'saved_records':saved_records,'discarded_records':discarded_records,'raw_ids':raw_ids,'poses':ns['poses'],'histories':histories,'rejection_stats':ns['rejection_stats']}
  with open(str(cp)+'.tmp','wb') as fh:pickle.dump(state,fh,protocol=5)
  os.replace(str(cp)+'.tmp',cp)
 started=time.time()
 for fr in range(start,N):
  preds=read(fr)
  d=ns['gated_dist'](t,active,last_seen,lens,np.full(len(preds),fr),preds) if len(preds) else np.empty((0,5))
  matches=fast_matches(d);ma=set(matches[:,0]);mb=set(matches[:,1])
  unmatched_past=[i for i in np.flatnonzero(active) if i not in ma];unmatched_current=[i for i in range(len(preds)) if i not in mb]
  for slot,di in matches:
   t[fr,slot]=preds[di];last_seen[slot,:-1]=last_seen[slot,1:];last_seen[slot,-1]=fr;lens[slot]+=1
  for slot in unmatched_past:
   last=last_seen[slot,-1];x,y=t[last,slot,:2];butt=np.all(t[last_seen[slot,5:],slot,2]==1)
   if fr-last>track.get_look_back(x,y,butt):
    save(slot);active[slot]=False;t[:,slot]=0
  for di in unmatched_current:
   free=np.flatnonzero(~active)
   if not len(free):
    grow=max(128,capacity//4);t=np.pad(t,((0,0),(0,grow),(0,0)));active=np.pad(active,(0,grow));last_seen=np.pad(last_seen,((0,grow),(0,0)),constant_values=-1);lens=np.pad(lens,(0,grow),constant_values=-1);free=np.array([capacity]);capacity+=grow
   slot=free[0];t[fr,slot]=preds[di];active[slot]=True;last_seen[slot]=fr;lens[slot]=1;raw_ids+=1
  if fr%100==0 or fr==N-1:
   progress={'state':'tracking','video':v,'frame':fr,'total_frames':N,'active_tracks':int(active.sum()),'saved_tracks':tr_nb,'raw_ids':raw_ids,'seconds_this_run':time.time()-started,'frames_this_run':fr-start+1,'updated':time.time()}
   atomic_json(base/'tracking_progress.json',progress);print(json.dumps(progress),flush=True)
  if fr%300==0 or fr==args.stop_at or fr==N-1:checkpoint(fr)
  if fr==args.stop_at:print('PAUSED_FOR_RESUME_TEST',flush=True);return
 for slot in np.flatnonzero(active):save(slot)
 items=[]
 for i in range(tr_nb):
  rows=np.load(tracks/f'{i:06d}.npy');items.append({'fragment':i,'start':int(rows[0,0]),'end':int(rows[-1,0]),'observations':len(rows)})
 items.sort(key=lambda x:(x['start'],-(x['end']-x['start']+1)))
 for tid,item in enumerate(items,1):item['track_id']=tid
 result={'video':v,'frames':N,'retained_ids':tr_nb,'raw_ids':raw_ids,'saved_observations':saved_records,'discarded_observations':discarded_records,'tracks':items,'rejection_stats':ns['rejection_stats'],'continuous_full_video':True}
 atomic_json(ready,result);print('TRACKING_COMPLETE',v,tr_nb,flush=True)
if __name__=='__main__':main()
