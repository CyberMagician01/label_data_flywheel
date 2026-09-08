import os,sys,json,time,gzip,argparse,copy
from pathlib import Path
from functools import lru_cache
import numpy as np
from full_common import *
from sweep_overlap_thresholds import suppress
def main():
 ap=argparse.ArgumentParser();ap.add_argument('video');ap.add_argument('--regression');args=ap.parse_args();v=args.video
 if args.regression:
  base=O/'regression'/args.regression;meta=json.load(open(R/'gt_eval_01_03_20260907'/args.regression/'inputs.json'))
  N=len(meta['records']);offset=meta['records'][0]['source_frame'];dest=base/'final'
  def load(fr):return [{'bbox_xyxy':d['box'],'keypoints':d['keypoints']} for d in meta['records'][fr]['detections']]
 else:
  base=O/v;N=json.load(open(O/'config.json'))['videos'][v];offset=0;dest=O/'final'
  def load(fr):return load_detections(v,fr)
 frameout=dest/'frames'/v;frameout.mkdir(parents=True,exist_ok=True);motout=dest/'mot';motout.mkdir(exist_ok=True)
 ready=base/'export_complete.json'
 if ready.exists():print('EXPORT_ALREADY_COMPLETE',v,flush=True);return
 result=json.load(open(base/'tracking_complete.json'))
 observed=[[] for _ in range(N)];fills=[[] for _ in range(N)]
 for item in result['tracks']:
  rows=np.load(base/'track_fragments'/f"{item['fragment']:06d}.npy");tid=item['track_id']
  for row in rows:observed[int(row[0])].append((tid,int(row[1]),int(row[2])))
  for a,b in zip(rows[:-1],rows[1:]):
   left,right=int(a[0]),int(b[0]);gap=right-left
   if not 1<gap<=90:continue
   spec=(tid,left,right,int(a[1]),int(a[2]),int(b[1]),int(b[2]),len(rows),float(np.linalg.norm(b[1:3]-a[1:3])))
   for fr in range(left+1,right):fills[fr].append(spec)
 @lru_cache(maxsize=256)
 def lookup(fr):
  return {(int((d['bbox_xyxy'][0]+d['bbox_xyxy'][2])/2),int((d['bbox_xyxy'][1]+d['bbox_xyxy'][3])/2)):d for d in load(fr)}
 totals={'observed':0,'interpolated_before':0,'interpolated_kept':0,'interpolated_hidden':0,'visible':0,'frames':N};seenids=set();start=time.time()
 motfile=motout/(v+'.txt');tmpmot=Path(str(motfile)+'.tmp')
 with open(tmpmot,'w') as mot:
  for fr in range(N):
   sf=fr+offset;p=frameout/f'frame_{sf:08d}.json.gz'
   if p.exists():
    with gzip.open(p,'rt') as fh:data=json.load(fh)
   else:
    obs=[];ins=[];priority=[]
    current=lookup(fr)
    for tid,x,y in observed[fr]:
     d=copy.deepcopy(current[(x,y)]);d['source_track_id']=d.get('track_id');d['track_id']=tid;d['origin']='observed';obs.append(d)
    for tid,left,right,x1,y1,x2,y2,length,disp in fills[fr]:
     a=lookup(left)[(x1,y1)];b=lookup(right)[(x2,y2)];alpha=(fr-left)/(right-left)
     bbox=((1-alpha)*np.array(a['bbox_xyxy'])+alpha*np.array(b['bbox_xyxy'])).tolist()
     ins.append({'track_id':tid,'bbox_xyxy':bbox,'origin':'linear_interpolation','interpolation':{'left_source_frame':left+offset,'right_source_frame':right+offset,'alpha':alpha},'keypoints':{}})
     priority.append((right-left,-length,disp,tid))
    f={'frame':sf,'observed':[[d['track_id'],*d['bbox_xyxy']] for d in obs],'interpolated':[[d['track_id'],*d['bbox_xyxy']] for d in ins],'priority_order':sorted(range(len(ins)),key=lambda i:priority[i])}
    hidden=suppress(f,.2);kept=[];suppressed=[]
    for d in ins:
     evidence=hidden.get(str(d['track_id']))
     if evidence is None:kept.append(d)
     else:d['suppression']=evidence;suppressed.append(d)
    data={'video':v,'frame':sf,'fps':30,'image_size':[1920,1080],'coordinate_space':'full_frame','overlap_threshold':.2,'overlap_definition':'intersection/minimum_box_area','detections':obs+kept,'temporarily_hidden_detections':suppressed,'counts':{'observed':len(obs),'interpolated_before':len(ins),'interpolated_kept':len(kept),'interpolated_hidden':len(suppressed),'visible':len(obs)+len(kept)}}
    with gzip.open(str(p)+'.tmp','wt',encoding='utf-8',compresslevel=1) as fh:json.dump(data,fh,ensure_ascii=False,separators=(',',':'))
    os.replace(str(p)+'.tmp',p)
   counts=data['counts'];ds=data['detections'];hid=data['temporarily_hidden_detections']
   assert len(ds)==counts['visible'] and counts['interpolated_before']==counts['interpolated_kept']+counts['interpolated_hidden']
   assert counts['observed']==len(observed[fr])
   assert len({d['track_id'] for d in ds+hid})==len(ds)+len(hid)
   assert all(d['origin']=='linear_interpolation' for d in hid)
   for key in counts:totals[key]+=counts[key]
   for d in ds:
    tid=d['track_id'];seenids.add(tid);a,b,c,e=d['bbox_xyxy'];mot.write(f'{sf+1},{tid},{a:.6f},{b:.6f},{c-a:.6f},{e-b:.6f},1,-1,-1,-1\n')
   if fr%300==0 or fr==N-1:
    atomic_json(base/'export_progress.json',{'video':v,'frame':sf,'frames_done':fr+1,'total_frames':N,'seconds':time.time()-start,'updated':time.time()})
    print('EXPORT',v,fr+1,'/',N,flush=True)
 os.replace(tmpmot,motfile)
 assert totals['observed']==result['saved_observations']
 assert seenids==set(range(1,result['retained_ids']+1))
 assert len(list(frameout.glob('frame_*.json.gz')))==N
 totals.update({'video':v,'unique_visible_ids':len(seenids),'maximum_id':max(seenids,default=0),'original_observations_hidden':0,'id_history_preserved':True,'full_frame_30fps':True,'threshold':.2,'all_checks_passed':True,'seconds':time.time()-start,'frames_path':str(frameout),'mot_path':str(motfile)})
 atomic_json(ready,totals);print('EXPORT_COMPLETE',json.dumps(totals),flush=True)
if __name__=='__main__':main()
