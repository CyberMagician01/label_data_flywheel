import os,sys,time,json,subprocess,sqlite3,fcntl,hashlib
from pathlib import Path
from full_common import *
lock=open(O/'manager.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
python=str(R/'env/bin/python')
assert json.load(open(O/'regression_verified.json'))['all_observation_rows_and_ids_exact_match']
assert json.load(open(O/'export_regression_verified.json'))['exact_match_at_frame6400']
atomic_json(O/'implementation_hashes.json',{name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in ['full_common.py','full_embed_worker.py','full_track_worker.py','full_export_worker.py','track_gt_eval.py','sweep_overlap_thresholds.py']})
atomic_json(O/'manager_pid.json',{'pid':os.getpid(),'started':time.time()})
jobs={};attempts={};failed={};gpu_retries={0:0,1:0}
def launch(v,stage):
 script='full_track_worker.py' if stage=='tracking' else 'full_export_worker.py'
 log=open(O/v/(stage+'.log'),'a')
 p=subprocess.Popen([python,str(R/script),v],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 jobs[v]=(stage,p);attempts[(v,stage)]=attempts.get((v,stage),0)+1
 print('LAUNCH',v,stage,p.pid,flush=True)
def gpu_processes():
 found={}
 for path in Path('/proc').glob('[0-9]*'):
  try:
   args=(path/'cmdline').read_bytes().split(b'\0')
   if str(R/'full_embed_worker.py').encode() in args:
    pos=args.index(str(R/'full_embed_worker.py').encode());found[int(args[pos+1])]=int(path.name)
  except (OSError,ValueError,IndexError):pass
 return found
started=time.time()
while True:
 db=sqlite3.connect(O/'embedding_jobs.sqlite',timeout=60)
 gp=gpu_processes()
 for gpu in (0,1):
  if gpu in gp:continue
  pending=db.execute("select count(*) from jobs where status='pending' or (status in ('running','failed') and gpu=?)",(gpu,)).fetchone()[0]
  if pending:
   if gpu_retries[gpu]>=2:failed['gpu'+str(gpu)]='embedding worker repeatedly exited';continue
   db.execute("update jobs set status='pending',gpu=null where status in ('running','failed') and gpu=?",(gpu,));db.commit()
   log=open(O/f'gpu{gpu}.log','a');p=subprocess.Popen([python,str(R/'full_embed_worker.py'),str(gpu)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);gpu_retries[gpu]+=1
   print('RESTART_GPU',gpu,p.pid,flush=True)
 counts=[{'video':a,'status':b,'frames':c} for a,b,c in db.execute('select video,status,sum(end-start) from jobs group by video,status')];db.close()
 for v in VIDEOS:
  if v in failed:continue
  if v in jobs:
   stage,p=jobs[v]
   if p.poll() is None:continue
   del jobs[v]
   done=O/v/('tracking_complete.json' if stage=='tracking' else 'export_complete.json')
   if p.returncode!=0 or not done.exists():
    if attempts[(v,stage)]<3:launch(v,stage)
    else:failed[v]={'stage':stage,'exit_code':p.returncode,'log':str(O/v/(stage+'.log'))}
    continue
  if (O/v/'export_complete.json').exists():continue
  launch(v,'export' if (O/v/'tracking_complete.json').exists() else 'tracking')
 videos={}
 for v in VIDEOS:
  videos[v]={}
  for tag,name in [('tracking','tracking_progress.json'),('export','export_progress.json'),('result','export_complete.json')]:
   p=O/v/name
   if p.exists():videos[v][tag]=json.load(open(p))
 gpu_usage=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
 complete=all((O/v/'export_complete.json').exists() for v in VIDEOS)
 status={'started':started,'updated':time.time(),'manager_pid':os.getpid(),'completed':complete,'failed':failed,'running':{v:{'stage':stage,'pid':p.pid} for v,(stage,p) in jobs.items()},'embedding_jobs':counts,'gpu_usage':gpu_usage,'videos':videos}
 atomic_json(O/'status.json',status)
 if complete:
  results={v:json.load(open(O/v/'export_complete.json')) for v in VIDEOS}
  atomic_json(O/'completion.json',{'completed':True,'completed_at':time.time(),'total_frames':sum(x['frames'] for x in results.values()),'total_visible_boxes':sum(x['visible'] for x in results.values()),'results':results,'output_root':str(O/'final')})
  print('ALL_FOUR_VIDEOS_COMPLETE',flush=True);break
 if failed and not jobs:
  print('STOPPED_WITH_FAILURES',json.dumps(failed),flush=True);break
 time.sleep(30)
