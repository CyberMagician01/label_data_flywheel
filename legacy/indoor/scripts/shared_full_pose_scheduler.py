"""GPU按独立帧段分工；ID关联、去重和插值仍按整段视频执行。"""
import json,os,subprocess,time,shutil,sys
from pathlib import Path

ROOT=Path('/root/autodl-tmp/indoor_ids_full_latest_20260906')
VIDEOS=['B-5-3','B-5-1','B-5-4','B-5-2']
EXPECTED={'B-5-1':8996,'B-5-2':8996,'B-5-3':9013,'B-5-4':9005}
STAGES=['prepare','pose_pre','prefilter','track','pose_owner','owner','pose_dense','dense','pose_repair','repair','audit_export']
CACHE={'pose_pre':'01_input','pose_owner':'04_ownership','pose_dense':'05_dense','pose_repair':'06_repair'}
GPU=set(CACHE);CHUNK=300
SCRIPT=Path('/root/autodl-tmp/run_full_id_stage.py')
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')

def completed(v,s):return (ROOT/'done'/f'{v}_{s}.json').exists()

def cache_frames(v,s):
    folder=ROOT/CACHE[s]/'pose_cache'/v
    return {int(p.name.split('_')[1].split('.')[0]) for pattern in ['frame_*.json','frame_*.json.gz'] for p in folder.glob(pattern)}

def next_chunk(total,finished,occupied,size=CHUNK):
    for start in range(total):
        if start in finished or any(a<=start<b for a,b in occupied):continue
        end=min(start+size,total)
        for a,b in occupied:
            if start<a<end:end=a
        return start,end
    return None

class AdoptedCPU:
    """替换调度器时保留CPU进程，完成标记负责确认成功。"""
    def __init__(self,pid,video,stage):self.pid=pid;self.video=video;self.stage=stage
    def poll(self):
        p=Path(f'/proc/{self.pid}/cmdline')
        if p.exists() and p.read_bytes():return None
        return 0 if completed(self.video,self.stage) else 1
    def terminate(self):
        if self.poll() is None:os.kill(self.pid,15)

def main():
    ROOT.mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
    statepath=ROOT/'status.json';previous=json.loads(statepath.read_text()) if statepath.exists() else {}
    started=previous.get('started',time.time());running={};failures={};attempts={};gpu_progress={}
    for key,r in previous.get('running',{}).items():
        v=r.get('video',key);p=Path(f"/proc/{r['pid']}/cmdline")
        if r['stage'] not in GPU and p.exists() and b'run_full_id_stage.py' in p.read_bytes():
            path=ROOT/'logs'/f"{v}_{r['stage']}.log"
            running[v]=dict(video=v,stage=r['stage'],p=AdoptedCPU(r['pid'],v,r['stage']),device=None,started=r['started'],path=path,log=None)
    manifestpath=ROOT/'run_manifest.json';manifest=json.loads(manifestpath.read_text())
    manifest.update(gpu_scheduling='independent_frame_chunks_shared_by_two_gpus',pose_chunk_frames=CHUNK)
    manifestpath.write_text(json.dumps(manifest,indent=2))
    def write_state():
        record=dict(started=started,updated=time.time(),manager_pid=os.getpid(),completed=all(completed(v,STAGES[-1]) for v in VIDEOS),failed=failures,free_disk_gb=round(shutil.disk_usage(ROOT).free/1024**3,2),gpu_progress=gpu_progress,running={key:dict(video=r['video'],stage=r['stage'],pid=r['p'].pid,device=r['device'],started=r['started'],frame_start=r.get('frame_start'),frame_end=r.get('frame_end')) for key,r in running.items()},videos={v:dict(completed_stages=[s for s in STAGES if completed(v,s)],completed=completed(v,STAGES[-1])) for v in VIDEOS})
        tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));os.replace(tmp,statepath)
    def launch(v,s,device=None,bounds=None):
        key=v if device is None else f'{v}@gpu{device}'
        path=ROOT/'logs'/f'{v}_{s}.log';log=path.open('a')
        args=[sys.executable,'-u',str(SCRIPT),'--video',v,'--stage',s]
        if device is not None:args+=['--device',str(device),'--frame-start',str(bounds[0]),'--frame-end',str(bounds[1])]
        p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        running[key]=dict(video=v,p=p,stage=s,device=device,log=log,path=path,started=time.time())
        if bounds:running[key].update(frame_start=bounds[0],frame_end=bounds[1])
        print(json.dumps(dict(start=v,stage=s,device=device,pid=p.pid,bounds=bounds)),flush=True)
    try:
        while True:
            for key,r in list(running.items()):
                code=r['p'].poll()
                if code is None:continue
                if r['log']:r['log'].close()
                del running[key]
                if code!=0:
                    attempt=(r['video'],r['stage'],r.get('frame_start'));attempts[attempt]=attempts.get(attempt,0)+1
                    if attempts[attempt]>=2:failures[r['video']]=dict(stage=r['stage'],exit_code=code,log=str(r['path']))
                    print(json.dumps(dict(video=r['video'],stage=r['stage'],exit_code=code,attempt=attempts[attempt])),flush=True)
            if shutil.disk_usage(ROOT).free<1024**3:
                failures['storage']=dict(reason='less_than_1GiB_free');break
            ready=[];gpu_progress={};finished_by_video={}
            for v in VIDEOS:
                if v in failures:continue
                for i,s in enumerate(STAGES):
                    if completed(v,s):continue
                    if s in GPU:
                        finished=cache_frames(v,s);finished_by_video[v]=finished
                        gpu_progress[v]=dict(stage=s,frames=len(finished),total=EXPECTED[v])
                        if finished==set(range(EXPECTED[v])) and not any(r['video']==v for r in running.values()):
                            marker=ROOT/'done'/f'{v}_{s}.json';marker.parent.mkdir(exist_ok=True)
                            marker.write_text(json.dumps(dict(completed=True,time=time.time(),frames=len(finished),shared_gpus=True)))
                            continue
                    ready.append((i,v,s));break
            cpu=sum(r['device'] is None for r in running.values())
            for i,v,s in sorted(ready,reverse=True):
                if s in GPU or v in running or cpu>=2:continue
                launch(v,s);cpu+=1
            used={r['device'] for r in running.values() if r['device'] is not None}
            for device in [g for g in [0,1] if g not in used]:
                for i,v,s in sorted(ready,reverse=True):
                    if s not in GPU:continue
                    occupied=[(r['frame_start'],r['frame_end']) for r in running.values() if r['video']==v and r['device'] is not None]
                    bounds=next_chunk(EXPECTED[v],finished_by_video[v],occupied)
                    if bounds is not None:launch(v,s,device,bounds);break
            write_state()
            if not running and all(v in failures or completed(v,STAGES[-1]) for v in VIDEOS):break
            time.sleep(3)
    finally:
        for r in running.values():
            if r['p'].poll() is None:r['p'].terminate()
            if r['log']:r['log'].close()
        write_state()

if __name__=='__main__':main()
