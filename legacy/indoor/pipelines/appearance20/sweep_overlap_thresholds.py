import json,sys,csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
r=Path('/root/bee_tracking_pilot_20260907');o=r/'gt_eval_01_03_20260907';out=o/'overlap_threshold_sweep'
def prepare():
 out.mkdir(exist_ok=True);seqs=[s['sequence'] for s in json.load(open(o/'protocol.json'))['sequences']]
 prepared=[]
 for seq in seqs:
  base=o/seq;gt=json.load(open(base/'gt.json'));indices={x['frame'] for x in gt};frames=json.load(open(base/'outputs.json'));start=frames[0]['frame'];quality={}
  for p in sorted((base/'tracking/tracks').glob('[0-9]*.txt')):
   tid=int(p.stem)+1;rows=np.loadtxt(p,delimiter=',',ndmin=2).astype(int)
   for a,b in zip(rows[:-1],rows[1:]):
    gap=int(b[0]-a[0])
    if not 1<gap<=90:continue
    displacement=float(np.linalg.norm(b[1:3]-a[1:3]))
    for fr in range(int(a[0])+1,int(b[0])):
     if fr+start in indices:quality[(fr+start,tid)]=(gap,-len(rows),displacement,tid)
  subset=[]
  for f in frames:
   if f['frame'] not in indices:continue
   f['priority_order']=sorted(range(len(f['interpolated'])),key=lambda i:quality[(f['frame'],int(f['interpolated'][i][0]))])
   subset.append(f)
  prepared.append({'sequence':seq,'gt':gt,'frames':subset})
 json.dump(prepared,open(out/'inputs.json','w'))
 return prepared
def matrices(a,b):
 a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
 inter=np.prod(np.maximum(np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2]),0),axis=2)
 aa=np.prod(a[:,2:]-a[:,:2],axis=1);bb=np.prod(b[:,2:]-b[:,:2],axis=1)
 return inter/np.maximum(np.minimum(aa[:,None],bb[None,:]),1e-9),inter
def suppress(f,threshold):
 ob=f['observed'];ins=f['interpolated'];ib=[x[1:] for x in ins]
 mo,ao=matrices(ib,[x[1:] for x in ob]);mi,ai=matrices(ib,ib)
 co=(mo>=threshold)&(ao>=16);ci=(mi>=threshold)&(ai>=16)
 hidden={};kept=[]
 for i,row in enumerate(ins):
  js=np.flatnonzero(co[i])
  if len(js):hidden[str(int(row[0]))]={'reason':'observed','kept_id':int(ob[int(js[np.argmax(mo[i,js])])][0])}
 for i in f['priority_order']:
  row=ins[i]
  if str(int(row[0])) in hidden:continue
  clashes=[j for j in kept if ci[i,j]]
  if clashes:hidden[str(int(row[0]))]={'reason':'interpolated','kept_id':int(ins[clashes[0]][0])}
  else:kept.append(i)
 assert not co[kept].any()
 assert not np.triu(ci[np.ix_(kept,kept)],1).any()
 return hidden
def worker(threshold):
 src=(r/'score_gt_eval.py').read_text();ns={};exec(src[:src.index('# Sanity:')],ns)
 metrics=ns['metrics'];make=ns['data_for'];simple=ns['simple']
 allres={};rows=[];allhidden={}
 for item in json.load(open(out/'inputs.json')):
  seq=item['sequence'];gt=item['gt'];frames={f['frame']:f for f in item['frames']};suppressed=0;added=0
  for fr,f in frames.items():
   f['hidden']=suppress(f,threshold);suppressed+=len(f['hidden']);added+=len(f['interpolated'])
  allhidden[seq]={str(fr):f['hidden'] for fr,f in frames.items()}
  dat=make(gt,frames,'hidden70');res={m.get_name():m.eval_sequence(dat) for m in metrics};allres[seq]=res
  rows.append({'scope':seq,'threshold':threshold,**simple(res),'gt_boxes':dat['num_gt_dets'],'pred_boxes':dat['num_tracker_dets'],'added_before':added,'added_kept':added-suppressed,'added_hidden':suppressed})
 seqs=list(allres)
 for scope,ss in [('ALL',seqs),('01',[s for s in seqs if s.startswith('01_')]),('03',[s for s in seqs if s.startswith('03_')])]:
  res={m.get_name():m.combine_sequences({s:allres[s][m.get_name()] for s in ss}) for m in metrics}
  selected=[x for x in rows if x['scope'] in ss]
  rows.append({'scope':scope,'threshold':threshold,**simple(res),**{k:sum(x[k] for x in selected) for k in ['gt_boxes','pred_boxes','added_before','added_kept','added_hidden']}})
 for x in rows:assert abs(x['MOTA']-(1-(x['CLR_FP']+x['CLR_FN']+x['IDSW'])/x['gt_boxes']))<1e-12
 tag=f"{int(threshold*100):02d}"
 json.dump({'rows':rows},open(out/f'results_{tag}.json','w'))
 json.dump(allhidden,open(out/f'hidden_{tag}.json','w'))
 print(json.dumps(next(x for x in rows if x['scope']=='ALL')),flush=True)
 return rows
if __name__=='__main__':
 prepare();thresholds=[.05,.1,.15,.2,.3,.5]
 json.dump({'thresholds':thresholds,'overlap_definition':'intersection / min(box_area)','minimum_intersection_pixels':16,'only_threshold_changes':True,'evaluation_is_parameter_tuning_not_independent_test':True},open(out/'config.json','w'),indent=2)
 with ProcessPoolExecutor(max_workers=6) as pool:rows=[x for batch in pool.map(worker,thresholds) for x in batch]
 previous=json.load(open(o/'dedup_v5/results.json'))['rows']
 for a in [x for x in rows if x['threshold']==.1]:
  b=next(x for x in previous if x['mode']=='dedup_v5' and x['scope']==a['scope'])
  for k in ['MOTA','IDF1','CLR_FP','CLR_FN','IDSW']:assert abs(a[k]-b[k])<1e-12,(a['scope'],k)
 json.dump({'rows':rows,'baseline':json.load(open(o/'results.json'))['rows']},open(out/'results.json','w'),indent=2)
 with open(out/'results.csv','w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('SWEEP_COMPLETE',flush=True)
