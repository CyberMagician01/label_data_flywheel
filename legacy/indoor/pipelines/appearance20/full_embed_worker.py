import os,sys,time,json,sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
gpu=int(sys.argv[1]);os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);os.environ['TF_CPP_MIN_LOG_LEVEL']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from full_common import *
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior();sys.path.insert(0,str(R/'repo'))
from tracking import inception
from utils.func import crop
reuse=json.load(open(O/'reuse_index.json'));db=sqlite3.connect(O/'embedding_jobs.sqlite',timeout=60)
graph=tf.Graph()
with graph.as_default():
 with tf.device('/gpu:0'),tf.name_scope('tower_0') as scope:
  inp=tf.placeholder(tf.float32,[None,80,80,1]);vec,_=inception.inception_v3(inp,is_training=tf.constant(False),scope=scope,num_classes=64)
 saver=tf.train.Saver(tf.global_variables())
 cfg=tf.ConfigProto(allow_soft_placement=False,intra_op_parallelism_threads=4,inter_op_parallelism_threads=2);cfg.gpu_options.allow_growth=True
 sess=tf.Session(config=cfg);checkpoint=next(R.glob('data/**/model_005000.ckpt.index'));saver.restore(sess,str(checkpoint)[:-6])
def prepare_frame(v,fr):
 path=O/v/'embeddings'/f'{fr:06d}.npy'
 if path.exists():return fr,None,None
 ds=load_detections(v,fr);xy=np.array([[int((d['bbox_xyxy'][0]+d['bbox_xyxy'][2])/2),int((d['bbox_xyxy'][1]+d['bbox_xyxy'][3])/2)] for d in ds],dtype=np.int32).reshape(-1,2)
 old=reuse.get(f'{v}/{fr}')
 if old:
  arr=np.load(old)
  assert np.array_equal(arr[:,:2],xy),(v,fr,'cached input mismatch')
  os.link(old,path);return fr,None,None
 if not len(xy):
  with open(str(path)+'.tmp','wb') as f:np.save(f,np.empty((0,68),np.float32))
  os.replace(str(path)+'.tmp',path);return fr,None,None
 im=np.asarray(Image.open(IMAGES/v/f'frame_{fr:08d}.jpg').convert('L'),np.float32)
 im=(im-im.min())/max(float(im.max()-im.min()),1)*2-1
 crops=np.array([crop(im,int(x),int(y)) for x,y in xy],np.float32)[...,None]
 return fr,xy,crops
total=0;started=time.time()
with ThreadPoolExecutor(max_workers=4) as pool:
 while True:
  db.execute('begin immediate')
  job=db.execute("select video,start,end from jobs where status='pending' order by rowid limit 1").fetchone()
  if job is None:db.commit();break
  v,a,b=job;db.execute("update jobs set status='running',gpu=?,updated=? where video=? and start=?",(gpu,time.time(),v,a));db.commit()
  try:
   pending={i:pool.submit(prepare_frame,v,i) for i in range(a,min(a+8,b))}
   for k in range(a,b,4):
    pack=[]
    for fr in range(k,min(k+4,b)):
     pack.append(pending.pop(fr).result())
     nxt=fr+8
     if nxt<b:pending[nxt]=pool.submit(prepare_frame,v,nxt)
    active=[x for x in pack if x[1] is not None]
    if active:
     imgs=np.concatenate([x[2] for x in active]);vectors=np.concatenate([sess.run(vec,{inp:imgs[j:j+512]}) for j in range(0,len(imgs),512)])
     offset=0
     for fr,xy,_ in active:
      arr=np.column_stack([xy,np.zeros((len(xy),2)),vectors[offset:offset+len(xy)]]).astype(np.float32);offset+=len(xy)
      p=O/v/'embeddings'/f'{fr:06d}.npy'
      with open(str(p)+'.tmp','wb') as f:np.save(f,arr)
      os.replace(str(p)+'.tmp',p)
    total+=len(pack)
    if (k-a)%40==0:
     atomic_json(O/f'gpu{gpu}_progress.json',{'gpu':gpu,'video':v,'frame':k,'processed_this_run':total,'seconds':time.time()-started,'updated':time.time()})
   db.execute("update jobs set status='done',updated=? where video=? and start=?",(time.time(),v,a));db.commit()
   print('CHUNK_DONE',gpu,v,a,b,'frames',total,'seconds',round(time.time()-started,1),flush=True)
  except Exception:
   db.execute("update jobs set status='failed',updated=? where video=? and start=?",(time.time(),v,a));db.commit();raise
atomic_json(O/f'gpu{gpu}_complete.json',{'processed_frames':total,'seconds':time.time()-started})
print('GPU_COMPLETE',gpu,flush=True)
