"""确认全量加速的候选集合与原实现一致，并核对空帧及压缩路径。"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from full_id_storage import JsonPath
from dedup_overlap_tracks import geometry
from keep_interpolation_and_ownership import body_evidence
from reassociate_final_bees import metrics,describe

r=np.random.default_rng(20260906);c=r.uniform(0,300,(5000,2));t=r.integers(0,9013,5000);xy=cKDTree(c);xyz=cKDTree(np.column_stack([c,t*.3]))
for i in range(200):
    q=c[i];f=int(t[i]);old={j for j in xy.query_ball_point(q,45) if 0<t[j]-f<=150}
    new={j for j in xyz.query_ball_point(np.r_[q,f*.3],45+1e-9,p=np.inf) if 0<t[j]-f<=150 and np.linalg.norm(c[j]-q)<=45}
    assert old==new
assert geometry([])[0].shape==(0,0)
assert body_evidence([],{'indexes':[],'points':[]})[0]=={}
assert describe(np.zeros((10,10)),[]).shape==(0,64)
d={'track_id':1,'bbox_xyxy':[0,0,10,10]};m=metrics({0:{'detections':[d]},1:{'detections':[]},2:{'detections':[d]}})
assert m['internal_gaps']==1 and m['frames']==3
root=JsonPath('/root/autodl-tmp/indoor_ids_full_latest_20260906')
counts={v:len(list((root/'01_input/frames'/v).glob('*.json'))) for v in ['B-5-1','B-5-2','B-5-3','B-5-4']}
assert counts=={'B-5-1':8996,'B-5-2':8996,'B-5-3':9013,'B-5-4':9005}
report=dict(candidate_equivalence_checks=200,empty_frames_supported=True,compressed_input_frames=counts,passed=True)
(Path(root)/'runtime_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
