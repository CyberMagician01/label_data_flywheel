import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


def test_outdoor_forward_reverse_stitch_and_id_only_release(tmp_path):
    root=Path(__file__).resolve().parents[1]
    extracted=tmp_path/"extracted"
    labels=extracted/"annotations/outdoor/A-5-1/frames"
    images=extracted/"frames/A-5-1"
    labels.mkdir(parents=True); images.mkdir(parents=True)
    rng=np.random.default_rng(34)
    background=rng.integers(0,30,(128,128),dtype=np.uint8)
    originals=[]
    for i in range(6):
        image=background.copy(); image[25:45,20+i:40+i]=180
        Image.fromarray(image).save(images/f"frame_{i:08d}.jpg")
        row={"frame":i,"width":128,"height":128,"detections":[
            {"class_id":0,"bbox_xyxy":[20+i,25,40+i,45],"confidence":.95,
             "keypoints":{"head":[25+i,30,.9],"abdomen_tip":[35+i,40,.9]}},
            {"class_id":1,"bbox_xyxy":[80,80,100,100],"confidence":.8}]}
        originals.append(row)
        (labels/f"frame_{i:08d}.json").write_text(json.dumps(row))
    cfg=tmp_path/"config.json"
    cfg.write_text(json.dumps({"workspace_root":str(tmp_path/"work"),"extracted_root":str(extracted),
                              "sequences":{"A-5-1":{"frames":6}}}))
    p=subprocess.run([sys.executable,str(root/"legacy/outdoor/final_id_only/run_final.py"),
                      "--config",str(cfg)],capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
    output=tmp_path/"work/outputs/optimized_v1/11_id_only_final"
    ids=[]
    for i, original in enumerate(originals):
        row=json.loads((output/f"annotations/outdoor/A-5-1/frame_{i:08d}.json").read_text())
        ids.append(row["detections"][0].pop("track_id"))
        assert row["detections"]==original["detections"]
    assert len(set(ids))==1
    release=json.loads((output/"release_manifest.json").read_text())
    assert release["all_non_id_fields_preserved"] and release["counts"]["outdoor_frames"]==6
