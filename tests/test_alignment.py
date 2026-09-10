import copy
import json
from pathlib import Path

import pytest
from label_data_flywheel.io import normalize, labelme_frame, write_frames, read_json, iter_frames
from label_data_flywheel.behavior import analyze
from label_data_flywheel.review import apply_reviews
from label_data_flywheel.pipeline import run_round
from label_data_flywheel.pipeline import colony_knowledge_reviews
from label_data_flywheel.training_data import export_coco
from label_data_flywheel.feedback import evidence_quality, feedback_targets
from label_data_flywheel.knowledge import default_graph, update_graph, apply_behavior_knowledge
from label_data_flywheel.review_tasks import export_tasks, import_tasks
from label_data_flywheel.behavior_supervision import individual_records, colony_records


def frame(index=0, domain="IR_in", boxes=None):
    boxes = boxes if boxes is not None else [{"bbox_xyxy": [10+index, 10, 30+index, 30],
            "track_id": 7, "confidence": .9,
            "keypoints": {"head": [15+index, 15, .9], "abdomen_tip": [25+index, 25, .9]}}]
    return normalize({"frame": index, "image_size": [100, 100], "image_path": "/data/image.jpg",
                      "detections": boxes}, domain=domain, video="B" if domain == "IR_in" else "A",
                     group="01", split="train", source="human")


def test_shadow_same_group_never_overwrites_bee_or_supplies_pose_or_identity(tmp_path):
    source = {"imageWidth":100, "imageHeight":100, "shapes":[
        {"label":"bee", "shape_type":"rectangle", "group_id":7, "points":[[10,10],[30,30]]},
        {"label":"bee_shadow", "shape_type":"rectangle", "group_id":7, "points":[[40,40],[60,60]]},
        {"label":"head", "shape_type":"point", "group_id":7, "points":[[15,15]]},
        {"label":"tail", "shape_type":"point", "group_id":7, "points":[[25,25]]},
    ]}
    parsed = labelme_frame(source,0)
    assert len(parsed["detections"]) == 2
    bee, shadow = parsed["detections"]
    assert bee["track_id"] == 7 and bee["keypoints"]
    assert shadow["class_id"] == 1 and shadow["track_id"] is None and not shadow["keypoints"]
    f = frame(boxes=parsed["detections"])
    assert analyze([f])["group_windows"][0]["raw_count"] == 1
    coco = export_coco([f],tmp_path/"coco.json","/")
    assert len(coco["annotations"]) == 1
    assert not coco["images"][0]["annotation_complete"] and not coco["images"][0]["is_unlabeled"]
    source_path=tmp_path/"frame_00000000.json"
    source_path.write_text(json.dumps(source),encoding="utf-8")
    assert len(list(iter_frames(source_path,domain="RGB_out",video="A"))[0]["detections"]) == 2


def test_review_promotes_candidates_refreshes_masks_and_adds_missing_objects(tmp_path):
    f=frame(boxes=[])
    f["ignore_regions"]=[{"entity_id":"candidate","bbox_xyxy":[1,1,5,5],"keypoints":{},
                         "track_id":None,"label_status":"ignore","supervision_mask":{"pose":False,"tracking":False}}]
    decisions=[{"action":"correct","entity_id":"candidate","track_id":8,
                "keypoints":{"head":[2,2,1],"abdomen_tip":[4,4,1]},
                "behavior_labels":{"motion":"walking"}},
               {"action":"add","entity_id":"new","frame_id":f["sample_id"],"bbox_xyxy":[60,60,80,80]}]
    result=apply_reviews([f],decisions,"reviewer",tmp_path/"audit.jsonl")
    assert f["detections"] == []
    assert result[0]["ignore_regions"] == []
    d=result[0]["detections"][0]
    assert all(d["supervision_mask"].values())
    coco=export_coco(result,tmp_path/"coco.json","/")
    assert coco["annotations"][0]["pose_mask"] and coco["annotations"][0]["track_mask"]
    assert len(coco["annotations"]) == 2
    result=apply_reviews(result,[{"action":"reject","entity_id":"new"}],"reviewer",tmp_path/"audit.jsonl")
    coco=export_coco(result,tmp_path/"rejected.json","/")
    assert coco["images"][0]["verified_background_boxes"] == [[60,60,20,20]]


def test_round_rebuilds_fill_after_id_change_and_exports_real_quality(tmp_path):
    frames=[frame(0),frame(1,boxes=[]),frame(2)]
    frames[1]["detections"]=[{"entity_id":"old-fill","bbox_xyxy":[11,10,31,30],
                              "track_id":7,"origin":"linear_interpolation"}]
    source=tmp_path/"source.jsonl"; write_frames(source,frames)
    decisions=[{"action":"correct","entity_id":f["detections"][0]["entity_id"],"track_id":8}
               for f in (frames[0],frames[2])]
    reviews=tmp_path/"reviews.json"; reviews.write_text(json.dumps(decisions))
    out=tmp_path/"round"
    run_round({"inputs":[{"path":str(source)}],"review_decisions":str(reviews),
               "reviewer":"reviewer","export_training":True},out)
    result=list(iter_frames(out/"annotations.jsonl.gz"))
    assert result[1]["detections"][0]["track_id"] == 8
    assert result[1]["superseded_interpolations"][0]["track_id"] == 7
    quality={r["sample_id"]:r["quality"] for r in read_json(out/"quality.json") if r["layer"]=="L1"}
    coco=read_json(out/"training_snapshot/all.coco.json")
    assert all(a["hierarchy_quality"] == quality[a["source_entity_id"]] for a in coco["annotations"])
    assert all(not im["annotation_complete"] for im in coco["images"])


def test_q2_decreases_actual_training_weight_for_conflicting_observations(tmp_path):
    good=frame(0)
    bad=frame(1,boxes=[{"bbox_xyxy":[10,10,30,30],"track_id":7},
                      {"bbox_xyxy":[11,11,31,31],"track_id":8}])
    path=tmp_path/"source.jsonl"; write_frames(path,[good,bad])
    out=tmp_path/"round"
    run_round({"inputs":[{"path":str(path)}],"export_training":True},out)
    anns=read_json(out/"training_snapshot/train.coco.json")["annotations"]
    assert anns[0]["hierarchy_quality"] > anns[1]["hierarchy_quality"]
    assert anns[1]["hierarchy_quality"] < .7
    cube=read_json(out/"error_cube.json")
    assert all(r["video"]=="B" and r["scale_bin"]!="unknown" for r in cube)


def test_reference_pose_vote_and_missing_reference_enter_review():
    frames=[frame(i) for i in range(3)]
    refs=copy.deepcopy(frames)
    for f in frames:
        d=f["detections"][0]; d["keypoints"]["head"],d["keypoints"]["abdomen_tip"]=d["keypoints"]["abdomen_tip"],d["keypoints"]["head"]
    refs[0]["detections"].append({"entity_id":"missing","bbox_xyxy":[70,70,80,80],"keypoints":{},"track_id":9})
    quality=evidence_quality(frames,analyze(frames),refs)
    poses=[r for r in quality if r["layer"]=="L2"]
    assert all(r["track_pose_vote"]["action"]=="swap_candidate" for r in poses)
    assert any(r["root_cause"]=="reference_miss" for r in quality)


def test_knowledge_changes_evidence_and_error_feedback_changes_next_targets():
    graph=default_graph()
    event={"knowledge_source":"visual_kinematics","domain":"IR_in","video":"B","evidence_confidence":.8}
    old=apply_behavior_knowledge({"events":[event],"raw_count":3},graph)
    decision={"knowledge_id":"visual_kinematics","review_id":"1","reviewer":"r","action":"confirm","domain":"IR_in","video":"B"}
    graph=update_graph(graph,[decision,decision])
    new=apply_behavior_knowledge({"events":[event],"raw_count":3},graph)
    assert new["events"][0]["evidence_confidence"] > old["events"][0]["evidence_confidence"]
    assert new["raw_count"] == old["raw_count"]
    assert graph["nodes"][0]["alpha"]==2
    records=[{"video":"A"},{"video":"B"}]
    target=feedback_targets(records,[{"video":"A","mean_error":1},{"video":"B","mean_error":0}],{"axes":["video"]})
    assert target["video"]["A"] > target["video"]["B"]


def test_labelme_task_roundtrip_changes_pose_and_id_without_confirming_everything(tmp_path):
    f=frame(); d=f["detections"][0]
    export_tasks([f],[{"sample_id":d["entity_id"],"frame_id":f["sample_id"]}],tmp_path/"tasks")
    assert import_tasks(tmp_path/"tasks") == []
    path=next((tmp_path/"tasks/editable").glob("*.json")); data=read_json(path)
    for s in data["shapes"]: s["group_id"]=42
    data["shapes"][1]["points"]=[[16,16]]
    path.write_text(json.dumps(data))
    decisions=import_tasks(tmp_path/"tasks")
    assert len(decisions)==1 and decisions[0]["action"]=="correct" and decisions[0]["track_id"]==42
    assert decisions[0]["keypoints"]["head"][:2]==[16,16]


def test_behavior_confirmation_produces_trainable_features_without_group_broadcast():
    fs=[frame(0),frame(1)]
    fs[-1]["detections"][0].update(label_status="human_confirmed",reviewer="r",behavior_labels={"motion":"walking"})
    records=individual_records(fs,analyze(fs))
    assert len(records)==1 and records[0]["speed_bl_per_source_frame"] > 0
    report={"scopes":[{"domain":"IR_in","video":"B","group":"01","windows":[{"index":0,"mean_observed_count":2}]}],
            "confirmed_group_windows":[{"domain":"IR_in","video":"B","group":"01","window_index":0,
            "source_frame_range":[0,1],"event_id":"group","human_label":"聚集","reviewer":"r","status":"human_confirmed"}]}
    groups=colony_records(fs,report)
    assert groups[0]["labels"]=={"colony":"聚集"} and groups[0]["split"]=="train"
    assert len(individual_records(fs,analyze(fs)))==1


def test_next_round_reads_sampling_policy_knowledge_and_prompt_memory(tmp_path):
    fs=[frame(0),frame(1)]
    source=tmp_path/"source.jsonl"; write_frames(source,fs)
    d=fs[0]["detections"][0]
    decisions=tmp_path/"review.json"
    decisions.write_text(json.dumps([{"action":"confirm","entity_id":d["entity_id"],
                                     "knowledge_id":"visual_kinematics"}]))
    first=tmp_path/"r1"; second=tmp_path/"r2"
    run_round({"inputs":[{"path":str(source)}],"review_decisions":str(decisions),"reviewer":"r"},first)
    run_round({"inputs":[{"path":str(first/"annotations.jsonl.gz")}],
               "previous_round_policy":str(first/"next_round_policy.json")},second)
    assert read_json(second/"knowledge_graph.json")==read_json(first/"knowledge_graph.json")
    assert read_json(second/"prompt_memory.json")["examples"][0]["entity_id"]==d["entity_id"]
    assert read_json(second/"next_round_policy.json")["sampling_targets"]
    event={"domain":"IR_in","video":"B","group":"01","source_frame_range":[0,1]}
    audit=[{"before":event,"decision_id":"g1","reviewer":"r","decision":{"action":"confirm"}}]
    graph=update_graph(default_graph(),colony_knowledge_reviews(fs,audit))
    assert next(n for n in graph["nodes"] if n["id"]=="colony_temporal_change")["alpha"]==2
    fs[1]["split"]="test"
    assert colony_knowledge_reviews(fs,audit)==[]


def test_rejected_box_and_shadow_do_not_hide_valid_interpolation():
    from label_data_flywheel.postprocess import suppress_interpolation
    f=frame(boxes=[{"bbox_xyxy":[10,10,30,30],"label_status":"invalid"},
                  {"bbox_xyxy":[10,10,30,30],"class_id":1},
                  {"bbox_xyxy":[10,10,30,30],"track_id":5,"origin":"linear_interpolation"}])
    result=suppress_interpolation(f)
    assert result["counts"]["interpolated_kept"]==1


def test_internal_coco_does_not_relabel_shadow_as_bee(tmp_path):
    from label_data_flywheel.annotation_package import export_annotations
    source=tmp_path/"source.jsonl"
    f=frame(domain="RGB_out",boxes=[{"bbox_xyxy":[10,10,30,30],"track_id":1},
                                  {"bbox_xyxy":[40,40,60,60],"class_id":1,"track_id":1}])
    write_frames(source,[f])
    out=tmp_path/"export"
    result=export_annotations({"version":"test","inputs":[{"path":str(source)}]},out)
    assert result["counts"]["audit_only"]==1 and result["counts"]["confirmed"]==1
    coco=read_json(next(out.rglob("confirmed.coco.json")))
    assert len(coco["annotations"])==1 and coco["annotations"][0]["bbox"]==[10,10,20,20]


def test_reconfirmed_interpolation_becomes_direct_human_evidence(tmp_path):
    f=frame(boxes=[{"bbox_xyxy":[10,10,30,30],"track_id":1,"origin":"linear_interpolation"}])
    eid=f["detections"][0]["entity_id"]
    rejected=apply_reviews([f],[{"entity_id":eid,"action":"reject"}],"r",tmp_path/"a.jsonl")
    confirmed=apply_reviews(rejected,[{"entity_id":eid,"action":"confirm"}],"r",tmp_path/"a.jsonl")
    assert confirmed[0]["verified_background_regions"]==[]
    assert confirmed[0]["detections"][0]["origin"]=="human_review"
    assert len(analyze(confirmed)["observations"])==1
