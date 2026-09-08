"""有出处、有输入门槛的养蜂复核建议；输出从不作为疾病或行为真值。"""

from importlib.resources import files
import json
import math
import copy


def literature():
    return json.loads(
        files("label_data_flywheel")
        .joinpath("assets/bee_knowledge.json")
        .read_text(encoding="utf-8")
    )


def interpret_colony(report, context_records=None):
    """外部证据必须有来源、作用域和源时间窗，不跨蜂箱/视频拼接。"""
    records = context_records or []
    conclusions, queue = [], []
    for scope in report["scopes"]:
        key = (scope["domain"], scope["video"], scope["group"])
        changes = scope["temporal"]["change_candidates"]
        for window in scope["windows"]:
            wi = window["index"]
            candidates = [c for c in changes if c["window_index"] == wi]
            context = {}
            sources = []
            for row in records:
                if (row.get("domain"), row.get("video"), row.get("group")) != key:
                    continue
                if row.get("window_index") == wi and row.get("source"):
                    context.update(
                        {
                            k: v
                            for k, v in row.get("measurements", {}).items()
                            if v is not None
                        }
                    )
                    sources.append(row["source"])
            good = (
                window["coverage"] >= report["config"]["min_window_coverage"]
                and window["tracked_fraction"] >= 0.8
            )
            if any(
                c["metric"] in ("active_fraction", "median_speed_bl_proxy_s")
                for c in candidates
            ):
                good = good and window["motion_observable_fraction"] >= 0.5
            visual = bool(candidates)

            def number(name):
                value = context.get(name)
                return (
                    value
                    if type(value) in (int, float) and math.isfinite(value)
                    else None
                )

            temperature = number("temperature_c")
            baseline = number("temperature_baseline_c")
            hot = (
                temperature is not None
                and baseline is not None
                and temperature > baseline
            )
            rain, wind = number("rain_mm"), number("wind_m_s")
            weight = number("hive_weight_delta_kg")
            dead = number("dead_bee_count")
            scenarios = [
                (
                    "weather",
                    "天气影响与热环境复核",
                    ["temperature_c", "temperature_baseline_c"],
                    hot
                    or (rain is not None and rain > 0)
                    or (wind is not None and wind > 0),
                    ["heat_response", "multimodal_time"],
                    "核对同时间段温度、风雨、遮挡与蜜粉源条件；若热环境升高，现场检查水源、遮阳与通风状态。",
                    "天气、采集任务、摄像机变化都可能影响视觉活动。",
                ),
                (
                    "swarming",
                    "分蜂相关线索复核",
                    ["queen_cell_verified", "vibration_swarm_candidate"],
                    context.get("queen_cell_verified") is True
                    or context.get("vibration_swarm_candidate") is True
                    or (weight is not None and weight < 0),
                    ["swarming", "entrance_activity"],
                    "优先人工核查王台和实际蜂群状态，并调阅前后较长时间的巢口、振动与称重记录。",
                    "群体转移、正常出勤、称重操作或环境变化均可能出现类似线索。",
                ),
                (
                    "pesticide",
                    "疑似暴露事件复核",
                    ["pesticide_exposure_record", "dead_bee_count"],
                    context.get("pesticide_exposure_record") is True
                    and dead is not None
                    and dead > 0,
                    ["pesticide", "multimodal_time"],
                    "核查施药时间地点、天气和死蜂记录，保留对应样本与视频供专业人员排查原因。",
                    "视频不能确认农药危害；病虫害、温度和采集条件也是候选原因。",
                ),
            ]
            for (
                sid,
                title,
                required,
                corroboration,
                refs,
                action,
                alternatives,
            ) in scenarios:
                status = "insufficient_evidence"
                if good and visual and corroboration:
                    status = "review_candidate"
                elif good and sources and all(k in context for k in required):
                    status = "no_joint_signal_in_this_window"
                row = {
                    "domain": key[0],
                    "video": key[1],
                    "group": key[2],
                    "window_index": wi,
                    "scenario": sid,
                    "title": title,
                    "status": status,
                    "visual_candidates": candidates,
                    "external_measurements": context,
                    "external_sources": sources,
                    "missing_reference_fields": [
                        k for k in required if k not in context
                    ],
                    "data_quality_gate_passed": good,
                    "references": refs,
                    "review_action": action,
                    "alternative_explanations": alternatives,
                    "diagnosis": None,
                    "meaning": "候选仅用于人工复核排序；无联合信号不等于排除该蜂学问题",
                }
                conclusions.append(row)
            if candidates:
                queue.append(
                    {
                        "sample_id": f"{key[0]}/{key[2]}/{key[1]}/{window['first_frame']:08d}",
                        "event_id": f"{key[0]}/{key[2]}/{key[1]}/colony/{window['first_frame']}-{window['last_frame']}",
                        "layer": "L4",
                        "reason": "colony_temporal_change"
                        if good
                        else "colony_observability_check",
                        "domain": key[0],
                        "video": key[1],
                        "group": key[2],
                        "source_frame_range": [
                            window["first_frame"],
                            window["last_frame"],
                        ],
                        "window_index": wi,
                        "status": "unconfirmed",
                        "events": candidates,
                        "review_instruction": "先检查漏检、重复、ID跳变与遮挡，再判读群体行为；不得直接回流为行为真值。",
                    }
                )
    return {
        "literature": literature(),
        "interpretations": conclusions,
        "review_queue": queue,
        "validation_status": "机制已实现；真实蜂学事件识别性能须用具名复核与独立事件真值评估",
    }


def review_colony(report, decisions, reviewer):
    """群体事件独立复核，不把一次群体判读广播成全部个体的行为真值。"""
    if not reviewer.strip():
        raise ValueError("群体复核需要具名 reviewer")
    result = copy.deepcopy(report)
    audit = result.setdefault("colony_review_audit", [])
    done = {row["decision_id"]: row for row in audit}
    queue = result["apiculture"]["review_queue"]
    lookup = {row["event_id"]: row for row in queue}
    for decision in decisions:
        did = decision["decision_id"]
        if did in done:
            if done[did]["decision"] != decision or done[did]["reviewer"] != reviewer:
                raise ValueError("decision_id 已用于另一项复核")
            continue
        target = lookup[decision["event_id"]]
        action = decision["action"]
        if action not in ("confirm", "correct", "reject"):
            raise ValueError("操作须为 confirm/correct/reject")
        if action != "reject" and not decision.get("human_label", "").strip():
            raise ValueError("确认群体事件须填写实际人工标签")
        before = copy.deepcopy(target)
        target.update(
            status="rejected" if action == "reject" else "human_confirmed",
            human_label=decision.get("human_label"),
            reviewer=reviewer,
            review_note=decision.get("note", ""),
        )
        row = {
            "decision_id": did,
            "decision": copy.deepcopy(decision),
            "reviewer": reviewer,
            "before": before,
            "after": copy.deepcopy(target),
        }
        audit.append(row)
        done[did] = row
    result["confirmed_group_windows"] = [
        copy.deepcopy(row) for row in queue if row["status"] == "human_confirmed"
    ]
    return result
