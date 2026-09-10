"""官方DDS接口：T-Rex2视觉/嵌入提示，GroundingDINO文本与协同候选。

T-Rex2公开API没有给出可验证的原生文本请求格式，因此文本侧明确标为
GroundingDINO，不把两次请求的融合冒称T-Rex2内部文本-视觉网络。
"""

import base64
import copy
import os
import time
from pathlib import Path
from .geometry import overlap
from .io import write_json, read_json


class DDSClient:
    def __init__(
        self, token=None, token_env="TREX2_API_TOKEN", session=None, timeout=120
    ):
        import requests

        self.token = token or os.environ.get(token_env)
        if not self.token:
            raise RuntimeError(f"缺少{token_env}")
        self.session = session or requests.Session()
        self.timeout = timeout

    @staticmethod
    def image(path):
        p = Path(path)
        mime = "png" if p.suffix.lower() == ".png" else "jpeg"
        return f"data:image/{mime};base64," + base64.b64encode(p.read_bytes()).decode()

    def task(self, endpoint, body):
        headers = {"Token": self.token, "Content-Type": "application/json"}
        response = self.session.post(
            "https://api.deepdataspace.com" + endpoint,
            json=body,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("msg") != "ok":
            raise RuntimeError("DDS拒绝任务；请检查账户权限和请求参数")
        task_id = payload["data"]["task_uuid"]
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            response = self.session.get(
                "https://api.deepdataspace.com/v2/task_status/" + task_id,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            status = payload["data"]["status"]
            if status == "success":
                return {"task_uuid": task_id, "result": payload["data"]["result"]}
            if status not in ("waiting", "running"):
                raise RuntimeError("DDS任务失败，未产出检测结果")
            time.sleep(1)
        raise TimeoutError("DDS任务尚未完成，请按task_uuid检查服务状态")

    def visual(self, image, references, embedding=False):
        prompts = copy.deepcopy(references)
        for p in prompts:
            p["image"] = self.image(p["image"])
        return self.task(
            "/v2/task/trex/detection",
            {
                "model": "T-Rex-2.0",
                "image": self.image(image),
                "targets": ["bbox", "embedding"] if embedding else ["bbox"],
                "prompt": {"type": "visual_images", "visual_images": prompts},
            },
        )

    def embedding(self, image, embedding):
        return self.task(
            "/v2/task/trex/detection",
            {
                "model": "T-Rex-2.0",
                "image": self.image(image),
                "targets": ["bbox"],
                "prompt": {"type": "embedding", "embedding": embedding},
            },
        )

    def visual_from_memory(self, image, memory, domain, tags=(), limit=4):
        references = select_prompts(memory, domain, tags, limit)
        references += select_prompts(memory, domain, tags, limit, negative=True)
        if not references:
            raise ValueError("当前域没有已确认提示")
        return self.visual(image, references)

    def text(self, image, text="bee"):
        return self.task(
            "/v2/task/grounding_dino/detection",
            {
                "model": "GroundingDino-1.6-Pro",
                "image": self.image(image),
                "targets": ["bbox"],
                "prompt": {"type": "text", "text": text},
            },
        )

    def hybrid(self, image, references, text="bee"):
        return {
            "visual": self.visual(image, references),
            "text": self.text(image, text),
            "fusion_level": "candidate_evidence",
            "native_trex_text": False,
        }


def cluster_evidence(results, iou_threshold=0.5):
    """每个实体聚合同空间候选，保留各专家原始分数供域内校准。"""
    clusters = []
    for name, result in results.items():
        for obj in result["result"].get("objects", []):
            box = obj["bbox"]
            score = obj["score"]
            ious = overlap([box], [c["bbox_xyxy"] for c in clusters])[2]
            index = int(ious[0].argmax()) if ious.size else None
            if index is None or ious[0, index] < iou_threshold:
                clusters.append({"bbox_xyxy": box, "experts": []})
                index = len(clusters) - 1
            clusters[index]["experts"].append(
                {"name": name, "family": name, "score": score}
            )
    return clusters


def update_prompt_memory(path, examples):
    memory = read_json(path) if Path(path).exists() else {"examples": []}
    old = {x["entity_id"]: x for x in memory["examples"]}
    for item in examples:
        if item.get("label_status") != "human_confirmed" or not item.get("reviewer"):
            raise ValueError("提示记忆只接受人工确认的正/负样例")
        if item["split"] == "test":
            raise ValueError("测试集不能更新提示记忆")
        old[item["entity_id"]] = item
    memory["examples"] = list(old.values())
    write_json(path, memory)
    return memory


def select_prompts(memory, domain, tags=(), limit=4, negative=False):
    examples = [
        e
        for e in memory["examples"]
        if e["domain"] == domain and bool(e.get("negative")) == negative
    ]
    examples.sort(key=lambda e: len(set(tags) & set(e.get("tags", []))), reverse=True)
    return [
        {
            "image": e["image"],
            "interactions": [
                {
                    "type": "rect",
                    "category_id": 2 if negative else 1,
                    "rect": e["bbox_xyxy"],
                }
            ],
        }
        for e in examples[:limit]
    ]
