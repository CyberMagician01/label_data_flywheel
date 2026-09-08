"""模型在各自Python环境运行，共享JSON协议。"""

import argparse
import copy
import importlib.util
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from label_data_flywheel.io import read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["yolo", "vitpose", "density"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config")
    ap.add_argument("--vitpose-root", type=Path)
    ap.add_argument("--image", required=True)
    ap.add_argument("--detections")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-det", type=int, default=1200)
    args = ap.parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    if args.model == "yolo":
        from ultralytics import YOLO

        result = YOLO(args.checkpoint).predict(
            image[:, :, ::-1].copy(),
            device=args.device,
            verbose=False,
            imgsz=args.imgsz,
            conf=args.conf,
            max_det=args.max_det,
        )[0]
        output = {
            "detections": [
                {"bbox_xyxy": b, "confidence": s, "class_id": int(c)}
                for b, s, c in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                )
            ]
        }
        if result.keypoints is not None:
            for d, k in zip(output["detections"], result.keypoints.data.cpu().tolist()):
                d["keypoints"] = {"head": k[0], "abdomen_tip": k[1]}
    elif args.model == "vitpose":
        if args.vitpose_root:
            sys.path.insert(0, str(args.vitpose_root))
        from mmpose.apis import init_pose_model, inference_top_down_pose_model
        from mmpose.datasets import DatasetInfo
        from mmpose.datasets.builder import PIPELINES

        @PIPELINES.register_module(force=True)
        class SetBeeDatasetIndex:
            def __init__(self, dataset_idx=0):
                self.dataset_idx = dataset_idx

            def __call__(self, result):
                result["dataset_idx"] = self.dataset_idx
                return result

        model = init_pose_model(args.config, args.checkpoint, device=args.device)
        data = read_json(args.detections)
        ds = data["detections"]
        if "test_pipeline" not in model.cfg:
            model.cfg.test_pipeline = copy.deepcopy(model.cfg.data.test.pipeline)
        model.cfg.test_pipeline.insert(
            0, dict(type="SetBeeDatasetIndex", dataset_idx=0)
        )
        dataset_info = DatasetInfo(model.cfg.data.test.dataset_info)
        people = [{"bbox": np.r_[d["bbox_xyxy"], 1.0]} for d in ds]
        results, _ = inference_top_down_pose_model(
            model,
            image[:, :, ::-1].copy(),
            people,
            format="xyxy",
            dataset_info=dataset_info,
        )
        if len(results) != len(ds):
            raise ValueError("姿态输出数量与输入框不同")
        for d, r in zip(ds, results):
            d["keypoints"] = {
                "head": r["keypoints"][0].tolist(),
                "abdomen_tip": r["keypoints"][1].tolist(),
            }
        output = data
    else:
        import torch

        p = ROOT / "legacy/server_experiments/train_bee_density.py"
        spec = importlib.util.spec_from_file_location("bee_density", p)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        opts = saved.get("args", {})
        width = opts.get("width", 640)
        height = opts.get("height", 352)
        model = module.DensityNet(pretrained=False).to(args.device)
        model.load_state_dict(saved["model"])
        model.eval()
        x = np.asarray(Image.fromarray(image).resize((width, height)), np.float32) / 255
        x = torch.from_numpy(x.transpose(2, 0, 1))
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None]
        with torch.no_grad():
            density = model(x[None].to(args.device))[0, 0].cpu().numpy()
        map_path = Path(args.output).with_suffix(".npy")
        map_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(map_path, density)
        output = {
            "estimated_count": float(density.sum()),
            "density_path": str(map_path),
            "shape": list(density.shape),
            "source_image_size": [image.shape[1], image.shape[0]],
            "checkpoint": args.checkpoint,
        }
    output.setdefault("source_image_size", [image.shape[1], image.shape[0]])
    output.setdefault("image_size", [image.shape[1], image.shape[0]])
    output.setdefault("image_path", args.image)
    write_json(args.output, output)


if __name__ == "__main__":
    main()
