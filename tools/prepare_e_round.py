"""把飞轮快照绑定到E训练器；不改原实验配置或已有权重。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "legacy/route_e/ecdetseg"))
from label_data_flywheel.io import read_json, write_json, sha256


def main():
    import yaml
    from engine.core.yaml_utils import load_config
    from engine.data.dataset.coco_dataset import CocoDetection

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--image-root", default="/")
    ap.add_argument("--sampling-plan")
    ap.add_argument("--split-contract")
    ap.add_argument("--checkpoint")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    snapshot = read_json(Path(args.snapshot) / "snapshot.json")
    config = load_config(args.base_config)
    for role, name in [
        ("train", "train_dataloader"),
        ("calibration", "val_dataloader"),
    ]:
        source = snapshot["splits"][role]
        document = read_json(source["path"])
        CocoDetection.validate_schema_dict(document, 2)
        dataset = config[name]["dataset"]
        dataset.update(
            ann_file=source["path"],
            img_folder=args.image_root,
            expected_ann_sha256=source["sha256"],
        )
    if args.split_contract:
        config["bee_e_split_contract"] = read_json(args.split_contract)
    if config.get("enable_continuous_stage_training"):
        from tools.bee_e.split_contract import validate_schema_v2_split_contract

        validate_schema_v2_split_contract(
            config["bee_e_split_contract"],
            config["train_dataloader"]["dataset"],
            config["val_dataloader"]["dataset"],
        )
    if args.sampling_plan:
        plan = read_json(args.sampling_plan)
        if not plan["gate"]["passed"]:
            raise ValueError("采样计划未通过覆盖/ESS/双域门")
        config["flywheel_input_plan"] = {
            "path": str(Path(args.sampling_plan).resolve()),
            "sha256": sha256(args.sampling_plan),
            "applied_via": "COCO images[].sampling_weight",
        }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("配置输出已存在")
    output.parent.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output.parent / "training_output")
    output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    command = [
        sys.executable,
        str(ROOT / "legacy/route_e/ecdetseg/train.py"),
        "-c",
        str(output),
    ]
    if args.checkpoint:
        command.extend(["--resume" if args.resume else "--tuning", args.checkpoint])
    write_json(
        output.with_suffix(".launch.json"),
        {
            "argv": command,
            "snapshot_sha256": sha256(Path(args.snapshot) / "snapshot.json"),
            "executed": False,
            "checkpoint_lineage_preserved": bool(args.resume),
        },
    )
    print(json.dumps({"config": str(output), "argv": command}, ensure_ascii=False))


if __name__ == "__main__":
    main()
