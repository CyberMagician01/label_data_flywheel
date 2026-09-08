"""Generate a single-GPU CountGD++ BEE24 configuration and launch training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_BEE24_ROOT = WORKSPACE.parent / "data" / "BEE24"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CountGD++ on BEE24 COCO boxes.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_BEE24_ROOT,
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_BEE24_ROOT / "prepared_models" / "countgdpp" / "annotations",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("third_party/CountGDPlusPlus/training/countgd_plusplus_training"),
    )
    parser.add_argument(
        "--pretrained-model",
        type=Path,
        default=Path("third_party/CountGDPlusPlus/checkpoints/countgd_plusplus.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("work_dirs/countgdpp_bee24"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-train-samples", type=int, default=0)
    parser.add_argument("--limit-val-samples", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def make_limited_coco(source: Path, destination: Path, limit: int) -> Path:
    """保留一张图及其标注，用于验证一次前向与反向传播。"""
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["images"] = payload["images"][:limit]
    image_ids = {image["id"] for image in payload["images"]}
    payload["annotations"] = [
        annotation
        for annotation in payload.get("annotations", [])
        if annotation["image_id"] in image_ids
    ]
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


def make_limited_odvg(source: Path, destination: Path, limit: int) -> Path:
    """保留一条 JSONL 训练记录，用于验证 CountGD++ 的训练数据链路。"""
    records = source.read_text(encoding="utf-8").splitlines()[:limit]
    destination.write_text("\n".join(records) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    args = parse_args()
    workspace = WORKSPACE
    data_root = args.data_root.resolve()
    annotations_dir = args.annotations_dir.resolve()
    training_root = (workspace / args.training_root).resolve()
    pretrained_model = (workspace / args.pretrained_model).resolve()
    output_dir = (workspace / args.output_dir).resolve()
    for path in (
        data_root,
        training_root / "main.py",
        pretrained_model,
        annotations_dir / "bee24_train_odvg.jsonl",
        annotations_dir / "bee24_val.json",
        annotations_dir / "bee24_label_map.json",
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required path is missing: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_annotation = annotations_dir / "bee24_train_odvg.jsonl"
    val_annotation = annotations_dir / "bee24_val.json"
    label_map = annotations_dir / "bee24_label_map.json"
    train_limit = 1 if args.smoke_test else args.limit_train_samples
    val_limit = 1 if args.smoke_test else args.limit_val_samples
    if train_limit:
        train_annotation = make_limited_odvg(
            train_annotation, output_dir / "bee24_train_limited.jsonl", train_limit
        )
    if val_limit:
        val_annotation = make_limited_coco(
            val_annotation, output_dir / "bee24_val_limited.json", val_limit
        )
    dataset_config = {
        "train": [
            {
                "root": str(data_root),
                "anno": str(train_annotation),
                "label_map": str(label_map),
                "dataset_mode": "odvg",
            }
        ],
        "val": [
            {
                "root": str(data_root),
                "anno": str(val_annotation),
                "label_map": None,
                "dataset_mode": "coco",
            }
        ],
    }
    dataset_path = output_dir / "datasets_bee24.json"
    dataset_path.write_text(json.dumps(dataset_config, indent=2), encoding="utf-8")

    base_config = training_root / "config" / "cfg_fsc147_vit_b_debug.py"
    epoch_count = 1 if args.smoke_test else args.epochs
    config_path = output_dir / "cfg_bee24.py"
    config_path.write_text(
        base_config.read_text(encoding="utf-8")
        + "\n# BEE24 single-GPU overrides\n"
        + f"batch_size = {args.batch_size}\n"
        + f"epochs = {epoch_count}\n"
        + f"lr_drop = {max(1, epoch_count // 2)}\n"
        + f"lr_drop_list = [{max(1, epoch_count // 2)}]\n"
        + f"save_checkpoint_interval = {max(1, epoch_count // 2)}\n"
        # CountGD++ handles its training labels as zero-based indices.  Its own
        # non-COCO evaluation path still uses the generated COCO boxes, while
        # avoiding the hard-coded 80-class COCO label mapping.
        + "use_coco_eval = False\n"
        + "label_list = ['bee']\n"
        + "val_label_list = ['bee']\n",
        encoding="utf-8",
    )

    bert_path = training_root / "checkpoints" / "bert-base-uncased"
    command = [
        sys.executable,
        "main.py",
        "--output_dir",
        str(output_dir),
        "--config_file",
        str(config_path),
        "--datasets",
        str(dataset_path),
        "--pretrain_model_path",
        str(pretrained_model),
        "--num_workers",
        str(args.workers),
        "--amp",
        "--options",
        f"text_encoder_type={bert_path}",
    ]
    print("Launching:", " ".join(command))
    if args.prepare_only:
        print(f"Generated CountGD++ BEE24 configuration in: {output_dir}")
        return
    subprocess.run(command, cwd=training_root, check=True)


if __name__ == "__main__":
    main()
