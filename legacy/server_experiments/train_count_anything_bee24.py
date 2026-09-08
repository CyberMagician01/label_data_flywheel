"""Create a low-memory Count Anything configuration and launch BEE24 training."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_BEE24_ROOT = WORKSPACE.parent / "data" / "BEE24"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Count Anything on BEE24.")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_BEE24_ROOT / "prepared_models" / "count_anything",
    )
    parser.add_argument(
        "--sam3-checkpoint",
        type=Path,
        default=WORKSPACE / "third_party" / "count-anything" / "pretrained" / "sam3.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "work_dirs" / "count_anything_bee24",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=560)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-train-samples", type=int, default=0)
    parser.add_argument("--limit-val-samples", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def replace_image_size(node, image_size: int) -> None:
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if key in {"size", "sizes", "resolution"} and value == 1008:
                node[key] = image_size
            else:
                replace_image_size(value, image_size)
    elif isinstance(node, list):
        for value in node:
            replace_image_size(value, image_size)


def configure_dataset(dataset, annotation_file: Path, training: bool) -> None:
    dataset["ann_file"] = str(annotation_file)
    loader = dataset["coco_json_loader"]
    loader["class_name"] = "bee"
    loader["prompt_text"] = "bee"
    dataset["training"] = training


def main() -> None:
    args = parse_args()
    try:
        from omegaconf import OmegaConf
    except ImportError as error:
        raise SystemExit(
            "缺少 omegaconf。请先完成 Count Anything 环境安装后再运行此脚本。"
        ) from error

    workspace = WORKSPACE
    repo_root = workspace / "third_party" / "count-anything"
    annotations_dir = args.annotations_dir.resolve()
    sam3_checkpoint = args.sam3_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    train_json, val_json = annotations_dir / "train.json", annotations_dir / "val.json"
    for path in (train_json, val_json, sam3_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"Required file is missing: {path}")

    cfg = OmegaConf.load(repo_root / "config" / "count_anything_train_cloc.yaml")
    cfg.paths.train_annotation_file = str(train_json)
    cfg.paths.val_annotation_file = str(val_json)
    cfg.paths.stage1_checkpoint_path = str(sam3_checkpoint)
    cfg.paths.experiment_log_dir = str(output_dir)
    cfg.trainer.model.checkpoint_path = str(sam3_checkpoint)
    cfg.trainer.model.load_from_HF = False
    cfg.trainer.max_epochs = 1 if args.smoke_test else args.epochs
    # BEE24's configured loader yields one batch per iteration.  SAM3's grouped
    # accumulation path expects a list of batches and aborts with this dataset.
    cfg.trainer.gradient_accumulation_steps = 1
    cfg.trainer.validate_before_train = not args.smoke_test
    cfg.trainer.data.train.batch_size = args.batch_size
    cfg.trainer.data.val.batch_size = args.batch_size
    cfg.trainer.data.train.num_workers = args.workers
    cfg.trainer.data.val.num_workers = args.workers
    configure_dataset(cfg.trainer.data.train.dataset, train_json, True)
    configure_dataset(cfg.trainer.data.val.dataset, val_json, False)
    train_limit = 1 if args.smoke_test else args.limit_train_samples
    val_limit = 1 if args.smoke_test else args.limit_val_samples
    if train_limit:
        cfg.trainer.data.train.dataset.limit_ids = train_limit
    if val_limit:
        cfg.trainer.data.val.dataset.limit_ids = val_limit
    cfg.launcher.gpus_per_node = 1
    cfg.launcher.num_nodes = 1
    cfg.launcher.experiment_log_dir = str(output_dir)
    if os.name == "nt":
        cfg.trainer.distributed.backend = "gloo"
    cfg.trainer.checkpoint.save_dir = str(output_dir / "checkpoints")
    cfg.trainer.logging.log_dir = str(output_dir / "logs")
    cfg.trainer.logging.tensorboard_writer.log_dir = str(output_dir / "tensorboard")
    for scheduler in cfg.trainer.optim.options.lr:
        if "scheduler" in scheduler:
            if "max_epochs" in scheduler.scheduler:
                scheduler.scheduler.max_epochs = cfg.trainer.max_epochs
            if "end_epoch" in scheduler.scheduler:
                scheduler.scheduler.end_epoch = cfg.trainer.max_epochs
    replace_image_size(cfg, args.image_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "count_anything_bee24.yaml"
    OmegaConf.save(cfg, config_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["USE_LIBUV"] = "0"
    command = [
        sys.executable,
        str(repo_root / "sam3" / "train" / "train.py"),
        "--config",
        str(config_path),
        "--num-gpus",
        "1",
        "--use-cluster",
        "0",
    ]
    print("Launching:", " ".join(command))
    subprocess.run(command, cwd=repo_root, env=env, check=True)


if __name__ == "__main__":
    main()
