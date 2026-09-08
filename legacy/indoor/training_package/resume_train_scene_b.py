import os
import sys
from pathlib import Path
from ultralytics import YOLO

PACKAGE_ROOT = Path(__file__).resolve().parent

def main():
    print("=" * 70, flush=True)
    print("🚀 [纯单卡 GPU 0 无缝续训: Scene B 室内密蜂 YOLOv8x-P2]", flush=True)
    print("📌 机制说明: 锁定 GPU 0 独占训练，继承第 14 轮 checkpoint 与精确学习率衰减进度", flush=True)
    print("=" * 70, flush=True)

    last_pt_path = PACKAGE_ROOT.parent / 'runs' / 'unified' / 'scene_b_unified_yolov8x_p2' / 'weights' / 'last.pt'
    
    if not last_pt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {last_pt_path}")

    # 1. Load checkpoint
    model = YOLO(str(last_pt_path))

    # 2. Resume training strictly on Single GPU 0
    results = model.train(
        resume=True,
        device=0,
        amp=True
    )

    print("🎉 Scene B YOLOv8x-P2 续训全部完成！", flush=True)
    print(f"最优模型权重: {results.save_dir}/weights/best.pt", flush=True)

if __name__ == '__main__':
    main()
