import os
import sys
from pathlib import Path
from ultralytics import YOLO

PACKAGE_ROOT = Path(__file__).resolve().parent
WORK_ROOT = PACKAGE_ROOT.parent


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_bool(name, default):
    value = os.environ.get(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}

def main():
    print("=" * 70, flush=True)
    print("🚀 [启动 Scene B 室内密蜂 YOLOv8x-P2 训练 (微小目标特化)]", flush=True)
    print("=" * 70, flush=True)

    resume_path = os.environ.get('RESUME_PATH')
    if resume_path:
        print(f"Resuming full training state from: {resume_path}", flush=True)
        model = YOLO(resume_path)
        results = model.train(
            resume=True,
            device=os.environ.get('DEVICE', '0,1'),
            batch=env_int('BATCH', 32),
            workers=env_int('WORKERS', 12),
            cache=os.environ.get('CACHE', 'disk'),
            save_period=env_int('SAVE_PERIOD', 10),
            patience=env_int('PATIENCE', 50),
            plots=True,
        )
        print("Scene B YOLOv8x-P2 resumed training completed!", flush=True)
        return

    # 1. Initialize YOLOv8x-P2 architecture (with P2/4 160x160 shallow HD feature head)
    model = YOLO(str(PACKAGE_ROOT / 'configs' / 'yolov8x-p2.yaml'))
    
    # 2. Transfer official COCO pre-trained weights from YOLOv8x
    try:
        model.load(str(PACKAGE_ROOT / 'weights' / 'yolov8x.pt'))
        print("✅ Successfully transferred YOLOv8x pre-trained backbone weights!", flush=True)
    except Exception as e:
        print(f"Notice: Initializing directly ({e})", flush=True)

    # 3. Train on sliced unified Scene B dataset (640x640)
    # Using batch=16 (8 per GPU) for optimal VRAM efficiency on dense Scenes (1500+ bees/crop)
    results = model.train(
        data=str(PACKAGE_ROOT / 'configs' / 'scene_b_unified.yaml'),
        epochs=env_int('EPOCHS', 300),
        batch=env_int('BATCH', 16),
        imgsz=640,
        device=os.environ.get('DEVICE', '0,1'),
        workers=env_int('WORKERS', 12),
        cache=os.environ.get('CACHE', 'ram'),
        amp=True,
        seed=env_int('SEED', 2026),
        deterministic=True,
        project=os.environ.get('PROJECT', str(WORK_ROOT / 'runs' / 'unified')),
        name=os.environ.get('RUN_NAME', 'scene_b_annotators_01_02_03_yolov8x_p2_seed2026'),
        exist_ok=env_bool('EXIST_OK', False),
        pretrained=True,
        optimizer='auto',
        patience=env_int('PATIENCE', 50),
        save=True,
        save_period=env_int('SAVE_PERIOD', 10),
        plots=True
    )

    print("🎉 Scene B YOLOv8x-P2 unified training completed!", flush=True)
    save_dir = getattr(results, 'save_dir', None)
    if save_dir is not None:
        print(f"Best model saved to: {save_dir}/weights/best.pt", flush=True)

if __name__ == '__main__':
    main()
