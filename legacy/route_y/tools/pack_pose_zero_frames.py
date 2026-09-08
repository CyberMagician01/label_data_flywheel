#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path


ROOT = Path("/data/bee26/beeposetrack_y_20260827")
DATASET = ROOT / "datasets" / "y_unified_20260901_labelme5_a1_strict_v8_annotator01_val_03_test"
MANIFEST = DATASET / "dataset_manifest.jsonl"
OUTDIR = Path("/tmp/y_strict_v8_pose_zero_frames")
ARCHIVE = Path("/tmp/y_strict_v8_pose_zero_frames.tar.gz")


def main() -> None:
    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    OUTDIR.mkdir(parents=True)

    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pose = sum(1 for inst in row["instances"] if inst.get("pose_mask"))
        if row["split"] == "train" and pose == 0:
            rows.append(row)

    summary = []
    for idx, row in enumerate(rows, 1):
        det = len(row["instances"])
        section = Path(row["image_path"]).name.split("__frame", 1)[0]
        name = f"{idx:02d}__{section}__frame{row['frame_id']:06d}__det{det}_pose0"
        frame_dir = OUTDIR / name
        frame_dir.mkdir(parents=True)

        src_json = Path(row["source_json"])
        src_img = Path(row["source_image_path"])
        y_img = Path(row["image_path"])
        y_lab = Path(str(y_img).replace("/images/", "/labels/")).with_suffix(".txt")
        for label, path in [
            ("original_labelme", src_json),
            ("original_image", src_img),
            ("derived_image", y_img),
            ("derived_yolo_label", y_lab),
        ]:
            if path.exists():
                shutil.copy2(path, frame_dir / f"{label}{path.suffix}")

        (frame_dir / "manifest_record.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        labels = {}
        try:
            data = json.loads(src_json.read_text(encoding="utf-8"))
            for shape in data.get("shapes", []):
                key = str(shape.get("label")).lower()
                labels[key] = labels.get(key, 0) + 1
        except Exception as exc:  # pragma: no cover - diagnostic script
            labels = {"read_error": str(exc)}

        summary.append(
            {
                "folder": name,
                "split": row["split"],
                "video_id": row["video_id"],
                "section_id": row["section_id"],
                "frame_id": row["frame_id"],
                "det_instances": det,
                "pose_instances": 0,
                "source_json": str(src_json),
                "source_image_path": str(src_img),
                "shape_label_counts": labels,
            }
        )

    (OUTDIR / "pose_zero_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# strict_v8 train pose=0 frames",
        "",
        "These frames have detection/tracking instances but zero complete head+tail pose labels in the derived YOLO view.",
        "",
        "| folder | video | section | frame | det | original shape labels |",
        "|---|---|---|---:|---:|---|",
    ]
    for item in summary:
        lines.append(
            "| `{folder}` | {video_id} | {section_id} | {frame_id} | {det_instances} | `{shape_label_counts}` |".format(
                **item
            )
        )
    (OUTDIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with tarfile.open(ARCHIVE, "w:gz") as tar:
        tar.add(OUTDIR, arcname=OUTDIR.name)

    print(f"pose_zero_frames {len(rows)}")
    print(f"archive {ARCHIVE}")
    for item in summary:
        print(item["folder"], item["shape_label_counts"])


if __name__ == "__main__":
    main()
