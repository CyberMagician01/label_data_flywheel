"""从带 Track ID 的头尾关键点结果计算运动学、密度和交互网络指标。"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--interaction-radius", type=float, default=2.0,
                        help="以个体头尾长度为单位的交互半径")
    parser.add_argument("--min-track-length", type=int, default=3)
    return parser.parse_args()


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_heatmap(points, width, height, target):
    if not points:
        return
    xy = np.asarray(points)
    heatmap, _, _ = np.histogram2d(
        xy[:, 1], xy[:, 0], bins=(72, 128), range=((0, height), (0, width))
    )
    heatmap = np.log1p(heatmap)
    heatmap /= max(float(heatmap.max()), 1e-6)
    red = np.clip(1.8 * heatmap, 0, 1)
    green = np.clip(1.8 * (1 - np.abs(heatmap - 0.55) / 0.55), 0, 1)
    blue = np.clip(1.4 * (1 - heatmap), 0, 1)
    color = np.stack((red, green, blue), axis=2)
    Image.fromarray((color * 255).astype(np.uint8)).resize(
        (1280, 720), Image.Resampling.BILINEAR
    ).save(target)
    np.save(target.with_suffix(".npy"), heatmap.astype(np.float32))


def main():
    args = parse_args()
    dataset = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in dataset["images"]}
    predicted = {}
    if args.predictions:
        result = json.loads(args.predictions.read_text(encoding="utf-8"))
        predicted = {item["annotation_id"]: item["pred_keypoints"] for item in result["predictions"]}

    observations = []
    for annotation in dataset["annotations"]:
        image = images[annotation["image_id"]]
        keypoints = predicted.get(annotation["id"], annotation["keypoints"])
        head = np.asarray(keypoints[0:2], dtype=float)
        tail = np.asarray(keypoints[3:5], dtype=float)
        center = (head + tail) * 0.5
        length = max(float(np.linalg.norm(tail - head)), 1e-6)
        observations.append({
            "source": image.get("source", ""), "scene": image.get("scene", ""),
            "video": image.get("video", ""), "frame": int(image.get("frame", image["id"])),
            "segment": str(Path(image.get("source_json", image["file_name"])).parent),
            "track_id": str(annotation.get("track_id", annotation["id"])),
            "x": float(center[0]), "y": float(center[1]), "length": length,
            "angle": math.atan2(tail[1] - head[1], tail[0] - head[0]),
        })

    tracks = defaultdict(list)
    frames = defaultdict(list)
    for item in observations:
        track_key = (item["source"], item["scene"], item["video"], item["segment"], item["track_id"])
        frame_key = (item["source"], item["scene"], item["video"], item["segment"], item["frame"])
        tracks[track_key].append(item)
        frames[frame_key].append(item)

    track_rows = []
    observation_rows = []
    valid_tracks = {}
    for key, items in tracks.items():
        items.sort(key=lambda item: item["frame"])
        if len(items) < args.min_track_length:
            continue
        valid_tracks[key] = items
        frame = np.asarray([item["frame"] for item in items], dtype=float)
        xy = np.asarray([[item["x"], item["y"]] for item in items])
        length = np.asarray([item["length"] for item in items])
        angle = np.unwrap(np.asarray([item["angle"] for item in items]))
        dt = np.diff(frame) / args.fps
        valid = dt > 0
        speed = np.linalg.norm(np.diff(xy, axis=0), axis=1)[valid] / dt[valid]
        angular_velocity = np.degrees(np.diff(angle)[valid] / dt[valid])
        body_speed = speed / max(float(np.median(length)), 1e-6)
        window = min(len(angle), max(3, int(round(args.fps / max(np.median(np.diff(frame)), 1)))))
        trend = np.convolve(angle, np.ones(window) / window, mode="same")
        margin = window // 2
        residual = angle[margin:len(angle) - margin] - trend[margin:len(angle) - margin]
        wag_proxy = float(np.degrees(np.percentile(np.abs(residual), 95))) if residual.size else 0.0
        track_rows.append({
            "source": key[0], "scene": key[1], "video": key[2], "track_id": key[4],
            "observations": len(items), "span_seconds": (frame[-1] - frame[0]) / args.fps,
            "mean_speed_px_s": float(speed.mean()) if speed.size else 0.0,
            "mean_speed_body_lengths_s": float(body_speed.mean()) if body_speed.size else 0.0,
            "mean_abs_angular_velocity_deg_s": float(np.abs(angular_velocity).mean()) if angular_velocity.size else 0.0,
            "orientation_oscillation_amplitude_deg": wag_proxy,
        })
        for index, item in enumerate(items):
            if index == 0 or frame[index] <= frame[index - 1]:
                instant_speed = 0.0
                instant_body_speed = 0.0
                instant_angular_velocity = 0.0
            else:
                instant_dt = (frame[index] - frame[index - 1]) / args.fps
                instant_speed = float(np.linalg.norm(xy[index] - xy[index - 1]) / instant_dt)
                instant_body_speed = instant_speed / max(float(np.median(length)), 1e-6)
                instant_angular_velocity = float(
                    np.degrees(wrap_angle(angle[index] - angle[index - 1]) / instant_dt)
                )
            observation_rows.append({
                "source": key[0], "scene": key[1], "video": key[2],
                "track_id": key[4], "frame": int(frame[index]),
                "time_seconds": float((frame[index] - frame[0]) / args.fps),
                "x": float(xy[index, 0]), "y": float(xy[index, 1]),
                "absolute_orientation_deg": float(np.degrees(wrap_angle(angle[index]))),
                "instant_speed_px_s": instant_speed,
                "instant_speed_body_lengths_s": instant_body_speed,
                "instant_angular_velocity_deg_s": instant_angular_velocity,
            })

    median_length = float(np.median([item["length"] for item in observations]))
    radius = median_length * args.interaction_radius
    edges = defaultdict(float)
    starts = defaultdict(int)
    ends = defaultdict(int)
    for items in valid_tracks.values():
        first, last = items[0], items[-1]
        first_key = (first["source"], first["scene"], first["video"], first["segment"], first["frame"])
        last_key = (last["source"], last["scene"], last["video"], last["segment"], last["frame"])
        starts[first_key] += 1
        ends[last_key] += 1

    frame_rows = []
    for key, items in sorted(frames.items()):
        positions = np.asarray([[item["x"], item["y"]] for item in items])
        interaction_count = 0
        for first in range(len(items)):
            for second in range(first + 1, len(items)):
                if float(np.linalg.norm(positions[first] - positions[second])) <= radius:
                    a, b = sorted((items[first]["track_id"], items[second]["track_id"]))
                    edges[(key[0], key[1], key[2], key[3], a, b)] += 1.0 / args.fps
                    interaction_count += 1
        frame_rows.append({
            "source": key[0], "scene": key[1], "video": key[2], "frame": key[4],
            "bee_count": len(items), "interaction_pairs": interaction_count,
            "observed_track_starts": starts[key],
            "observed_track_ends": ends[key],
            "observed_net_flux_proxy": starts[key] - ends[key],
            "mean_orientation_deg": float(np.degrees(np.angle(np.mean(np.exp(1j * np.asarray([x["angle"] for x in items])))))),
        })
    edge_rows = [
        {"source": key[0], "scene": key[1], "video": key[2],
         "track_a": key[4], "track_b": key[5], "interaction_seconds": value}
        for key, value in sorted(edges.items(), key=lambda item: item[1], reverse=True)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_track_behavior.csv", track_rows)
    write_csv(args.output_dir / "per_observation_kinematics.csv", observation_rows)
    write_csv(args.output_dir / "per_frame_group_stats.csv", frame_rows)
    write_csv(args.output_dir / "interaction_network.csv", edge_rows)
    width = max(item["width"] for item in dataset["images"])
    height = max(item["height"] for item in dataset["images"])
    render_heatmap([(item["x"], item["y"]) for item in observations], width, height,
                   args.output_dir / "group_density_heatmap.png")
    summary = {
        "observations": len(observations), "valid_tracks": len(valid_tracks),
        "median_body_length_px": median_length, "interaction_radius_px": radius,
        "mean_count_per_annotated_frame": float(np.mean([row["bee_count"] for row in frame_rows])),
        "interaction_edges": len(edge_rows),
        "flux_note": "observed_net_flux_proxy 只表示轨迹在观测片段中的开始数减结束数；需要标定巢口 ROI 和内外方向后才能解释为真实巢口通量。",
        "important_note": "orientation_oscillation_amplitude_deg 是头尾身体轴摆动代理；没有胸部关键点时不能解释为腹部相对胸部的真实摆尾角。",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
