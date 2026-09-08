"""bee-flywheel命令行：导入、去重、质量/行为、采样、复核及原模型调用。"""

import argparse
import json
from pathlib import Path
from .io import iter_frames, write_frames, read_json, write_json


def frame_args(ap):
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--domain", choices=["RGB_out", "IR_in"], required=True)
    ap.add_argument("--group", default="unknown")
    ap.add_argument("--split", default="unassigned")
    ap.add_argument("--source", default="prediction")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest="command", required=True)
    for name in ("ingest", "dedup", "analyze", "review", "run"):
        sub = subs.add_parser(name)
        frame_args(sub)
        if name in ("dedup", "run"):
            sub.add_argument("--threshold", type=float, default=0.2)
            sub.add_argument("--interpolate", action="store_true")
        if name == "review":
            sub.add_argument("--decisions", required=True)
            sub.add_argument("--reviewer", required=True)
    sample = subs.add_parser("sample")
    sample.add_argument("--input", required=True)
    sample.add_argument("--targets", required=True)
    sample.add_argument("--output", required=True)
    round_parser = subs.add_parser("round")
    round_parser.add_argument("--config", required=True)
    round_parser.add_argument("--output", required=True)
    cal = subs.add_parser("calibrate-experts")
    cal.add_argument("--input", required=True)
    cal.add_argument("--output", required=True)
    annotations = subs.add_parser(
        "export-annotations", help="只导出数据飞轮标注包，不生成参赛推理程序"
    )
    annotations.add_argument("--config", required=True)
    annotations.add_argument("--output", required=True)
    validation = subs.add_parser("validate-annotations")
    validation.add_argument("--input", required=True)
    validation.add_argument("--output")
    behavior_train = subs.add_parser("train-behavior")
    behavior_train.add_argument("--input", required=True)
    behavior_train.add_argument("--output", required=True)
    behavior_train.add_argument("--epochs", type=int, default=100)
    behavior_train.add_argument("--device", default="cpu")
    export = subs.add_parser("export-training")
    frame_args(export)
    export.add_argument("--image-root", required=True)
    tracking = subs.add_parser("associate")
    frame_args(tracking)
    tracking.add_argument("--retention", type=int, default=150)
    evaluation = subs.add_parser("evaluate")
    frame_args(evaluation)
    evaluation.add_argument("--gt", required=True)
    evaluation.add_argument("--tracking", action="store_true")
    backend = subs.add_parser("backend")
    backend.add_argument("name")
    backend.add_argument("--config", required=True)
    backend.add_argument("--dry-run", action="store_true")
    args, extra = ap.parse_known_args(argv)
    if extra and args.command != "backend":
        ap.error("不认识的参数：" + " ".join(extra))
    if args.command == "round":
        from .pipeline import run_round

        print(
            json.dumps(
                run_round(read_json(args.config), args.output), ensure_ascii=False
            )
        )
        return
    if args.command == "export-annotations":
        from .annotation_package import export_annotations

        result = export_annotations(read_json(args.config), args.output)
        print(
            json.dumps(
                {
                    "version": result["version"],
                    "counts": result["counts"],
                    "pending": result["pending"],
                },
                ensure_ascii=False,
            )
        )
        return
    if args.command == "validate-annotations":
        from .annotation_package import validate_annotations

        result = validate_annotations(args.input)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False))
        if not result["passed"]:
            raise SystemExit(1)
        return
    if args.command == "calibrate-experts":
        from .experts import fit_experts

        write_json(args.output, fit_experts(read_json(args.input)))
        return
    if args.command == "train-behavior":
        from .behavior_learning import train

        result = train(read_json(args.input), args.output, args.epochs, args.device)
        write_json(Path(args.output).with_suffix(".training.json"), result)
        return
    if args.command == "backend":
        from .backends import run_backend

        arguments = extra[1:] if extra[:1] == ["--"] else extra
        print(
            json.dumps(
                run_backend(args.name, read_json(args.config), arguments, args.dry_run),
                ensure_ascii=False,
            )
        )
        return
    if args.command == "sample":
        from .sampling import calibrate_weights, balanced_batches, distribution_report

        records = read_json(args.input)
        targets = read_json(args.targets)
        weights = calibrate_weights(records, targets)
        write_json(
            args.output,
            {
                "weights": weights.tolist(),
                "batches": balanced_batches(records, weights),
                "diagnostics": distribution_report(records, weights, targets),
            },
        )
        return
    frames = list(
        iter_frames(
            args.input,
            video=args.video,
            domain=args.domain,
            group=args.group,
            split=args.split,
            source=args.source,
        )
    )
    if args.command == "export-training":
        from .training_data import export_round

        export_round(frames, args.output, args.image_root)
        return
    if args.command == "associate":
        from .tracking import AssociationTracker

        tracker = AssociationTracker(args.retention)
        write_frames(
            args.output,
            (tracker.update(f) for f in sorted(frames, key=lambda f: f["frame"])),
        )
        return
    if args.command == "evaluate":
        from .metrics import evaluate

        gt = list(
            iter_frames(
                args.gt,
                video=args.video,
                domain=args.domain,
                group=args.group,
                split=args.split,
                source="human",
            )
        )
        write_json(args.output, evaluate(gt, frames, args.tracking))
        return
    if args.command == "ingest":
        write_frames(args.output, frames)
    elif args.command in ("dedup", "run"):
        from .postprocess import suppress_interpolation, interpolate

        if args.interpolate:
            frames = interpolate(frames)
        frames = [suppress_interpolation(f, args.threshold) for f in frames]
        if args.command == "dedup":
            write_frames(args.output, frames)
    elif args.command == "review":
        from .review import apply_reviews, feedback_snapshot

        result = apply_reviews(
            frames,
            read_json(args.decisions),
            args.reviewer,
            Path(args.output).with_suffix(".audit.jsonl"),
        )
        write_frames(args.output, result)
        write_frames(
            Path(args.output).with_suffix(".confirmed.jsonl"), feedback_snapshot(result)
        )
    if args.command in ("analyze", "run"):
        from .quality import frame_evidence, q2
        from .behavior import analyze
        from .review import prioritize, error_cube
        from .curriculum import meta_policy

        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        quality = q2([r for f in frames for r in frame_evidence(f)])
        cube = error_cube(quality)
        write_json(out / "quality.json", quality)
        write_json(out / "error_cube.json", cube)
        write_json(out / "behavior.json", analyze(frames))
        review = [
            dict(
                r,
                uncertainty=1 - r["quality"],
                disagreement=r["error"],
                features=[r["error"], r["quality"]],
            )
            for r in quality
        ]
        write_json(out / "review_queue.json", prioritize(review, 50))
        write_json(out / "next_round_policy.json", meta_policy(cube, 50))
        if args.command == "run":
            write_frames(out / "annotations.jsonl.gz", frames)
        write_json(
            out / "run_summary.json",
            {
                "frames": len(frames),
                "boxes": sum(len(f["detections"]) for f in frames),
                "source": args.input,
                "quality_is_geometric_proxy": True,
                "behavior_is_candidate": True,
            },
        )


if __name__ == "__main__":
    main()
