#!/usr/bin/env python3
"""Bind a fully validated public-preadaptation checkpoint to fresh E-S0."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


OFFICIAL_COCO_SHA = "c4cdf8bcd3b27c7903e422acd03caf733b5e1bfd664550bce43e50e2a3bbdc6e"
REQUIRED_STATE = {"model", "optimizer", "lr_scheduler", "scaler", "ema", "last_epoch"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def validate_checkpoint(path: Path, expected_sha: str | None = None) -> tuple[dict, str]:
    if not path.is_file() or path.stat().st_size < 100_000_000:
        raise ValueError(f"Checkpoint is absent or incomplete: {path}")
    actual_sha = sha256(path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            f"Checkpoint SHA mismatch: expected={expected_sha}, actual={actual_sha}"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing = REQUIRED_STATE - set(state)
    if missing:
        raise ValueError(f"Checkpoint misses complete training state: {sorted(missing)}")
    if not isinstance(state["ema"], dict) or "module" not in state["ema"]:
        raise ValueError("Checkpoint EMA state is incomplete.")
    return state, actual_sha


def render_template(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for placeholder in sorted(replacements, key=len, reverse=True):
        rendered = rendered.replace(placeholder, replacements[placeholder])
    if "PUBLIC_" in rendered:
        raise ValueError("Rendered E-S0 config still contains an unresolved placeholder.")
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--p0a-checkpoint", type=Path, required=True)
    parser.add_argument("--p0b-output", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--rendered-config", type=Path, required=True)
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Public data manifest not found: {args.manifest}")
    manifest_sha = sha256(args.manifest)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if set(manifest.get("public_datasets", [])) != {
        "BEE24", "BeePose", "MendeleyBeePose"
    }:
        raise ValueError("Public data manifest does not cover all required datasets.")
    if any(manifest.get("leakage_checks", {}).values()):
        raise ValueError("Public data leakage gate failed.")

    p0a_state, p0a_sha = validate_checkpoint(args.p0a_checkpoint)
    if int(p0a_state["last_epoch"]) != 0:
        raise ValueError("E-P0A handoff requires the complete epoch0 checkpoint.")
    selection_path = args.p0b_output / "selected_checkpoint.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_checkpoint = Path(selection["checkpoint"]).expanduser().resolve()
    selected_state, selected_sha = validate_checkpoint(
        selected_checkpoint, str(selection["sha256"]).lower()
    )
    selected_epoch = int(selection["epoch"])
    if not 0 <= selected_epoch < 12:
        raise ValueError("Selected E-P0B epoch is outside the complete 12-cycle stage.")
    if int(selected_state["last_epoch"]) != selected_epoch:
        raise ValueError("Selected E-P0B epoch disagrees with the checkpoint state.")

    evidence = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_official_checkpoint_sha256": OFFICIAL_COCO_SHA,
        "public_datasets": ["BEE24", "BeePose", "MendeleyBeePose"],
        "public_data_manifest": str(args.manifest.resolve()),
        "public_data_manifest_sha256": manifest_sha,
        "p0a": {
            "checkpoint": str(args.p0a_checkpoint.resolve()),
            "checkpoint_sha256": p0a_sha,
            "last_epoch": int(p0a_state["last_epoch"]),
        },
        "p0b": {
            "selected_epoch": selected_epoch,
            "selected_checkpoint": str(selected_checkpoint),
            "metrics": selection.get("metrics", {}),
            "last_epoch": int(selected_state["last_epoch"]),
        },
        "selected_checkpoint_sha256": selected_sha,
        "handoff_policy": "model_and_ema_tuning_with_fresh_E-S0_optimizer_scheduler",
    }
    atomic_text(
        args.evidence,
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    evidence_sha = sha256(args.evidence)

    rendered = render_template(
        args.template.read_text(encoding="utf-8"),
        {
            "PUBLIC_SELECTED_CHECKPOINT": str(selected_checkpoint),
            "PUBLIC_SELECTED_SHA256": selected_sha,
            "PUBLIC_PREADAPTATION_EVIDENCE": str(args.evidence.resolve()),
            "PUBLIC_PREADAPTATION_EVIDENCE_SHA256": evidence_sha,
        },
    )
    if evidence_sha not in rendered:
        raise ValueError("Rendered E-S0 config is not bound to the evidence SHA256.")
    atomic_text(args.rendered_config, rendered)
    print(json.dumps({
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": selected_sha,
        "evidence": str(args.evidence),
        "evidence_sha256": evidence_sha,
        "rendered_config": str(args.rendered_config),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
