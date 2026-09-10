"""沿用最终选定参数，顺序运行室外双向关联、轨迹片段拼接和 ID-only 回写。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def run(config_path, sequences=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sequences = sequences or list(config["sequences"])
    here = Path(__file__).resolve().parent
    env = {**os.environ, "BEE_OUTDOOR_CONFIG": str(config_path)}
    for sequence in sequences:
        commands = [
            ["bidirectional_tracker.py", "--sequence", sequence, "--mode", "full",
             "--max-age", "8", "--unmatched-cost", "2.1", "--single-flow-cost", ".92",
             "--segment-frames", "600", "--overlap-frames", "120", "--flow-scale", ".5",
             "--run-name", "full_bidir_age8_v1"],
            ["tracklet_stitcher.py", "--sequence", sequence,
             "--source-run", "full_bidir_age8_v1", "--run-name", "full_stitch_v1",
             "--max-gap", "12", "--history", "4", "--max-cost", "1.35",
             "--min-margin", ".08", "--min-appearance", "-.25", "--max-speed", "85"],
            ["promote_id_only.py", "--sequence", sequence],
        ]
        for command in commands:
            subprocess.run([sys.executable, str(here / command[0]), *command[1:]],
                           check=True, env=env)
    if set(sequences) == set(config["sequences"]):
        subprocess.run([sys.executable, str(here / "promote_id_only.py"), "--aggregate"],
                       check=True, env=env)
    return str((config_path.parent / config["workspace_root"] /
                "outputs/optimized_v1/11_id_only_final/annotations/outdoor").resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--sequences", nargs="+")
    args = parser.parse_args()
    print(run(args.config, args.sequences))
