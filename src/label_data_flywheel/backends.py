"""隔离不同模型环境；参数以argv传递，拒绝shell拼接。"""

import os
from pathlib import Path
import subprocess
import sys


def repository_root():
    configured = os.environ.get("BEE_FLYWHEEL_ROOT")
    root = Path(configured) if configured else Path(__file__).resolve().parents[2]
    if not (root / "legacy").exists():
        raise FileNotFoundError("请从完整仓库运行或设置BEE_FLYWHEEL_ROOT")
    return root


def backend_command(name, config, arguments):
    root = repository_root()
    entry = config["backends"][name]
    script = Path(entry["script"])
    script = script if script.is_absolute() else root / script
    if not script.is_file():
        raise FileNotFoundError(script)
    return [
        str(v).replace("{root}", str(root))
        for v in [
            entry.get("python", sys.executable),
            str(script),
            *entry.get("defaults", []),
            *arguments,
        ]
    ]


def run_backend(name, config, arguments, dry_run=False):
    argv = backend_command(name, config, arguments)
    if dry_run:
        return {"argv": argv, "executed": False}
    root = repository_root()
    entry = config["backends"][name]
    env = os.environ.copy()
    env.update(
        {
            k: str(v).replace("{root}", str(root))
            for k, v in entry.get("env", {}).items()
        }
    )
    cwd = Path(str(entry.get("cwd", root)).replace("{root}", str(root)))
    if not cwd.is_absolute():
        cwd = root / cwd
    subprocess.run(argv, check=True, env=env, cwd=cwd)
    return {"argv": argv, "executed": True}


def trex_visual(image, references, token_env="TREX2_API_TOKEN"):
    # 官方API wrapper：https://github.com/IDEA-Research/T-Rex/tree/trex2
    from .open_set import DDSClient

    return DDSClient(token_env=token_env).visual(image, references)
