"""Run provenance, replacing the old `dict(os.environ)` dump (which bloated every result file
and could leak credentials). Only an explicit allow-list is recorded."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

SLURM_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_PARTITION",
    "SLURM_CPUS_PER_TASK",
)


@lru_cache(maxsize=8)
def _git_info(directory: str) -> dict[str, Any] | None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except Exception:
        return None


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def collect_provenance(
    packages: tuple[str, ...] = (), git_dir: Path | None = None
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": {p: _version(p) for p in packages},
        "slurm": {k: os.environ[k] for k in SLURM_KEYS if k in os.environ},
    }
    if git_dir is not None:
        info["git"] = _git_info(str(git_dir))
    return info
