"""Verified sample models (`fitchip samples pull <name>`)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import yaml

_REGISTRY = Path(__file__).parent / "registry.yaml"


def registry() -> dict[str, dict]:
    return yaml.safe_load(_REGISTRY.read_text())["samples"]


def pull(name: str, dest_dir: str | Path = ".") -> Path:
    samples = registry()
    if name not in samples:
        raise KeyError(
            f"Unknown sample '{name}'. Available: {', '.join(sorted(samples))}"
        )
    entry = samples[name]
    dest = Path(dest_dir) / entry["filename"]
    urllib.request.urlretrieve(entry["url"], dest)  # noqa: S310 — pinned https URLs
    return dest
