"""Target Registry — hardware profiles are data, not code.

Adding a new board = dropping one YAML file into this directory (or any
directory passed via `extra_dirs` / the FITCHIP_TARGETS_DIR env var).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from fitchip.core.cal.backend import TargetProfile

_BUILTIN_DIR = Path(__file__).parent


class TargetRegistry:
    def __init__(self, extra_dirs: list[str | Path] | None = None) -> None:
        self._targets: dict[str, TargetProfile] = {}
        dirs: list[Path] = [_BUILTIN_DIR]
        if os.environ.get("FITCHIP_TARGETS_DIR"):
            dirs.append(Path(os.environ["FITCHIP_TARGETS_DIR"]))
        dirs.extend(Path(d) for d in (extra_dirs or []))
        for directory in dirs:
            for path in sorted(directory.glob("*.yaml")):
                profile = _load_profile(path)
                self._targets[profile.id] = profile

    def all(self) -> list[TargetProfile]:
        return sorted(self._targets.values(), key=lambda t: t.id)

    def get(self, target_id: str) -> TargetProfile:
        try:
            return self._targets[target_id]
        except KeyError:
            raise KeyError(
                f"Unknown target '{target_id}'. "
                f"Available targets: {', '.join(sorted(self._targets))}"
            ) from None

    def ids(self) -> list[str]:
        return sorted(self._targets)


def _load_profile(path: Path) -> TargetProfile:
    data = yaml.safe_load(path.read_text())
    known = set(TargetProfile.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown target profile keys in {path}: {sorted(unknown)}")
    return TargetProfile(**data)
