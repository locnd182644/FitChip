"""Backend manifest loading.

The Selection Engine never hard-codes knowledge about a compiler: everything
it needs to filter and rank backends is declared in the backend's
manifest.yaml. Updating op tables or priorities requires no code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fitchip.core.cal.backend import TargetProfile


@dataclass
class BackendManifest:
    id: str
    display_name: str
    input_formats: list[str]
    output_artifacts: list[str]
    targets: list[dict]                     # [{match: {...}}] rules
    quantization: list[str | None]
    priority: int = 0
    convertible_from: list[str] = field(default_factory=list)
    ops_supported_file: str | None = None
    docker_image: str | None = None         # unused in MVP (in-process execution)
    supports_autotune: bool = False
    timeout_s: int = 600

    def matches_target(self, target: TargetProfile) -> bool:
        """A target matches when at least one `targets` rule matches.

        A rule matches when every key equals the target's attribute; list
        values mean "any of". Example: {has_os: true, isa: [armv8, x86_64]}.
        """
        for rule in self.targets:
            match = rule.get("match", {})
            if all(_attr_matches(target, key, want) for key, want in match.items()):
                return True
        return False

    def supports_quantization(self, quantization: str | None) -> bool:
        wanted = quantization if quantization is not None else "none"
        normalized = ["none" if q in (None, "none") else q for q in self.quantization]
        return wanted in normalized


def _attr_matches(target: TargetProfile, key: str, want) -> bool:
    actual = getattr(target, key, None)
    if isinstance(want, list):
        return actual in want
    return actual == want


def load_manifest(path: str | Path) -> BackendManifest:
    data = yaml.safe_load(Path(path).read_text())
    known = {f for f in BackendManifest.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown manifest keys in {path}: {sorted(unknown)}")
    return BackendManifest(**data)
