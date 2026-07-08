"""Compiler Abstraction Layer (CAL) — the single contract every backend implements.

Adding a compiler = one adapter class + one manifest.yaml. No core changes.
"""

from fitchip.core.cal.backend import (
    ArtifactKind,
    CompileRequest,
    CompileResult,
    CompilerBackend,
    ErrorCode,
    ModelFormat,
    NormalizedError,
    TargetProfile,
)
from fitchip.core.cal.manifest import BackendManifest, load_manifest
from fitchip.core.cal.registry import BackendRegistry

__all__ = [
    "ArtifactKind",
    "BackendManifest",
    "BackendRegistry",
    "CompileRequest",
    "CompileResult",
    "CompilerBackend",
    "ErrorCode",
    "ModelFormat",
    "NormalizedError",
    "TargetProfile",
    "load_manifest",
]
