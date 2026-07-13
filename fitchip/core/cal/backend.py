"""The CAL contract: dataclasses + the CompilerBackend ABC.

This module is the stable public interface of FitChip. Every adapter (TFLM,
TVM, ONNX Runtime, TensorRT...) implements CompilerBackend; the GUI, CLI and
orchestrator only ever talk to this interface and never to a concrete backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ModelFormat(str, Enum):
    ONNX = "onnx"
    TFLITE = "tflite"
    PYTORCH = "pt"          # TorchScript archive (.pt / .pth)
    PT2 = "pt2"             # torch.export ExportedProgram archive
    PTE = "pte"             # ExecuTorch program
    KERAS = "h5"            # Keras model (.h5 / .hdf5 / .keras)
    SAVED_MODEL = "pb"      # TF SavedModel dir or frozen GraphDef file
    CKPT = "ckpt"           # weights-only checkpoint — rejected at inspection

    @classmethod
    def from_path(cls, path: str) -> "ModelFormat":
        p = Path(path)
        if p.is_dir():
            if (p / "saved_model.pb").is_file():
                return cls.SAVED_MODEL
            raise ValueError(
                f"'{path}' is a directory but not a TF SavedModel "
                "(no saved_model.pb inside)."
            )
        suffix = path.rsplit(".", 1)[-1].lower()
        fmt = _SUFFIX_TO_FORMAT.get(suffix)
        if fmt is None:
            supported = sorted(_SUFFIX_TO_FORMAT)
            raise ValueError(
                f"Cannot infer model format from '{path}'. "
                f"Supported extensions: {', '.join('.' + s for s in supported)}"
            )
        return cls(fmt)


# Several user-facing extensions map onto one canonical format.
_SUFFIX_TO_FORMAT = {
    "onnx": "onnx",
    "tflite": "tflite",
    "pt": "pt",
    "pth": "pt",
    "pt2": "pt2",
    "pte": "pte",
    "h5": "h5",
    "hdf5": "h5",
    "keras": "h5",
    "pb": "pb",
    "ckpt": "ckpt",
}


class ArtifactKind(str, Enum):
    C_SOURCE_PROJECT = "c_source_project"   # ZIP/dir of an ESP-IDF/PlatformIO project
    SHARED_LIB = "shared_lib"               # .so for Linux edge targets
    SERIALIZED_MODEL = "serialized_model"   # optimized .onnx/.engine


class ErrorCode(str, Enum):
    """Normalized error codes shared across all backends.

    These are the vocabulary of the future AI error agent — keep them stable.
    """

    OP_UNSUPPORTED = "OP_UNSUPPORTED"
    OOM_ARENA = "OOM_ARENA"
    OOM_FLASH = "OOM_FLASH"
    CONVERT_FAIL = "CONVERT_FAIL"
    QUANTIZE_FAIL = "QUANTIZE_FAIL"
    QUANT_UNSUPPORTED = "QUANT_UNSUPPORTED"
    FORMAT_UNSUPPORTED = "FORMAT_UNSUPPORTED"
    TARGET_UNSUPPORTED = "TARGET_UNSUPPORTED"
    MODEL_INVALID = "MODEL_INVALID"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    INTERNAL = "INTERNAL"


@dataclass
class TargetProfile:
    """Hardware profile, loaded from the Target Registry (targets/*.yaml)."""

    id: str                             # "esp32s3", "jetson-orin-nano"...
    display_name: str
    isa: str                            # "xtensa-lx7", "armv8", "x86_64"
    ram_kb: int
    flash_kb: int | None
    has_os: bool                        # bare-metal vs Linux
    accelerators: list[str] = field(default_factory=list)
    vendor: str = ""
    psram_kb: int | None = None
    toolchains: list[str] = field(default_factory=list)


@dataclass
class CompileRequest:
    model_path: str
    model_format: ModelFormat
    target: TargetProfile
    quantization: str | None = None     # "int8_full", "fp16", None
    calibration_data: str | None = None
    optimize_for: str = "size"          # "size" | "speed"
    options: dict = field(default_factory=dict)  # backend-specific knobs


@dataclass
class NormalizedError:
    """Cross-backend normalized error — the raw compiler stderr stays in `raw`,
    while `code`/`message`/`hints` are what users (and later the AI agent) see."""

    code: str                           # an ErrorCode value
    message: str                        # human-friendly explanation
    raw: str = ""                       # original compiler stderr
    hints: list[str] = field(default_factory=list)


@dataclass
class CompileResult:
    success: bool
    artifacts: list[dict] = field(default_factory=list)  # [{kind, path, sha256}]
    report: dict = field(default_factory=dict)  # sizes, arena estimate, op coverage...
    logs: str = ""                      # raw logs — future input for the AI error agent
    error: NormalizedError | None = None


class CompilerBackend(ABC):
    """Every adapter (TFLM, TVM, ORT, TensorRT...) implements this class."""

    @abstractmethod
    def capabilities(self) -> dict:
        """Declared abilities, read from the manifest: input/output formats,
        supported targets, ops, quantization modes."""

    @abstractmethod
    def validate(self, req: CompileRequest, model_meta: dict) -> list[NormalizedError]:
        """Pre-compile checks: op coverage, memory forecast. Cheap; runs on the
        orchestrator (outside Docker), a few hundred ms at most."""

    @abstractmethod
    def compile(self, req: CompileRequest, workspace: str) -> CompileResult:
        """The real compilation. In wave 2 this runs inside the backend's own
        Docker container; the MVP TFLM backend runs in-process."""

    @abstractmethod
    def estimate(self, req: CompileRequest, model_meta: dict) -> dict:
        """Fast estimate (arena size, flash) shown before the user hits compile."""
