"""Model Inspector — parses a model file into format-neutral metadata.

Everything downstream (selection, validation, estimation) works off
ModelMeta, never off the raw model, so backends stay format-agnostic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from fitchip.core.cal.backend import ModelFormat


@dataclass
class ModelMeta:
    format: str                         # ModelFormat value
    file_size_bytes: int
    num_ops: int
    op_counts: dict[str, int]           # {"CONV_2D": 12, "SOFTMAX": 1, ...}
    inputs: list[dict]                  # [{name, shape, dtype}]
    outputs: list[dict]
    weights_bytes: int
    # Peak of live intermediate/input/output tensors over the execution order
    # (liveness analysis). This is the core of the arena estimate. None when
    # it cannot be computed for this format (e.g. ONNX before conversion).
    intermediate_peak_bytes: int | None = None
    is_quantized: bool | None = None
    custom_ops: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def op_names(self) -> set[str]:
        return set(self.op_counts)

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_model(model_path: str | Path, model_format: ModelFormat | None = None) -> ModelMeta:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    fmt = model_format or ModelFormat.from_path(str(path))
    if fmt == ModelFormat.TFLITE:
        from fitchip.core.inspector.tflite import inspect_tflite

        return inspect_tflite(path)
    if fmt == ModelFormat.ONNX:
        from fitchip.core.inspector.onnx import inspect_onnx

        return inspect_onnx(path)
    raise NotImplementedError(
        f"Inspection for '{fmt.value}' models is not implemented yet. "
        "Export your PyTorch model to ONNX first (torch.onnx.export)."
    )
