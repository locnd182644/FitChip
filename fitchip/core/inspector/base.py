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


def stat_only_meta(
    path: Path, fmt: ModelFormat, warnings: list[str], file_size_bytes: int | None = None
) -> ModelMeta:
    """Minimal metadata when a format's full inspector needs an optional
    dependency that is not installed. Downstream treats the model like a
    pre-conversion format: op checks are deferred, memory fit is neutral."""
    return ModelMeta(
        format=fmt.value,
        file_size_bytes=path.stat().st_size if file_size_bytes is None else file_size_bytes,
        num_ops=0,
        op_counts={},
        inputs=[],
        outputs=[],
        weights_bytes=0,
        warnings=warnings,
    )


def inspect_model(model_path: str | Path, model_format: ModelFormat | None = None) -> ModelMeta:
    path = Path(model_path)
    fmt = model_format or ModelFormat.from_path(str(path))
    if fmt == ModelFormat.SAVED_MODEL and path.is_dir():
        pass  # SavedModel is a directory; every other format is a single file
    elif not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    if fmt == ModelFormat.TFLITE:
        from fitchip.core.inspector.tflite import inspect_tflite

        return inspect_tflite(path)
    if fmt == ModelFormat.ONNX:
        from fitchip.core.inspector.onnx import inspect_onnx

        return inspect_onnx(path)
    if fmt == ModelFormat.KERAS:
        from fitchip.core.inspector.keras import inspect_keras

        return inspect_keras(path)
    if fmt == ModelFormat.SAVED_MODEL:
        from fitchip.core.inspector.tf_graph import inspect_saved_model

        return inspect_saved_model(path)
    if fmt == ModelFormat.PYTORCH:
        from fitchip.core.inspector.pytorch import inspect_torchscript

        return inspect_torchscript(path)
    if fmt in (ModelFormat.PT2, ModelFormat.PTE):
        from fitchip.core.inspector.executorch import inspect_executorch

        return inspect_executorch(path, fmt)
    if fmt == ModelFormat.CKPT:
        raise ValueError(
            f"'{path.name}' is a training checkpoint — it contains weights but no "
            "computation graph, so it cannot be compiled directly. Re-export the "
            "full model from your training code: Keras → model.save('model.h5') "
            "or tf.saved_model.save(...); PyTorch → torch.export.save(...) (.pt2) "
            "or torch.onnx.export(...) (.onnx)."
        )
    raise ValueError(f"Inspection for '{fmt.value}' models is not supported.")
