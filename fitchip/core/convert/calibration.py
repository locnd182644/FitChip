"""Shared calibration-sample loading for INT8 conversion paths.

Every converter that quantizes (onnx2tf, keras->tflite, and later the
ExecuTorch PT2E flow) reads representative samples the same way: a .npy
file, or a directory containing .npy files. Samples are expected to be
already preprocessed (mean=0/std=1 handling is the exporter's job).
"""

from __future__ import annotations

from pathlib import Path


def resolve_npy(calibration_data: str | None) -> Path | None:
    """Path of the first usable .npy sample file, or None."""
    if not calibration_data:
        return None
    path = Path(calibration_data)
    if path.is_dir():
        npys = sorted(path.glob("*.npy"))
        return npys[0] if npys else None
    return path if path.suffix == ".npy" else None


def load_samples(calibration_data: str | None):
    """The samples as a numpy array (first dim = sample index), or None."""
    npy = resolve_npy(calibration_data)
    if npy is None:
        return None
    import numpy as np

    return np.load(str(npy))


def onnx2tf_calibration_arg(model_path: Path, calibration_data: str) -> list | None:
    """Build onnx2tf's [[input_name, npy_path, mean, std], ...] argument.
    Returns None when nothing usable is found — onnx2tf then falls back to its
    built-in random calibration (accuracy warning is raised upstream)."""
    npy = resolve_npy(calibration_data)
    if npy is None:
        return None

    import onnx

    graph = onnx.load(str(model_path)).graph
    initializers = {init.name for init in graph.initializer}
    input_names = [vi.name for vi in graph.input if vi.name not in initializers]
    if not input_names:
        return None
    # mean=0, std=1: samples are expected to be already preprocessed.
    return [[input_names[0], str(npy), 0.0, 1.0]]
