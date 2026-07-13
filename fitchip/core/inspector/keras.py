"""Keras model inspector (.h5 / .hdf5 / .keras).

Both containers carry the architecture as a JSON model config, so the op
list comes from layer class names without loading TensorFlow:
- .keras is a ZIP holding config.json (stdlib only).
- .h5/.hdf5 stores the config in an HDF5 attribute (needs h5py; falls back
  to stat-only metadata when it is not installed).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from fitchip.core.cal.backend import ModelFormat
from fitchip.core.inspector.base import ModelMeta, stat_only_meta

_DEFERRED = (
    "Arena estimate and exact op compatibility are computed after the "
    "keras -> tflite conversion step."
)


def inspect_keras(path: Path) -> ModelMeta:
    if zipfile.is_zipfile(path):
        return _inspect_keras_zip(path)
    return _inspect_h5(path)


def _inspect_keras_zip(path: Path) -> ModelMeta:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if "config.json" not in names:
            raise ValueError(
                f"'{path.name}' is not a Keras v3 archive (no config.json inside)."
            )
        config = json.loads(zf.read("config.json"))
        weights_bytes = sum(
            info.file_size for info in zf.infolist()
            if info.filename.endswith((".h5", ".weights.h5", ".npy"))
        )
    return _meta_from_config(path, config, weights_bytes)


def _inspect_h5(path: Path) -> ModelMeta:
    try:
        import h5py
    except ImportError:
        return stat_only_meta(
            path,
            ModelFormat.KERAS,
            warnings=[
                "h5py is not installed — layer/op details unavailable "
                "(pip install h5py or 'fitchip[quantize]'). " + _DEFERRED,
            ],
        )

    with h5py.File(path, "r") as f:
        raw_config = f.attrs.get("model_config")
        if raw_config is None:
            raise ValueError(
                f"'{path.name}' is an HDF5 file but has no Keras model_config — "
                "is it a weights-only file? Save the full model with model.save()."
            )
        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode("utf-8")
        config = json.loads(raw_config)

        weights_bytes = 0

        def _accumulate(name: str, obj) -> None:
            nonlocal weights_bytes
            if isinstance(obj, h5py.Dataset):
                weights_bytes += obj.size * obj.dtype.itemsize

        if "model_weights" in f:
            f["model_weights"].visititems(_accumulate)
    return _meta_from_config(path, config, weights_bytes)


def _meta_from_config(path: Path, config: dict, weights_bytes: int) -> ModelMeta:
    layers = _collect_layers(config.get("config", config))
    op_counts: dict[str, int] = {}
    for layer in layers:
        cls = layer.get("class_name", "Unknown")
        if cls == "InputLayer":
            continue
        op_counts[cls] = op_counts.get(cls, 0) + 1

    inputs = [
        {
            "name": layer.get("config", {}).get("name", "input"),
            "shape": layer.get("config", {}).get("batch_shape")
            or layer.get("config", {}).get("batch_input_shape")
            or [],
            "dtype": layer.get("config", {}).get("dtype", "float32"),
        }
        for layer in layers
        if layer.get("class_name") == "InputLayer"
    ]

    return ModelMeta(
        format=ModelFormat.KERAS.value,
        file_size_bytes=path.stat().st_size,
        num_ops=sum(op_counts.values()),
        op_counts=op_counts,
        inputs=inputs,
        outputs=[],  # not recoverable from the config without building the model
        weights_bytes=weights_bytes,
        is_quantized=False,  # Keras saves float models; INT8 happens at conversion
        warnings=[
            "Op names are Keras layer classes; exact TFLite op mapping is "
            "checked after the keras -> tflite conversion step."
        ],
    )


def _collect_layers(config: dict) -> list[dict]:
    """Layers of Sequential/Functional models, flattening nested submodels."""
    layers: list[dict] = []
    for layer in config.get("layers", []):
        inner = layer.get("config", {})
        if isinstance(inner, dict) and "layers" in inner:
            layers.extend(_collect_layers(inner))
        else:
            layers.append(layer)
    return layers
