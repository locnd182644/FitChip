"""TensorFlow graph inspector (.pb frozen GraphDef or SavedModel directory).

Parsing either container needs the TensorFlow protobuf schemas, which only
ship with tensorflow itself — so full inspection is gated on the `quantize`
extra and degrades to stat-only metadata without it.
"""

from __future__ import annotations

from pathlib import Path

from fitchip.core.cal.backend import ModelFormat
from fitchip.core.inspector.base import ModelMeta, stat_only_meta

_DEFERRED = (
    "Arena estimate and exact op compatibility are computed after the "
    "tensorflow -> tflite conversion step."
)


def inspect_saved_model(path: Path) -> ModelMeta:
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        return stat_only_meta(
            path,
            ModelFormat.SAVED_MODEL,
            warnings=[
                "tensorflow is not installed — op details unavailable "
                "(pip install 'fitchip[quantize]'). " + _DEFERRED,
            ],
            file_size_bytes=_total_size(path),
        )

    if path.is_dir():
        graph_def = _saved_model_graph_def(path)
        file_size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    else:
        graph_def = _frozen_graph_def(path)
        file_size = path.stat().st_size

    op_counts: dict[str, int] = {}
    weights_bytes = 0
    inputs = []
    for node in graph_def.node:
        if node.op in ("Const", "NoOp"):
            if node.op == "Const":
                tensor = node.attr["value"].tensor
                weights_bytes += len(tensor.tensor_content)
            continue
        if node.op == "Placeholder":
            inputs.append({"name": node.name, "shape": [], "dtype": "unknown"})
            continue
        op_counts[node.op] = op_counts.get(node.op, 0) + 1

    return ModelMeta(
        format=ModelFormat.SAVED_MODEL.value,
        file_size_bytes=file_size,
        num_ops=sum(op_counts.values()),
        op_counts=op_counts,
        inputs=inputs,
        outputs=[],
        weights_bytes=weights_bytes,
        is_quantized=None,
        warnings=[
            "Op names are TensorFlow graph ops; exact TFLite op mapping is "
            "checked after the tensorflow -> tflite conversion step."
        ],
    )


def _saved_model_graph_def(path: Path):
    from tensorflow.core.protobuf import saved_model_pb2

    sm = saved_model_pb2.SavedModel()
    sm.ParseFromString((path / "saved_model.pb").read_bytes())
    if not sm.meta_graphs:
        raise ValueError(f"'{path}' contains an empty SavedModel.")
    return sm.meta_graphs[0].graph_def


def _frozen_graph_def(path: Path):
    from tensorflow.core.framework import graph_pb2

    gd = graph_pb2.GraphDef()
    try:
        gd.ParseFromString(path.read_bytes())
    except Exception as exc:
        raise ValueError(
            f"'{path.name}' is not a frozen TensorFlow GraphDef. If it came "
            "from a SavedModel, pass the SavedModel directory instead."
        ) from exc
    return gd


def _total_size(path: Path) -> int:
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size
