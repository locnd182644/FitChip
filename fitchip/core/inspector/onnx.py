"""ONNX model inspection via the onnx package (no runtime needed)."""

from __future__ import annotations

from pathlib import Path

from fitchip.core.inspector.base import ModelMeta

_ELEM_TYPE_NAME = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
    6: "int32", 7: "int64", 9: "bool", 10: "float16", 11: "float64",
}


def inspect_onnx(path: Path) -> ModelMeta:
    import onnx

    try:
        model = onnx.load(str(path))
    except Exception as exc:
        raise ValueError(f"'{path}' is not a valid ONNX model: {exc}") from exc

    graph = model.graph
    op_counts: dict[str, int] = {}
    custom_ops: list[str] = []
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        if node.domain not in ("", "ai.onnx") and node.domain not in custom_ops:
            custom_ops.append(node.domain)

    initializer_names = {init.name for init in graph.initializer}

    def value_info(vi) -> dict:
        shape = []
        ttype = vi.type.tensor_type
        for dim in ttype.shape.dim:
            shape.append(dim.dim_value if dim.dim_value > 0 else -1)
        return {
            "name": vi.name,
            "shape": shape,
            "dtype": _ELEM_TYPE_NAME.get(ttype.elem_type, f"type_{ttype.elem_type}"),
        }

    inputs = [value_info(vi) for vi in graph.input if vi.name not in initializer_names]
    outputs = [value_info(vi) for vi in graph.output]
    weights_bytes = sum(len(init.raw_data) for init in graph.initializer)

    warnings = []
    if any(-1 in i["shape"] for i in inputs):
        warnings.append(
            "Model has dynamic input dimensions; they are treated as batch=1 "
            "during conversion and estimation."
        )

    return ModelMeta(
        format="onnx",
        file_size_bytes=path.stat().st_size,
        num_ops=len(graph.node),
        op_counts=op_counts,
        inputs=inputs,
        outputs=outputs,
        weights_bytes=weights_bytes,
        # Arena depends on the post-conversion TFLite graph; unknown here.
        intermediate_peak_bytes=None,
        is_quantized=None,
        custom_ops=custom_ops,
        warnings=warnings,
    )
