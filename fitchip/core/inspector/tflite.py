"""TFLite model inspection via flatbuffers schema bindings (no TensorFlow).

Also runs a tensor-liveness analysis over the (topological) operator order to
compute the peak of simultaneously-live activation tensors — the main input
to the tensor-arena estimate.
"""

from __future__ import annotations

from pathlib import Path

from fitchip.core.inspector.base import ModelMeta

# Bytes per element for tflite.TensorType values.
_DTYPE_SIZE = {
    0: 4,   # FLOAT32
    1: 2,   # FLOAT16
    2: 4,   # INT32
    3: 1,   # UINT8
    4: 8,   # INT64
    6: 1,   # BOOL
    7: 2,   # INT16
    9: 1,   # INT8
    10: 8,  # FLOAT64
    16: 4,  # UINT32
}

_DTYPE_NAME = {
    0: "float32", 1: "float16", 2: "int32", 3: "uint8", 4: "int64",
    5: "string", 6: "bool", 7: "int16", 8: "complex64", 9: "int8",
    10: "float64", 16: "uint32",
}


def inspect_tflite(path: Path) -> ModelMeta:
    import tflite

    buf = path.read_bytes()
    # Flatbuffers reads are lazy: a corrupt buffer blows up at first access,
    # not at load. Check the file identifier up front and wrap the traversal.
    if len(buf) < 8 or buf[4:8] != b"TFL3":
        raise ValueError(f"'{path}' is not a valid TFLite flatbuffer (missing TFL3 identifier).")
    try:
        return _inspect(tflite, path, buf)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"'{path}' is not a valid TFLite flatbuffer: {exc}") from exc


def _inspect(tflite, path: Path, buf: bytes) -> ModelMeta:
    model = tflite.Model.GetRootAsModel(buf, 0)

    warnings: list[str] = []
    if model.SubgraphsLength() > 1:
        warnings.append(
            f"Model has {model.SubgraphsLength()} subgraphs; only the main subgraph is analyzed."
        )
    graph = model.Subgraphs(0)

    op_counts: dict[str, int] = {}
    custom_ops: list[str] = []
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        opcode = model.OperatorCodes(op.OpcodeIndex())
        # Schema quirk: builtin codes >127 only fit in the new int32 field.
        code = max(opcode.DeprecatedBuiltinCode(), opcode.BuiltinCode())
        if code == tflite.BuiltinOperator.CUSTOM:
            raw_name = opcode.CustomCode()
            name = raw_name.decode() if raw_name else "CUSTOM"
            if name not in custom_ops:
                custom_ops.append(name)
        else:
            name = tflite.opcode2name(code)
        op_counts[name] = op_counts.get(name, 0) + 1

    def tensor_info(idx: int) -> dict:
        t = graph.Tensors(idx)
        shape = [t.Shape(j) for j in range(t.ShapeLength())]
        return {
            "name": (t.Name() or b"").decode(),
            "shape": shape,
            "dtype": _DTYPE_NAME.get(t.Type(), f"type_{t.Type()}"),
        }

    def tensor_bytes(idx: int) -> int:
        t = graph.Tensors(idx)
        elems = 1
        for j in range(t.ShapeLength()):
            dim = t.Shape(j)
            elems *= dim if dim > 0 else 1  # dynamic dim: assume batch 1
        return elems * _DTYPE_SIZE.get(t.Type(), 4)

    def is_constant(idx: int) -> bool:
        buffer = model.Buffers(graph.Tensors(idx).Buffer())
        return buffer.DataLength() > 0

    weights_bytes = sum(
        model.Buffers(b).DataLength() for b in range(model.BuffersLength())
    )

    inputs = [tensor_info(graph.Inputs(i)) for i in range(graph.InputsLength())]
    outputs = [tensor_info(graph.Outputs(i)) for i in range(graph.OutputsLength())]

    # int8/uint8 graph inputs are the fingerprint of full-integer quantization.
    input_dtypes = {i["dtype"] for i in inputs}
    is_quantized = bool(input_dtypes) and input_dtypes <= {"int8", "uint8", "int16"}

    peak = _liveness_peak(graph, tensor_bytes, is_constant)

    return ModelMeta(
        format="tflite",
        file_size_bytes=path.stat().st_size,
        num_ops=graph.OperatorsLength(),
        op_counts=op_counts,
        inputs=inputs,
        outputs=outputs,
        weights_bytes=weights_bytes,
        intermediate_peak_bytes=peak,
        is_quantized=is_quantized,
        custom_ops=custom_ops,
        warnings=warnings,
    )


def _liveness_peak(graph, tensor_bytes, is_constant) -> int:
    """Peak sum of live activation tensors across the operator schedule.

    A tensor lives from the op that produces it (graph inputs: from step 0)
    to the last op that consumes it (graph outputs: to the end). Constants
    live in flash, not the arena, so they are excluded.
    """
    n_ops = graph.OperatorsLength()
    born: dict[int, int] = {}
    last_use: dict[int, int] = {}

    for i in range(graph.InputsLength()):
        born[graph.Inputs(i)] = 0
    for step in range(n_ops):
        op = graph.Operators(step)
        for j in range(op.InputsLength()):
            idx = op.Inputs(j)
            if idx >= 0 and not is_constant(idx):
                last_use[idx] = step
        for j in range(op.OutputsLength()):
            idx = op.Outputs(j)
            born.setdefault(idx, step)
    for i in range(graph.OutputsLength()):
        last_use[graph.Outputs(i)] = n_ops - 1

    peak = 0
    live = 0
    events: dict[int, list[int]] = {}  # step -> tensor deltas applied before it
    for idx, start in born.items():
        end = last_use.get(idx, start)
        size = tensor_bytes(idx)
        events.setdefault(start, []).append(size)
        events.setdefault(end + 1, []).append(-size)
    for step in sorted(events):
        live += sum(events[step])
        peak = max(peak, live)
    return peak
