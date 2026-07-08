"""Shared fixtures.

`tiny_tflite` builds a minimal-but-valid .tflite flatbuffer (one float32 ADD
op) directly with the flatbuffers builder — no TensorFlow required, so the
whole suite runs against the light core install.
"""

from __future__ import annotations

import struct
from pathlib import Path

import flatbuffers
import pytest
import tflite


def build_tiny_tflite(path: Path, opcode: int = None, quantized: bool = False) -> Path:
    opcode = tflite.BuiltinOperator.ADD if opcode is None else opcode
    b = flatbuffers.Builder(1024)
    dtype = 9 if quantized else 0  # INT8 vs FLOAT32

    tflite.BufferStart(b)
    empty_buf = tflite.BufferEnd(b)
    dvec = b.CreateByteVector(struct.pack("<4b", 1, 1, 1, 1) if quantized
                              else struct.pack("<4f", 1.0, 1.0, 1.0, 1.0))
    tflite.BufferStart(b)
    tflite.BufferAddData(b, dvec)
    const_buf = tflite.BufferEnd(b)

    def make_tensor(name: str, buffer_idx: int):
        nm = b.CreateString(name)
        tflite.TensorStartShapeVector(b, 2)
        b.PrependInt32(4)
        b.PrependInt32(1)
        shape = b.EndVector()
        tflite.TensorStart(b)
        tflite.TensorAddShape(b, shape)
        tflite.TensorAddType(b, dtype)
        tflite.TensorAddBuffer(b, buffer_idx)
        tflite.TensorAddName(b, nm)
        return tflite.TensorEnd(b)

    t_in = make_tensor("input", 0)
    t_const = make_tensor("weights", 1)
    t_out = make_tensor("output", 0)

    tflite.OperatorStartInputsVector(b, 2)
    b.PrependInt32(1)
    b.PrependInt32(0)
    op_inputs = b.EndVector()
    tflite.OperatorStartOutputsVector(b, 1)
    b.PrependInt32(2)
    op_outputs = b.EndVector()
    tflite.OperatorStart(b)
    tflite.OperatorAddOpcodeIndex(b, 0)
    tflite.OperatorAddInputs(b, op_inputs)
    tflite.OperatorAddOutputs(b, op_outputs)
    op = tflite.OperatorEnd(b)

    tflite.SubGraphStartTensorsVector(b, 3)
    b.PrependUOffsetTRelative(t_out)
    b.PrependUOffsetTRelative(t_const)
    b.PrependUOffsetTRelative(t_in)
    tensors = b.EndVector()
    tflite.SubGraphStartInputsVector(b, 1)
    b.PrependInt32(0)
    g_inputs = b.EndVector()
    tflite.SubGraphStartOutputsVector(b, 1)
    b.PrependInt32(2)
    g_outputs = b.EndVector()
    tflite.SubGraphStartOperatorsVector(b, 1)
    b.PrependUOffsetTRelative(op)
    ops = b.EndVector()
    tflite.SubGraphStart(b)
    tflite.SubGraphAddTensors(b, tensors)
    tflite.SubGraphAddInputs(b, g_inputs)
    tflite.SubGraphAddOutputs(b, g_outputs)
    tflite.SubGraphAddOperators(b, ops)
    subgraph = tflite.SubGraphEnd(b)

    tflite.OperatorCodeStart(b)
    tflite.OperatorCodeAddDeprecatedBuiltinCode(b, min(opcode, 127))
    tflite.OperatorCodeAddBuiltinCode(b, opcode)
    opcode_off = tflite.OperatorCodeEnd(b)

    tflite.ModelStartOperatorCodesVector(b, 1)
    b.PrependUOffsetTRelative(opcode_off)
    opcodes = b.EndVector()
    tflite.ModelStartSubgraphsVector(b, 1)
    b.PrependUOffsetTRelative(subgraph)
    subgraphs = b.EndVector()
    tflite.ModelStartBuffersVector(b, 2)
    b.PrependUOffsetTRelative(const_buf)
    b.PrependUOffsetTRelative(empty_buf)
    buffers = b.EndVector()

    tflite.ModelStart(b)
    tflite.ModelAddVersion(b, 3)
    tflite.ModelAddOperatorCodes(b, opcodes)
    tflite.ModelAddSubgraphs(b, subgraphs)
    tflite.ModelAddBuffers(b, buffers)
    model = tflite.ModelEnd(b)
    b.Finish(model, file_identifier=b"TFL3")
    path.write_bytes(b.Output())
    return path


@pytest.fixture
def tiny_tflite(tmp_path: Path) -> Path:
    return build_tiny_tflite(tmp_path / "tiny.tflite")


@pytest.fixture
def tiny_tflite_int8(tmp_path: Path) -> Path:
    return build_tiny_tflite(tmp_path / "tiny_int8.tflite", quantized=True)


@pytest.fixture
def tiny_onnx(tmp_path: Path) -> Path:
    """Minimal ONNX model (one Relu op) — onnx is a core dependency."""
    import onnx
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])
    graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path
