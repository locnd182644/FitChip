from fitchip.core.cal.backend import ModelFormat
from fitchip.core.convert.chain import ConverterChain


def test_same_format_is_zero_hops():
    chain = ConverterChain()
    assert chain.shortest_path(ModelFormat.TFLITE, ModelFormat.TFLITE) == []
    assert chain.hops(ModelFormat.TFLITE, ModelFormat.TFLITE) == 0


def test_onnx_to_tflite_is_one_hop():
    assert ConverterChain().hops(ModelFormat.ONNX, ModelFormat.TFLITE) == 1


def test_unreachable_returns_none():
    chain = ConverterChain()
    assert chain.hops(ModelFormat.TFLITE, ModelFormat.ONNX) is None
    assert chain.shortest_path(ModelFormat.PYTORCH, ModelFormat.TFLITE) is None


def test_registered_edge_enables_multi_hop_path():
    chain = ConverterChain()
    chain.register(ModelFormat.PYTORCH, ModelFormat.ONNX, lambda p, w, r=None: p)
    path = chain.shortest_path(ModelFormat.PYTORCH, ModelFormat.TFLITE)
    assert path == [
        (ModelFormat.PYTORCH, ModelFormat.ONNX),
        (ModelFormat.ONNX, ModelFormat.TFLITE),
    ]
