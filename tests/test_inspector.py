import pytest

from fitchip.core.inspector import inspect_model


def test_tflite_ops_and_io(tiny_tflite):
    meta = inspect_model(tiny_tflite)
    assert meta.format == "tflite"
    assert meta.op_counts == {"ADD": 1}
    assert meta.num_ops == 1
    assert meta.inputs == [{"name": "input", "shape": [1, 4], "dtype": "float32"}]
    assert meta.outputs == [{"name": "output", "shape": [1, 4], "dtype": "float32"}]
    assert meta.is_quantized is False


def test_tflite_weights_and_liveness_peak(tiny_tflite):
    meta = inspect_model(tiny_tflite)
    assert meta.weights_bytes == 16          # 4 x float32 constant
    # input (16 B) and output (16 B) are live simultaneously; the constant
    # lives in flash and must not count toward the arena peak.
    assert meta.intermediate_peak_bytes == 32


def test_int8_model_detected_as_quantized(tiny_tflite_int8):
    meta = inspect_model(tiny_tflite_int8)
    assert meta.is_quantized is True
    assert meta.inputs[0]["dtype"] == "int8"


def test_invalid_file_raises_value_error(tmp_path):
    bogus = tmp_path / "bogus.tflite"
    bogus.write_bytes(b"not a flatbuffer")
    with pytest.raises(ValueError):
        inspect_model(bogus)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_model(tmp_path / "missing.tflite")


def test_unknown_extension_rejected(tmp_path):
    weird = tmp_path / "model.bin"
    weird.write_bytes(b"")
    with pytest.raises(ValueError, match="format"):
        inspect_model(weird)
