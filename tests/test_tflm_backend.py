from pathlib import Path

import pytest

from fitchip.backends.tflm.adapter import TflmBackend
from fitchip.core.cal.backend import CompileRequest, ModelFormat, TargetProfile


@pytest.fixture
def backend() -> TflmBackend:
    return TflmBackend()


def _target(**overrides) -> TargetProfile:
    base = dict(id="esp32s3", display_name="ESP32-S3", isa="xtensa-lx7",
                ram_kb=512, flash_kb=8192, has_os=False,
                accelerators=["esp-nn-simd"], psram_kb=8192)
    base.update(overrides)
    return TargetProfile(**base)


def _req(model_path="m.tflite", **overrides) -> CompileRequest:
    base = dict(model_path=model_path, model_format=ModelFormat.TFLITE, target=_target())
    base.update(overrides)
    return CompileRequest(**base)


def _meta(**overrides) -> dict:
    base = dict(
        format="tflite", file_size_bytes=100_000, num_ops=4,
        op_counts={"CONV_2D": 2, "FULLY_CONNECTED": 1, "SOFTMAX": 1},
        inputs=[], outputs=[], weights_bytes=90_000,
        intermediate_peak_bytes=40_000, is_quantized=True,
        custom_ops=[], warnings=[],
    )
    base.update(overrides)
    return base


def test_validate_clean_model(backend):
    assert backend.validate(_req(), _meta()) == []


def test_validate_flags_unsupported_op(backend):
    errors = backend.validate(_req(), _meta(op_counts={"CONV_2D": 1, "GELU": 2}))
    assert errors[0].code == "OP_UNSUPPORTED"
    assert "GELU" in errors[0].message
    assert any("GELU" in h for h in errors[0].hints)  # substitution suggested


def test_validate_flags_arena_oom(backend):
    errors = backend.validate(
        _req(target=_target(ram_kb=64)), _meta(intermediate_peak_bytes=10_000_000)
    )
    codes = [e.code for e in errors]
    assert "OOM_ARENA" in codes
    # psram hint surfaces for boards that have it
    oom = next(e for e in errors if e.code == "OOM_ARENA")
    assert any("PSRAM" in h for h in oom.hints)


def test_validate_warns_on_random_calibration(backend):
    errors = backend.validate(
        _req(quantization="int8_full"), _meta(is_quantized=False)
    )
    assert any(e.code == "WARNING" and "calibration" in e.message for e in errors)


def test_validate_skips_op_check_for_non_native_format(backend):
    # ONNX op names (Relu, Conv...) must not be matched against the TFLite op
    # table — op compatibility is re-checked by compile() after conversion.
    errors = backend.validate(
        _req(model_path="m.onnx", model_format=ModelFormat.ONNX),
        _meta(format="onnx", num_ops=1, op_counts={"Relu": 1},
              intermediate_peak_bytes=None, is_quantized=None),
    )
    assert all(e.code == "WARNING" for e in errors), errors
    assert any("after conversion" in e.message for e in errors)


def test_estimate_omits_op_coverage_for_non_native_format(backend):
    est = backend.estimate(
        _req(model_path="m.onnx", model_format=ModelFormat.ONNX),
        _meta(format="onnx", num_ops=1, op_counts={"Relu": 1},
              intermediate_peak_bytes=None, is_quantized=None),
    )
    assert "op_coverage" not in est  # engine defaults to 1.0
    assert est["esp_nn_accelerated_ops"] == []


def test_estimate_op_coverage(backend):
    est = backend.estimate(_req(), _meta(op_counts={"CONV_2D": 3, "GELU": 1}, num_ops=4))
    assert est["op_coverage"] == pytest.approx(0.75)


def test_estimate_esp_nn_ops_require_accelerator(backend):
    est = backend.estimate(_req(target=_target(accelerators=[])), _meta())
    assert est["esp_nn_accelerated_ops"] == []
    est = backend.estimate(_req(), _meta())
    assert "CONV_2D" in est["esp_nn_accelerated_ops"]


def test_compile_generates_buildable_project(backend, tiny_tflite, tmp_path):
    req = _req(model_path=str(tiny_tflite), options={"out_dir": str(tmp_path / "out")})
    result = backend.compile(req, str(tmp_path))
    assert result.success, result.error
    project = Path(result.artifacts[0]["path"])
    for expected in [
        "CMakeLists.txt", "platformio.ini", "README.md", "NOTICE",
        "main/main.cc", "main/model_data.cc", "main/model_data.h",
        "main/CMakeLists.txt", "main/idf_component.yml",
    ]:
        assert (project / expected).is_file(), f"missing {expected}"

    main_cc = (project / "main/main.cc").read_text()
    assert "resolver.AddAdd();" in main_cc          # only the op the model uses
    assert "MicroMutableOpResolver<1>" in main_cc
    model_cc = (project / "main/model_data.cc").read_text()
    assert f"g_model_data_len = {tiny_tflite.stat().st_size}" in model_cc


def test_compile_rejects_quantizing_float_tflite(backend, tiny_tflite, tmp_path):
    req = _req(model_path=str(tiny_tflite), quantization="int8_full",
               options={"out_dir": str(tmp_path / "out")})
    result = backend.compile(req, str(tmp_path))
    assert not result.success
    assert result.error.code == "QUANTIZE_FAIL"
    assert result.error.hints


def test_compile_accepts_already_quantized_model(backend, tiny_tflite_int8, tmp_path):
    req = _req(model_path=str(tiny_tflite_int8), quantization="int8_full",
               options={"out_dir": str(tmp_path / "out")})
    result = backend.compile(req, str(tmp_path))
    assert result.success, result.error
    assert result.report["quantized"] is True
