"""Selection Engine tests with fake backends — proves that ranking and
filtering are driven purely by manifests, not by backend identity."""

from __future__ import annotations

from fitchip.core.cal.backend import (
    CompileRequest,
    CompileResult,
    CompilerBackend,
    ModelFormat,
    NormalizedError,
    TargetProfile,
)
from fitchip.core.cal.manifest import BackendManifest
from fitchip.core.convert.chain import ConverterChain
from fitchip.core.inspector.base import ModelMeta
from fitchip.core.selection.engine import SelectionEngine


class FakeBackend(CompilerBackend):
    def __init__(self, manifest: BackendManifest, estimate: dict | None = None,
                 errors: list[NormalizedError] | None = None):
        self.manifest = manifest
        self._estimate = estimate or {"op_coverage": 1.0, "arena_kb": 50, "flash_kb": 500}
        self._errors = errors or []

    def capabilities(self):
        return {"id": self.manifest.id}

    def validate(self, req, model_meta):
        return self._errors

    def compile(self, req, workspace):
        return CompileResult(success=True)

    def estimate(self, req, model_meta):
        return self._estimate


class FakeRegistry:
    def __init__(self, backends):
        self._backends = {b.manifest.id: b for b in backends}

    def all(self):
        return list(self._backends.values())

    def get(self, backend_id):
        return self._backends[backend_id]


def _manifest(id: str, priority: int = 0, input_formats=None, quantization=None,
              targets=None) -> BackendManifest:
    return BackendManifest(
        id=id,
        display_name=id,
        input_formats=input_formats or ["tflite"],
        output_artifacts=["c_source_project"],
        targets=targets or [{"match": {"has_os": False}}],
        quantization=quantization or ["int8_full", "none"],
        priority=priority,
    )


def _target(**overrides) -> TargetProfile:
    base = dict(id="mcu", display_name="MCU", isa="xtensa-lx7", ram_kb=512,
                flash_kb=4096, has_os=False)
    base.update(overrides)
    return TargetProfile(**base)


def _req(**overrides) -> CompileRequest:
    base = dict(model_path="m.tflite", model_format=ModelFormat.TFLITE, target=_target())
    base.update(overrides)
    return CompileRequest(**base)


def _meta() -> ModelMeta:
    return ModelMeta(
        format="tflite", file_size_bytes=100_000, num_ops=10,
        op_counts={"CONV_2D": 10}, inputs=[], outputs=[], weights_bytes=90_000,
        intermediate_peak_bytes=40_000,
    )


def test_higher_priority_wins_all_else_equal():
    engine = SelectionEngine(
        FakeRegistry([FakeBackend(_manifest("low", 10)), FakeBackend(_manifest("high", 90))])
    )
    report = engine.select(_req(), _meta())
    assert [c.backend_id for c in report.candidates] == ["high", "low"]
    assert report.best.backend_id == "high"
    assert [c.backend_id for c in report.fallback_chain] == ["low"]


def test_conversion_hops_cost_points():
    # Same priority; one needs onnx->tflite conversion (input is onnx).
    direct = FakeBackend(_manifest("direct", 50, input_formats=["onnx"]))
    needs_convert = FakeBackend(_manifest("convert", 50, input_formats=["tflite"]))
    engine = SelectionEngine(FakeRegistry([direct, needs_convert]), ConverterChain())
    report = engine.select(_req(model_format=ModelFormat.ONNX, model_path="m.onnx"), _meta())
    assert report.best.backend_id == "direct"
    assert report.candidates[1].conversion_hops == 1


def test_unreachable_format_is_rejected():
    onnx_only = FakeBackend(_manifest("onnx_only", input_formats=["onnx"]))
    engine = SelectionEngine(FakeRegistry([onnx_only]), ConverterChain())
    # tflite -> onnx has no converter edge
    report = engine.select(_req(), _meta())
    assert not report.candidates
    assert report.rejected[0][1].code == "FORMAT_UNSUPPORTED"


def test_target_mismatch_is_rejected():
    mcu_only = FakeBackend(_manifest("mcu_only"))
    engine = SelectionEngine(FakeRegistry([mcu_only]))
    report = engine.select(_req(target=_target(has_os=True)), _meta())
    assert report.rejected[0][1].code == "TARGET_UNSUPPORTED"


def test_unsupported_quantization_is_rejected():
    b = FakeBackend(_manifest("b", quantization=["none"]))
    engine = SelectionEngine(FakeRegistry([b]))
    report = engine.select(_req(quantization="int8_full"), _meta())
    assert report.rejected[0][1].code == "QUANT_UNSUPPORTED"


def test_validation_error_rejects_candidate():
    bad = FakeBackend(
        _manifest("bad", 99),
        errors=[NormalizedError(code="OP_UNSUPPORTED", message="nope")],
    )
    good = FakeBackend(_manifest("good", 1))
    engine = SelectionEngine(FakeRegistry([bad, good]))
    report = engine.select(_req(), _meta())
    assert report.best.backend_id == "good"
    assert report.rejected[0][0] == "bad"


def test_validation_warning_does_not_reject():
    b = FakeBackend(
        _manifest("warned"),
        errors=[NormalizedError(code="WARNING", message="heads up")],
    )
    report = SelectionEngine(FakeRegistry([b])).select(_req(), _meta())
    assert report.best.backend_id == "warned"
    assert report.best.warnings[0].message == "heads up"


def test_forced_backend_option_bypasses_others():
    a = FakeBackend(_manifest("a", 99))
    b = FakeBackend(_manifest("b", 1))
    engine = SelectionEngine(FakeRegistry([a, b]))
    report = engine.select(_req(options={"backend": "b"}), _meta())
    assert [c.backend_id for c in report.candidates] == ["b"]
