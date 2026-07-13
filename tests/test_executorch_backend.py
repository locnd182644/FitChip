"""ExecuTorch backend tests.

Everything here runs against the light core install: the adapter constructs,
validates, estimates and generates the CMake project without torch/executorch.
The torch-gated lane (.pt2 -> PT2E INT8 -> lower) is exercised in
test_executorch_torch_lane.py behind importorskip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fitchip.backends.executorch.adapter import ExecutorchBackend, _base_op_name
from fitchip.core.cal.backend import CompileRequest, ModelFormat, TargetProfile
from fitchip.core.pipeline import Pipeline
from fitchip.core.targets import TargetRegistry


@pytest.fixture
def backend() -> ExecutorchBackend:
    return ExecutorchBackend()


@pytest.fixture
def tiny_pte(tmp_path: Path) -> Path:
    """Header-valid .pte (ET identifier at offset 4); content is opaque to the
    ungated code paths, which treat the program as a byte blob."""
    path = tmp_path / "tiny.pte"
    path.write_bytes(b"\x00\x00\x00\x00ET13" + b"\x00" * 56)
    return path


def _target(**overrides) -> TargetProfile:
    base = dict(id="stm32f746", display_name="STM32F746", isa="cortex-m7",
                ram_kb=320, flash_kb=1024, has_os=False, vendor="st",
                accelerators=["cmsis-nn"], toolchains=["cmake"])
    base.update(overrides)
    return TargetProfile(**base)


def _req(model_path="m.pte", model_format=ModelFormat.PTE, **overrides) -> CompileRequest:
    base = dict(model_path=model_path, model_format=model_format, target=_target())
    base.update(overrides)
    return CompileRequest(**base)


def _meta(**overrides) -> dict:
    base = dict(
        format="pte", file_size_bytes=64_000, num_ops=3,
        op_counts={"aten::conv2d": 2, "aten::relu": 1},
        inputs=[], outputs=[], weights_bytes=50_000,
        intermediate_peak_bytes=100_000, is_quantized=None,
        custom_ops=[], warnings=[],
    )
    base.update(overrides)
    return base


# ------------------------------------------------------- manifest & targets

def test_manifest_loads(backend):
    m = backend.manifest
    assert m.id == "executorch"
    assert m.input_formats == ["pte", "pt2"]
    assert m.priority == 90            # below TFLM's 100 by design


def test_manifest_matches_cortex_m_only(backend):
    m = backend.manifest
    assert m.matches_target(_target())                            # cortex-m7
    assert m.matches_target(_target(isa="cortex-m4f"))
    assert not m.matches_target(_target(isa="xtensa-lx7"))        # ESP32
    assert not m.matches_target(_target(has_os=True, isa="cortex-m7"))


def test_stm32_targets_registered():
    reg = TargetRegistry()
    assert {"nucleo_f446re", "stm32f746", "stm32h743"} <= set(reg.ids())
    f746 = reg.get("stm32f746")
    assert f746.vendor == "st"
    assert f746.isa == "cortex-m7"
    assert "cmsis-nn" in f746.accelerators


# ----------------------------------------------------------------- validate

def test_validate_clean_pte(backend):
    assert backend.validate(_req(), _meta()) == []


def test_validate_pte_int8_is_quantize_fail(backend):
    # Design decision 3: never post-quantize an already-lowered .pte.
    errors = backend.validate(_req(quantization="int8_full"), _meta())
    assert errors[0].code == "QUANTIZE_FAIL"
    assert any(".pt2" in h for h in errors[0].hints)


def test_validate_pt2_int8_is_allowed_with_calibration_warning(backend):
    errors = backend.validate(
        _req(model_format=ModelFormat.PT2, quantization="int8_full"),
        _meta(format="pt2", intermediate_peak_bytes=None),
    )
    assert all(e.code == "WARNING" for e in errors)
    assert any("calibration" in e.message for e in errors)


def test_validate_unknown_op_warns_never_rejects(backend):
    # Portable-kernel fallback: unlike TFLM, a missing op is a warning.
    errors = backend.validate(
        _req(), _meta(op_counts={"aten::gelu": 1, "aten::conv2d": 1})
    )
    assert [e.code for e in errors] == ["WARNING"]
    assert "gelu" in errors[0].message
    assert any("gelu" in h for h in errors[0].hints)   # substitution suggested


def test_validate_exact_arena_oom(backend):
    errors = backend.validate(
        _req(target=_target(ram_kb=64)), _meta(intermediate_peak_bytes=100_000)
    )
    oom = [e for e in errors if e.code == "OOM_ARENA"]
    assert oom and "exact" in oom[0].message


def test_validate_arena_not_checked_for_pt2(backend):
    # A .pt2 has no memory plan yet — no false OOM before lowering.
    errors = backend.validate(
        _req(model_format=ModelFormat.PT2, target=_target(ram_kb=64)),
        _meta(format="pt2", intermediate_peak_bytes=100_000),
    )
    assert all(e.code != "OOM_ARENA" for e in errors)


# ----------------------------------------------------------------- estimate

def test_estimate_exact_arena_from_pte_plan(backend):
    est = backend.estimate(_req(), _meta(intermediate_peak_bytes=100_000))
    assert est["arena_kb"] == 97
    assert est["arena_exact"] is True


def test_estimate_no_arena_claim_without_plan(backend):
    est = backend.estimate(
        _req(model_format=ModelFormat.PT2), _meta(format="pt2", intermediate_peak_bytes=None)
    )
    assert est["arena_kb"] is None     # honest: no plan, no number
    assert est["arena_exact"] is False


def test_estimate_cmsis_nn_ops_require_accelerator(backend):
    est = backend.estimate(_req(), _meta())
    assert est["cmsis_nn_accelerated_ops"] == ["conv2d", "relu"]
    est = backend.estimate(_req(target=_target(accelerators=[])), _meta())
    assert est["cmsis_nn_accelerated_ops"] == []


def test_base_op_name_normalization():
    assert _base_op_name("aten::conv2d") == "conv2d"
    assert _base_op_name("aten.conv2d.default") == "conv2d"
    assert _base_op_name("aten.linear.out") == "linear"
    assert _base_op_name("_softmax.default") == "softmax"


# ------------------------------------------------------------------ compile

def test_compile_pte_generates_cmake_project(backend, tiny_pte, tmp_path):
    req = _req(model_path=str(tiny_pte), options={"out_dir": str(tmp_path / "out")})
    result = backend.compile(req, str(tmp_path))
    assert result.success, result.error
    project = Path(result.artifacts[0]["path"])
    assert project.name == "stm32f746-project"
    for expected in [
        "CMakeLists.txt", "cmake/arm-none-eabi.cmake", "main.cpp",
        "model_pte.c", "model_pte.h", "README.md", "NOTICE",
    ]:
        assert (project / expected).is_file(), f"missing {expected}"

    cmake = (project / "CMakeLists.txt").read_text()
    assert "FetchContent" in cmake and "pytorch/executorch" in cmake
    assert 'GIT_TAG        v' in cmake                     # pinned, never a branch
    toolchain = (project / "cmake/arm-none-eabi.cmake").read_text()
    assert "-mcpu=cortex-m7" in toolchain
    model_c = (project / "model_pte.c").read_text()
    assert f"g_model_pte_len = {tiny_pte.stat().st_size}" in model_c
    assert result.report["next_steps"]                     # CLI prints these


def test_compile_pte_int8_fails_before_codegen(backend, tiny_pte, tmp_path):
    req = _req(model_path=str(tiny_pte), quantization="int8_full",
               options={"out_dir": str(tmp_path / "out")})
    result = backend.compile(req, str(tmp_path))
    assert not result.success
    assert result.error.code == "QUANTIZE_FAIL"
    assert not (tmp_path / "out").exists()


def test_compile_pt2_without_deps_is_dependency_missing(backend, tmp_path):
    pytest.importorskip("torch")  # covers only the torch-present/executorch-absent case
    try:
        import executorch  # noqa: F401
        pytest.skip("executorch installed — the gated e2e test covers this")
    except ImportError:
        pass
    torch = __import__("torch")

    class Net(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x)

    pt2 = tmp_path / "m.pt2"
    torch.export.save(torch.export.export(Net(), (torch.zeros(1, 4),)), str(pt2))
    req = _req(model_path=str(pt2), model_format=ModelFormat.PT2,
               options={"out_dir": str(tmp_path / "out")})
    result = backend.compile(req, str(tmp_path))
    assert not result.success
    assert result.error.code == "DEPENDENCY_MISSING"
    assert any("executorch" in h for h in result.error.hints)


def test_placeholder_arena_when_plan_unknown(backend, tiny_pte, tmp_path):
    # Stat-only inspection (no executorch): the runner must say PLACEHOLDER,
    # never pass a guess off as the exact plan.
    try:
        import executorch  # noqa: F401
        pytest.skip("executorch installed — deep inspection yields exact numbers")
    except ImportError:
        pass
    req = _req(model_path=str(tiny_pte), options={"out_dir": str(tmp_path / "out")})
    result = backend.compile(req, str(tmp_path))
    assert result.success, result.error
    assert result.report["arena_exact"] is False
    main_cpp = (Path(result.artifacts[0]["path"]) / "main.cpp").read_text()
    assert "PLACEHOLDER" in main_cpp
    # RAM/4 of the 320 KB target
    assert "kActivationBytes = 81920" in main_cpp


# ------------------------------------------------- selection matrix (Phase 2)

def test_selection_matrix_stm32_vs_esp32(tiny_pte, tiny_tflite):
    pipeline = Pipeline()

    # .pte on STM32 -> executorch only; TFLM must not match (vendor: st).
    req = pipeline.build_request(str(tiny_pte), "stm32f746")
    _, selection = pipeline.inspect(req)
    assert [c.backend_id for c in selection.candidates] == ["executorch"]
    assert any(bid == "tflm" for bid, _ in selection.rejected)

    # .tflite on ESP32 -> tflm only; executorch must not match (ISA).
    req = pipeline.build_request(str(tiny_tflite), "esp32s3")
    _, selection = pipeline.inspect(req)
    assert [c.backend_id for c in selection.candidates] == ["tflm"]
    assert any(bid == "executorch" for bid, _ in selection.rejected)


def test_selection_tflite_on_stm32_rejected_everywhere(tiny_tflite):
    # No conversion edge feeds ExecuTorch, and TFLM doesn't serve ST boards:
    # the report must show two clean rejections, not a crash.
    pipeline = Pipeline()
    req = pipeline.build_request(str(tiny_tflite), "stm32f746")
    _, selection = pipeline.inspect(req)
    assert not selection.candidates
    codes = {bid: err.code for bid, err in selection.rejected}
    assert codes["tflm"] == "TARGET_UNSUPPORTED"
    assert codes["executorch"] == "FORMAT_UNSUPPORTED"


def test_cli_lists_new_backend_and_targets():
    from click.testing import CliRunner

    from fitchip.cli.main import cli

    runner = CliRunner()
    assert "stm32f746" in runner.invoke(cli, ["targets"]).output
    out = runner.invoke(cli, ["backends"]).output
    assert "executorch" in out and "tflm" in out


def test_cli_compile_pte_end_to_end(tiny_pte, tmp_path):
    from click.testing import CliRunner

    from fitchip.cli.main import cli

    result = CliRunner().invoke(
        cli,
        ["compile", str(tiny_pte), "--target", "stm32f746",
         "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert "Project generated" in result.output
    assert "cmake" in result.output          # backend-specific next steps
    assert "idf.py" not in result.output     # the old hardcoded ESP-IDF line is gone
    assert (tmp_path / "out" / "stm32f746-project" / "main.cpp").is_file()
