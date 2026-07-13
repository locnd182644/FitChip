"""Torch-gated lane: real .pt2 export -> lower -> STM32 project.

Runs only with `pip install -e '.[executorch]'`. The e2e INT8 path and the
exact-arena assertion live here because they need real torch + executorch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("executorch")

from fitchip.backends.executorch.adapter import ExecutorchBackend  # noqa: E402
from fitchip.core.cal.backend import CompileRequest, ModelFormat, TargetProfile  # noqa: E402
from fitchip.core.inspector import inspect_model  # noqa: E402


class TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return torch.relu(self.fc(x))


@pytest.fixture
def tiny_pt2(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.pt2"
    ep = torch.export.export(TinyNet().eval(), (torch.zeros(1, 4),))
    torch.export.save(ep, str(path))
    return path


def _req(model_path: str, **overrides) -> CompileRequest:
    base = dict(
        model_path=model_path,
        model_format=ModelFormat.PT2,
        target=TargetProfile(
            id="stm32f746", display_name="STM32F746", isa="cortex-m7",
            ram_kb=320, flash_kb=1024, has_os=False, vendor="st",
            accelerators=["cmsis-nn"], toolchains=["cmake"],
        ),
    )
    base.update(overrides)
    return CompileRequest(**base)


def test_pt2_compiles_to_stm32_project(tiny_pt2, tmp_path):
    backend = ExecutorchBackend()
    req = _req(str(tiny_pt2), options={"out_dir": str(tmp_path / "out")})
    result = backend.compile(req, str(tmp_path))
    assert result.success, result.error

    project = Path(result.artifacts[0]["path"])
    model_c = (project / "model_pte.c").read_text()
    assert "g_model_pte" in model_c

    # The lowered program embeds a memory plan -> the arena number is exact
    # and the runner must not carry the PLACEHOLDER warning.
    assert result.report["arena_exact"] is True
    assert result.report["arena_kb"] is not None
    main_cpp = (project / "main.cpp").read_text()
    assert "EXACT" in main_cpp and "PLACEHOLDER" not in main_cpp


def test_pt2_int8_pt2e_flow(tiny_pt2, tmp_path):
    import numpy as np

    calib = tmp_path / "calib.npy"
    np.save(calib, np.random.rand(8, 4).astype("float32"))

    backend = ExecutorchBackend()
    req = _req(str(tiny_pt2), quantization="int8_full",
               calibration_data=str(calib),
               options={"out_dir": str(tmp_path / "out")})
    result = backend.compile(req, str(tmp_path))
    assert result.success, result.error
    assert result.report["quantized"] is True
    assert "quantize" in result.logs.lower() or "PT2E" in result.logs


def test_lowered_pte_deep_inspection_arena_matches_report(tiny_pt2, tmp_path):
    # Lower once via the backend, then deep-inspect the emitted .pte: the
    # inspector's planned arena must agree with the compile report.
    backend = ExecutorchBackend()
    req = _req(str(tiny_pt2), options={"out_dir": str(tmp_path / "out")})
    result = backend.compile(req, str(tmp_path))
    assert result.success, result.error

    model_c = (Path(result.artifacts[0]["path"]) / "model_pte.c").read_text()
    hex_bytes = [int(tok, 16) for tok in
                 __import__("re").findall(r"0x([0-9a-f]{2})", model_c)]
    pte = tmp_path / "roundtrip.pte"
    pte.write_bytes(bytes(hex_bytes))

    meta = inspect_model(pte)
    assert meta.format == "pte"
    if meta.intermediate_peak_bytes is not None:  # deep deserializer available
        assert meta.intermediate_peak_bytes // 1024 == result.report["arena_kb"]
        assert meta.num_ops > 0
