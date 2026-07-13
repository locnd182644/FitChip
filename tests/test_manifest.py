from pathlib import Path

import pytest

from fitchip.core.cal.backend import TargetProfile
from fitchip.core.cal.manifest import BackendManifest, load_manifest

TFLM_MANIFEST = Path("fitchip/backends/tflm/manifest.yaml")


def _target(**overrides) -> TargetProfile:
    base = dict(
        id="t", display_name="T", isa="xtensa-lx7", ram_kb=512,
        flash_kb=4096, has_os=False, vendor="espressif",
    )
    base.update(overrides)
    return TargetProfile(**base)


def test_tflm_manifest_loads():
    m = load_manifest(TFLM_MANIFEST)
    assert m.id == "tflm"
    assert m.input_formats == ["tflite"]
    assert m.priority == 100


def test_target_match_espressif_bare_metal_only():
    # Regression: TFLM's codegen only knows ESP-IDF, so its manifest must
    # not claim every bare-metal board — an STM32 (vendor: st) would be
    # selected and then fail INTERNAL at codegen.
    m = load_manifest(TFLM_MANIFEST)
    assert m.matches_target(_target())
    assert not m.matches_target(_target(has_os=True))
    assert not m.matches_target(_target(vendor="st", isa="cortex-m7"))
    assert not m.matches_target(_target(vendor=""))


def test_target_match_list_means_any_of():
    m = BackendManifest(
        id="x", display_name="X", input_formats=["onnx"], output_artifacts=[],
        targets=[{"match": {"has_os": True, "isa": ["armv8", "x86_64"]}}],
        quantization=["none"],
    )
    assert m.matches_target(_target(has_os=True, isa="armv8"))
    assert m.matches_target(_target(has_os=True, isa="x86_64"))
    assert not m.matches_target(_target(has_os=True, isa="riscv32"))
    assert not m.matches_target(_target(has_os=False, isa="armv8"))


def test_quantization_none_aliases():
    m = load_manifest(TFLM_MANIFEST)
    assert m.supports_quantization(None)
    assert m.supports_quantization("int8_full")
    assert not m.supports_quantization("fp16")


def test_unknown_manifest_key_rejected(tmp_path):
    bad = tmp_path / "manifest.yaml"
    bad.write_text(
        "id: x\ndisplay_name: X\ninput_formats: [onnx]\noutput_artifacts: []\n"
        "targets: []\nquantization: [none]\ntypo_field: 1\n"
    )
    with pytest.raises(ValueError, match="typo_field"):
        load_manifest(bad)
