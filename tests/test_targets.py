import pytest

from fitchip.core.targets import TargetRegistry


def test_builtin_targets_load():
    reg = TargetRegistry()
    assert {"esp32", "esp32s3", "esp32c3"} <= set(reg.ids())


def test_profile_fields():
    esp32s3 = TargetRegistry().get("esp32s3")
    assert esp32s3.ram_kb == 512
    assert esp32s3.has_os is False
    assert "esp-nn-simd" in esp32s3.accelerators
    assert esp32s3.psram_kb == 8192


def test_unknown_target_lists_available():
    with pytest.raises(KeyError, match="esp32s3"):
        TargetRegistry().get("nonexistent")


def test_extra_dir_adds_target(tmp_path):
    (tmp_path / "myboard.yaml").write_text(
        "id: myboard\ndisplay_name: My Board\nisa: armv7\n"
        "ram_kb: 128\nflash_kb: 1024\nhas_os: false\n"
    )
    reg = TargetRegistry(extra_dirs=[tmp_path])
    assert reg.get("myboard").ram_kb == 128
