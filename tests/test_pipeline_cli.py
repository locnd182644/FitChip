"""End-to-end tests through the Pipeline and the click CLI."""

from pathlib import Path

from click.testing import CliRunner

from fitchip.cli.main import cli
from fitchip.core.pipeline import Pipeline


def test_pipeline_end_to_end(tiny_tflite, tmp_path):
    pipeline = Pipeline()
    req = pipeline.build_request(str(tiny_tflite), "esp32s3")
    meta, selection = pipeline.inspect(req)
    assert meta.op_counts == {"ADD": 1}
    assert selection.best.backend_id == "tflm"

    result = pipeline.compile(req, tmp_path / "out")
    assert result.success, result.error
    assert result.report["attempts"] == ["tflm: ok"]
    assert Path(result.artifacts[0]["path"]).name == "esp32s3-project"


def test_pipeline_onnx_reaches_selection(tiny_onnx, tmp_path):
    # Regression for A1: ONNX input must not be rejected by the pre-conversion
    # op-check (ONNX op names never match the TFLite op table).
    pipeline = Pipeline()
    req = pipeline.build_request(str(tiny_onnx), "esp32s3")
    meta, selection = pipeline.inspect(req)
    assert [c.backend_id for c in selection.candidates] == ["tflm"]
    assert any("after conversion" in w.message for w in selection.best.warnings)

    result = pipeline.compile(req, tmp_path / "out")
    try:
        import onnx2tf  # noqa: F401
    except ImportError:
        # Without the converter the failure must be clean and actionable,
        # not "no installed backend can compile this model".
        assert not result.success
        assert result.error.code == "DEPENDENCY_MISSING"
    else:
        assert result.success, result.error


def test_cli_compile(tiny_tflite, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["compile", str(tiny_tflite), "--target", "esp32s3",
         "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert "Project generated" in result.output
    assert (tmp_path / "out" / "esp32s3-project" / "main" / "main.cc").is_file()


def test_cli_inspect_with_target(tiny_tflite):
    runner = CliRunner()
    result = runner.invoke(cli, ["inspect", str(tiny_tflite), "--target", "esp32"])
    assert result.exit_code == 0, result.output
    assert "ADD" in result.output
    assert "tflm" in result.output


def test_cli_unknown_target_fails_cleanly(tiny_tflite):
    runner = CliRunner()
    result = runner.invoke(cli, ["compile", str(tiny_tflite), "--target", "nope"])
    assert result.exit_code != 0
    assert "Available targets" in result.output


def test_cli_targets_and_backends():
    runner = CliRunner()
    assert "esp32s3" in runner.invoke(cli, ["targets"]).output
    assert "tflm" in runner.invoke(cli, ["backends"]).output


def test_fail_prints_raw_compiler_error(capsys):
    # Regression: CONVERT_FAIL used to hide the underlying framework error
    # (e.g. Keras 3's "Unknown layer: 'TFOpLambda'"), leaving users blind.
    from fitchip.cli.main import _fail
    from fitchip.core.cal.backend import NormalizedError

    _fail(NormalizedError(
        code="CONVERT_FAIL",
        message="keras -> tflite conversion failed for 'm.h5'.",
        raw="Unknown layer: 'TFOpLambda'. Please ensure ...",
        hints=["Run `fitchip inspect` first."],
    ))
    err = capsys.readouterr().err
    assert "CONVERT_FAIL" in err
    assert "TFOpLambda" in err          # the raw cause is visible
    assert "fitchip inspect" in err


def test_fail_truncates_huge_raw_logs(capsys):
    from fitchip.cli.main import _fail
    from fitchip.core.cal.backend import NormalizedError

    _fail(NormalizedError(code="CONVERT_FAIL", message="x", raw="y" * 10_000))
    err = capsys.readouterr().err
    assert "[…]" in err
    assert len(err) < 5_000
