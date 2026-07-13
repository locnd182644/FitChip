"""Tests for the multi-format ingestion layer.

Covers the three honesty groups from the integration plan:
- self-contained formats are inspected (keras zip / h5 / pte header),
- weights-only files (.ckpt, state_dict .pt) are rejected with export
  guidance instead of a traceback,
- everything reaches the pipeline through ModelFormat.from_path and the
  converter-chain edges (h5/pb 1 hop, torchscript 2 hops).

All fixtures are built programmatically (stdlib zip / h5py / raw bytes) —
no TensorFlow or torch needed for the ungated part of the suite.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from fitchip.cli.main import cli
from fitchip.core.cal.backend import ModelFormat
from fitchip.core.cal.quant import normalize_quantize
from fitchip.core.convert.chain import ConverterChain
from fitchip.core.inspector import inspect_model
from fitchip.core.pipeline import Pipeline

# --------------------------------------------------------------- fixtures

_KERAS_CONFIG = {
    "class_name": "Sequential",
    "config": {
        "name": "tiny",
        "layers": [
            {"class_name": "InputLayer",
             "config": {"name": "input_layer", "batch_shape": [None, 4],
                        "dtype": "float32"}},
            {"class_name": "Dense", "config": {"name": "dense", "units": 8}},
            {"class_name": "Dense", "config": {"name": "dense_1", "units": 2}},
        ],
    },
}


@pytest.fixture
def tiny_keras_zip(tmp_path: Path) -> Path:
    """Minimal Keras v3 archive: config.json + a weights blob (stdlib only)."""
    path = tmp_path / "tiny.keras"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("config.json", json.dumps(_KERAS_CONFIG))
        zf.writestr("model.weights.h5", b"\x00" * 64)
    return path


@pytest.fixture
def tiny_h5(tmp_path: Path) -> Path:
    """Legacy .h5 container: model_config attribute + weight datasets."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as f:
        f.attrs["model_config"] = json.dumps(_KERAS_CONFIG)
        grp = f.create_group("model_weights/dense")
        grp.create_dataset("kernel", data=[[0.0] * 2] * 4, dtype="float32")  # 32 B
        grp.create_dataset("bias", data=[0.0] * 2, dtype="float32")          # 8 B
    return path


@pytest.fixture
def state_dict_pt(tmp_path: Path) -> Path:
    """torch.save() zip layout (data.pkl, no constants.pkl/code/) — the
    weights-only case users hit most often."""
    path = tmp_path / "weights.pt"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", b"\x80\x02")
        zf.writestr("archive/version", "3")
    return path


@pytest.fixture
def torchscript_layout_pt(tmp_path: Path) -> Path:
    """torch.jit.save() zip layout — recognizably compilable without torch."""
    path = tmp_path / "scripted.pt"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/constants.pkl", b"\x80\x02")
        zf.writestr("archive/code/__torch__.py", "def forward(self, x): ...")
        zf.writestr("archive/data.pkl", b"\x80\x02")
    return path


@pytest.fixture
def tiny_pte(tmp_path: Path) -> Path:
    """8+ bytes with the ExecuTorch flatbuffer identifier at offset 4."""
    path = tmp_path / "tiny.pte"
    path.write_bytes(b"\x00\x00\x00\x00ET13" + b"\x00" * 24)
    return path


# --------------------------------------------- A1: ModelFormat.from_path

@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("model.pt", ModelFormat.PYTORCH),
        ("model.pth", ModelFormat.PYTORCH),
        ("model.pt2", ModelFormat.PT2),
        ("model.pte", ModelFormat.PTE),
        ("model.h5", ModelFormat.KERAS),
        ("model.hdf5", ModelFormat.KERAS),
        ("model.keras", ModelFormat.KERAS),
        ("MODEL.KERAS", ModelFormat.KERAS),   # extension matching is case-insensitive
        ("model.pb", ModelFormat.SAVED_MODEL),
        ("model.ckpt", ModelFormat.CKPT),
    ],
)
def test_from_path_maps_extensions(filename, expected):
    assert ModelFormat.from_path(filename) is expected


def test_from_path_saved_model_directory(tmp_path):
    sm = tmp_path / "my_model"
    sm.mkdir()
    (sm / "saved_model.pb").write_bytes(b"")
    assert ModelFormat.from_path(str(sm)) is ModelFormat.SAVED_MODEL


def test_from_path_plain_directory_rejected(tmp_path):
    with pytest.raises(ValueError, match="saved_model.pb"):
        ModelFormat.from_path(str(tmp_path))


def test_from_path_unknown_extension_lists_supported():
    with pytest.raises(ValueError, match=r"\.keras"):
        ModelFormat.from_path("model.bin")


# ------------------------------- clean rejection of weights-only formats

def test_ckpt_rejected_with_export_guidance(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"\x00" * 16)
    with pytest.raises(ValueError) as exc:
        inspect_model(ckpt)
    message = str(exc.value)
    assert "checkpoint" in message
    assert "torch.export.save" in message or "model.save" in message


def test_state_dict_pt_rejected_with_export_guidance(state_dict_pt):
    with pytest.raises(ValueError) as exc:
        inspect_model(state_dict_pt)
    message = str(exc.value)
    assert "state_dict" in message
    assert ".pt2" in message and "torch.onnx.export" in message


def test_non_zip_pt_rejected(tmp_path):
    legacy = tmp_path / "legacy.pth"
    legacy.write_bytes(b"not a zip at all")
    with pytest.raises(ValueError, match="Re-export"):
        inspect_model(legacy)


def test_h5_without_model_config_rejected(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "weights_only.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("kernel", data=[0.0] * 4)
    with pytest.raises(ValueError, match="weights-only"):
        inspect_model(path)


# --------------------------------------------------- A2: new inspectors

def test_keras_zip_inspection(tiny_keras_zip):
    meta = inspect_model(tiny_keras_zip)
    assert meta.format == "h5"
    assert meta.op_counts == {"Dense": 2}      # InputLayer is not an op
    assert meta.num_ops == 2
    assert meta.inputs == [
        {"name": "input_layer", "shape": [None, 4], "dtype": "float32"}
    ]
    assert meta.weights_bytes == 64
    assert meta.is_quantized is False          # INT8 happens at conversion


def test_keras_zip_without_config_rejected(tmp_path):
    bogus = tmp_path / "bogus.keras"
    with zipfile.ZipFile(bogus, "w") as zf:
        zf.writestr("something_else.txt", "hi")
    with pytest.raises(ValueError, match="config.json"):
        inspect_model(bogus)


def test_h5_inspection_reads_config_and_weights(tiny_h5):
    meta = inspect_model(tiny_h5)
    assert meta.format == "h5"
    assert meta.op_counts == {"Dense": 2}
    assert meta.weights_bytes == 40            # 4x2 + 2 float32


def test_nested_submodel_layers_are_flattened(tmp_path):
    config = {
        "class_name": "Functional",
        "config": {
            "layers": [
                {"class_name": "InputLayer",
                 "config": {"name": "in", "batch_shape": [None, 4]}},
                {"class_name": "Sequential",
                 "config": {"layers": [
                     {"class_name": "Conv2D", "config": {"name": "conv"}},
                     {"class_name": "ReLU", "config": {"name": "relu"}},
                 ]}},
            ],
        },
    }
    path = tmp_path / "nested.keras"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("config.json", json.dumps(config))
    meta = inspect_model(path)
    assert meta.op_counts == {"Conv2D": 1, "ReLU": 1}


def test_pte_stat_only_meta(tiny_pte):
    meta = inspect_model(tiny_pte)
    assert meta.format == "pte"
    assert meta.num_ops == 0                   # op checks deferred to backend
    assert any("compile time" in w for w in meta.warnings)


def test_corrupt_pte_rejected(tmp_path):
    bogus = tmp_path / "bogus.pte"
    bogus.write_bytes(b"definitely not a pte")
    with pytest.raises(ValueError, match="ExecuTorch"):
        inspect_model(bogus)


def test_torchscript_layout_accepted_without_torch(torchscript_layout_pt):
    try:
        import torch  # noqa: F401
    except ImportError:
        # Fallback lane: the ZIP layout says "compilable", details deferred.
        meta = inspect_model(torchscript_layout_pt)
        assert meta.format == "pt"
        assert any("torch is not installed" in w for w in meta.warnings)
    else:
        # With torch installed the fake archive must fail at jit.load,
        # not slip through as a valid model.
        with pytest.raises(Exception):
            inspect_model(torchscript_layout_pt)


def test_real_torchscript_full_inspection(tmp_path):
    torch = pytest.importorskip("torch")

    class Net(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x)

    path = tmp_path / "real.pt"
    torch.jit.save(torch.jit.trace(Net(), torch.zeros(1, 4)), str(path))
    meta = inspect_model(path)
    assert meta.format == "pt"
    assert any(op.startswith("aten::relu") for op in meta.op_counts)


def test_saved_model_dir_inspection(tmp_path):
    sm = tmp_path / "sm"
    sm.mkdir()
    (sm / "saved_model.pb").write_bytes(b"")
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        meta = inspect_model(sm)               # stat-only fallback, no crash
        assert meta.format == "pb"
        assert any("tensorflow is not installed" in w for w in meta.warnings)
    else:
        with pytest.raises(ValueError, match="empty SavedModel"):
            inspect_model(sm)


# ------------------------------------------------ A3: converter-chain edges

def test_new_edges_reach_tflite():
    chain = ConverterChain()
    assert chain.hops(ModelFormat.KERAS, ModelFormat.TFLITE) == 1
    assert chain.hops(ModelFormat.SAVED_MODEL, ModelFormat.TFLITE) == 1
    # TorchScript reaches MCU backends through onnx — BFS finds it unaided.
    assert chain.shortest_path(ModelFormat.PYTORCH, ModelFormat.TFLITE) == [
        (ModelFormat.PYTORCH, ModelFormat.ONNX),
        (ModelFormat.ONNX, ModelFormat.TFLITE),
    ]


def test_weights_only_formats_have_no_route():
    chain = ConverterChain()
    assert chain.hops(ModelFormat.CKPT, ModelFormat.TFLITE) is None
    assert chain.hops(ModelFormat.PTE, ModelFormat.TFLITE) is None


# ------------------------------------------ A4: front-ends + quant helper

@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("int8", "int8_full"),
        ("int8_full", "int8_full"),
        (" INT8 ", "int8_full"),               # whitespace/case tolerated
        ("none", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_quantize_aliases(value, expected):
    assert normalize_quantize(value) == expected


def test_normalize_quantize_rejects_unknown():
    # Regression for C1: unknown values must fail loud, never become None.
    with pytest.raises(ValueError, match="fp16"):
        normalize_quantize("fp16")


def test_pipeline_keras_reaches_selection(tiny_keras_zip):
    pipeline = Pipeline()
    req = pipeline.build_request(str(tiny_keras_zip), "esp32s3")
    meta, selection = pipeline.inspect(req)
    assert meta.format == "h5"
    assert selection.best.backend_id == "tflm"
    assert selection.best.conversion_hops == 1
    assert any("after conversion" in w.message for w in selection.best.warnings)


def test_cli_ckpt_fails_with_guidance_not_traceback(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"\x00" * 16)
    result = CliRunner().invoke(
        cli, ["compile", str(ckpt), "--target", "esp32s3"]
    )
    assert result.exit_code != 0
    assert "checkpoint" in result.output
    assert "Traceback" not in result.output


def test_cli_inspect_state_dict_fails_cleanly(state_dict_pt):
    result = CliRunner().invoke(cli, ["inspect", str(state_dict_pt)])
    assert result.exit_code != 0
    assert "state_dict" in result.output
    assert "Traceback" not in result.output
