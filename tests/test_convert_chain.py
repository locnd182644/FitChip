import sys
from types import SimpleNamespace

import pytest

from fitchip.core.cal.backend import ModelFormat
from fitchip.core.convert.chain import ConverterChain
from fitchip.core.convert.tf_convert import _load_keras_model


def test_same_format_is_zero_hops():
    chain = ConverterChain()
    assert chain.shortest_path(ModelFormat.TFLITE, ModelFormat.TFLITE) == []
    assert chain.hops(ModelFormat.TFLITE, ModelFormat.TFLITE) == 0


def test_onnx_to_tflite_is_one_hop():
    assert ConverterChain().hops(ModelFormat.ONNX, ModelFormat.TFLITE) == 1


def test_unreachable_returns_none():
    chain = ConverterChain()
    assert chain.hops(ModelFormat.TFLITE, ModelFormat.ONNX) is None
    assert chain.shortest_path(ModelFormat.CKPT, ModelFormat.TFLITE) is None


def test_registered_edge_enables_multi_hop_path():
    chain = ConverterChain()
    chain.register(ModelFormat.PTE, ModelFormat.ONNX, lambda p, w, r=None: p)
    path = chain.shortest_path(ModelFormat.PTE, ModelFormat.TFLITE)
    assert path == [
        (ModelFormat.PTE, ModelFormat.ONNX),
        (ModelFormat.ONNX, ModelFormat.TFLITE),
    ]


# Regression: legacy Keras-2 .h5 files (TFOpLambda layers) cannot be loaded
# by Keras 3 (tf.keras since TF 2.16) — the converter must fall back to
# tf_keras instead of failing, and must not depend on whether something else
# (e.g. onnx2tf) flipped TF_USE_LEGACY_KERAS earlier in the process.

def _keras3_like_tf():
    """A tf module whose Keras-3 loader rejects legacy archives."""
    def load_model(path, compile):  # noqa: A002 — keras kwarg name
        raise ValueError("Unknown layer: 'TFOpLambda'. Please ensure ...")
    return SimpleNamespace(keras=SimpleNamespace(models=SimpleNamespace(load_model=load_model)))


def test_legacy_h5_falls_back_to_tf_keras(monkeypatch, tmp_path):
    sentinel = object()
    fake_tf_keras = SimpleNamespace(
        models=SimpleNamespace(load_model=lambda path, compile: sentinel)
    )
    monkeypatch.setitem(sys.modules, "tf_keras", fake_tf_keras)
    assert _load_keras_model(_keras3_like_tf(), tmp_path / "legacy.h5") is sentinel


def test_legacy_h5_surfaces_original_error_without_tf_keras(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "tf_keras", None)  # import raises ImportError
    with pytest.raises(ValueError, match="TFOpLambda"):
        _load_keras_model(_keras3_like_tf(), tmp_path / "legacy.h5")


def test_legacy_h5_original_error_when_both_loaders_fail(monkeypatch, tmp_path):
    def also_fails(path, compile):
        raise RuntimeError("tf_keras choked differently")
    monkeypatch.setitem(
        sys.modules, "tf_keras",
        SimpleNamespace(models=SimpleNamespace(load_model=also_fails)),
    )
    with pytest.raises(ValueError, match="TFOpLambda"):  # not the RuntimeError
        _load_keras_model(_keras3_like_tf(), tmp_path / "legacy.h5")
