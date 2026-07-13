"""TensorFlow-family converters: keras (.h5/.keras) and SavedModel/frozen
GraphDef (.pb) into .tflite.

Both require tensorflow (the `fitchip[quantize]` extra). INT8 happens here,
at conversion time, driven by req.quantization + req.calibration_data —
the same policy as the onnx2tf edge: a float32 .tflite is never
post-quantized.
"""

from __future__ import annotations

from pathlib import Path

from fitchip.core.cal.backend import CompileRequest, ErrorCode, NormalizedError
from fitchip.core.convert.calibration import load_samples

_HINT_INSTALL = "Install the conversion extra:  pip install 'fitchip[quantize]'"


def keras_to_tflite(model_path: Path, workspace: Path, req: CompileRequest | None = None) -> Path:
    tf = _import_tf("Keras input requires the keras->tflite converter")
    from fitchip.core.convert.chain import ConversionError

    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        input_shapes = [list(t.shape) for t in model.inputs]
        _apply_quantization(tf, converter, req, input_shapes)
        tflite_bytes = converter.convert()
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(
            NormalizedError(
                code=ErrorCode.CONVERT_FAIL,
                message=f"keras -> tflite conversion failed for '{model_path.name}'.",
                raw=str(exc),
                hints=[
                    "Run `fitchip inspect` to check the layer list first.",
                    "Custom layers need a concrete implementation at load time "
                    "(tf.keras.utils.custom_object_scope).",
                ],
            )
        ) from exc
    return _write(workspace, model_path.stem, tflite_bytes)


def saved_model_to_tflite(
    model_path: Path, workspace: Path, req: CompileRequest | None = None
) -> Path:
    tf = _import_tf("TensorFlow .pb input requires the tf->tflite converter")
    from fitchip.core.convert.chain import ConversionError

    try:
        if model_path.is_dir():
            converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
        else:
            converter = _frozen_graph_converter(tf, model_path)
        _apply_quantization(tf, converter, req, input_shapes=None)
        tflite_bytes = converter.convert()
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(
            NormalizedError(
                code=ErrorCode.CONVERT_FAIL,
                message=f"tensorflow -> tflite conversion failed for '{model_path.name}'.",
                raw=str(exc),
                hints=[
                    "A SavedModel directory converts more reliably than a "
                    "frozen .pb — prefer tf.saved_model.save(...).",
                ],
            )
        ) from exc
    return _write(workspace, model_path.stem or "model", tflite_bytes)


def _frozen_graph_converter(tf, model_path: Path):
    """Best-effort converter for a single-file frozen GraphDef: inputs are
    the Placeholder nodes, outputs the nodes nothing else consumes."""
    from tensorflow.core.framework import graph_pb2

    gd = graph_pb2.GraphDef()
    gd.ParseFromString(model_path.read_bytes())
    inputs = [n.name for n in gd.node if n.op == "Placeholder"]
    consumed = {inp.split(":")[0].lstrip("^") for n in gd.node for inp in n.input}
    outputs = [
        n.name for n in gd.node
        if n.name not in consumed and n.op not in ("Const", "Placeholder", "NoOp")
    ]
    if not inputs or not outputs:
        from fitchip.core.convert.chain import ConversionError

        raise ConversionError(
            NormalizedError(
                code=ErrorCode.CONVERT_FAIL,
                message=f"Could not infer the input/output nodes of '{model_path.name}'.",
                hints=["Convert the original SavedModel directory instead of a frozen .pb."],
            )
        )
    return tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        str(model_path), input_arrays=inputs, output_arrays=outputs
    )


def _apply_quantization(tf, converter, req: CompileRequest | None, input_shapes) -> None:
    if not (req and req.quantization == "int8_full"):
        return
    samples = load_samples(req.calibration_data)
    if samples is None:
        # Random calibration — the accuracy warning is raised upstream by
        # the backend's validate(), mirroring the onnx2tf edge.
        if not input_shapes:
            from fitchip.core.convert.chain import ConversionError

            raise ConversionError(
                NormalizedError(
                    code=ErrorCode.QUANTIZE_FAIL,
                    message="INT8 quantization needs calibration samples for this "
                    "input format (input shapes are not statically known).",
                    hints=["Pass --calibration-data samples.npy"],
                )
            )
        import numpy as np

        shape = [1 if d is None else d for d in input_shapes[0]]
        samples = np.random.rand(8, *shape[1:]).astype("float32")

    def _representative():
        for sample in samples:
            yield [sample[None, ...].astype("float32")]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8


def _import_tf(context: str):
    try:
        import tensorflow as tf

        return tf
    except ImportError:
        from fitchip.core.convert.chain import ConversionError

        raise ConversionError(
            NormalizedError(
                code=ErrorCode.DEPENDENCY_MISSING,
                message=f"{context}, which is not installed.",
                hints=[_HINT_INSTALL],
            )
        ) from None


def _write(workspace: Path, stem: str, tflite_bytes: bytes) -> Path:
    out_dir = workspace / "tf_convert_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}.tflite"
    out.write_bytes(tflite_bytes)
    return out
