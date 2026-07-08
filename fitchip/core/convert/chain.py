"""Converter chain — a shared graph of model-format conversions.

Converters are edges in a graph; the Selection Engine asks for the shortest
path from the user's format to a format the backend accepts, and each extra
hop costs score points. Adding a converter = registering one edge.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Callable

from fitchip.core.cal.backend import CompileRequest, ErrorCode, ModelFormat, NormalizedError


class ConversionError(Exception):
    def __init__(self, error: NormalizedError):
        super().__init__(error.message)
        self.error = error


# A converter takes (input_path, workspace_dir, request) and returns the
# output path. The request is passed because some conversions are also the
# natural place to quantize (onnx2tf emits INT8 during conversion).
Converter = Callable[[Path, Path, CompileRequest | None], Path]


class ConverterChain:
    def __init__(self) -> None:
        self._edges: dict[tuple[ModelFormat, ModelFormat], Converter] = {}
        self._edges[(ModelFormat.ONNX, ModelFormat.TFLITE)] = _onnx_to_tflite

    def register(self, src: ModelFormat, dst: ModelFormat, converter: Converter) -> None:
        self._edges[(src, dst)] = converter

    def shortest_path(
        self, src: ModelFormat, dst: ModelFormat
    ) -> list[tuple[ModelFormat, ModelFormat]] | None:
        """BFS over the conversion graph. Returns the list of edges to apply,
        [] when src == dst, or None when unreachable."""
        if src == dst:
            return []
        queue: deque[tuple[ModelFormat, list]] = deque([(src, [])])
        seen = {src}
        while queue:
            fmt, path = queue.popleft()
            for (edge_src, edge_dst), _ in self._edges.items():
                if edge_src != fmt or edge_dst in seen:
                    continue
                new_path = path + [(edge_src, edge_dst)]
                if edge_dst == dst:
                    return new_path
                seen.add(edge_dst)
                queue.append((edge_dst, new_path))
        return None

    def hops(self, src: ModelFormat, dst: ModelFormat) -> int | None:
        path = self.shortest_path(src, dst)
        return None if path is None else len(path)

    def convert(
        self,
        model_path: Path,
        src: ModelFormat,
        dst: ModelFormat,
        workspace: Path,
        req: CompileRequest | None = None,
    ) -> Path:
        path = self.shortest_path(src, dst)
        if path is None:
            raise ConversionError(
                NormalizedError(
                    code=ErrorCode.CONVERT_FAIL,
                    message=f"No conversion route from {src.value} to {dst.value}.",
                )
            )
        current = model_path
        for edge in path:
            current = self._edges[edge](current, workspace, req)
        return current


def _onnx_to_tflite(model_path: Path, workspace: Path, req: CompileRequest | None = None) -> Path:
    """onnx -> tflite via onnx2tf (requires the `fitchip[quantize]` extra)."""
    try:
        import onnx2tf  # noqa: F401
    except ImportError:
        raise ConversionError(
            NormalizedError(
                code=ErrorCode.DEPENDENCY_MISSING,
                message="ONNX input requires the onnx->tflite converter, which is not installed.",
                hints=["Install the conversion extra:  pip install 'fitchip[quantize]'"],
            )
        ) from None

    out_dir = workspace / "onnx2tf_out"
    # INT8 happens here, at conversion time: onnx2tf drives the TFLite
    # converter with a representative dataset. Quantizing an already-exported
    # float32 .tflite (without its source model) has no supported path.
    want_int8 = bool(req and req.quantization == "int8_full")
    kwargs: dict = {}
    if want_int8:
        kwargs["output_integer_quantized_tflite"] = True
        calib = _calibration_arg(model_path, req.calibration_data) if req.calibration_data else None
        if calib:
            kwargs["custom_input_op_name_np_data_path"] = calib
    try:
        onnx2tf.convert(
            input_onnx_file_path=str(model_path),
            output_folder_path=str(out_dir),
            output_signaturedefs=False,
            non_verbose=True,
            **kwargs,
        )
    except Exception as exc:
        raise ConversionError(
            NormalizedError(
                code=ErrorCode.CONVERT_FAIL,
                message=f"onnx2tf failed to convert '{model_path.name}'.",
                raw=str(exc),
                hints=[
                    "Run `fitchip inspect` to check op compatibility before converting.",
                    "Simplify the ONNX graph first: pip install onnxsim && onnxsim in.onnx out.onnx",
                    "See docs/onnx-conversion.md for known-problematic ops.",
                ],
            )
        ) from exc

    patterns = (
        ["*_full_integer_quant.tflite", "*_integer_quant.tflite"]
        if want_int8
        else ["*_float32.tflite", "*.tflite"]
    )
    for pattern in patterns:
        candidates = sorted(out_dir.glob(pattern))
        if candidates:
            return candidates[0]
    raise ConversionError(
        NormalizedError(
            code=ErrorCode.CONVERT_FAIL,
            message="onnx2tf finished but produced no .tflite file"
            + (" with INT8 quantization" if want_int8 else "")
            + ".",
        )
    )


def _calibration_arg(model_path: Path, calibration_data: str) -> list | None:
    """Build onnx2tf's [[input_name, npy_path, mean, std], ...] argument from
    a .npy file (or a directory containing one) of representative samples.
    Returns None when nothing usable is found — onnx2tf then falls back to its
    built-in random calibration (accuracy warning is raised upstream)."""
    calib_path = Path(calibration_data)
    if calib_path.is_dir():
        npys = sorted(calib_path.glob("*.npy"))
        if not npys:
            return None
        calib_path = npys[0]
    if calib_path.suffix != ".npy":
        return None

    import onnx

    graph = onnx.load(str(model_path)).graph
    initializers = {init.name for init in graph.initializer}
    input_names = [vi.name for vi in graph.input if vi.name not in initializers]
    if not input_names:
        return None
    # mean=0, std=1: samples are expected to be already preprocessed.
    return [[input_names[0], str(calib_path), 0.0, 1.0]]
