"""TorchScript (.pt/.pth) -> ONNX converter.

Opens the 2-hop route torchscript -> onnx -> tflite to MCU backends. Best
effort by design: torch.onnx.export needs example inputs, which a traced
TorchScript archive carries as static shapes but a scripted one often does
not — in that case the user supplies them via options or exports ONNX
directly from the training code.
"""

from __future__ import annotations

from pathlib import Path

from fitchip.core.cal.backend import CompileRequest, ErrorCode, NormalizedError


def torchscript_to_onnx(
    model_path: Path, workspace: Path, req: CompileRequest | None = None
) -> Path:
    from fitchip.core.convert.chain import ConversionError

    try:
        import torch
    except ImportError:
        raise ConversionError(
            NormalizedError(
                code=ErrorCode.DEPENDENCY_MISSING,
                message="TorchScript input requires the torchscript->onnx converter, "
                "which needs torch (not installed).",
                hints=["pip install torch  —  or export to ONNX from your training code."],
            )
        ) from None

    try:
        module = torch.jit.load(str(model_path), map_location="cpu").eval()
    except Exception as exc:
        raise ConversionError(
            NormalizedError(
                code=ErrorCode.MODEL_INVALID,
                message=f"'{model_path.name}' could not be loaded as TorchScript.",
                raw=str(exc),
            )
        ) from exc

    shapes = _input_shapes(module, req)
    if shapes is None:
        raise ConversionError(
            NormalizedError(
                code=ErrorCode.CONVERT_FAIL,
                message="The TorchScript graph does not carry static input shapes, "
                "so example inputs for the ONNX export cannot be built.",
                hints=[
                    'Pass shapes explicitly: --option input_shapes=[[1,3,224,224]] '
                    "(orchestrator/web: options.input_shapes)",
                    "Or export ONNX directly: torch.onnx.export(model, example, 'model.onnx')",
                ],
            )
        )

    example = tuple(torch.zeros(*shape) for shape in shapes)
    out = workspace / f"{model_path.stem}.onnx"
    try:
        torch.onnx.export(module, example, str(out))
    except Exception as exc:
        raise ConversionError(
            NormalizedError(
                code=ErrorCode.CONVERT_FAIL,
                message=f"torch.onnx.export failed for '{model_path.name}'.",
                raw=str(exc),
                hints=[
                    "Models with data-dependent control flow rarely survive the "
                    "TorchScript->ONNX trip; export ONNX from the eager model instead.",
                ],
            )
        ) from exc
    return out


def _input_shapes(module, req: CompileRequest | None) -> list[list[int]] | None:
    explicit = (req.options.get("input_shapes") if req else None)
    if explicit:
        return [[int(d) for d in shape] for shape in explicit]
    shapes = []
    for value in list(module.graph.inputs())[1:]:  # arg 0 is `self`
        try:
            sizes = value.type().sizes()
        except RuntimeError:
            return None
        if sizes is None or any(d is None for d in sizes):
            return None
        shapes.append([max(int(d), 1) for d in sizes])
    return shapes or None
