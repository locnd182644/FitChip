"""Every torch/executorch import of this backend lives in this module.

The executorch Python API is young and churns between releases (the
integration plan lists this as the top risk), so the blast radius of an
upstream rename is confined to this file. Nothing here is imported at
module level by the adapter — only called inside compile().

All failures are raised as EtError with a normalized ErrorCode so the
adapter can hand them to the CAL without translation.
"""

from __future__ import annotations

from pathlib import Path

from fitchip.core.cal.backend import ErrorCode

_INSTALL_HINT = "Install the ExecuTorch extra:  pip install 'fitchip[executorch]'"


class EtError(Exception):
    def __init__(self, code: str, message: str, hints: list[str] | None = None, raw: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hints = hints or []
        self.raw = raw


def compile_pt2(
    pt2_path: Path,
    *,
    int8: bool,
    calibration_data: str | None,
    use_cmsis_nn: bool,
) -> tuple[bytes, int | None, list[str]]:
    """.pt2 ExportedProgram -> (pte bytes, exact planned arena, log notes).

    INT8 runs the PT2E flow (prepare -> calibrate -> convert) before lowering,
    mirroring the TFLM policy: quantization happens at conversion time only.
    """
    torch = _import_torch()
    _import_executorch()
    notes: list[str] = []

    try:
        ep = torch.export.load(str(pt2_path))
    except Exception as exc:
        raise EtError(
            ErrorCode.MODEL_INVALID,
            f"'{pt2_path.name}' could not be loaded as a torch.export archive.",
            hints=["Re-export with torch.export.save(torch.export.export(model, example), path)."],
            raw=str(exc),
        ) from exc

    if int8:
        ep, quant_notes = _quantize_pt2e(torch, ep, calibration_data)
        notes.extend(quant_notes)

    pte_bytes, arena, lower_notes = _lower(ep, use_cmsis_nn)
    notes.extend(lower_notes)
    return pte_bytes, arena, notes


# ----------------------------------------------------------------- imports

def _import_torch():
    try:
        import torch

        return torch
    except ImportError:
        raise EtError(
            ErrorCode.DEPENDENCY_MISSING,
            "Compiling a .pt2 requires torch, which is not installed.",
            hints=[_INSTALL_HINT],
        ) from None


def _import_executorch():
    try:
        import executorch  # noqa: F401

        return executorch
    except ImportError:
        raise EtError(
            ErrorCode.DEPENDENCY_MISSING,
            "Lowering to a .pte requires the executorch package, which is not installed.",
            hints=[_INSTALL_HINT],
        ) from None


# ------------------------------------------------------------ PT2E / lower

def _quantize_pt2e(torch, ep, calibration_data: str | None):
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

    notes: list[str] = []
    example_args, example_kwargs = _example_inputs(ep)
    try:
        module = ep.module()
        prepared = prepare_pt2e(module, _make_quantizer(notes))
        for sample in _calibration_batches(torch, example_args, calibration_data, notes):
            prepared(*sample, **example_kwargs)
        converted = convert_pt2e(prepared)
        return torch.export.export(converted, example_args, example_kwargs), notes
    except EtError:
        raise
    except Exception as exc:
        raise EtError(
            ErrorCode.QUANTIZE_FAIL,
            "PT2E INT8 quantization failed for this ExportedProgram.",
            hints=[
                "Try compiling without --quantize to isolate the problem.",
                "Quantize at export time instead and ship the resulting .pte.",
            ],
            raw=str(exc),
        ) from exc


def _make_quantizer(notes: list[str]):
    # Prefer an Arm/Cortex-M-aware quantizer when this executorch build has
    # one; the XNNPACK symmetric INT8 config is the portable fallback and
    # produces kernels CMSIS-NN can consume.
    try:
        from executorch.backends.cortex_m.quantizer import CortexMQuantizer  # type: ignore

        return CortexMQuantizer()
    except ImportError:
        pass
    try:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
    except ImportError:
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
    notes.append("[executorch] PT2E quantizer: XNNPACK symmetric INT8 config")
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())
    return quantizer


def _example_inputs(ep):
    example = getattr(ep, "example_inputs", None)
    if not example or example[0] is None:
        raise EtError(
            ErrorCode.QUANTIZE_FAIL,
            "The .pt2 carries no example inputs, so PT2E calibration cannot run.",
            hints=["Re-export with torch.export.export(model, example_inputs) and save again."],
        )
    args, kwargs = example
    return tuple(args), dict(kwargs or {})


def _calibration_batches(torch, example_args, calibration_data: str | None, notes: list[str]):
    from fitchip.core.convert.calibration import load_samples

    samples = load_samples(calibration_data)
    if samples is None:
        # Random calibration — the accuracy warning was already raised by
        # validate(), same contract as the TF-lane converters.
        notes.append("[executorch] no calibration data — using 8 random batches")
        for _ in range(8):
            yield tuple(torch.rand_like(arg) for arg in example_args)
        return
    for sample in samples:
        yield (torch.as_tensor(sample).unsqueeze(0).to(example_args[0].dtype),)


def _lower(ep, use_cmsis_nn: bool) -> tuple[bytes, int | None, list[str]]:
    from executorch.exir import to_edge

    notes: list[str] = []
    try:
        edge = to_edge(ep)
        partitioner = _cmsis_nn_partitioner() if use_cmsis_nn else None
        if partitioner is not None:
            edge = edge.to_backend(partitioner)
            notes.append("[executorch] CMSIS-NN partitioner applied")
        elif use_cmsis_nn:
            notes.append(
                "[executorch] this executorch build ships no Cortex-M partitioner — "
                "portable kernels only (correct but slower)"
            )
        et_program = edge.to_executorch()
        pte_bytes = et_program.buffer
    except Exception as exc:
        raise EtError(
            ErrorCode.CONVERT_FAIL,
            "Lowering the ExportedProgram to an ExecuTorch .pte failed.",
            hints=[
                "Models with data-dependent control flow may need torch.cond rewrites.",
                "See docs/pytorch-path.md for the supported export recipe.",
            ],
            raw=str(exc),
        ) from exc
    return pte_bytes, _planned_arena(et_program), notes


def _cmsis_nn_partitioner():
    try:
        from executorch.backends.cortex_m.partitioner import CortexMPartitioner  # type: ignore

        return CortexMPartitioner()
    except ImportError:
        return None


def _planned_arena(et_program) -> int | None:
    try:
        plan = et_program.executorch_program.execution_plan[0]
        return sum(size for size in plan.non_const_buffer_sizes if size > 0)
    except Exception:
        return None  # report stays honest: no exact plan, no number
