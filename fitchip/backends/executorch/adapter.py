"""ExecuTorch backend adapter — Cortex-M (STM32 & friends) + CMSIS-NN.

Design decisions:
- export/lower/PT2E-quantize run inside compile(), not through the
  ConverterChain: quantization needs the source ExportedProgram +
  calibration, which a pure path->path converter cannot model;
- `.pt2` + int8 -> PT2E flow; `.pte` + int8 -> QUANTIZE_FAIL (mirror of the
  TFLM policy: never post-quantize an already-lowered artifact);
- the arena number is EXACT when it comes from the .pte's memory plan;
- torch/executorch are imported only inside _et.py, at call time — this
  module must construct and validate with the light core install.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from fitchip.backends.executorch import estimate as est
from fitchip.backends.executorch._et import EtError, compile_pt2
from fitchip.backends.executorch.codegen import generate_project
from fitchip.core.cal.backend import (
    ArtifactKind,
    CompileRequest,
    CompileResult,
    CompilerBackend,
    ErrorCode,
    NormalizedError,
)
from fitchip.core.cal.manifest import load_manifest
from fitchip.core.inspector import inspect_model

_HERE = Path(__file__).parent

_PTE_INT8_ERROR = NormalizedError(
    code=ErrorCode.QUANTIZE_FAIL,
    message=(
        "This model is already a lowered .pte — ExecuTorch has no supported "
        "path to post-quantize it without the source ExportedProgram."
    ),
    hints=[
        "Provide the .pt2 instead — FitChip runs PT2E INT8 during lowering "
        "(--calibration-data recommended), or",
        "quantize at export time and ship the resulting .pte with --quantize none.",
    ],
)


class ExecutorchBackend(CompilerBackend):
    def __init__(self) -> None:
        self.manifest = load_manifest(_HERE / "manifest.yaml")
        ops_data = yaml.safe_load((_HERE / self.manifest.ops_supported_file).read_text())
        self.ops: dict[str, dict] = ops_data["ops"]
        self.substitutions: dict[str, str] = ops_data.get("substitutions", {})

    # ------------------------------------------------------------------ CAL
    def capabilities(self) -> dict:
        return {
            "id": self.manifest.id,
            "display_name": self.manifest.display_name,
            "input_formats": self.manifest.input_formats,
            "convertible_from": self.manifest.convertible_from,
            "output_artifacts": self.manifest.output_artifacts,
            "quantization": self.manifest.quantization,
            "ops_supported": sorted(self.ops),
            "cmsis_nn_accelerated": sorted(
                op for op, spec in self.ops.items() if spec.get("cmsis_nn")
            ),
        }

    def validate(self, req: CompileRequest, model_meta: dict) -> list[NormalizedError]:
        errors: list[NormalizedError] = []

        if req.quantization == "int8_full" and model_meta.get("format") == "pte":
            errors.append(_PTE_INT8_ERROR)

        # Ops absent from the table still run — ExecuTorch falls back to its
        # portable reference kernels — so this is a latency warning, never a
        # rejection (unlike TFLM's OP_UNSUPPORTED).
        unknown = self._unknown_ops(model_meta)
        if unknown:
            hints = [
                f"{op}: {self.substitutions[_base_op_name(op)]}"
                for op in sorted(unknown)
                if _base_op_name(op) in self.substitutions
            ]
            errors.append(
                NormalizedError(
                    code="WARNING",
                    message=(
                        f"{len(unknown)} operator(s) not in the declared table "
                        f"({', '.join(sorted(unknown))}) — they will run on portable "
                        "kernels (correct but slower)."
                    ),
                    hints=hints,
                )
            )

        arena = est.exact_arena_bytes(model_meta)
        if arena is not None and req.target.ram_kb and arena // 1024 > req.target.ram_kb:
            errors.append(
                NormalizedError(
                    code=ErrorCode.OOM_ARENA,
                    message=(
                        f"The memory plan baked into this .pte needs {arena // 1024} KB "
                        f"of activations — more than {req.target.display_name} RAM "
                        f"({req.target.ram_kb} KB). This number is exact, not an estimate."
                    ),
                    hints=["Re-export with a smaller input resolution or model width."],
                )
            )

        will_quantize = self._will_quantize(req, model_meta)
        flash = est.estimate_flash_kb(model_meta, will_quantize)
        if req.target.flash_kb and flash > req.target.flash_kb:
            errors.append(
                NormalizedError(
                    code=ErrorCode.OOM_FLASH,
                    message=(
                        f"Estimated flash footprint ≈ {flash} KB exceeds "
                        f"{req.target.display_name} flash ({req.target.flash_kb} KB)."
                    ),
                    hints=["Quantize to INT8 (--quantize int8), or prune/distill the model."],
                )
            )

        if will_quantize and not req.calibration_data:
            errors.append(
                NormalizedError(
                    code="WARNING",
                    message=(
                        "INT8 (PT2E) without --calibration-data uses random calibration: "
                        "expect a real accuracy drop. Provide a .npy of representative "
                        "inputs for production use."
                    ),
                )
            )
        return errors

    def estimate(self, req: CompileRequest, model_meta: dict) -> dict:
        will_quantize = self._will_quantize(req, model_meta)
        arena = est.exact_arena_bytes(model_meta)
        return {
            "arena_kb": arena // 1024 if arena is not None else None,
            "arena_exact": arena is not None,
            "flash_kb": est.estimate_flash_kb(model_meta, will_quantize),
            "model_kb": est.estimate_model_kb(model_meta, will_quantize),
            "quantized_output": will_quantize or bool(model_meta.get("is_quantized")),
            "cmsis_nn_accelerated_ops": self._accelerated_ops(req, model_meta),
        }

    def compile(self, req: CompileRequest, workspace: str) -> CompileResult:
        meta = inspect_model(req.model_path, req.model_format)
        logs: list[str] = [
            f"[executorch] model: {req.model_path} ({meta.file_size_bytes} bytes)"
        ]

        errors = [e for e in self.validate(req, meta.to_dict()) if e.code != "WARNING"]
        if errors:
            return CompileResult(success=False, logs="\n".join(logs), error=errors[0])

        use_cmsis_nn = any(a == "cmsis-nn" for a in req.target.accelerators)
        if meta.format == "pte":
            pte_bytes = Path(req.model_path).read_bytes()
            arena = est.exact_arena_bytes(meta.to_dict())
        else:  # .pt2 — export/quantize/lower happens here, per design decision 1
            try:
                pte_bytes, arena, notes = compile_pt2(
                    Path(req.model_path),
                    int8=req.quantization == "int8_full",
                    calibration_data=req.calibration_data,
                    use_cmsis_nn=use_cmsis_nn,
                )
                logs.extend(notes)
            except EtError as exc:
                return CompileResult(
                    success=False,
                    logs="\n".join(logs),
                    error=NormalizedError(
                        code=exc.code, message=exc.message, hints=exc.hints, raw=exc.raw
                    ),
                )
            logs.append(f"[executorch] lowered to .pte ({len(pte_bytes)} bytes)")

        out_root = Path(req.options.get("out_dir", "out"))
        project_dir = out_root / f"{req.target.id}-project"
        arena_bytes = arena if arena else est.fallback_arena_bytes(req.target)
        try:
            generate_project(
                project_dir=project_dir,
                pte_bytes=pte_bytes,
                meta=meta,
                target=req.target,
                arena_bytes=arena_bytes,
                arena_exact=arena is not None,
                cmsis_nn_ops=self._accelerated_ops(req, meta.to_dict()),
                use_cmsis_nn=use_cmsis_nn,
            )
            logs.append(f"[executorch] project generated at {project_dir}")
        except Exception as exc:  # codegen must never leak a raw traceback
            return CompileResult(
                success=False,
                logs="\n".join(logs),
                error=NormalizedError(
                    code=ErrorCode.INTERNAL,
                    message=f"Project generation failed: {exc}",
                    raw=repr(exc),
                ),
            )

        return CompileResult(
            success=True,
            artifacts=[
                {
                    "kind": ArtifactKind.C_SOURCE_PROJECT.value,
                    "path": str(project_dir),
                    "sha256": hashlib.sha256(pte_bytes).hexdigest(),
                }
            ],
            report={
                "backend": self.manifest.id,
                "target": req.target.id,
                "model_size_bytes": len(pte_bytes),
                "num_ops": meta.num_ops,
                "quantized": req.quantization == "int8_full" or bool(meta.is_quantized),
                "arena_kb": arena // 1024 if arena is not None else None,
                "arena_exact": arena is not None,
                "flash_kb": est.RUNTIME_FLASH_KB + len(pte_bytes) // 1024,
                "cmsis_nn_accelerated_ops": self._accelerated_ops(req, meta.to_dict()),
                "next_steps": [
                    f"Next:  cd {project_dir} && "
                    "cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/arm-none-eabi.cmake && "
                    "cmake --build build",
                    "       Link the static library into your CubeMX/BSP firmware — "
                    "see the project README.",
                ],
            },
            logs="\n".join(logs),
        )

    # ------------------------------------------------------------- helpers
    def _unknown_ops(self, model_meta: dict) -> set[str]:
        return {
            op for op in model_meta.get("op_counts", {})
            if _base_op_name(op) not in self.ops
        }

    def _accelerated_ops(self, req: CompileRequest, model_meta: dict) -> list[str]:
        if not any(a == "cmsis-nn" for a in req.target.accelerators):
            return []
        return sorted(
            {
                _base_op_name(op)
                for op in model_meta.get("op_counts", {})
                if self.ops.get(_base_op_name(op), {}).get("cmsis_nn")
            }
        )

    @staticmethod
    def _will_quantize(req: CompileRequest, model_meta: dict) -> bool:
        return req.quantization == "int8_full" and model_meta.get("format") == "pt2"


def _base_op_name(name: str) -> str:
    """'aten::conv2d', 'aten.conv2d.default', '_softmax.out' -> table key."""
    base = name.strip().lower().rsplit("::", 1)[-1]
    base = base.removeprefix("aten.")
    base = base.split(".", 1)[0]
    return base.lstrip("_")
