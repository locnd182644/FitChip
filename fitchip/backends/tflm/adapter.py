"""TFLM backend adapter — the first (and MVP-only) CompilerBackend.

Wraps TensorFlow Lite Micro + esp-nn: op compatibility from ops_tflm.yaml,
arena/flash heuristics, and generation of a complete ESP-IDF/PlatformIO
project. GUI/CLI never import this module directly — it is discovered through
the `fitchip.backends` entry point.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from fitchip.backends.tflm import estimate as est
from fitchip.backends.tflm.codegen import generate_project
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


class TflmBackend(CompilerBackend):
    def __init__(self) -> None:
        self.manifest = load_manifest(_HERE / "manifest.yaml")
        ops_data = yaml.safe_load((_HERE / self.manifest.ops_supported_file).read_text())
        self.ops: dict[str, dict] = ops_data["ops"]
        self.custom_ops: dict[str, dict] = ops_data.get("custom_ops", {})
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
            "ops_supported": sorted(self.ops) + sorted(self.custom_ops),
            "esp_nn_accelerated": sorted(
                op for op, spec in self.ops.items() if spec.get("esp_nn")
            ),
        }

    def validate(self, req: CompileRequest, model_meta: dict) -> list[NormalizedError]:
        errors: list[NormalizedError] = []

        # The op table (ops_tflm.yaml) names TFLite builtins — it only applies
        # to models already in a format this backend reads directly. Pre-conversion
        # metadata (e.g. ONNX op names) would never match; compile() re-validates
        # after the converter chain has produced a .tflite.
        if not self._is_native_format(model_meta):
            errors.append(
                NormalizedError(
                    code="WARNING",
                    message=(
                        f"Model is '{model_meta.get('format')}' — op compatibility "
                        "will be checked after conversion to tflite."
                    ),
                )
            )
        else:
            unsupported = self._unsupported_ops(model_meta)
            if unsupported:
                hints = [
                    f"{op}: {self.substitutions[op]}"
                    for op in unsupported
                    if op in self.substitutions
                ] or ["See the op support table: docs/op-support.md"]
                errors.append(
                    NormalizedError(
                        code=ErrorCode.OP_UNSUPPORTED,
                        message=(
                            f"{len(unsupported)} operator(s) not supported by TFLM: "
                            f"{', '.join(sorted(unsupported))}"
                        ),
                        hints=hints,
                    )
                )

        will_quantize = self._will_quantize(req, model_meta)
        arena = est.estimate_arena_bytes(model_meta, will_quantize)
        if arena is not None and req.target.ram_kb and arena // 1024 > req.target.ram_kb:
            errors.append(
                NormalizedError(
                    code=ErrorCode.OOM_ARENA,
                    message=(
                        f"Estimated tensor arena ≈ {arena // 1024} KB exceeds "
                        f"{req.target.display_name} RAM ({req.target.ram_kb} KB)."
                    ),
                    hints=(
                        ["Enable INT8 quantization (--quantize int8) to shrink activations ~4x."]
                        if not will_quantize and not model_meta.get("is_quantized")
                        else ["Reduce input resolution or model width at training time."]
                    )
                    + (
                        [f"{req.target.display_name} has PSRAM — consider placing the arena there."]
                        if req.target.psram_kb
                        else []
                    ),
                )
            )

        flash = est.estimate_flash_kb(model_meta, will_quantize)
        if req.target.flash_kb and flash > req.target.flash_kb:
            errors.append(
                NormalizedError(
                    code=ErrorCode.OOM_FLASH,
                    message=(
                        f"Estimated flash footprint ≈ {flash} KB exceeds "
                        f"{req.target.display_name} flash ({req.target.flash_kb} KB)."
                    ),
                    hints=["Quantize to INT8, or prune/distill the model."],
                )
            )

        if (
            req.quantization == "int8_full"
            and not model_meta.get("is_quantized")
            and not req.calibration_data
        ):
            errors.append(
                NormalizedError(
                    code="WARNING",
                    message=(
                        "INT8 quantization without --calibration-data uses random "
                        "calibration: expect a real accuracy drop. Provide a .npy of "
                        "representative inputs for production use."
                    ),
                )
            )
        return errors

    def estimate(self, req: CompileRequest, model_meta: dict) -> dict:
        will_quantize = self._will_quantize(req, model_meta)
        arena = est.estimate_arena_bytes(model_meta, will_quantize)
        estimate = {
            "arena_kb": arena // 1024 if arena is not None else None,
            "flash_kb": est.estimate_flash_kb(model_meta, will_quantize),
            "model_kb": est.estimate_model_kb(model_meta, will_quantize),
            "quantized_output": will_quantize or bool(model_meta.get("is_quantized")),
        }
        if not self._is_native_format(model_meta):
            # Pre-conversion op names never match the TFLite op table: coverage
            # would read 0 and no esp-nn op would match. Omit op_coverage so the
            # engine defaults to 1.0; real numbers come from post-conversion meta.
            estimate["esp_nn_accelerated_ops"] = []
            return estimate
        supported = model_meta["num_ops"] - self._unsupported_op_instances(model_meta)
        estimate["op_coverage"] = (
            supported / model_meta["num_ops"] if model_meta["num_ops"] else 1.0
        )
        estimate["esp_nn_accelerated_ops"] = sorted(
            op
            for op in model_meta["op_counts"]
            if self.ops.get(op, {}).get("esp_nn")
            and any(a.startswith("esp-nn") for a in req.target.accelerators)
        )
        return estimate

    def compile(self, req: CompileRequest, workspace: str) -> CompileResult:
        meta = inspect_model(req.model_path)
        logs: list[str] = [f"[tflm] model: {req.model_path} ({meta.file_size_bytes} bytes)"]

        # Quantization contract: INT8 happens at conversion time (ONNX input
        # arrives here already quantized). A float32 .tflite cannot be
        # post-quantized without its source model — fail with guidance instead
        # of silently shipping a float model the target cannot afford.
        if req.quantization == "int8_full" and not meta.is_quantized:
            return CompileResult(
                success=False,
                logs="\n".join(logs),
                error=NormalizedError(
                    code=ErrorCode.QUANTIZE_FAIL,
                    message=(
                        "This .tflite is float32 and TFLite offers no supported path to "
                        "quantize an exported .tflite without its source model."
                    ),
                    hints=[
                        "Export from the original framework with INT8 PTQ enabled, or",
                        "provide the ONNX model instead — FitChip quantizes during "
                        "onnx→tflite conversion (--calibration-data recommended).",
                    ],
                ),
            )

        errors = [e for e in self.validate(req, meta.to_dict()) if e.code != "WARNING"]
        if errors:
            return CompileResult(success=False, logs="\n".join(logs), error=errors[0])

        out_root = Path(req.options.get("out_dir", "out"))
        project_dir = out_root / f"{req.target.id}-project"
        try:
            resolver_methods = self._resolver_methods(meta.to_dict())
            estimate = self.estimate(req, meta.to_dict())
            generate_project(
                project_dir=project_dir,
                model_path=Path(req.model_path),
                meta=meta,
                target=req.target,
                resolver_methods=resolver_methods,
                arena_bytes=(estimate["arena_kb"] or 64) * 1024,
                esp_nn_ops=estimate["esp_nn_accelerated_ops"],
            )
            logs.append(f"[tflm] project generated at {project_dir}")
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

        model_sha = hashlib.sha256(Path(req.model_path).read_bytes()).hexdigest()
        return CompileResult(
            success=True,
            artifacts=[
                {
                    "kind": ArtifactKind.C_SOURCE_PROJECT.value,
                    "path": str(project_dir),
                    "sha256": model_sha,
                }
            ],
            report={
                "backend": self.manifest.id,
                "target": req.target.id,
                "model_size_bytes": meta.file_size_bytes,
                "num_ops": meta.num_ops,
                "quantized": bool(meta.is_quantized),
                **estimate,
            },
            logs="\n".join(logs),
        )

    # ------------------------------------------------------------- helpers
    def _is_native_format(self, model_meta: dict) -> bool:
        return model_meta.get("format") in self.manifest.input_formats

    def _unsupported_ops(self, model_meta: dict) -> set[str]:
        builtin_unsupported = {
            op
            for op in model_meta["op_counts"]
            if op not in self.ops and op not in model_meta.get("custom_ops", [])
        }
        custom_unsupported = {
            op for op in model_meta.get("custom_ops", []) if op not in self.custom_ops
        }
        return builtin_unsupported | custom_unsupported

    def _unsupported_op_instances(self, model_meta: dict) -> int:
        unsupported = self._unsupported_ops(model_meta)
        return sum(count for op, count in model_meta["op_counts"].items() if op in unsupported)

    def _resolver_methods(self, model_meta: dict) -> list[str]:
        methods = []
        for op in sorted(model_meta["op_counts"]):
            spec = self.ops.get(op) or self.custom_ops.get(op)
            if spec:
                methods.append(spec["resolver"])
        return sorted(set(methods))

    @staticmethod
    def _will_quantize(req: CompileRequest, model_meta: dict) -> bool:
        return req.quantization == "int8_full" and not model_meta.get("is_quantized")
