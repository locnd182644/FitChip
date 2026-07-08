"""End-to-end compile pipeline shared by CLI, orchestrator and web GUI.

None of the front-ends know a concrete backend exists — they hand a
CompileRequest to `run_compile()` and get a CompileResult back.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

from fitchip.core.cal.backend import CompileRequest, CompileResult, ModelFormat, NormalizedError
from fitchip.core.cal.registry import BackendRegistry
from fitchip.core.convert.chain import ConversionError, ConverterChain
from fitchip.core.inspector.base import ModelMeta, inspect_model
from fitchip.core.selection.engine import Candidate, SelectionEngine, SelectionReport
from fitchip.core.targets import TargetRegistry


class Pipeline:
    def __init__(self) -> None:
        self.backends = BackendRegistry()
        self.targets = TargetRegistry()
        self.chain = ConverterChain()
        self.selector = SelectionEngine(self.backends, self.chain)

    def inspect(self, req: CompileRequest) -> tuple[ModelMeta, SelectionReport]:
        """The fast lane: parse + validate + estimate. No compilation."""
        meta = inspect_model(req.model_path, req.model_format)
        return meta, self.selector.select(req, meta)

    def compile(self, req: CompileRequest, out_dir: str | Path) -> CompileResult:
        """The slow lane. Runs in-process for the MVP; wave 2 moves each
        backend into its own ephemeral Docker container behind Celery."""
        meta = inspect_model(req.model_path, req.model_format)
        selection = self.selector.select(req, meta)
        if not selection.candidates:
            return CompileResult(
                success=False,
                error=_no_backend_error(selection),
            )

        attempts: list[str] = []
        last_result: CompileResult | None = None
        for candidate in selection.candidates:  # best first, remainder = fallback chain
            result = self._compile_with(candidate, req, meta, Path(out_dir))
            attempts.append(f"{candidate.backend_id}: {'ok' if result.success else 'failed'}")
            result.report.setdefault("selection", _selection_summary(selection))
            result.report["attempts"] = attempts
            if result.success:
                return result
            last_result = result
        return last_result  # all candidates failed; surface the last normalized error

    def _compile_with(
        self, candidate: Candidate, req: CompileRequest, meta: ModelMeta, out_dir: Path
    ) -> CompileResult:
        manifest = candidate.backend.manifest
        with tempfile.TemporaryDirectory(prefix="fitchip-") as tmp:
            workspace = Path(tmp)
            effective_req = req
            model_path = Path(req.model_path)

            # Convert to a format the backend accepts, if needed.
            if req.model_format.value not in manifest.input_formats:
                dst = self.selector._best_input_format(manifest, req)
                try:
                    model_path = self.chain.convert(
                        model_path, req.model_format, dst, workspace, req
                    )
                except ConversionError as exc:
                    return CompileResult(success=False, error=exc.error)
                # The backend re-inspects the converted graph itself.
                effective_req = dataclasses.replace(
                    req, model_path=str(model_path), model_format=dst
                )

            effective_req.options.setdefault("out_dir", str(out_dir))
            return candidate.backend.compile(effective_req, str(workspace))

    def build_request(
        self,
        model_path: str,
        target_id: str,
        quantize: str | None = None,
        calibration_data: str | None = None,
        optimize_for: str = "size",
        backend: str | None = None,
        options: dict | None = None,
    ) -> CompileRequest:
        opts = dict(options or {})
        if backend:
            opts["backend"] = backend
        return CompileRequest(
            model_path=model_path,
            model_format=ModelFormat.from_path(model_path),
            target=self.targets.get(target_id),
            quantization=quantize,
            calibration_data=calibration_data,
            optimize_for=optimize_for,
            options=opts,
        )


def _selection_summary(selection: SelectionReport) -> dict:
    return {
        "ranking": [
            {
                "backend": c.backend_id,
                "score": c.score,
                "op_coverage": c.op_coverage,
                "conversion_hops": c.conversion_hops,
            }
            for c in selection.candidates
        ],
        "rejected": [
            {"backend": bid, "code": err.code, "reason": err.message}
            for bid, err in selection.rejected
        ],
    }


def _no_backend_error(selection: SelectionReport) -> NormalizedError:
    reasons = [f"  - {bid}: {err.message}" for bid, err in selection.rejected]
    return NormalizedError(
        code="TARGET_UNSUPPORTED",
        message="No installed backend can handle this model/target combination.\n"
        + "\n".join(reasons),
        hints=["Run `fitchip inspect <model> --target <target>` for the full report."],
    )
