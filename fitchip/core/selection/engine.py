"""Selection Engine — rule-based filtering + scoring over backend manifests.

The pipeline (per the architecture doc):

  1. Hard filter   — format reachable? target matches manifest rules? quant mode?
  2. Validate      — backend.validate(): op coverage, memory forecast
  3. Score & rank  — priority + op_coverage*W - conversion_hops*W + memory fit
  4. Fallback      — the ranked remainder becomes the fallback chain

Deliberately data-driven: adding a backend never means editing rules here.
The (request, ranking, outcome) logs this produces are the training data for
the future ML-based selector — swap the scorer, keep the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fitchip.core.cal.backend import (
    CompileRequest,
    CompilerBackend,
    ModelFormat,
    NormalizedError,
)
from fitchip.core.cal.registry import BackendRegistry
from fitchip.core.convert.chain import ConverterChain
from fitchip.core.inspector.base import ModelMeta

# Scoring weights. Static for the MVP; replaced by a learned model later.
W_OP_COVERAGE = 50.0
W_MEMORY_FIT = 30.0
W_CONVERSION_HOP = 10.0


@dataclass
class Candidate:
    backend: CompilerBackend
    backend_id: str
    score: float
    op_coverage: float                  # 0..1
    conversion_hops: int
    estimate: dict = field(default_factory=dict)
    warnings: list[NormalizedError] = field(default_factory=list)


@dataclass
class SelectionReport:
    candidates: list[Candidate] = field(default_factory=list)   # ranked, best first
    rejected: list[tuple[str, NormalizedError]] = field(default_factory=list)

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def fallback_chain(self) -> list[Candidate]:
        return self.candidates[1:]


class SelectionEngine:
    def __init__(self, registry: BackendRegistry, chain: ConverterChain | None = None) -> None:
        self.registry = registry
        self.chain = chain or ConverterChain()

    def select(self, req: CompileRequest, meta: ModelMeta) -> SelectionReport:
        report = SelectionReport()
        forced = req.options.get("backend")
        backends = [self.registry.get(forced)] if forced else self.registry.all()

        for backend in backends:
            manifest = backend.manifest  # convention: every adapter exposes its manifest
            rejection = self._hard_filter(manifest, req)
            if rejection is not None:
                report.rejected.append((manifest.id, rejection))
                continue

            errors = backend.validate(req, meta.to_dict())
            fatal = [e for e in errors if e.code != "WARNING"]
            if fatal:
                report.rejected.append((manifest.id, fatal[0]))
                continue

            hops = self.chain.hops(req.model_format, self._best_input_format(manifest, req))
            estimate = backend.estimate(req, meta.to_dict())
            coverage = float(estimate.get("op_coverage", 1.0))
            score = (
                manifest.priority
                + coverage * W_OP_COVERAGE
                + _memory_fit(estimate, req) * W_MEMORY_FIT
                - (hops or 0) * W_CONVERSION_HOP
            )
            report.candidates.append(
                Candidate(
                    backend=backend,
                    backend_id=manifest.id,
                    score=round(score, 2),
                    op_coverage=coverage,
                    conversion_hops=hops or 0,
                    estimate=estimate,
                    warnings=[e for e in errors if e.code == "WARNING"],
                )
            )

        report.candidates.sort(key=lambda c: c.score, reverse=True)
        return report

    def _hard_filter(self, manifest, req: CompileRequest) -> NormalizedError | None:
        if self._best_input_format(manifest, req) is None:
            return NormalizedError(
                code="FORMAT_UNSUPPORTED",
                message=(
                    f"Backend '{manifest.id}' accepts {manifest.input_formats} and no "
                    f"conversion route exists from '{req.model_format.value}'."
                ),
            )
        if not manifest.matches_target(req.target):
            return NormalizedError(
                code="TARGET_UNSUPPORTED",
                message=f"Backend '{manifest.id}' does not support target '{req.target.id}'.",
            )
        if not manifest.supports_quantization(req.quantization):
            return NormalizedError(
                code="QUANT_UNSUPPORTED",
                message=(
                    f"Backend '{manifest.id}' does not support quantization "
                    f"'{req.quantization}'. Supported: {manifest.quantization}."
                ),
            )
        return None

    def _best_input_format(self, manifest, req: CompileRequest) -> ModelFormat | None:
        """The accepted input format reachable in the fewest conversion hops."""
        best: tuple[int, ModelFormat] | None = None
        for fmt_name in manifest.input_formats:
            fmt = ModelFormat(fmt_name)
            hops = self.chain.hops(req.model_format, fmt)
            if hops is not None and (best is None or hops < best[0]):
                best = (hops, fmt)
        return best[1] if best else None


def _memory_fit(estimate: dict, req: CompileRequest) -> float:
    """1.0 = fits comfortably, 0.0 = at the limit; candidates that clearly
    do not fit were already rejected by backend.validate()."""
    arena_kb = estimate.get("arena_kb")
    if not arena_kb or not req.target.ram_kb:
        return 0.5  # unknown — neutral
    headroom = 1.0 - (arena_kb / req.target.ram_kb)
    return max(0.0, min(1.0, headroom))
