"""ExecuTorch-lane inspector (.pt2 ExportedProgram / .pte ExecuTorch program).

Full inspection needs torch / executorch (the `executorch` extra). Without
them the model still enters the pipeline with stat-only metadata: op checks
are deferred to the backend, exactly like pre-conversion formats. The deep
.pte inspection (op list + exact planned arena from the baked-in memory
plan) lands together with the ExecuTorch backend.
"""

from __future__ import annotations

from pathlib import Path

from fitchip.core.cal.backend import ModelFormat
from fitchip.core.inspector.base import ModelMeta, stat_only_meta

_INSTALL_HINT = "pip install 'fitchip[executorch]'"


def inspect_executorch(path: Path, fmt: ModelFormat) -> ModelMeta:
    if fmt == ModelFormat.PT2:
        return _inspect_pt2(path)
    return _inspect_pte(path)


def _inspect_pt2(path: Path) -> ModelMeta:
    try:
        import torch
    except ImportError:
        return stat_only_meta(
            path,
            ModelFormat.PT2,
            warnings=[
                "torch is not installed — graph details unavailable "
                f"({_INSTALL_HINT}). Op compatibility is checked by the "
                "backend at compile time.",
            ],
        )

    ep = torch.export.load(str(path))
    op_counts: dict[str, int] = {}
    for node in ep.graph.nodes:
        if node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", str(node.target))
        op_counts[name] = op_counts.get(name, 0) + 1
    weights_bytes = sum(
        t.numel() * t.element_size() for t in ep.state_dict.values()
    )
    inputs = [
        {"name": spec.arg.name, "shape": [], "dtype": "unknown"}
        for spec in ep.graph_signature.input_specs
        if spec.kind.name == "USER_INPUT"
    ]
    return ModelMeta(
        format=ModelFormat.PT2.value,
        file_size_bytes=path.stat().st_size,
        num_ops=sum(op_counts.values()),
        op_counts=op_counts,
        inputs=inputs,
        outputs=[],
        weights_bytes=weights_bytes,
        is_quantized=None,
        warnings=[],
    )


def _inspect_pte(path: Path) -> ModelMeta:
    # Sanity check the flatbuffer identifier ("ET??" at offset 4) so corrupt
    # uploads fail loud instead of reaching the backend.
    header = path.read_bytes()[:8]
    if len(header) < 8 or not header[4:6] == b"ET":
        raise ValueError(
            f"'{path.name}' is not an ExecuTorch program (missing ET flatbuffer "
            "identifier). Export with executorch's to_executorch() and save "
            "the .pte buffer."
        )
    try:
        return _deep_pte_meta(path)
    except ImportError:
        return stat_only_meta(
            path,
            ModelFormat.PTE,
            warnings=[
                "Deep .pte inspection (op list, planned memory) requires the "
                f"ExecuTorch backend ({_INSTALL_HINT}); op compatibility is "
                "checked at compile time.",
            ],
        )
    except Exception as exc:  # deserializer churn must never block the pipeline
        return stat_only_meta(
            path,
            ModelFormat.PTE,
            warnings=[
                f"executorch is installed but deep .pte inspection failed ({exc}); "
                "op compatibility is checked at compile time.",
            ],
        )


def _deep_pte_meta(path: Path) -> ModelMeta:
    """Op table + EXACT planned activation bytes from the .pte's baked-in
    memory plan — the real advantage over heuristic arena estimates. The
    deserializer is an internal executorch API whose location moved between
    releases, hence the import fallbacks (isolated here on purpose)."""
    try:
        from executorch.exir._serialize import _deserialize_pte_binary
    except ImportError:
        from executorch.exir._serialize._program import (  # 0.6+
            deserialize_pte_binary as _deserialize_pte_binary,
        )

    program = _deserialize_pte_binary(path.read_bytes())
    plan = program.execution_plan[0]
    op_counts: dict[str, int] = {}
    for op in plan.operators:
        name = op.name if not getattr(op, "overload", "") else f"{op.name}.{op.overload}"
        op_counts[name] = op_counts.get(name, 0) + 1
    planned = sum(size for size in plan.non_const_buffer_sizes if size > 0)
    return ModelMeta(
        format=ModelFormat.PTE.value,
        file_size_bytes=path.stat().st_size,
        num_ops=sum(op_counts.values()),
        op_counts=op_counts,
        inputs=[],
        outputs=[],
        weights_bytes=0,  # constants live in the program segment, not reported separately
        intermediate_peak_bytes=planned,  # exact: memory plan, not a liveness estimate
        is_quantized=None,
        warnings=[
            "Op names are the program's kernel table entries (one count per "
            "registered kernel, not per call site)."
        ],
    )
