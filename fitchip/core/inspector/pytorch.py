"""PyTorch file inspector (.pt / .pth).

A .pt/.pth file is one of three very different things, and only one of them
is compilable:

- a TorchScript archive (torch.jit.save) — self-contained graph + weights ✅
- a torch.save() pickle of a state_dict — weights only, no graph ❌
- a torch.save() pickle of a whole nn.Module — needs the user's class code ❌

The ZIP layout distinguishes them without importing torch (TorchScript
archives contain constants.pkl + code/; torch.save archives contain
data.pkl), so the rejection message is dependable even on a bare install.
Unpickling is never attempted: torch pickles execute arbitrary code.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from fitchip.core.cal.backend import ModelFormat
from fitchip.core.inspector.base import ModelMeta, stat_only_meta

_EXPORT_GUIDANCE = (
    "Re-export from your training code: torch.export.save(...) (.pt2, "
    "preferred for MCU targets), torch.onnx.export(...) (.onnx), or "
    "torch.jit.script/trace + torch.jit.save (TorchScript .pt)."
)


def inspect_torchscript(path: Path) -> ModelMeta:
    if not zipfile.is_zipfile(path):
        raise ValueError(
            f"'{path.name}' is not a loadable PyTorch model (legacy or corrupt "
            "torch.save format). " + _EXPORT_GUIDANCE
        )
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    is_torchscript = any(n.endswith("constants.pkl") for n in names) and any(
        "/code/" in n or n.startswith("code/") for n in names
    )
    if not is_torchscript:
        raise ValueError(
            f"'{path.name}' is a torch.save() pickle (state_dict or pickled "
            "module) — it cannot be compiled standalone: a state_dict has no "
            "graph, and unpickling a module requires your class code. "
            + _EXPORT_GUIDANCE
        )

    try:
        import torch
    except ImportError:
        return stat_only_meta(
            path,
            ModelFormat.PYTORCH,
            warnings=[
                "torch is not installed — graph details unavailable "
                "(pip install torch). Op compatibility is checked after the "
                "torchscript -> onnx conversion step.",
            ],
        )

    module = torch.jit.load(str(path), map_location="cpu")
    op_counts: dict[str, int] = {}
    for node in module.graph.nodes():
        kind = node.kind()  # e.g. "aten::conv2d"
        if kind.startswith("prim::"):
            continue
        op_counts[kind] = op_counts.get(kind, 0) + 1
    weights_bytes = sum(
        p.numel() * p.element_size() for p in module.parameters()
    )
    inputs = [
        {"name": inp.debugName(), "shape": _shape_of(inp), "dtype": "unknown"}
        for inp in list(module.graph.inputs())[1:]  # arg 0 is `self`
    ]
    return ModelMeta(
        format=ModelFormat.PYTORCH.value,
        file_size_bytes=path.stat().st_size,
        num_ops=sum(op_counts.values()),
        op_counts=op_counts,
        inputs=inputs,
        outputs=[],
        weights_bytes=weights_bytes,
        is_quantized=None,
        warnings=[
            "Op names are TorchScript ops; exact op mapping is checked after "
            "the torchscript -> onnx conversion step."
        ],
    )


def _shape_of(value) -> list:
    try:
        sizes = value.type().sizes()
        return list(sizes) if sizes is not None else []
    except RuntimeError:
        return []
