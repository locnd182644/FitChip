"""Memory/footprint numbers for the ExecuTorch backend.

The headline difference vs. the TFLM heuristics: a .pte carries its memory
plan, so the arena number is **exact** (the inspector surfaces it as
`intermediate_peak_bytes`) — no 1.25x safety factor. Flash stays a labeled
heuristic: the ExecuTorch runtime + portable kernels add roughly 250 KB of
code on top of the program itself in an -Os arm-none-eabi build.
"""

from __future__ import annotations

from fitchip.core.cal.backend import TargetProfile

# Flash consumed by the ExecuTorch runtime, portable kernels and generated
# glue on top of the embedded .pte (heuristic; the map file has the truth).
RUNTIME_FLASH_KB = 250

# Expected shrink of a float32 .pt2 under PT2E full-integer quantization
# (weights 4 -> 1 byte; scales/zero-points survive) — same rationale as TFLM.
INT8_SIZE_FACTOR = 0.29


def exact_arena_bytes(meta: dict) -> int | None:
    """The memory-planned activation size baked into a .pte, or None when the
    model is a pre-lowering .pt2 / deep inspection was unavailable."""
    if meta.get("format") != "pte":
        return None
    return meta.get("intermediate_peak_bytes")


def fallback_arena_bytes(target: TargetProfile) -> int:
    """Placeholder static-buffer size emitted when the plan is unknown:
    a quarter of the target's RAM, at least 16 KB. Labeled as a placeholder
    everywhere it surfaces — never reported as `arena_kb`."""
    return max(16, (target.ram_kb or 64) // 4) * 1024


def estimate_flash_kb(meta: dict, will_quantize: bool) -> int:
    return estimate_model_kb(meta, will_quantize) + RUNTIME_FLASH_KB


def estimate_model_kb(meta: dict, will_quantize: bool) -> int:
    model_bytes = meta["file_size_bytes"]
    if will_quantize:
        model_bytes = int(model_bytes * INT8_SIZE_FACTOR)
    return model_bytes // 1024
