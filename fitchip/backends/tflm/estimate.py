"""Memory/footprint heuristics for the TFLM backend.

These are *estimates*, labeled as such everywhere they surface. The generated
project prints `arena_used_bytes()` at boot so users can feed real numbers
back (that feedback loop is the community op-table/model-zoo data).
"""

from __future__ import annotations

# Liveness peak -> arena: TFLM adds per-tensor metadata, alignment padding and
# kernel scratch buffers on top of the raw activation bytes.
ARENA_OVERHEAD_FACTOR = 1.25
ARENA_FIXED_OVERHEAD_BYTES = 16 * 1024

# Flash consumed by the TFLM runtime + generated glue on top of the model
# array, calibrated against ESP-IDF 5.1 -Os builds of comparable projects.
RUNTIME_FLASH_KB = 350

# Expected shrink of a float32 model under full-integer quantization
# (weights go 4->1 byte; ~15% survives as scale/zero-point metadata).
INT8_SIZE_FACTOR = 0.29


def estimate_arena_bytes(meta: dict, will_quantize: bool) -> int | None:
    peak = meta.get("intermediate_peak_bytes")
    if peak is None:
        return None
    if will_quantize:
        peak = peak / 4  # float32 activations become int8
    return int(peak * ARENA_OVERHEAD_FACTOR + ARENA_FIXED_OVERHEAD_BYTES)


def estimate_flash_kb(meta: dict, will_quantize: bool) -> int:
    model_bytes = meta["file_size_bytes"]
    if will_quantize:
        model_bytes = int(model_bytes * INT8_SIZE_FACTOR)
    return model_bytes // 1024 + RUNTIME_FLASH_KB


def estimate_model_kb(meta: dict, will_quantize: bool) -> int:
    model_bytes = meta["file_size_bytes"]
    if will_quantize:
        model_bytes = int(model_bytes * INT8_SIZE_FACTOR)
    return model_bytes // 1024
