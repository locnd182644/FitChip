"""Single source of truth for user-facing quantization mode names.

Every front-end (CLI, web, orchestrator) accepts the same aliases and maps
them to the internal mode names that backend manifests declare. An unknown
value raises ValueError so front-ends fail loud (HTTP 422 / exit 1) instead
of silently compiling without quantization.
"""

from __future__ import annotations

# user-facing value -> internal mode (None = no quantization)
_ALIASES: dict[str, str | None] = {
    "int8": "int8_full",
    "int8_full": "int8_full",
    "none": None,
    "": None,
}

# What front-ends should offer in choice widgets / --help.
QUANT_CHOICES = ["int8", "none"]


def normalize_quantize(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown quantization mode '{value}'. Supported: {', '.join(QUANT_CHOICES)}."
        )
    return _ALIASES[key]
