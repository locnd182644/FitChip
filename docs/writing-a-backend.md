# Writing a Backend Adapter

Wrapping a compiler = **one adapter class + one manifest**. The core never
imports your code by name — it discovers backends through the
`fitchip.backends` entry-point group, so your adapter can live in a separate
pip package.

## 1. The manifest (`manifest.yaml`)

Everything the Selection Engine knows about your backend lives here:

```yaml
id: mycompiler
display_name: "My Compiler"
docker_image: fitchip/backend-mycompiler:1.0   # reserved for wave-2 isolation
input_formats: [onnx]              # formats you accept directly
output_artifacts: [c_source_project]
targets:
  - match: {has_os: false}                    # rules against TargetProfile
  - match: {has_os: true, isa: [armv8]}       # list value = any-of
quantization: [int8_full, none]
priority: 80                       # tie-breaker when several backends fit
timeout_s: 600
```

## 2. The adapter

Implement `CompilerBackend` (`fitchip/core/cal/backend.py`) — four methods:

```python
from pathlib import Path
from fitchip.core.cal.backend import (
    CompilerBackend, CompileRequest, CompileResult, NormalizedError, ErrorCode,
)
from fitchip.core.cal.manifest import load_manifest

class MyBackend(CompilerBackend):
    def __init__(self):
        # Convention: expose the manifest; the Selection Engine reads it.
        self.manifest = load_manifest(Path(__file__).parent / "manifest.yaml")

    def capabilities(self) -> dict:
        ...  # formats, targets, ops, quant modes — usually straight from the manifest

    def validate(self, req, model_meta) -> list[NormalizedError]:
        ...  # CHEAP pre-compile checks: op coverage, memory forecast.
             # code="WARNING" entries warn without disqualifying you.

    def estimate(self, req, model_meta) -> dict:
        ...  # {"op_coverage": 0..1, "arena_kb": int|None, "flash_kb": int}

    def compile(self, req, workspace) -> CompileResult:
        ...  # the real work. Write artifacts under req.options["out_dir"].
```

Rules of the contract:

- **Never raise from `compile`** — return `CompileResult(success=False,
  error=NormalizedError(...))`. Keep the raw compiler stderr in `error.raw`
  and put actionable advice in `error.hints`.
- Use the shared `ErrorCode` vocabulary (`OP_UNSUPPORTED`, `OOM_ARENA`,
  `CONVERT_FAIL`…) so errors stay comparable across backends.
- Keep your op table in a YAML file next to the manifest (see
  `backends/tflm/ops_tflm.yaml`) so it can be updated without a code change.

## 3. Registration

In your package's `pyproject.toml`:

```toml
[project.entry-points."fitchip.backends"]
mycompiler = "fitchip_backend_mycompiler.adapter:MyBackend"
```

`pip install` your package and `fitchip backends` lists it; the Selection
Engine starts ranking it immediately.

## 4. Test it

`tests/test_selection.py` shows the pattern: a `FakeBackend` plus manifests
is enough to test ranking. For the real adapter, compile the sample models
(`fitchip samples pull micro-speech`) and assert on the generated artifacts —
see `tests/test_tflm_backend.py`.
