# Contributing to FitChip

The fastest ways to help, in order of impact:

1. **Test a board.** Compile a sample (`fitchip samples pull micro-speech`),
   flash the generated project, and PR the boot log's arena/latency numbers
   into [docs/op-support.md](docs/op-support.md).
2. **Add a target profile.** One YAML file — [guide](docs/adding-a-target.md).
3. **Write a backend adapter.** ~200 lines + a manifest —
   [guide](docs/writing-a-backend.md).
4. **Report broken models.** A failing `.onnx`/`.tflite` you're allowed to
   share is a gift — open an issue with the file and the full error output.

## Development setup

```bash
git clone https://github.com/locnd182644/fitchip && cd fitchip
pip install -e ".[dev]"          # light core — no TensorFlow needed
python -m pytest                 # the whole suite runs in <1s, offline
ruff check .
```

Optional extras when your change touches them:

```bash
pip install -e ".[quantize]"     # onnx conversion / INT8 (pulls TensorFlow)
pip install -e ".[server]"       # FastAPI orchestrator
pip install -e ".[web]"          # Streamlit GUI
```

## Ground rules
 
- The core must keep working **without** TensorFlow installed. Heavy imports
  go behind function-level imports and fail with a `DEPENDENCY_MISSING`
  normalized error that names the extra to install.
- Backends never raise from `compile()` — they return a
  `CompileResult` with a `NormalizedError` (stable `code`, friendly
  `message`, raw stderr in `raw`, actionable `hints`).
- Op tables, target profiles and manifests are data (YAML), not code.
- On source targets (`c_source_project`), generated output stays 100%
  readable C/C++ — never emit opaque blobs there. Binary artifact kinds
  (`shared_lib`, `serialized_model`) are legitimate, but only for
  OS-based targets that declare them.
- Add a test for what you change; `tests/conftest.py` builds `.tflite`
  fixtures without TensorFlow.

All contributors are credited in release notes.
