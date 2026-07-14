# FitChip Architecture

FitChip is an **orchestrator, not another compiler**. Its core design goal:
any ML compiler (TVM, ONNX Runtime, TensorRT, ExecuTorch…) and any target
(ESP32, STM32, Jetson, x86…) can be plugged in later **without touching the
core**.

```
 GUI (Streamlit → React)          CLI (fitchip compile ...)
        │                                  │
        ▼                                  ▼
 ┌─────────────────────────────────────────────────────────┐
 │                ORCHESTRATOR (FastAPI)                   │
 │    CompileRequest → select backend → dispatch → result  │
 └──────┬──────────────────┬─────────────────────┬─────────┘
        ▼                  ▼                     ▼
 ┌────────────┐     ┌─────────────┐     ┌──────────────────┐
 │  Model     │     │  Target     │     │ Selection Engine │
 │  Inspector │     │  Registry   │     │   (rule-based)   │
 └────────────┘     └─────────────┘     └────────┬─────────┘
                                                 ▼
 ┌─────────────────────────────────────────────────────────┐
 │          COMPILER ABSTRACTION LAYER (CAL)               │
 │  contract: capabilities / validate / compile / estimate │
 ├──────────────┬──────────────┬───────────────────────────┤
 │ Adapter TFLM │ Adapter TVM  │  future: ORT, TensorRT…   │
 └──────────────┴──────────────┴───────────────────────────┘
```

## The five moving parts

| Component | Code | Extending it |
|---|---|---|
| **CAL** — the contract every compiler adapter implements | `fitchip/core/cal/backend.py` | stable interface; don't break it |
| **Model Inspector** — parses `.tflite`/`.onnx` into neutral `ModelMeta` | `fitchip/core/inspector/` | add a format = add one module |
| **Target Registry** — hardware profiles as YAML | `fitchip/core/targets/*.yaml` | add a board = add a file ([guide](adding-a-target.md)) |
| **Selection Engine** — filter + score over backend manifests | `fitchip/core/selection/engine.py` | never edited when backends are added |
| **Converter Chain** — format-conversion graph (BFS shortest path) | `fitchip/core/convert/chain.py` | add a converter = register one edge |

## The CAL contract

Every backend implements four methods:

- `capabilities()` — declared abilities, read from `manifest.yaml`
- `validate(req, meta)` — cheap pre-compile checks (op coverage, memory forecast); returns `NormalizedError`s
- `estimate(req, meta)` — arena/flash estimate shown *before* the user compiles
- `compile(req, workspace)` — the real work; returns a `CompileResult`

Errors are normalized across backends (`NormalizedError`: stable `code`,
friendly `message`, raw compiler stderr, actionable `hints`). This error
vocabulary is deliberately stable — it is the input for the future
AI-assisted error diagnosis.

## Selection

1. **Hard filter** — input format reachable through the converter chain?
   target matches the manifest's `targets.match` rules? quantization mode
   supported?
2. **Validate** — `backend.validate()`: op coverage, memory fit.
3. **Score** — `priority + op_coverage·w₁ + memory_fit·w₂ − conversion_hops·w₃`.
4. **Fallback** — ranked losers form the fallback chain; if the winner's
   compile fails, the next candidate is tried automatically.

## Execution: MVP vs. wave 2

The MVP runs backends **in-process** — TFLM and ExecuTorch are pure
Python + codegen, need no isolation. The seams for wave 2 are already in
place, so the switch is an adapter swap, not a rewrite:

- **`JobRunner`** (`fitchip/orchestrator/jobs.py`) — the execution seam.
  Endpoints hand compile work to a runner; the MVP `InProcessRunner` uses a
  single worker thread (so compiles never block the event loop and the
  `/v1/inspect` fast lane stays fast) and expires finished jobs after a TTL.
  Wave 2 implements the same protocol on Celery + Redis — endpoints stay
  unchanged.
- each backend gets its own Docker image (declared as `docker_image` in the
  manifest, currently unused),
- the orchestrator's `/v1/compile` already returns a `job_id` and clients
  can already poll `/v1/jobs/{id}`, so they survive the switch to async
  (202 + poll) unchanged.

The rule: infrastructure is added when the need lands, not before.
Concretely:

| Infrastructure | Add it when | Not when |
|---|---|---|
| Celery + Redis | a backend compiles for >~15 min **locally inside the orchestrator** (e.g. TVM autotuning) | the TVM adapter lands but heavy autotuning runs in FitChip Cloud |
| Docker per backend | a hosted service accepts models from strangers (sandboxing untrusted uploads), **or** two backends have unresolvable dependency conflicts | a backend merely being "big" |
| Cheaper middle step | dependency conflicts alone → try per-backend subprocess + venv before reaching for Docker | |
