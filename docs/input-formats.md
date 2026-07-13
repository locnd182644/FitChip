# Input formats

FitChip accepts the common "trained model" artifacts. Not every file that
comes out of a training run contains enough information to compile, so the
behavior is honest per group:

| Group | Formats | Behavior |
|---|---|---|
| Self-contained (graph + weights) | `.tflite` · `.onnx` · `.h5`/`.hdf5`/`.keras` · `.pb` (SavedModel dir / frozen graph) · `.pt`/`.pth` **TorchScript** · `.pt2` · `.pte` | inspected, converted where needed, compiled |
| Weights only, no graph | `.ckpt` · `.pt`/`.pth` **state_dict** | rejected cleanly with export guidance (never a traceback) |
| Needs your class code | `.pt` pickled `nn.Module` | rejected cleanly with export guidance |

## Conversion routes (computed, not hardcoded)

The converter chain is a graph; the Selection Engine takes the shortest
path and each hop costs score points:

| From \ Backend | TFLM (needs `.tflite`) | ExecuTorch (needs `.pte`/`.pt2`) |
|---|---|---|
| `.tflite` | 0 hops | — |
| `.onnx` | 1 hop | — |
| `.h5` / `.keras` | 1 hop | — |
| `.pb` | 1 hop | — |
| `.pt` (TorchScript) | 2 hops (`pt → onnx → tflite`) | — (ET needs an ExportedProgram — re-export as `.pt2`) |
| `.pt2` / `.pte` | — | 0 hops (export/lower happens inside `compile()`) |
| `.ckpt` / state_dict | rejected + guidance | rejected + guidance |

"—" means no route: the selection report shows a clean
`FORMAT_UNSUPPORTED` rejection for that backend.

## Notes per format

- **`.keras` / `.h5`** — inspected without TensorFlow (zip config / h5py);
  the `h5 → tflite` conversion itself needs `fitchip[quantize]`.
- **`.pb`** — pass the SavedModel *directory* on the CLI; the web/HTTP
  front-ends take single files only (zip the directory or convert first).
  Frozen graphs are best-effort: prefer the SavedModel.
- **`.pt` / `.pth`** — FitChip distinguishes TorchScript archives from
  `torch.save()` pickles by ZIP layout, without unpickling (torch pickles
  execute arbitrary code; they are never loaded).
- **`.pt2` / `.pte`** — the PyTorch → Cortex-M lane, see
  [pytorch-path.md](pytorch-path.md). INT8 applies to `.pt2` only.
- **Quantization happens at conversion/lowering time, never after.** A
  float32 `.tflite` or a lowered `.pte` cannot be post-quantized; FitChip
  fails with guidance instead of shipping a model that will not fit.
