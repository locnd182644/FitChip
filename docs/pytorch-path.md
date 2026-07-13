# The PyTorch → MCU path

FitChip compiles PyTorch models to Cortex-M boards through the
**ExecuTorch backend** — no ONNX detour. This page is the supported recipe.

## TL;DR

```python
# In your training environment (torch >= 2.5):
import torch

model.eval()
example = (torch.zeros(1, 1, 28, 28),)          # one real-shaped input
ep = torch.export.export(model, example)
torch.export.save(ep, "model.pt2")
```

```bash
pip install 'fitchip[executorch]'
fitchip compile model.pt2 --target stm32f746 --quantize int8 \
    --calibration-data samples.npy
```

Output: a CMake project (static library + reference runner + pinned
ExecuTorch fetch) for the arm-none-eabi toolchain. See the generated
README for how it links into your CubeMX/BSP firmware.

## Which file should I export?

| You have | Do this |
|---|---|
| a live `nn.Module` | `torch.export.save(...)` → **`.pt2`** (preferred: enables PT2E INT8) |
| an already-lowered program | ship the **`.pte`** with `--quantize none` |
| a TorchScript archive (`.pt`) | works via the 2-hop `pt → onnx → tflite` route to ESP32; for Cortex-M re-export as `.pt2` |
| a `state_dict` checkpoint (`.pt`/`.pth`/`.ckpt`) | cannot be compiled — re-export from training code (FitChip rejects it with this exact guidance) |

## Quantization contract

- **`.pt2` + `--quantize int8`** runs the PT2E flow during lowering:
  prepare → calibrate → convert → lower. Pass `--calibration-data
  samples.npy` (a float32 array, first dim = sample index, already
  preprocessed); without it FitChip calibrates on random data and warns.
- **`.pte` + `--quantize int8` fails** with `QUANTIZE_FAIL`: a lowered
  program cannot be post-quantized without its source ExportedProgram.
  This mirrors the TFLM policy for float32 `.tflite` files.

## Memory numbers you can trust

A `.pte` carries its memory plan, so the activation ("arena") number
FitChip reports for `.pte` inputs is **exact** — read from the plan, no
1.25× safety factor. For `.pt2` inputs the exact number appears in the
compile report after lowering. Flash remains a labeled heuristic
(program size + ~250 KB runtime).

## Known limits

- Data-dependent control flow needs `torch.cond` rewrites before
  `torch.export` succeeds — export errors happen in *your* environment,
  where they are actionable.
- CMSIS-NN kernel coverage is narrow; ops outside the table in
  `ops_executorch.yaml` run on portable kernels (correct but slower).
  `fitchip inspect model.pt2 --target stm32f746` lists what accelerates.
- The generated project is a library + runner, **not** a complete
  firmware image: startup code, linker script and HAL come from your BSP.
