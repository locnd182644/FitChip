# ONNX Conversion Notes

ONNX input reaches MCU targets through an automatic `onnx → tflite` hop
(via [onnx2tf](https://github.com/PINTO0309/onnx2tf)), which requires the
conversion extra:

```bash
pip install 'fitchip[quantize]'
```

## Quantization happens at conversion time

This is the part people trip over, so FitChip is explicit about it:

- **`.onnx` + `--quantize int8`** ✅ — onnx2tf drives TFLite full-integer
  quantization during conversion. Pass `--calibration-data samples.npy`
  (a preprocessed array of representative inputs); without it, random
  calibration is used and FitChip prints an accuracy warning.
- **float32 `.tflite` + `--quantize int8`** ❌ — TFLite has no supported path
  to quantize an already-exported flatbuffer without its source model.
  FitChip fails with `QUANTIZE_FAIL` instead of silently shipping a float
  model. Export with PTQ enabled in your training framework, or provide the
  ONNX instead.
- **already-quantized `.tflite`** ✅ — used as-is.

## Known-problematic graphs

Conversion coverage is good but not universal. `fitchip inspect model.onnx
--target esp32s3` tells you up front whether the graph survives the trip.
Frequent offenders:

| Symptom | Fix |
|---|---|
| Dynamic input shapes | Re-export with a fixed batch size (FitChip assumes batch=1) |
| `LSTM` / `GRU` nodes | Unroll at export time, or use keras → tflite directly |
| `GELU`, exotic activations | Export with an approximation flag (most exporters have one) |
| NCHW-heavy graphs with many `Transpose` | Usually fine — onnx2tf removes most of them; check op count after conversion |
| Ops in custom domains | Not convertible; rewrite the graph before export |

When conversion fails, the `CONVERT_FAIL` error carries the raw onnx2tf
output plus hints; simplifying the graph first
(`onnxsim in.onnx out.onnx`) resolves a surprising share of failures.
