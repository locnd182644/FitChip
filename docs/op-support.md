# Operator Support

The source of truth is machine-readable:
[`fitchip/backends/tflm/ops_tflm.yaml`](../fitchip/backends/tflm/ops_tflm.yaml)
— ~95 TFLite builtin ops supported by TFLite Micro, with the esp-nn
accelerated subset flagged (`esp_nn: true`: CONV_2D, DEPTHWISE_CONV_2D,
FULLY_CONNECTED, ADD, MUL, pooling, RELU/RELU6/PRELU, SOFTMAX).

Check a specific model instead of reading tables:

```bash
fitchip inspect model.tflite --target esp32s3
```

## Verified on real hardware

"Supported" above means *the kernel exists*. The column that matters —
*measured working on a physical board* — is community-filled:

| Op | ESP32 | ESP32-S3 | ESP32-C3 |
|---|---|---|---|
| CONV_2D (int8) | ✅ | ✅ (SIMD) | 🙋 needs tester |
| DEPTHWISE_CONV_2D (int8) | ✅ | ✅ (SIMD) | 🙋 needs tester |
| FULLY_CONNECTED (int8) | ✅ | ✅ (SIMD) | 🙋 needs tester |
| SOFTMAX (int8) | ✅ | ✅ | 🙋 needs tester |

Tested a board we haven't? Run the generated project's boot log (it prints
arena usage and per-inference latency) and PR your numbers — see
[CONTRIBUTING.md](../CONTRIBUTING.md).
