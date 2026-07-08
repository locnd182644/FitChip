# Adding a Hardware Target

A target is **one YAML file** — no code. Drop it into `fitchip/core/targets/`
(or point `FITCHIP_TARGETS_DIR` at your own directory) and it shows up in
`fitchip targets`.

```yaml
# targets/esp32s3.yaml
id: esp32s3               # what users type after --target
display_name: "ESP32-S3"
vendor: espressif
isa: xtensa-lx7           # matched against backend manifests
ram_kb: 512               # budget for the tensor-arena check
psram_kb: 8192            # optional
flash_kb: 8192            # budget for the flash-footprint check
has_os: false             # bare-metal (false) or Linux-class (true)
accelerators: [esp-nn-simd]
toolchains: [esp-idf, platformio]
```

Field notes:

- `ram_kb` should be the **usable** RAM for the application, not the chip's
  headline SRAM (e.g. ESP32 ships 520 KB SRAM but ~320 KB is realistically
  available to the arena).
- `isa` and `has_os` are what backend manifests match against — check
  `backends/*/manifest.yaml` to see which backends will pick your board up.
- If the board needs a new PlatformIO board id or IDF chip name for project
  generation, add it to the maps at the top of
  `fitchip/backends/tflm/codegen.py` (one line each).

Then verify:

```bash
fitchip targets                              # your board is listed
fitchip inspect model.tflite --target yourboard
python -m pytest tests/test_targets.py
```

PR checklist: YAML file + (if you own the hardware) measured numbers for the
op-support table.
