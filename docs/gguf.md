# GGUF

tenspile lists a little-endian GGUF v2 or v3 file, prints its metadata, extracts tensors, and rewrites metadata by copying the tensor blob. Quantized tensors stay opaque block bytes. Creating a file, or adding, replacing, or deleting a tensor, is refused.

The layout and type table are [gguf.yaml](../tensdataspecs_readonly/specs/formats/gguf.yaml). See [Import and export](import-export.md) for operands.

## Format version

`version: 2` or `version: 3` on `-i` is the little-endian number after the `GGUF` magic. A model name, a filename token, and the metadata key `general.version` are separate fields.

| Version | What tenspile does |
| --- | --- |
| 1 | Refuses the file. Counts and lengths are 32-bit. |
| 2 | Reads it. A metadata rewrite keeps version 2. |
| 3 | Reads the little-endian form. A big-endian file is refused. |

`ggml`, `ggmf`, and `ggjt` stop with `is legacy GGML/GGMF/GGJT; tenspile reads GGUF`. A removed ggml type code stops with `was removed from GGUF`. Type 40 is `NVFP4`. `COUNT` is the sentinel after the last type, not a tensor type.

## List and info

Shapes on `-t` and `-i` are ggml `ne[]`: `ne[0]` is the innermost axis. A note says that a conversion reverses them. `bytes` is the span from this tensor's offset to the next tensor, padding included. A quantized tensor says `quantized span includes alignment padding` in `detail`.

## Extract

A dense type with a NumPy descr extracts as `.npy`, `.npz`, or safetensors. Those outputs drop alignment padding and reverse `ne[]` to the canonical shape. `BF16` copies into safetensors or rawtensor and has no `.npy` descr. A quantized type extracts with `--format raw`: the manifest stores the canonical id (for example `ggml_q8_0`), the reversed shape, and the block bytes without the file's alignment padding.

```
tenspile -x -f model.gguf -C npy weight bias
tenspile -x -f model.gguf --format raw -o weight.rawtensor.json weight
tenspile -cf copied.safetensors weight=model.gguf:weight
```

`--format gguf` on extract stops with `extract cannot write GGUF`. Adding or deleting a tensor stops with `GGUF tensors cannot be added, replaced, or deleted`. `tenspile -c -f new.gguf` stops with `creating a GGUF file is not supported`.

## Metadata

`-u` with `--metadata` or `--metadata-file` rewrites metadata and copies the tensor blob. Offsets inside the blob stay the same.

```
tenspile -u -f model.gguf --metadata general.name=renamed
tenspile -u -f model.gguf --metadata-file meta.json
```

`--metadata` writes a string. Other values are a JSON object with `type` and `value`. Types: `uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `uint64`, `int64`, `float32`, `float64`, `bool`, `string`, `array`. An array is one level deep. `general.alignment` has to be a power of two and at least 8. A missing key means 32.
