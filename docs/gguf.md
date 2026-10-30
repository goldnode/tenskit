# GGUF

tenspile lists a little-endian GGUF v2 or v3 file, prints its metadata, extracts tensors, and rewrites metadata by copying the tensor blob. Quantized tensors stay opaque block bytes. Creating a file, or adding, replacing, or deleting a tensor, is refused.

The layout and type table are `gguf.yaml`. See [Import and export](import-export.md) for operands.

## Format version

`version: 2` or `version: 3` on `-i` is the little-endian number after the `GGUF` magic. A model name, a filename token, and the metadata key `general.version` are separate fields.

| Version | What tenspile does |
| --- | --- |
| 1 | Refuses the file. Counts and lengths are 32-bit. |
| 2 | Reads it. A metadata rewrite keeps version 2. |
| 3 | Reads the little-endian form. A big-endian file is refused. |

`ggml`, `ggmf`, and `ggjt` stop with `is legacy GGML/GGMF/GGJT; tenskit reads GGUF`. A removed ggml type code stops with `was removed from GGUF`. Type 40 is `NVFP4`. `COUNT` is the sentinel after the last type, not a tensor type.

## List and info

Shapes on `-t` and `-i` are ggml `ne[]`: `ne[0]` is the innermost axis. A note says that a conversion reverses them. `bytes` is the tensor's own length, from its type and shape. When alignment padding follows a tensor, `detail` says how many bytes, for example `16 bytes of alignment padding follow the tensor`.

```
$ tensinfo -t small.gguf
file:         small.gguf (gguf v3), 1.4 KiB of tensor data
model:        llama
parameters:   532 in 5 tensors
dtypes:       F32 48%, Q8_0 48%, F16 2%, BF16 1%, I32 1%
quantization: GGUF blocks Q8_0
tokenizer:    none: no tokenizer.ggml.model, so llama.cpp will not load the file

name    dtype  shape    bytes  offset  params  detail
w_f32   F32    [64, 4]  1 KiB     384     256
b_f16   F16    [8]       16 B    1408       8  16 bytes of alignment padding follow the tensor
i_i32   I32    [2, 3]    24 B    1440       6  8 bytes of alignment padding follow the tensor
e_bf16  BF16   [3, 2]    12 B    1472       6  20 bytes of alignment padding follow the tensor
w_q8_0  Q8_0   [64, 4]  272 B    1504     256  16 bytes of alignment padding follow the tensor
```

A key `<name>.orig_shape` restores a shape that GGUF cannot store, such as a scalar written as `ne = [1]`, and is not listed as metadata (GGUF-023).

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
tenspile -u -f model.gguf -o renamed.gguf --metadata general.name=renamed
```

`-o` writes a new file and leaves the original alone.

`--metadata` writes a string. In `--metadata-file`, each value is a JSON object with `type` and `value`. Types: `uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `uint64`, `int64`, `float32`, `float64`, `bool`, `string`, `array`. An array also needs `elem`, its element type, and is one level deep. `general.alignment` has to be a power of two and at least 8. A missing key means 32. A file such as `meta.json`:

```json
{
  "general.name": {"type": "string", "value": "demo"},
  "llama.context_length": {"type": "uint32", "value": 4096},
  "llama.rope.freq_base": {"type": "float32", "value": 10000.0},
  "tokenizer.ggml.add_bos_token": {"type": "bool", "value": true},
  "demo.ids": {"type": "array", "elem": "int32", "value": [1, 2]}
}
```

Converting to and from GGUF, and carrying its metadata into other formats, is [tensconv](tensconv.md). Dequantizing its blocks is [tensquant](tensquant.md).
