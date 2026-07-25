# GGUF

tenspile lists a GGUF file, prints its metadata, extracts tensors into another container, and rewrites metadata by copying the tensor blob. Quantized tensors stay opaque bytes. Creating a file, or adding, replacing, or deleting a tensor, is refused. The messages are at the end of [Metadata](#metadata).

See [Import and export](import-export.md) for the shared operand and dtype rules.

## Format version

`version: 2` or `version: 3` on `-i` is the file-format version: the little-endian number immediately after the `GGUF` magic. A model name such as `Llama-3`, the `v1.0` token in a [GGUF filename](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/docs/gguf.md#gguf-naming-convention), and the metadata key `general.version` are separate fields.

tenspile reads little-endian versions 2 and 3. They use the same field widths, so listing, extraction, and a metadata rewrite behave the same. The rewrite writes the file's own version back, and a version 2 file stays version 2. Current writers emit version 3. The layout is the ggml [file structure](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/docs/gguf.md#file-structure) at commit `3612c0a` (2026-10-12).

| Version | What tenspile does |
| --- | --- |
| 1 | Refuses the file. Counts and lengths are 32-bit. See [Older versions](#older-versions). |
| 2 | Reads it. Same little-endian layout as version 3. |
| 3 | Reads the little-endian form. A big-endian file is refused. |

A version word that reads as more than 65536 is treated as big-endian. A big-endian 3 is the bytes `00 00 00 03`, and it stops with `looks like big-endian GGUF, which tenspile does not read`. The spec defines no endian flag. Any other version stops with `model.gguf: GGUF version 4 is not supported; tenspile reads little-endian versions 2 and 3`.

A new metadata key does not change the format version. tenspile prints an unknown key and can replace it.

`ggml`, `ggmf`, and `ggjt` are older containers. They stop with `is legacy GGML/GGMF/GGJT; tenspile reads GGUF`.

## List and info

This file has two `F32` vectors, `weight` and `bias`, each with two values. Alignment is 32. `weight` is stored in a 32-byte slot so that `bias` stays aligned; the eight payload bytes are `1.0` and `2.0`.

```
tenspile -i -f model.gguf
```

```
format: gguf
version: 3
path: model.gguf
members: 2
data_bytes: 40
metadata:
  general.architecture=test
  general.alignment=32
  general.name=demo
dtypes:
  F32     2       4
note: alignment 32
note: data starts at 224
```

A version 2 file of this same layout prints `version: 2`. Listing, extraction, and a metadata rewrite are the same, and the rewrite leaves the version word at 2.

```
tenspile -tvf model.gguf
```

```
name     dtype  shape  bytes  offset  params  shard  detail
weight   F32    [2]    32     224     2       -      stored span 32 includes padding; dense payload is 8 bytes
bias     F32    [2]    8      256     2       -      -
```

`bytes` is the span from this tensor's offset to the next tensor, or to the end of the blob. `data_bytes` is the whole blob, padding included. A quantized tensor says `quantized span includes alignment padding` in `detail`.

On disk the file is the magic `GGUF`, the version `uint32`, the metadata pairs, the tensor records, padding out to `general.alignment`, then the blob. A production file's metadata is long. `-i` prints each value, shortens a string past 120 characters to a prefix plus its byte length, and prints an array of more than eight elements as `array<string>[32000]`.

## Extract

A dense type with a NumPy descr (`F32`, `F16`, `F64`, `I8`, `I16`, `I32`, `I64`) extracts as `.npy`, `.npz`, or safetensors. Those outputs use the logical payload, so the alignment padding is left behind. `weight.npy` from this file is 8 payload bytes, while the raw span is 32.

```
tenspile -x -f model.gguf -C npy weight bias
tenspile -x -f model.gguf --format raw -o weight.raw weight
```

`BF16` has a fixed width and can be copied into safetensors or raw. It has no `.npy` descr, so an `.npy` extract stops with `dtype BF16 has no .npy encoding; use --format raw`.

A quantized type (`Q4_0`, `Q4_K`, `IQ4_NL`, and the other GGML block types) has no logical dense size here. Extract it with `--format raw`. The message is `tensor weight is a quantized span; extract it with --format raw`. The raw file is the stored span, including padding. The values are still quantized.

Copy one dense tensor into a new safetensors file:

```
tenspile -cf copied.safetensors weight=model.gguf:weight
```

`--format gguf` on extract stops with `extract cannot write GGUF; choose npy, raw, npz, or safetensors`.

## Metadata

`-u` with `--metadata` or `--metadata-file` is the GGUF write. The tensor records and the blob are copied. Offsets inside the blob stay the same. A longer header moves `data starts at`.

```
tenspile -u -f model.gguf --metadata general.name=renamed
```

`--metadata` writes a string. Numbers, bools, and arrays go in a JSON object. A bare JSON string is also a GGUF string. Anything else is an object with `type` and `value`:

```json
{
  "general.name": { "type": "string", "value": "typed" },
  "general.file_type": { "type": "uint32", "value": 1 },
  "tokenizer.ggml.tokens": {
    "type": "array",
    "elem": "string",
    "value": ["hello", "world"]
  }
}
```

```
tenspile -u -f model.gguf --metadata-file meta.json
```

Types: `uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `uint64`, `int64`, `float32`, `float64`, `bool`, `string`, `array`. An array is one level deep. `elem` names the element type. Up to eight elements print in full on `-i`; past that, the line is `array<string>[N]`.

`general.alignment` is a `uint32`. The [spec](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/docs/gguf.md#file-structure) says it is a multiple of 8, and that a missing key means 32. tenspile accepts a power of two that is at least 8. Every tensor offset in the blob has to be a multiple of the new value.

Adding or deleting a tensor stops with `GGUF tensors cannot be added, replaced, or deleted; a metadata update copies the tensor blob`. `tenspile -c -f new.gguf` stops with `creating a GGUF file is not supported`.

## Older versions

Little-endian version 2 and version 3 are both supported. They use the same layout. List, extract, and metadata rewrite work the same way on either file. A metadata rewrite keeps the version number already stored in the file, so a version 2 file stays version 2.

These format numbers are not the `v1.0` token in a filename, and they are not the older `ggml`, `ggmf`, and `ggjt` containers.

Version 1 is not supported. Its tensor count, metadata count, string lengths, array lengths, and dimension sizes are 32-bit. Version 2 and version 3 store those as 64-bit, so opening a version 1 file with this layout would read the wrong number of bytes. The dimension count is 32-bit in version 2 and version 3 as well, and the tensor-data offset is 64-bit in both. Version 1 has no `uint64`, `int64`, or `float64` metadata values. It was only written for a few days in August 2023. Opening one stops with:

```
model.gguf: GGUF version 1 is not supported. That layout used 32-bit counts, string lengths, array lengths, and dimension sizes, and it was only written from 21 to 27 August 2023. llama.cpp removed the version 1 reader on 2 November 2023. tenspile reads little-endian versions 2 and 3
```

A big-endian file is not supported, including a big-endian version 3. A version 2 file is little-endian. Alignment is the same for version 2 and version 3: a missing `general.alignment` means 32.
