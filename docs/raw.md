# Raw bytes

A raw file is the tensor payload and nothing else. Dtype and shape come from the command. There is no raw-file specification. The dtype names are the safetensors names in [Import and export](import-export.md), and `version: raw` is tenspile's label for this headerless file.

## Reading and writing a headerless file

The member name is the file stem. `token=w.bin` written to `blob.raw` lists as `blob`.

```
tenspile -c -f embed.raw --dtype F32 --shape 2 embed.npy
tenspile -i --format raw --dtype F32 --shape 2 -f embed.raw
tenspile -tvf embed.raw --format raw --dtype F32 --shape 2
```

```
format: raw
version: raw
path: embed.raw
members: 1
data_bytes: 8
metadata: none
dtypes:
  F32     1       2
```

```
name    dtype   shape   bytes   offset  params  shard   detail
embed   F32     [2]     8       0       2       -       -
```

The file itself is those 8 bytes:

```
00 00 80 3f 00 00 00 40     f32 1.0, f32 2.0
```

The length has to equal the packed size. A mismatch names both sizes: `embed.bin is 8 bytes; dtype F32 shape [4] needs 16`. `--shape 2x2` is `[2, 2]`. `--shape []` is one element, so `F32` is 4 bytes and `F4` needs two elements to fill one byte.

`--format raw`, or any file whose magic tenspile does not recognize, is read this way when both flags are present. A `.bin` of plain bytes works with those flags. A `.bin`, `.pt`, `.pth`, or `.pkl` that is actually pickle is refused.

There is no metadata. Append and delete are refused: `a raw file holds one tensor`. Update replaces the bytes:

```
tenspile -u -f embed.raw --dtype F32 --shape 2 norm.bin
```

## Extracting to raw

Extracting a headed tensor to raw copies the stored payload. Leave `--dtype` and `--shape` off on that command. The raw file has nowhere to store them, so pass the flags again when that file is the source.

```
tenspile -x -f out.safetensors --format raw -o embed.raw embed
tenspile -x -f model.gguf --format raw -o weight.raw weight
```

An `.npy`, `.npz`, or safetensors extract is the default when the dtype has a NumPy descr. `--format raw` forces the headerless copy, including for `BF16`, `F8_*`, `F6_*`, and `F4`. For a GGUF tensor the raw file is the stored span, padding included. The `.npy` extract of a dense GGUF tensor drops that padding. See [GGUF](gguf.md).

A Fortran-order array and a quantized GGUF tensor also extract this way. The bytes stay in stored order.
