# Import and export

tenspile moves tensors between containers. Create, append, and update read source files into an archive. Extract writes members back out. List and info read headers and leave the weight bytes on disk.

- [safetensors](safetensors.md)
- [NumPy `.npy` and `.npz`](numpy.md)
- [raw bytes](raw.md)
- [GGUF](gguf.md)
- [shard index](shards.md)

Each of those pages links to the specification it follows. `version` on `-i` is the file's own format version when it has one: `2` or `3` for little-endian GGUF, and `1.0`, `2.0`, or `3.0` for `.npy`. The other labels are tenspile's: `header+blob`, `index`, `zip`, and `raw`.

## Operands

```
tenspile -cf out.safetensors embed=embed.npy
tenspile -cf out.safetensors embed=embed.bin --dtype F32 --shape 2
tenspile -cf out.safetensors pack.npz
tenspile -cf out.safetensors embed=model.gguf:weight
tenspile -xf model.safetensors -C out embed
tenspile -xf model.safetensors -o layer.npz embed norm
```

| Operand | Meaning |
| --- | --- |
| `KEY=PATH` | Read that file as one tensor named `KEY`. |
| a bare `.npy` path | Same, with the file stem as the key. |
| a bare `.npz` path | Read every array. The array names are the keys. |
| `KEY=ARCHIVE:MEMBER` | Copy one member from safetensors, GGUF, `.npz`, or a shard index. |
| a bare name | On list, extract, and delete, select that member. |

A path that exists is a file, including a name that contains a colon. `ARCHIVE:MEMBER` applies when the whole string is not a file and the text before the last colon is a file.

`--dtype` and `--shape` describe every headerless source in that command. On a file that already has a header they have to match the header. One command carries one dtype and one shape, so two raw files of different shapes take two commands. Headed files of different shapes can share a command when those flags are left off.

`--shape` accepts `2`, `2,2`, `2x2`, and `[2, 2]`. `[]` is a scalar: one element.

## Where a tensor can go

| Source | safetensors | `.npy` | `.npz` | raw | GGUF | shard index |
| --- | --- | --- | --- | --- | --- | --- |
| safetensors member | yes | with a NumPy descr | with a NumPy descr | yes | — | edit an existing index |
| `.npy` | yes | one array | yes | yes | — | edit an existing index |
| `.npz` member | yes | with a NumPy descr | with a NumPy descr | yes | — | edit an existing index |
| raw bytes | yes | with a NumPy descr | with a NumPy descr | yes | — | edit an existing index |
| GGUF `F32`, `F16`, `F64`, `I8`, `I16`, `I32`, `I64` | yes, padding dropped | yes, padding dropped | yes, padding dropped | yes, stored span | metadata rewrite | — |
| GGUF `BF16` | yes | — | — | yes | metadata rewrite | — |
| GGUF quantized type | — | — | — | yes | metadata rewrite | — |

A cell marked — is refused before a write. The GGUF rows are little-endian format versions 2 and 3. Version 1 and a big-endian file are refused, as described in [GGUF](gguf.md). GGUF tensors cannot be added or deleted. A shard index is edited; creating one is refused. Bytes are copied as stored. The dtype and shape stay as they are.

Extract writes one `.npy` per member when the dtype has a NumPy descr. `BF16`, `F4`, `F6_E2M3`, `F6_E3M2`, and the `F8_*` names are written as `.raw`. `--format npy` or `--format raw` forces that choice, and `--format` wins over the output extension.

`--format npz`, `--format safetensors`, or an `-o` name ending in `.npz` or `.safetensors`, writes one archive. Several members and an `-o` that is neither of those need an explicit archive format. A tensor named `model.layers.0.weight` becomes `model.layers.0.weight.npy`. The dots stay.

`-C` is the directory for per-member files. The default is the current directory. `--force` overwrites an existing output file. `-n` prints the plan and writes nothing. A rewrite uses a temporary file in the destination directory, then renames it into place.

## Dtypes that round-trip through `.npy`

Safetensors names are the [`Dtype` enum](https://github.com/huggingface/safetensors/blob/e246a2560645b7525f5775669ed816eb57c5bcc8/safetensors/src/tensor.rs#L808-L910) at commit `e246a25`. The descrs are the little-endian NumPy forms from the [`_format_impl.py`](https://github.com/numpy/numpy/blob/3ae3cc9657c32f530700831273bca9f680488403/numpy/lib/_format_impl.py) docstring at commit `3ae3cc9`. tenspile writes the descrs in the table. On read it also accepts `?` for `BOOL`, and `u1` / `i1` for `U8` / `I8`.

| Name | `.npy` descr |
| --- | --- |
| `BOOL` | `\|b1` |
| `U8` | `\|u1` |
| `I8` | `\|i1` |
| `U16` | `<u2` |
| `I16` | `<i2` |
| `F16` | `<f2` |
| `U32` | `<u4` |
| `I32` | `<i4` |
| `F32` | `<f4` |
| `U64` | `<u8` |
| `I64` | `<i8` |
| `F64` | `<f8` |
| `C64` | `<c8` |

These names have no `.npy` descr. They stay in safetensors or in a raw file:

`BF16`, `F4`, `F6_E2M3`, `F6_E3M2`, `F8_E4M3`, `F8_E5M2`, `F8_E8M0`, `F8_E4M3FNUZ`, `F8_E5M2FNUZ`.

`F4` is 4 bits and `F6_*` is 6 bits. The shape has to fill a whole number of bytes. Two `F4` values are one byte.

A big-endian descr and a structured descr are listed as that descr text. The default extract is a raw payload. Import into safetensors stops because those descrs have no safetensors dtype. An object-dtype `.npy` is refused on open: it can carry pickle. A Fortran-order array can be listed, and its default extract is raw. The element order stays as stored.

## What list and info print

`tenspile -t -f out.safetensors` prints one name per line. `-v` adds columns, separated by tabs. The sample is aligned with spaces:

```
name    dtype   shape   bytes   offset  params  shard   detail
embed   F32     [2]     8       104     2       -       -
```

`bytes` is the stored length. For a GGUF tensor, that includes alignment padding up to the next tensor. `offset` is where that payload starts in the file. Inside `.npz` it is the offset inside that array's `.npy` entry, so two arrays can both report offset 128. `params` is the element count. `shard` is filled for a shard index. `detail` is a padding note, `fortran_order`, `zip deflate`, or the shard file name.

`tenspile -i -f out.safetensors` adds format, version, member count, data bytes, metadata, and a dtype tally (`dtype`, count, element count). `-i --json` and `-t --json` print one document. `-t` limits `members` to the names you gave. Keys are sorted:

```json
{
  "data_bytes": 8,
  "format": "safetensors",
  "members": [
    {
      "bytes": 8,
      "detail": null,
      "dtype": "F32",
      "name": "embed",
      "numel": 2,
      "offset": 104,
      "shape": [
        2
      ],
      "shard": null
    }
  ],
  "metadata": {
    "format": "pt"
  },
  "notes": [],
  "path": "out.safetensors",
  "version": "header+blob"
}
```

A larger file repeats objects in `members`. Metadata values in this document are strings.

## Files that are refused

A PyTorch pickle — `.pt`, `.pth`, a zip of `.pkl`, or bytes that start with a pickle opcode — stops with `is a PyTorch pickle file and was not opened`. A `.bin` whose bytes are ordinary tensor data can be imported as raw when `--dtype` and `--shape` are set. Legacy `ggml`, `ggmf`, and `ggjt` stop with `is legacy GGML/GGMF/GGJT; tenspile reads GGUF`.
