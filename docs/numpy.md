# NumPy `.npy` and `.npz`

`.npy` is one array. `.npz` is a zip of `.npy` members. Both are archives tenspile can list, and both are sources for the other containers. See [Import and export](import-export.md) for the dtype table and the operand rules.

The reader follows the docstring of [`numpy/lib/_format_impl.py`](https://github.com/numpy/numpy/blob/3ae3cc9657c32f530700831273bca9f680488403/numpy/lib/_format_impl.py) at NumPy commit `3ae3cc9` (2026-10-12). There `ARRAY_ALIGN` is 64. [NEP 1](https://github.com/numpy/numpy/blob/3ae3cc9657c32f530700831273bca9f680488403/doc/neps/nep-0001-npy-format.rst) is the original note. Its alignment figure is 16, and it stops at version 2.0. `.npz` is the zip of `.npy` members described in the same module: `numpy.savez` stores entries uncompressed, and `numpy.savez_compressed` uses deflate.

## One `.npy`

The member name is the file stem. The key on the create command applies while the file is being built. `tenspile -c -f weights.npy token=w.bin --dtype F32 --shape 2` lists as `weights`.

```
tenspile -c -f embed.npy --dtype F32 --shape 2 embed.bin
tenspile -i -f embed.npy
```

```
format: npy
version: 1.0
path: embed.npy
members: 1
data_bytes: 8
metadata:
  descr=<f4
dtypes:
  F32     1       2
```

`descr` is the NumPy header field. There is no metadata map to edit. `--metadata` stops with `npy files have no string metadata`.

Append and delete are refused, because the file holds one array: `a npy file holds one tensor; append into an npz or safetensors file`. Update replaces that array:

```
tenspile -u -f embed.npy --dtype F32 --shape 2 norm.bin
```

`version:` on `-i` is the file's own major.minor. Version 1.0 stores the header length in a `uint16` and encodes the header as Latin-1. Version 2.0 uses a `uint32` length for a large header. Version 3.0 keeps that `uint32` length and encodes the header as UTF-8. tenspile reads all three. It writes version 1.0, padded so the payload starts on a 64-byte boundary. A header that does not fit in a `uint16` stops with `npy header does not fit in version 1.0`. For this two-element file the payload starts at offset 128:

```
93 4e 55 4d 50 59  01 00     \x93NUMPY, version 1.0
76 00                          body length 118; the payload starts at offset 128
{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }
… spaces and a final newline …
00 00 80 3f 00 00 00 40        1.0, 2.0
```

A longer shape is still one dict. `(2, 3)` is rank 2. A one-element shape is written `(2,)` with the trailing comma, which is Python's one-tuple. A scalar is `()`.

Fortran order can be listed. The default extract is the payload in stored order, under a `.raw` name. `--format npy`, `.npz`, and safetensors stop with `member embed is fortran-order; extract it with --format raw`. Import stops with `embed.npy is fortran-order; tenspile does not transpose it`. An object dtype (`|O`) is refused on open.

A big-endian or structured descr is listed as that descr text. The default extract is a raw payload. Import into safetensors stops with `embed.npy descr >f4 has no safetensors dtype`.

## An `.npz` archive

A bare `.npz` path on create, append, or update imports every array. `KEY=pack.npz` is refused. One array is `KEY=pack.npz:embed`.

```
tenspile -c -f pack.npz embed.npy
tenspile -r -f pack.npz norm=norm.bin --dtype F32 --shape 2
tenspile -tvf pack.npz
```

```
name    dtype   shape   bytes   offset  params  shard   detail
embed   F32     [2]     8       128     2       -       -
norm    F32     [2]     8       128     2       -       -
```

`offset` is inside each `.npy` entry, which is why both rows say 128. `tenspile -i` reports `version: zip`, `metadata: none`, and `data_bytes` as the sum of the arrays.

`version: zip` is tenspile's label for the archive. Each entry is written as `{key}.npy` with zip method "stored". A deflate entry, as from `numpy.savez_compressed`, still lists, and `-v` puts `zip deflate` in `detail`. A zip whose members are pickle is refused.

```
tenspile -u -f pack.npz embed=embed.npy
tenspile --delete -f pack.npz norm
```

Update replaces or adds. Delete rewrites the zip. The last array cannot be removed: `an npz archive needs at least one array`. `--metadata` stops with `npz files have no string metadata`.

Inside the zip, a small archive looks like this. A large one continues with one `{name}.npy` per array.

```
PK…                         zip local header
embed.npy                   stored .npy, header plus 8 payload bytes
PK…
norm.npy                    stored .npy, header plus 8 payload bytes
… central directory …
```

Extract the whole archive as `.npy` files, or pull one array back out:

```
tenspile -xf pack.npz -C out
tenspile -cf again.safetensors pack.npz
tenspile -cf one.safetensors embed=pack.npz:embed
```
