# NumPy `.npy` and `.npz`

`.npy` is one array. `.npz` is a zip of `.npy` members. Both can be listed, and both are sources for the other containers.

The readers follow `npy.yaml` and `npz.yaml`. Writers emit `.npy` version 1.0 with the payload on a 64-byte boundary. Versions 1.0, 2.0, and 3.0 are read. A file whose data is not 64-byte aligned is still read, and `-i` notes NPY-005.

See [Import and export](import-export.md) for operands. Descr codes come from `dtype-mapping.json`. On read, `?` is accepted for `BOOL`, and `u1` / `i1` for `U8` / `I8`.

## One `.npy`

The member name is the file stem. Append and delete are refused, because the file holds one array. Update replaces that array. There is no metadata map. `--metadata` stops with `npy files have no string metadata`.

```
tenspile -c -f embed.npy --dtype F32 --shape 2 embed.bin
tenspile -i -f embed.npy
tenspile -u -f embed.npy --dtype F32 --shape 2 norm.bin
```

`version:` on `-i` is the file's major.minor. `descr` is the NumPy header field.

Fortran order can be listed. Import into safetensors stops with `is fortran-order; tenspile does not transpose it`. Extract to `.npy` writes the array as stored, with `fortran_order: True`, so `tenspile -xf` works on any `np.savez` archive. Extract to `.npz` or safetensors stops with the same message. `--format raw` stops as well, because rawtensor bytes are C order (RAW-009). A big-endian descr such as `>f4` or `>c8` is read as its little-endian dtype and the bytes are swapped on the way in; a complex value swaps each part. numpy never writes the native order `=`; a hand-written `=f4` is read as `<f4` with a note (NPY-013). An object dtype (`|O`) is refused on open.

A structured descr is listed as that descr text. Import into safetensors stops because the descr has no safetensors dtype. bfloat16 saved by NumPy is a void descr such as `<V2`. The spec treats that as ambiguous, so tenspile does not guess that it is `bfloat16`. The bytes can still be listed.

## An `.npz` archive

A bare `.npz` path on create, append, or update imports every array. `KEY=pack.npz` is refused. One array is `KEY=pack.npz:embed`.

```
tenspile -c -f pack.npz embed.npy
tenspile -r -f pack.npz norm=norm.bin --dtype F32 --shape 2
tenspile -tvf pack.npz
tenspile -u -f pack.npz embed=embed.npy
tenspile --delete -f pack.npz norm
```

`version: zip` is tenspile's label. Each entry is written as `{key}.npy` with zip method stored. A deflate entry still lists, and `-v` puts `zip deflate` in `detail`. A member is an array when it starts with the npy magic; any other member is skipped with a note (NPZ-007). The key is the member name with one `.npy` removed, and a `/` in it is kept. Two members with the same key are refused (NPZ-006). An empty archive lists with no arrays (NPZ-005). A ZIP with neither `.npy` members nor a PyTorch `data.pkl` is not a tensor file. An edit copies every member it does not change as stored: name, header, bytes, and compression (NPZ-008, NPZ-009). So an array with a descr that has no safetensors type (such as `<V2`), a member that is not an array, and a deflated member all survive unchanged. A replaced array keeps its member's compression. A new array is deflated when every member of the archive is deflated, as in a `np.savez_compressed` file, and stored otherwise. The last array cannot be removed. `--metadata` stops with `npz files have no string metadata`.

`offset` on `-v` is the file offset of a stored entry's array data, and 0 for a deflated entry. `data_bytes` is the sum of the arrays.

```
tenspile -xf pack.npz -C out
tenspile -cf again.safetensors pack.npz
tenspile -cf one.safetensors embed=pack.npz:embed
```
