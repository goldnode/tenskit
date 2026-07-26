# Safetensors

tenspile can create, list, extract, append, replace, and delete members in a safetensors file. String metadata lives under `__metadata__`.

The layout, dtype names, and validation ids are [safetensors.yaml](../tensdataspecs_readonly/specs/formats/safetensors.yaml). `version: header+blob` is tenspile's label. The file has no format-version field.

See [Import and export](import-export.md) for operands and the JSON listing.

## Create, edit, and list

```
tenspile -cf out.safetensors embed=embed.npy --metadata format=pt
tenspile -tvf out.safetensors
tenspile -i -f out.safetensors
```

Append adds a missing key. Update replaces a key or adds it. Delete rewrites the file without the named members. An append of a key that is already there stops with `tensor embed already exists; use -u to replace it`, and the file is left untouched.

```
tenspile -r -f out.safetensors --dtype F32 --shape 2 norm=norm.bin
tenspile -u -f out.safetensors --metadata-file meta.json
tenspile --delete -f out.safetensors norm
```

`meta.json` is an object of strings. A value that is not a string stops with `metadata kind is not a string; safetensors metadata is text`. The tensor name `__metadata__` is reserved.

Copy one tensor out of another container:

```
tenspile -cf copied.safetensors w=out.safetensors:embed
```

## Extract

With no `--format`, each member becomes an `.npy` when the dtype mapping has a NumPy descr.

```
tenspile -xf out.safetensors -C out embed
tenspile -x -f out.safetensors --format raw -o embed.rawtensor.json embed
tenspile -x -f out.safetensors --format safetensors -o layer.safetensors embed norm
```

The rawtensor extract is a manifest plus a sibling `.bin`. It records the canonical dtype id. Several members and `--format safetensors` write one new file with the tensors and no metadata.

## What the reader enforces

The checks are the `validation` list in the spec. Holes, overlaps, trailing bytes, duplicate keys, and a dtype outside the spec are refused. A header longer than 100 MiB is refused until `--max-header-size` raises the cap (ST-002). A header whose length is not a multiple of 8 is still read, and `-i` notes ST-016. F4 and F6 tensors are copied as bytes; `-i` notes ST-Q2 and ST-Q1 because the packing order is still open.
