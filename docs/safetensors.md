# Safetensors

A safetensors file is an 8-byte header length, a JSON object padded with spaces to a multiple of 8, then the tensor bytes packed in header order. tenspile can create it, list it, extract it, append, replace, and delete members. String metadata lives under `__metadata__`.

The layout is the [format section](https://github.com/huggingface/safetensors/blob/e246a2560645b7525f5775669ed816eb57c5bcc8/README.md#format) of the safetensors README at commit `e246a25` (2026-09-22). Dtype names are the [`Dtype` enum](https://github.com/huggingface/safetensors/blob/e246a2560645b7525f5775669ed816eb57c5bcc8/safetensors/src/tensor.rs#L808-L910) in `safetensors/src/tensor.rs` at that commit. The JSON Schema dtype list is behind that enum, and the enum is the list this tool accepts. `version: header+blob` is tenspile's name for this layout. The file has no format-version field.

See [Import and export](import-export.md) for operands, dtypes, and the JSON listing.

## Create, edit, and list

`embed.npy` holds two little-endian `f32` values, `1.0` and `2.0`.

```
tenspile -cf out.safetensors embed=embed.npy --metadata format=pt
tenspile -tvf out.safetensors
```

```
name    dtype   shape   bytes   offset  params  shard   detail
embed   F32     [2]     8       104     2       -       -
```

```
tenspile -i -f out.safetensors
```

```
format: safetensors
version: header+blob
path: out.safetensors
members: 1
data_bytes: 8
metadata:
  format=pt
dtypes:
  F32     1       2
```

Append adds a missing key. Update replaces a key or adds it. Delete rewrites the file without the named members. An append of a key that is already there stops with `tensor embed already exists; use -u to replace it`, and the file is left untouched.

```
tenspile -r -f out.safetensors --dtype F32 --shape 2 norm=norm.bin
tenspile -u -f out.safetensors --metadata-file meta.json
```

`meta.json` is an object of strings. Flags after the file replace the same key.

```json
{ "format": "pt", "kind": "demo" }
```

```
metadata:
  format=pt
  kind=demo
```

A value that is not a string stops with `metadata kind is not a string; safetensors metadata is text`. The tensor name `__metadata__` is reserved.

Copy one tensor out of another container:

```
tenspile -cf copied.safetensors w=out.safetensors:embed
```

## Extract

With no `--format`, each member becomes an `.npy` when it has a NumPy descr.

```
tenspile -xf out.safetensors -C out embed
```

That writes `out/embed.npy`. Raw bytes are the payload alone, with no header to read back later:

```
tenspile -x -f out.safetensors --format raw -o embed.raw embed
```

`embed.raw` is 8 bytes: `00 00 80 3f 00 00 00 40`.

Several members can be written as one new safetensors file. That file has the tensors and no metadata:

```
tenspile -x -f out.safetensors --format safetensors -o layer.safetensors embed norm
tenspile -i -f layer.safetensors
```

```
members: 2
data_bytes: 16
metadata: none
```

Delete rewrites the file without the named members:

```
tenspile --delete -f out.safetensors norm
```

## What the file looks like

This is the one-tensor file from the create command above. A real checkpoint repeats the tensor object once per key; the header can be megabytes, which is what `-t` and `-i` are for.

```
60 00 00 00 00 00 00 00     little-endian header length, 96
{"__metadata__":{"format":"pt"},"embed":{"data_offsets":[0,8],"dtype":"F32","shape":[2]}}
… spaces out to 96 bytes …
00 00 80 3f 00 00 00 40     two f32 values, 1.0 and 2.0
```

`data_offsets` are relative to the first payload byte. Here that byte is at file offset 104, which is what `-v` prints. The ranges have to cover the blob with no gap and no overlap. A repeated JSON key is refused.

The header cap is 100 MiB. Raise it with `--max-header-size 200M` when a file is over the cap. The message names the size and the flag.
