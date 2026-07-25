# Shard index

`model.safetensors.index.json` is one logical safetensors archive. `weight_map` names the shard file for each tensor. The shard files sit next to the index and hold the bytes. tenspile lists, extracts, appends, replaces, and deletes through the index. `tenspile -c -f model.safetensors.index.json` stops with `creating a shard index is not supported; edit an existing index instead`.

This is the Hugging Face index described in the [safetensors metadata parsing doc](https://github.com/huggingface/safetensors/blob/main/docs/source/metadata_parsing.mdx): a `weight_map` from tensor name to shard file, plus `metadata.total_size`. [`SafetensorsIndexJson`](https://github.com/huggingface/huggingface.js/blob/main/packages/hub/src/lib/parse-safetensors-metadata.ts) in huggingface.js is the same object. Shard names are relative. The index is not a concatenation of the shard files. `version: index` is tenspile's label. Each shard is a [safetensors](safetensors.md) file. Extract formats are in [Import and export](import-export.md).

## What you start with

Two shards and an index that points at them:

```json
{
  "metadata": { "total_size": 8 },
  "weight_map": {
    "a": "shard1.safetensors",
    "b": "shard2.safetensors"
  }
}
```

A Hub index has the same two fields. `weight_map` continues for every tensor, and `total_size` is the sum of the tensor bytes. Shard names are relative to the index directory. An absolute path or a `..` component is refused. Every tensor in a shard has to appear in the map, and every map entry has to exist in that shard.

```
tenspile -i -f model.safetensors.index.json
```

```
format: safetensors
version: index
path: model.safetensors.index.json
members: 2
data_bytes: 8
metadata:
  total_size=8
dtypes:
  U8      2       8
note: 2 shards
```

```
tenspile -tv -f model.safetensors.index.json
```

```
name  dtype  shape  bytes  offset  params  shard                 detail
a     U8     [4]    4      64      4       shard1.safetensors    shard1.safetensors
b     U8     [4]    4      64      4       shard2.safetensors    shard2.safetensors
```

`offset` is the payload offset inside that shard file. Extract uses the same rules as one safetensors file:

```
tenspile -x -f model.safetensors.index.json --format raw -o a.raw a
tenspile -xf model.safetensors.index.json -C out
```

`--json` stringifies metadata, so `total_size` appears as `"8"` in that listing. The index file keeps the number `8`.

## Append, replace, and delete

New tensors are written into the shard whose file name sorts last. With zero-padded Hub names, that is the highest shard number (`model-00002-of-00002.safetensors` sorts after `model-00001-of-00002.safetensors`). Here `shard2.safetensors` sorts after `shard1.safetensors`, so `c` lands in shard 2. An index with no shards has nowhere to put a new tensor.

```
tenspile -r -f model.safetensors.index.json --dtype U8 --shape 4 c=c.bin
```

```json
{
  "metadata": { "total_size": 12 },
  "weight_map": {
    "a": "shard1.safetensors",
    "b": "shard2.safetensors",
    "c": "shard2.safetensors"
  }
}
```

`shard1.safetensors` is left as it was. `shard2.safetensors` is rewritten with `b` and `c`. A shard's own `__metadata__` is kept. The index metadata is separate.

`-u` replaces a tensor in the shard that already holds it, or appends a new name onto the last shard. A duplicate append stops with `tensor a already exists; use -u to replace it`.

```
tenspile --delete -f model.safetensors.index.json b
tenspile --delete -f model.safetensors.index.json c
```

After `b` is removed, `shard2.safetensors` still exists because `c` points at it, and `total_size` is 8. After `c` is removed, nothing points at shard 2, so that file is deleted and `total_size` is 4. `total_size` is recomputed from the tensors. A value passed with `--metadata total_size=…` is replaced by that sum.

```json
{
  "metadata": { "total_size": 4 },
  "weight_map": { "a": "shard1.safetensors" }
}
```

Other index metadata can be set as strings:

```
tenspile -u -f model.safetensors.index.json --metadata note=hi
```

`-o` has to stay in the index directory. A different directory stops with `-o for a shard index must stay in the same directory as the index`. Shard files are removed only when the index is rewritten onto itself. An `-o` of another name in that directory writes the new index and leaves every shard file on disk.

`-n` prints `rewrite shard2.safetensors` and `drop shard2.safetensors` without changing either file.
