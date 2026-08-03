# Shard index

`model.safetensors.index.json` is one logical safetensors archive. `weight_map` names the shard file for each tensor. The shard files sit next to the index and hold the bytes. tenspile lists, extracts, appends, replaces, and deletes through the index. Creating an index stops with `creating a shard index is not supported; edit an existing index instead`.

A directory is that weight set. One `*.safetensors.index.json` is opened. With no index, one `*.safetensors` that is not a shard name (`model-00001-of-00005.safetensors`) is opened. Several of either, or none, is an error. `-i` notes the other files in the directory, such as `config.json` and `tokenizer.json`. Those files are not parsed.

A `.safetensors` path whose directory contains `config.json` stops, because the file is likely one piece of that set. Pass the directory, or put `--force-single-file` immediately before that path to read the one file. The flag does not apply to the next path. An index path is already the whole weight set, so `config.json` beside it does not stop the command.

The index is the sharding section of [safetensors.yaml](../tensdataspecs_readonly/specs/formats/safetensors.yaml). `version: index` is tenspile's label. Each shard is a [safetensors](safetensors.md) file. Extract formats are in [Import and export](import-export.md).

## What you start with

```json
{
  "metadata": { "total_size": 8 },
  "weight_map": {
    "a": "shard1.safetensors",
    "b": "shard2.safetensors"
  }
}
```

Shard names are relative to the index directory. An absolute path or a `..` component is refused. Every tensor in a shard has to appear in the map, and every map entry has to exist in that shard.

```
tenspile -i -f model.safetensors.index.json
tenspile -tv -f model.safetensors.index.json
tenspile -x -f model.safetensors.index.json --format raw -o a.rawtensor.json a
tenspile -xf model.safetensors.index.json -C out
```

`offset` is the payload offset inside that shard file. `--json` stringifies metadata, so `total_size` appears as a string in that listing. The index file keeps the number.

## Append, replace, and delete

New tensors are written into the shard whose file name sorts last. With zero-padded Hub names, that is the highest shard number. An index with no shards has nowhere to put a new tensor.

```
tenspile -r -f model.safetensors.index.json --dtype U8 --shape 4 c=c.bin
tenspile -u -f model.safetensors.index.json --metadata note=hi
tenspile --delete -f model.safetensors.index.json b
```

A shard that no longer has tensors is deleted when the index is rewritten onto itself. `total_size` is recomputed from the tensors. A shard's own `__metadata__` is kept. `-o` has to stay in the index directory. An `-o` of another name in that directory writes the new index and leaves every shard file on disk.

`-n` prints the rewrite plan and does not change the files.
