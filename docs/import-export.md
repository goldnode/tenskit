# Import and export

tenspile moves tensors between containers. Create, append, and update read source files into an archive. Extract writes members back out. List and info read headers and leave the weight bytes on disk.

Byte layouts, dtype codes, and validation rules are the files under [tensdataspecs_readonly/specs](../tensdataspecs_readonly/specs/README.md). This page says which operations exist.

- [safetensors](safetensors.md) follows [safetensors.yaml](../tensdataspecs_readonly/specs/formats/safetensors.yaml)
- [NumPy `.npy` and `.npz`](numpy.md) follow [npy.yaml](../tensdataspecs_readonly/specs/formats/npy.yaml) and [npz.yaml](../tensdataspecs_readonly/specs/formats/npz.yaml)
- [rawtensor/1](raw.md) follows [raw-sidecar.yaml](../tensdataspecs_readonly/specs/formats/raw-sidecar.yaml)
- [GGUF](gguf.md) follows [gguf.yaml](../tensdataspecs_readonly/specs/formats/gguf.yaml)
- [shard index](shards.md) is the sharding section of the safetensors spec. A directory with those weights is the same archive. A `.safetensors` file next to `config.json` is refused unless `--force-single-file` is written immediately before that path.
- [PyTorch zip checkpoints](pytorch.md) follow [pytorch.yaml](../tensdataspecs_readonly/specs/formats/pytorch.yaml). `--format pytorch` reads and writes a state_dict.
- [ONNX initializers](onnx.md) follow [onnx.yaml](../tensdataspecs_readonly/specs/formats/onnx.yaml). `--format onnx` reads and writes those tensors, and an edit keeps the graph.

`version` on `-i` is the file's own format version when it has one: `2` or `3` for little-endian GGUF, `1.0`, `2.0`, or `3.0` for `.npy`, `1` for rawtensor, the text of the pytorch `version` record, and `ir N` for ONNX. The other labels are tenspile's: `header+blob`, `index`, and `zip`.

Which formats exist in a build is [local/supported-formats.txt](../local/supported-formats.txt). A format left out of that file is not compiled in.

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
| `KEY=ARCHIVE:MEMBER` | Copy one member from safetensors, GGUF, `.npz`, rawtensor, pytorch, ONNX, or a shard index. |
| a bare name | On list, extract, and delete, select that member. |

A path that exists is a file, including a name that contains a colon. `ARCHIVE:MEMBER` applies when the whole string is not a file and the text before the last colon is a file.

`--dtype` and `--shape` describe a headerless source. They are not a stored format. On a file that already has a header they have to match the header. A safetensors name (`F32`) and the canonical id (`float32`) name the same type. The ids and the per-format codes are [dtypes.yaml](../tensdataspecs_readonly/specs/mappings/dtypes.yaml) and [dtype-mapping.json](../tensdataspecs_readonly/specs/mappings/dtype-mapping.json).

`--shape` accepts `2`, `2,2`, `2x2`, and `[2, 2]`. `[]` is a scalar: one element.

## Where a tensor can go

| Source | safetensors | `.npy` | `.npz` | rawtensor | GGUF | shard index |
| --- | --- | --- | --- | --- | --- | --- |
| safetensors member | yes | when the mapping has a NumPy descr | when the mapping has a NumPy descr | yes | — | edit an existing index |
| `.npy` | yes | one array | yes | yes | — | edit an existing index |
| `.npz` member | yes | when the mapping has a NumPy descr | when the mapping has a NumPy descr | yes | — | edit an existing index |
| headerless bytes | yes, with `--dtype` and `--shape` | with a NumPy descr | with a NumPy descr | yes | — | edit an existing index |
| GGUF element type with a NumPy descr | yes, padding dropped, shape reversed to canonical | yes | yes | yes | metadata rewrite | — |
| GGUF `BF16` | yes | — | — | yes | metadata rewrite | — |
| GGUF quantized type | — | — | — | yes, block bytes, not the alignment pad | metadata rewrite | — |

A cell marked — is refused before a write. GGUF list and info show `ne[]`, innermost first. An export reverses that to the canonical shape. Bytes are copied as stored. F4 and F6 tensors are copied without interpreting the open packing questions in the safetensors spec; `-i` names ST-Q1 and ST-Q2 when those dtypes are present.

Extract writes one `.npy` per member when the dtype has a NumPy descr. `BF16`, `F4`, `F6_*`, and the `F8_*` names are written as rawtensor. `--format npy` or `--format raw` forces that choice, and `--format` wins over the output extension.

`--format npz`, `--format safetensors`, `--format raw`, `--format pytorch`, `--format onnx`, or an `-o` name ending in `.npz`, `.safetensors`, `.rawtensor.json`, `.pt`, or `.onnx`, writes one archive. A tensor named `model.layers.0.weight` becomes `model.layers.0.weight.npy` or `model.layers.0.weight.rawtensor.json`.

`-C` is the directory for per-member files. The default is the current directory. `--force` overwrites an existing output file. `-n` prints the plan and writes nothing. A rewrite uses a temporary file in the destination directory, then renames it into place.

## What list and info print

`tenspile -t -f out.safetensors` prints one name per line. `-v` adds columns, separated by tabs:

```
name    dtype   shape   bytes   offset  params  shard   detail
embed   F32     [2]     8       104     2       -       -
```

| Column | Meaning |
| --- | --- |
| `name` | The member name. |
| `dtype` | The container's own type name where it has one (`F32` in safetensors, npy, npz, and GGUF; a raw descr such as `<V2` when there is no mapping), otherwise the canonical id (`float32` in rawtensor, PyTorch, and ONNX). |
| `shape` | The stored shape. GGUF shows `ne[]`, innermost first. |
| `bytes` | The tensor's own length. Padding after a GGUF tensor is not counted; `detail` says how much follows. |
| `offset` | Where the bytes start in the file: for a shard index, in that shard; for `.npz`, in the zip file when the entry is stored. It is 0 when there is no single position, as for a deflated `.npz` entry or an ONNX tensor in typed fields. |
| `params` | The element count. |
| `shard` | The shard file, for a shard index. |
| `detail` | A note: GGUF padding, `fortran_order`, `zip deflate`, or a quantization pack. |

GPTQ, AWQ, bitsandbytes, compressed-tensors, and MLX tensors stay packed. `detail` names the pack, and [tensquant](tensquant.md) dequantizes it.

```
tenspile -t --columns -f model.safetensors
tenspile -t --columns --fold -f model.safetensors
```

`--columns` prints the same fields aligned for a terminal, with a heading row. A shared parent is a heading, and the leaf name is indented under it. `bytes` is a power-of-two size (`64 MiB`) and `params` is a decimal count (`16.8M`). `offset` stays an integer. `shard` and `detail` are omitted when every value is empty. `--fold` implies `--columns` and merges one numeric name segment across a consecutive run when dtype, shape, stored size, shard, and detail all match. `model.layers.[3-5].self_attn.` is one copy of each leaf, not the sum, and the offset column is `-`. A tensor that differs, or a gap in the numbers, stays on its own line. A second number in the name, such as an expert id, is not folded in the same pass. `--json` stays the tab-free machine listing; `--columns` and `--fold` are refused with it.

`tenspile -i` adds format, version, member count, data bytes, metadata, a dtype tally, and any spec notes. Each tally row is a dtype, the number of tensors of that type, and their total element count. A note often names a spec rule id, such as `NPY-005`, that can be looked up in that format's spec. On a directory, a sibling `config.json` with `quantization_config.quant_method` or an MLX `quantization` object adds one note with the method, bit width, and group size. The weights stay packed. `-i --json` and `-t --json` print one document.

## The other commands

- [tensdiff](tensdiff.md) compares two containers, including a file and its conversion.
- [tensconv](tensconv.md) converts a whole container, and `--rename` maps Hugging Face names to GGUF.
- [tensquant](tensquant.md) dequantizes GGUF blocks and GPTQ, AWQ, bitsandbytes, compressed-tensors, and MLX packs, and writes Q8_0 or Q4_0 GGUF.

## Files that are refused

A PyTorch zip checkpoint is read with a restricted unpickler that handles the opcode set of PT-007 and no other. Detection follows the order in conversion-considerations section 14: a safetensors header is tested before any pickle test, a legacy `torch.save` pickle is recognized by its 14-byte prefix and a tar checkpoint by `ustar`, and both stop with a message that names pickle. A file name alone never makes a file a pickle. A ZIP whose `data.pkl` is not inside a folder is refused (PT-001). No code from the file runs. A global outside the tensor allowlist is rejected. A `.bin` whose bytes are ordinary tensor data can be imported when `--dtype` and `--shape` are set. Legacy `ggml`, `ggmf`, and `ggjt` stop with `is legacy GGML/GGMF/GGJT; tenspile reads GGUF`. Removed ggml type codes are rejected. GGUF version 1 and big-endian GGUF are refused. A strided PyTorch tensor is refused rather than gathered.
