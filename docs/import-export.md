# Import and export

tenspile moves tensors between containers. Create, append, and update read source files into an archive. Extract writes members back out. List and info read headers and leave the weight bytes on disk.

Byte layouts, dtype codes, and validation rules are the files under [tensdataspecs_readonly/specs](../tensdataspecs_readonly/specs/README.md). This page says which operations exist.

- [safetensors](safetensors.md) follows [safetensors.yaml](../tensdataspecs_readonly/specs/formats/safetensors.yaml)
- [NumPy `.npy` and `.npz`](numpy.md) follow [npy.yaml](../tensdataspecs_readonly/specs/formats/npy.yaml) and [npz.yaml](../tensdataspecs_readonly/specs/formats/npz.yaml)
- [rawtensor/1](raw.md) follows [raw-sidecar.yaml](../tensdataspecs_readonly/specs/formats/raw-sidecar.yaml)
- [GGUF](gguf.md) follows [gguf.yaml](../tensdataspecs_readonly/specs/formats/gguf.yaml)
- [shard index](shards.md) is the sharding section of the safetensors spec. A directory with those weights is the same archive. A `.safetensors` file next to `config.json` is refused unless `--force-single-file` is written immediately before that path.
- PyTorch zip checkpoints follow [pytorch.yaml](../tensdataspecs_readonly/specs/formats/pytorch.yaml). `--format pytorch` reads and writes a state_dict.
- ONNX initializers follow [onnx.yaml](../tensdataspecs_readonly/specs/formats/onnx.yaml). `--format onnx` reads and writes those tensors, not the graph.

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

`bytes` is the stored length. For a GGUF tensor, that includes alignment padding up to the next tensor. `offset` is where that payload starts in the file. Inside `.npz` it is the offset inside that array's `.npy` entry. `params` is the element count. `shard` is filled for a shard index. `detail` is a padding note, `fortran_order`, `zip deflate`, a quant-pack name, or the shard file name. GPTQ, AWQ, bitsandbytes, compressed-tensors, and MLX stay packed; the note names the pack and the bytes are not dequantized.

```
tenspile -t --columns -f model.safetensors
tenspile -t --columns --fold -f model.safetensors
```

`--columns` prints the same fields aligned for a terminal, with a heading row. A shared parent is a heading, and the leaf name is indented under it. `bytes` is a power-of-two size (`64 MiB`) and `params` is a decimal count (`16.8M`). `offset` stays an integer. `shard` and `detail` are omitted when every value is empty. `--fold` implies `--columns` and merges one numeric name segment across a consecutive run when dtype, shape, stored size, shard, and detail all match. `model.layers.[3-5].self_attn.` is one copy of each leaf, not the sum, and the offset column is `-`. A tensor that differs, or a gap in the numbers, stays on its own line. A second number in the name, such as an expert id, is not folded in the same pass. `--json` stays the tab-free machine listing; `--columns` and `--fold` are refused with it.

`tenspile -i` adds format, version, member count, data bytes, metadata, a dtype tally, and any spec notes. On a directory, a sibling `config.json` with `quantization_config.quant_method` or an MLX `quantization` object adds one note with the method, bit width, and group size. The weights stay packed. `-i --json` and `-t --json` print one document.

## Comparing two files

`tensdiff LEFT RIGHT` compares two containers. `tenspile` does not. The report lists a format difference, member names only on one side, and one line for each tensor that differs or was paired under another name. Metadata keys stay on their own lines. Comparison continues after the first mismatch.

A tensor line is `name : dtype : shape`. `=>` appears only on a field that differs, so a shared dtype or shape is written once:

```
blk.18.ssm_conv1d.weight => model.language_model.layers.17.linear_attn.conv1d.weight : F32 : [6144, 4] => [6144, 1, 4]
blk.19.ssm_conv1d.weight : F32 : [6144, 4] => [6144, 1, 4]
```

When the dtype and shape match and the bytes do not, the line ends with `values differ at byte N`. A renamed pair whose dtype, shape, and bytes match is printed and does not by itself make the exit status 1. Tensor lines and names only on one side are sorted by name, and a run of digits compares as a number, so `.10.` follows `.9.`.

`--fold` merges tensor lines that are the same except for a consecutive number. A number that advances on both sides is written as a range on both sides. A number that stays the same is written once. A gap is not bridged.

```
blk.[20-21].attn_norm.weight => model.language_model.layers.[20-21].input_layernorm.weight : F32 => BF16 : [1024]
```

A dtype match uses the canonical id, so `F32` and `float32` are the same type. A GGUF shape is reversed from `ne[]` before it is compared. The bytes compared are the logical tensor: GGUF alignment padding is dropped, and a quantized tensor is the block bytes, not a dequantized float. Exit 0 means the names, types, shapes, values, and metadata match. Exit 1 means a difference was printed. `tensdiff` exits 2 when a file cannot be read. `--json` prints the same findings as a document.

`--match-names` pairs names that differ through `tables/gguf-tensor-roles.json`. A GGUF `general.architecture` value selects that architecture. Otherwise every architecture is tried, and the smallest role set that explains every remaining name wins. When several architectures still fit, the report says `names: may be refact, qwen2vl, or granite`. Each pair lists both names. `language_model.` is removed before the lookup. Vision and audio tensors are left unpaired in a language file. Qwen3.5 `dt_bias` and `mtp.*` names use the rewrites in `qwen35_tensor_names_to_gguf`. If nothing covers the names, the report says `names: no table matched` and falls back to exact names. Paired Q and K weights stay in the order stored in the file, so a permuted or quantized tensor still shows a value or dtype difference.

```bash
tensdiff --match-names model.safetensors model.gguf
```

`--fold` on a listing stays on `tenspile`. `--fold` on a diff stays on `tensdiff`.

## Converting a container

`tensconv` reads one container and writes one container. A directory is the safetensors weight set, the same rule `tenspile` uses. Shapes are outermost-first. A GGUF `ne[]` axis is reversed on the way in and on the way out. A tensor whose bytes change, by a widened dtype or a value change, is held in memory until the output is written. A GGUF name of 64 bytes or more, or a rank outside 1..=4, stops the write. A rank-0 tensor is stored as `ne = [1]`, and `<name>.orig_shape` records its shape `[]` (GGUF-023). Reading that key back restores the shape.

Metadata is carried over. GGUF keys keep their types when the target is GGUF or rawtensor, and become `gguf.<key>` JSON strings tagged with the type in safetensors and ONNX. Tokenizer arrays are left out of those string maps with a warning. String metadata from other formats is written to GGUF as `<format>.metadata.<key>`. npy, npz, and PyTorch output carry no metadata, and a warning names how many keys were dropped.

A dtype is copied when the target has it. BF16 widens to F32, and the FP8 and FP4 types that fit in F16 widen to F16. E8M0 widens to F32. Anything else the target cannot store stops the command. Narrowing is not implemented. Quantized blocks are not written here; `tensquant` does that.

```bash
tensconv -o model.gguf model.safetensors
tensconv --rename -o model.gguf model/
```

`--rename` takes the architecture from `general.architecture`, or from `config.json` `architectures[0]` (`text_config.architectures[0]` when it is present). The converter class is the one registered for that HF class. Without `config.json`, an architecture with more than one converter class stops: there is no preference order. Names come from `tables/gguf-tensor-roles.json`. Hyperparameters written into GGUF follow `hf_config_to_gguf`: `text_config` is merged over the root object first, then the first listed key is used. A sibling `config.json` is read only for those keys.

Value changes run with `--rename` unless `--names-only` is set. A converter record with `complete: true` applies the listed branch (`+ 1`, unsqueeze, squeeze, the clip `view`/`permute`, `-exp`, and the recorded renames). The expression text is not evaluated. Any other expression stops the conversion and prints that expression. LLaMA rows of tensors ending in `q_proj.weight`, `q_proj.bias`, `k_proj.weight`, or `k_proj.bias` are permuted when that converter has `undo_permute`. Qwen3.5 applies the V-head reorder, `A_log` as `-exp`, norms other than `linear_attn.norm.weight` as `weight + 1`, and the `conv1d` squeeze. Full-attention Q and K are not permuted, and the reorder is skipped when the linear head counts match. An architecture whose record is `complete: false`, and that is not the LLaMA or Qwen3.5 case, stops and names `unexpanded`. Warnings go to stderr.

## Dequantizing and the reference quantizers

`tensquant` reads a container and writes float32, or the reverse for the two reference quantizers. Dequant does not recover the original pre-quant floats.

```bash
tensquant --dequant -o model.safetensors model.gguf
tensquant --quantize q8_0 -o model.gguf weights.safetensors
```

GGML blocks use the `dequantize` snippets. Q8_1 is refused. A pack is read only with `config.json` `quantization_config`, because bits, group size, and the zero convention are not in the tensors; tensor names that only look like a pack stop with a message that names the guess. bitsandbytes is the exception when each weight has its own `quant_state`.

- GPTQ uses the 2, 4, and 8 bit cuda pack, which is also the stored exllama layout. `bits` or `w_bit` gives the width. Zeros are incremented before the scale for `checkpoint_format` `gptq`, and used as stored for `gptq_v2`. `g_idx` selects the row.
- AWQ uses the gemm nibble order. `version` is read without case and defaults to gemm.
- bitsandbytes reads `<m>.weight`, `<m>.weight.absmax`, and `<m>.weight.quant_state.bitsandbytes__nf4` (or `__fp4`), and takes the shape from that quant_state. A nested absmax is decoded with `nested_quant_map`, `nested_absmax`, and `nested_offset`. A file without quant_state needs the config's quant type and blocksize.
- compressed-tensors picks each module's scheme from `config_groups` (exact names, then `re:` patterns, then class names), unpacks the codes, and applies `(q - zero) * (scale / global_scale)` for the tensor, channel, group, and block strategies, rounding each step to the scale's dtype. The checkpoint does not record module classes, so a class target such as `Linear` is assumed to match and a warning says so.
- MLX uses `quantization` in `config.json`, with per-module entries replacing the defaults, and a per-expert nvfp4 `global_scale`.

Pack outputs are named `<module>.weight` and shaped `[out_features, in_features]`.

GPTQ 3-bit, the marlin tile inverse, EXL2 (`EXL2-Q1`), and ModelOpt NVFP4 (`CT-Q1`) stop, because those formulas are not written out. `--quantize` accepts `q4_0` and `q8_0` only.

## Files that are refused

A PyTorch zip checkpoint is read with a restricted unpickler that handles the opcode set of PT-007 and no other. Detection follows the order in conversion-considerations section 14: a safetensors header is tested before any pickle test, a legacy `torch.save` pickle is recognized by its 14-byte prefix and a tar checkpoint by `ustar`, and both stop with a message that names pickle. A file name alone never makes a file a pickle. A ZIP whose `data.pkl` is not inside a folder is refused (PT-001). No code from the file runs. A global outside the tensor allowlist is rejected. A `.bin` whose bytes are ordinary tensor data can be imported when `--dtype` and `--shape` are set. Legacy `ggml`, `ggmf`, and `ggjt` stop with `is legacy GGML/GGMF/GGJT; tenspile reads GGUF`. Removed ggml type codes are rejected. GGUF version 1 and big-endian GGUF are refused. A strided PyTorch tensor is refused rather than gathered.
