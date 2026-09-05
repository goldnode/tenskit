# tensconv

`tensconv` reads one container and writes the whole of it into another. The output name picks the format, or `--format` names it.

```bash
tensconv -o model.gguf model.safetensors        # safetensors to GGUF, names kept
tensconv --rename -o model.gguf model/          # Hugging Face directory to GGUF names
tensconv -o weights.npz model.gguf              # GGUF to NumPy, shapes reversed
tensconv -o state.pt model.safetensors          # to a PyTorch state_dict
tensconv --format safetensors -o weights.bin model.pt   # the name does not say the format
```

The source can be any format tenspile reads. A directory is the safetensors weight set inside it: its shard index, or its one loose `.safetensors` file. The outputs are safetensors, GGUF, `.npy` (one tensor), `.npz`, rawtensor, PyTorch, and ONNX. An existing output is left in place unless `--force` is set. Warnings go to stderr, and the exit status is 0 when the file was written.

## What is copied

Shapes are stored outermost-first. A GGUF `ne[]` is reversed on the way in and on the way out, so a `[4, 64]` safetensors tensor is `ne = [64, 4]` in GGUF. A rank-0 tensor in GGUF is stored as `ne = [1]`, and the key `<name>.orig_shape` records `[]` so that reading it back restores the scalar (GGUF-023). A GGUF name of 64 bytes or more, or a rank outside 1 to 4, stops the write.

A dtype is copied when the target has it. When it does not, the value is widened only where that is exact:

| Source dtype | Becomes |
| --- | --- |
| BF16, in a target without BF16 (npy, npz) | F32 |
| FP8 and FP4 types that F16 holds exactly | F16 |
| E8M0 | F32 |
| anything else the target lacks | the command stops |

Narrowing is never done. Quantized GGUF blocks are not converted. The command stops and names `tensquant`:

```
$ tensconv -o x.npz small.gguf
tensconv: tensor w_q8_0 is quantized; dequantize it with tensquant
```

Unchanged tensors are streamed from the source. A tensor whose bytes change (widened, or changed by `--rename`) is held in memory until the output is written.

## Metadata

No metadata is dropped silently. The rules are conversion-considerations section 16.

- **GGUF to GGUF or rawtensor:** keys keep their types.
- **GGUF to safetensors or ONNX:** these have string-only metadata. Each key becomes `gguf.<key>` holding a JSON envelope with the GGUF type. A 64-bit integer above 2^53 is a decimal string. An F32 is written as the f64 that round-trips, so `0.1` becomes `0.10000000149011612`. The `tokenizer.ggml.*` arrays are left out with a warning; they belong in `tokenizer.json`.
- **Back to GGUF:** each `gguf.<key>` envelope is restored as the typed `<key>`. Keys that the writer sets itself win, and a warning shows a different source value.
- **Other string metadata into GGUF:** stored as `<format>.metadata.<key>`, where `<format>` is `safetensors`, `onnx`, or `raw-sidecar`. A rawtensor value keeps its type when GGUF has it. A `bytes` or `json` value becomes a string envelope such as `{"type":"BYTES","value":"AAE="}`, and converting that GGUF file back to rawtensor restores the typed `<key>`.
- **npy, npz, and PyTorch output:** these hold no metadata, so a warning says how many keys were left behind.

A GGUF file survives a trip through safetensors:

```
$ tensconv -o back.safetensors llama.gguf
$ tenspile -i back.safetensors
format: safetensors
version: header+blob
path: back.safetensors
members: 30
data_bytes: 9952
metadata:
  gguf.general.architecture={"type":"STRING","value":"llama"}
  gguf.llama.block_count={"type":"UINT32","value":3}
  gguf.llama.context_length={"type":"UINT32","value":64}
  …
  gguf.llama.rope.freq_base={"type":"FLOAT32","value":10000.0}
  gguf.llama.attention.layer_norm_rms_epsilon={"type":"FLOAT32","value":9.999999747378752e-6}
  gguf.general.file_type={"type":"UINT32","value":0}
  gguf.safetensors.metadata.format={"type":"STRING","value":"pt"}
dtypes:
  F32	30	2488
$ tensconv -o back.gguf back.safetensors
$ tensdiff llama.gguf back.gguf
$ echo $?
0
```

## Renaming for llama.cpp

`--rename` maps Hugging Face names to GGUF names through `tables/gguf-tensor-roles.json`. It also applies the value changes that llama.cpp's converter makes, and writes the hyperparameters from `config.json`.

```
$ ls hf
config.json  model.safetensors
$ tensconv --rename -o llama.gguf hf
tensconv: LlamaModel: LLaMA Q/K rows were permuted; other unexpanded arms were left alone
tensconv: renamed model.embed_tokens.weight -> token_embd.weight
tensconv: renamed model.layers.0.self_attn.q_proj.weight -> blk.0.attn_q.weight
…
tensconv: renamed lm_head.weight -> output.weight
$ tenspile -t --columns --fold -f llama.gguf
name                  dtype  shape    bytes  offset  params
token_embd.
  weight              F32    [8, 32]  1 KiB    2304     256
blk.[0-2].
  attn_q.weight       F32    [8, 8]   256 B       -      64
  attn_k.weight       F32    [8, 8]   256 B       -      64
  attn_v.weight       F32    [8, 8]   256 B       -      64
  attn_output.weight  F32    [8, 8]   256 B       -      64
  ffn_gate.weight     F32    [8, 16]  512 B       -     128
  ffn_up.weight       F32    [8, 16]  512 B       -     128
  ffn_down.weight     F32    [16, 8]  512 B       -     128
  attn_norm.weight    F32    [8]       32 B       -       8
  ffn_norm.weight     F32    [8]       32 B       -       8
output_norm.
  weight              F32    [8]       32 B   11200       8
output.
  weight              F32    [8, 32]  1 KiB   11232     256
```

How the architecture and its changes are chosen:

- The architecture is `general.architecture`, or `config.json` `architectures[0]`. `text_config.architectures[0]` is used when it is present, except for the few classes llama.cpp keeps at the top level. The converter class is the one registered for that Hugging Face class. When there is no `config.json` and an architecture has more than one converter class, the command stops, because the spec gives no preference order.
- Hyperparameters follow `hf_config_to_gguf`. `text_config` is merged over the root object, and the first listed key wins. A sibling `config.json` is read only for those keys.
- Value changes run unless `--names-only` is set. A converter record with `complete: true` applies its listed branch: `+ 1`, unsqueeze, squeeze, the clip `view`/`permute`, `-exp`, and the recorded renames. The expression text is never evaluated. Any other expression stops the conversion and prints it.
- LLaMA permutes the rows of `q_proj` and `k_proj` weights and biases when the converter has `undo_permute`. That is why `tensdiff --match-names` reports their values as different.
- Qwen3.5 applies the V-head reorder, `A_log` as `-exp`, norms other than `linear_attn.norm.weight` as `weight + 1`, and the `conv1d` squeeze. Full-attention Q and K are not permuted, and the reorder is skipped when the linear head counts match.
- Any other architecture whose record is `complete: false` stops. The message explains why, suggests `--names-only`, and lists the `unexpanded` steps one per line, as the spec quotes them from llama.cpp's Python source.
- Into GGUF, the tensors llama.cpp's converter writes as F32 are widened to F32 when they are F16 or BF16 (GGUF-024 steps 2 and 3): rank 1 and below, `*_norm.weight`, names that do not end in `.weight`, such as biases, and a few roles such as `ffn_gate_inp`. The widen is exact. llama.cpp's CPU backend cannot multiply by a BF16 norm, so a file without it stops with `binary_op: unsupported types`. One warning gives the count. `--names-only` skips this too.

`--names-only` renames and skips every value change, which is useful for inspecting the mapping.

## See also

[tensdiff](tensdiff.md) for checking a conversion, [tensquant](tensquant.md) for quantized tensors, and `man tensconv`.
