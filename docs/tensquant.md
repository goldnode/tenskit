# tensquant

`tensquant --dequant` turns quantized tensors back into float32. `tensquant --quantize` writes GGUF with one of two reference quantizers, Q8_0 or Q4_0. Dequant does not recover the original floats; it reproduces the values the quantized file stands for.

```bash
tensquant --dequant -o model-f32.safetensors model.gguf
tensquant --dequant -o model-f32.safetensors gptq-model/
tensquant --dequant -o weights.npz bnb-model/
tensquant --quantize q8_0 -o model-q8.gguf model/
```

An existing output is left in place unless `--force` is set. Warnings go to stderr.

## Dequantizing

The output is safetensors, `.npy` (one tensor), or `.npz`. What happens to each tensor:

- **Quantized tensors and packs:** each becomes one float32 `<module>.weight`, shaped `[out_features, in_features]` (CONV-003). Stacked experts are `[experts, out, in]`.
- **Plain float tensors (F16, BF16, F32):** widened to float32.
- **Integer and bool tensors:** copied unchanged, because float32 cannot hold every int32 or int64 value.

```
$ tensquant --dequant -o dequant.safetensors small.gguf
$ tenspile -t --columns -f dequant.safetensors
name    dtype  shape    bytes  offset  params
w_f32   F32    [4, 64]  1 KiB     336     256
b_f16   F32    [8]       32 B    1360       8
i_i32   I32    [3, 2]    24 B    1392       6
e_bf16  F32    [2, 3]    24 B    1416       6
w_q8_0  F32    [4, 64]  1 KiB    1440     256
```

### GGUF

Each GGML block type is dequantized with the `dequantize` snippet in gguf.yaml. That covers Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K through Q8_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, and Q2_0. Q8_1 is refused, because the spec infers its dequant rather than taking it from llama.cpp. The canonical shape is the reversed `ne[]`.

### Quantization packs in safetensors

GPTQ, AWQ, compressed-tensors, and MLX store their bit width, group size, and zero convention in `config.json`, not in the tensors. Pass the model directory, or a file beside its `config.json`, so the config is read. A file whose tensor names only look like a pack stops and says so:

```
$ tensquant --dequant -o x.safetensors lone.safetensors
tensquant: the tensor names look like a GPTQ pack, but there is no config.json quantization_config; dequant needs it
```

`tenspile -i` on the directory shows what was found, and `tenspile -t` marks the pack tensors:

```
$ cat gptq/config.json
{"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 32, "desc_act": false}}
$ tenspile -i gptq | tail -2
note: directory gptq also contains config.json
note: quantization gptq bits 4 group 32
$ tenspile -t --columns -f gptq
name       dtype  shape     bytes  offset  params  detail
gptq.
  scales   F32    [8, 32]   1 KiB     296     256  GPTQ pack; bytes stay packed
  g_idx    I32    [256]     1 KiB    1320     256  GPTQ pack; bytes stay packed
  qweight  I32    [32, 32]  4 KiB    2344      1K  GPTQ pack; bytes stay packed
  qzeros   I32    [8, 4]    128 B    6440      32  GPTQ pack; bytes stay packed
$ tensquant --dequant -o gptq.safetensors gptq
$ tenspile -t --columns -f gptq.safetensors
name      dtype  shape       bytes  offset  params
gptq.
  weight  F32    [32, 256]  32 KiB      88    8.2K
```

The `quantization_config` that each pack reads:

| Pack | Keys read | Notes |
| --- | --- | --- |
| GPTQ | `bits` (or `w_bit`), `group_size`, `checkpoint_format` | 2, 4, and 8 bits in the cuda pack, which is also the stored exllama layout. `checkpoint_format: gptq` (the default) adds 1 to the zeros; `gptq_v2` uses them as stored. `g_idx` selects each row's group. |
| AWQ | `bits` (or `w_bit`), `group_size`, `version` | The gemm nibble order. `version` is read without regard to case and defaults to gemm. |
| bitsandbytes | `bnb_4bit_quant_type`, `bnb_4bit_quant_blocksize` | Needed only when a weight has no `quant_state` tensor. The Hugging Face layout (`<W>`, `<W>.absmax`, `<W>.quant_state.bitsandbytes__nf4`) carries its own type, blocksize, and shape. A nested absmax is decoded with `nested_absmax`, `nested_quant_map`, and `nested_offset`. |
| compressed-tensors | `config_groups`, `ignore`, `format` | See below. |
| MLX | `quantization` (`bits`, `group_size`, `mode`, and per-module entries) | See below. |

Some packs are refused with the open question named: GPTQ 3-bit, the marlin tile layout, EXL2 (`EXL2-Q1`), and ModelOpt NVFP4 (`CT-Q1`). Their formulas are not written out in the specs.

### compressed-tensors

A module counts as quantized when its tensors say so (CT-009), whatever the targets say:

- **pack-quantized:** `weight_packed` with `weight_shape`, codes bit-packed into int32.
- **naive-, int-, or float-quantized:** `weight` beside `weight_scale`, one code per element. An int scheme stores int8, int16, or int32 by `num_bits`. An 8-bit float scheme stores float8_e4m3fn. A zero point is stored unpacked.

The scheme comes from `config_groups`, read in file order (CT-010). Each group lists `targets`, and the match order is: exact module names, then `re:` patterns, then class names.

- A `re:` pattern is matched from the start of the module name, as Python's `re.match` does (CT-011). Groups and alternation work. Lookahead is refused, with the pattern named.
- A target that is not `re:` and names no module in the checkpoint is a class name such as `Linear`. The checkpoint does not record module classes, so a class target matches any quantized module. When class targets of two different schemes compete, the packed width and the zero point decide, or the module is refused.
- A class name in `ignore` cannot be checked either. It gives a warning, and the stored tensors decide.
- A `config_groups` key whose value is a target list is a preset name, such as `W4A16`, `W8A8`, `FP8_DYNAMIC`, or `FP8_BLOCK`, expanded as CT-012 lists it. An unknown preset is refused. A key whose value is a full scheme object is only a label.

```json
{"quantization_config": {
  "quant_method": "compressed-tensors",
  "format": "int-quantized",
  "config_groups": {"group_0": {
    "targets": ["Linear"],
    "weights": {"num_bits": 8, "type": "int", "symmetric": true, "strategy": "channel"}
  }},
  "ignore": ["lm_head"]
}}
```

Values are `(q - zero_point) * (scale / global_scale)` for the tensor, channel, group, tensor_group, and block strategies. Each step is rounded to the scale's dtype, as the reference does.

### MLX

`quantization` gives the defaults: `bits`, `group_size`, and `mode` (`affine`, `mxfp4`, `nvfp4`, or `mxfp8`). A `quantization.<module>` entry replaces them for that module, and `false` leaves the module dense. A per-expert nvfp4 `global_scale` vector is applied expert by expert (MLX-014).

A module with `.scales` but no entry has its settings inferred, as mlx-lm's `infer_quant_config` does (MLX-016):

- For `model_type: llama`, the input width of each attention, MLP, embedding, and `lm_head` module comes from `config.json` (MLX-017). That gives `bits` and `group_size` directly.
- Otherwise, uint8 scales name nvfp4, mxfp4, or mxfp8 from the stored widths. Other scales use the top-level values only when `bits * group_size` matches the stored widths, and the module is refused when it does not.

## Quantizing

```
$ tensquant --quantize q8_0 -o model-q8.gguf model/
```

The output is always GGUF, and the tensor names are kept as stored. `general.file_type` is the requested type, MOSTLY_Q8_0 (7) or MOSTLY_Q4_0 (2) (GGUF-025). The source's other metadata is carried as `tensconv` carries it, so a GGUF source keeps its architecture and hyperparameters. K-quants and IQ types are not written.

A tensor that is already quantized is not quantized again, because that loses more than quantizing once. As with llama.cpp's `llama-quantize`, the command stops:

```
$ tensquant --quantize q4_0 -o model-q4.gguf model-q8.gguf
tensquant: tensor output.weight is already quantized (Q8_0); dequantize the file with tensquant --dequant first
```

For a file that llama.cpp can load, rename first and then quantize the GGUF. Quantizing a Hugging Face directory directly keeps the Hugging Face names:

```
$ tensconv --rename -o model.gguf model/
$ tensquant --quantize q8_0 -o model-q8.gguf model.gguf
```

Neither command builds a tokenizer (`tokenizer.ggml.*`), which llama.cpp needs before it can load the model (GGUF-033); the spec has not transcribed how llama.cpp's converter builds one (GGUF-Q5). A GGUF source's tokenizer keys are copied. When the output has no `tokenizer.ggml.model`, both commands warn.

Which tensors are quantized follows llama.cpp's `convert_hf_to_gguf.py` (GGUF-024). These stay F32:

- rank 1 and below;
- names ending in `_norm.weight`;
- names that do not end in `.weight`, `.lora_a`, or `.lora_b`;
- a fixed list of roles, such as `ffn_gate_inp`, `position_embd`, `token_types`, `ssm_conv1d`, and the `time_mix_*` tensors.

Hugging Face names are first resolved to their GGUF role, using `config.json` `architectures[0]` (GGUF-026). For example, a Qwen2-MoE `model.layers.0.mlp.gate.weight` is the router, `ffn_gate_inp`, and stays F32. A name with no role, or every name when the architecture is unknown, is decided by rank and the `.weight` suffix only, and one warning lists those names. The 36 per-architecture `tensor_force_quant` overrides in llama.cpp are not applied.

A matrix whose innermost size is not a multiple of 32 cannot be split into blocks. It is written F16 with a warning, as llama.cpp does, and `general.file_type` still names the requested type (GGUF-027). Integer tensors are copied.

### Compared with llama-quantize

tensquant follows `convert_hf_to_gguf.py`, not `llama-quantize`, so it does not claim to match llama-quantize's output (GGUF-032). The blocks of a tensor both tools quantize to the same type are the same reference packing. The differences are in which tensors are quantized, and to what (GGUF-031):

- llama-quantize gives the output tensor a higher type: `output.weight`, or `token_embd.weight` when the model shares its input and output embeddings. With Q4_0 it asks for Q6_K, which becomes Q8_0 when the row width is not a multiple of 256. tensquant writes Q4_0, and warns with the type llama-quantize would use.
- In a model with 8 experts, llama-quantize writes `attn_v`, `attn_qkv`, `attn_kv_b`, and `attn_k` as Q8_0 at Q4_0. tensquant warns with the count.
- llama-quantize leaves every tensor it does not quantize in its stored type, so a BF16 norm stays BF16. tensquant writes the GGUF-024 tensors as F32, which every llama.cpp backend can run (GGUF-028).
- llama-quantize's list of tensors it never quantizes differs from GGUF-024's in other details. Among them are `altup`, `laurel`, and some vision tensors.

At Q8_0, the output-tensor and expert rules change nothing, so only the last two points apply.

## See also

[tensconv](tensconv.md) for conversions without quantization, and `man tensquant`.
