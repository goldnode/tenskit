# tensinfo

`tensinfo` says what a tensor container holds. By default it gives a short overview of the model: what kind it is,
how big, in which dtypes, how it is laid out, and whether it is quantized. Use it before deciding what to do with a
file. To list or edit the members themselves, use [tenspile](import-export.md).

```bash
tensinfo model/                        # overview of a Hugging Face directory
tensinfo model.gguf                    # overview of a GGUF file
tensinfo -t --fold model.gguf          # add the tensor table, one line per layer pattern
tensinfo -m model.gguf                 # add every metadata key
tensinfo --json model.safetensors      # everything, for a script
```

## The overview

```
$ tensinfo smollm2-135m
file:         smollm2-135m/model.safetensors (safetensors), 256.6 MiB of tensor data
model:        llama (LlamaForCausalLM)
parameters:   134.5M in 272 tensors
dtypes:       BF16 (all)
layers:       30, hidden 576, heads 9 (3 kv), context 8192
vocabulary:   49152
quantization: none
tokenizer:    tokenizer.json beside the weights
embeddings:   tied input and output (tie_word_embeddings)
```

The same model as a quantized GGUF file:

```
$ tensinfo smol-gguf/SmolLM2-135M-Instruct-Q4_K_M.gguf
file:         smol-gguf/SmolLM2-135M-Instruct-Q4_K_M.gguf (gguf v3), 98.9 MiB of tensor data
model:        llama (Smollm2 135M 8k Lc100K Mix1 Ep2, 135M)
parameters:   134.5M in 272 tensors
dtypes:       Q5_0 58%, Q8_0 22%, Q4_K 11%, Q6_K 9%, F32 <1%
layers:       30, hidden 576, heads 9 (3 kv), context 8192
vocabulary:   49152
quantization: GGUF blocks Q4_K, Q5_0, Q6_K, Q8_0
tokenizer:    gpt2 (tokenizer.ggml.* keys)
embeddings:   the output layer shares token_embd
```

Each line, and where it comes from. A line whose source is missing is left out.

| Line | From |
| --- | --- |
| `file` | The file read, its container format, and the size of the tensor data. A directory is the safetensors weight set inside it, as for tenspile. |
| `model` | GGUF: `general.architecture`, with `general.name` and `general.size_label`. Hugging Face: `config.json` `model_type` and `architectures[0]`. |
| `parameters` | The total element count and the number of tensors. A packed quantization (GPTQ, AWQ, bitsandbytes) counts its packed values as stored, and says so. |
| `dtypes` | Each dtype's share of the parameters, largest first. |
| `layers` | Layer count, hidden size, attention heads and key-value heads, context length, and experts. GGUF: the `{arch}.*` keys. Hugging Face: the `config.json` names that `hf_config_to_gguf` lists, with `text_config` merged over the top level for a multimodal model. |
| `vocabulary` | GGUF: `{arch}.vocab_size`, the `tokenizer.ggml.tokens` count, or the rows of `token_embd`. Hugging Face: `vocab_size`. |
| `quantization` | GGUF: the block types present. Hugging Face: the method, bits, and group size from `quantization_config` or MLX `quantization`; a pack found without one is named. |
| `tokenizer` | GGUF: `tokenizer.ggml.model`; a GGUF file without it does not load in llama.cpp. Otherwise `tokenizer.json` or `tokenizer.model` beside the weights. |
| `embeddings` | Whether the output layer reuses the input embedding: no `output.weight` beside `token_embd.weight`, or `tie_word_embeddings`. |
| `note` | Shards, shared PyTorch storage, and similar facts. Notes about the file's layout, such as GGUF alignment, are in `tenspile -i` and `--json`. |

A quantized Hugging Face pack:

```
$ tensinfo qwen-gptq
file:         qwen-gptq/model.safetensors (safetensors), 438 MiB of tensor data
model:        qwen2 (Qwen2ForCausalLM)
parameters:   184.3M in 794 tensors, counting packed values as stored
dtypes:       F16 75%, I32 25%
layers:       24, hidden 896, heads 14 (2 kv), context 32768
vocabulary:   151936
quantization: gptq bits 4 group 128
tokenizer:    tokenizer.json beside the weights
embeddings:   tied input and output (tie_word_embeddings)
```

## The tensor table: `-t`, `--fold`, `--sort`

`-t` adds an aligned table after the overview: name, dtype, shape, bytes, offset, and parameters. A shared parent is a heading, and the leaf name is indented under it. `bytes` is a power-of-two size (`64 MiB`) and `params` a decimal count (`16.8M`). `shard` and `detail` columns appear only when a tensor has them. Tensor names after the file limit the table to those tensors.

```
$ tensinfo -t small.gguf
file:         small.gguf (gguf v3), 1.4 KiB of tensor data
model:        llama
parameters:   532 in 5 tensors
dtypes:       F32 48%, Q8_0 48%, F16 2%, BF16 1%, I32 1%
quantization: GGUF blocks Q8_0
tokenizer:    none: no tokenizer.ggml.model, so llama.cpp will not load the file

name    dtype  shape    bytes  offset  params  detail
w_f32   F32    [64, 4]  1 KiB     384     256
b_f16   F16    [8]       16 B    1408       8  16 bytes of alignment padding follow the tensor
i_i32   I32    [2, 3]    24 B    1440       6  8 bytes of alignment padding follow the tensor
e_bf16  BF16   [3, 2]    12 B    1472       6  20 bytes of alignment padding follow the tensor
w_q8_0  Q8_0   [64, 4]  272 B    1504     256  16 bytes of alignment padding follow the tensor
```

GGUF shapes are the file's own `ne[]` order, innermost axis first; a conversion reverses them.

`--fold` merges one numeric name segment across a consecutive run when dtype, shape, stored size, shard, and detail all match. `model.layers.[0-29].` is one copy of each leaf, not the sum, and the offset column is `-`. A tensor that differs, or a gap in the numbers, stays on its own line. A second number in the name, such as an expert id, is not folded in the same pass.

```
$ tensinfo -t --fold smollm2-135m
...
name                               dtype  shape           bytes     offset  params
model.
  embed_tokens.weight              BF16   [49152, 576]   54 MiB      30536   28.3M
model.layers.[0-29].
  input_layernorm.weight           BF16   [576]         1.1 KiB          -     576
model.layers.[0-29].mlp.
  down_proj.weight                 BF16   [576, 1536]   1.7 MiB          -  884.7K
  gate_proj.weight                 BF16   [1536, 576]   1.7 MiB          -  884.7K
  up_proj.weight                   BF16   [1536, 576]   1.7 MiB          -  884.7K
model.layers.[0-29].
  post_attention_layernorm.weight  BF16   [576]         1.1 KiB          -     576
model.layers.[0-29].self_attn.
  k_proj.weight                    BF16   [192, 576]    216 KiB          -  110.6K
  o_proj.weight                    BF16   [576, 576]    648 KiB          -  331.8K
  q_proj.weight                    BF16   [576, 576]    648 KiB          -  331.8K
  v_proj.weight                    BF16   [192, 576]    216 KiB          -  110.6K
model.
  norm.weight                      BF16   [576]         1.1 KiB  269059400     576
```

The table is in file order. `--sort` sorts it by name instead, one dot-separated segment at a time. A segment that is all digits sorts by its number and before any other segment, so `model.layers.2.` comes before `model.layers.10.`. Other segments sort by bytes. `--fold` and `--sort` each imply `-t`.

## Metadata: `-m`

`-m` lists every metadata key after the overview. A value longer than 120 characters, such as a GGUF vocabulary, is cut; a cut array gives its item count and the length of its full JSON text. `--json` gives every value whole.

## For a script: `--json`

`--json` prints one document: the overview (with the exact parameter count as `parameters_exact`), every metadata key in full, every tensor, and every note.

```bash
tensinfo --json small.safetensors | jq -r '.members[] | "\(.name) \(.numel)"'
tensinfo --json model/ | jq -r '.overview.model'
```

`-t`, `-m`, and `--fold` do not apply to `--json`, which already holds everything.

## tenspile's own listing

`tenspile -t` lists names, `tenspile -tv` the same fields as tab-separated text, and `tenspile -i` the header summary, for scripts that read lines. The options `--columns`, `--fold`, `--sort`, and `--json` moved from tenspile to tensinfo; given to tenspile, they stop with a pointer to the tensinfo command.

## See also

`man tensinfo`, [Import and export](import-export.md) for tenspile, and [tensdiff](tensdiff.md) to compare two files.
