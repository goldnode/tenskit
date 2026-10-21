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

One warning per pair of dtypes gives the count and the first tensor's name.

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

Tensors that share one storage in a PyTorch file, such as a tied `lm_head.weight`, are each written in full into safetensors, which cannot share storage. A warning names each pair. safetensors' `save_model` and Hugging Face's `save_pretrained` keep one name instead and re-tie on load. Writing every name in full is valid, and CONV-010 allows it. The output is larger but loads either way.

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
- llama.cpp's converter reads transformers' `AutoConfig` dict, not `config.json` itself (GGUF-039). That dict adds the defaults of the model's config class. tenspile has no Python, so it reads `config.json` (GGUF-040) and adds only the defaults GGUF-039 lists. For 14 model types, `tables/hf-config-defaults.json` gives every default `to_dict()` adds: `llama`, `mistral`, `mixtral`, the Qwen 2 and 3 types, the Gemma types, `phi3`, `smollm3`, `olmo2`, and `granite`. A default applies only to a key that `config.json` lacks. The derived keys follow GGUF-039's formulas:
  - `head_dim` for `llama`;
  - `num_key_value_heads` copied from the attention heads where transformers copies it;
  - `sliding_window` null for the Qwen types unless `use_sliding_window` is true;
  - `layer_types` for Qwen, Gemma 2 and 3, and SmolLM3, with Gemma 3's halved window under bidirectional attention.

  Another model type gets only the common `PretrainedConfig` keys, such as `max_length` 20, and a warning.

  A `text_config` gets the defaults for its own `model_type`, and the common keys, before it is merged. Its values then win over the top level, as they do in transformers' dict. For Gemma 3, the top-level `eos_token_id` [1, 106] becomes 1, and `architectures` becomes null. The converter class and the `hf_arch` in filter conditions still come from the top-level `architectures`. When `config.json` has no `text_config`, a composite type gets one built as transformers builds it. The built `text_config` starts from the top-level keys the composite passes on, then gets the text model type's defaults and derived formulas, then its default `text_config` for anything still missing:
  - `qwen2_vl`, `qwen2_5_vl`, `glm4v`, and `glm4v_moe` pass on every top-level key they do not take themselves;
  - `fuyu`, `paligemma`, and a few others pass on a listed set;
  - most, such as `llava`, pass on nothing, so their text defaults replace the top-level values in the merge, as they do in llama.cpp.

  A warning is given only when the text model type is not in the defaults table.

  It renames `llm_config`, `lm_config`, `thinker_config.text_config`, `language_config`, and `lfm` to `text_config`, in that order, before merging. The last one present wins and replaces any `text_config`. A null one, or a `thinker_config` without `text_config`, stops the conversion, as it stops llama.cpp's. One warning says that a key depending on another transformers default may differ from llama.cpp's.
- Hyperparameters follow `hf_config_to_gguf`, whose rows transcribe llama.cpp's `TextModel.set_gguf_parameters`. `text_config` is merged over the root object, and the first listed key that is present wins, as llama.cpp's `find_hparam` takes it. A null value counts as present: a base row is then not written, and a class rule that needs the value stops the conversion, because llama.cpp's converter would fail writing it. A sibling `config.json` is read only for those keys. A row is written only when its condition holds:
  - `attention.head_count_kv` only when the config gives it, and `attention.causal` only as false, when `is_causal` is false or `use_bidirectional_attention` is true;
  - `attention.key_length` and `value_length` only from an explicit `head_dim`;
  - the rope parameters P are `rope_parameters` when that key is present, else `rope_scaling`, and a null gives an empty set. Unless P is nested (`full_attention` or `sliding_attention`), the top-level `rope_theta`, `partial_rotary_factor` (or `rope_pct`), `original_max_position_embeddings`, and the local theta keys fill what P lacks, and `type` becomes `rope_type` when P has none. A value already in P is never replaced;
  - `rope.freq_base` from the rope parameters' `rope_theta`, and `rope.freq_base_swa` from `sliding_attention.rope_theta`;
  - `rope.scaling.*` by `rope_type`: `linear` and `yarn` write the type and factor, and `yarn` also `original_context_length` and the `yarn_*` keys; `su` or `longrope` writes only the type `longrope`; `llama3` and `dynamic` write none;
  - `rope.dimension_sections` from `mrope_section`, `rope.freq_base_swa`, the expert counts and group counts, and `expert_gating_func` (softmax 1, sigmoid 2, sqrtsoftplus 4).

  The base writes no `vocab_size`, `rope.dimension_count`, or `attention.sliding_window`.
- Each converter class then writes its own keys, as llama.cpp's does (GGUF-037). tensconv follows the class's `set_gguf_parameters` chain in `tables/gguf-config-keys.json`, base first. The base rows apply only when the chain reaches `TextModel`. Phi-3 and Gemma-2, for example, never do. For every other step, tensconv evaluates each key's guard and value from `config.json` and writes the key with its listed type. Examples are `llama.vocab_size` from LlamaModel, and RWKV6Qwen2's constant context length. Steps run in source order, and each `super()` call runs the rest of the chain at its line, so a key a class writes before calling `super()`, as DistilBERT's `layer_norm_epsilon` does, is replaced by the base row. A later write of the same key replaces the earlier one.

  Before the keys are read, the converter's own changes to its config dict are applied, in run order (`hparams_mutations`). Those in `__init__`, `prepare_tensors`, and `prepare_metadata` before the key writes count. They include TextModel's merge of `text_config`, CommandR2 taking `max_position_embeddings` from `model_max_length`, Dots1 copying `n_routed_experts`, and NemotronH setting `num_hidden_layers` from `layers_block_type`. A local dict that an `__init__` passes on as `hparams=` is the config dict itself, because llama.cpp's converter passes no `hparams`. A change that cannot be evaluated is named in a warning, because the keys read after it may differ.

  Classes that `gguf_class_parameters` transcribes by hand follow the transcription instead of the index:
  - **Pooling (`_try_set_pooling_type`, called by Qwen2, BERT, and about 40 converters):** `pooling_type` comes from sentence-transformers' `modules.json` and its pooling `config.json`, or from `embedding_config.json`. A plain causal LM has neither file and writes none. `classifier_pooling` and `classifier_activation` give the classifier keys. An unsupported mode stops the conversion, as it stops llama.cpp's.
  - **LLaMA:** follows the index, which adds `vocab_size`. `rope.dimension_count` is `head_dim`, else `hidden_size // num_attention_heads`, as the `hf_config_to_gguf` row states, because the index cannot evaluate it.
  - **Phi-3 and PhiMoE:** these write their own list of keys. `rope.scaling.original_context_length` comes from the top-level `original_max_position_embeddings`, and `rope.dimension_count` is `int(partial_rotary_factor * hidden_size) // num_attention_heads`. `attention.sliding_window` is 0 when the config has none. Neither class writes `rope.scaling.type` or `factor`. With `long_factor`, they also write `rope.scaling.attn_factor`: Phi-3.5-mini gets `sqrt(1 + ln 32 / ln 4096)` = 1.1902381. Factor lists whose length is not half the rope dimension stop the conversion, as they stop llama.cpp's converter. These keys were worth about 2% perplexity on Phi-3.5.
  - **Gemma-2:** writes its own list, with `attn_logit_softcapping`, `final_logit_softcapping`, `key_length` and `value_length` from `head_dim`, and `attention.sliding_window`, but no `rope.freq_base`. A config without any one of them stops the conversion. llama.cpp would otherwise use 50 and 30 for the softcapping and run another variant silently wrong.
  - **Gemma-3:** this transcription also applies to Gemma-3n, Gemma-4, and EmbeddingGemma, whose chains pass through Gemma-3. It writes after the base rows, with defaults when the config has no value. `final_logit_softcapping` is written only when it is set and not 0. `attention.sliding_window` is left out when `sliding_window_pattern` is 1. A config that sets `attn_logit_softcapping` stops the conversion, because llama.cpp's converter asserts that it is absent.

  The evaluator understands a small part of Python: config lookups, `find_hparam`, comparisons, arithmetic, single `for` comprehensions and generator expressions (list, set, or in a call), `re.match` with `.group`, and a few built-ins (`int`, `float`, `len`, `any`, `all`, `sum`, `sorted`, `min`, `max`). `self.model_tensors` holds the checkpoint's tensor names after the `filter_tensors` chain and the global filters, so a rule that counts layers from the names, as DeepSeek V4 DSpark's `block_count` does, is evaluated on them. Attributes follow `converter_attributes`. `self.hparams` is the config dict, `self.rope_parameters` is P, `self.dir_model` is the model directory, and `self.is_mistral_format` is false. `no_mtp` and `mtp_only` are false, because they are command-line switches that default to off. Any other `self.<attr>`, `block_count` included, is evaluated by running the converter's `attribute_assignments` in order. Each site's guard is checked first, then its value is assigned with `=` or added with `+=`, reading the site's own guarded locals. So the next-token layers that Qwen3.5, GLM-4.5, DeepSeek, Bailing, Cohere2-MoE, and others add to `block_count` are counted from `config.json`. Each assignment's phase places it in llama.cpp's run order. Assignments in `__init__`, `prepare_tensors`, and `prepare_metadata` before the key writes count. Those after the key writes, such as LLaMA's `dir_model` in `set_vocab`, or in an `__init__` the chain never reaches, are skipped. An assignment the evaluator cannot follow, or one whose place in the order is not known while its guard may hold, leaves the key out with the warning. A key whose guard or value depends on anything else, such as a local variable assigned in more than one place or an enum constant, is left out. An array value is typed as gguf-py types it (GGUF-041): the element type is fixed where the writer fixes it, and otherwise taken from the first element (string, nested array, float F32, bool, int I32). A per-layer list of ints is therefore INT32, and an empty list is left out. One warning names all such keys. A config without a block count, a `yarn` scaling without `original_max_position_embeddings`, or an unknown gating function stops the conversion, as it stops llama.cpp's converter.
- The sampler defaults in `generation_config.json`, beside `config.json`, become `general.sampling.*` (GGUF-038). These are `temperature` as `temp`, `top_k`, `top_p`, `min_p`, `repetition_penalty` as `penalty_repeat`, and the `xtc_*`, `mirostat*`, `penalty_last_n`, and `sequence` fields. Each is written only when it is present and not null, whatever `do_sample` says. llama.cpp uses them as defaults that a command-line option overrides, so Qwen2.5's `repetition_penalty: 1.1` changes its greedy output unless `--repeat-penalty` is given.
- Value changes run unless `--names-only` is set. A converter record with `complete: true` applies its listed branch: `+ 1`, unsqueeze, squeeze, the clip `view`/`permute`, `-exp`, and the recorded renames. The expression text is never evaluated. Any other expression stops the conversion and prints it. A BF16 tensor is changed in F32, as llama.cpp's converter does before it changes any value (GGUF-036), so a Gemma norm's `weight + 1` is exact. It returns to BF16 afterwards unless it is one of the tensors written as F32 below. An F16 tensor is changed in F16, as in llama.cpp.
- LLaMA (the `LlamaModel` converter, architecture `llama`) permutes the rows of `q_proj` and `k_proj` weights and biases. That is why `tensdiff --match-names` reports their values as different. It also removes a `text_model.` segment, and adds `model.` in front of every name when `architectures[0]` is `LlamaModel`. An EAGLE-3 draft (`draft_vocab_size` with one layer) and a Mixtral-style checkpoint with `block_sparse_moe.experts` tensors are refused, because the spec's LLaMA transcription does not cover them. Other converters that permute like LLaMA, such as Granite, SmolLM3, and Arcee, are not transcribed and stop.
- Qwen3.5 applies the V-head reorder, `A_log` as `-exp`, norms other than `linear_attn.norm.weight` as `weight + 1`, and the `conv1d` squeeze. Full-attention Q and K are not permuted, and the reorder is skipped when the linear head counts match.
- Any other architecture whose record is `complete: false` stops. The message explains why, suggests `--names-only`, and lists the `unexpanded` steps one per line, as the spec quotes them from llama.cpp's Python source.
- A class that inherits llama.cpp's base `modify_tensors` is `rename_only`: its tensors are renamed and their values kept (Phi-3, StarCoder2, OLMo 2, Command R, and others).
- Which classes are followed is GGUF-030. A class whose record lists a hook that changes the tensor set (`prepare_tensors`, `index_tensors`, `get_tensors`, `dequant_model`, or an `__init__` that edits the tensors) stops, and names the hooks. So does a class with a `filter_tensors` the spec has not expanded. A complete `filter_tensors`, which covers the methods it inherits too, leaves out the tensors its conditions name, with a warning for each. Then, for a text model, llama.cpp's general text-model filter leaves out vision, audio, and projector tensors (`visual.`, `vision_`, `audio.`, `mlp…` at the top level, and others), again with a warning for each. LLaMA (`LlamaModel` for `llama`) and Qwen3.5 (`Qwen3_5TextModel` for `qwen35`) are transcribed in their own spec sections and are not checked this way; Qwen3.5 MoE is.
- `--rename` into GGUF does not build the tokenizer keys, and warns that llama.cpp will not load the file until `tokenizer.ggml.*` is added (GGUF-033, GGUF-034). The spec has not yet transcribed how llama.cpp's converter builds them (GGUF-Q5). A GGUF source's keys are copied.
- Every `--rename` leaves out names ending in `.attention.masked_bias`, `.attention.bias`, or `.rotary_emb.inv_freq`, which llama.cpp's converter skips for every model.
- Some converters also compute tensors (`generate_extra_tensors`). The spec lists each one, with the condition under which llama.cpp writes them (GGUF-035). tensconv does not write them, and warns once, only when the condition holds for the model's `config.json`. LLaMA writes `rope_freqs.weight` only when `rope_type` is `llama3`, so most LLaMA models, SmolLM2 among them, get no warning. Phi-3 with `rope_scaling.long_factor` would get `rope_factors_long` and `rope_factors_short`; llama.cpp loads the file without them, but runs RoPE without the factors, so its answers can differ. A condition tenspile cannot test, such as a file beside the weights, is quoted in the warning. A converter that repacks checkpoint tensors (GPT-OSS, DeepSeek V4, and others with MXFP4 experts) stops, like the other hooks.
- Into GGUF, the tensors llama.cpp's converter writes as F32 are widened to F32 when they are F16 or BF16 (GGUF-024 steps 2 and 3, as GGUF-029 asks of any writer that uses llama.cpp's names): rank 1 and below, `*_norm.weight`, names that do not end in `.weight`, such as biases, and a few roles such as `ffn_gate_inp`. The widen is exact. No llama.cpp backend can apply a BF16 norm or bias, and only CUDA and Vulkan can apply an F16 one (GGUF-028). On the CPU, a file without the widen stops with `binary_op: unsupported types`. llama-quantize does not fix this later, because it never converts a norm. One warning gives the count. `--names-only` skips this too.

`--names-only` renames and skips every value change, which is useful for inspecting the mapping.

## See also

[tensdiff](tensdiff.md) for checking a conversion, [tensquant](tensquant.md) for quantized tensors, and `man tensconv`.
