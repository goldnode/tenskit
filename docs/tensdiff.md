# tensdiff

`tensdiff LEFT RIGHT` compares two tensor containers of any supported format. It checks the container format, the member names, each tensor's dtype, shape, and bytes, and the metadata. Comparison does not stop at the first difference.

```bash
tensdiff before.safetensors after.safetensors
tensdiff model.safetensors model.gguf
tensdiff --match-names --fold model/ model.gguf
tensdiff --json a.npz b.npz
```

A directory is the safetensors weight set inside it, the same rule `tenspile` uses. A `.safetensors` file beside `config.json` is refused unless `--force-single-file` is written immediately before that path.

## Reading the report

```
$ tensdiff small.safetensors small.gguf
format: small.safetensors safetensors header+blob  vs  small.gguf gguf 3
only in small.safetensors: s_f32
only in small.gguf: w_q8_0
metadata: general.architecture  only in small.gguf
metadata: test.answer  only in small.gguf
metadata: test.note  only in small.gguf
```

| Line | Meaning |
| --- | --- |
| `format: A … vs B …` | The containers differ. Each side shows its path, format, and version. |
| `only in PATH: NAME` | A tensor name on one side only. |
| `NAME : DTYPE : SHAPE` | A tensor that differs, or a pair matched under another name. `=>` marks a field that differs, so a field written once is the same on both sides. |
| `… : values differ at byte N` | Dtype and shape match and the bytes do not. N is the first differing byte of the logical tensor. |
| `metadata: KEY  only in PATH` | A key on one side only. |
| `metadata: KEY  LEFT  vs  RIGHT` | A key whose values differ. |
| `names: ARCH` | With `--match-names`, the architecture whose role table paired the names. |

A tensor that matches on both sides is not printed. Lines are sorted by name, and a run of digits sorts as a number, so `blk.10` follows `blk.9`.

The comparison is on canonical values:

- dtypes by canonical id, so `F32`, `float32`, and GGUF `F32` are the same;
- shapes in canonical order, so a GGUF `ne[]` is reversed first;
- bytes of the logical tensor, so GGUF alignment padding is ignored, a big-endian source is byteswapped first, and a quantized tensor is compared as its block bytes (not dequantized).

## Exit status

| Status | Meaning |
| --- | --- |
| 0 | Same format, names, dtypes, shapes, bytes, and metadata. |
| 1 | A difference was printed. A different container format counts, so a file and its conversion exit 1 even when every tensor matches. |
| 2 | A file could not be read, or the command line was not valid. |

To check that a conversion kept every tensor, look for a report whose only line is `format:`:

```
$ tensconv -o small.pt small.safetensors
$ tensdiff small.safetensors small.pt
format: small.safetensors safetensors header+blob  vs  small.pt pytorch 3
```

With `--json`, that is a `findings` list whose only entry has `"kind": "format"`.

## Pairing renamed tensors

`--match-names` pairs names that differ through the GGUF role table (`tables/gguf-tensor-roles.json`). It is meant for comparing a Hugging Face checkpoint with its GGUF conversion. Here `hf/` is a small LLaMA checkpoint, and `llama.gguf` is that checkpoint after `tensconv --rename`:

```
$ tensdiff --match-names --fold hf llama.gguf
format: hf/model.safetensors safetensors header+blob  vs  llama.gguf gguf 3
names: llama
lm_head.weight => output.weight : F32 : [32, 8]
model.embed_tokens.weight => token_embd.weight : F32 : [32, 8]
model.layers.[0-2].input_layernorm.weight => blk.[0-2].attn_norm.weight : F32 : [8]
model.layers.[0-2].mlp.down_proj.weight => blk.[0-2].ffn_down.weight : F32 : [8, 16]
model.layers.[0-2].mlp.gate_proj.weight => blk.[0-2].ffn_gate.weight : F32 : [16, 8]
model.layers.[0-2].mlp.up_proj.weight => blk.[0-2].ffn_up.weight : F32 : [16, 8]
model.layers.[0-2].post_attention_layernorm.weight => blk.[0-2].ffn_norm.weight : F32 : [8]
model.layers.[0-2].self_attn.k_proj.weight => blk.[0-2].attn_k.weight : F32 : [8, 8] : values differ at byte 35
model.layers.[0-2].self_attn.o_proj.weight => blk.[0-2].attn_output.weight : F32 : [8, 8]
model.layers.[0-2].self_attn.q_proj.weight => blk.[0-2].attn_q.weight : F32 : [8, 8] : values differ at byte 35
model.layers.[0-2].self_attn.v_proj.weight => blk.[0-2].attn_v.weight : F32 : [8, 8]
model.norm.weight => output_norm.weight : F32 : [8]
metadata: format  only in hf/model.safetensors
metadata: general.architecture  only in llama.gguf
…
```

Each pair is printed so you can check the mapping. A pair whose bytes match does not by itself make the exit status 1. Q and K differ because the LLaMA converter permutes their rows. That is expected, and tensdiff does not undo it.

How the architecture is chosen:

- A GGUF `general.architecture` value selects that architecture (`names: llama` above).
- Otherwise every architecture is tried, and the smallest role set that explains every remaining name wins. A tie is reported as `names: may be A, B, or C`.
- If nothing covers the names, the report says `names: no table matched` and compares exact names.

`language_model.` is removed before the lookup. Vision and audio tensors are left unpaired in a language-only file. Qwen3.5 `dt_bias` and `mtp.*` names use the rewrites in `qwen35_tensor_names_to_gguf`.

## Folding layers

`--fold` merges lines that are the same except for one consecutive number. `blk.20` and `blk.21` become `blk.[20-21]`. A number that advances on both sides is written as a range on both sides, and a gap is not bridged:

```
blk.[20-21].attn_norm.weight => model.language_model.layers.[20-21].input_layernorm.weight : F32 => BF16 : [1024]
```

`--fold` does not apply to `--json`.

## JSON

`--json` prints `{"same": bool, "findings": [...]}`. Each finding has a `kind` of `format`, `names`, `only_left`, `only_right`, `tensor`, `metadata_only`, or `metadata`, with the paths, names, and fields involved:

```
$ tensdiff --json small.safetensors small.gguf
{
  "same": false,
  "findings": [
    {
      "kind": "format",
      "left": "small.safetensors safetensors header+blob",
      "right": "small.gguf gguf 3"
    },
    {
      "kind": "only_left",
      "path": "small.safetensors",
      "name": "s_f32"
    },
    …
```
