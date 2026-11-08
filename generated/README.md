# generated

Having standalone static executables has eased use but made this data incredably messy (well, even more messy). A better structure is really needed and is being experimented with.

| Path | What it is |
| --- | --- |
| `spec_tables.rs` | The dtype registry, the format ids, the GGML types, and the name tables, as Rust statics. `src/dtype.rs` includes it. |
| `enabled_formats.rs` | The format modules compiled in. `src/formats/mod.rs` includes it. |
| `formats.txt` | Every specification format id: `+` is compiled in, `-` is not. `build.rs` sets a `format_<id>` cfg for each `+`. |
| `ggml_tables.rs` | The GGML IQ sign masks, IQ4/FP4 value tables, and IQ codebook grids, as f32 arrays. `src/ggml_dequant.rs` includes it. |
| `tensor_roles.rs` | The GGUF tensor roles: per architecture, the Hugging Face to GGUF name patterns (held once in `PAIRS` and indexed) and the converter records, plus a sorted Hugging Face class to architecture index. `src/gguf_names.rs` includes it. |
| `hf_defaults.rs` | The Hugging Face config defaults: the common keys, each model_type's defaults and derived keys, and the composite text_config defaults, with the values as `Lit` constants. `src/gguf_keys.rs` includes it. |
| `config_keys.rs` | The GGUF-037 config-keys index: converters, methods, assignment sites, and set_gguf_parameters sequences as typed records, sorted for binary search, with every reference between them resolved to an index and checked when it is generated, and each expression string held once in `EXPR`. `src/gguf_keys.rs` includes it. |
| `test-vectors/` | The specification's fixtures, for the tests. |
| `bnb_tables.rs` | The bitsandbytes NF4 and FP4 codebooks, as f32 arrays. `src/quant.rs` includes it. |
| `spec-expected/` | The tables above as JSON, for the tests only. |
