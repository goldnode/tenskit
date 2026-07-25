# tenspile

Command-line utility for inspecting and editing tensor containers: `.safetensors`, GGUF format versions 2 and 3, `.npy`, `.npz`, raw tensors, and a safetensors shard index.

Nick Reynolds ([goldnode](https://github.com/goldnode) on GitHub, [goldnode](https://x.com/goldnode) on X).

```
tenspile -tvf model.safetensors
tenspile -i model.gguf
tenspile -xf model.safetensors -C out embed
tenspile -cf out.safetensors embed=embed.npy --metadata format=pt
tenspile -uf model.safetensors norm=new_norm.npy
tenspile --delete -f model.safetensors optimizer.step
```

`cargo build --release` builds the binary. `tenspile -h` lists the flags.

How tensors move between these formats is in [docs/import-export.md](docs/import-export.md).

## Specifications

Parsers follow these documents. tenspile does not vendor them, and its MIT license does not replace theirs.

- **safetensors.** Format section of the [safetensors README](https://github.com/huggingface/safetensors/blob/e246a2560645b7525f5775669ed816eb57c5bcc8/README.md) and the `Dtype` enum in `safetensors/src/tensor.rs` at commit `e246a25` (2026-09-22). Apache-2.0. The JSON Schema dtype pattern lags the enum; the enum is the list of names this tool accepts.
- **GGUF.** Little-endian file-format versions 2 and 3, from the [GGUF specification](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/docs/gguf.md) and [`include/gguf.h`](https://github.com/ggml-org/ggml/blob/3612c0af93f2cfc8f048295300a61187723b12ae/include/gguf.h) in ggml at commit `3612c0a` (2026-10-12). MIT. The two versions share a field layout. Version 3 is what current writers emit. Version 1 and big-endian files are refused. See [docs/gguf.md](docs/gguf.md). The file layout is the ggml document. The [gguf-py](https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/gguf-py/README.md) tools (llama.cpp `cbe28bb`, 2026-10-12) are prior art for metadata commands.
- **NumPy `.npy` and `.npz`.** [NEP 1](https://github.com/numpy/numpy/blob/3ae3cc9657c32f530700831273bca9f680488403/doc/neps/nep-0001-npy-format.rst) (public domain; its alignment figure is stale) and the living writer in [`numpy/lib/_format_impl.py`](https://github.com/numpy/numpy/blob/3ae3cc9657c32f530700831273bca9f680488403/numpy/lib/_format_impl.py) at commit `3ae3cc9` (2026-10-12), where `ARRAY_ALIGN` is 64. NumPy is BSD-3-Clause.
- **Safetensors shard index.** `model.safetensors.index.json` as parsed by Hugging Face: a `weight_map` from tensor name to shard file and `metadata.total_size`. See the [safetensors metadata parsing doc](https://github.com/huggingface/safetensors/blob/main/docs/source/metadata_parsing.mdx) and [`SafetensorsIndexJson` in huggingface.js](https://github.com/huggingface/huggingface.js/blob/main/packages/hub/src/lib/parse-safetensors-metadata.ts). Shard names are relative. The index is not a concatenation of the shard files.

## License

[MIT](LICENSE). Copyright 2026 Nick Reynolds (goldnode).
