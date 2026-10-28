# tenskit

A suite of command-line utilities for tensor containers: `.safetensors`, GGUF format versions 2 and 3, `.npy`, `.npz`, rawtensor/1, PyTorch zip checkpoints, ONNX initializers, and a safetensors shard index. tenskit lists, edits, compares, converts, and quantizes them.

Nick Reynolds ([goldnode](https://github.com/goldnode) on GitHub, [goldnode](https://x.com/goldnode) on X).

```bash
tenspile -tvf model.safetensors                       # list tensors with dtype, shape, and size
tenspile -t --columns --fold -f model/                # aligned listing, repeated layers folded
tenspile -i model.gguf                                # header summary and metadata
tenspile -xf model.safetensors -C out embed           # extract one tensor as out/embed.npy
tenspile -cf out.safetensors embed=embed.npy --metadata format=pt
tenspile -uf model.safetensors norm=new_norm.npy      # replace a tensor
tenspile --delete -f model.safetensors optimizer.step
tensdiff model.safetensors model.gguf                 # compare two files of any format
tensconv --rename -o model.gguf model/                # Hugging Face directory to GGUF names
tensquant --dequant -o model-f32.safetensors model.gguf
tensquant --quantize q8_0 -o model-q8.gguf model/
```

tenskit has four utilities, which share one library:

- `tenspile` lists, extracts, creates, and edits a container, like `tar` for tensors.
- `tensdiff` compares two containers by canonical dtype, shape, bytes, and metadata. It exits 1 on a difference and 2 when a file cannot be read.
- `tensconv` copies a whole container into another format. `--rename` maps Hugging Face names to GGUF.
- `tensquant` dequantizes GGUF blocks and GPTQ, AWQ, bitsandbytes, compressed-tensors, and MLX packs, and writes Q8_0 or Q4_0 GGUF.

`cargo build --release` builds all four. Each takes `-h` for a short help. Section 1 manual pages are in [man/](man/). The build copies them to `target/release/share/man/man1` (or the debug directory), so `MANPATH=target/release/share/man man tenspile` shows one.

## Documentation

Start with [Getting started](docs/getting-started.md), a five-minute walk-through on the small files that come with the specs.

- [Import and export](docs/import-export.md): operands, where each type can go, and what the listing columns mean
- Formats: [safetensors](docs/safetensors.md), [shard index](docs/shards.md), [GGUF](docs/gguf.md), [NumPy `.npy` and `.npz`](docs/numpy.md), [rawtensor](docs/raw.md), [PyTorch](docs/pytorch.md), [ONNX](docs/onnx.md)
- Utilities: tenspile (the pages above), [tensdiff](docs/tensdiff.md), [tensconv](docs/tensconv.md), [tensquant](docs/tensquant.md)

## Specifications

[tensdataspecs_readonly](tensdataspecs_readonly/README.md) is the source of truth for every format tenskit reads or writes. The dtype registry, the cross-format mapping, and the format ids are loaded at build time from that tree. If a spec and tenskit disagree, the spec wins unless the spec maintainer changes it. Do not edit that folder here.

Which of those formats are compiled in is the list in [local/supported-formats.txt](local/supported-formats.txt). `build.rs` reads it and includes only those modules from [src/formats/](src/formats/). Delete a line to leave a format out of the binary. Delete the matching file under `src/formats/` as well when you want it gone from the source tree.

The formats implemented now:

- [safetensors.yaml](tensdataspecs_readonly/specs/formats/safetensors.yaml), including the shard index
- [gguf.yaml](tensdataspecs_readonly/specs/formats/gguf.yaml) — little-endian versions 2 and 3
- [npy.yaml](tensdataspecs_readonly/specs/formats/npy.yaml) and [npz.yaml](tensdataspecs_readonly/specs/formats/npz.yaml)
- [raw-sidecar.yaml](tensdataspecs_readonly/specs/formats/raw-sidecar.yaml) — rawtensor/1
- [pytorch.yaml](tensdataspecs_readonly/specs/formats/pytorch.yaml) — zip checkpoints, with a restricted unpickler
- [onnx.yaml](tensdataspecs_readonly/specs/formats/onnx.yaml) — graph initializers
- GPTQ, AWQ, bitsandbytes, compressed-tensors, and MLX, which are naming conventions inside those containers. Packed tensors stay packed.

The other files in `tensdataspecs_readonly/specs/formats/` are in scope for this project and are not implemented yet. Overview specs stay unimplemented until they describe a byte layout. Legacy GGML, a legacy PyTorch pickle, and the PyTorch tar format are recognized and refused. Pinned upstream commits live in each spec's `sources`, not in this README.

## License

[MIT](LICENSE). Copyright 2026 Nick Reynolds (goldnode).
