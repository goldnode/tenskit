# tenskit

A suite of command-line utilities for tensor containers: `.safetensors`, GGUF format versions 2 and 3, `.npy`, `.npz`, rawtensor/1, PyTorch zip checkpoints, ONNX initializers, and a safetensors shard index. tenskit lists, edits, compares, converts, and quantizes them.

```bash
tenspile -tvf model.safetensors                       # list tensors with dtype, shape, and size
tensinfo model/                                       # short overview: model type, size, dtypes, quantization
tensinfo -t --fold model/                             # aligned tensor table, repeated layers folded
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

tenskit has five utilities, which share one library:

- `tenspile` lists, extracts, creates, and edits a container, like `tar` for tensors. Its listing is plain text: names, or tab-separated fields.
- `tensinfo` describes a container for a person: a short overview of the model, and on request an aligned tensor table, every metadata key, or JSON.
- `tensdiff` compares two containers by canonical dtype, shape, bytes, and metadata. It exits 1 on a difference and 2 when a file cannot be read.
- `tensconv` copies a whole container into another format. `--rename` maps Hugging Face names to GGUF.
- `tensquant` dequantizes GGUF blocks and GPTQ, AWQ, bitsandbytes, compressed-tensors, and MLX packs, and writes Q8_0 or Q4_0 GGUF.

`cargo build --release` builds all five. Each takes `-h` for a short help. Section 1 manual pages are in [man/](man/). The build copies them to `target/release/share/man/man1` (or the debug directory) with the version from `Cargo.toml` filled in, so `MANPATH=target/release/share/man man tenspile` shows one.

## Install

Binaries for Linux on 64-bit Intel and ARM are available from the [Releases](https://github.com/goldnode/tenskit/releases) page. macOS and Windows to follow.

On Linux, packages can also be installed and updated using your OS tools, such as `apt`, `dnf`, and `zypper`. See [packages.goldnode.org](https://packages.goldnode.org/) for how to set this up.

[Installing tenskit](docs/install.md) has detailed instructions, including how to install without root and how to clone and build with Cargo.

## Documentation

Start with [Getting started](docs/getting-started.md), a five-minute walk-through on the small files that come with the specs.

- [Import and export](docs/import-export.md): operands, where each type can go, and what the listing columns mean
- Formats: [safetensors](docs/safetensors.md), [shard index](docs/shards.md), [GGUF](docs/gguf.md), [NumPy `.npy` and `.npz`](docs/numpy.md), [rawtensor](docs/raw.md), [PyTorch](docs/pytorch.md), [ONNX](docs/onnx.md)
- Utilities: tenspile (the pages above), [tensinfo](docs/tensinfo.md), [tensdiff](docs/tensdiff.md), [tensconv](docs/tensconv.md), [tensquant](docs/tensquant.md)

## Specifications

Every format tenskit reads or writes follows a set of YAML and JSON format specifications. If a specification and tenskit disagree, the specification wins. The spec file names below, such as `gguf.yaml`, name those specifications.

Which formats are compiled in is listed in [generated/formats.txt](generated/formats.txt). Only those modules from [src/formats/](src/formats/) are built.

The formats implemented now:

- `safetensors.yaml`, including the shard index
- `gguf.yaml` — little-endian versions 2 and 3
- `npy.yaml` and `npz.yaml`
- `raw-sidecar.yaml` — rawtensor/1
- `pytorch.yaml` — zip checkpoints, with a restricted unpickler
- `onnx.yaml` — graph initializers
- GPTQ, AWQ, bitsandbytes, compressed-tensors, and MLX, which are naming conventions inside those containers. Packed tensors stay packed.

The other format specifications are in scope for this project and are not implemented yet. Overview specs stay unimplemented until they describe a byte layout. Legacy GGML, a legacy PyTorch pickle, and the PyTorch tar format are recognized and refused. Pinned upstream commits live in each spec's `sources`, not in this README.

## License

[MIT](LICENSE). Copyright 2026 Nick Reynolds (goldnode).
