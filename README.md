# tenspile

Command-line utility for inspecting and editing tensor containers: `.safetensors`, GGUF format versions 2 and 3, `.npy`, `.npz`, rawtensor/1, PyTorch zip checkpoints, ONNX initializers, and a safetensors shard index.

Nick Reynolds ([goldnode](https://github.com/goldnode) on GitHub, [goldnode](https://x.com/goldnode) on X).

```
tenspile -tvf model.safetensors
tenspile -t --columns --fold -f model.safetensors
tenspile -i model.gguf
tenspile -xf model.safetensors -C out embed
tenspile -cf out.safetensors embed=embed.npy --metadata format=pt
tenspile -uf model.safetensors norm=new_norm.npy
tenspile --delete -f model.safetensors optimizer.step
tensdiff a.safetensors b.gguf
```

`cargo build --release` builds `tenspile`, `tensdiff`, `tensconv`, and `tensquant`. `tenspile -h` lists the archive flags. `tensdiff LEFT RIGHT` compares container type, member names, canonical dtypes, canonical shapes, logical tensor bytes, and metadata. `--match-names` pairs tensors through the GGUF role table. `general.architecture` selects the architecture, and a tie is reported as `may be A, B, or C`. GGUF alignment padding is not part of the value. Exit 1 means they differ. `tensdiff` exits 2 when a file cannot be read. `tensconv` copies one container into another, and `--rename` applies the GGUF name table. `tensquant --dequant` writes float32 from a transcribed pack, and `--quantize q4_0` or `q8_0` writes those two GGUF types. `-v` stays tab-separated. `--columns` aligns that listing for a terminal, and `--fold` collapses a consecutive run of identical layers such as `model.layers.[3-5]`. The commands are described in [docs/import-export.md](docs/import-export.md). Section 1 manual pages are in [man/](man/). `man man/tenspile.1` shows one.

## Specifications

[tensdataspecs_readonly](tensdataspecs_readonly/README.md) is the source of truth for every format this tool reads or writes. The dtype registry, the cross-format mapping, and the format ids are loaded at build time from that tree. If a spec and this program disagree, the spec wins unless the spec maintainer changes it. Do not edit that folder here.

Which of those formats are compiled in is the list in [local/supported-formats.txt](local/supported-formats.txt). `build.rs` reads it and includes only those modules from [src/formats/](src/formats/). Delete a line to leave a format out of the binary. Delete the matching file under `src/formats/` as well when you want it gone from the source tree.

The formats implemented now:

- [safetensors.yaml](tensdataspecs_readonly/specs/formats/safetensors.yaml), including the shard index
- [gguf.yaml](tensdataspecs_readonly/specs/formats/gguf.yaml) — little-endian versions 2 and 3
- [npy.yaml](tensdataspecs_readonly/specs/formats/npy.yaml) and [npz.yaml](tensdataspecs_readonly/specs/formats/npz.yaml)
- [raw-sidecar.yaml](tensdataspecs_readonly/specs/formats/raw-sidecar.yaml) — rawtensor/1
- [pytorch.yaml](tensdataspecs_readonly/specs/formats/pytorch.yaml) — zip checkpoints, with a restricted unpickler
- [onnx.yaml](tensdataspecs_readonly/specs/formats/onnx.yaml) — graph initializers
- GPTQ, AWQ, bitsandbytes, compressed-tensors, and MLX, which are naming conventions inside those containers. Packed tensors stay packed.

The other files in `tensdataspecs_readonly/specs/formats/` are in scope for this project and are not implemented yet. Overview specs stay unimplemented until they describe a byte layout. Legacy GGML and a PyTorch pickle that is not a zip checkpoint are recognized and refused. Pinned upstream commits live in each spec's `sources`, not in this README.

How tensors move between the implemented formats is in [docs/import-export.md](docs/import-export.md).

## License

[MIT](LICENSE). Copyright 2026 Nick Reynolds (goldnode).
