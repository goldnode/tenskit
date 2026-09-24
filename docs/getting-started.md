# Getting started

This walk-through uses the small files that come with the specs, so every command and every line of output below can be reproduced. It takes about five minutes.

```bash
cargo build --release
export PATH="$PWD/target/release:$PATH"
vectors="$PWD/tensdataspecs_readonly/specs/test-vectors"
mkdir -p /tmp/tour && cd /tmp/tour
cp "$vectors"/{small.safetensors,small.gguf,small.npz,w_f32.npy} .
```

Each fixture holds the same five small tensors in a different container. `small.gguf` has a `Q8_0` copy of `w_f32` in place of the scalar `s_f32`.

## Look inside a file

`-t` lists member names. `-v` adds the columns, separated by tabs, so the output can go to `cut` or `awk`:

```
$ tenspile -tvf small.safetensors
name	dtype	shape	bytes	offset	params	shard	detail
s_f32	F32	[]	4	328	1	-	-
w_f32	F32	[4, 64]	1024	332	256	-	-
i_i32	I32	[3, 2]	24	1356	6	-	-
e_bf16	BF16	[2, 3]	12	1380	6	-	-
b_f16	F16	[8]	16	1392	8	-	-
```

`--columns` lines the same fields up for reading, and scales the sizes:

```
$ tenspile -t --columns -f small.gguf
name    dtype  shape    bytes  offset  params  detail
w_f32   F32    [64, 4]  1 KiB     384     256
b_f16   F16    [8]       16 B    1408       8  16 bytes of alignment padding follow the tensor
i_i32   I32    [2, 3]    24 B    1440       6  8 bytes of alignment padding follow the tensor
e_bf16  BF16   [3, 2]    12 B    1472       6  20 bytes of alignment padding follow the tensor
w_q8_0  Q8_0   [64, 4]  272 B    1504     256  16 bytes of alignment padding follow the tensor
```

The GGUF shapes are reversed compared with safetensors. GGUF stores `ne[]` with the innermost axis first. tenspile shows the file's own order, and every conversion reverses it. `bytes` is the tensor itself; the padding after it is in `detail`.

`-i` summarizes the header:

```
$ tenspile -i small.gguf
format: gguf
version: 3
path: small.gguf
members: 5
data_bytes: 1408
metadata:
  general.architecture=llama
  test.answer=42
  test.note=tensor-format-specs
dtypes:
  BF16	1	6
  F16	1	8
  F32	1	256
  I32	1	6
  Q8_0	1	256
note: alignment 32
note: data starts at 384
note: shapes are ggml ne[] (innermost first); a conversion reverses them
```

Each `dtypes` row is the type, the number of tensors, and their total element count. A `note` is something the reader noticed. It is often a spec rule id that you can look up under `tensdataspecs_readonly/specs`.

For a large model, `--fold` collapses repeated layers into one block. `--sort` lists the members by name, with layer numbers in numeric order, instead of in file order. The [tensconv](tensconv.md) page shows one.

## Pull tensors out

```
$ tenspile -xf small.safetensors -C out
$ ls out
b_f16.npy
e_bf16.bin
e_bf16.rawtensor.json
i_i32.npy
s_f32.npy
w_f32.npy
```

Each member becomes an `.npy` file when NumPy has a type for it. NumPy has no bfloat16, so `e_bf16` is written as [rawtensor](raw.md), which is a JSON manifest plus a `.bin` file. A quantized GGUF tensor is written the same way, with its block bytes unchanged:

```
$ tenspile -x -f small.gguf --format raw -o q8.rawtensor.json w_q8_0
$ cat q8.rawtensor.json
{
  "format": "rawtensor",
  "version": 1,
  "byte_order": "little",
  "alignment": 64,
  "tensors": [
    {
      "name": "w_q8_0",
      "dtype": "ggml_q8_0",
      "shape": [
        4,
        64
      ],
      "file": "q8.bin",
      "offset": 0,
      "nbytes": 272
    }
  ]
}
```

## Build and edit a file

An operand `KEY=FILE` reads a file as one tensor. `KEY=ARCHIVE:MEMBER` copies one tensor out of another container:

```
$ tenspile -cf new.safetensors w=w_f32.npy b=small.gguf:b_f16 --metadata note=demo
```

Bytes without a header need `--dtype` and `--shape`. `-n` shows the plan without writing:

```
$ head -c 8 /dev/zero > zeros.bin
$ tenspile -n -r -f new.safetensors --dtype F32 --shape 2 z=zeros.bin
dry-run: write new.safetensors
w F32 [4, 64] 1024
b F16 [8] 16
z F32 [2] 8
metadata note=demo
$ tenspile -r -f new.safetensors --dtype F32 --shape 2 z=zeros.bin
$ tenspile -r -f new.safetensors --dtype F32 --shape 2 z=zeros.bin
tenspile: tensor z already exists; use -u to replace it
```

`-r` (append) refuses a name that exists, and leaves the file unchanged. `-u` (update) replaces it, with the new tensor's own dtype and shape; a warning names any change, such as BF16 to F32 from an `.npy` file. `--delete` removes members:

```
$ tenspile --delete -f new.safetensors b
$ tenspile -tvf new.safetensors
name	dtype	shape	bytes	offset	params	shard	detail
w	F32	[4, 64]	1024	160	256	-	-
z	F32	[2]	8	1184	2	-	-
```

Every rewrite goes to a temporary file that is renamed into place, so an interrupted command leaves the original file whole.

GGUF tensors cannot be added or removed, but the metadata can be rewritten. `-o` writes a new file and leaves the original alone:

```
$ tenspile -u -f small.gguf -o named.gguf --metadata general.name=demo
```

## Compare files

```
$ tensdiff small.gguf named.gguf
metadata: general.name  only in named.gguf
```

The exit status is 1 because the files differ. Two different containers that hold the same tensors still differ by their format:

```
$ tensdiff small.safetensors small.gguf
format: small.safetensors safetensors header+blob  vs  small.gguf gguf 3
only in small.safetensors: s_f32
only in small.gguf: w_q8_0
metadata: general.architecture  only in small.gguf
metadata: test.answer  only in small.gguf
metadata: test.note  only in small.gguf
```

`w_f32`, `i_i32`, `e_bf16`, and `b_f16` are not listed. Their types, canonical shapes, and bytes match, even though GGUF stores the shapes reversed and pads the data. See [tensdiff](tensdiff.md).

## Convert and dequantize

`tensconv` copies a whole container into another one. The output name picks the format:

```
$ tensconv -o small.pt small.safetensors
$ tensdiff small.safetensors small.pt
format: small.safetensors safetensors header+blob  vs  small.pt pytorch 3
```

Only the format line differs, so the tensors survived the round trip.

`tensquant --dequant` turns quantized tensors into float32. Float tensors are widened to float32, and integer tensors are copied unchanged:

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

See [tensconv](tensconv.md) and [tensquant](tensquant.md).

## Use it from a script

`--json` prints one JSON document for a listing, for info, and for a diff:

```bash
tenspile -t --json -f small.safetensors | jq -r '.members[] | "\(.name) \(.numel)"'
tensdiff --json small.safetensors small.gguf | jq '.findings[] | select(.kind == "only_left")'
```

Exit status 0 means success, and 1 means an error. For `tensdiff`, 1 means a difference was found and 2 means a file could not be read.

## Where next

- [Import and export](import-export.md): operands, where each tensor type can go, and the listing columns
- [safetensors](safetensors.md), [shard index](shards.md), [GGUF](gguf.md), [NumPy](numpy.md), [rawtensor](raw.md), [PyTorch](pytorch.md), and [ONNX](onnx.md)
- [tensdiff](tensdiff.md), [tensconv](tensconv.md), and [tensquant](tensquant.md)
- `man tenspile` and the other section 1 pages in [man/](../man/)
