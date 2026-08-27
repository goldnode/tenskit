# ONNX initializers

An ONNX model keeps its weights as graph initializers. tenspile reads and writes those tensors. The operators, inputs, and outputs are left alone. The layout and validation ids are in [onnx.yaml](../tensdataspecs_readonly/specs/formats/onnx.yaml). `version` on `-i` is `ir N`, the model's IR version.

The protobuf is walked directly, so no ONNX library or code generator is needed.

## Listing and extracting

```
$ tenspile -t --columns -f small-typed.onnx
name   dtype    shape   bytes  offset  params
t_f32  float32  [2, 2]   16 B       0       4
t_f16  float16  [3]       6 B       0       3
t_i8   int8     [3]       3 B       0       3
t_i32  int32    [3]      12 B       0       3
t_i64  int64    [3]      24 B       0       3
t_u32  uint32   [2]       8 B       0       2
```

An initializer's bytes come from one of three places:

- **`raw_data`:** the bytes are read where they lie, and `offset` is their position in the `.onnx` file.
- **A typed field** (`float_data`, `int32_data`, `int64_data`, and so on): the values are decoded into canonical bytes (ONNX-007, ONNX-008). They have no single position in the file, so `offset` is 0.
- **External data:** the bytes are in the file that `location` names, beside the model. A location outside the model directory is refused (ONNX-003).

Sparse initializers and tensors split into segments are refused. Two initializers with the same name are refused (ONNX-005).

```bash
tenspile -xf model.onnx -C out
tenspile -x -f model.onnx --format safetensors -o weights.safetensors
tensconv -o weights.safetensors model.onnx
```

## Writing a new file

`tenspile -c`, `tensconv`, and extract with `--format onnx` write a model that holds only the initializers, as onnx.yaml's writer does:

- a graph named `weights`, with one initializer per tensor in `raw_data` and one graph output per tensor;
- opset 21;
- the lowest IR version that has every dtype, and never below 10.

String metadata goes to `metadata_props`:

```
$ tenspile -cf model.onnx w=w_f32.npy --metadata producer=demo
$ tenspile -i model.onnx
format: onnx
version: ir 10
path: model.onnx
members: 1
data_bytes: 1024
metadata:
  producer=demo
dtypes:
  float32	1	256
```

tenspile does not write external data. A model of 2 GiB or more is refused (ONNX-006).

## Editing a model

`-r`, `-u`, and `--delete` rewrite only what they change (ONNX-009). Everything else in the model is copied:

- graph nodes, inputs, outputs, value info, functions, training info, quantization annotations, the producer, and fields tenspile does not know;
- `ir_version` and `opset_import`, as stored (ONNX-013);
- each initializer the edit does not touch, in its original encoding, including typed fields and external-data references.

A replaced or added initializer is written with `raw_data`. When its dtype is newer than the model's IR, `ir_version` is raised to the first IR that has it, and the edit says so (ONNX-014):

| dtype | lowest IR |
| --- | --- |
| bfloat16 | 4 |
| the four float8 types | 9 |
| int4, uint4 | 10 |
| float4_e2m1fn | 11 |
| float8_e8m0fnu | 12 |
| int2, uint2 | 13 |
| float6 types | 14 |
| every other dtype | 1, so an IR 3 model stays IR 3 |

onnx 1.23.2's checker does not test this yet, but newer checkers do, so the IR is raised anyway. `--metadata k=v` sets that key in `metadata_props`; the other keys stay.

When `-o` puts the model in another directory, each external-data file that a kept initializer names is copied beside it, so the reference still resolves (ONNX-003). A file of that name that is already there is used if it holds the same bytes. Otherwise the edit stops and leaves it alone, because other models may share it (ONNX-015):

```
$ tenspile -u -f ext.onnx -o moved/ext.onnx W=w_new.npy
tenspile: moved/ext.onnx: copied external data ext.data beside the new model (ONNX-015)
$ tenspile -u -f ext.onnx -o moved/ext2.onnx W=w_new.npy
tenspile: moved/ext2.onnx: uses the identical external data ext.data already beside the new model (ONNX-015)
```

Deleting an initializer also removes the graph input, graph output, and `value_info` entry with its name (ONNX-010, ONNX-012). The delete is refused when the model still uses the name (ONNX-011, ONNX-012):

- a node input, in the main graph or in a subgraph of an `If`, `Loop`, or `Scan` node, unless the subgraph defines the name itself;
- a node input in a training `initialization` or `algorithm` graph, which can read the main graph's initializers;
- a key of a training `initialization_binding` or `update_binding`;
- a quantization annotation, as the tensor or as one of its scale or zero-point tensors.

A model whose `If` node reads `S` only inside its `then_branch`:

```
$ tenspile --delete -f model.onnx S
tenspile: initializer S is an input of a graph node; deleting it would break the model (ONNX-011)
```

Edits were checked with onnx 1.23.2: every edited model passes `onnx.checker.check_model(full_check=True)`, and apart from the edit it is the same `ModelProto`.
