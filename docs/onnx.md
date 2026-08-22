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

`-r`, `-u`, and `--delete` rewrite only what they change. Everything else in the model is copied:

- graph nodes, inputs, outputs, value info, opsets, the producer, and other fields;
- each initializer the edit does not touch, in its original encoding, including typed fields and external-data references.

A replaced or added initializer is written with `raw_data`. A metadata update replaces `metadata_props`. Deleting an initializer also removes a graph output of the same name. Deleting one that a node still reads is refused, because the model would no longer run:

```
tenspile: initializer w is an input of a graph node; deleting it would break the model
```
