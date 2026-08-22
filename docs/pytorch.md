# PyTorch checkpoints

tenspile reads and writes the zip checkpoint that `torch.save` has written since PyTorch 1.6 (`.pt`, `.pth`, `.bin`, `.ckpt`). The layout, the pickle subset, and the validation ids are in [pytorch.yaml](../tensdataspecs_readonly/specs/formats/pytorch.yaml). `version` on `-i` is the text of the checkpoint's `version` record.

## Safe loading

The pickle inside the checkpoint is read by tenspile's own restricted unpickler. It handles only the opcodes listed in PT-007, and only the globals that rebuild tensors, storages, dtypes, and `OrderedDict` (PT-002). Any other global stops the read, and no code from the file runs. This is the same idea as `torch.load(weights_only=True)`, without Python.

Older formats are recognized and refused, not unpickled:

```
$ tenspile -tvf legacy.pt
tenspile: legacy.pt is a legacy PyTorch pickle or tar checkpoint and was not opened
```

That covers a bare pickle stream (detected by its 14-byte prefix) and the tar container (`ustar`). A ZIP whose `data.pkl` sits at the root is refused too (PT-001). A file name alone never makes a file a pickle: a `.bin` that holds plain tensor bytes can still be imported with `--dtype` and `--shape`.

## Listing and extracting

```
$ tenspile -tvf small.pt
name	dtype	shape	bytes	offset	params	shard	detail
w_f32	float32	[4, 64]	1024	1024	256	-	-
b_f16	float16	[8]	16	2112	8	-	-
e_bf16	bfloat16	[2, 3]	12	2240	6	-	-
i_i32	int32	[3, 2]	24	2368	6	-	-
s_f32	float32	[]	4	2496	1	-	-
```

dtypes are canonical ids. `offset` is where the tensor's bytes start in the zip, inside its `data/<key>` record.

- **Names:** a tensor's name is its path in the pickled object. A nested dict such as `{'state_dict': {'w': …}}` lists as `state_dict.w`.
- **Other values:** a string, number, or bool is shown as metadata. Anything else is skipped with a note.
- **Shared storage:** tensors that share storage, such as tied weights, are listed once each.
- **Byte order:** a big-endian checkpoint (`byteorder` record `big`) is byteswapped on the way in (PT-006).
- **Strided tensors:** a non-contiguous tensor is refused rather than gathered.

```bash
tenspile -xf small.pt -C out                                   # one .npy per tensor
tenspile -x -f small-state-dict.pt --format safetensors -o lin.safetensors lin.weight lin.bias
tensconv -o model.safetensors pytorch_model.bin
```

## Writing

`tenspile -c`, `tensconv`, and extract with `--format pytorch` write a flat state_dict: one top-level dict of name → tensor, pickled with protocol 2. This follows the writer in pytorch.yaml:

- Types with a legacy storage class (`FloatStorage`, `HalfStorage`, `BFloat16Storage`, and the rest) use `_rebuild_tensor_v2`.
- uint16, uint32, uint64, and the fp8 types use `_rebuild_tensor_v3` with an `UntypedStorage` and the dtype.
- Each `data/` record is stored uncompressed and 64-byte aligned.
- `.format_version` is left out, so torch reads each record's offset from its local header (PT-009).

A state_dict written here has no string metadata, so `--metadata` is refused:

```
$ tenspile -cf m2.pt w=w_f32.npy --metadata a=b
tenspile: a pytorch state_dict written here has no string metadata
```

## Editing

`-r`, `-u`, and `--delete` rewrite the checkpoint as a flat state_dict. That is only lossless when the file already is one. A checkpoint that holds nested dicts, or values that are not tensors (an `epoch`, optimizer state), is refused rather than flattened:

```
tenspile: ckpt.pt holds nested dicts or values that are not tensors, and tenspile writes a flat dict of tensors; a rewrite would drop them. Use tensconv to write a new state_dict on purpose
```

`tensconv -o weights.pt ckpt.pt` writes the tensors to a new flat state_dict, and warns about what it leaves behind.
