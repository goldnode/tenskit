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
- **Views:** a tensor that is a strided view of its storage, such as one row or every second element, is gathered into C order. Its `offset` is 0, because its bytes are not in one place.
- **Conj and neg bits:** a view saved with `{'conj': True}` (from `c.conj()`) has its imaginary parts negated, and one saved with `{'neg': True}` (from `c.conj().imag`) has every element negated, as `torch.load` does (PT-010). Without this the values would be wrong with no error.

`small-pt-fp8.pt` in the spec's test vectors has all of these: two fp8 tensors, a conj view, a neg view, and three names on one storage.

```
$ tenspile -tvf small-pt-fp8.pt
name	dtype	shape	bytes	offset	params	shard	detail
f8e4m3	float8_e4m3fn	[2, 3]	6	1216	6	-	-
f8e5m2	float8_e5m2	[4]	4	1344	4	-	-
c64_conj	complex64	[2]	16	0	2	-	-
imag_neg	float32	[2]	8	0	2	-	-
tied_a	float32	[2, 2]	16	1728	4	-	-
tied_b	float32	[2, 2]	16	1728	4	-	-
row1	float32	[2]	8	1736	2	-	-
```

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

`-r`, `-u`, and `--delete` write the pickled object back as it was read (PT-011). Only the tensors the edit names change:

- dicts and `OrderedDict`s, key order, and an `OrderedDict`'s `_metadata`, so a `model.state_dict()` keeps its version records;
- nested dicts such as `{'state_dict': …, 'epoch': 3}`, and ints, floats, strings, bools, `None`, lists, and tuples;
- each tensor the edit leaves alone, with its storage, offset, strides, conj and neg bits, and `nn.Parameter` wrapper;
- one `data/` record per storage, so tied weights stay tied. Replacing one name of a tied pair gives that name a storage of its own.

A new name goes into the deepest dict its dotted path names, so `state_dict.extra` lands inside `state_dict`. Otherwise it is a new top-level key. A replaced tensor is written with `_rebuild_tensor_v2`, or `_rebuild_tensor_v3` for the dtypes listed above.

Deleting `f8e5m2` keeps the shared storage and the views:

```
$ tenspile --delete -f small-pt-fp8.pt -o fp8b.pt f8e5m2
tenspile: fp8b.pt: left out .format_version, so torch reads each record's offset from its header (PT-009)
$ tensdiff small-pt-fp8.pt fp8b.pt
only in small-pt-fp8.pt: f8e5m2
```

The edit prints what it re-encoded: `.format_version` is always left out (PT-009), and a compressed storage is stored uncompressed (PT-005).

A checkpoint that holds something tenspile cannot write back exactly is refused rather than changed: a set, a dtype or storage class stored as a value, a tensor with backward hooks or extra state, the old `_rebuild_tensor` call, or two entries that list under the same name.

```
tenspile: ckpt.pt holds a set, which tenspile cannot write back (PT-011); use tensconv to write a new state_dict on purpose
```

`tensconv -o weights.pt ckpt.pt` writes the tensors to a new flat state_dict, and warns about what it leaves behind.
