# PyTorch checkpoints

tenspile reads and writes the zip checkpoint that `torch.save` has written since PyTorch 1.6 (`.pt`, `.pth`, `.bin`, `.ckpt`). The layout, the pickle subset, and the validation ids are in `pytorch.yaml`. `version` on `-i` is the text of the checkpoint's `version` record.

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
w_f32	F32	[4, 64]	1024	1024	256	-	-
b_f16	F16	[8]	16	2112	8	-	-
e_bf16	BF16	[2, 3]	12	2240	6	-	-
i_i32	I32	[3, 2]	24	2368	6	-	-
s_f32	F32	[]	4	2496	1	-	-
```

A dtype is listed by its safetensors name, such as `F16`, when it has one, and otherwise by its canonical id. `offset` is where the tensor's bytes start in the zip, inside its `data/<key>` record.

- **Names:** a tensor's name is its path in the pickled object, joined with `.` (PT-012). A string key is used as it is, and an integer key or a list index is its number. So `{'state_dict': {'w': …}}` lists as `state_dict.w`, and an Adam optimizer's state lists as `optimizer.state.0.exp_avg`.
- **Unnamed tensors:** a tensor reached through any other kind of key, such as `None` or a tuple, has no name. A note gives its path, dtype, and shape, for example `the tensor at root[None] (float32 [2]) has no name`. It is not listed or extracted, and an edit keeps it as stored.
- **Other values:** a string, number, or bool is shown as metadata under its path, such as `optimizer.param_groups.0.lr=0.001`. `None` is left out. Anything else is skipped with a note.
- **Shared storage:** tensors that share storage, such as tied weights, are listed once each.
- **Byte order:** a big-endian checkpoint (`byteorder` record `big`) is byteswapped on the way in (PT-006).
- **Views:** a tensor that is a strided view of its storage, such as one row or every second element, is gathered into C order. Its `offset` is 0, because its bytes are not in one place.
- **Conj and neg bits:** a view saved with `{'conj': True}` (from `c.conj()`) has its imaginary parts negated, and one saved with `{'neg': True}` (from `c.conj().imag`) has every element negated, as `torch.load` does (PT-010). Floats flip their sign bit. int8 to int64 and uint8 negate with wraparound, so int8 -128 stays -128. A neg bit on bool, uint16 to uint64, or an fp8 type is refused, because torch cannot resolve it either. Without this step the values would be wrong with no error.

`small-pt-fp8.pt` in the spec's test vectors has all of these: two fp8 tensors, a conj view, a neg view, and three names on one storage.

```
$ tenspile -tvf small-pt-fp8.pt
name	dtype	shape	bytes	offset	params	shard	detail
f8e4m3	F8_E4M3	[2, 3]	6	1216	6	-	-
f8e5m2	F8_E5M2	[4]	4	1344	4	-	-
c64_conj	C64	[2]	16	0	2	-	-
imag_neg	F32	[2]	8	0	2	-	-
tied_a	F32	[2, 2]	16	1728	4	-	-
tied_b	F32	[2, 2]	16	1728	4	-	-
row1	F32	[2]	8	1736	2	-	-
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

A new name goes into the deepest dict whose name, followed by `.`, starts it (PT-013). The rest of the name is one key, dots included: `state_dict.extra` lands inside `state_dict` as `extra`, and `state_dict.layer.9.w` as `layer.9.w`. Otherwise it is a new top-level key. No new nested dict is made, nothing is added to a list or tuple, and `_metadata` is left alone, because `load_state_dict` needs no entry for a new key. A checkpoint whose root is a list has no dict to add to, so adding is refused. Deleting a tensor inside a list or tuple is refused too, even the last item, because the items after it would change names and code may index the list by position. Replacing it in place is allowed. Deleting a dict key removes only that key, and a dict left empty stays. A replaced tensor is written with `_rebuild_tensor_v2`, or `_rebuild_tensor_v3` for the dtypes listed above.

Deleting `f8e5m2` keeps the shared storage and the views:

```
$ tenspile --delete -f small-pt-fp8.pt -o fp8b.pt f8e5m2
$ tensdiff small-pt-fp8.pt fp8b.pt
only in small-pt-fp8.pt: f8e5m2
```

The edit prints what it re-encodes: a compressed storage is stored uncompressed (PT-005). `.format_version` is always left out (PT-009) without a message. torch uses it only to decide how to find record offsets, so no loaded value changes.

A checkpoint that holds something tenspile cannot write back exactly is refused rather than changed: a set, a dtype or storage class stored as a value, a tensor with backward hooks or extra state, or the old `_rebuild_tensor` call. So is a file where two paths give one name, such as `{'a.b': t1, 'a': {'b': t2}}` (PT-012), and an edit that would create such a name.

A training checkpoint with its optimizer keeps everything else when one weight is replaced:

```
$ tenspile -u -f ckpt.pt model.0.bias=new_bias.npy
$ python3 -c "import torch; c = torch.load('ckpt.pt', weights_only=True); print(list(c), c['optimizer']['param_groups'][0]['lr'])"
['model', 'optimizer', 'epoch', 'best', 'note'] 0.001
```

```
tenspile: ckpt.pt holds a set, which tenspile cannot write back (PT-011); use tensconv to write a new state_dict on purpose
```

`tensconv -o weights.pt ckpt.pt` writes the tensors to a new flat state_dict, and warns about what it leaves behind.
