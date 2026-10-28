# rawtensor/1

The stored raw format is rawtensor/1, defined by [raw-sidecar.yaml](../tensdataspecs_readonly/specs/formats/raw-sidecar.yaml). A `.rawtensor.json` manifest names every tensor. A sibling `<stem>.bin` holds the bytes in canonical layout: little-endian, C-order, packed. `version` on `-i` is `1`. `--format raw` selects this format.

Headerless bytes are not an archive. `--dtype` and `--shape` exist only so a headerless file can be imported into a container. The names are safetensors names or canonical ids from [dtypes.yaml](../tensdataspecs_readonly/specs/mappings/dtypes.yaml).

## Create, list, and edit

```
tenspile -c -f embed.rawtensor.json --dtype F32 --shape 2 embed.bin
tenspile -i -f embed.rawtensor.json
tenspile -tvf embed.rawtensor.json
```

The manifest records `dtype: float32`. The bytes are in `embed.bin`, aligned to 64. Several tensors share that one `.bin`, with zero padding between them. Append, update, and delete rewrite the manifest and the `.bin`:

- `source` is kept as it was, because it records where the data came from (RAW-010). So are manifest fields tenspile does not know.
- Each edit appends `{"converter": "tenspile 0.1.0", "action": "replaced w; deleted b"}` to the `history` array (RAW-010).
- A tensor the edit leaves alone keeps `attrs`, `native`, `sha256`, and any other field of its entry. A replaced or new tensor has none of them (RAW-011).
- When every tensor in the input had `sha256`, a replaced or new tensor gets one too, so the whole file can still be checked (RAW-012). String metadata is stored as a typed string value. Keys that start with `rawtensor.` are reserved.

```
tenspile -r -f embed.rawtensor.json --dtype F32 --shape 2 norm=norm.bin
tenspile -u -f embed.rawtensor.json --metadata source=demo
tenspile --delete -f embed.rawtensor.json norm
```

A manifest that is not named `<stem>.rawtensor.json` is refused on write. On read, a JSON file whose `format` is `rawtensor` and whose `version` is `1` is accepted. Paths in the manifest have to stay inside the manifest directory. `nbytes` has to match the canonical size. A `sha256` field, when present, has to match the bytes. A big-endian manifest is byteswapped for element types wider than one byte. Block dtypes in a big-endian manifest are not swapped; `-i` says that field order is an open question.

## Extracting to rawtensor

```
tenspile -x -f out.safetensors --format raw -o embed.rawtensor.json embed
tenspile -x -f model.gguf --format raw -o weight.rawtensor.json weight
```

One `--format raw` command writes one manifest. A GGUF quantized tensor is stored as its canonical block dtype and block bytes, without the GGUF alignment padding. A Fortran-order array is refused, because rawtensor bytes are C order and tenskit does not transpose (RAW-009).
