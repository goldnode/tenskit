//! rawtensor/1, the manifest-plus-bytes format in `formats/raw-sidecar.yaml`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use crate::dtype::{self, canonical_id, canonical_nbytes, canon_bits, canon_kind};
use crate::error::{io_err, msg, Result};
use crate::io_util::{self, atomic_write, relative_child};
use crate::payload::Payload;
use crate::sha256;

pub const FORMAT: &str = "rawtensor";
pub const VERSION: u64 = 1;
pub const ALIGNMENT: u64 = 64;
const MANIFEST_SUFFIX: &str = ".rawtensor.json";

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TensorRec {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub file: String,
    pub offset: u64,
    pub nbytes: u64,
    pub fortran: bool,
    pub payload: Payload,
    /// Entry fields other than the six that locate the bytes: `attrs`,
    /// `native`, `sha256`, and any this reader does not know. An edit keeps
    /// them on a tensor it leaves alone (RAW-011).
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RawFile {
    pub path: PathBuf,
    pub byte_order: String,
    pub alignment: u64,
    pub metadata: Map<String, Value>,
    /// Manifest fields other than the ones this reader uses: `source`,
    /// `history`, and any it does not know. An edit keeps them (RAW-010).
    pub extra: Map<String, Value>,
    pub tensors: Vec<TensorRec>,
    pub notes: Vec<String>,
}

/// The TensorEntry fields that locate a tensor's bytes; a writer sets them.
const ENTRY_FIELDS: [&str; 6] = ["name", "dtype", "shape", "file", "offset", "nbytes"];

/// The manifest fields a writer sets.
const MANIFEST_FIELDS: [&str; 6] = ["format", "version", "byte_order", "alignment", "tensors", "metadata"];

pub fn is_manifest_name(name: &str) -> bool {
    name.to_ascii_lowercase().ends_with(MANIFEST_SUFFIX)
}

pub fn data_file_name(manifest: &Path) -> Result<String> {
    let name = manifest
        .file_name()
        .and_then(|text| text.to_str())
        .ok_or_else(|| msg(format!("{} is not a rawtensor manifest name", manifest.display())))?;
    let stem = name
        .strip_suffix(MANIFEST_SUFFIX)
        .or_else(|| name.strip_suffix(".rawtensor.JSON"))
        .filter(|stem| !stem.is_empty())
        .ok_or_else(|| {
            msg(format!(
                "{} must be named <stem>.rawtensor.json",
                manifest.display()
            ))
        })?;
    Ok(format!("{stem}.bin"))
}

pub fn read_file(path: &Path) -> Result<RawFile> {
    let bytes = fs::read(path).map_err(|err| io_err(path, err))?;
    let value: Value = serde_json::from_slice(&bytes).map_err(|err| {
        msg(format!(
            "{} is not a rawtensor/1 manifest: {err}",
            path.display()
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        msg(format!(
            "{} is not a rawtensor/1 manifest",
            path.display()
        ))
    })?;
    let format = object.get("format").and_then(Value::as_str);
    let version = object.get("version").and_then(Value::as_u64);
    if format != Some(FORMAT) || version != Some(VERSION) {
        return Err(msg(format!(
            "{} is not rawtensor/1",
            path.display()
        )));
    }
    let byte_order = object
        .get("byte_order")
        .and_then(Value::as_str)
        .unwrap_or("little")
        .to_string();
    if byte_order != "little" && byte_order != "big" {
        return Err(msg(format!(
            "{}: byte_order {byte_order} is not little or big",
            path.display()
        )));
    }
    let alignment = object
        .get("alignment")
        .and_then(Value::as_u64)
        .unwrap_or(ALIGNMENT);
    let metadata = match object.get("metadata") {
        Some(Value::Object(map)) => map.clone(),
        Some(_) => {
            return Err(msg(format!(
                "{}: metadata is not an object",
                path.display()
            )))
        }
        None => Map::new(),
    };
    let entries = object
        .get("tensors")
        .and_then(Value::as_array)
        .ok_or_else(|| msg(format!("{}: tensors is not an array", path.display())))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut names = BTreeSet::new();
    let mut tensors = Vec::with_capacity(entries.len());
    let mut notes = Vec::new();
    let mut ranges: Vec<(String, u64, u64)> = Vec::new();
    for entry in entries {
        let tensor = entry.as_object().ok_or_else(|| {
            msg(format!("{}: a tensor entry is not an object", path.display()))
        })?;
        let name = tensor
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| msg(format!("{}: a tensor has no name", path.display())))?
            .to_string();
        if !names.insert(name.clone()) {
            return Err(msg(format!(
                "{}: duplicate tensor name {name}",
                path.display()
            )));
        }
        let dtype = tensor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| msg(format!("{}: tensor {name} has no dtype", path.display())))?;
        if dtype::canonical_id(dtype).is_err() && canon_kind(dtype).is_none() {
            return Err(msg(format!(
                "{}: tensor {name} dtype {dtype} is not a canonical id",
                path.display()
            )));
        }
        if canon_kind(dtype).is_none() {
            return Err(msg(format!(
                "{}: tensor {name} dtype {dtype} is not a canonical id",
                path.display()
            )));
        }
        let shape = json_shape(path, &name, tensor.get("shape"))?;
        let file_name = tensor
            .get("file")
            .and_then(Value::as_str)
            .ok_or_else(|| msg(format!("{}: tensor {name} has no file", path.display())))?;
        let data_path = relative_child(base, file_name)?;
        let offset = json_u64(tensor.get("offset")).ok_or_else(|| {
            msg(format!(
                "{}: tensor {name} offset is not an integer",
                path.display()
            ))
        })?;
        let nbytes = json_u64(tensor.get("nbytes")).ok_or_else(|| {
            msg(format!(
                "{}: tensor {name} nbytes is not an integer",
                path.display()
            ))
        })?;
        let expected = canonical_nbytes(dtype, &shape)?;
        if nbytes != expected {
            return Err(msg(format!(
                "{}: tensor {name} nbytes is {nbytes} and dtype {dtype} needs {expected}",
                path.display()
            )));
        }
        let file_len = fs::metadata(&data_path)
            .map_err(|err| io_err(&data_path, err))?
            .len();
        if offset.checked_add(nbytes).map(|end| end > file_len).unwrap_or(true) {
            return Err(msg(format!(
                "{}: tensor {name} extends past {}",
                path.display(),
                data_path.display()
            )));
        }
        if alignment > 1 && offset % alignment != 0 {
            notes.push(format!(
                "tensor {name} offset {offset} is not aligned to {alignment}"
            ));
        }
        // RAW-009: the bytes are C order and `native` is informational only.
        let fortran = false;
        if tensor
            .get("native")
            .and_then(|native| native.get("fortran_order"))
            .and_then(Value::as_bool)
            == Some(true)
        {
            notes.push(format!(
                "tensor {name} has native.fortran_order; rawtensor bytes are read as C order (RAW-009)"
            ));
        }
        let mut payload = Payload::Range {
            path: data_path,
            offset,
            len: nbytes,
        };
        if byte_order == "big" {
            payload = byteswap_if_needed(path, &name, dtype, payload)?;
            if canon_kind(dtype) == Some("block") {
                notes.push(format!(
                    "tensor {name} is a big-endian block dtype; block field order is an open question and the bytes were not swapped"
                ));
            }
        }
        if let Some(digest) = tensor.get("sha256").and_then(Value::as_str) {
            let bytes = payload.read_vec()?;
            let actual = sha256::hex(&bytes);
            if !actual.eq_ignore_ascii_case(digest) {
                return Err(msg(format!(
                    "{}: tensor {name} sha256 does not match",
                    path.display()
                )));
            }
        }
        ranges.push((file_name.to_string(), offset, nbytes));
        let extra = tensor
            .iter()
            .filter(|(key, _)| !ENTRY_FIELDS.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        tensors.push(TensorRec {
            name,
            dtype: dtype.to_string(),
            shape,
            file: file_name.to_string(),
            offset,
            nbytes,
            fortran,
            payload,
            extra,
        });
    }
    note_overlaps(&mut notes, &ranges);
    Ok(RawFile {
        path: path.to_path_buf(),
        byte_order,
        alignment,
        metadata,
        extra: object
            .iter()
            .filter(|(key, _)| !MANIFEST_FIELDS.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        tensors,
        notes,
    })
}

pub fn display_metadata(metadata: &Map<String, Value>) -> Vec<(String, String)> {
    metadata
        .iter()
        .map(|(key, value)| (key.clone(), display_typed(value)))
        .collect()
}

fn display_typed(value: &Value) -> String {
    let Some(object) = value.as_object() else {
        return value.to_string();
    };
    if object.get("type").and_then(Value::as_str) == Some("string") {
        if let Some(text) = object.get("value").and_then(Value::as_str) {
            return text.to_string();
        }
    }
    value.to_string()
}

pub fn string_metadata(pairs: &[(String, String)]) -> Result<Map<String, Value>> {
    let mut map = Map::new();
    for (key, value) in pairs {
        if key.starts_with("rawtensor.") {
            return Err(msg(format!("metadata key {key} is reserved")));
        }
        map.insert(
            key.clone(),
            json!({"type": "string", "value": value}),
        );
    }
    Ok(map)
}

pub fn merge_metadata(
    base: &Map<String, Value>,
    updates: &[(String, String)],
) -> Result<Map<String, Value>> {
    let mut merged = base.clone();
    let incoming = string_metadata(updates)?;
    for (key, value) in incoming {
        merged.insert(key, value);
    }
    Ok(merged)
}

#[derive(Clone)]
pub struct OutTensor {
    pub name: String,
    pub canonical: String,
    pub shape: Vec<u64>,
    pub payload: Payload,
    pub fortran: bool,
    /// Entry fields written as given, such as `attrs`, `native`, or `sha256`.
    pub extra: Map<String, Value>,
}

pub fn write_archive(
    manifest: &Path,
    metadata: &Map<String, Value>,
    tensors: &[OutTensor],
) -> Result<()> {
    write_archive_with(manifest, metadata, &Map::new(), tensors)
}

/// `write_archive`, plus manifest fields such as `source` and `history`.
pub fn write_archive_with(
    manifest: &Path,
    metadata: &Map<String, Value>,
    extra: &Map<String, Value>,
    tensors: &[OutTensor],
) -> Result<()> {
    if !is_manifest_name(
        manifest
            .file_name()
            .and_then(|text| text.to_str())
            .unwrap_or(""),
    ) {
        return Err(msg(format!(
            "{} must be named <stem>.rawtensor.json",
            manifest.display()
        )));
    }
    let data_path = data_path_for(manifest)?;
    let data_name = data_file_name(manifest)?;
    io_util::check_unique_names(tensors.iter().map(|tensor| tensor.name.as_str()))?;
    if let Some(key) = metadata.keys().find(|key| key.starts_with("rawtensor.")) {
        return Err(msg(format!("metadata key {key} is reserved")));
    }
    let mut entries = Vec::with_capacity(tensors.len());
    let mut offset = 0u64;
    for tensor in tensors {
        if tensor.fortran {
            return Err(msg(format!(
                "tensor {} is fortran-order; rawtensor bytes are C order (RAW-009) and tenskit does not transpose",
                tensor.name
            )));
        }
        offset += (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;
        let nbytes = tensor.payload.len();
        let expected = canonical_nbytes(&tensor.canonical, &tensor.shape)?;
        if nbytes != expected {
            return Err(msg(format!(
                "tensor {} is {nbytes} bytes and dtype {} needs {expected}",
                tensor.name, tensor.canonical
            )));
        }
        let mut entry = json!({
            "name": tensor.name,
            "dtype": tensor.canonical,
            "shape": tensor.shape,
            "file": data_name,
            "offset": offset,
            "nbytes": nbytes,
        });
        if let Value::Object(fields) = &mut entry {
            for (key, value) in &tensor.extra {
                if !ENTRY_FIELDS.contains(&key.as_str()) {
                    fields.insert(key.clone(), value.clone());
                }
            }
        }
        entries.push(entry);
        offset += nbytes;
    }
    let mut document = Map::new();
    document.insert("format".into(), json!(FORMAT));
    document.insert("version".into(), json!(VERSION));
    document.insert("byte_order".into(), json!("little"));
    document.insert("alignment".into(), json!(ALIGNMENT));
    document.insert("tensors".into(), Value::Array(entries));
    if !metadata.is_empty() {
        document.insert("metadata".into(), Value::Object(metadata.clone()));
    }
    for (key, value) in extra {
        if !MANIFEST_FIELDS.contains(&key.as_str()) {
            document.insert(key.clone(), value.clone());
        }
    }
    let mut body = serde_json::to_vec_pretty(&Value::Object(document))?;
    body.push(b'\n');
    // The data file is streamed to a temp name and renamed before the
    // manifest, so the manifest never names bytes that were not written.
    atomic_write(&data_path, |file| {
        use std::io::Write;
        let mut written = 0u64;
        for tensor in tensors {
            let pad = (ALIGNMENT - (written % ALIGNMENT)) % ALIGNMENT;
            file.write_all(&vec![0u8; pad as usize])?;
            written += pad;
            tensor.payload.copy_to(file)?;
            written += tensor.payload.len();
        }
        Ok(())
    })?;
    atomic_write(manifest, |file| {
        use std::io::Write;
        file.write_all(&body)?;
        Ok(())
    })
}

/// The data file written beside `<stem>.rawtensor.json`.
pub fn data_path_for(manifest: &Path) -> Result<PathBuf> {
    Ok(manifest
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(data_file_name(manifest)?))
}

pub fn to_out(
    name: &str,
    dtype: &str,
    shape: &[u64],
    payload: &Payload,
    fortran: bool,
) -> Result<OutTensor> {
    let canonical = canonical_id(dtype)
        .map_err(|err| msg(format!("member {name}: {err}")))?
        .to_string();
    Ok(OutTensor {
        name: name.to_string(),
        canonical,
        shape: shape.to_vec(),
        payload: payload.clone(),
        fortran,
        extra: Map::new(),
    })
}

/// Big-endian data to canonical little-endian bytes, per swap unit
/// (SCHEMA.md section 7: complex types swap each part).
fn byteswap_if_needed(path: &Path, name: &str, dtype: &str, payload: Payload) -> Result<Payload> {
    let kind = canon_kind(dtype).unwrap_or("");
    let bits = canon_bits(dtype).unwrap_or(0);
    if kind == "block" || bits <= 8 {
        return Ok(payload);
    }
    let mut bytes = payload.read_vec()?;
    crate::dtype::byteswap(&mut bytes, dtype)
        .map_err(|err| msg(format!("{}: tensor {name}: {err}", path.display())))?;
    Ok(Payload::Bytes(bytes))
}

fn note_overlaps(notes: &mut Vec<String>, ranges: &[(String, u64, u64)]) {
    let mut grouped: BTreeSet<&str> = BTreeSet::new();
    for (file, _, _) in ranges {
        grouped.insert(file);
    }
    for file in grouped {
        let mut spans: Vec<(u64, u64)> = ranges
            .iter()
            .filter(|(name, _, _)| name == file)
            .map(|(_, offset, len)| (*offset, offset + len))
            .collect();
        spans.sort_unstable();
        let mut cursor = 0u64;
        for (begin, end) in spans {
            if begin < cursor {
                notes.push(format!("byte ranges in {file} overlap"));
                break;
            }
            cursor = end;
        }
    }
}

fn json_shape(path: &Path, name: &str, value: Option<&Value>) -> Result<Vec<u64>> {
    let array = value.and_then(Value::as_array).ok_or_else(|| {
        msg(format!(
            "{}: tensor {name} shape is not an array",
            path.display()
        ))
    })?;
    array
        .iter()
        .map(|dim| {
            json_u64(Some(dim)).ok_or_else(|| {
                msg(format!(
                    "{}: tensor {name} has a non-integer dimension",
                    path.display()
                ))
            })
        })
        .collect()
}

fn json_u64(value: Option<&Value>) -> Option<u64> {
    let value = value?;
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|number| u64::try_from(number).ok()))
}
