use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::Path;

use serde_json::Value;

use crate::dtype::{ggml_type_name, numel, require_ggml};
use crate::error::{io_err, msg, Result};
use crate::io_util::{atomic_write, copy_range};
use crate::payload::Payload;

pub const MAGIC: &[u8] = b"GGUF";
/// Current writer version. Little-endian version 3 is what GGUF writers emit.
pub const VERSION: u32 = 3;
/// Oldest little-endian version this reader accepts. Version 2 uses the same field widths.
const MIN_VERSION: u32 = 2;
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// GGUF-022: a reader may accept up to 8 dims and 1024-byte names, with a
/// warning. ggml itself stops at 4 dims and 63 bytes (GGUF-009, GGUF-010).
const MAX_DIMS: u32 = 8;
const MAX_NAME: u64 = 1024;
const GGML_MAX_DIMS: usize = 4;
const GGML_MAX_NAME: usize = 63;
/// Smallest KV entry (key length, type, one byte) and tensor info (name
/// length, n_dims, type, offset), for the GGUF-019 count bounds.
const MIN_KV: u64 = 12;
const MIN_TENSOR_INFO: u64 = 24;

const TYPE_U8: u32 = 0;
const TYPE_I8: u32 = 1;
const TYPE_U16: u32 = 2;
const TYPE_I16: u32 = 3;
const TYPE_U32: u32 = 4;
pub(crate) const TYPE_I32: u32 = 5;
const TYPE_F32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_U64: u32 = 10;
/// Element type of a `.orig_shape` array (GGUF-023).
pub const TYPE_ARRAY_U64: u32 = TYPE_U64;
const TYPE_I64: u32 = 11;
const TYPE_F64: u32 = 12;

#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub ggml_type: u32,
    /// ggml `ne[]`, innermost first.
    pub shape: Vec<u64>,
    /// Canonical shape from `<name>.orig_shape` (GGUF-023), when present
    /// and its element count matches.
    pub orig_shape: Option<Vec<u64>>,
    /// Offset within the tensor blob.
    pub offset: u64,
    /// Bytes from this offset to the next tensor, including alignment padding.
    pub span: u64,
    /// Bytes the type and shape need, without padding.
    pub nbytes: u64,
    /// File offset of the first stored byte.
    pub file_offset: u64,
}

#[derive(Debug, Clone)]
pub struct MetaField {
    pub key: String,
    pub value: MetaValue,
    /// File offset of the whole entry: key, type, and value.
    pub entry_offset: u64,
    pub entry_len: u64,
}

#[derive(Debug, Clone)]
pub struct GgufFile {
    pub path: std::path::PathBuf,
    pub version: u32,
    pub alignment: u64,
    pub data_start: u64,
    pub data_len: u64,
    pub metadata: Vec<MetaField>,
    pub tensors: Vec<GgufTensor>,
    pub warnings: Vec<String>,
}

struct In {
    file: File,
    pos: u64,
    len: u64,
}

impl In {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|err| io_err(path, err))?;
        let len = file.metadata().map_err(|err| io_err(path, err))?.len();
        Ok(Self { file, pos: 0, len })
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        self.file.read_exact(buf)?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn remaining(&self) -> u64 {
        self.len.saturating_sub(self.pos)
    }

    /// GGUF-003: a length is bounded by the bytes left in the file.
    fn read_string(&mut self, limit: u64) -> Result<String> {
        let len = self.read_u64()?;
        if len > limit {
            return Err(msg(format!("gguf string is {len} bytes, over the {limit} byte limit")));
        }
        if len > self.remaining() {
            return Err(msg(format!(
                "gguf string of {len} bytes runs past the end of the file (GGUF-003)"
            )));
        }
        let mut buf = vec![0u8; len as usize];
        self.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|_| msg("gguf string is not utf-8"))
    }
}

pub fn read_file(path: &Path) -> Result<GgufFile> {
    let mut input = In::open(path)?;
    let mut magic = [0u8; 4];
    input.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(msg(format!("{} is not a GGUF file", path.display())));
    }
    let version = input.read_u32()?;
    if !(MIN_VERSION..=VERSION).contains(&version) {
        return Err(unsupported_version(path, version));
    }
    let tensor_count = input.read_u64()?;
    let kv_count = input.read_u64()?;
    // GGUF-019: each count is bounded by the smallest record that fits.
    if kv_count > input.remaining() / MIN_KV
        || tensor_count > input.remaining() / MIN_TENSOR_INFO
    {
        return Err(msg(format!(
            "{}: gguf counts ({tensor_count} tensors, {kv_count} keys) do not fit in the file (GGUF-003)",
            path.display()
        )));
    }
    let mut warnings = Vec::new();
    let mut metadata: Vec<MetaField> = Vec::with_capacity(kv_count as usize);
    let mut keys = std::collections::HashSet::new();
    for _ in 0..kv_count {
        let entry_offset = input.pos;
        let key = input.read_string(65_535)?;
        if !keys.insert(key.clone()) {
            return Err(msg(format!("gguf metadata key {key} appears twice (GGUF-006)")));
        }
        let value_type = input.read_u32()?;
        let value = read_value(&mut input, value_type, true)?;
        let entry_len = input.pos - entry_offset;
        metadata.push(MetaField {
            key,
            value,
            entry_offset,
            entry_len,
        });
    }
    let alignment = alignment_of(&metadata, &mut warnings)?;
    let mut tensors_raw = Vec::with_capacity(tensor_count as usize);
    let mut names = std::collections::HashSet::new();
    for _ in 0..tensor_count {
        let name = input.read_string(MAX_NAME)?;
        if !names.insert(name.clone()) {
            return Err(msg(format!("gguf tensor {name} appears twice (GGUF-006)")));
        }
        if name.len() > GGML_MAX_NAME {
            warnings.push(format!(
                "tensor {name} has a name of {} bytes; ggml loads names of at most 63 (GGUF-022)",
                name.len()
            ));
        }
        let n_dims = input.read_u32()?;
        if n_dims > MAX_DIMS {
            return Err(msg(format!("gguf tensor {name} has {n_dims} dimensions")));
        }
        if n_dims as usize > GGML_MAX_DIMS {
            warnings.push(format!(
                "tensor {name} has {n_dims} dimensions; ggml loads at most 4 (GGUF-022)"
            ));
        }
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(input.read_u64()?);
        }
        if numel(&shape).is_none_or(|count| count > i64::MAX as u64) {
            return Err(msg(format!("gguf tensor {name} has too many elements (GGUF-010)")));
        }
        let ggml_type = input.read_u32()?;
        let row = require_ggml(ggml_type).map_err(|err| {
            msg(format!("{}: tensor {name}: {err}", path.display()))
        })?;
        let block = u64::from(row.block_elements.max(1));
        if shape.first().is_some_and(|ne0| ne0 % block != 0) {
            return Err(msg(format!(
                "gguf tensor {name} ne[0] is not a multiple of the {} block of {block} (GGUF-012)",
                ggml_type_name(ggml_type)
            )));
        }
        let offset = input.read_u64()?;
        if offset % alignment != 0 {
            return Err(msg(format!(
                "gguf tensor {name} offset {offset} is not aligned to {alignment} (GGUF-013)"
            )));
        }
        tensors_raw.push((name, ggml_type, shape, offset));
    }
    // GGUF-021: ggml seeks to the aligned data section only when there are
    // tensors, so a file without tensors may end right after its last KV.
    let data_start = if tensor_count == 0 {
        input.pos.min(input.len)
    } else {
        align_up(input.pos, alignment)
    };
    if data_start > input.len {
        return Err(msg("gguf tensor data starts past the end of the file"));
    }
    let data_len = input.len - data_start;
    let mut tensors = spans(tensors_raw, data_start, data_len)?;
    // GGUF-023: `<name>.orig_shape` restores the canonical shape and is not
    // reported as model metadata.
    let mut hidden = std::collections::HashSet::new();
    for tensor in &mut tensors {
        let key = format!("{}.orig_shape", tensor.name);
        let Some(field) = metadata.iter().find(|field| field.key == key) else {
            continue;
        };
        let orig = match &field.value {
            MetaValue::Array { elem: TYPE_U64, values } => values
                .iter()
                .map(|value| match value {
                    MetaValue::U64(dim) => Some(*dim),
                    _ => None,
                })
                .collect::<Option<Vec<u64>>>(),
            _ => None,
        };
        match orig {
            Some(orig) if numel(&orig) == numel(&tensor.shape) => {
                tensor.orig_shape = Some(orig);
                hidden.insert(key);
            }
            _ => warnings.push(format!(
                "{key} is not an ARRAY of UINT64 with the stored element count; the stored shape is kept (GGUF-023)"
            )),
        }
    }
    metadata.retain(|field| !hidden.contains(&field.key));
    Ok(GgufFile {
        path: path.to_path_buf(),
        version,
        alignment,
        data_start,
        data_len,
        metadata,
        tensors,
        warnings,
    })
}

/// GGUF-008: `general.alignment`, when present, is a UINT32 non-zero power
/// of two. GGUF-020: a multiple of 8 is expected; 1, 2, and 4 are read with
/// a warning.
fn alignment_of(metadata: &[MetaField], warnings: &mut Vec<String>) -> Result<u64> {
    let Some(field) = metadata.iter().find(|field| field.key == "general.alignment") else {
        return Ok(DEFAULT_ALIGNMENT);
    };
    let MetaValue::U32(value) = field.value else {
        return Err(msg(format!(
            "general.alignment is {}, not uint32 (GGUF-008)",
            gguf_type_name(field.value.type_id())
        )));
    };
    if value == 0 || !value.is_power_of_two() {
        return Err(msg(format!(
            "general.alignment {value} is not a non-zero power of two (GGUF-008)"
        )));
    }
    if value % 8 != 0 {
        warnings.push(format!(
            "general.alignment {value} is not a multiple of 8 (GGUF-020)"
        ));
    }
    Ok(u64::from(value))
}

fn spans(
    mut raw: Vec<(String, u32, Vec<u64>, u64)>,
    data_start: u64,
    data_len: u64,
) -> Result<Vec<GgufTensor>> {
    let mut order: Vec<usize> = (0..raw.len()).collect();
    order.sort_by_key(|&index| raw[index].3);
    let mut span_of = vec![0u64; raw.len()];
    for (rank, &index) in order.iter().enumerate() {
        let offset = raw[index].3;
        if offset > data_len {
            return Err(msg(format!(
                "gguf tensor {} offset {offset} is past the tensor blob; the file may be truncated or damaged",
                raw[index].0
            )));
        }
        let next = order.get(rank + 1).map(|&next| raw[next].3);
        let end = next.unwrap_or(data_len);
        if end < offset {
            return Err(msg(format!(
                "gguf tensor {} overlaps the next tensor",
                raw[index].0
            )));
        }
        span_of[index] = end - offset;
    }
    raw.drain(..)
        .enumerate()
        .map(|(index, (name, ggml_type, shape, offset))| {
            let nbytes = crate::dtype::ggml_payload_bytes(ggml_type, &shape)?;
            // GGUF-015: the tensor's bytes lie before the next tensor and the
            // end of the file.
            if nbytes > span_of[index] {
                return Err(msg(format!(
                    "gguf tensor {name} needs {nbytes} bytes and has {} before the next tensor or the end of the file (GGUF-015); the file may be truncated or damaged",
                    span_of[index]
                )));
            }
            Ok(GgufTensor {
                name,
                ggml_type,
                shape,
                orig_shape: None,
                offset,
                span: span_of[index],
                nbytes,
                file_offset: data_start + offset,
            })
        })
        .collect()
}

pub fn align_up(offset: u64, alignment: u64) -> u64 {
    let alignment = alignment.max(1);
    offset.saturating_add(alignment - 1) / alignment * alignment
}

/// Read one value. `top` is false inside an array: ggml rejects nested
/// arrays (GGUF-005), so an ARRAY element type is refused.
fn read_value(input: &mut In, value_type: u32, top: bool) -> Result<MetaValue> {
    Ok(match value_type {
        TYPE_U8 => MetaValue::U8(input.read_u8()?),
        TYPE_I8 => MetaValue::I8(input.read_u8()? as i8),
        TYPE_U16 => {
            let mut buf = [0u8; 2];
            input.read_exact(&mut buf)?;
            MetaValue::U16(u16::from_le_bytes(buf))
        }
        TYPE_I16 => {
            let mut buf = [0u8; 2];
            input.read_exact(&mut buf)?;
            MetaValue::I16(i16::from_le_bytes(buf))
        }
        TYPE_U32 => MetaValue::U32(input.read_u32()?),
        TYPE_I32 => MetaValue::I32(input.read_u32()? as i32),
        TYPE_F32 => MetaValue::F32(f32::from_bits(input.read_u32()?)),
        TYPE_BOOL => match input.read_u8()? {
            0 => MetaValue::Bool(false),
            1 => MetaValue::Bool(true),
            value => return Err(msg(format!("gguf bool value {value} is invalid (GGUF-007)"))),
        },
        TYPE_STRING => MetaValue::String(input.read_string(u64::MAX)?),
        TYPE_ARRAY if top => {
            let elem = input.read_u32()?;
            if elem == TYPE_ARRAY {
                return Err(msg("gguf metadata holds a nested array (GGUF-005)"));
            }
            let size = elem_min_size(elem)?;
            let count = input.read_u64()?;
            if count > input.remaining() / size {
                return Err(msg(format!(
                    "gguf array of {count} elements runs past the end of the file (GGUF-003)"
                )));
            }
            let mut values = Vec::with_capacity(count as usize);
            for _ in 0..count {
                values.push(read_value(input, elem, false)?);
            }
            MetaValue::Array { elem, values }
        }
        TYPE_ARRAY => return Err(msg("gguf metadata holds a nested array (GGUF-005)")),
        TYPE_U64 => MetaValue::U64(input.read_u64()?),
        TYPE_I64 => MetaValue::I64(input.read_u64()? as i64),
        TYPE_F64 => MetaValue::F64(f64::from_bits(input.read_u64()?)),
        other => return Err(msg(format!("unknown gguf value type {other} (GGUF-004)"))),
    })
}

/// Smallest encoded size of one array element (GGUF-019).
fn elem_min_size(elem: u32) -> Result<u64> {
    Ok(match elem {
        TYPE_U8 | TYPE_I8 | TYPE_BOOL => 1,
        TYPE_U16 | TYPE_I16 => 2,
        TYPE_U32 | TYPE_I32 | TYPE_F32 => 4,
        TYPE_U64 | TYPE_I64 | TYPE_F64 | TYPE_STRING => 8,
        other => return Err(msg(format!("unknown gguf array element type {other} (GGUF-004)"))),
    })
}

/// The type id for a gguf-py type name, in either case.
pub fn gguf_type_id(name: &str) -> Option<u32> {
    [TYPE_U8, TYPE_I8, TYPE_U16, TYPE_I16, TYPE_U32, TYPE_I32, TYPE_F32, TYPE_BOOL, TYPE_STRING, TYPE_ARRAY, TYPE_U64, TYPE_I64, TYPE_F64]
        .into_iter()
        .find(|id| gguf_type_name(*id).eq_ignore_ascii_case(name))
}

pub fn gguf_type_name(id: u32) -> &'static str {
    match id {
        TYPE_U8 => "uint8",
        TYPE_I8 => "int8",
        TYPE_U16 => "uint16",
        TYPE_I16 => "int16",
        TYPE_U32 => "uint32",
        TYPE_I32 => "int32",
        TYPE_F32 => "float32",
        TYPE_BOOL => "bool",
        TYPE_STRING => "string",
        TYPE_ARRAY => "array",
        TYPE_U64 => "uint64",
        TYPE_I64 => "int64",
        TYPE_F64 => "float64",
        _ => "unknown",
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array { elem: u32, values: Vec<MetaValue> },
}

impl MetaValue {
    pub fn type_id(&self) -> u32 {
        match self {
            Self::U8(_) => TYPE_U8,
            Self::I8(_) => TYPE_I8,
            Self::U16(_) => TYPE_U16,
            Self::I16(_) => TYPE_I16,
            Self::U32(_) => TYPE_U32,
            Self::I32(_) => TYPE_I32,
            Self::F32(_) => TYPE_F32,
            Self::Bool(_) => TYPE_BOOL,
            Self::String(_) => TYPE_STRING,
            Self::U64(_) => TYPE_U64,
            Self::I64(_) => TYPE_I64,
            Self::F64(_) => TYPE_F64,
            Self::Array { .. } => TYPE_ARRAY,
        }
    }

    /// The value as metadata text: a string as itself, any other value as
    /// JSON (safetensors.yaml interop: "JSON-encode non-string values").
    pub fn text(&self) -> String {
        match self {
            Self::String(v) => v.clone(),
            other => other.json().to_string(),
        }
    }

    pub fn json(&self) -> Value {
        match self {
            Self::U8(v) => Value::from(*v),
            Self::I8(v) => Value::from(*v),
            Self::U16(v) => Value::from(*v),
            Self::I16(v) => Value::from(*v),
            Self::U32(v) => Value::from(*v),
            Self::I32(v) => Value::from(*v),
            Self::F32(v) => Value::from(*v),
            Self::Bool(v) => Value::from(*v),
            Self::String(v) => Value::from(v.clone()),
            Self::U64(v) => Value::from(*v),
            Self::I64(v) => Value::from(*v),
            Self::F64(v) => Value::from(*v),
            Self::Array { values, .. } => Value::Array(values.iter().map(MetaValue::json).collect()),
        }
    }

    /// raw-sidecar TypedValue: `{"type": T, "value": V}`, arrays with
    /// `elem`, 64-bit integers above 2^53 as decimal strings, and non-finite
    /// floats as "nan", "inf", "-inf".
    pub fn typed_value(&self) -> Value {
        self.tagged(false)
    }

    /// The CONV-004 envelope for a string map: TypedValue with the upper-case
    /// gguf-py type names. Floats are f64 shortest round trip (CONV-005).
    pub fn envelope(&self) -> Value {
        self.tagged(true)
    }

    /// The inverse of `typed_value` and `envelope` (either case), for the
    /// types GGUF has; None for anything else, or a value that does not fit
    /// its type.
    pub fn from_typed(tagged: &Value) -> Option<MetaValue> {
        let object = tagged.as_object()?;
        let kind = gguf_type_id(object.get("type")?.as_str()?)?;
        let value = object.get("value")?;
        if kind == TYPE_ARRAY {
            let elem = gguf_type_id(object.get("elem")?.as_str()?)?;
            if elem == TYPE_ARRAY {
                return None;
            }
            let values = value
                .as_array()?
                .iter()
                .map(|item| Self::plain_from(elem, item))
                .collect::<Option<Vec<_>>>()?;
            return Some(MetaValue::Array { elem, values });
        }
        Self::plain_from(kind, value)
    }

    fn plain_from(kind: u32, value: &Value) -> Option<MetaValue> {
        let int = || -> Option<i128> {
            match value {
                Value::Number(n) => n.as_i64().map(i128::from).or_else(|| n.as_u64().map(i128::from)),
                Value::String(text) => text.parse().ok(),
                _ => None,
            }
        };
        let float = || -> Option<f64> {
            match value {
                Value::Number(n) => n.as_f64(),
                Value::String(text) => match text.as_str() {
                    "nan" => Some(f64::NAN),
                    "inf" => Some(f64::INFINITY),
                    "-inf" => Some(f64::NEG_INFINITY),
                    _ => None,
                },
                _ => None,
            }
        };
        Some(match kind {
            TYPE_U8 => MetaValue::U8(int()?.try_into().ok()?),
            TYPE_I8 => MetaValue::I8(int()?.try_into().ok()?),
            TYPE_U16 => MetaValue::U16(int()?.try_into().ok()?),
            TYPE_I16 => MetaValue::I16(int()?.try_into().ok()?),
            TYPE_U32 => MetaValue::U32(int()?.try_into().ok()?),
            TYPE_I32 => MetaValue::I32(int()?.try_into().ok()?),
            TYPE_U64 => MetaValue::U64(int()?.try_into().ok()?),
            TYPE_I64 => MetaValue::I64(int()?.try_into().ok()?),
            TYPE_F32 => MetaValue::F32(float()? as f32),
            TYPE_F64 => MetaValue::F64(float()?),
            TYPE_BOOL => MetaValue::Bool(value.as_bool()?),
            TYPE_STRING => MetaValue::String(value.as_str()?.to_string()),
            _ => return None,
        })
    }

    fn tagged(&self, upper: bool) -> Value {
        let name = |id: u32| {
            let name = gguf_type_name(id);
            if upper {
                name.to_ascii_uppercase()
            } else {
                name.to_string()
            }
        };
        fn plain(value: &MetaValue) -> Value {
            const EXACT: u64 = 1 << 53;
            let float = |v: f64| {
                if v.is_nan() {
                    Value::from("nan")
                } else if v.is_infinite() {
                    Value::from(if v > 0.0 { "inf" } else { "-inf" })
                } else {
                    Value::from(v)
                }
            };
            match value {
                MetaValue::U64(v) if *v > EXACT => Value::from(v.to_string()),
                MetaValue::I64(v) if v.unsigned_abs() > EXACT => Value::from(v.to_string()),
                MetaValue::F32(v) => float(f64::from(*v)),
                MetaValue::F64(v) => float(*v),
                MetaValue::Array { values, .. } => Value::Array(values.iter().map(plain).collect()),
                other => other.json(),
            }
        }
        match self {
            Self::Array { elem, .. } => serde_json::json!({
                "type": name(TYPE_ARRAY),
                "elem": name(*elem),
                "value": plain(self),
            }),
            other => serde_json::json!({
                "type": name(other.type_id()),
                "value": plain(other),
            }),
        }
    }

    #[allow(dead_code)]
    pub fn display(&self) -> String {
        match self {
            Self::U8(v) => v.to_string(),
            Self::I8(v) => v.to_string(),
            Self::U16(v) => v.to_string(),
            Self::I16(v) => v.to_string(),
            Self::U32(v) => v.to_string(),
            Self::I32(v) => v.to_string(),
            Self::F32(v) => v.to_string(),
            Self::Bool(v) => v.to_string(),
            Self::String(v) => v.clone(),
            Self::U64(v) => v.to_string(),
            Self::I64(v) => v.to_string(),
            Self::F64(v) => v.to_string(),
            Self::Array { elem, values } => {
                if values.len() > 8 {
                    format!("array<{}>[{}]", gguf_type_name(*elem), values.len())
                } else {
                    format!(
                        "[{}]",
                        values
                            .iter()
                            .map(MetaValue::display)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
        }
    }
}

pub fn parse_typed_metadata(value: &Value) -> Result<Vec<(String, MetaValue)>> {
    let object = value
        .as_object()
        .ok_or_else(|| msg("GGUF metadata file must be a JSON object"))?;
    let mut out = Vec::new();
    for (key, entry) in object {
        out.push((key.clone(), parse_meta_value(entry)?));
    }
    Ok(out)
}

fn parse_meta_value(value: &Value) -> Result<MetaValue> {
    let object = value
        .as_object()
        .ok_or_else(|| msg("GGUF metadata values are objects with type and value"))?;
    let type_name = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| msg("GGUF metadata entry is missing type"))?;
    let raw = object
        .get("value")
        .ok_or_else(|| msg("GGUF metadata entry is missing value"))?;
    match type_name {
        "uint8" => Ok(MetaValue::U8(narrow(json_u64(raw)?)?)),
        "int8" => Ok(MetaValue::I8(narrow(json_i64(raw)?)?)),
        "uint16" => Ok(MetaValue::U16(narrow(json_u64(raw)?)?)),
        "int16" => Ok(MetaValue::I16(narrow(json_i64(raw)?)?)),
        "uint32" => Ok(MetaValue::U32(narrow(json_u64(raw)?)?)),
        "int32" => Ok(MetaValue::I32(narrow(json_i64(raw)?)?)),
        "float32" => Ok(MetaValue::F32(json_f64(raw)? as f32)),
        "bool" => Ok(MetaValue::Bool(
            raw.as_bool()
                .ok_or_else(|| msg("bool metadata needs true or false"))?,
        )),
        "string" => Ok(MetaValue::String(
            raw.as_str()
                .ok_or_else(|| msg("string metadata needs a string"))?
                .to_string(),
        )),
        "uint64" => Ok(MetaValue::U64(json_u64(raw)?)),
        "int64" => Ok(MetaValue::I64(json_i64(raw)?)),
        "float64" => Ok(MetaValue::F64(json_f64(raw)?)),
        "array" => {
            let elem_name = object
                .get("elem")
                .and_then(Value::as_str)
                .ok_or_else(|| msg("array metadata needs elem"))?;
            let elem = type_id_from_name(elem_name)?;
            let items = raw
                .as_array()
                .ok_or_else(|| msg("array value must be a JSON array"))?;
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                values.push(parse_plain_elem(elem, item)?);
            }
            Ok(MetaValue::Array { elem, values })
        }
        other => Err(msg(format!("unknown GGUF metadata type {other}"))),
    }
}

fn parse_plain_elem(elem: u32, value: &Value) -> Result<MetaValue> {
    match elem {
        TYPE_STRING => Ok(MetaValue::String(
            value
                .as_str()
                .ok_or_else(|| msg("array string is not a string"))?
                .to_string(),
        )),
        TYPE_U8 => Ok(MetaValue::U8(narrow(json_u64(value)?)?)),
        TYPE_I8 => Ok(MetaValue::I8(narrow(json_i64(value)?)?)),
        TYPE_U16 => Ok(MetaValue::U16(narrow(json_u64(value)?)?)),
        TYPE_I16 => Ok(MetaValue::I16(narrow(json_i64(value)?)?)),
        TYPE_U32 => Ok(MetaValue::U32(narrow(json_u64(value)?)?)),
        TYPE_I32 => Ok(MetaValue::I32(narrow(json_i64(value)?)?)),
        TYPE_U64 => Ok(MetaValue::U64(json_u64(value)?)),
        TYPE_I64 => Ok(MetaValue::I64(json_i64(value)?)),
        TYPE_F32 => Ok(MetaValue::F32(json_f64(value)? as f32)),
        TYPE_F64 => Ok(MetaValue::F64(json_f64(value)?)),
        TYPE_BOOL => Ok(MetaValue::Bool(
            value
                .as_bool()
                .ok_or_else(|| msg("array bool is invalid"))?,
        )),
        _ => Err(msg(
            "nested GGUF arrays in metadata files are not supported",
        )),
    }
}

fn type_id_from_name(name: &str) -> Result<u32> {
    Ok(match name {
        "uint8" => TYPE_U8,
        "int8" => TYPE_I8,
        "uint16" => TYPE_U16,
        "int16" => TYPE_I16,
        "uint32" => TYPE_U32,
        "int32" => TYPE_I32,
        "float32" => TYPE_F32,
        "bool" => TYPE_BOOL,
        "string" => TYPE_STRING,
        "uint64" => TYPE_U64,
        "int64" => TYPE_I64,
        "float64" => TYPE_F64,
        _ => return Err(msg(format!("unknown GGUF element type {name}"))),
    })
}

fn narrow<T: TryFrom<S>, S: std::fmt::Display + Copy>(value: S) -> Result<T> {
    T::try_from(value).map_err(|_| msg(format!("metadata value {value} does not fit its type")))
}

fn json_u64(value: &Value) -> Result<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|n| u64::try_from(n).ok()))
        .ok_or_else(|| msg("metadata number is not an unsigned integer"))
}

fn json_i64(value: &Value) -> Result<i64> {
    value
        .as_i64()
        .ok_or_else(|| msg("metadata number is not an integer"))
}

fn json_f64(value: &Value) -> Result<f64> {
    value
        .as_f64()
        .ok_or_else(|| msg("metadata number is not a float"))
}

enum MetaWrite {
    Copy { offset: u64, len: u64 },
    New { key: String, value: MetaValue },
}

/// Copy the tensor blob and replace metadata. Tensor bytes stay at the same blob offsets.
pub fn rewrite_metadata(
    src: &GgufFile,
    dest: &Path,
    updates: &[(String, MetaValue)],
) -> Result<()> {
    if !(MIN_VERSION..=VERSION).contains(&src.version) {
        return Err(msg(format!(
            "refusing to write GGUF version {}; tenspile writes little-endian versions {MIN_VERSION} and {VERSION}",
            src.version
        )));
    }
    let mut alignment = src.alignment;
    let mut update_keys = std::collections::HashSet::new();
    for (key, value) in updates {
        if !update_keys.insert(key.as_str()) {
            return Err(msg(format!("metadata key {key} is given twice")));
        }
        if key == "general.alignment" {
            alignment = match value {
                MetaValue::U32(v) => u64::from(*v),
                _ => return Err(msg("general.alignment must be uint32 (GGUF-008)")),
            };
            if alignment < 8 || !alignment.is_power_of_two() {
                return Err(msg(
                    "general.alignment must be a power of two and at least 8",
                ));
            }
        }
        validate_key(key)?;
    }
    for tensor in &src.tensors {
        if tensor.offset % alignment != 0 {
            return Err(msg(format!(
                "tensor {} offset {} is not aligned to {alignment}",
                tensor.name, tensor.offset
            )));
        }
    }
    let mut plan: Vec<MetaWrite> = Vec::new();
    let mut replaced = vec![false; updates.len()];
    for field in &src.metadata {
        if let Some(index) = updates.iter().position(|(key, _)| key == &field.key) {
            replaced[index] = true;
            plan.push(MetaWrite::New {
                key: field.key.clone(),
                value: updates[index].1.clone(),
            });
        } else {
            plan.push(MetaWrite::Copy {
                offset: field.entry_offset,
                len: field.entry_len,
            });
        }
    }
    for (index, (key, value)) in updates.iter().enumerate() {
        if !replaced[index] {
            plan.push(MetaWrite::New {
                key: key.clone(),
                value: value.clone(),
            });
        }
    }

    atomic_write(dest, |out| {
        out.write_all(MAGIC)?;
        // A version 2 file stays version 2. The field layout matches version 3.
        out.write_all(&src.version.to_le_bytes())?;
        out.write_all(&(src.tensors.len() as u64).to_le_bytes())?;
        out.write_all(&(plan.len() as u64).to_le_bytes())?;
        for item in &plan {
            match item {
                MetaWrite::Copy { offset, len } => {
                    copy_range(&src.path, *offset, *len, out)?;
                }
                MetaWrite::New { key, value } => write_kv(out, key, value)?,
            }
        }
        for tensor in &src.tensors {
            write_string(out, &tensor.name)?;
            out.write_all(&(tensor.shape.len() as u32).to_le_bytes())?;
            for dim in &tensor.shape {
                out.write_all(&dim.to_le_bytes())?;
            }
            out.write_all(&tensor.ggml_type.to_le_bytes())?;
            out.write_all(&tensor.offset.to_le_bytes())?;
        }
        let written = out.stream_position()?;
        let padded = align_up(written, alignment);
        let gap = padded - written;
        if gap > 0 {
            out.write_all(&vec![0u8; gap as usize])?;
        }
        copy_range(&src.path, src.data_start, src.data_len, out)?;
        Ok(())
    })
}

fn validate_key(key: &str) -> Result<()> {
    if key.is_empty() || key.len() > 65_535 {
        return Err(msg(format!(
            "gguf metadata key {key} has an invalid length"
        )));
    }
    if !key.is_ascii() {
        return Err(msg(format!("gguf metadata key {key} must be ASCII")));
    }
    Ok(())
}

fn write_kv(out: &mut File, key: &str, value: &MetaValue) -> Result<()> {
    write_string(out, key)?;
    out.write_all(&value.type_id().to_le_bytes())?;
    write_value(out, value)
}

fn write_value(out: &mut File, value: &MetaValue) -> Result<()> {
    match value {
        MetaValue::U8(v) => out.write_all(&[*v])?,
        MetaValue::I8(v) => out.write_all(&[*v as u8])?,
        MetaValue::U16(v) => out.write_all(&v.to_le_bytes())?,
        MetaValue::I16(v) => out.write_all(&v.to_le_bytes())?,
        MetaValue::U32(v) => out.write_all(&v.to_le_bytes())?,
        MetaValue::I32(v) => out.write_all(&v.to_le_bytes())?,
        MetaValue::F32(v) => out.write_all(&v.to_le_bytes())?,
        MetaValue::Bool(v) => out.write_all(&[*v as u8])?,
        MetaValue::String(v) => write_string(out, v)?,
        MetaValue::U64(v) => out.write_all(&v.to_le_bytes())?,
        MetaValue::I64(v) => out.write_all(&v.to_le_bytes())?,
        MetaValue::F64(v) => out.write_all(&v.to_le_bytes())?,
        MetaValue::Array { elem, values } => {
            out.write_all(&elem.to_le_bytes())?;
            out.write_all(&(values.len() as u64).to_le_bytes())?;
            for item in values {
                if item.type_id() != *elem {
                    return Err(msg("gguf array element type does not match"));
                }
                write_value(out, item)?;
            }
        }
    }
    Ok(())
}

fn write_string(out: &mut impl Write, text: &str) -> Result<()> {
    out.write_all(&(text.len() as u64).to_le_bytes())?;
    out.write_all(text.as_bytes())?;
    Ok(())
}

/// One dense element tensor. `ne` is ggml order, innermost first.
pub struct DenseTensor {
    pub name: String,
    pub ggml_type: u32,
    pub ne: Vec<u64>,
    pub payload: Payload,
}

/// Write a little-endian GGUF v3 file of dense element tensors.
pub fn write_dense(
    path: &Path,
    metadata: &[(String, MetaValue)],
    tensors: &[DenseTensor],
) -> Result<()> {
    crate::io_util::check_unique_names(tensors.iter().map(|tensor| tensor.name.as_str()))?;
    let mut keys = std::collections::HashSet::new();
    for (key, _) in metadata {
        if !keys.insert(key.as_str()) {
            return Err(msg(format!("gguf metadata key {key} is given twice (GGUF-006)")));
        }
    }
    // The data follows the alignment the metadata names; without the key it
    // is 32, and the key is not written (gguf.yaml write algorithm).
    let alignment = match metadata.iter().find(|(key, _)| key == "general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some((_, MetaValue::U32(value))) if *value >= 8 && value.is_power_of_two() => {
            u64::from(*value)
        }
        Some(_) => {
            return Err(msg(
                "general.alignment must be a uint32 power of two of at least 8 (GGUF-008, GGUF-020)",
            ))
        }
    };
    let metadata = metadata.to_vec();
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut cursor = 0u64;
    for tensor in tensors {
        if tensor.name.is_empty() || tensor.name.len() >= 64 {
            return Err(msg(format!(
                "GGUF tensor name {} must be shorter than 64 bytes",
                tensor.name
            )));
        }
        if tensor.ne.is_empty() || tensor.ne.len() > 4 {
            return Err(msg(format!(
                "GGUF tensor {} has rank {}; the file stores 1 to 4 dimensions",
                tensor.name,
                tensor.ne.len()
            )));
        }
        let expect = crate::dtype::ggml_payload_bytes(tensor.ggml_type, &tensor.ne)?;
        if tensor.payload.len() != expect {
            return Err(msg(format!(
                "tensor {} is {} bytes and ggml type {} needs {expect}",
                tensor.name,
                tensor.payload.len(),
                ggml_type_name(tensor.ggml_type)
            )));
        }
        let offset = align_up(cursor, alignment);
        offsets.push(offset);
        cursor = offset + tensor.payload.len();
    }
    let mut header = 4 + 4 + 8 + 8;
    for (key, value) in &metadata {
        validate_key(key)?;
        header += kv_len(key, value)?;
    }
    for tensor in tensors {
        header += string_len(&tensor.name) + 4 + 8 * tensor.ne.len() as u64 + 4 + 8;
    }
    let data_start = align_up(header, alignment);
    atomic_write(path, |out| {
        out.write_all(MAGIC)?;
        out.write_all(&VERSION.to_le_bytes())?;
        out.write_all(&(tensors.len() as u64).to_le_bytes())?;
        out.write_all(&(metadata.len() as u64).to_le_bytes())?;
        for (key, value) in &metadata {
            write_kv(out, key, value)?;
        }
        for (tensor, offset) in tensors.iter().zip(offsets.iter()) {
            write_string(out, &tensor.name)?;
            out.write_all(&(tensor.ne.len() as u32).to_le_bytes())?;
            for dim in &tensor.ne {
                out.write_all(&dim.to_le_bytes())?;
            }
            out.write_all(&tensor.ggml_type.to_le_bytes())?;
            out.write_all(&offset.to_le_bytes())?;
        }
        let written = out.stream_position()?;
        if written != header {
            return Err(msg(format!(
                "GGUF header is {written} bytes and was planned as {header}"
            )));
        }
        if data_start > written {
            out.write_all(&vec![0u8; (data_start - written) as usize])?;
        }
        let mut place = 0u64;
        for (tensor, offset) in tensors.iter().zip(offsets.iter()) {
            if *offset > place {
                out.write_all(&vec![0u8; (*offset - place) as usize])?;
            }
            tensor.payload.copy_to(out)?;
            place = offset + tensor.payload.len();
        }
        Ok(())
    })
}

fn string_len(text: &str) -> u64 {
    8 + text.len() as u64
}

fn kv_len(key: &str, value: &MetaValue) -> Result<u64> {
    Ok(string_len(key) + 4 + value_len(value)?)
}

fn value_len(value: &MetaValue) -> Result<u64> {
    Ok(match value {
        MetaValue::U8(_) | MetaValue::I8(_) | MetaValue::Bool(_) => 1,
        MetaValue::U16(_) | MetaValue::I16(_) => 2,
        MetaValue::U32(_) | MetaValue::I32(_) | MetaValue::F32(_) => 4,
        MetaValue::U64(_) | MetaValue::I64(_) | MetaValue::F64(_) => 8,
        MetaValue::String(text) => string_len(text),
        MetaValue::Array { values, .. } => {
            let mut total = 4 + 8;
            for item in values {
                total += value_len(item)?;
            }
            total
        }
    })
}

/// Build a small version-3 file. Used by metadata rewrites' tests and as a fixture writer.
pub fn write_fixture(path: &Path, name: &str, tensor_name: &str, f32_values: &[f32]) -> Result<()> {
    let mut payload = Vec::with_capacity(f32_values.len() * 4);
    for value in f32_values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    let alignment = DEFAULT_ALIGNMENT;
    atomic_write(path, |out| {
        out.write_all(MAGIC)?;
        out.write_all(&VERSION.to_le_bytes())?;
        out.write_all(&1u64.to_le_bytes())?;
        out.write_all(&3u64.to_le_bytes())?;
        write_kv(
            out,
            "general.architecture",
            &MetaValue::String("test".into()),
        )?;
        write_kv(out, "general.alignment", &MetaValue::U32(alignment as u32))?;
        write_kv(out, "general.name", &MetaValue::String(name.into()))?;
        write_string(out, tensor_name)?;
        out.write_all(&1u32.to_le_bytes())?;
        out.write_all(&(f32_values.len() as u64).to_le_bytes())?;
        out.write_all(&0u32.to_le_bytes())?; // F32
        out.write_all(&0u64.to_le_bytes())?; // offset
        let written = out.stream_position()?;
        let padded = align_up(written, alignment);
        out.write_all(&vec![0u8; (padded - written) as usize])?;
        out.write_all(&payload)?;
        Ok(())
    })?;
    Ok(())
}

fn unsupported_version(path: &Path, version: u32) -> crate::error::Error {
    // GGUF-002i: a version word whose low 16 bits are zero is byte-swapped.
    if version & 0xFFFF == 0 {
        return msg(format!(
            "{} looks like big-endian GGUF, which tenspile does not read",
            path.display()
        ));
    }
    if version == 1 {
        return msg(format!(
            "{}: GGUF version 1 is not supported. That layout used 32-bit counts, string lengths, array lengths, and dimension sizes, and it was only written from 21 to 27 August 2023. llama.cpp removed the version 1 reader on 2 November 2023. tenspile reads little-endian versions {MIN_VERSION} and {VERSION}",
            path.display()
        ));
    }
    msg(format!(
        "{}: GGUF version {version} is not supported; tenspile reads little-endian versions {MIN_VERSION} and {VERSION}",
        path.display()
    ))
}

pub fn type_label(tensor: &GgufTensor) -> String {
    ggml_type_name(tensor.ggml_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cut_tensor_blob_suggests_truncation() {
        // ggml type 0 is F32: 4 elements need 16 bytes.
        let past = spans(vec![("w".into(), 0, vec![4], 32)], 0, 16).unwrap_err().to_string();
        assert!(past.ends_with("the file may be truncated or damaged"), "{past}");
        let short = spans(vec![("w".into(), 0, vec![4], 0)], 0, 8).unwrap_err().to_string();
        assert!(short.ends_with("the file may be truncated or damaged"), "{short}");
    }

    #[test]
    fn metadata_rewrite_keeps_tensor_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        write_fixture(&path, "old", "weight", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let original = read_file(&path).unwrap();
        let before =
            crate::io_util::read_range(&path, original.tensors[0].file_offset, 16).unwrap();
        let updated = dir.path().join("updated.gguf");
        rewrite_metadata(
            &original,
            &updated,
            &[("general.name".into(), MetaValue::String("new".into()))],
        )
        .unwrap();
        let after = read_file(&updated).unwrap();
        let name = after
            .metadata
            .iter()
            .find(|field| field.key == "general.name")
            .unwrap();
        assert_eq!(name.value.text(), "new");
        let bytes = crate::io_util::read_range(&updated, after.tensors[0].file_offset, 16).unwrap();
        assert_eq!(bytes, before);
        assert_eq!(after.tensors[0].name, "weight");
        assert_eq!(after.tensors[0].span, 16);
        assert_eq!(after.version, VERSION);
        let header = std::fs::read(&updated).unwrap();
        assert_eq!(&header[4..8], &VERSION.to_le_bytes());
    }

    #[test]
    fn version_2_reads_and_rewrite_keeps_version_2() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        write_fixture(&path, "old", "weight", &[1.0, 2.0]).unwrap();
        set_version_word(&path, 2);
        let original = read_file(&path).unwrap();
        assert_eq!(original.version, 2);
        assert_eq!(original.tensors[0].name, "weight");
        assert_eq!(original.metadata[2].value.text(), "old");
        let before = crate::io_util::read_range(&path, original.tensors[0].file_offset, 8).unwrap();
        let updated = dir.path().join("updated.gguf");
        rewrite_metadata(
            &original,
            &updated,
            &[("general.name".into(), MetaValue::String("kept".into()))],
        )
        .unwrap();
        let after = read_file(&updated).unwrap();
        assert_eq!(after.version, 2);
        let name = after
            .metadata
            .iter()
            .find(|field| field.key == "general.name")
            .unwrap();
        assert_eq!(name.value.text(), "kept");
        let bytes = crate::io_util::read_range(&updated, after.tensors[0].file_offset, 8).unwrap();
        assert_eq!(bytes, before);
        let header = std::fs::read(&updated).unwrap();
        assert_eq!(&header[0..4], MAGIC);
        assert_eq!(&header[4..8], &2u32.to_le_bytes());
    }

    #[test]
    fn version_1_and_other_versions_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        write_fixture(&path, "old", "weight", &[1.0, 2.0]).unwrap();

        set_version_word(&path, 1);
        let err = read_file(&path).unwrap_err().to_string();
        assert!(err.contains("version 1"), "{err}");
        assert!(err.contains("32-bit"), "{err}");
        assert!(err.contains("August 2023"), "{err}");

        set_version_word(&path, 4);
        let err = read_file(&path).unwrap_err().to_string();
        assert!(err.contains("version 4"), "{err}");
        assert!(err.contains("versions 2 and 3"), "{err}");

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[4..8].copy_from_slice(&3u32.to_be_bytes());
        std::fs::write(&path, bytes).unwrap();
        let err = read_file(&path).unwrap_err().to_string();
        assert!(err.contains("big-endian"), "{err}");
    }

    fn set_version_word(path: &std::path::Path, version: u32) {
        let mut bytes = std::fs::read(path).unwrap();
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        std::fs::write(path, bytes).unwrap();
    }
}
