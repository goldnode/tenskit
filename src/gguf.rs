use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde_json::Value;

use crate::dtype::{ggml_dense, ggml_type_name, numel};
use crate::error::{io_err, msg, Result};
use crate::io_util::{atomic_write, copy_range};

pub const MAGIC: &[u8] = b"GGUF";
/// Current writer version. Little-endian version 3 is what GGUF writers emit.
pub const VERSION: u32 = 3;
/// Oldest little-endian version this reader accepts. Version 2 uses the same field widths.
const MIN_VERSION: u32 = 2;
pub const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_DIMS: u32 = 8;
const MAX_STRING: u64 = 16 * 1024 * 1024;
const MAX_ARRAY: u64 = 50_000_000;
const DISPLAY_STRING: usize = 120;

const TYPE_U8: u32 = 0;
const TYPE_I8: u32 = 1;
const TYPE_U16: u32 = 2;
const TYPE_I16: u32 = 3;
const TYPE_U32: u32 = 4;
const TYPE_I32: u32 = 5;
const TYPE_F32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_U64: u32 = 10;
const TYPE_I64: u32 = 11;
const TYPE_F64: u32 = 12;

#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub ggml_type: u32,
    pub shape: Vec<u64>,
    /// Offset within the tensor blob.
    pub offset: u64,
    /// Bytes from this offset to the next tensor, including alignment padding.
    pub span: u64,
    /// File offset of the first stored byte.
    pub file_offset: u64,
}

#[derive(Debug, Clone)]
pub struct MetaField {
    pub key: String,
    #[allow(dead_code)]
    pub value_type: u32,
    /// File offset of the whole entry: key, type, and value.
    pub entry_offset: u64,
    pub entry_len: u64,
    pub display: String,
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

    fn read_string(&mut self, limit: u64) -> Result<String> {
        let len = self.read_u64()?;
        if len > limit {
            return Err(msg(format!("gguf string is {len} bytes")));
        }
        let mut buf = vec![0u8; len as usize];
        self.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|_| msg("gguf string is not utf-8"))
    }

    fn skip(&mut self, n: u64) -> Result<()> {
        if self.pos.checked_add(n).is_none_or(|end| end > self.len) {
            return Err(msg("gguf read runs past the end of the file"));
        }
        self.file.seek(SeekFrom::Current(n as i64))?;
        self.pos += n;
        Ok(())
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
    // Versions 2 and 3 share little-endian field widths. A big-endian version
    // word is larger than 65536 when those bytes are read as little-endian.
    if !(MIN_VERSION..=VERSION).contains(&version) {
        return Err(unsupported_version(path, version));
    }
    let tensor_count = input.read_u64()?;
    let kv_count = input.read_u64()?;
    if tensor_count > 10_000_000 || kv_count > 10_000_000 {
        return Err(msg("gguf counts are unreasonably large"));
    }
    let mut metadata = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        let entry_offset = input.pos;
        let key = input.read_string(65_535)?;
        let value_type = input.read_u32()?;
        let display = read_value_display(&mut input, value_type, 0)?;
        let entry_len = input.pos - entry_offset;
        metadata.push(MetaField {
            key,
            value_type,
            entry_offset,
            entry_len,
            display,
        });
    }
    let alignment = alignment_of(&metadata);
    let mut tensors_raw = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = input.read_string(1024)?;
        let n_dims = input.read_u32()?;
        if n_dims > MAX_DIMS {
            return Err(msg(format!("gguf tensor {name} has {n_dims} dimensions")));
        }
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(input.read_u64()?);
        }
        let ggml_type = input.read_u32()?;
        let offset = input.read_u64()?;
        tensors_raw.push((name, ggml_type, shape, offset));
    }
    let data_start = align_up(input.pos, alignment);
    if data_start > input.len {
        return Err(msg("gguf tensor data starts past the end of the file"));
    }
    let data_len = input.len - data_start;
    let tensors = spans(tensors_raw, data_start, data_len)?;
    Ok(GgufFile {
        path: path.to_path_buf(),
        version,
        alignment,
        data_start,
        data_len,
        metadata,
        tensors,
    })
}

fn alignment_of(metadata: &[MetaField]) -> u64 {
    for field in metadata {
        if field.key == "general.alignment" {
            if let Some(value) = parse_alignment_display(&field.display) {
                if value >= 8 && value.is_power_of_two() {
                    return value;
                }
            }
        }
    }
    DEFAULT_ALIGNMENT
}

fn parse_alignment_display(text: &str) -> Option<u64> {
    text.parse().ok()
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
                "gguf tensor {} offset {offset} is past the tensor blob",
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
    Ok(raw
        .drain(..)
        .enumerate()
        .map(|(index, (name, ggml_type, shape, offset))| GgufTensor {
            name,
            ggml_type,
            shape,
            offset,
            span: span_of[index],
            file_offset: data_start + offset,
        })
        .collect())
}

pub fn align_up(offset: u64, alignment: u64) -> u64 {
    let alignment = alignment.max(1);
    offset.saturating_add(alignment - 1) / alignment * alignment
}

fn read_value_display(input: &mut In, value_type: u32, depth: u32) -> Result<String> {
    if depth > 8 {
        return Err(msg("gguf array nesting is too deep"));
    }
    match value_type {
        TYPE_U8 => Ok(input.read_u8()?.to_string()),
        TYPE_I8 => Ok((input.read_u8()? as i8).to_string()),
        TYPE_U16 => {
            let mut buf = [0u8; 2];
            input.read_exact(&mut buf)?;
            Ok(u16::from_le_bytes(buf).to_string())
        }
        TYPE_I16 => {
            let mut buf = [0u8; 2];
            input.read_exact(&mut buf)?;
            Ok(i16::from_le_bytes(buf).to_string())
        }
        TYPE_U32 => Ok(input.read_u32()?.to_string()),
        TYPE_I32 => Ok((input.read_u32()? as i32).to_string()),
        TYPE_F32 => {
            let mut buf = [0u8; 4];
            input.read_exact(&mut buf)?;
            Ok(f32::from_le_bytes(buf).to_string())
        }
        TYPE_BOOL => {
            let value = input.read_u8()?;
            match value {
                0 => Ok("false".to_string()),
                1 => Ok("true".to_string()),
                _ => Err(msg(format!("gguf bool value {value} is invalid"))),
            }
        }
        TYPE_STRING => {
            let text = input.read_string(MAX_STRING)?;
            if text.chars().count() > DISPLAY_STRING {
                let prefix: String = text.chars().take(DISPLAY_STRING).collect();
                Ok(format!("{prefix}… ({} bytes)", text.len()))
            } else {
                Ok(text)
            }
        }
        TYPE_ARRAY => {
            let elem = input.read_u32()?;
            let count = input.read_u64()?;
            if count > MAX_ARRAY {
                return Err(msg(format!("gguf array has {count} elements")));
            }
            if count > 8 {
                for _ in 0..count {
                    skip_value(input, elem, depth + 1)?;
                }
                Ok(format!("array<{}>[{count}]", gguf_type_name(elem)))
            } else {
                let mut items = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    items.push(read_value_display(input, elem, depth + 1)?);
                }
                Ok(format!("[{}]", items.join(", ")))
            }
        }
        TYPE_U64 => Ok(input.read_u64()?.to_string()),
        TYPE_I64 => Ok((input.read_u64()? as i64).to_string()),
        TYPE_F64 => {
            let mut buf = [0u8; 8];
            input.read_exact(&mut buf)?;
            Ok(f64::from_le_bytes(buf).to_string())
        }
        other => Err(msg(format!("unknown gguf value type {other}"))),
    }
}

fn skip_value(input: &mut In, value_type: u32, depth: u32) -> Result<()> {
    if depth > 8 {
        return Err(msg("gguf array nesting is too deep"));
    }
    match value_type {
        TYPE_U8 | TYPE_I8 | TYPE_BOOL => {
            input.skip(1)?;
        }
        TYPE_U16 | TYPE_I16 => input.skip(2)?,
        TYPE_U32 | TYPE_I32 | TYPE_F32 => input.skip(4)?,
        TYPE_U64 | TYPE_I64 | TYPE_F64 => input.skip(8)?,
        TYPE_STRING => {
            let len = input.read_u64()?;
            if len > MAX_STRING {
                return Err(msg(format!("gguf string is {len} bytes")));
            }
            input.skip(len)?;
        }
        TYPE_ARRAY => {
            let elem = input.read_u32()?;
            let count = input.read_u64()?;
            if count > MAX_ARRAY {
                return Err(msg(format!("gguf array has {count} elements")));
            }
            for _ in 0..count {
                skip_value(input, elem, depth + 1)?;
            }
        }
        other => return Err(msg(format!("unknown gguf value type {other}"))),
    }
    Ok(())
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

#[derive(Debug, Clone)]
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
        "uint8" => Ok(MetaValue::U8(json_u64(raw)? as u8)),
        "int8" => Ok(MetaValue::I8(json_i64(raw)? as i8)),
        "uint16" => Ok(MetaValue::U16(json_u64(raw)? as u16)),
        "int16" => Ok(MetaValue::I16(json_i64(raw)? as i16)),
        "uint32" => Ok(MetaValue::U32(json_u64(raw)? as u32)),
        "int32" => Ok(MetaValue::I32(json_i64(raw)? as i32)),
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
        TYPE_U8 => Ok(MetaValue::U8(json_u64(value)? as u8)),
        TYPE_I8 => Ok(MetaValue::I8(json_i64(value)? as i8)),
        TYPE_U16 => Ok(MetaValue::U16(json_u64(value)? as u16)),
        TYPE_I16 => Ok(MetaValue::I16(json_i64(value)? as i16)),
        TYPE_U32 => Ok(MetaValue::U32(json_u64(value)? as u32)),
        TYPE_I32 => Ok(MetaValue::I32(json_i64(value)? as i32)),
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
    for (key, value) in updates {
        if key == "general.alignment" {
            alignment = match value {
                MetaValue::U32(v) => u64::from(*v),
                MetaValue::U64(v) => *v,
                MetaValue::I32(v) if *v > 0 => *v as u64,
                _ => return Err(msg("general.alignment must be an unsigned integer")),
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
    if version > 65_536 {
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

pub fn logical_dense_bytes(tensor: &GgufTensor) -> Option<u64> {
    let dtype = ggml_dense(tensor.ggml_type)?;
    let elements = numel(&tensor.shape)?;
    elements.checked_mul(u64::from(dtype.bits / 8))
}

pub fn type_label(tensor: &GgufTensor) -> String {
    ggml_type_name(tensor.ggml_type)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(name.display, "new");
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
        assert_eq!(original.metadata[2].display, "old");
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
        assert_eq!(name.display, "kept");
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
