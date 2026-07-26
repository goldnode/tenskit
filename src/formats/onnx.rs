//! ONNX models, read as graph initializers.
//!
//! The reader walks the protobuf for `ModelProto.graph.initializer` and does
//! not load a protobuf code generator. Graph nodes are not tensors and are
//! left in the file. Remove this file and the `onnx` line in
//! `local/supported-formats.txt` to drop the format.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::archive::{Archive, Inner, Member};
use crate::cli::FormatName;
use crate::dtype::{self, canonical_nbytes};
use crate::error::{io_err, msg, Result};
use crate::io_util::relative_child;
use crate::payload::Payload;

struct Found {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    payload: Payload,
    file_offset: u64,
}

pub fn open(path: &Path) -> Result<Archive> {
    let bytes = std::fs::read(path).map_err(|err| io_err(path, err))?;
    if bytes.len() as u64 >= 2 * 1024 * 1024 * 1024 {
        return Err(msg(format!(
            "{} is at least 2 GiB; ONNX requires external data (ONNX-006)",
            path.display()
        )));
    }
    let (ir, metadata, found, mut notes) = parse_model(&bytes, path)?;
    if found.is_empty() {
        return Err(msg(format!(
            "{} has no ONNX initializers",
            path.display()
        )));
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut members = Vec::with_capacity(found.len());
    let mut data_bytes = 0u64;
    for tensor in found {
        if !seen.insert(tensor.name.clone()) {
            return Err(msg(format!(
                "ONNX initializer {} is duplicated (ONNX-005)",
                tensor.name
            )));
        }
        let nbytes = tensor.payload.len();
        let expected = canonical_nbytes(&tensor.dtype, &tensor.shape)?;
        if nbytes != expected {
            return Err(msg(format!(
                "ONNX initializer {} is {nbytes} bytes; {} needs {expected} (ONNX-002)",
                tensor.name, tensor.dtype
            )));
        }
        data_bytes = data_bytes.saturating_add(nbytes);
        members.push(Member {
            name: tensor.name,
            dtype: tensor.dtype,
            numel: dtype::numel(&tensor.shape),
            shape: tensor.shape,
            file_offset: tensor.file_offset,
            nbytes,
            payload: tensor.payload,
            shard: None,
            detail: None,
            fortran: false,
        });
    }
    notes.sort();
    notes.dedup();
    Ok(Archive {
        format: FormatName::Onnx,
        version: format!("ir {ir}"),
        path: path.to_path_buf(),
        members,
        metadata,
        data_bytes,
        notes,
        inner: Inner::Onnx,
    })
}

fn parse_model(bytes: &[u8], path: &Path) -> Result<(u64, Vec<(String, String)>, Vec<Found>, Vec<String>)> {
    let mut ir = None;
    let mut metadata = Vec::new();
    let mut found = Vec::new();
    let mut notes = Vec::new();
    for field in fields(bytes, 0)? {
        match field {
            Field::Varint { id: 1, value } => ir = Some(value),
            Field::Bytes { id: 7, data, at: _ } => {
                parse_graph(&data, path, &mut found, &mut notes)?;
            }
            Field::Bytes { id: 14, data, .. } => {
                if let Some((key, value)) = string_pair(&data)? {
                    metadata.push((key, value));
                }
            }
            _ => {}
        }
    }
    let ir = ir.ok_or_else(|| {
        msg(format!(
            "{} is not an ONNX ModelProto; it has no ir_version",
            path.display()
        ))
    })?;
    Ok((ir, metadata, found, notes))
}

fn parse_graph(bytes: &[u8], path: &Path, found: &mut Vec<Found>, notes: &mut Vec<String>) -> Result<()> {
    for field in fields(bytes, 0)? {
        match field {
            Field::Bytes { id: 5, data, at } => {
                found.push(parse_tensor(&data, at, path, notes)?);
            }
            Field::Bytes { id: 15, .. } => {
                return Err(msg(
                    "ONNX sparse initializers are not expanded by tenspile",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_tensor(bytes: &[u8], base: u64, path: &Path, notes: &mut Vec<String>) -> Result<Found> {
    let mut dims = Vec::new();
    let mut data_type = None;
    let mut name = None;
    let mut raw: Option<(u64, Vec<u8>)> = None;
    let mut float_data = None;
    let mut int32_data = None;
    let mut int64_data = None;
    let mut double_data = None;
    let mut uint64_data = None;
    let mut external = Vec::new();
    let mut location_external = false;
    for field in fields(bytes, base)? {
        match field {
            Field::Varint { id: 1, value } => dims.push(value),
            Field::Varint { id: 2, value } => data_type = Some(value),
            Field::Bytes { id: 8, data, .. } => {
                name = Some(
                    String::from_utf8(data).map_err(|_| msg("ONNX tensor name is not utf-8"))?,
                );
            }
            Field::Bytes { id: 9, data, at } => raw = Some((at, data)),
            Field::Bytes { id: 4, data, .. } => float_data = Some(data),
            Field::Bytes { id: 5, data, .. } => int32_data = Some(data),
            Field::Bytes { id: 7, data, .. } => int64_data = Some(data),
            Field::Bytes { id: 10, data, .. } => double_data = Some(data),
            Field::Bytes { id: 11, data, .. } => uint64_data = Some(data),
            Field::Bytes { id: 13, data, .. } => {
                if let Some(pair) = string_pair(&data)? {
                    external.push(pair);
                }
            }
            Field::Varint { id: 14, value } if value == 1 => location_external = true,
            _ => {}
        }
    }
    let code = data_type.ok_or_else(|| msg("ONNX initializer has no data_type"))?;
    let dtype = onnx_dtype(code)?;
    let name = name.unwrap_or_default();
    if name.is_empty() {
        return Err(msg("ONNX initializer has an empty name"));
    }
    let sources = [
        raw.is_some(),
        float_data.is_some() || int32_data.is_some() || int64_data.is_some() || double_data.is_some() || uint64_data.is_some(),
        location_external,
    ]
    .into_iter()
    .filter(|flag| *flag)
    .count();
    if sources != 1 {
        return Err(msg(format!(
            "ONNX initializer {name} needs exactly one data source (ONNX-001)"
        )));
    }
    let (file_offset, payload) = if let Some((at, data)) = raw {
        (at, Payload::Bytes(data))
    } else if location_external {
        let map: std::collections::BTreeMap<_, _> = external.into_iter().collect();
        let location = map.get("location").ok_or_else(|| {
            msg(format!("ONNX initializer {name} external data has no location"))
        })?;
        if location.split(['/', '\\']).any(|part| part == "..") {
            return Err(msg(format!(
                "ONNX external data for {name} leaves the model directory (ONNX-003)"
            )));
        }
        let file = relative_child(path.parent().unwrap_or_else(|| Path::new(".")), location)?;
        let offset: u64 = map.get("offset").map(|text| text.parse()).transpose().map_err(|_| {
            msg(format!("ONNX external offset for {name} is not an integer"))
        })?.unwrap_or(0);
        let length: Option<u64> = match map.get("length") {
            Some(text) => Some(text.parse().map_err(|_| {
                msg(format!("ONNX external length for {name} is not an integer"))
            })?),
            None => None,
        };
        if offset % 4096 != 0 {
            notes.push(format!(
                "ONNX external data for {name} is not 4096-byte aligned (ONNX-004)"
            ));
        }
        let file_len = std::fs::metadata(&file).map_err(|err| io_err(&file, err))?.len();
        let len = length.unwrap_or(file_len.saturating_sub(offset));
        (
            offset,
            Payload::Range {
                path: file,
                offset,
                len,
            },
        )
    } else {
        let bytes = pack_typed(code, float_data, int32_data, int64_data, double_data, uint64_data)?;
        (0, Payload::Bytes(bytes))
    };
    Ok(Found {
        name,
        dtype,
        shape: dims,
        payload,
        file_offset,
    })
}

fn pack_typed(
    code: u64,
    float_data: Option<Vec<u8>>,
    int32_data: Option<Vec<u8>>,
    int64_data: Option<Vec<u8>>,
    double_data: Option<Vec<u8>>,
    uint64_data: Option<Vec<u8>>,
) -> Result<Vec<u8>> {
    if matches!(code, 1 | 14) {
        return float_data.ok_or_else(|| msg("ONNX FLOAT data is missing float_data"));
    }
    if matches!(code, 11 | 15) {
        return double_data.ok_or_else(|| msg("ONNX DOUBLE data is missing double_data"));
    }
    if matches!(code, 7) {
        return int64_data.ok_or_else(|| msg("ONNX INT64 data is missing int64_data"));
    }
    if matches!(code, 12 | 13) {
        return uint64_data.ok_or_else(|| msg("ONNX UINT64 data is missing uint64_data"));
    }
    let packed = int32_data.ok_or_else(|| msg("ONNX typed data is missing int32_data"))?;
    let width = match code {
        2 | 3 | 9 => 1,
        4 | 5 | 10 | 16 | 17 | 18 | 19 | 20 | 24 => 2,
        6 => 4,
        _ => {
            return Err(msg(format!(
                "ONNX data type {code} is not read from a typed field"
            )))
        }
    };
    if width == 4 {
        return Ok(packed);
    }
    if packed.len() % 4 != 0 {
        return Err(msg("ONNX int32_data is not a sequence of int32"));
    }
    let mut out = Vec::with_capacity(packed.len() / 4 * width);
    for chunk in packed.chunks_exact(4) {
        let value = u32::from_le_bytes(chunk.try_into().unwrap());
        if width == 1 {
            out.push(value as u8);
        } else {
            out.extend_from_slice(&(value as u16).to_le_bytes());
        }
    }
    Ok(out)
}

fn onnx_dtype(code: u64) -> Result<String> {
    let name = match code {
        1 => "float32",
        2 => "uint8",
        3 => "int8",
        4 => "uint16",
        5 => "int16",
        6 => "int32",
        7 => "int64",
        9 => "bool",
        10 => "float16",
        11 => "float64",
        12 => "uint32",
        13 => "uint64",
        14 => "complex64",
        15 => "complex128",
        16 => "bfloat16",
        17 => "float8_e4m3fn",
        18 => "float8_e4m3fnuz",
        19 => "float8_e5m2",
        20 => "float8_e5m2fnuz",
        21 => "uint4",
        22 => "int4",
        23 => "float4_e2m1fn",
        24 => "float8_e8m0fnu",
        25 => "uint2",
        26 => "int2",
        27 => "float6_e2m3fn",
        28 => "float6_e3m2fn",
        0 | 8 => {
            return Err(msg(format!(
                "ONNX data type {code} is not a numeric tensor"
            )))
        }
        other => {
            return Err(msg(format!("unknown ONNX data type {other}")))
        }
    };
    Ok(name.to_string())
}

fn string_pair(bytes: &[u8]) -> Result<Option<(String, String)>> {
    let mut key = None;
    let mut value = None;
    for field in fields(bytes, 0)? {
        if let Field::Bytes { id, data, .. } = field {
            let text = String::from_utf8(data).map_err(|_| msg("ONNX string field is not utf-8"))?;
            if id == 1 {
                key = Some(text);
            } else if id == 2 {
                value = Some(text);
            }
        }
    }
    match (key, value) {
        (Some(key), Some(value)) => Ok(Some((key, value))),
        _ => Ok(None),
    }
}

enum Field {
    Varint { id: u64, value: u64 },
    Bytes { id: u64, data: Vec<u8>, at: u64 },
}

fn fields(bytes: &[u8], base: u64) -> Result<Vec<Field>> {
    let mut out = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        let (key, next) = varint_at(bytes, index)?;
        index = next;
        let id = key >> 3;
        let wire = key & 7;
        match wire {
            0 => {
                let (value, next) = varint_at(bytes, index)?;
                index = next;
                out.push(Field::Varint { id, value });
            }
            1 => {
                index = index.checked_add(8).ok_or_else(|| msg("ONNX protobuf is truncated"))?;
            }
            2 => {
                let (len, next) = varint_at(bytes, index)?;
                index = next;
                let len = usize::try_from(len).map_err(|_| msg("ONNX protobuf field is too large"))?;
                let end = index.checked_add(len).ok_or_else(|| msg("ONNX protobuf is truncated"))?;
                if end > bytes.len() {
                    return Err(msg("ONNX protobuf is truncated"));
                }
                let at = base + index as u64;
                out.push(Field::Bytes {
                    id,
                    data: bytes[index..end].to_vec(),
                    at,
                });
                index = end;
            }
            5 => {
                index = index.checked_add(4).ok_or_else(|| msg("ONNX protobuf is truncated"))?;
            }
            other => return Err(msg(format!("ONNX protobuf wire type {other} is not supported"))),
        }
    }
    Ok(out)
}

fn varint_at(bytes: &[u8], mut index: usize) -> Result<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let byte = *bytes.get(index).ok_or_else(|| msg("ONNX protobuf is truncated"))?;
        index += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Ok((value, index));
        }
        shift += 7;
        if shift > 63 {
            return Err(msg("ONNX protobuf varint is too long"));
        }
    }
}

pub struct OutTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub payload: Payload,
}

pub fn write_archive(path: &Path, metadata: &[(String, String)], tensors: &[OutTensor]) -> Result<()> {
    if tensors.is_empty() {
        return Err(msg("an ONNX file needs at least one initializer"));
    }
    let mut initializers = Vec::new();
    for tensor in tensors {
        let canonical = dtype::canonical_id(&tensor.dtype)?;
        let code = onnx_code(canonical)?;
        let raw = tensor.payload.read_vec()?;
        let expected = canonical_nbytes(canonical, &tensor.shape)?;
        if raw.len() as u64 != expected {
            return Err(msg(format!(
                "tensor {} is {} bytes; {canonical} needs {expected}",
                tensor.name,
                raw.len()
            )));
        }
        let mut body = Vec::new();
        for dim in &tensor.shape {
            varint_field(&mut body, 1, *dim);
        }
        varint_field(&mut body, 2, code);
        bytes_field(&mut body, 8, tensor.name.as_bytes());
        bytes_field(&mut body, 9, &raw);
        bytes_field(&mut initializers, 5, &body);
    }
    let mut graph = Vec::new();
    bytes_field(&mut graph, 2, b"weights");
    graph.extend_from_slice(&initializers);
    let mut model = Vec::new();
    varint_field(&mut model, 1, 10);
    bytes_field(&mut model, 7, &graph);
    let mut opset = Vec::new();
    bytes_field(&mut opset, 1, b"");
    varint_field(&mut opset, 2, 21);
    bytes_field(&mut model, 8, &opset);
    for (key, value) in metadata {
        let mut pair = Vec::new();
        bytes_field(&mut pair, 1, key.as_bytes());
        bytes_field(&mut pair, 2, value.as_bytes());
        bytes_field(&mut model, 14, &pair);
    }
    crate::io_util::atomic_write(path, |file: &mut File| {
        file.write_all(&model)?;
        Ok(())
    })
}

fn onnx_code(dtype: &str) -> Result<u64> {
    Ok(match dtype {
        "float32" => 1,
        "uint8" => 2,
        "int8" => 3,
        "uint16" => 4,
        "int16" => 5,
        "int32" => 6,
        "int64" => 7,
        "bool" => 9,
        "float16" => 10,
        "float64" => 11,
        "uint32" => 12,
        "uint64" => 13,
        "complex64" => 14,
        "complex128" => 15,
        "bfloat16" => 16,
        "float8_e4m3fn" => 17,
        "float8_e4m3fnuz" => 18,
        "float8_e5m2" => 19,
        "float8_e5m2fnuz" => 20,
        "uint4" => 21,
        "int4" => 22,
        "float4_e2m1fn" => 23,
        "float8_e8m0fnu" => 24,
        "uint2" => 25,
        "int2" => 26,
        "float6_e2m3fn" => 27,
        "float6_e3m2fn" => 28,
        other => return Err(msg(format!("dtype {other} has no ONNX data type"))),
    })
}

fn varint_field(out: &mut Vec<u8>, field: u64, value: u64) {
    write_varint(out, field << 3);
    write_varint(out, value);
}

fn bytes_field(out: &mut Vec<u8>, field: u64, data: &[u8]) {
    write_varint(out, (field << 3) | 2);
    write_varint(out, data.len() as u64);
    out.extend_from_slice(data);
}

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}
