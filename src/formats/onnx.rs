//! ONNX models, read as graph initializers.
//!
//! The reader walks the protobuf for `ModelProto.graph.initializer` and does
//! not load a protobuf code generator. Graph nodes are not tensors and are
//! left in the file. Remove this file and the `onnx` line in
//! `local/supported-formats.txt` to drop the format.

use std::collections::BTreeSet;
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
    let len = std::fs::metadata(path).map_err(|err| io_err(path, err))?.len();
    if len >= 2 * 1024 * 1024 * 1024 {
        return Err(msg(format!(
            "{} is at least 2 GiB; ONNX requires external data (ONNX-006)",
            path.display()
        )));
    }
    let bytes = std::fs::read(path).map_err(|err| io_err(path, err))?;
    let top = fields(&bytes, 0)?;
    // ONNX-007: TensorProto.data_type is field 2 as a varint; ModelProto
    // field 2 is producer_name, a string.
    let is_tensor = top
        .iter()
        .any(|field| matches!(field, Field::Varint { id: 2, .. }));
    let is_model = top
        .iter()
        .any(|field| matches!(field, Field::Bytes { id: 7, .. }));
    let (version, metadata, found, mut notes) = if is_tensor {
        let mut notes = Vec::new();
        let mut tensor = parse_tensor(&bytes, 0, path, &mut notes)?;
        if tensor.name.is_empty() {
            tensor.name = file_stem(path);
        }
        ("TensorProto".to_string(), Vec::new(), vec![tensor], notes)
    } else if is_model {
        let (ir, metadata, found, notes) = parse_model(&top, path)?;
        (format!("ir {ir}"), metadata, found, notes)
    } else {
        return Err(msg(format!(
            "{} is neither an ONNX ModelProto nor a TensorProto (ONNX-007); pass --format to name it",
            path.display()
        )));
    };
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
        version,
        path: path.to_path_buf(),
        members,
        metadata,
        data_bytes,
        notes,
        inner: Inner::Onnx,
    })
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("tensor")
        .to_string()
}

fn parse_model(top: &[Field], path: &Path) -> Result<(u64, Vec<(String, String)>, Vec<Found>, Vec<String>)> {
    let mut ir = None;
    let mut metadata = Vec::new();
    let mut found = Vec::new();
    let mut notes = Vec::new();
    for field in top {
        match field {
            Field::Varint { id: 1, value } => ir = Some(*value),
            Field::Bytes { id: 7, data, at } => {
                parse_graph(data, *at, path, &mut found, &mut notes)?;
            }
            Field::Bytes { id: 14, data, .. } => {
                if let Some((key, value)) = string_pair(data)? {
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

fn parse_graph(
    bytes: &[u8],
    base: u64,
    path: &Path,
    found: &mut Vec<Found>,
    notes: &mut Vec<String>,
) -> Result<()> {
    for field in fields(bytes, base)? {
        match field {
            Field::Bytes { id: 5, data, at } => {
                found.push(parse_tensor(data, at, path, notes)?);
            }
            Field::Bytes { id: 15, .. } => {
                return Err(msg(
                    "ONNX sparse initializers are not supported by tenspile",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Values of one repeated scalar field. protobuf allows the packed form
/// (one length-delimited chunk, possibly several) and the unpacked form (one
/// field per value); both are accepted and concatenated in order.
#[derive(Default)]
struct Repeated {
    varints: Vec<u64>,
    fixed: Vec<u8>,
    present: bool,
}

impl Repeated {
    fn add_packed_varints(&mut self, data: &[u8]) -> Result<()> {
        self.present = true;
        let mut index = 0;
        while index < data.len() {
            let (value, next) = varint_at(data, index)?;
            self.varints.push(value);
            index = next;
        }
        Ok(())
    }
}

fn parse_tensor(bytes: &[u8], base: u64, path: &Path, notes: &mut Vec<String>) -> Result<Found> {
    let mut dims = Repeated::default();
    let mut data_type = None;
    let mut name = None;
    let mut raw: Option<(u64, u64)> = None;
    let mut float_data = Repeated::default();
    let mut int32_data = Repeated::default();
    let mut int64_data = Repeated::default();
    let mut double_data = Repeated::default();
    let mut uint64_data = Repeated::default();
    let mut external = Vec::new();
    let mut location_external = false;
    for field in fields(bytes, base)? {
        match field {
            Field::Varint { id: 1, value } => {
                dims.present = true;
                dims.varints.push(value);
            }
            Field::Bytes { id: 1, data, .. } => dims.add_packed_varints(data)?,
            Field::Varint { id: 2, value } => data_type = Some(value),
            Field::Bytes { id: 8, data, .. } => {
                name = Some(
                    String::from_utf8(data.to_vec())
                        .map_err(|_| msg("ONNX tensor name is not utf-8"))?,
                );
            }
            Field::Bytes { id: 9, data, at } => raw = Some((at, data.len() as u64)),
            Field::Bytes { id: 4, data, .. } => {
                float_data.present = true;
                float_data.fixed.extend_from_slice(data);
            }
            Field::Fixed { id: 4, data } if data.len() == 4 => {
                float_data.present = true;
                float_data.fixed.extend_from_slice(data);
            }
            Field::Bytes { id: 10, data, .. } => {
                double_data.present = true;
                double_data.fixed.extend_from_slice(data);
            }
            Field::Fixed { id: 10, data } if data.len() == 8 => {
                double_data.present = true;
                double_data.fixed.extend_from_slice(data);
            }
            Field::Bytes { id: 5, data, .. } => int32_data.add_packed_varints(data)?,
            Field::Varint { id: 5, value } => {
                int32_data.present = true;
                int32_data.varints.push(value);
            }
            Field::Bytes { id: 7, data, .. } => int64_data.add_packed_varints(data)?,
            Field::Varint { id: 7, value } => {
                int64_data.present = true;
                int64_data.varints.push(value);
            }
            Field::Bytes { id: 11, data, .. } => uint64_data.add_packed_varints(data)?,
            Field::Varint { id: 11, value } => {
                uint64_data.present = true;
                uint64_data.varints.push(value);
            }
            Field::Bytes { id: 13, data, .. } => {
                if let Some(pair) = string_pair(data)? {
                    external.push(pair);
                }
            }
            Field::Bytes { id: 3, .. } => {
                return Err(msg("ONNX tensor segments are not supported by tenspile"));
            }
            Field::Varint { id: 14, value } if value == 1 => location_external = true,
            _ => {}
        }
    }
    let code = data_type.ok_or_else(|| msg("ONNX initializer has no data_type"))?;
    let dtype = onnx_dtype(code)?;
    let name = name.unwrap_or_default();
    let typed = [&float_data, &int32_data, &int64_data, &double_data, &uint64_data]
        .iter()
        .any(|field| field.present);
    let sources = [raw.is_some(), typed, location_external]
        .into_iter()
        .filter(|flag| *flag)
        .count();
    if sources != 1 {
        return Err(msg(format!(
            "ONNX initializer {name} needs exactly one data source (ONNX-001)"
        )));
    }
    let (file_offset, payload) = if let Some((at, len)) = raw {
        (
            at,
            Payload::Range {
                path: path.to_path_buf(),
                offset: at,
                len,
            },
        )
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
        shape: dims.varints,
        payload,
        file_offset,
    })
}

/// ONNX-008: little-endian element bytes from the typed fields.
fn pack_typed(
    code: u64,
    float_data: Repeated,
    int32_data: Repeated,
    int64_data: Repeated,
    double_data: Repeated,
    uint64_data: Repeated,
) -> Result<Vec<u8>> {
    let missing = |field: &str| msg(format!("ONNX data type {code} is missing {field}"));
    match code {
        // FLOAT and COMPLEX64 (pairs of floats): packed fixed32.
        1 | 14 => float_data.present.then_some(float_data.fixed).ok_or_else(|| missing("float_data")),
        // DOUBLE and COMPLEX128: packed fixed64.
        11 | 15 => double_data
            .present
            .then_some(double_data.fixed)
            .ok_or_else(|| missing("double_data")),
        7 => {
            if !int64_data.present {
                return Err(missing("int64_data"));
            }
            Ok(int64_data.varints.iter().flat_map(|value| value.to_le_bytes()).collect())
        }
        13 => {
            if !uint64_data.present {
                return Err(missing("uint64_data"));
            }
            Ok(uint64_data.varints.iter().flat_map(|value| value.to_le_bytes()).collect())
        }
        12 => {
            if !uint64_data.present {
                return Err(missing("uint64_data"));
            }
            let mut out = Vec::with_capacity(uint64_data.varints.len() * 4);
            for value in uint64_data.varints {
                let narrow = u32::try_from(value)
                    .map_err(|_| msg("ONNX UINT32 value does not fit in 32 bits"))?;
                out.extend_from_slice(&narrow.to_le_bytes());
            }
            Ok(out)
        }
        _ => {
            // One int32_data entry per element, narrowed to the element width.
            let width = match code {
                2 | 3 | 9 | 17 | 18 | 19 | 20 | 24 => 1,
                4 | 5 | 10 | 16 => 2,
                6 => 4,
                _ => {
                    return Err(msg(format!(
                        "ONNX data type {code} is not read from a typed field by tenspile"
                    )))
                }
            };
            if !int32_data.present {
                return Err(missing("int32_data"));
            }
            let mut out = Vec::with_capacity(int32_data.varints.len() * width);
            for value in int32_data.varints {
                // int32 is sign-extended to 64 bits in the varint.
                let low = (value as u32).to_le_bytes();
                out.extend_from_slice(&low[..width]);
            }
            Ok(out)
        }
    }
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
            let text = String::from_utf8(data.to_vec()).map_err(|_| msg("ONNX string field is not utf-8"))?;
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

enum Field<'a> {
    Varint { id: u64, value: u64 },
    /// Length-delimited data and its file offset.
    Bytes { id: u64, data: &'a [u8], at: u64 },
    /// fixed32 or fixed64.
    Fixed { id: u64, data: &'a [u8] },
}

fn fields(bytes: &[u8], base: u64) -> Result<Vec<Field<'_>>> {
    let mut out = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        let (key, next) = varint_at(bytes, index)?;
        index = next;
        let id = key >> 3;
        let wire = key & 7;
        let fixed = |index: usize, width: usize| -> Result<(&[u8], usize)> {
            let end = index
                .checked_add(width)
                .filter(|end| *end <= bytes.len())
                .ok_or_else(|| msg("ONNX protobuf is truncated"))?;
            Ok((&bytes[index..end], end))
        };
        match wire {
            0 => {
                let (value, next) = varint_at(bytes, index)?;
                index = next;
                out.push(Field::Varint { id, value });
            }
            1 => {
                let (data, next) = fixed(index, 8)?;
                index = next;
                out.push(Field::Fixed { id, data });
            }
            2 => {
                let (len, next) = varint_at(bytes, index)?;
                index = next;
                let len = usize::try_from(len).map_err(|_| msg("ONNX protobuf field is too large"))?;
                let (data, end) = fixed(index, len)?;
                out.push(Field::Bytes {
                    id,
                    data,
                    at: base + index as u64,
                });
                index = end;
            }
            5 => {
                let (data, next) = fixed(index, 4)?;
                index = next;
                out.push(Field::Fixed { id, data });
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

/// A new model holding only initializers, as onnx.yaml `write` builds it:
/// graph `weights`, one initializer and one graph output per tensor, and
/// opset 21.
pub fn write_archive(path: &Path, metadata: &[(String, String)], tensors: &[OutTensor]) -> Result<()> {
    if tensors.is_empty() {
        return Err(msg("writing an ONNX file with no initializers is not supported"));
    }
    crate::io_util::check_unique_names(tensors.iter().map(|tensor| tensor.name.as_str()))?;
    let mut initializers = Vec::new();
    let mut outputs = Vec::new();
    let mut ir_version = 10;
    for tensor in tensors {
        let (body, code) = encode_tensor(tensor)?;
        ir_version = ir_version.max(min_ir(code));
        bytes_field(&mut initializers, 5, &body);
        bytes_field(&mut outputs, 12, &value_info(tensor, code));
    }
    let mut graph = Vec::new();
    bytes_field(&mut graph, 2, b"weights");
    graph.extend_from_slice(&initializers);
    graph.extend_from_slice(&outputs);
    let mut model = Vec::new();
    varint_field(&mut model, 1, ir_version);
    bytes_field(&mut model, 7, &graph);
    let mut opset = Vec::new();
    bytes_field(&mut opset, 1, b"");
    varint_field(&mut opset, 2, 21);
    bytes_field(&mut model, 8, &opset);
    model.extend_from_slice(&metadata_fields(metadata));
    finish(path, &model)
}

/// Rewrite an existing model with a new set of initializers and metadata
/// (ONNX-009 to ONNX-013). Every other field is copied as stored, unknown
/// fields included, and so is an initializer named in `unchanged`, with its
/// typed fields or external data reference. A replaced or added initializer
/// is written with `raw_data`, and `ir_version` is raised only when its
/// dtype needs a newer IR. A removed initializer also leaves the graph
/// inputs, outputs, and value_info. Removing one that a node reads, in the
/// graph or in a subgraph, that a training binding names, or that a
/// quantization annotation names, is refused. Returns notes on what changed
/// besides the edit.
pub fn rewrite_archive(
    source: &Path,
    path: &Path,
    metadata: &[(String, String)],
    tensors: &[OutTensor],
    unchanged: &BTreeSet<String>,
) -> Result<Vec<String>> {
    if tensors.is_empty() {
        return Err(msg("writing an ONNX file with no initializers is not supported"));
    }
    crate::io_util::check_unique_names(tensors.iter().map(|tensor| tensor.name.as_str()))?;
    let bytes = std::fs::read(source).map_err(|err| io_err(source, err))?;
    let keep: BTreeSet<&str> = tensors.iter().map(|tensor| tensor.name.as_str()).collect();
    let removed: BTreeSet<String> = initializer_names(&bytes)?
        .into_iter()
        .filter(|name| !keep.contains(name.as_str()))
        .collect();
    check_removals(&bytes, &removed)?;
    let mut notes = Vec::new();
    let mut written = BTreeSet::new();
    let mut external = BTreeSet::new();
    let mut ir_version = None;
    let mut needed_ir = 0;
    let mut model = Vec::new();
    let mut metadata_written = false;
    for field in fields(&bytes, 0)? {
        match field {
            Field::Varint { id: 1, value } => ir_version = Some(value),
            Field::Bytes { id: 7, data, .. } => {
                let mut graph = Vec::new();
                for item in fields(data, 0)? {
                    match item {
                        Field::Bytes { id: 5, data, .. } => {
                            let name = tensor_name(data)?;
                            if !keep.contains(name.as_str()) {
                                continue;
                            }
                            if unchanged.contains(&name) {
                                if let Some(location) = external_location(data)? {
                                    external.insert(location);
                                }
                                bytes_field(&mut graph, 5, data);
                            } else {
                                let tensor = tensors.iter().find(|tensor| tensor.name == name).unwrap();
                                let (body, code) = encode_tensor(tensor)?;
                                needed_ir = needed_ir.max(dtype_ir(code));
                                bytes_field(&mut graph, 5, &body);
                            }
                            written.insert(name);
                        }
                        // A graph input, output, or value_info naming a removed
                        // initializer goes too (ONNX-010, ONNX-012).
                        Field::Bytes { id: 11..=13, data, .. }
                            if removed.contains(&value_info_name(data)?) => {}
                        other => copy_field(&mut graph, &other),
                    }
                }
                for tensor in tensors.iter().filter(|tensor| !written.contains(&tensor.name)) {
                    let (body, code) = encode_tensor(tensor)?;
                    needed_ir = needed_ir.max(dtype_ir(code));
                    bytes_field(&mut graph, 5, &body);
                }
                bytes_field(&mut model, 7, &graph);
            }
            Field::Bytes { id: 14, .. } => {
                if !metadata_written {
                    model.extend_from_slice(&metadata_fields(metadata));
                    metadata_written = true;
                }
            }
            other => copy_field(&mut model, &other),
        }
    }
    if !metadata_written {
        model.extend_from_slice(&metadata_fields(metadata));
    }
    // ir_version and opset_import stay as stored; the checker rejects a
    // data type newer than the model's IR (ONNX-009, ONNX-013).
    if needed_ir > ir_version.unwrap_or(0) {
        notes.push(format!(
            "raised ir_version from {} to {needed_ir} for the new initializers' data types (ONNX-009)",
            ir_version.map_or("none".to_string(), |value| value.to_string())
        ));
        ir_version = Some(needed_ir);
    }
    let mut out = Vec::with_capacity(model.len() + 4);
    if let Some(value) = ir_version {
        varint_field(&mut out, 1, value);
    }
    out.extend_from_slice(&model);
    // External data the kept initializers name must still resolve beside the
    // new model (ONNX-003, ONNX-009).
    let from = source.parent().unwrap_or_else(|| Path::new("."));
    let to = path.parent().unwrap_or_else(|| Path::new("."));
    if !external.is_empty() && !same_dir(from, to) {
        for location in &external {
            let target = relative_child(to, location)?;
            if target.exists() {
                return Err(msg(format!(
                    "{} exists; the edited model would reference it in place of {}'s external data",
                    target.display(),
                    source.display()
                )));
            }
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(|err| io_err(parent, err))?;
            }
            let origin = relative_child(from, location)?;
            std::fs::copy(&origin, &target).map_err(|err| io_err(&origin, err))?;
            notes.push(format!("copied external data {location} beside the new model (ONNX-009)"));
        }
    }
    finish(path, &out)?;
    Ok(notes)
}

fn same_dir(left: &Path, right: &Path) -> bool {
    let resolve = |dir: &Path| {
        let dir = if dir.as_os_str().is_empty() { Path::new(".") } else { dir };
        dir.canonicalize().ok()
    };
    matches!((resolve(left), resolve(right)), (Some(left), Some(right)) if left == right)
}

/// Refuse to remove an initializer that the model still names: a node input
/// in the graph or in any nested subgraph, a training binding key
/// (ONNX-011), or a quantization annotation (ONNX-012).
fn check_removals(model: &[u8], removed: &BTreeSet<String>) -> Result<()> {
    if removed.is_empty() {
        return Ok(());
    }
    for field in fields(model, 0)? {
        match field {
            Field::Bytes { id: 7, data, .. } => {
                for item in fields(data, 0)? {
                    match item {
                        Field::Bytes { id: 1, data, .. } => {
                            let mut reads = BTreeSet::new();
                            node_reads(data, &mut reads)?;
                            if let Some(name) = reads.intersection(removed).next() {
                                return Err(msg(format!(
                                    "initializer {name} is an input of a graph node; deleting it would break the model (ONNX-011)"
                                )));
                            }
                        }
                        // TensorAnnotation: tensor_name (1), and
                        // quant_parameter_tensor_names (2) entries.
                        Field::Bytes { id: 14, data, .. } => {
                            for entry in fields(data, 0)? {
                                let named = match entry {
                                    Field::Bytes { id: 1, data, .. } => Some(text(data)?),
                                    Field::Bytes { id: 2, data, .. } => string_pair(data)?.map(|(_, value)| value),
                                    _ => None,
                                };
                                if let Some(name) = named.filter(|name| removed.contains(name)) {
                                    return Err(msg(format!(
                                        "initializer {name} is named by a quantization annotation; deleting it would lose that data (ONNX-012)"
                                    )));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            // TrainingInfoProto: initialization_binding (3) and
            // update_binding (4) keys name initializers.
            Field::Bytes { id: 20, data, .. } => {
                for entry in fields(data, 0)? {
                    if let Field::Bytes { id: 3 | 4, data, .. } = entry {
                        if let Some((name, _)) = string_pair(data)?.filter(|(key, _)| removed.contains(key)) {
                            return Err(msg(format!(
                                "initializer {name} is bound by the model's training_info; deleting it would break the model (ONNX-011)"
                            )));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Names a node reads: its inputs, and what the graphs in its attributes
/// (`g`, `graphs`) read from outer scopes, recursively (ONNX-011).
fn node_reads(node: &[u8], out: &mut BTreeSet<String>) -> Result<()> {
    for field in fields(node, 0)? {
        match field {
            Field::Bytes { id: 1, data, .. } => {
                out.insert(text(data)?);
            }
            Field::Bytes { id: 5, data, .. } => {
                for attribute in fields(data, 0)? {
                    if let Field::Bytes { id: 6 | 11, data, .. } = attribute {
                        subgraph_reads(data, out)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Names a subgraph's nodes read that the subgraph does not define itself
/// as an input, an initializer, or a node output.
fn subgraph_reads(graph: &[u8], out: &mut BTreeSet<String>) -> Result<()> {
    let mut defined = BTreeSet::new();
    let mut reads = BTreeSet::new();
    for field in fields(graph, 0)? {
        match field {
            Field::Bytes { id: 1, data, .. } => {
                node_reads(data, &mut reads)?;
                for item in fields(data, 0)? {
                    if let Field::Bytes { id: 2, data, .. } = item {
                        defined.insert(text(data)?);
                    }
                }
            }
            Field::Bytes { id: 5, data, .. } => {
                defined.insert(tensor_name(data)?);
            }
            Field::Bytes { id: 11, data, .. } => {
                defined.insert(value_info_name(data)?);
            }
            _ => {}
        }
    }
    out.extend(reads.difference(&defined).cloned());
    Ok(())
}

/// The `location` of an initializer stored as external data.
fn external_location(tensor: &[u8]) -> Result<Option<String>> {
    let mut external = false;
    let mut location = None;
    for field in fields(tensor, 0)? {
        match field {
            Field::Varint { id: 14, value: 1 } => external = true,
            Field::Bytes { id: 13, data, .. } => {
                if let Some((key, value)) = string_pair(data)? {
                    if key == "location" {
                        location = Some(value);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(location.filter(|_| external))
}

fn finish(path: &Path, model: &[u8]) -> Result<()> {
    if model.len() as u64 >= 2 * 1024 * 1024 * 1024 {
        return Err(msg(
            "the ONNX model would be 2 GiB or more; tenspile does not write external data (ONNX-006)",
        ));
    }
    crate::io_util::atomic_write(path, |file: &mut File| {
        file.write_all(model)?;
        Ok(())
    })
}

/// TensorProto with `raw_data`, and its ONNX data type code.
fn encode_tensor(tensor: &OutTensor) -> Result<(Vec<u8>, u64)> {
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
    Ok((body, code))
}

/// ValueInfoProto for a graph output: name, and a tensor type with the
/// element type and every dimension as `dim_value`.
fn value_info(tensor: &OutTensor, code: u64) -> Vec<u8> {
    let mut shape = Vec::new();
    for dim in &tensor.shape {
        let mut dimension = Vec::new();
        varint_field(&mut dimension, 1, *dim);
        bytes_field(&mut shape, 1, &dimension);
    }
    let mut tensor_type = Vec::new();
    varint_field(&mut tensor_type, 1, code);
    bytes_field(&mut tensor_type, 2, &shape);
    let mut type_proto = Vec::new();
    bytes_field(&mut type_proto, 1, &tensor_type);
    let mut info = Vec::new();
    bytes_field(&mut info, 1, tensor.name.as_bytes());
    bytes_field(&mut info, 2, &type_proto);
    info
}

fn metadata_fields(metadata: &[(String, String)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (key, value) in metadata {
        let mut pair = Vec::new();
        bytes_field(&mut pair, 1, key.as_bytes());
        bytes_field(&mut pair, 2, value.as_bytes());
        bytes_field(&mut out, 14, &pair);
    }
    out
}

fn copy_field(out: &mut Vec<u8>, field: &Field) {
    match field {
        Field::Varint { id, value } => varint_field(out, *id, *value),
        Field::Bytes { id, data, .. } => bytes_field(out, *id, data),
        Field::Fixed { id, data } => {
            let wire = if data.len() == 8 { 1 } else { 5 };
            write_varint(out, (id << 3) | wire);
            out.extend_from_slice(data);
        }
    }
}

fn text(data: &[u8]) -> Result<String> {
    String::from_utf8(data.to_vec()).map_err(|_| msg("ONNX string is not utf-8"))
}

fn tensor_name(tensor: &[u8]) -> Result<String> {
    for field in fields(tensor, 0)? {
        if let Field::Bytes { id: 8, data, .. } = field {
            return text(data);
        }
    }
    Err(msg("ONNX initializer has no name"))
}

fn value_info_name(info: &[u8]) -> Result<String> {
    for field in fields(info, 0)? {
        if let Field::Bytes { id: 1, data, .. } = field {
            return text(data);
        }
    }
    Ok(String::new())
}

fn initializer_names(model: &[u8]) -> Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    for field in fields(model, 0)? {
        if let Field::Bytes { id: 7, data, .. } = field {
            for item in fields(data, 0)? {
                if let Field::Bytes { id: 5, data, .. } = item {
                    names.insert(tensor_name(data)?);
                }
            }
        }
    }
    Ok(names)
}

/// Lowest IR version for a new file. The writer never goes below IR 10.
fn min_ir(code: u64) -> u64 {
    dtype_ir(code).max(10)
}

/// The IR version that introduced a data type (onnx.yaml `versions`), or 0
/// for the types every IR has.
fn dtype_ir(code: u64) -> u64 {
    match code {
        16 => 4,
        17..=20 => 9,
        21 | 22 => 10,
        23 => 11,
        24 => 12,
        25 | 26 => 13,
        27 | 28 => 14,
        _ => 0,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A model with initializers `w` and `b`, a graph output for each, and an
    /// Identity node that reads `w`.
    fn model_with_node(path: &Path) {
        let tensor = |name: &str| {
            let mut body = Vec::new();
            varint_field(&mut body, 1, 2);
            varint_field(&mut body, 2, 1);
            bytes_field(&mut body, 8, name.as_bytes());
            bytes_field(&mut body, 9, &[0u8; 8]);
            body
        };
        let mut node = Vec::new();
        bytes_field(&mut node, 1, b"w");
        bytes_field(&mut node, 2, b"y");
        bytes_field(&mut node, 4, b"Identity");
        let mut graph = Vec::new();
        bytes_field(&mut graph, 1, &node);
        bytes_field(&mut graph, 2, b"g");
        for name in ["w", "b"] {
            bytes_field(&mut graph, 5, &tensor(name));
            let out = OutTensor {
                name: name.into(),
                dtype: "float32".into(),
                shape: vec![2],
                payload: Payload::Bytes(vec![0; 8]),
            };
            bytes_field(&mut graph, 12, &value_info(&out, 1));
        }
        let mut model = Vec::new();
        varint_field(&mut model, 1, 9);
        bytes_field(&mut model, 2, b"producer");
        bytes_field(&mut model, 7, &graph);
        std::fs::write(path, model).unwrap();
    }

    fn graph_fields(path: &Path) -> Vec<u64> {
        let bytes = std::fs::read(path).unwrap();
        let mut ids = Vec::new();
        for field in fields(&bytes, 0).unwrap() {
            if let Field::Bytes { id: 7, data, .. } = field {
                for item in fields(data, 0).unwrap() {
                    let (Field::Bytes { id, .. } | Field::Varint { id, .. } | Field::Fixed { id, .. }) = item;
                    ids.push(id);
                }
            }
        }
        ids
    }

    #[test]
    fn a_rewrite_keeps_the_graph() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.onnx");
        model_with_node(&path);
        let archive = open(&path).unwrap();
        let out = |names: &[&str]| -> Vec<OutTensor> {
            archive
                .members
                .iter()
                .filter(|member| names.contains(&member.name.as_str()))
                .map(|member| OutTensor {
                    name: member.name.clone(),
                    dtype: member.dtype.clone(),
                    shape: member.shape.clone(),
                    payload: member.payload.clone(),
                })
                .collect()
        };
        let all: BTreeSet<String> = ["w".to_string(), "b".to_string()].into();
        // A metadata change keeps the node, both outputs, and the producer.
        let edited = dir.path().join("meta.onnx");
        rewrite_archive(&path, &edited, &[("k".into(), "v".into())], &out(&["w", "b"]), &all).unwrap();
        assert_eq!(graph_fields(&edited), vec![1, 2, 5, 12, 5, 12]);
        let reread = open(&edited).unwrap();
        assert_eq!(reread.metadata, vec![("k".to_string(), "v".to_string())]);
        assert!(std::fs::read(&edited).unwrap().windows(8).any(|window| window == b"producer"));
        // Deleting b drops its initializer and its graph output.
        let deleted = dir.path().join("del.onnx");
        rewrite_archive(&path, &deleted, &[], &out(&["w"]), &all).unwrap();
        assert_eq!(graph_fields(&deleted), vec![1, 2, 5, 12]);
        // w feeds the node, so deleting it is refused.
        let err = rewrite_archive(&path, &dir.path().join("x.onnx"), &[], &out(&["b"]), &all).unwrap_err();
        assert!(err.to_string().contains("input of a graph node"), "{err}");
        // A new file has one graph output per initializer (onnx.yaml write).
        let fresh = dir.path().join("new.onnx");
        write_archive(&fresh, &[], &out(&["w", "b"])).unwrap();
        assert_eq!(graph_fields(&fresh), vec![2, 5, 5, 12, 12]);
    }

    fn float_tensor(name: &str) -> OutTensor {
        OutTensor {
            name: name.into(),
            dtype: "float32".into(),
            shape: vec![2],
            payload: Payload::Bytes(vec![0; 8]),
        }
    }

    /// An IR 3 model with initializers `w` and `b`, each also a graph input
    /// and a value_info, plus `graph_extra` and `model_extra` fields.
    fn model_with(path: &Path, graph_extra: &[u8], model_extra: &[u8]) {
        let mut graph = Vec::new();
        bytes_field(&mut graph, 2, b"g");
        for name in ["w", "b"] {
            let (body, _) = encode_tensor(&float_tensor(name)).unwrap();
            bytes_field(&mut graph, 5, &body);
            bytes_field(&mut graph, 11, &value_info(&float_tensor(name), 1));
            bytes_field(&mut graph, 13, &value_info(&float_tensor(name), 1));
        }
        graph.extend_from_slice(graph_extra);
        let mut model = Vec::new();
        varint_field(&mut model, 1, 3);
        bytes_field(&mut model, 7, &graph);
        model.extend_from_slice(model_extra);
        std::fs::write(path, model).unwrap();
    }

    fn delete_b(path: &Path, dest: &Path) -> Result<Vec<String>> {
        let all: BTreeSet<String> = ["w".to_string(), "b".to_string()].into();
        rewrite_archive(path, dest, &[], &[float_tensor("w")], &all)
    }

    #[test]
    fn deleting_an_initializer_removes_its_input_and_value_info() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.onnx");
        model_with(&path, &[], &[]);
        let dest = dir.path().join("d.onnx");
        delete_b(&path, &dest).unwrap();
        assert_eq!(graph_fields(&dest), vec![2, 5, 11, 13]);
        // IR 3 stays IR 3 when nothing needs a newer one (ONNX-013).
        let bytes = std::fs::read(&dest).unwrap();
        assert!(matches!(fields(&bytes, 0).unwrap()[0], Field::Varint { id: 1, value: 3 }));
    }

    #[test]
    fn a_new_dtype_raises_ir_version_only_as_far_as_it_needs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.onnx");
        model_with(&path, &[], &[]);
        let dest = dir.path().join("bf.onnx");
        let unchanged: BTreeSet<String> = ["w".to_string()].into();
        let bf16 = OutTensor {
            name: "b".into(),
            dtype: "bfloat16".into(),
            shape: vec![2],
            payload: Payload::Bytes(vec![0; 4]),
        };
        let notes = rewrite_archive(&path, &dest, &[], &[float_tensor("w"), bf16], &unchanged).unwrap();
        assert!(notes[0].contains("from 3 to 4"), "{notes:?}");
        let bytes = std::fs::read(&dest).unwrap();
        assert!(matches!(fields(&bytes, 0).unwrap()[0], Field::Varint { id: 1, value: 4 }));
    }

    #[test]
    fn subgraphs_training_bindings_and_annotations_block_a_delete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.onnx");
        let dest = dir.path().join("d.onnx");
        // An If node whose then_branch reads b from the outer graph.
        let mut inner = Vec::new();
        bytes_field(&mut inner, 1, b"b");
        bytes_field(&mut inner, 2, b"y");
        bytes_field(&mut inner, 4, b"Identity");
        let mut branch = Vec::new();
        bytes_field(&mut branch, 1, &inner);
        let mut attribute = Vec::new();
        bytes_field(&mut attribute, 1, b"then_branch");
        bytes_field(&mut attribute, 6, &branch);
        let mut node = Vec::new();
        bytes_field(&mut node, 1, b"cond");
        bytes_field(&mut node, 4, b"If");
        bytes_field(&mut node, 5, &attribute);
        let mut graph_extra = Vec::new();
        bytes_field(&mut graph_extra, 1, &node);
        model_with(&path, &graph_extra, &[]);
        let err = delete_b(&path, &dest).unwrap_err().to_string();
        assert!(err.contains("ONNX-011"), "{err}");
        // A subgraph that defines b itself does not read the outer one.
        let mut defines = Vec::new();
        bytes_field(&mut defines, 2, b"g2");
        let (body, _) = encode_tensor(&float_tensor("b")).unwrap();
        bytes_field(&mut defines, 5, &body);
        bytes_field(&mut defines, 1, &inner);
        let mut attribute = Vec::new();
        bytes_field(&mut attribute, 6, &defines);
        let mut node = Vec::new();
        bytes_field(&mut node, 5, &attribute);
        let mut graph_extra = Vec::new();
        bytes_field(&mut graph_extra, 1, &node);
        model_with(&path, &graph_extra, &[]);
        delete_b(&path, &dest).unwrap();
        // A training binding key.
        let mut binding = Vec::new();
        bytes_field(&mut binding, 1, b"b");
        bytes_field(&mut binding, 2, b"b_next");
        let mut training = Vec::new();
        bytes_field(&mut training, 4, &binding);
        let mut model_extra = Vec::new();
        bytes_field(&mut model_extra, 20, &training);
        model_with(&path, &[], &model_extra);
        let err = delete_b(&path, &dest).unwrap_err().to_string();
        assert!(err.contains("training_info"), "{err}");
        // A quantization annotation.
        let mut pair = Vec::new();
        bytes_field(&mut pair, 1, b"SCALE_TENSOR");
        bytes_field(&mut pair, 2, b"b");
        let mut annotation = Vec::new();
        bytes_field(&mut annotation, 1, b"x");
        bytes_field(&mut annotation, 2, &pair);
        let mut graph_extra = Vec::new();
        bytes_field(&mut graph_extra, 14, &annotation);
        model_with(&path, &graph_extra, &[]);
        let err = delete_b(&path, &dest).unwrap_err().to_string();
        assert!(err.contains("ONNX-012"), "{err}");
    }

    #[test]
    fn external_data_is_copied_when_the_model_moves() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.onnx");
        std::fs::write(dir.path().join("w.bin"), [0u8; 8]).unwrap();
        let mut tensor = Vec::new();
        varint_field(&mut tensor, 1, 2);
        varint_field(&mut tensor, 2, 1);
        bytes_field(&mut tensor, 8, b"w");
        let mut location = Vec::new();
        bytes_field(&mut location, 1, b"location");
        bytes_field(&mut location, 2, b"w.bin");
        bytes_field(&mut tensor, 13, &location);
        varint_field(&mut tensor, 14, 1);
        let mut graph = Vec::new();
        bytes_field(&mut graph, 5, &tensor);
        let mut model = Vec::new();
        varint_field(&mut model, 1, 10);
        bytes_field(&mut model, 7, &graph);
        std::fs::write(&path, model).unwrap();
        let archive = open(&path).unwrap();
        let member = &archive.members[0];
        let kept = OutTensor {
            name: "w".into(),
            dtype: member.dtype.clone(),
            shape: member.shape.clone(),
            payload: member.payload.clone(),
        };
        let unchanged: BTreeSet<String> = ["w".to_string()].into();
        let moved = dir.path().join("out/m.onnx");
        std::fs::create_dir_all(moved.parent().unwrap()).unwrap();
        let notes = rewrite_archive(&path, &moved, &[], &[kept], &unchanged).unwrap();
        assert!(notes[0].contains("w.bin"), "{notes:?}");
        let reread = open(&moved).unwrap();
        assert_eq!(reread.members[0].payload.read_vec().unwrap(), vec![0u8; 8]);
    }
}
