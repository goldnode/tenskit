use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use serde_json::{Map, Value};

use crate::dtype::{format_shape, packed_bytes, safe_dtype};
use crate::error::{io_err, msg, Result};
use crate::io_util::atomic_write;
use crate::payload::Payload;

pub const METADATA_KEY: &str = "__metadata__";
const MAX_OFFSET: u64 = 281_474_976_710_655;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub begin: u64,
    pub end: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeFile {
    pub path: std::path::PathBuf,
    pub header_len: u64,
    pub data_start: u64,
    pub data_len: u64,
    pub metadata: Vec<(String, String)>,
    pub tensors: Vec<TensorEntry>,
    pub warnings: Vec<String>,
}

pub fn read_file(path: &Path, max_header: u64) -> Result<SafeFile> {
    let mut file = File::open(path).map_err(|err| io_err(path, err))?;
    let file_len = file.metadata().map_err(|err| io_err(path, err))?.len();
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).map_err(|err| {
        if err.kind() == std::io::ErrorKind::UnexpectedEof {
            msg(format!("{} is too small to be safetensors", path.display()))
        } else {
            io_err(path, err)
        }
    })?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > max_header {
        return Err(msg(format!(
            "{}: safetensors header is {header_len} bytes, over the {max_header} byte cap (raise it with --max-header-size)",
            path.display()
        )));
    }
    if file_len < 8 + header_len {
        return Err(msg(format!(
            "{}: safetensors header length {header_len} runs past the end of the file",
            path.display()
        )));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|err| io_err(path, err))?;
    if header.first() != Some(&b'{') {
        return Err(msg(format!(
            "{}: safetensors header does not start with '{{'",
            path.display()
        )));
    }
    let header_text = std::str::from_utf8(&header).map_err(|_| {
        msg(format!(
            "{}: safetensors header is not utf-8",
            path.display()
        ))
    })?;
    check_duplicate_keys(header_text)?;
    let mut warnings = Vec::new();
    if header_len % 8 != 0 {
        warnings.push(
            "safetensors header length is not a multiple of 8 (ST-016); the file is still read"
                .to_string(),
        );
    }
    let space_trimmed = header_text.trim_end_matches(' ');
    if header_text.trim_end().len() != space_trimmed.len() {
        warnings.push(
            "safetensors header padding is not only spaces (ST-006); the file is still read"
                .to_string(),
        );
    }
    let value: Value = serde_json::from_str(header_text)?;
    let object = value.as_object().ok_or_else(|| {
        msg(format!(
            "{}: safetensors header is not an object",
            path.display()
        ))
    })?;
    let mut parsed = parse_object(path, header_len, file_len, object)?;
    for tensor in &parsed.tensors {
        if tensor.dtype == "F4" {
            warnings.push(
                "F4 nibble order is an open question (ST-Q2); bytes are copied and not interpreted"
                    .to_string(),
            );
        }
        if tensor.dtype == "F6_E2M3" || tensor.dtype == "F6_E3M2" {
            warnings.push(
                "F6 bit packing is an open question (ST-Q1); bytes are copied and not interpreted"
                    .to_string(),
            );
        }
    }
    warnings.sort();
    warnings.dedup();
    parsed.warnings = warnings;
    Ok(parsed)
}

fn parse_object(
    path: &Path,
    header_len: u64,
    file_len: u64,
    object: &Map<String, Value>,
) -> Result<SafeFile> {
    let mut metadata = Vec::new();
    let mut tensors = Vec::new();
    if let Some(meta) = object.get(METADATA_KEY) {
        let meta_object = meta
            .as_object()
            .ok_or_else(|| msg(format!("{}: __metadata__ is not an object", path.display())))?;
        for (key, value) in meta_object {
            let text = value.as_str().ok_or_else(|| {
                msg(format!(
                    "{}: metadata {key} is not a string",
                    path.display()
                ))
            })?;
            metadata.push((key.clone(), text.to_string()));
        }
    }
    for (name, value) in object {
        if name == METADATA_KEY {
            continue;
        }
        let entry = value.as_object().ok_or_else(|| {
            msg(format!(
                "{}: tensor {name} is not an object",
                path.display()
            ))
        })?;
        let dtype = entry
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                msg(format!(
                    "{}: tensor {name} is missing dtype",
                    path.display()
                ))
            })?
            .to_string();
        safe_dtype(&dtype)?;
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                msg(format!(
                    "{}: tensor {name} is missing shape",
                    path.display()
                ))
            })?;
        let mut dims = Vec::with_capacity(shape.len());
        for dim in shape {
            let dim = dim.as_u64().ok_or_else(|| {
                msg(format!(
                    "{}: tensor {name} has a non-integer shape",
                    path.display()
                ))
            })?;
            if dim > MAX_OFFSET {
                return Err(msg(format!(
                    "{}: tensor {name} shape exceeds 48 bits",
                    path.display()
                )));
            }
            dims.push(dim);
        }
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                msg(format!(
                    "{}: tensor {name} is missing data_offsets",
                    path.display()
                ))
            })?;
        if offsets.len() != 2 {
            return Err(msg(format!(
                "{}: tensor {name} data_offsets must have two numbers",
                path.display()
            )));
        }
        let begin = json_u64(&offsets[0]).ok_or_else(|| {
            msg(format!(
                "{}: tensor {name} has a bad offset",
                path.display()
            ))
        })?;
        let end = json_u64(&offsets[1]).ok_or_else(|| {
            msg(format!(
                "{}: tensor {name} has a bad offset",
                path.display()
            ))
        })?;
        if begin > end || begin > MAX_OFFSET || end > MAX_OFFSET {
            return Err(msg(format!(
                "{}: tensor {name} offsets [{begin}, {end}) are invalid",
                path.display()
            )));
        }
        let expected = packed_bytes(safe_dtype(&dtype)?, &dims)?;
        if end - begin != expected {
            return Err(msg(format!(
                "{}: tensor {name} covers {} bytes, dtype {dtype} shape {} needs {expected}",
                path.display(),
                end - begin,
                format_shape(&dims)
            )));
        }
        tensors.push(TensorEntry {
            name: name.clone(),
            dtype,
            shape: dims,
            begin,
            end,
        });
    }
    let data_start = 8 + header_len;
    let data_len = file_len - data_start;
    check_coverage(&tensors, data_len)?;
    Ok(SafeFile {
        path: path.to_path_buf(),
        header_len,
        data_start,
        data_len,
        metadata,
        tensors,
        warnings: Vec::new(),
    })
}

fn json_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|n| u64::try_from(n).ok()))
}

fn check_coverage(tensors: &[TensorEntry], data_len: u64) -> Result<()> {
    let mut ranges: Vec<(u64, u64)> = tensors
        .iter()
        .map(|tensor| (tensor.begin, tensor.end))
        .collect();
    ranges.sort_unstable();
    let mut cursor = 0u64;
    for (begin, end) in ranges {
        if begin != cursor {
            return Err(msg(format!(
                "safetensors data has a hole or overlap at byte {cursor} (next range starts at {begin})"
            )));
        }
        cursor = end;
    }
    if cursor != data_len {
        return Err(msg(format!(
            "safetensors data is {data_len} bytes and the tensors cover {cursor}"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct OutTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: Payload,
}

pub fn write_file(path: &Path, metadata: &[(String, String)], tensors: &[OutTensor]) -> Result<()> {
    let mut cursor = 0u64;
    let mut header = Map::new();
    if !metadata.is_empty() {
        let mut meta = Map::new();
        for (key, value) in metadata {
            meta.insert(key.clone(), Value::String(value.clone()));
        }
        header.insert(METADATA_KEY.to_string(), Value::Object(meta));
    }
    for tensor in tensors {
        let dtype = safe_dtype(&tensor.dtype)?;
        let expected = packed_bytes(dtype, &tensor.shape)?;
        if expected != tensor.bytes.len() {
            return Err(msg(format!(
                "tensor {} is {} bytes on disk and dtype {} needs {expected}",
                tensor.name,
                tensor.bytes.len(),
                tensor.dtype
            )));
        }
        let end = cursor + tensor.bytes.len();
        if end > MAX_OFFSET {
            return Err(msg("safetensors data offset exceeds 48 bits"));
        }
        let mut entry = Map::new();
        entry.insert("dtype".to_string(), Value::String(tensor.dtype.clone()));
        entry.insert(
            "shape".to_string(),
            Value::Array(
                tensor
                    .shape
                    .iter()
                    .copied()
                    .map(|dim| Value::from(dim))
                    .collect(),
            ),
        );
        entry.insert(
            "data_offsets".to_string(),
            Value::Array(vec![Value::from(cursor), Value::from(end)]),
        );
        header.insert(tensor.name.clone(), Value::Object(entry));
        cursor = end;
    }
    let json = serde_json::to_vec(&Value::Object(header))?;
    let mut padded = json;
    let pad = (8 - (padded.len() % 8)) % 8;
    padded.extend(std::iter::repeat(b' ').take(pad));
    let header_len = padded.len() as u64;
    atomic_write(path, |file| {
        file.write_all(&header_len.to_le_bytes())?;
        file.write_all(&padded)?;
        for tensor in tensors {
            tensor.bytes.copy_to(file)?;
        }
        Ok(())
    })
}

fn check_duplicate_keys(text: &str) -> Result<()> {
    let bytes = text.as_bytes();
    let mut index = 0;
    skip_ws(bytes, &mut index);
    if bytes.get(index) != Some(&b'{') {
        return Err(msg("safetensors header does not start with '{'"));
    }
    index += 1;
    let mut seen = HashSet::new();
    loop {
        skip_ws(bytes, &mut index);
        if bytes.get(index) == Some(&b'}') {
            return Ok(());
        }
        let key = parse_json_string(bytes, &mut index)?;
        if !seen.insert(key.clone()) {
            return Err(msg(format!("duplicate safetensors key {key}")));
        }
        skip_ws(bytes, &mut index);
        if bytes.get(index) != Some(&b':') {
            return Err(msg(format!("safetensors key {key} is missing ':'")));
        }
        index += 1;
        skip_value(bytes, &mut index)?;
        skip_ws(bytes, &mut index);
        match bytes.get(index) {
            Some(b',') => {
                index += 1;
                skip_ws(bytes, &mut index);
                if bytes.get(index) == Some(&b'}') {
                    return Err(msg("safetensors header has a trailing comma"));
                }
            }
            Some(b'}') => return Ok(()),
            _ => return Err(msg("safetensors header is not valid json")),
        }
    }
}

fn skip_ws(bytes: &[u8], index: &mut usize) {
    while bytes.get(*index).is_some_and(u8::is_ascii_whitespace) {
        *index += 1;
    }
}

fn parse_json_string(bytes: &[u8], index: &mut usize) -> Result<String> {
    if bytes.get(*index) != Some(&b'"') {
        return Err(msg("expected a json string"));
    }
    *index += 1;
    let mut out = Vec::new();
    while *index < bytes.len() {
        let ch = bytes[*index];
        *index += 1;
        match ch {
            b'"' => {
                return String::from_utf8(out).map_err(|_| msg("json key is not utf-8"));
            }
            b'\\' => {
                let esc = *bytes.get(*index).ok_or_else(|| msg("bad json escape"))?;
                *index += 1;
                match esc {
                    b'"' | b'\\' | b'/' => out.push(esc),
                    b'b' => out.push(0x08),
                    b'f' => out.push(0x0c),
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'u' => {
                        if *index + 4 > bytes.len() {
                            return Err(msg("short unicode escape"));
                        }
                        let hex = std::str::from_utf8(&bytes[*index..*index + 4])
                            .map_err(|_| msg("bad unicode escape"))?;
                        let code =
                            u32::from_str_radix(hex, 16).map_err(|_| msg("bad unicode escape"))?;
                        *index += 4;
                        let ch = char::from_u32(code).ok_or_else(|| msg("bad unicode escape"))?;
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                    _ => return Err(msg("bad json escape")),
                }
            }
            _ => out.push(ch),
        }
    }
    Err(msg("unterminated json string"))
}

fn skip_value(bytes: &[u8], index: &mut usize) -> Result<()> {
    skip_ws(bytes, index);
    match bytes.get(*index).copied() {
        Some(b'"') => {
            parse_json_string(bytes, index)?;
            Ok(())
        }
        Some(b'{') => {
            *index += 1;
            loop {
                skip_ws(bytes, index);
                if bytes.get(*index) == Some(&b'}') {
                    *index += 1;
                    return Ok(());
                }
                parse_json_string(bytes, index)?;
                skip_ws(bytes, index);
                if bytes.get(*index) != Some(&b':') {
                    return Err(msg("json object is missing ':'"));
                }
                *index += 1;
                skip_value(bytes, index)?;
                skip_ws(bytes, index);
                match bytes.get(*index) {
                    Some(b',') => *index += 1,
                    Some(b'}') => {
                        *index += 1;
                        return Ok(());
                    }
                    _ => return Err(msg("json object is not closed")),
                }
            }
        }
        Some(b'[') => {
            *index += 1;
            loop {
                skip_ws(bytes, index);
                if bytes.get(*index) == Some(&b']') {
                    *index += 1;
                    return Ok(());
                }
                skip_value(bytes, index)?;
                skip_ws(bytes, index);
                match bytes.get(*index) {
                    Some(b',') => *index += 1,
                    Some(b']') => {
                        *index += 1;
                        return Ok(());
                    }
                    _ => return Err(msg("json array is not closed")),
                }
            }
        }
        Some(b't') => eat(bytes, index, b"true"),
        Some(b'f') => eat(bytes, index, b"false"),
        Some(b'n') => eat(bytes, index, b"null"),
        Some(b'-') | Some(b'0'..=b'9') => {
            if bytes.get(*index) == Some(&b'-') {
                *index += 1;
            }
            while bytes.get(*index).is_some_and(|ch| ch.is_ascii_digit()) {
                *index += 1;
            }
            if bytes.get(*index) == Some(&b'.') {
                *index += 1;
                while bytes.get(*index).is_some_and(|ch| ch.is_ascii_digit()) {
                    *index += 1;
                }
            }
            if matches!(bytes.get(*index), Some(b'e' | b'E')) {
                *index += 1;
                if matches!(bytes.get(*index), Some(b'+' | b'-')) {
                    *index += 1;
                }
                while bytes.get(*index).is_some_and(|ch| ch.is_ascii_digit()) {
                    *index += 1;
                }
            }
            Ok(())
        }
        _ => Err(msg("safetensors header is not valid json")),
    }
}

fn eat(bytes: &[u8], index: &mut usize, word: &[u8]) -> Result<()> {
    if bytes.get(*index..*index + word.len()) == Some(word) {
        *index += word.len();
        Ok(())
    } else {
        Err(msg("safetensors header is not valid json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn round_trip_preserves_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let raw = dir.path().join("raw.bin");
        {
            let mut file = File::create(&raw).unwrap();
            file.write_all(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        }
        let out = dir.path().join("model.safetensors");
        write_file(
            &out,
            &[("format".into(), "pt".into())],
            &[OutTensor {
                name: "w".into(),
                dtype: "F32".into(),
                shape: vec![2],
                bytes: crate::payload::Payload::Range {
                    path: raw,
                    offset: 0,
                    len: 8,
                },
            }],
        )
        .unwrap();
        let parsed = read_file(&out, crate::cli::DEFAULT_MAX_HEADER).unwrap();
        assert_eq!(parsed.metadata, vec![("format".into(), "pt".into())]);
        assert_eq!(parsed.tensors[0].name, "w");
        assert_eq!(parsed.tensors[0].end - parsed.tensors[0].begin, 8);
        let bytes = crate::io_util::read_range(&out, parsed.data_start, 8).unwrap();
        assert_eq!(bytes, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
