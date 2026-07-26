use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::Path;

use crate::dtype::{numpy_descr_to_safe, packed_bytes, SafeDtype};
use crate::error::{io_err, msg, Result};

const MAGIC: &[u8] = b"\x93NUMPY";
const ALIGN: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpyInfo {
    pub major: u8,
    pub minor: u8,
    pub descr: String,
    pub fortran_order: bool,
    pub shape: Vec<u64>,
    /// Absolute file offset of the first payload byte.
    pub data_offset: u64,
    pub nbytes: u64,
    pub dtype: Option<SafeDtype>,
}

pub fn read_info(path: &Path) -> Result<NpyInfo> {
    let mut file = File::open(path).map_err(|err| io_err(path, err))?;
    read_info_from(path, &mut file)
}

pub fn read_info_from(path: &Path, file: &mut (impl Read + Seek)) -> Result<NpyInfo> {
    let mut preamble = [0u8; 8];
    file.read_exact(&mut preamble)
        .map_err(|err| io_err(path, err))?;
    if &preamble[..6] != MAGIC {
        return Err(msg(format!("{} is not an .npy file", path.display())));
    }
    let major = preamble[6];
    let minor = preamble[7];
    let (header_len, encoding_utf8) = match (major, minor) {
        (1, 0) => {
            let mut buf = [0u8; 2];
            file.read_exact(&mut buf).map_err(|err| io_err(path, err))?;
            (u16::from_le_bytes(buf) as u64, false)
        }
        (2, 0) | (3, 0) => {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf).map_err(|err| io_err(path, err))?;
            (u32::from_le_bytes(buf) as u64, major == 3)
        }
        _ => {
            return Err(msg(format!(
                "{}: unsupported npy version {major}.{minor}",
                path.display()
            )));
        }
    };
    if header_len > 1024 * 1024 {
        return Err(msg(format!(
            "{}: npy header is {header_len} bytes",
            path.display()
        )));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|err| io_err(path, err))?;
    let header_text = if encoding_utf8 {
        String::from_utf8(header)
            .map_err(|_| msg(format!("{}: npy header is not utf-8", path.display())))?
    } else {
        header.iter().map(|byte| *byte as char).collect()
    };
    let (descr, fortran_order, shape) = parse_header_dict(&header_text)?;
    let data_offset = file.stream_position()?;
    let dtype = match numpy_descr_to_safe(&descr) {
        Ok(dtype) => Some(dtype),
        Err(err) => {
            // Structured arrays and object arrays still list. Object arrays
            // are refused at import time by the caller when a conversion is required.
            if descr_is_object(&descr) {
                return Err(err);
            }
            None
        }
    };
    let nbytes = if let Some(dtype) = dtype {
        packed_bytes(dtype, &shape)?
    } else {
        // Unknown but non-object descr: size from the trailing digits, if present.
        descr_itemsize(&descr)
            .unwrap_or(0)
            .saturating_mul(crate::dtype::numel(&shape).unwrap_or(0))
    };
    Ok(NpyInfo {
        major,
        minor,
        descr,
        fortran_order,
        shape,
        data_offset,
        nbytes,
        dtype,
    })
}

fn descr_is_object(descr: &str) -> bool {
    let rest = descr.strip_prefix(['<', '>', '|', '=']).unwrap_or(descr);
    rest.starts_with('O')
}

fn descr_itemsize(descr: &str) -> Option<u64> {
    let digits: String = descr
        .chars()
        .rev()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    digits.parse().ok()
}

pub fn write_header(descr: &str, shape: &[u64]) -> Result<Vec<u8>> {
    let shape_text = python_shape(shape);
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}");
    // NumPy pads with spaces and a final newline so the data starts on a
    // 64-byte boundary. A remainder of 0 still takes a full extra pad, which
    // keeps the newline from overwriting the dict.
    let header = dict.into_bytes();
    let preamble = 10usize;
    let with_newline = header.len() + 1;
    let padlen = ALIGN - ((preamble + with_newline) % ALIGN);
    let body_len = header.len() + padlen + 1;
    if body_len > u16::MAX as usize {
        return Err(msg("npy header does not fit in version 1.0"));
    }
    let mut out = Vec::with_capacity(preamble + body_len);
    out.extend_from_slice(MAGIC);
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(body_len as u16).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend(std::iter::repeat(b' ').take(padlen));
    out.push(b'\n');
    Ok(out)
}

fn python_shape(shape: &[u64]) -> String {
    match shape {
        [] => "()".to_string(),
        [one] => format!("({one},)"),
        _ => {
            let body = shape
                .iter()
                .map(|dim| dim.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("({body})")
        }
    }
}

fn parse_header_dict(text: &str) -> Result<(String, bool, Vec<u64>)> {
    let descr = quoted_field(text, "descr")?;
    let fortran_order = bool_field(text, "fortran_order")?;
    let shape = shape_field(text)?;
    Ok((descr, fortran_order, shape))
}

fn quoted_field(text: &str, key: &str) -> Result<String> {
    // The key is itself a quoted Python string, so the quote after the key
    // text closes the key. The value quote comes after the colon.
    let key_at = [format!("'{key}'"), format!("\"{key}\"")]
        .into_iter()
        .find_map(|pattern| text.find(&pattern).map(|index| index + pattern.len()))
        .ok_or_else(|| msg(format!("npy header is missing {key}")))?;
    let after = text[key_at..].trim_start();
    let after = after
        .strip_prefix(':')
        .ok_or_else(|| msg(format!("npy header {key} is not a string")))?
        .trim_start();
    let bytes = after.as_bytes();
    if bytes.is_empty() || (bytes[0] != b'\'' && bytes[0] != b'"') {
        return Err(msg(format!("npy header {key} is not a string")));
    }
    let quote = bytes[0];
    let mut index = 1;
    let mut value = String::new();
    while index < bytes.len() {
        let ch = bytes[index];
        if ch == b'\\' {
            index += 1;
            if index >= bytes.len() {
                break;
            }
            value.push(bytes[index] as char);
        } else if ch == quote {
            return Ok(value);
        } else {
            value.push(ch as char);
        }
        index += 1;
    }
    Err(msg(format!("npy header {key} string is not closed")))
}

fn bool_field(text: &str, key: &str) -> Result<bool> {
    let start = text
        .find(key)
        .ok_or_else(|| msg(format!("npy header is missing {key}")))?;
    let after = &text[start + key.len()..];
    let after = after
        .trim_start_matches(|ch: char| ch == ':' || ch == '\'' || ch == '"' || ch.is_whitespace());
    if after.starts_with("True") {
        Ok(true)
    } else if after.starts_with("False") {
        Ok(false)
    } else {
        Err(msg(format!("npy header {key} is not a boolean")))
    }
}

fn shape_field(text: &str) -> Result<Vec<u64>> {
    let start = text
        .find("shape")
        .ok_or_else(|| msg("npy header is missing shape"))?;
    let after = &text[start..];
    let open = after
        .find('(')
        .ok_or_else(|| msg("npy header shape is missing"))?;
    let close = after[open..]
        .find(')')
        .ok_or_else(|| msg("npy header shape is not closed"))?;
    let body = &after[open + 1..open + close];
    let mut shape = Vec::new();
    for part in body.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let dim: u64 = part
            .parse()
            .map_err(|_| msg(format!("invalid npy shape dimension {part}")))?;
        shape.push(dim);
    }
    Ok(shape)
}

pub fn write_file(
    path: &Path,
    descr: &str,
    shape: &[u64],
    payload: impl FnOnce(&mut File) -> Result<()>,
) -> Result<()> {
    let header = write_header(descr, shape)?;
    crate::io_util::atomic_write(path, |file| {
        file.write_all(&header)?;
        payload(file)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn header_is_aligned_and_round_trips() {
        let header = write_header("<f4", &[2, 3]).unwrap();
        assert_eq!(header.len() % 64, 0);
        let mut cursor = Cursor::new(header);
        let info = read_info_from(Path::new("mem.npy"), &mut cursor).unwrap();
        assert_eq!(info.descr, "<f4");
        assert!(!info.fortran_order);
        assert_eq!(info.shape, vec![2, 3]);
        assert_eq!(info.dtype.unwrap().name, "F32");
        assert_eq!(info.nbytes, 24);
    }

    #[test]
    fn object_dtype_is_refused() {
        let mut bytes = write_header("<f4", &[1]).unwrap();
        // Rewrite descr inside the header to an object dtype.
        let text = String::from_utf8(bytes.clone()).unwrap_or_default();
        let _ = text;
        let header = b"{'descr': '|O', 'fortran_order': False, 'shape': (1,), }\n";
        bytes = build_with_header(header);
        let mut cursor = Cursor::new(bytes);
        let err = read_info_from(Path::new("obj.npy"), &mut cursor).unwrap_err();
        assert!(err.to_string().contains("pickle"));
    }

    fn build_with_header(dict: &[u8]) -> Vec<u8> {
        let mut header = dict.to_vec();
        let pad = (64 - ((10 + header.len()) % 64)) % 64;
        header.extend(std::iter::repeat(b' ').take(pad));
        let last = header.len() - 1;
        header[last] = b'\n';
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(&header);
        out
    }
}
