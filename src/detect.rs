use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::error::{io_err, msg, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detected {
    Safetensors,
    Gguf,
    Npy,
    Zip,
    LegacyGgml,
    Pickle,
    Index,
    RawTensor,
    /// Present even when the pytorch module is left out of the build.
    #[allow(dead_code)]
    Pytorch,
    /// Present even when the onnx module is left out of the build.
    #[allow(dead_code)]
    Onnx,
}

pub fn sniff(path: &Path, max_header: u64) -> Result<Detected> {
    let mut file = File::open(path).map_err(|err| io_err(path, err))?;
    let len = file.metadata().map_err(|err| io_err(path, err))?.len();
    let mut head = [0u8; 16];
    let n = file.read(&mut head).map_err(|err| io_err(path, err))?;
    let head = &head[..n];
    if head.starts_with(b"GGUF") {
        return Ok(Detected::Gguf);
    }
    if head.starts_with(b"\x93NUMPY") {
        return Ok(Detected::Npy);
    }
    if head.starts_with(b"PK\x03\x04") || head.starts_with(b"PK\x05\x06") {
        #[cfg(format_pytorch)]
        if crate::formats::pytorch::is_torch_zip(path)? {
            return Ok(Detected::Pytorch);
        }
        return Ok(Detected::Zip);
    }
    if head.starts_with(b"ggml") || head.starts_with(b"ggmf") || head.starts_with(b"ggjt") {
        return Ok(Detected::LegacyGgml);
    }
    if head.first() == Some(&0x80) && matches!(head.get(1), Some(2..=5)) {
        return Ok(Detected::Pickle);
    }
    if let Some(kind) = safetensors_or_index(path, &mut file, len, max_header)? {
        return Ok(kind);
    }
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if matches!(ext.as_str(), "pt" | "pth" | "bin" | "pkl" | "ckpt") {
        return Ok(Detected::Pickle);
    }
    #[cfg(format_onnx)]
    if matches!(ext.as_str(), "onnx" | "pb") {
        return Ok(Detected::Onnx);
    }
    Err(msg(format!(
        "cannot tell what kind of tensor file {} is; pass --format",
        path.display()
    )))
}

fn safetensors_or_index(
    path: &Path,
    file: &mut File,
    len: u64,
    max_header: u64,
) -> Result<Option<Detected>> {
    if len >= 8 {
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(0))
            .map_err(|err| io_err(path, err))?;
        let mut len_buf = [0u8; 8];
        if file.read_exact(&mut len_buf).is_ok() {
            let header_len = u64::from_le_bytes(len_buf);
            if header_len <= max_header && len >= 8 + header_len && header_len > 0 {
                let mut first = [0u8; 1];
                if file.read_exact(&mut first).is_ok() && first[0] == b'{' {
                    return Ok(Some(Detected::Safetensors));
                }
            }
        }
    }
    let name = path
        .file_name()
        .and_then(|text| text.to_str())
        .unwrap_or("");
    let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    let maybe_json = ext.eq_ignore_ascii_case("json") || name.ends_with(".safetensors.index.json");
    if !maybe_json {
        return Ok(None);
    }
    if len > 512 * 1024 * 1024 {
        return Err(msg(format!(
            "{} is too large to be an index",
            path.display()
        )));
    }
    let bytes = std::fs::read(path).map_err(|err| io_err(path, err))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    if value.get("format").and_then(serde_json::Value::as_str) == Some("rawtensor") {
        return Ok(Some(Detected::RawTensor));
    }
    if value
        .get("weight_map")
        .and_then(|item| item.as_object())
        .is_some()
    {
        Ok(Some(Detected::Index))
    } else {
        Err(msg(format!(
            "{} is JSON without a weight_map or a rawtensor manifest",
            path.display()
        )))
    }
}

/// True when the file starts with a pickle protocol opcode. Extension-only
/// guesses are separate: those bytes can still be imported as raw.
pub fn has_pickle_magic(path: &Path) -> Result<bool> {
    let mut file = File::open(path).map_err(|err| io_err(path, err))?;
    let mut head = [0u8; 2];
    let n = file.read(&mut head).map_err(|err| io_err(path, err))?;
    Ok(n >= 2 && head[0] == 0x80 && matches!(head.get(1), Some(2..=5)))
}

pub fn refuse_unsafe(kind: Detected, path: &Path) -> Result<()> {
    match kind {
        Detected::Pickle => Err(msg(format!(
            "{} is a PyTorch pickle file and was not opened",
            path.display()
        ))),
        Detected::LegacyGgml => Err(msg(format!(
            "{} is legacy GGML/GGMF/GGJT; tenspile reads GGUF",
            path.display()
        ))),
        _ => Ok(()),
    }
}
