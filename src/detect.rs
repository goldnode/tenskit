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

/// `_legacy_save` with pickle protocol 2: PROTO 2, then LONG1 of the
/// magic number 0x1950a86a20f9469cfc6c (pytorch.yaml `magic`).
const LEGACY_PICKLE: [u8; 14] = [
    0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19,
];

/// The older PyTorch tar container: `ustar` at offset 257.
fn is_tar(path: &Path, file: &mut File, len: u64) -> Result<bool> {
    use std::io::{Seek, SeekFrom};
    if len < 262 {
        return Ok(false);
    }
    file.seek(SeekFrom::Start(257))
        .map_err(|err| io_err(path, err))?;
    let mut magic = [0u8; 5];
    file.read_exact(&mut magic).map_err(|err| io_err(path, err))?;
    Ok(&magic == b"ustar")
}

/// A ZIP that is not a PyTorch checkpoint is npz when it is empty or holds
/// a `.npy` member (section 14). Any other ZIP is not a tensor file.
fn classify_zip(path: &Path) -> Result<Detected> {
    let file = File::open(path).map_err(|err| io_err(path, err))?;
    let archive = zip::ZipArchive::new(file)
        .map_err(|err| msg(format!("{}: {err}", path.display())))?;
    let mut names = archive.file_names();
    if archive.is_empty() || names.any(|name| name.ends_with(".npy")) {
        return Ok(Detected::Zip);
    }
    if archive.file_names().any(|name| name == "data.pkl") {
        return Err(msg(format!(
            "{} has data.pkl at the root, not in a folder; it is not a PyTorch checkpoint (PT-001)",
            path.display()
        )));
    }
    Err(msg(format!(
        "{} is a ZIP with no data.pkl and no .npy members; it is not a tensor file",
        path.display()
    )))
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
        return classify_zip(path);
    }
    if head.starts_with(b"ggml") || head.starts_with(b"ggmf") || head.starts_with(b"ggjt") {
        return Ok(Detected::LegacyGgml);
    }
    // conversion-considerations.md section 14: the full safetensors test
    // runs before any pickle test. Header lengths such as 640 start with
    // the bytes 80 02, which a one-byte pickle check would take.
    if let Some(kind) = safetensors_or_index(path, &mut file, len, max_header)? {
        return Ok(kind);
    }
    if head.starts_with(&LEGACY_PICKLE) || is_tar(path, &mut file, len)? {
        return Ok(Detected::Pickle);
    }
    // ONNX has no signature. The extension picks it, and the protobuf parse
    // and ONNX-007 decide the rest.
    #[cfg(format_onnx)]
    {
        let ext = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if matches!(ext.as_str(), "onnx" | "pb") {
            return Ok(Detected::Onnx);
        }
    }
    let named_safetensors = path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"));
    if named_safetensors {
        return Err(msg(format!(
            "{} is not a valid safetensors file: {}",
            path.display(),
            safetensors_problem(file, len, max_header)
        )));
    }
    Err(msg(format!(
        "cannot tell what kind of tensor file {} is; pass --format",
        path.display()
    )))
}

/// Why the first bytes of a file named .safetensors are not a safetensors
/// header (ST-001, ST-002).
fn safetensors_problem(mut file: File, len: u64, max_header: u64) -> String {
    use std::io::{Seek, SeekFrom};
    if len < 8 {
        return format!("it is {len} bytes, shorter than the 8-byte header length");
    }
    let mut len_buf = [0u8; 8];
    if file.seek(SeekFrom::Start(0)).is_err() || file.read_exact(&mut len_buf).is_err() {
        return "its header length cannot be read".into();
    }
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > max_header && header_len > len - 8 {
        return format!(
            "its first 8 bytes do not hold a plausible header length ({header_len}); it does not look like safetensors data"
        );
    }
    if header_len > len - 8 {
        return format!(
            "the header length {header_len} is more than the {} bytes after it; the file may be truncated or damaged",
            len - 8
        );
    }
    if header_len > max_header {
        return format!(
            "the header length {header_len} is above the {max_header}-byte limit; --max-header-size raises it"
        );
    }
    if header_len < 2 {
        return format!("the header length {header_len} is too short for a JSON object");
    }
    "the header does not start with '{'".into()
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
            if (2..=max_header).contains(&header_len) && len - 8 >= header_len {
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

pub fn refuse_unsafe(kind: Detected, path: &Path) -> Result<()> {
    match kind {
        Detected::Pickle => Err(msg(format!(
            "{} is a legacy PyTorch pickle or tar checkpoint and was not opened",
            path.display()
        ))),
        Detected::LegacyGgml => Err(msg(format!(
            "{} is legacy GGML/GGMF/GGJT; tenspile reads GGUF",
            path.display()
        ))),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bad_safetensors_file_is_named_as_such() {
        let dir = tempfile::tempdir().unwrap();
        let check = |name: &str, bytes: &[u8], want: &str| {
            let path = dir.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            let err = sniff(&path, 100_000_000).unwrap_err().to_string();
            assert!(err.contains(want), "{err}");
        };
        check("text.safetensors", b"hello world, not tensors\n", "not a valid safetensors file: its first 8 bytes");
        check("short.safetensors", b"abc", "shorter than the 8-byte header length");
        let mut cut = 320u64.to_le_bytes().to_vec();
        cut.extend_from_slice(b"{\"w\":");
        check("cut.safetensors", &cut, "the file may be truncated or damaged");
        check("junk.bin", b"junk junk junk\n", "cannot tell what kind of tensor file");
    }
}
