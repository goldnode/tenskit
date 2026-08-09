use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::cli::FormatName;
use crate::detect::{self, Detected};
use crate::dtype;
use crate::error::{io_err, msg, Result};
use crate::io_util::relative_child;
use crate::payload::Payload;

#[derive(Debug, Clone)]
pub struct Member {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub file_offset: u64,
    pub nbytes: u64,
    pub numel: Option<u64>,
    pub payload: Payload,
    pub shard: Option<String>,
    pub detail: Option<String>,
    pub fortran: bool,
}

#[derive(Debug, Clone)]
pub struct Archive {
    pub format: FormatName,
    pub version: String,
    pub path: PathBuf,
    pub members: Vec<Member>,
    pub metadata: Vec<(String, String)>,
    pub data_bytes: u64,
    pub notes: Vec<String>,
    pub inner: Inner,
}

#[derive(Debug, Clone)]
pub enum Inner {
    #[cfg(format_safetensors)]
    Safe,
    #[cfg(format_gguf)]
    Gguf(crate::formats::gguf::GgufFile),
    #[cfg(format_npy)]
    Npy,
    #[cfg(format_npz)]
    Npz,
    #[cfg(format_safetensors)]
    Index(IndexFile),
    #[cfg(format_raw_sidecar)]
    Raw(crate::formats::rawtensor::RawFile),
    #[cfg(format_pytorch)]
    Pytorch,
    #[cfg(format_onnx)]
    Onnx,
}

#[derive(Debug, Clone)]
pub struct IndexFile {
    pub directory: PathBuf,
    pub metadata: Map<String, Value>,
    /// Tensor name to shard file name, in index order.
    pub weight_map: Vec<(String, String)>,
}

pub fn open(
    path: &Path,
    format: Option<FormatName>,
    max_header: u64,
    dtype: Option<&str>,
    shape: Option<&[u64]>,
) -> Result<Archive> {
    let _ = (dtype, shape);
    let mut archive = open_detected(path, format, max_header)?;
    annotate_packs(&mut archive);
    Ok(archive)
}

fn open_detected(path: &Path, format: Option<FormatName>, max_header: u64) -> Result<Archive> {
    #[cfg(format_raw_sidecar)]
    if crate::formats::rawtensor::is_manifest_name(file_name(path)) {
        return open_rawtensor(path);
    }
    let detected = detect::sniff(path, max_header)?;
    detect::refuse_unsafe(detected, path)?;
    let forced = format;
    match (forced, detected) {
        #[cfg(format_safetensors)]
        (Some(FormatName::Safetensors), Detected::Safetensors) | (None, Detected::Safetensors) => {
            open_safe(path, max_header)
        }
        #[cfg(format_gguf)]
        (Some(FormatName::Gguf), Detected::Gguf) | (None, Detected::Gguf) => open_gguf(path),
        #[cfg(format_npy)]
        (Some(FormatName::Npy), Detected::Npy) | (None, Detected::Npy) => open_npy(path),
        #[cfg(format_npz)]
        (Some(FormatName::Npz), Detected::Zip) | (None, Detected::Zip) => open_npz(path),
        #[cfg(format_safetensors)]
        (None, Detected::Index) => open_index(path, max_header),
        #[cfg(format_safetensors)]
        (Some(FormatName::Safetensors), Detected::Index) => open_index(path, max_header),
        #[cfg(format_raw_sidecar)]
        (Some(FormatName::Raw), Detected::RawTensor) | (None, Detected::RawTensor) => {
            open_rawtensor(path)
        }
        #[cfg(format_pytorch)]
        (Some(FormatName::Pytorch), Detected::Pytorch) | (None, Detected::Pytorch) => {
            open_pytorch(path)
        }
        #[cfg(format_onnx)]
        (Some(FormatName::Onnx), Detected::Onnx) | (None, Detected::Onnx) => {
            crate::formats::onnx::open(path)
        }
        (Some(forced), detected) => Err(msg(format!(
            "{} is {detected:?}, not {}",
            path.display(),
            forced.as_str()
        ))),
        (None, other) => Err(msg(format!(
            "{} has an unsupported layout ({other:?})",
            path.display()
        ))),
    }
}

#[cfg(format_pytorch)]
fn open_pytorch(path: &Path) -> Result<Archive> {
    crate::formats::pytorch::open(path)
}

fn annotate_packs(archive: &mut Archive) {
    let owned: Vec<String> = archive
        .members
        .iter()
        .map(|member| member.name.clone())
        .collect();
    let peers: Vec<&str> = owned.iter().map(String::as_str).collect();
    for member in &mut archive.members {
        if member.detail.is_some() {
            continue;
        }
        if let Some(note) = pack_note(&member.name, &peers) {
            member.detail = Some(note.to_string());
        }
    }
    #[cfg(format_mlx)]
    if archive
        .members
        .iter()
        .any(|member| member.dtype.contains('V'))
        || archive
            .metadata
            .iter()
            .any(|(_, value)| value.contains('V'))
    {
        if let Some(note) = crate::formats::mlx::void_note("<V2") {
            archive.notes.push(note.to_string());
        }
    }
}

fn pack_note(name: &str, peers: &[&str]) -> Option<&'static str> {
    #[cfg(format_gptq)]
    if let Some(note) = crate::formats::gptq::detail(name, peers) {
        return Some(note);
    }
    #[cfg(format_awq)]
    if let Some(note) = crate::formats::awq::detail(name, peers) {
        return Some(note);
    }
    #[cfg(format_bitsandbytes)]
    if let Some(note) = crate::formats::bitsandbytes::detail(name, peers) {
        return Some(note);
    }
    #[cfg(format_compressed_tensors)]
    if let Some(note) = crate::formats::compressed_tensors::detail(name) {
        return Some(note);
    }
    #[cfg(format_mlx)]
    if let Some(note) = crate::formats::mlx::quant_detail(name, peers) {
        return Some(note);
    }
    let _ = (name, peers);
    None
}

#[cfg(format_raw_sidecar)]
fn file_name(path: &Path) -> &str {
    path.file_name()
        .and_then(|text| text.to_str())
        .unwrap_or("")
}

#[cfg(format_raw_sidecar)]
fn open_rawtensor(path: &Path) -> Result<Archive> {
    let file = crate::formats::rawtensor::read_file(path)?;
    let members = file
        .tensors
        .iter()
        .map(|tensor| Member {
            name: tensor.name.clone(),
            dtype: tensor.dtype.clone(),
            shape: tensor.shape.clone(),
            file_offset: tensor.offset,
            nbytes: tensor.nbytes,
            numel: dtype::numel(&tensor.shape),
            payload: tensor.payload.clone(),
            shard: None,
            detail: if tensor.fortran {
                Some("fortran_order recorded in native; bytes were not transposed".into())
            } else {
                None
            },
            fortran: tensor.fortran,
        })
        .collect();
    let data_bytes = file
        .tensors
        .iter()
        .fold(0u64, |acc, tensor| acc.saturating_add(tensor.nbytes));
    let metadata = crate::formats::rawtensor::display_metadata(&file.metadata);
    let notes = file.notes.clone();
    Ok(Archive {
        format: FormatName::Raw,
        version: crate::formats::rawtensor::VERSION.to_string(),
        path: path.to_path_buf(),
        members,
        metadata,
        data_bytes,
        notes,
        inner: Inner::Raw(file),
    })
}

#[cfg(format_safetensors)]
fn open_safe(path: &Path, max_header: u64) -> Result<Archive> {
    let file = crate::formats::safetensors::read_file(path, max_header)?;
    let mut members: Vec<_> = file
        .tensors
        .iter()
        .map(|tensor| Member {
            name: tensor.name.clone(),
            dtype: tensor.dtype.clone(),
            shape: tensor.shape.clone(),
            file_offset: file.data_start + tensor.begin,
            nbytes: tensor.end - tensor.begin,
            numel: dtype::numel(&tensor.shape),
            payload: Payload::Range {
                path: path.to_path_buf(),
                offset: file.data_start + tensor.begin,
                len: tensor.end - tensor.begin,
            },
            shard: None,
            detail: None,
            fortran: false,
        })
        .collect();
    members.sort_by_key(|member| member.file_offset);
    let data_bytes = file.data_len;
    let metadata = file.metadata.clone();
    let notes = file.warnings.clone();
    Ok(Archive {
        format: FormatName::Safetensors,
        version: "header+blob".into(),
        path: path.to_path_buf(),
        members,
        metadata,
        data_bytes,
        notes,
        inner: Inner::Safe,
    })
}

#[cfg(format_gguf)]
fn open_gguf(path: &Path) -> Result<Archive> {
    let file = crate::formats::gguf::read_file(path)?;
    let mut members: Vec<_> = file
        .tensors
        .iter()
        .map(|tensor| {
            let padding = tensor.span - tensor.nbytes;
            let detail = (padding > 0).then(|| {
                format!("{padding} bytes of alignment padding follow the tensor")
            });
            // Member shapes for GGUF are ne order. A GGUF-023 orig_shape is
            // canonical order, so it is reversed into the same convention.
            let shape = match &tensor.orig_shape {
                Some(orig) => orig.iter().rev().copied().collect(),
                None => tensor.shape.clone(),
            };
            Member {
                name: tensor.name.clone(),
                dtype: crate::formats::gguf::type_label(tensor),
                shape,
                file_offset: tensor.file_offset,
                nbytes: tensor.nbytes,
                numel: dtype::numel(&tensor.shape),
                payload: Payload::Range {
                    path: path.to_path_buf(),
                    offset: tensor.file_offset,
                    len: tensor.nbytes,
                },
                shard: None,
                detail,
                fortran: false,
            }
        })
        .collect();
    members.sort_by_key(|member| member.file_offset);
    let metadata = file
        .metadata
        .iter()
        .map(|field| (field.key.clone(), field.value.text()))
        .collect();
    let gguf_warnings = file.warnings.clone();
    let data_bytes = file.data_len;
    let alignment = file.alignment;
    let version = file.version.to_string();
    Ok(Archive {
        format: FormatName::Gguf,
        version,
        path: path.to_path_buf(),
        members,
        metadata,
        data_bytes,
        notes: [
            format!("alignment {alignment}"),
            format!("data starts at {}", file.data_start),
            "shapes are ggml ne[] (innermost first); a conversion reverses them".into(),
        ]
        .into_iter()
        .chain(gguf_warnings)
        .collect(),
        inner: Inner::Gguf(file),
    })
}

#[cfg(format_npy)]
fn open_npy(path: &Path) -> Result<Archive> {
    let info = crate::formats::npy::read_info(path)?;
    let name = path
        .file_stem()
        .and_then(|text| text.to_str())
        .unwrap_or("array")
        .to_string();
    let dtype_name = info
        .dtype
        .map(|dtype| dtype.name.to_string())
        .unwrap_or_else(|| info.descr.clone());
    let mut notes = Vec::new();
    if info.fortran_order {
        notes.push("fortran_order: import into safetensors is refused".into());
    }
    if info.data_offset % 64 != 0 {
        notes.push("npy data is not 64-byte aligned (NPY-005); the array is still read".into());
    }
    if info.native_order {
        notes.push(format!("descr {} names the native byte order; it is read as little-endian (NPY-013)", info.descr));
    }
    Ok(Archive {
        format: FormatName::Npy,
        version: format!("{}.{}", info.major, info.minor),
        path: path.to_path_buf(),
        data_bytes: info.nbytes,
        metadata: vec![("descr".into(), info.descr.clone())],
        notes,
        inner: Inner::Npy,
        members: vec![Member {
            name,
            dtype: dtype_name,
            shape: info.shape.clone(),
            file_offset: info.data_offset,
            nbytes: info.nbytes,
            numel: dtype::numel(&info.shape),
            payload: info.canonical_payload(Payload::Range {
                path: path.to_path_buf(),
                offset: info.data_offset,
                len: info.nbytes,
            })?,
            shard: None,
            detail: info.fortran_order.then(|| "fortran_order".into()),
            fortran: info.fortran_order,
        }],
    })
}

#[cfg(format_npz)]
fn open_npz(path: &Path) -> Result<Archive> {
    let (entries, mut notes) = crate::formats::npz::read_entries(path)?;
    let mut data_bytes = 0u64;
    let mut members = Vec::with_capacity(entries.len());
    let mut any_compressed = false;
    for entry in &entries {
        any_compressed |= entry.compressed;
        data_bytes = data_bytes.saturating_add(entry.info.nbytes);
        let dtype_name = entry
            .info
            .dtype
            .map(|dtype| dtype.name.to_string())
            .unwrap_or_else(|| entry.info.descr.clone());
        members.push(Member {
            name: entry.key.clone(),
            dtype: dtype_name,
            shape: entry.info.shape.clone(),
            file_offset: entry
                .data_start
                .map_or(0, |start| start + entry.info.data_offset),
            nbytes: entry.info.nbytes,
            numel: dtype::numel(&entry.info.shape),
            payload: entry.info.canonical_payload(Payload::Zip {
                path: path.to_path_buf(),
                entry: entry.zip_name.clone(),
                skip: entry.info.data_offset,
                len: entry.info.nbytes,
            })?,
            shard: None,
            detail: if entry.compressed {
                Some("zip deflate".into())
            } else if entry.info.fortran_order {
                Some("fortran_order".into())
            } else {
                None
            },
            fortran: entry.info.fortran_order,
        });
    }
    if any_compressed {
        notes.push("some arrays are deflate-compressed inside the zip".into());
    }
    Ok(Archive {
        format: FormatName::Npz,
        version: "zip".into(),
        path: path.to_path_buf(),
        members,
        metadata: Vec::new(),
        data_bytes,
        notes,
        inner: Inner::Npz,
    })
}

#[cfg(format_safetensors)]
fn open_index(path: &Path, max_header: u64) -> Result<Archive> {
    let bytes = fs::read(path).map_err(|err| io_err(path, err))?;
    let value: Value = serde_json::from_slice(&bytes)?;
    let object = value
        .as_object()
        .ok_or_else(|| msg("index is not a JSON object"))?;
    let weight_map = object
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| msg("index is missing weight_map"))?;
    let metadata = object
        .get("metadata")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let directory = path
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let mut weight_vec = Vec::new();
    let mut seen = BTreeSet::new();
    for (tensor, shard_value) in weight_map {
        let shard_name = shard_value
            .as_str()
            .ok_or_else(|| msg(format!("weight_map entry {tensor} is not a file name")))?;
        if !seen.insert(tensor.clone()) {
            return Err(msg(format!("duplicate index key {tensor}")));
        }
        weight_vec.push((tensor.clone(), shard_name.to_string()));
    }
    let mut shards = BTreeMap::new();
    for (_, shard_name) in &weight_vec {
        if shards.contains_key(shard_name) {
            continue;
        }
        let shard_path = relative_child(&directory, shard_name)?;
        let file = crate::formats::safetensors::read_file(&shard_path, max_header)?;
        shards.insert(shard_name.clone(), file);
    }
    let mut claimed: BTreeSet<(String, String)> = BTreeSet::new();
    let mut members = Vec::with_capacity(weight_vec.len());
    for (tensor, shard_name) in &weight_vec {
        let file = shards.get(shard_name).expect("inserted");
        let entry = file
            .tensors
            .iter()
            .find(|item| &item.name == tensor)
            .ok_or_else(|| {
                msg(format!(
                    "index maps {tensor} to {shard_name}, which does not contain it"
                ))
            })?;
        claimed.insert((shard_name.clone(), tensor.clone()));
        let shard_path = relative_child(&directory, shard_name)?;
        members.push(Member {
            name: tensor.clone(),
            dtype: entry.dtype.clone(),
            shape: entry.shape.clone(),
            file_offset: file.data_start + entry.begin,
            nbytes: entry.end - entry.begin,
            numel: dtype::numel(&entry.shape),
            payload: Payload::Range {
                path: shard_path,
                offset: file.data_start + entry.begin,
                len: entry.end - entry.begin,
            },
            shard: Some(shard_name.clone()),
            detail: Some(shard_name.clone()),
            fortran: false,
        });
    }
    for (shard_name, file) in &shards {
        for tensor in &file.tensors {
            if !claimed.contains(&(shard_name.clone(), tensor.name.clone())) {
                return Err(msg(format!(
                    "{shard_name} contains {} which is absent from the index",
                    tensor.name
                )));
            }
        }
    }
    let data_bytes = members
        .iter()
        .fold(0u64, |acc, member| acc.saturating_add(member.nbytes));
    let meta_pairs = metadata
        .iter()
        .map(|(key, value)| {
            let text = match value {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            (key.clone(), text)
        })
        .collect();
    Ok(Archive {
        format: FormatName::Safetensors,
        version: "index".into(),
        path: path.to_path_buf(),
        members,
        metadata: meta_pairs,
        data_bytes,
        notes: vec![format!("{} shards", shards.len())],
        inner: Inner::Index(IndexFile {
            directory,
            metadata,
            weight_map: weight_vec,
        }),
    })
}

/// A path the user named, after a directory is turned into its weight file.
#[derive(Debug)]
pub struct ResolvedPath {
    pub path: PathBuf,
    /// Set when the user named a directory. `-i` prints it.
    pub note: Option<String>,
}

/// Open `path` after applying the Hugging Face directory rule.
pub fn open_user(
    path: &Path,
    force_single: bool,
    format: Option<FormatName>,
    max_header: u64,
    dtype: Option<&str>,
    shape: Option<&[u64]>,
) -> Result<Archive> {
    let resolved = resolve_user_path(path, force_single)?;
    let mut archive = open(&resolved.path, format, max_header, dtype, shape)?;
    if let Some(note) = resolved.note {
        archive.notes.push(note);
    }
    Ok(archive)
}

/// A directory is the safetensors weight set inside it. A `.safetensors` file
/// whose directory contains `config.json` is refused unless `force_single`.
pub fn resolve_user_path(path: &Path, force_single: bool) -> Result<ResolvedPath> {
    let meta = fs::metadata(path).map_err(|err| io_err(path, err))?;
    if meta.is_dir() {
        return resolve_directory(path);
    }
    if !force_single && is_safetensors_name(path) && sibling_config(path) {
        let dir = path
            .parent()
            .filter(|dir| !dir.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        return Err(msg(format!(
            "{} is likely part of a larger set of files because config.json is in {}. Pass that directory, or put --force-single-file immediately before this path.",
            path.display(),
            dir.display()
        )));
    }
    Ok(ResolvedPath {
        path: path.to_path_buf(),
        note: None,
    })
}

fn sibling_config(path: &Path) -> bool {
    path.parent()
        .map(|dir| dir.join("config.json").is_file())
        .unwrap_or(false)
}

fn is_safetensors_name(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| {
            let lower = name.to_ascii_lowercase();
            lower.ends_with(".safetensors") && !lower.ends_with(".safetensors.index.json")
        })
        .unwrap_or(false)
}

fn resolve_directory(dir: &Path) -> Result<ResolvedPath> {
    let mut indexes = Vec::new();
    let mut loose = Vec::new();
    let mut shards = Vec::new();
    let mut others = Vec::new();
    let entries = fs::read_dir(dir).map_err(|err| io_err(dir, err))?;
    for entry in entries {
        let entry = entry.map_err(|err| io_err(dir, err))?;
        // Follow a symlink to a file. DirEntry::metadata does not, so a
        // Hugging Face snapshot (every name points at blobs/) looks empty.
        // The chosen path stays the name in this directory; open follows it.
        // A link to a directory is not walked. A dangling link is skipped.
        let meta = match fs::metadata(entry.path()) {
            Ok(meta) => meta,
            Err(_) => continue,
        };
        if !meta.is_file() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".safetensors.index.json") {
            indexes.push(name.to_string());
        } else if lower.ends_with(".safetensors") && is_shard_name(&lower) {
            shards.push(name.to_string());
        } else if lower.ends_with(".safetensors") {
            loose.push(name.to_string());
        } else {
            others.push(name.to_string());
        }
    }
    indexes.sort();
    loose.sort();
    shards.sort();
    others.sort();
    let chosen = if indexes.len() == 1 {
        indexes[0].clone()
    } else if indexes.len() > 1 {
        return Err(msg(format!(
            "{} has more than one shard index: {}",
            dir.display(),
            indexes.join(", ")
        )));
    } else if loose.len() == 1 {
        loose[0].clone()
    } else if loose.len() > 1 {
        return Err(msg(format!(
            "{} has more than one safetensors file: {}",
            dir.display(),
            loose.join(", ")
        )));
    } else if !shards.is_empty() {
        return Err(msg(format!(
            "{} has safetensors shards but no index: {}",
            dir.display(),
            shards.join(", ")
        )));
    } else {
        return Err(msg(format!("{} has no safetensors weights", dir.display())));
    };
    let mut listed = others;
    if indexes.len() == 1 {
        listed.extend(loose);
    }
    listed.retain(|name| name != &chosen);
    listed.sort();
    let note = if listed.is_empty() {
        None
    } else {
        Some(format!(
            "directory {} also contains {}",
            dir.display(),
            listed.join(", ")
        ))
    };
    Ok(ResolvedPath {
        path: dir.join(chosen),
        note,
    })
}

/// `prefix-00001-of-00005.safetensors` is one shard, not the weight set.
fn is_shard_name(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".safetensors") else {
        return false;
    };
    let Some((head, count)) = stem.rsplit_once("-of-") else {
        return false;
    };
    let Some((_, index)) = head.rsplit_once('-') else {
        return false;
    };
    !index.is_empty()
        && !count.is_empty()
        && index.bytes().all(|byte| byte.is_ascii_digit())
        && count.bytes().all(|byte| byte.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_json_refuses_a_lone_safetensors_file() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("model.safetensors");
        fs::write(&weights, b"not read").unwrap();
        fs::write(dir.path().join("config.json"), b"{}\n").unwrap();
        let err = resolve_user_path(&weights, false).unwrap_err();
        assert!(err.to_string().contains("config.json"), "{err}");
        assert!(err.to_string().contains("--force-single-file"), "{err}");
        let forced = resolve_user_path(&weights, true).unwrap();
        assert_eq!(forced.path, weights);
        assert!(forced.note.is_none());
        let via_dir = resolve_user_path(dir.path(), false).unwrap();
        assert_eq!(via_dir.path, weights);
        let note = via_dir.note.unwrap();
        assert!(note.contains("config.json"), "{note}");
        let index = dir.path().join("model.safetensors.index.json");
        fs::write(&index, b"{}").unwrap();
        let index_path = resolve_user_path(&index, false).unwrap();
        assert_eq!(index_path.path, index);
        assert!(index_path.note.is_none());
    }

    #[test]
    fn a_directory_prefers_one_index_and_rejects_two_loose_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model-00001-of-00002.safetensors"), b"a").unwrap();
        fs::write(dir.path().join("model-00002-of-00002.safetensors"), b"b").unwrap();
        fs::write(dir.path().join("model.safetensors.index.json"), b"{}").unwrap();
        fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        let resolved = resolve_user_path(dir.path(), false).unwrap();
        assert_eq!(
            resolved.path,
            dir.path().join("model.safetensors.index.json")
        );
        let note = resolved.note.unwrap();
        assert!(note.contains("tokenizer.json"), "{note}");
        assert!(!note.contains("model-00001"), "{note}");

        let loose = tempfile::tempdir().unwrap();
        fs::write(loose.path().join("a.safetensors"), b"a").unwrap();
        fs::write(loose.path().join("b.safetensors"), b"b").unwrap();
        let err = resolve_user_path(loose.path(), false).unwrap_err();
        assert!(
            err.to_string().contains("more than one safetensors"),
            "{err}"
        );

        let shards = tempfile::tempdir().unwrap();
        fs::write(shards.path().join("model-00001-of-00002.safetensors"), b"a").unwrap();
        let err = resolve_user_path(shards.path(), false).unwrap_err();
        assert!(err.to_string().contains("no index"), "{err}");
    }

    #[test]
    fn a_snapshot_directory_follows_symlinks_to_the_weight_files() {
        let root = tempfile::tempdir().unwrap();
        let blobs = root.path().join("blobs");
        let snap = root.path().join("snapshots").join("rev");
        fs::create_dir_all(&blobs).unwrap();
        fs::create_dir_all(&snap).unwrap();
        let index_blob = blobs.join("index");
        let shard_blob = blobs.join("shard");
        let config_blob = blobs.join("config");
        fs::write(&index_blob, b"{}").unwrap();
        fs::write(&shard_blob, b"shard").unwrap();
        fs::write(&config_blob, b"{}\n").unwrap();
        std::os::unix::fs::symlink(&index_blob, snap.join("model.safetensors.index.json")).unwrap();
        std::os::unix::fs::symlink(
            &shard_blob,
            snap.join("model.safetensors-00001-of-00001.safetensors"),
        )
        .unwrap();
        std::os::unix::fs::symlink(&config_blob, snap.join("config.json")).unwrap();
        let resolved = resolve_user_path(&snap, false).unwrap();
        assert_eq!(resolved.path, snap.join("model.safetensors.index.json"));
        let note = resolved.note.unwrap();
        assert!(note.contains("config.json"), "{note}");
        assert!(!note.contains("00001-of-00001"), "{note}");
    }
}
