use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::cli::FormatName;
use crate::detect::{self, Detected};
use crate::dtype::{self, ggml_dense};
use crate::error::{io_err, msg, Result};
use crate::gguf::{self, GgufFile};
use crate::io_util::relative_child;
use crate::npy;
use crate::npz;
use crate::payload::Payload;
use crate::safetensors;

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
    Safe,
    Gguf(GgufFile),
    Npy,
    Npz,
    Index(IndexFile),
    Raw,
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
    if format == Some(FormatName::Raw) {
        return open_raw(path, dtype, shape);
    }
    let detected = detect::sniff(path, max_header)?;
    detect::refuse_unsafe(detected, path)?;
    let forced = format;
    match (forced, detected) {
        (Some(FormatName::Safetensors), Detected::Safetensors) | (None, Detected::Safetensors) => {
            open_safe(path, max_header)
        }
        (Some(FormatName::Gguf), Detected::Gguf) | (None, Detected::Gguf) => open_gguf(path),
        (Some(FormatName::Npy), Detected::Npy) | (None, Detected::Npy) => open_npy(path),
        (Some(FormatName::Npz), Detected::Zip) | (None, Detected::Zip) => open_npz(path),
        (None, Detected::Index) => open_index(path, max_header),
        (Some(FormatName::Safetensors), Detected::Index) => open_index(path, max_header),
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

fn open_raw(path: &Path, dtype: Option<&str>, shape: Option<&[u64]>) -> Result<Archive> {
    let dtype_name = dtype.ok_or_else(|| msg("raw archives need --dtype"))?;
    let shape = shape
        .ok_or_else(|| msg("raw archives need --shape"))?
        .to_vec();
    let dtype = dtype::safe_dtype(dtype_name)?;
    let nbytes = dtype::packed_bytes(dtype, &shape)?;
    let file_len = fs::metadata(path).map_err(|err| io_err(path, err))?.len();
    if file_len != nbytes {
        return Err(msg(format!(
            "{} is {file_len} bytes and dtype {} shape {:?} needs {nbytes}",
            path.display(),
            dtype.name,
            shape
        )));
    }
    let name = path
        .file_stem()
        .and_then(|text| text.to_str())
        .unwrap_or("tensor")
        .to_string();
    let elements = dtype::numel(&shape);
    Ok(Archive {
        format: FormatName::Raw,
        version: "raw".into(),
        path: path.to_path_buf(),
        data_bytes: nbytes,
        metadata: Vec::new(),
        notes: Vec::new(),
        inner: Inner::Raw,
        members: vec![Member {
            name,
            dtype: dtype.name.to_string(),
            shape,
            file_offset: 0,
            nbytes,
            numel: elements,
            payload: Payload::Range {
                path: path.to_path_buf(),
                offset: 0,
                len: nbytes,
            },
            shard: None,
            detail: None,
            fortran: false,
        }],
    })
}

fn open_safe(path: &Path, max_header: u64) -> Result<Archive> {
    let file = safetensors::read_file(path, max_header)?;
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
    Ok(Archive {
        format: FormatName::Safetensors,
        version: "header+blob".into(),
        path: path.to_path_buf(),
        members,
        metadata,
        data_bytes,
        notes: Vec::new(),
        inner: Inner::Safe,
    })
}

fn open_gguf(path: &Path) -> Result<Archive> {
    let file = gguf::read_file(path)?;
    let mut members: Vec<_> = file
        .tensors
        .iter()
        .map(|tensor| {
            let dense = ggml_dense(tensor.ggml_type);
            let detail = if dtype::is_quant_ggml(tensor.ggml_type) {
                Some("quantized span includes alignment padding".to_string())
            } else if dense.is_some()
                && gguf::logical_dense_bytes(tensor).is_some_and(|logical| logical != tensor.span)
            {
                Some(format!(
                    "stored span {} includes padding; dense payload is {} bytes",
                    tensor.span,
                    gguf::logical_dense_bytes(tensor).unwrap_or(0)
                ))
            } else {
                None
            };
            Member {
                name: tensor.name.clone(),
                dtype: gguf::type_label(tensor),
                shape: tensor.shape.clone(),
                file_offset: tensor.file_offset,
                nbytes: tensor.span,
                numel: dtype::numel(&tensor.shape),
                payload: Payload::Range {
                    path: path.to_path_buf(),
                    offset: tensor.file_offset,
                    len: tensor.span,
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
        .map(|field| (field.key.clone(), field.display.clone()))
        .collect();
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
        notes: vec![
            format!("alignment {alignment}"),
            format!("data starts at {}", file.data_start),
        ],
        inner: Inner::Gguf(file),
    })
}

fn open_npy(path: &Path) -> Result<Archive> {
    let info = npy::read_info(path)?;
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
            payload: Payload::Range {
                path: path.to_path_buf(),
                offset: info.data_offset,
                len: info.nbytes,
            },
            shard: None,
            detail: info.fortran_order.then(|| "fortran_order".into()),
            fortran: info.fortran_order,
        }],
    })
}

fn open_npz(path: &Path) -> Result<Archive> {
    let entries = npz::read_entries(path)?;
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
            file_offset: entry.info.data_offset,
            nbytes: entry.info.nbytes,
            numel: dtype::numel(&entry.info.shape),
            payload: Payload::Zip {
                path: path.to_path_buf(),
                entry: entry.zip_name.clone(),
                skip: entry.info.data_offset,
                len: entry.info.nbytes,
            },
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
    let mut notes = Vec::new();
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
        let file = safetensors::read_file(&shard_path, max_header)?;
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
