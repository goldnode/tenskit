//! PyTorch zip checkpoints (`.pt`, `.pth`, `.bin`, `.ckpt`).
//!
//! Loading uses a restricted unpickler: a global outside the tensor rebuild
//! allowlist is rejected, and no code from the file runs. Legacy pickle streams
//! and the old tar container are recognized and refused. Remove this file and
//! the `pytorch` line in `local/supported-formats.txt` to drop the format.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::archive::{Archive, Inner, Member};
use crate::cli::FormatName;
use crate::dtype::{self, numel};
use crate::error::{io_err, msg, Result};
use crate::payload::Payload;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rebuild {
    V2,
    V3,
    /// `_rebuild_tensor`, which a rewrite does not reproduce.
    Legacy,
}

/// The persistent id `('storage', type, key, location, numel)`.
#[derive(Debug, Clone)]
struct StorageRef {
    module: String,
    type_name: String,
    key: String,
    location: String,
    numel: u64,
}

#[derive(Debug, Clone)]
struct TensorRef {
    dtype: String,
    shape: Vec<u64>,
    strides: Vec<u64>,
    storage: StorageRef,
    /// Storage size in bytes. A typed storage counts elements and an
    /// UntypedStorage counts bytes (PT-003).
    storage_bytes: u64,
    offset_elems: u64,
    itemsize: u64,
    rebuild: Rebuild,
    /// The torch dtype global that `_rebuild_tensor_v3` names.
    torch_dtype: Option<String>,
    requires_grad: bool,
    /// The conj and neg bits of the optional metadata argument (PT-010).
    conj: bool,
    neg: bool,
    /// `requires_grad` of an `nn.Parameter` wrapper.
    parameter: Option<bool>,
    /// False when the pickle carried something a rewrite cannot reproduce,
    /// such as backward hooks or tensor state.
    exact: bool,
}

#[derive(Debug, Clone)]
enum Obj {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<Obj>),
    List(Vec<Obj>),
    Set,
    /// A dict, or an `OrderedDict` with the state BUILD gave it (PT-007).
    Dict {
        pairs: Vec<(Obj, Obj)>,
        ordered: bool,
        state: Option<Box<Obj>>,
    },
    Global { module: String, name: String },
    Storage(StorageRef),
    Tensor(TensorRef),
    /// A tensor an edit writes, by index into its new tensors.
    Fresh { index: usize, parameter: Option<bool> },
}

/// What a rewrite needs from the file it edits.
#[derive(Debug, Clone)]
pub struct TorchFile {
    root: Obj,
    /// The archive folder, with its trailing `/`.
    prefix: String,
    byteorder: String,
    /// Why the pickle cannot be reproduced, if it cannot (PT-011).
    pub refused: Option<String>,
}

pub fn is_torch_zip(path: &Path) -> Result<bool> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(err) => return Err(io_err(path, err)),
    };
    let mut archive = match ZipArchive::new(file) {
        Ok(archive) => archive,
        Err(_) => return Ok(false),
    };
    for index in 0..archive.len() {
        let entry = archive
            .by_index(index)
            .map_err(|err| msg(format!("{}: {err}", path.display())))?;
        if is_data_pkl(entry.name()) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `<prefix>/data.pkl` with a non-empty top-level prefix (PT-001).
fn is_data_pkl(name: &str) -> bool {
    name.strip_suffix("/data.pkl")
        .is_some_and(|prefix| !prefix.is_empty() && !prefix.contains('/'))
}

fn record_text(zip: &mut ZipArchive<File>, name: &str) -> Option<String> {
    let mut entry = zip.by_name(name).ok()?;
    let mut text = String::new();
    entry.read_to_string(&mut text).ok()?;
    Some(text)
}

fn read_entry(zip: &mut ZipArchive<File>, name: &str, path: &Path) -> Result<Vec<u8>> {
    let mut entry = zip
        .by_name(name)
        .map_err(|err| msg(format!("{}: {err}", path.display())))?;
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes).map_err(|err| io_err(path, err))?;
    Ok(bytes)
}

fn is_c_order(shape: &[u64], strides: &[u64]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    let mut expect = 1u64;
    for index in (0..shape.len()).rev() {
        if shape[index] != 1 && strides[index] != expect {
            return false;
        }
        expect = expect.saturating_mul(shape[index].max(1));
    }
    true
}

struct NamedTensor {
    name: String,
    /// Positions from the root: a dict's pair index, or a list or tuple
    /// item index.
    path: Vec<usize>,
    tensor: TensorRef,
}

/// The names in a pickled object (PT-012). A path of keys and indices is
/// joined with `.`: a str key as is, an int key or a list or tuple index as
/// its decimal text. A tensor under any other key type has no name; it is
/// noted, not listed, and a rewrite keeps it as stored.
#[derive(Default)]
struct Walk {
    tensors: Vec<NamedTensor>,
    metadata: Vec<(String, String)>,
    notes: Vec<String>,
    /// Dicts that have a name, with their paths, for PT-013.
    dicts: Vec<(String, Vec<usize>)>,
}

impl Walk {
    fn of(root: &Obj) -> Walk {
        let mut walk = Walk::default();
        if matches!(root, Obj::Dict { .. } | Obj::List(_) | Obj::Tuple(_)) {
            walk.visit(root, Some(String::new()), &mut Vec::new(), "root".into());
        } else {
            walk.notes.push("the pickled object is not a dict of tensors".into());
        }
        walk
    }

    fn visit(&mut self, value: &Obj, name: Option<String>, path: &mut Vec<usize>, repr: String) {
        let child = |key: Option<String>| match (&name, key) {
            (Some(name), Some(key)) if name.is_empty() => Some(key),
            (Some(name), Some(key)) => Some(format!("{name}.{key}")),
            _ => None,
        };
        match value {
            Obj::Tensor(tensor) => match name {
                Some(name) => self.tensors.push(NamedTensor {
                    name,
                    path: path.clone(),
                    tensor: tensor.clone(),
                }),
                None => self.notes.push(format!(
                    "the tensor at {repr} ({} {}) has no name, because a key on its path is not a str or int (PT-012); it is not listed, and an edit keeps it",
                    tensor.dtype,
                    dtype::format_shape(&tensor.shape)
                )),
            },
            Obj::Dict { pairs, .. } => {
                if let Some(name) = &name {
                    self.dicts.push((name.clone(), path.clone()));
                }
                for (index, (key, item)) in pairs.iter().enumerate() {
                    let text = match key {
                        Obj::Str(text) => Some(text.clone()),
                        Obj::Int(number) => Some(number.to_string()),
                        _ => None,
                    };
                    path.push(index);
                    self.visit(item, child(text), path, format!("{repr}[{}]", key_repr(key)));
                    path.pop();
                }
            }
            Obj::List(items) | Obj::Tuple(items) => {
                for (index, item) in items.iter().enumerate() {
                    path.push(index);
                    self.visit(item, child(Some(index.to_string())), path, format!("{repr}[{index}]"));
                    path.pop();
                }
            }
            Obj::Str(text) => self.meta(name, text.clone()),
            Obj::Int(number) => self.meta(name, number.to_string()),
            Obj::Float(number) => self.meta(name, number.to_string()),
            Obj::Bool(flag) => self.meta(name, flag.to_string()),
            Obj::None => {}
            _ => {
                if let Some(name) = name {
                    self.notes.push(format!("{name} is not a tensor and was skipped"));
                }
            }
        }
    }

    fn meta(&mut self, name: Option<String>, value: String) {
        if let Some(name) = name {
            self.metadata.push((name, value));
        }
    }

    fn duplicate(&self) -> Option<&str> {
        let mut names = BTreeSet::new();
        self.tensors
            .iter()
            .find(|named| !names.insert(named.name.as_str()))
            .map(|named| named.name.as_str())
    }
}

/// A key as Python would print it, for a note.
fn key_repr(key: &Obj) -> String {
    match key {
        Obj::Str(text) => format!("{text:?}"),
        Obj::Int(number) => number.to_string(),
        Obj::Float(number) => number.to_string(),
        Obj::Bool(true) => "True".into(),
        Obj::Bool(false) => "False".into(),
        Obj::None => "None".into(),
        Obj::Tuple(_) => "(...)".into(),
        _ => "?".into(),
    }
}

/// The value at `path`.
fn slot_mut<'a>(obj: &'a mut Obj, path: &[usize]) -> Option<&'a mut Obj> {
    let Some((first, rest)) = path.split_first() else {
        return Some(obj);
    };
    let next = match obj {
        Obj::Dict { pairs, .. } => &mut pairs.get_mut(*first)?.1,
        Obj::List(items) | Obj::Tuple(items) => items.get_mut(*first)?,
        _ => return None,
    };
    slot_mut(next, rest)
}

/// Whether `path` passes through a list or tuple, whose later items would
/// be renamed if one were removed.
fn in_sequence(obj: &Obj, path: &[usize]) -> bool {
    let Some((first, rest)) = path.split_first() else {
        return false;
    };
    match obj {
        Obj::Dict { pairs, .. } => pairs.get(*first).is_some_and(|(_, value)| in_sequence(value, rest)),
        Obj::List(_) | Obj::Tuple(_) => true,
        _ => false,
    }
}

fn unpickle(bytes: &[u8]) -> Result<Obj> {
    let mut machine = Machine {
        data: bytes,
        index: 0,
        stack: Vec::new(),
        marks: Vec::new(),
        memo: HashMap::new(),
        budget: bytes.len().saturating_mul(64).saturating_add(1 << 20),
    };
    machine.load()
}

impl Obj {
    /// Node count, stopping once it passes `limit`.
    fn weight(&self, limit: usize) -> usize {
        let mut total = 1;
        let mut visit = |items: &mut dyn Iterator<Item = &Obj>| {
            for item in items {
                if total > limit {
                    break;
                }
                total += item.weight(limit - total.min(limit));
            }
        };
        match self {
            Obj::Tuple(items) | Obj::List(items) => visit(&mut items.iter()),
            Obj::Dict { pairs, .. } => visit(&mut pairs.iter().flat_map(|(k, v)| [k, v])),
            _ => {}
        }
        total
    }
}

struct Machine<'a> {
    data: &'a [u8],
    index: usize,
    stack: Vec<Obj>,
    marks: Vec<usize>,
    memo: HashMap<u32, Obj>,
    /// Nodes that BINGET may still copy. A memo copy is a deep clone, so a
    /// small pickle that fetches one object many times is refused here.
    budget: usize,
}

impl Machine<'_> {
    fn load(&mut self) -> Result<Obj> {
        while self.index < self.data.len() {
            let op = self.data[self.index];
            self.index += 1;
            match op {
                0x80 => {
                    self.byte()?;
                }
                0x2e => return self.pop(),
                0x28 => self.marks.push(self.stack.len()),
                0x7d => self.stack.push(Obj::Dict {
                    pairs: Vec::new(),
                    ordered: false,
                    state: None,
                }),
                0x29 => self.stack.push(Obj::Tuple(Vec::new())),
                0x4e => self.stack.push(Obj::None),
                0x89 => self.stack.push(Obj::Bool(false)),
                0x88 => self.stack.push(Obj::Bool(true)),
                0x5d => self.stack.push(Obj::List(Vec::new())),
                0x8f => self.stack.push(Obj::Set),
                0x71 => {
                    let index = u32::from(self.byte()?);
                    self.memo_set(index)?;
                }
                0x72 => {
                    let index = self.u32()?;
                    self.memo_set(index)?;
                }
                0x68 => {
                    let index = u32::from(self.byte()?);
                    self.memo_get(index)?;
                }
                0x6a => {
                    let index = self.u32()?;
                    self.memo_get(index)?;
                }
                0x58 => {
                    let len = self.u32()? as usize;
                    let text = self.utf8(len)?;
                    self.stack.push(Obj::Str(text));
                }
                0x55 => {
                    let len = self.byte()? as usize;
                    let text = self.utf8(len)?;
                    self.stack.push(Obj::Str(text));
                }
                0x8a => {
                    let len = self.byte()? as usize;
                    let value = self.long(len)?;
                    self.stack.push(Obj::Int(value));
                }
                0x47 => {
                    let bytes = self.raw(8)?;
                    let value = f64::from_be_bytes(bytes.try_into().unwrap());
                    self.stack.push(Obj::Float(value));
                }
                0x63 => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.allow(&module, &name)?;
                    self.stack.push(Obj::Global { module, name });
                }
                0x4b => {
                    let value = self.byte()? as i64;
                    self.stack.push(Obj::Int(value));
                }
                0x4d => {
                    let value = self.u16()? as i64;
                    self.stack.push(Obj::Int(value));
                }
                0x4a => {
                    let value = self.i32()? as i64;
                    self.stack.push(Obj::Int(value));
                }
                0x74 => {
                    let items = self.pop_mark()?;
                    self.stack.push(Obj::Tuple(items));
                }
                0x85 => self.push_tuple(1)?,
                0x86 => self.push_tuple(2)?,
                0x87 => self.push_tuple(3)?,
                0x51 => {
                    let pid = self.pop()?;
                    self.stack.push(persistent(&pid)?);
                }
                0x52 | 0x81 => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    self.stack.push(reduce(callable, args)?);
                }
                0x62 => {
                    let new_state = self.pop()?;
                    match self.stack.last_mut() {
                        // An OrderedDict carries _metadata here. A rewrite
                        // keeps it.
                        Some(Obj::Dict { state, .. }) => *state = Some(Box::new(new_state)),
                        // Tensor state does not change the bytes, but a
                        // rewrite would lose it.
                        Some(Obj::Tensor(tensor)) => tensor.exact = false,
                        _ => return Err(msg("pytorch pickle BUILD target is not a dict or tensor")),
                    }
                }
                0x73 => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    self.set_items(vec![key, value])?;
                }
                0x75 => {
                    let items = self.pop_mark()?;
                    self.set_items(items)?;
                }
                0x61 => {
                    let item = self.pop()?;
                    self.append(vec![item])?;
                }
                0x65 => {
                    let items = self.pop_mark()?;
                    self.append(items)?;
                }
                other => {
                    return Err(msg(format!(
                        "pytorch pickle opcode {other:#x} is not in the restricted reader"
                    )));
                }
            }
        }
        Err(msg("pytorch pickle ended without STOP"))
    }

    fn allow(&self, module: &str, name: &str) -> Result<()> {
        let ok = match module {
            "torch._utils" => name.starts_with("_rebuild_"),
            "collections" => name == "OrderedDict",
            "torch" | "torch.storage" => {
                name.ends_with("Storage")
                    || matches!(
                        name,
                        "float32"
                            | "float64"
                            | "float16"
                            | "bfloat16"
                            | "int64"
                            | "int32"
                            | "int16"
                            | "int8"
                            | "uint8"
                            | "bool"
                            | "complex64"
                            | "complex128"
                            | "float8_e4m3fn"
                            | "float8_e5m2"
                            | "float8_e4m3fnuz"
                            | "float8_e5m2fnuz"
                            | "float8_e8m0fnu"
                            | "uint16"
                            | "uint32"
                            | "uint64"
                    )
            }
            _ => false,
        };
        if ok {
            Ok(())
        } else {
            Err(msg(format!(
                "disallowed pickle global {module}.{name}"
            )))
        }
    }

    fn memo_set(&mut self, index: u32) -> Result<()> {
        let top = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| msg("pytorch pickle memo has no value"))?;
        self.memo.insert(index, top);
        Ok(())
    }

    fn memo_get(&mut self, index: u32) -> Result<()> {
        let value = self
            .memo
            .get(&index)
            .ok_or_else(|| msg("pytorch pickle memo index is missing"))?;
        let weight = value.weight(self.budget);
        if weight > self.budget {
            return Err(msg("pytorch pickle repeats memo objects past the reader's limit"));
        }
        self.budget -= weight;
        self.stack.push(value.clone());
        Ok(())
    }

    fn append(&mut self, items: Vec<Obj>) -> Result<()> {
        let Some(Obj::List(list)) = self.stack.last_mut() else {
            return Err(msg("pytorch pickle APPEND target is not a list"));
        };
        list.extend(items);
        Ok(())
    }

    /// LONG1: little-endian two's complement in `len` bytes.
    fn long(&mut self, len: usize) -> Result<i64> {
        if len > 8 {
            return Err(msg("pytorch pickle integer does not fit in 64 bits"));
        }
        let bytes = self.raw(len)?;
        let Some(last) = bytes.last() else {
            return Ok(0);
        };
        let mut buf = if last & 0x80 != 0 { [0xffu8; 8] } else { [0u8; 8] };
        buf[..len].copy_from_slice(bytes);
        Ok(i64::from_le_bytes(buf))
    }

    fn push_tuple(&mut self, count: usize) -> Result<()> {
        if self.stack.len() < count {
            return Err(msg("pytorch pickle tuple is short"));
        }
        let items = self.stack.split_off(self.stack.len() - count);
        self.stack.push(Obj::Tuple(items));
        Ok(())
    }

    fn pop_mark(&mut self) -> Result<Vec<Obj>> {
        let mark = self
            .marks
            .pop()
            .ok_or_else(|| msg("pytorch pickle is missing a mark"))?;
        if self.stack.len() < mark {
            return Err(msg("pytorch pickle mark is past the stack"));
        }
        Ok(self.stack.split_off(mark))
    }

    fn set_items(&mut self, items: Vec<Obj>) -> Result<()> {
        if items.len() % 2 != 0 {
            return Err(msg("pytorch pickle dict items are not in pairs"));
        }
        let Obj::Dict { pairs, .. } = self
            .stack
            .last_mut()
            .ok_or_else(|| msg("pytorch pickle SETITEMS has no dict"))?
        else {
            return Err(msg("pytorch pickle SETITEMS target is not a dict"));
        };
        let mut items = items.into_iter();
        while let Some(key) = items.next() {
            let value = items.next().unwrap();
            pairs.push((key, value));
        }
        Ok(())
    }

    fn pop(&mut self) -> Result<Obj> {
        self.stack
            .pop()
            .ok_or_else(|| msg("pytorch pickle stack underflow"))
    }

    fn byte(&mut self) -> Result<u8> {
        let value = *self
            .data
            .get(self.index)
            .ok_or_else(|| msg("pytorch pickle is truncated"))?;
        self.index += 1;
        Ok(value)
    }

    fn u16(&mut self) -> Result<u16> {
        let bytes = self.raw(2)?;
        Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        let bytes = self.raw(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32> {
        let bytes = self.raw(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn raw(&mut self, len: usize) -> Result<&[u8]> {
        let end = self
            .index
            .checked_add(len)
            .ok_or_else(|| msg("pytorch pickle is truncated"))?;
        if end > self.data.len() {
            return Err(msg("pytorch pickle is truncated"));
        }
        let slice = &self.data[self.index..end];
        self.index = end;
        Ok(slice)
    }

    fn utf8(&mut self, len: usize) -> Result<String> {
        let bytes = self.raw(len)?.to_vec();
        String::from_utf8(bytes).map_err(|_| msg("pytorch pickle string is not utf-8"))
    }

    fn line(&mut self) -> Result<String> {
        let start = self.index;
        while self.index < self.data.len() && self.data[self.index] != b'\n' {
            self.index += 1;
        }
        if self.index >= self.data.len() {
            return Err(msg("pytorch pickle global is truncated"));
        }
        let text = std::str::from_utf8(&self.data[start..self.index])
            .map_err(|_| msg("pytorch pickle global is not utf-8"))?
            .to_string();
        self.index += 1;
        Ok(text)
    }
}

fn persistent(pid: &Obj) -> Result<Obj> {
    let Obj::Tuple(items) = pid else {
        return Err(msg("pytorch pickle persistent id is not a tuple"));
    };
    if items.len() != 5 {
        return Err(msg("pytorch pickle persistent id has the wrong size"));
    }
    let Obj::Str(kind) = &items[0] else {
        return Err(msg("pytorch pickle persistent id is not a storage"));
    };
    if kind != "storage" {
        return Err(msg(format!(
            "pytorch pickle persistent id {kind} is not a storage"
        )));
    }
    let Obj::Global { module, name } = &items[1] else {
        return Err(msg("pytorch pickle storage type is not a global"));
    };
    let Obj::Str(key) = &items[2] else {
        return Err(msg("pytorch pickle storage key is not a string"));
    };
    let Obj::Str(location) = &items[3] else {
        return Err(msg("pytorch pickle storage location is not a string"));
    };
    let numel = as_u64(&items[4])?;
    Ok(Obj::Storage(StorageRef {
        module: module.clone(),
        type_name: name.clone(),
        key: key.clone(),
        location: location.clone(),
        numel,
    }))
}

fn reduce(callable: Obj, args: Obj) -> Result<Obj> {
    let Obj::Global { module, name } = callable else {
        return Err(msg("pytorch pickle REDUCE target is not a global"));
    };
    let Obj::Tuple(args) = args else {
        return Err(msg("pytorch pickle REDUCE arguments are not a tuple"));
    };
    if module == "collections" && name == "OrderedDict" {
        return Ok(Obj::Dict {
            pairs: Vec::new(),
            ordered: true,
            state: None,
        });
    }
    if name == "_rebuild_parameter" || name == "_rebuild_parameter_with_state" {
        // _rebuild_parameter(data, requires_grad, backward_hooks)
        let mut args = args.into_iter();
        let Some(Obj::Tensor(mut tensor)) = args.next() else {
            return Err(msg("pytorch parameter rebuild has no tensor"));
        };
        let requires_grad = args.next();
        let hooks = args.next();
        tensor.exact &= name == "_rebuild_parameter" && args.next().is_none() && is_empty_dict(hooks.as_ref());
        tensor.parameter = Some(match requires_grad {
            Some(Obj::Bool(flag)) => flag,
            _ => {
                tensor.exact = false;
                false
            }
        });
        return Ok(Obj::Tensor(tensor));
    }
    let rebuild = match name.as_str() {
        "_rebuild_tensor_v2" => Rebuild::V2,
        "_rebuild_tensor_v3" => Rebuild::V3,
        "_rebuild_tensor" => Rebuild::Legacy,
        _ => {
            return Err(msg(format!(
                "pytorch pickle reduce {module}.{name} is not a tensor rebuild"
            )))
        }
    };
    rebuild_tensor(rebuild, &args)
}

fn is_empty_dict(obj: Option<&Obj>) -> bool {
    matches!(obj, Some(Obj::Dict { pairs, state: None, .. }) if pairs.is_empty())
}

/// `_rebuild_tensor_v2(storage, offset, size, stride, requires_grad,
/// backward_hooks[, metadata])`; v3 has the dtype before metadata.
fn rebuild_tensor(rebuild: Rebuild, args: &[Obj]) -> Result<Obj> {
    if args.len() < 4 {
        return Err(msg("pytorch tensor rebuild is missing arguments"));
    }
    let Obj::Storage(storage) = &args[0] else {
        return Err(msg("pytorch tensor storage is not a persistent id"));
    };
    let (mut dtype, mut itemsize) = storage_layout(&storage.type_name)?;
    let storage_bytes = if storage.type_name == "UntypedStorage" {
        storage.numel
    } else {
        storage
            .numel
            .checked_mul(itemsize)
            .ok_or_else(|| msg("pytorch storage is too large"))?
    };
    let mut exact = rebuild != Rebuild::Legacy;
    let requires_grad = match args.get(4) {
        Some(Obj::Bool(flag)) => *flag,
        _ => {
            exact = false;
            false
        }
    };
    exact &= is_empty_dict(args.get(5));
    let mut next = 6;
    let mut torch_name = None;
    if rebuild == Rebuild::V3 {
        let Some(Obj::Global { module, name }) = args.get(6) else {
            return Err(msg("pytorch _rebuild_tensor_v3 has no dtype"));
        };
        let (canonical, bits) = torch_dtype(name)
            .filter(|_| module == "torch")
            .ok_or_else(|| msg(format!("pytorch dtype {module}.{name} has no canonical dtype")))?;
        dtype = canonical.into();
        itemsize = bits / 8;
        torch_name = Some(name.clone());
        next = 7;
    }
    let (mut conj, mut neg) = (false, false);
    if rebuild != Rebuild::Legacy {
        match args.get(next) {
            None => {}
            Some(Obj::Dict { pairs, .. }) => {
                for (key, value) in pairs {
                    match (key, value) {
                        (Obj::Str(key), Obj::Bool(flag)) if key == "conj" => conj = *flag,
                        (Obj::Str(key), Obj::Bool(flag)) if key == "neg" => neg = *flag,
                        _ => {
                            return Err(msg(
                                "pytorch tensor metadata holds more than the conj and neg bits (PT-010)",
                            ))
                        }
                    }
                }
            }
            Some(_) => return Err(msg("pytorch tensor metadata is not a dict (PT-010)")),
        }
        exact &= args.len() <= next + 1;
    }
    Ok(Obj::Tensor(TensorRef {
        dtype,
        shape: as_shape(&args[2])?,
        strides: as_shape(&args[3])?,
        storage: storage.clone(),
        storage_bytes,
        offset_elems: as_u64(&args[1])?,
        itemsize,
        rebuild,
        torch_dtype: torch_name,
        requires_grad,
        conj,
        neg,
        parameter: None,
        exact,
    }))
}

fn storage_layout(name: &str) -> Result<(String, u64)> {
    let (dtype, itemsize) = match name {
        "FloatStorage" => ("float32", 4),
        "DoubleStorage" => ("float64", 8),
        "HalfStorage" => ("float16", 2),
        "BFloat16Storage" => ("bfloat16", 2),
        "LongStorage" => ("int64", 8),
        "IntStorage" => ("int32", 4),
        "ShortStorage" => ("int16", 2),
        "CharStorage" => ("int8", 1),
        "ByteStorage" | "UntypedStorage" => ("uint8", 1),
        "BoolStorage" => ("bool", 1),
        "ComplexFloatStorage" => ("complex64", 8),
        "ComplexDoubleStorage" => ("complex128", 16),
        other => {
            return Err(msg(format!(
                "pytorch storage {other} has no canonical dtype"
            )))
        }
    };
    Ok((dtype.into(), itemsize))
}

fn torch_dtype(name: &str) -> Option<(&'static str, u64)> {
    Some(match name {
        "float32" => ("float32", 32),
        "float64" => ("float64", 64),
        "float16" => ("float16", 16),
        "bfloat16" => ("bfloat16", 16),
        "int64" => ("int64", 64),
        "int32" => ("int32", 32),
        "int16" => ("int16", 16),
        "int8" => ("int8", 8),
        "uint8" => ("uint8", 8),
        "uint16" => ("uint16", 16),
        "uint32" => ("uint32", 32),
        "uint64" => ("uint64", 64),
        "bool" => ("bool", 8),
        "complex64" => ("complex64", 64),
        "complex128" => ("complex128", 128),
        "float8_e4m3fn" => ("float8_e4m3fn", 8),
        "float8_e5m2" => ("float8_e5m2", 8),
        "float8_e4m3fnuz" => ("float8_e4m3fnuz", 8),
        "float8_e5m2fnuz" => ("float8_e5m2fnuz", 8),
        "float8_e8m0fnu" => ("float8_e8m0fnu", 8),
        _ => return None,
    })
}

fn as_u64(obj: &Obj) -> Result<u64> {
    match obj {
        Obj::Int(value) if *value >= 0 => Ok(*value as u64),
        _ => Err(msg("pytorch pickle expected a non-negative integer")),
    }
}

fn as_shape(obj: &Obj) -> Result<Vec<u64>> {
    let Obj::Tuple(items) = obj else {
        return Err(msg("pytorch tensor size is not a tuple"));
    };
    items.iter().map(as_u64).collect()
}

pub struct OutTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub payload: Payload,
}

/// Write a new checkpoint: a flat dict of tensors, as pytorch.yaml `write`.
pub fn write_archive(path: &Path, tensors: &[OutTensor]) -> Result<()> {
    if tensors.is_empty() {
        return Err(msg("writing a pytorch checkpoint with no tensors is not supported"));
    }
    let stem = path
        .file_stem()
        .and_then(|text| text.to_str())
        .filter(|text| !text.is_empty())
        .unwrap_or("archive");
    let root = Obj::Dict {
        pairs: tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                (
                    Obj::Str(tensor.name.clone()),
                    Obj::Fresh {
                        index,
                        parameter: None,
                    },
                )
            })
            .collect(),
        ordered: false,
        state: None,
    };
    let layout = Layout {
        prefix: format!("{stem}/"),
        byteorder: "little".into(),
        source: None,
        records: vec![("version".into(), b"3\n".to_vec())],
    };
    write_zip(path, &root, tensors, &layout).map(|_| ())
}

/// Rewrite `source` into `dest` (PT-011). The pickled structure is written
/// back as it was read: containers, key order, OrderedDict state, plain
/// values, and each tensor in `unchanged` with its storage, offset, strides,
/// and conj and neg bits. A storage that several tensors share stays one
/// `data/` record. Names missing from `tensors` are deleted; the rest are
/// replaced in place or added. Returns notes on what was re-encoded.
pub fn rewrite_archive(
    file: &TorchFile,
    source: &Path,
    dest: &Path,
    tensors: &[OutTensor],
    unchanged: &BTreeSet<String>,
) -> Result<Vec<String>> {
    if let Some(reason) = &file.refused {
        return Err(msg(format!(
            "{} holds {reason}, which tenspile cannot write back (PT-011); use tensconv to write a new state_dict on purpose",
            source.display()
        )));
    }
    let mut root = file.root.clone();
    let walk = Walk::of(&root);
    let mut removed = Vec::new();
    for named in &walk.tensors {
        if tensors.iter().any(|tensor| tensor.name == named.name) {
            continue;
        }
        if in_sequence(&root, &named.path) {
            return Err(msg(format!(
                "{} is in a list or tuple; deleting it would rename the items after it",
                named.name
            )));
        }
        removed.push(named.path.clone());
    }
    for (index, tensor) in tensors.iter().enumerate() {
        if unchanged.contains(&tensor.name) {
            continue;
        }
        let fresh = |parameter| Obj::Fresh { index, parameter };
        if let Some(named) = walk.tensors.iter().find(|named| named.name == tensor.name) {
            let slot = slot_mut(&mut root, &named.path)
                .ok_or_else(|| msg("internal error: a tensor path does not resolve"))?;
            // A replaced tensor stays where it is, and an nn.Parameter
            // stays a Parameter.
            let parameter = match slot {
                Obj::Tensor(old) => old.parameter,
                _ => None,
            };
            *slot = fresh(parameter);
            continue;
        }
        // PT-013: the deepest named dict whose name and a '.' start the new
        // name takes the rest as one key; otherwise the root dict does.
        let target = walk
            .dicts
            .iter()
            .filter(|(name, _)| {
                !name.is_empty()
                    && tensor.name.strip_prefix(name.as_str()).is_some_and(|rest| rest.starts_with('.'))
            })
            .max_by_key(|(name, path)| (path.len(), name.len()));
        let (path, key) = match target {
            Some((name, path)) => (path.clone(), tensor.name[name.len() + 1..].to_string()),
            None => (Vec::new(), tensor.name.clone()),
        };
        match slot_mut(&mut root, &path) {
            Some(Obj::Dict { pairs, .. }) => pairs.push((Obj::Str(key), fresh(None))),
            _ => {
                return Err(msg(format!(
                    "{} has no dict to add {} to; its root is not a dict (PT-013)",
                    source.display(),
                    tensor.name
                )))
            }
        }
    }
    // Remove the last positions first, so the earlier paths stay valid.
    removed.sort_unstable_by(|left, right| right.cmp(left));
    for path in removed {
        let (last, parent) = path.split_last().ok_or_else(|| msg("internal error: an empty tensor path"))?;
        if let Some(Obj::Dict { pairs, .. }) = slot_mut(&mut root, parent) {
            pairs.remove(*last);
        }
    }
    if let Some(name) = Walk::of(&root).duplicate() {
        return Err(msg(format!(
            "the edit would give two tensors the name {name} (PT-012)"
        )));
    }
    let mut zip = open_zip(source)?;
    let mut records = Vec::new();
    for index in 0..zip.len() {
        let mut entry = zip
            .by_index(index)
            .map_err(|err| msg(format!("{}: {err}", source.display())))?;
        let Some(rest) = entry.name().strip_prefix(&file.prefix).map(str::to_string) else {
            continue;
        };
        if entry.is_dir()
            || rest == "data.pkl"
            || rest.starts_with("data/")
            || matches!(rest.as_str(), ".format_version" | ".storage_alignment" | "byteorder")
        {
            continue;
        }
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).map_err(|err| io_err(source, err))?;
        records.push((rest, bytes));
    }
    let layout = Layout {
        prefix: file.prefix.clone(),
        byteorder: file.byteorder.clone(),
        source: Some(source),
        records,
    };
    // .format_version is left out without a note: PT-009 drops it, and it
    // changes no loaded value (CONV-009).
    write_zip(dest, &root, tensors, &layout)
}

fn open_zip(path: &Path) -> Result<ZipArchive<File>> {
    let file = File::open(path).map_err(|err| io_err(path, err))?;
    ZipArchive::new(file).map_err(|err| msg(format!("{}: {err}", path.display())))
}

/// Why `obj` cannot be pickled back as it was read, if it cannot.
fn unreproducible(obj: &Obj) -> Option<String> {
    match obj {
        Obj::Set => Some("a set".into()),
        Obj::Global { module, name } => Some(format!("the value {module}.{name}")),
        Obj::Storage(_) => Some("a storage outside a tensor".into()),
        Obj::Tensor(tensor) if !tensor.exact => {
            Some("a tensor with backward hooks, extra state, or an old rebuild call".into())
        }
        Obj::Tuple(items) | Obj::List(items) => items.iter().find_map(unreproducible),
        Obj::Dict { pairs, state, .. } => pairs
            .iter()
            .flat_map(|(key, value)| [key, value])
            .find_map(unreproducible)
            .or_else(|| state.as_deref().and_then(unreproducible)),
        _ => None,
    }
}

/// Where a checkpoint's records go, apart from data.pkl and data/.
struct Layout<'a> {
    prefix: String,
    byteorder: String,
    /// The checkpoint whose `data/` records unchanged tensors point at.
    source: Option<&'a Path>,
    /// Records written after `data/`, by name under the prefix.
    records: Vec<(String, Vec<u8>)>,
}

enum Record {
    /// A storage copied from the source checkpoint, by key.
    Kept(String),
    /// A new storage holding `tensors[index]`.
    Fresh { key: String, index: usize },
}

fn write_zip(path: &Path, root: &Obj, tensors: &[OutTensor], layout: &Layout) -> Result<Vec<String>> {
    let mut taken = BTreeSet::new();
    collect_keys(root, &mut taken);
    let mut emitter = Emitter {
        out: vec![0x80, 2],
        tensors,
        taken,
        next_key: 0,
        records: Vec::new(),
        seen: BTreeSet::new(),
    };
    emitter.emit(root)?;
    emitter.out.push(0x2e);
    let Emitter { out: pickle, records, .. } = emitter;
    let big = layout.byteorder == "big";
    let mut notes = Vec::new();
    let mut source = layout.source.map(open_zip).transpose()?;
    crate::io_util::with_temp_file(path, |tmp| {
        let file = crate::io_util::create_new(tmp)?;
        let mut writer = ZipWriter::new(file);
        let options = SimpleFileOptions::default()
            .compression_method(CompressionMethod::Stored)
            .large_file(true);
        // data/ records start on a 64-byte boundary, as .storage_alignment
        // says. The zip library's layout is not torch's miniz layout, so
        // .format_version is left out (PT-009): with it, torch can compute
        // the record offsets itself and read the wrong bytes.
        let aligned = options.with_alignment(64);
        let prefix = &layout.prefix;
        let put = |writer: &mut ZipWriter<File>, name: &str, bytes: &[u8]| -> Result<()> {
            writer
                .start_file(format!("{prefix}{name}"), options)
                .map_err(|err| msg(format!("pytorch: {err}")))?;
            writer.write_all(bytes)?;
            Ok(())
        };
        put(&mut writer, "data.pkl", &pickle)?;
        put(&mut writer, ".storage_alignment", b"64")?;
        put(&mut writer, "byteorder", layout.byteorder.as_bytes())?;
        for record in &records {
            let key = match record {
                Record::Kept(key) | Record::Fresh { key, .. } => key,
            };
            writer
                .start_file(format!("{prefix}data/{key}"), aligned)
                .map_err(|err| msg(format!("pytorch: {err}")))?;
            match record {
                Record::Kept(key) => {
                    let (Some(zip), Some(source_path)) = (source.as_mut(), layout.source) else {
                        return Err(msg("internal error: a kept storage has no source"));
                    };
                    let mut entry = zip.by_name(&format!("{prefix}data/{key}")).map_err(|_| {
                        msg(format!("{} is missing storage {key}", source_path.display()))
                    })?;
                    if entry.compression() != CompressionMethod::Stored {
                        notes.push(format!(
                            "storage {key} was compressed and is now stored uncompressed (PT-005)"
                        ));
                    }
                    std::io::copy(&mut entry, &mut writer).map_err(|err| io_err(source_path, err))?;
                }
                Record::Fresh { index, .. } => {
                    let tensor = &tensors[*index];
                    if big {
                        let mut bytes = tensor.payload.read_vec()?;
                        dtype::byteswap(&mut bytes, dtype::canonical_id(&tensor.dtype)?)?;
                        writer.write_all(&bytes)?;
                    } else {
                        tensor.payload.copy_to(&mut writer)?;
                    }
                }
            }
        }
        for (name, bytes) in &layout.records {
            put(&mut writer, name, bytes)?;
        }
        writer
            .finish()
            .map_err(|err| msg(format!("pytorch: {err}")))?
            .sync_all()?;
        Ok(())
    })?;
    Ok(notes)
}

/// Storage keys that unchanged tensors keep.
fn collect_keys(obj: &Obj, keys: &mut BTreeSet<String>) {
    match obj {
        Obj::Tensor(tensor) => {
            keys.insert(tensor.storage.key.clone());
        }
        Obj::Tuple(items) | Obj::List(items) => items.iter().for_each(|item| collect_keys(item, keys)),
        Obj::Dict { pairs, state, .. } => {
            for (key, value) in pairs {
                collect_keys(key, keys);
                collect_keys(value, keys);
            }
            if let Some(state) = state {
                collect_keys(state, keys);
            }
        }
        _ => {}
    }
}

struct Emitter<'a> {
    out: Vec<u8>,
    tensors: &'a [OutTensor],
    /// Keys of kept storages, which new storages must not reuse.
    taken: BTreeSet<String>,
    next_key: u64,
    records: Vec<Record>,
    seen: BTreeSet<String>,
}

impl Emitter<'_> {
    fn emit(&mut self, obj: &Obj) -> Result<()> {
        match obj {
            Obj::None => self.out.push(0x4e),
            Obj::Bool(flag) => self.out.push(if *flag { 0x88 } else { 0x89 }),
            Obj::Int(value) => put_signed(&mut self.out, *value),
            Obj::Float(value) => {
                self.out.push(0x47);
                self.out.extend_from_slice(&value.to_be_bytes());
            }
            Obj::Str(text) => put_unicode(&mut self.out, text),
            Obj::Tuple(items) if items.is_empty() => self.out.push(0x29),
            Obj::Tuple(items) => {
                self.out.push(0x28);
                for item in items {
                    self.emit(item)?;
                }
                self.out.push(0x74);
            }
            Obj::List(items) => {
                self.out.push(0x5d);
                if !items.is_empty() {
                    self.out.push(0x28);
                    for item in items {
                        self.emit(item)?;
                    }
                    self.out.push(0x65);
                }
            }
            Obj::Dict { pairs, ordered, state } => {
                if *ordered {
                    put_ordered_dict(&mut self.out);
                } else {
                    self.out.push(0x7d);
                }
                if !pairs.is_empty() {
                    self.out.push(0x28);
                    for (key, value) in pairs {
                        self.emit(key)?;
                        self.emit(value)?;
                    }
                    self.out.push(0x75);
                }
                if let Some(state) = state {
                    self.emit(state)?;
                    self.out.push(0x62);
                }
            }
            Obj::Tensor(tensor) => {
                if self.seen.insert(tensor.storage.key.clone()) {
                    self.records.push(Record::Kept(tensor.storage.key.clone()));
                }
                put_tensor(&mut self.out, tensor)?;
            }
            Obj::Fresh { index, parameter } => {
                let key = loop {
                    let key = self.next_key.to_string();
                    self.next_key += 1;
                    if !self.taken.contains(&key) {
                        break key;
                    }
                };
                let mut tensor = fresh_ref(&self.tensors[*index], key.clone())?;
                tensor.parameter = *parameter;
                self.records.push(Record::Fresh { key, index: *index });
                put_tensor(&mut self.out, &tensor)?;
            }
            Obj::Set | Obj::Global { .. } | Obj::Storage(_) => {
                return Err(msg("internal error: this pickle value cannot be written"));
            }
        }
        Ok(())
    }
}

/// The rebuild call for a new tensor. A dtype without a legacy storage class
/// goes through _rebuild_tensor_v3 with an UntypedStorage, whose numel counts
/// bytes (pytorch.yaml `write`).
fn fresh_ref(tensor: &OutTensor, key: String) -> Result<TensorRef> {
    let canonical = dtype::canonical_id(&tensor.dtype)?;
    let untyped = canonical_storage(canonical).is_err();
    let (module, storage, itemsize, torch_name) = if untyped {
        let (name, bits) = untyped_dtype(canonical)?;
        ("torch.storage", "UntypedStorage", bits / 8, Some(name.to_string()))
    } else {
        let (storage, itemsize) = canonical_storage(canonical)?;
        ("torch", storage, itemsize, None)
    };
    let count = numel(&tensor.shape).ok_or_else(|| msg("shape is too large to count"))?;
    let nbytes = count
        .checked_mul(itemsize)
        .ok_or_else(|| msg("pytorch tensor is too large"))?;
    if tensor.payload.len() != nbytes {
        return Err(msg(format!(
            "tensor {} is {} bytes; {canonical} needs {nbytes}",
            tensor.name,
            tensor.payload.len()
        )));
    }
    Ok(TensorRef {
        dtype: canonical.into(),
        shape: tensor.shape.clone(),
        strides: c_strides(&tensor.shape),
        storage: StorageRef {
            module: module.into(),
            type_name: storage.into(),
            key,
            location: "cpu".into(),
            numel: if untyped { nbytes } else { count },
        },
        storage_bytes: nbytes,
        offset_elems: 0,
        itemsize,
        rebuild: if untyped { Rebuild::V3 } else { Rebuild::V2 },
        torch_dtype: torch_name,
        requires_grad: false,
        conj: false,
        neg: false,
        parameter: None,
        exact: true,
    })
}

fn put_tensor(out: &mut Vec<u8>, tensor: &TensorRef) -> Result<()> {
    if !tensor.exact || tensor.rebuild == Rebuild::Legacy {
        return Err(msg("internal error: this tensor's rebuild call cannot be written"));
    }
    if let Some(requires_grad) = tensor.parameter {
        put_global(out, "torch._utils", "_rebuild_parameter");
        out.push(0x28);
        put_rebuild(out, tensor)?;
        out.push(if requires_grad { 0x88 } else { 0x89 });
        put_ordered_dict(out);
        out.push(0x74);
        out.push(0x52);
        Ok(())
    } else {
        put_rebuild(out, tensor)
    }
}

fn put_rebuild(out: &mut Vec<u8>, tensor: &TensorRef) -> Result<()> {
    let v3 = tensor.rebuild == Rebuild::V3;
    put_global(out, "torch._utils", if v3 { "_rebuild_tensor_v3" } else { "_rebuild_tensor_v2" });
    out.push(0x28);
    out.push(0x28);
    put_unicode(out, "storage");
    put_global(out, &tensor.storage.module, &tensor.storage.type_name);
    put_unicode(out, &tensor.storage.key);
    put_unicode(out, &tensor.storage.location);
    put_int(out, tensor.storage.numel);
    out.push(0x74);
    out.push(0x51);
    put_int(out, tensor.offset_elems);
    put_shape(out, &tensor.shape);
    put_shape(out, &tensor.strides);
    out.push(if tensor.requires_grad { 0x88 } else { 0x89 });
    put_ordered_dict(out);
    if v3 {
        let name = tensor
            .torch_dtype
            .as_deref()
            .ok_or_else(|| msg("internal error: a v3 rebuild has no dtype"))?;
        put_global(out, "torch", name);
    }
    if tensor.conj || tensor.neg {
        out.extend_from_slice(&[0x7d, 0x28]);
        for (key, set) in [("conj", tensor.conj), ("neg", tensor.neg)] {
            if set {
                put_unicode(out, key);
                out.push(0x88);
            }
        }
        out.push(0x75);
    }
    out.push(0x74);
    out.push(0x52);
    Ok(())
}

fn put_ordered_dict(out: &mut Vec<u8>) {
    put_global(out, "collections", "OrderedDict");
    out.push(0x29);
    out.push(0x52);
}

fn canonical_storage(dtype: &str) -> Result<(&'static str, u64)> {
    Ok(match dtype {
        "float32" => ("FloatStorage", 4),
        "float64" => ("DoubleStorage", 8),
        "float16" => ("HalfStorage", 2),
        "bfloat16" => ("BFloat16Storage", 2),
        "int64" => ("LongStorage", 8),
        "int32" => ("IntStorage", 4),
        "int16" => ("ShortStorage", 2),
        "int8" => ("CharStorage", 1),
        "uint8" => ("ByteStorage", 1),
        "bool" => ("BoolStorage", 1),
        "complex64" => ("ComplexFloatStorage", 8),
        "complex128" => ("ComplexDoubleStorage", 16),
        other => {
            return Err(msg(format!(
                "dtype {other} has no pytorch storage class"
            )))
        }
    })
}

/// The torch dtype name and bits of a canonical dtype that has no legacy
/// storage class. float4_e2m1fn_x2 packs two values per element and is not
/// written.
fn untyped_dtype(canonical: &str) -> Result<(&'static str, u64)> {
    match canonical {
        "uint16" | "uint32" | "uint64" | "float8_e4m3fn" | "float8_e5m2" | "float8_e4m3fnuz"
        | "float8_e5m2fnuz" | "float8_e8m0fnu" => torch_dtype(canonical)
            .ok_or_else(|| msg(format!("dtype {canonical} has no pytorch dtype"))),
        other => Err(msg(format!("dtype {other} has no pytorch storage class"))),
    }
}

fn c_strides(shape: &[u64]) -> Vec<u64> {
    let mut strides = vec![1u64; shape.len()];
    for index in (0..shape.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1].saturating_mul(shape[index + 1]);
    }
    strides
}

fn put_unicode(out: &mut Vec<u8>, text: &str) {
    out.push(0x58);
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

fn put_global(out: &mut Vec<u8>, module: &str, name: &str) {
    out.push(0x63);
    out.extend_from_slice(module.as_bytes());
    out.push(b'\n');
    out.extend_from_slice(name.as_bytes());
    out.push(b'\n');
}

fn put_int(out: &mut Vec<u8>, value: u64) {
    if value <= 0xff {
        out.push(0x4b);
        out.push(value as u8);
    } else if value <= 0xffff {
        out.push(0x4d);
        out.extend_from_slice(&(value as u16).to_le_bytes());
    } else if value <= i32::MAX as u64 {
        out.push(0x4a);
        out.extend_from_slice(&(value as i32).to_le_bytes());
    } else {
        // LONG1 with a trailing zero byte so the value stays positive.
        let bytes = value.to_le_bytes();
        let used = 8 - (value.leading_zeros() / 8) as usize;
        let len = if bytes[used - 1] & 0x80 != 0 { used + 1 } else { used };
        out.push(0x8a);
        out.push(len as u8);
        out.extend_from_slice(&bytes[..used]);
        out.resize(out.len() + (len - used), 0);
    }
}

fn put_signed(out: &mut Vec<u8>, value: i64) {
    if value >= 0 {
        put_int(out, value as u64);
    } else if value >= i64::from(i32::MIN) {
        out.push(0x4a);
        out.extend_from_slice(&(value as i32).to_le_bytes());
    } else {
        out.push(0x8a);
        out.push(8);
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn put_shape(out: &mut Vec<u8>, shape: &[u64]) {
    match shape.len() {
        0 => out.push(0x29),
        1 => {
            put_int(out, shape[0]);
            out.push(0x85);
        }
        2 => {
            put_int(out, shape[0]);
            put_int(out, shape[1]);
            out.push(0x86);
        }
        3 => {
            for dim in shape {
                put_int(out, *dim);
            }
            out.push(0x87);
        }
        _ => {
            out.push(0x28);
            for dim in shape {
                put_int(out, *dim);
            }
            out.push(0x74);
        }
    }
}

pub fn open(path: &Path) -> Result<Archive> {
    let mut zip = open_zip(path)?;
    // PT-001 / PT-001i: the first record names the archive folder, every
    // record sits under it, and data.pkl is directly inside it.
    let mut prefix: Option<String> = None;
    let mut pkl_name = None;
    for index in 0..zip.len() {
        let entry = zip
            .by_index(index)
            .map_err(|err| msg(format!("{}: {err}", path.display())))?;
        let name = entry.name().to_string();
        let top = match name.split_once('/') {
            Some((top, _)) if !top.is_empty() => top.to_string(),
            _ => {
                return Err(msg(format!(
                    "{} record {name} is not in a subdirectory (PT-001)",
                    path.display()
                )))
            }
        };
        match &prefix {
            None => prefix = Some(top),
            Some(first) if *first != top => {
                return Err(msg(format!(
                    "{} mixes zip prefixes (PT-001)",
                    path.display()
                )))
            }
            _ => {}
        }
        if is_data_pkl(&name) {
            if pkl_name.is_some() {
                return Err(msg(format!(
                    "{} has more than one data.pkl (PT-001)",
                    path.display()
                )));
            }
            pkl_name = Some(name);
        }
    }
    let pkl_name = pkl_name.ok_or_else(|| {
        msg(format!(
            "{} is not a pytorch zip; it has no <prefix>/data.pkl",
            path.display()
        ))
    })?;
    let prefix = pkl_name.trim_end_matches("data.pkl").to_string();
    let pickle = read_entry(&mut zip, &pkl_name, path)?;
    let obj = unpickle(&pickle)?;
    let Walk {
        tensors: found,
        metadata,
        mut notes,
        ..
    } = Walk::of(&obj);
    if found.is_empty() {
        return Err(msg(format!("{} pickle has no tensors", path.display())));
    }
    // PT-012: an ambiguous name is listed twice, and the file is not edited.
    let mut names = BTreeSet::new();
    let refused = unreproducible(&obj).or_else(|| {
        found
            .iter()
            .find(|named| !names.insert(named.name.as_str()))
            .map(|named| format!("two tensors named {} (PT-012)", named.name))
    });
    let byteorder =
        record_text(&mut zip, &format!("{prefix}byteorder")).unwrap_or_else(|| "little".into());
    if byteorder != "little" && byteorder != "big" {
        return Err(msg(format!(
            "{} byteorder {byteorder} is not little or big",
            path.display()
        )));
    }
    if byteorder == "big" {
        notes.push("byteorder is big; element values are byteswapped (PT-006)".into());
    }
    let version = record_text(&mut zip, &format!("{prefix}version"))
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "zip".into());
    // data_bytes counts each storage once, as the file holds it; a tied
    // weight is a second view of the same storage.
    let mut storages = BTreeMap::new();
    for named in &found {
        storages.insert(named.tensor.storage.key.clone(), named.tensor.storage_bytes);
    }
    let data_bytes = storages.values().fold(0u64, |acc, bytes| acc.saturating_add(*bytes));
    if storages.len() < found.len() {
        notes.push(format!(
            "{} tensors share {} storages; data_bytes counts each storage once",
            found.len(),
            storages.len()
        ));
    }
    let mut members = Vec::with_capacity(found.len());
    for named in found {
        let tensor = &named.tensor;
        let count = numel(&tensor.shape).ok_or_else(|| msg("shape is too large to count"))?;
        let nbytes = count
            .checked_mul(tensor.itemsize)
            .ok_or_else(|| msg("pytorch tensor is too large"))?;
        let payload = tensor_payload(&mut zip, path, &prefix, &named, &byteorder, &mut notes)?;
        let file_offset = match &payload {
            Payload::Range { offset, .. } => *offset,
            _ => 0,
        };
        members.push(Member {
            name: named.name,
            dtype: named.tensor.dtype,
            shape: named.tensor.shape,
            file_offset,
            nbytes,
            numel: Some(count),
            payload,
            shard: None,
            detail: None,
            fortran: false,
        });
    }
    Ok(Archive {
        format: FormatName::Pytorch,
        version,
        path: path.to_path_buf(),
        members,
        metadata,
        data_bytes,
        notes,
        inner: Inner::Pytorch(TorchFile {
            root: obj,
            prefix,
            byteorder,
            refused,
        }),
    })
}

/// A tensor's canonical bytes. A contiguous view of a little-endian,
/// uncompressed storage is read where it lies. Otherwise the view is read,
/// gathered when strided, byteswapped (PT-006), and has its conj and neg
/// bits applied (PT-010).
fn tensor_payload(
    zip: &mut ZipArchive<File>,
    path: &Path,
    prefix: &str,
    named: &NamedTensor,
    byteorder: &str,
    notes: &mut Vec<String>,
) -> Result<Payload> {
    let tensor = &named.tensor;
    let count = numel(&tensor.shape).ok_or_else(|| msg("shape is too large to count"))?;
    if tensor.strides.len() != tensor.shape.len() {
        return Err(msg(format!(
            "pytorch tensor {} has {} strides for {} dimensions",
            named.name,
            tensor.strides.len(),
            tensor.shape.len()
        )));
    }
    // Elements from the first to the last one the view reads (PT-004).
    let mut span = u64::from(count > 0);
    if count > 0 {
        for (size, stride) in tensor.shape.iter().zip(&tensor.strides) {
            span = (size - 1)
                .checked_mul(*stride)
                .and_then(|reach| span.checked_add(reach))
                .ok_or_else(|| msg("pytorch tensor view overflows"))?;
        }
    }
    let byte_off = tensor
        .offset_elems
        .checked_mul(tensor.itemsize)
        .ok_or_else(|| msg("pytorch storage offset overflows"))?;
    let span_bytes = span
        .checked_mul(tensor.itemsize)
        .ok_or_else(|| msg("pytorch tensor is too large"))?;
    let end = byte_off
        .checked_add(span_bytes)
        .ok_or_else(|| msg("pytorch storage offset overflows"))?;
    if count > 0 && tensor.storage_bytes < end {
        return Err(msg(format!(
            "pytorch tensor {} storage is smaller than its view (PT-004)",
            named.name
        )));
    }
    let key = &tensor.storage.key;
    let stored = zip
        .by_name(&format!("{prefix}data/{key}"))
        .map_err(|_| msg(format!("{} is missing storage {key}", path.display())))?;
    if stored.size() < tensor.storage_bytes {
        return Err(msg(format!(
            "pytorch storage {key} is shorter than its numel (PT-003)"
        )));
    }
    let compressed = stored.compression() != CompressionMethod::Stored;
    if compressed {
        notes.push(format!("storage {key} is compressed (PT-005)"));
    }
    let contiguous = is_c_order(&tensor.shape, &tensor.strides);
    if contiguous && !compressed && byteorder == "little" && !tensor.conj && !tensor.neg {
        return Ok(Payload::Range {
            path: path.to_path_buf(),
            offset: stored.data_start() + byte_off,
            len: span_bytes,
        });
    }
    // Read only up to the end of this view, not the whole storage.
    let mut stored = stored;
    std::io::copy(&mut (&mut stored).take(byte_off), &mut std::io::sink())
        .map_err(|err| io_err(path, err))?;
    let mut bytes = Vec::new();
    (&mut stored)
        .take(span_bytes)
        .read_to_end(&mut bytes)
        .map_err(|err| io_err(path, err))?;
    if bytes.len() as u64 != span_bytes {
        return Err(msg(format!("pytorch storage {key} slice is out of range")));
    }
    if !contiguous {
        bytes = gather(&bytes, &tensor.shape, &tensor.strides, tensor.itemsize as usize);
    }
    if byteorder == "big" {
        dtype::byteswap(&mut bytes, &tensor.dtype)?;
    }
    if tensor.conj {
        negate(&mut bytes, &tensor.dtype, true)?;
    }
    if tensor.neg {
        negate(&mut bytes, &tensor.dtype, false)?;
    }
    Ok(Payload::Bytes(bytes))
}

/// The elements of a strided view, in C order. `span` starts at the view's
/// first element and holds every element it reads.
fn gather(span: &[u8], shape: &[u64], strides: &[u64], itemsize: usize) -> Vec<u8> {
    let count = shape.iter().product::<u64>() as usize;
    let mut out = Vec::with_capacity(count * itemsize);
    let mut index = vec![0u64; shape.len()];
    for _ in 0..count {
        let at = index
            .iter()
            .zip(strides)
            .map(|(position, stride)| position * stride)
            .sum::<u64>() as usize
            * itemsize;
        out.extend_from_slice(&span[at..at + itemsize]);
        for dim in (0..shape.len()).rev() {
            index[dim] += 1;
            if index[dim] < shape[dim] {
                break;
            }
            index[dim] = 0;
        }
    }
    out
}

/// Negate every element, or with `imag_only` every imaginary part, as
/// torch resolves the bit (PT-010). Floats and complex parts flip the IEEE
/// sign bit, the top bit of each little-endian value's last byte. int8 to
/// int64 and uint8 negate in two's complement with wraparound. torch cannot
/// resolve the bit on other dtypes, so they are refused.
fn negate(bytes: &mut [u8], dtype: &str, imag_only: bool) -> Result<()> {
    let bit = if imag_only { "conj" } else { "neg" };
    let integer = match dtype {
        "int8" | "uint8" => Some(1),
        "int16" => Some(2),
        "int32" => Some(4),
        "int64" => Some(8),
        _ => None,
    };
    if let (Some(width), false) = (integer, imag_only) {
        for value in bytes.chunks_exact_mut(width) {
            // Two's complement: invert, then add one with carry.
            let mut carry = true;
            for byte in value.iter_mut() {
                let (sum, overflow) = (!*byte).overflowing_add(u8::from(carry));
                *byte = sum;
                carry = overflow;
            }
        }
        return Ok(());
    }
    let (width, complex) = match dtype {
        "float16" | "bfloat16" => (2, false),
        "float32" => (4, false),
        "float64" => (8, false),
        "complex64" => (4, true),
        "complex128" => (8, true),
        other => {
            return Err(msg(format!(
                "pytorch {bit} bit on a {other} tensor, which torch cannot resolve either (PT-010)"
            )))
        }
    };
    if imag_only && !complex {
        return Err(msg(format!(
            "pytorch conj bit on a {dtype} tensor, which is not complex (PT-010)"
        )));
    }
    for (index, part) in bytes.chunks_exact_mut(width).enumerate() {
        if !imag_only || index % 2 == 1 {
            part[width - 1] ^= 0x80;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    /// Rewrite `path`'s data.pkl through `wrap`, keeping every other record.
    fn repickle(path: &Path, wrap: impl Fn(&[u8]) -> Vec<u8>) {
        let mut source = zip::ZipArchive::new(File::open(path).unwrap()).unwrap();
        let mut records = Vec::new();
        for index in 0..source.len() {
            let mut entry = source.by_index(index).unwrap();
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).unwrap();
            records.push((entry.name().to_string(), bytes));
        }
        let mut writer = zip::ZipWriter::new(File::create(path).unwrap());
        for (name, bytes) in records {
            let bytes = if name.ends_with("data.pkl") { wrap(&bytes) } else { bytes };
            writer
                .start_file(name, SimpleFileOptions::default().compression_method(CompressionMethod::Stored))
                .unwrap();
            writer.write_all(&bytes).unwrap();
        }
        writer.finish().unwrap();
    }

    #[test]
    fn dtypes_without_a_storage_class_use_rebuild_v3() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("u.pt");
        let tensors = [
            OutTensor {
                name: "u16".into(),
                dtype: "uint16".into(),
                shape: vec![2],
                payload: Payload::Bytes(vec![1, 0, 2, 0]),
            },
            OutTensor {
                name: "f8".into(),
                dtype: "float8_e4m3fn".into(),
                shape: vec![3],
                payload: Payload::Bytes(vec![0x38, 0x40, 0x48]),
            },
        ];
        write_archive(&path, &tensors).unwrap();
        let archive = open(&path).unwrap();
        assert!(matches!(&archive.inner, Inner::Pytorch(file) if file.refused.is_none()));
        for (member, tensor) in archive.members.iter().zip(&tensors) {
            assert_eq!((member.name.as_str(), member.dtype.as_str()), (tensor.name.as_str(), tensor.dtype.as_str()));
            assert_eq!(member.shape, tensor.shape);
            assert_eq!(member.payload.read_vec().unwrap(), tensor.payload.read_vec().unwrap());
        }
    }

    fn names(path: &Path) -> Vec<String> {
        open(path).unwrap().members.into_iter().map(|member| member.name).collect()
    }

    fn values(path: &Path, name: &str) -> Vec<u8> {
        let archive = open(path).unwrap();
        let member = archive.members.iter().find(|member| member.name == name).unwrap();
        member.payload.read_vec().unwrap()
    }

    fn file(path: &Path) -> TorchFile {
        match open(path).unwrap().inner {
            Inner::Pytorch(file) => file,
            #[allow(unreachable_patterns)]
            _ => unreachable!(),
        }
    }

    fn pickle_of(path: &Path) -> Vec<u8> {
        let mut zip = open_zip(path).unwrap();
        let name = (0..zip.len())
            .map(|index| zip.by_index(index).unwrap().name().to_string())
            .find(|name| is_data_pkl(name))
            .unwrap();
        read_entry(&mut zip, &name, path).unwrap()
    }

    fn data_records(path: &Path) -> usize {
        let zip = open_zip(path).unwrap();
        zip.file_names().filter(|name| name.contains("/data/")).count()
    }

    #[test]
    fn a_nested_checkpoint_is_rewritten_with_its_structure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.pt");
        let tensor = OutTensor {
            name: "w".into(),
            dtype: "float32".into(),
            shape: vec![2],
            payload: Payload::Bytes(vec![0; 8]),
        };
        write_archive(&path, &[tensor]).unwrap();
        assert!(file(&path).refused.is_none());
        // {'state_dict': {...}, 'epoch': 3, 'loss': -0.5, 'tags': ['a', None]}
        repickle(&path, |pickle| {
            let mut out = vec![0x80, 2, 0x7d, 0x28];
            put_unicode(&mut out, "state_dict");
            out.extend_from_slice(&pickle[2..pickle.len() - 1]);
            put_unicode(&mut out, "epoch");
            out.extend_from_slice(&[0x4b, 3]);
            put_unicode(&mut out, "loss");
            out.push(0x47);
            out.extend_from_slice(&(-0.5f64).to_be_bytes());
            put_unicode(&mut out, "tags");
            out.extend_from_slice(&[0x5d, 0x28]);
            put_unicode(&mut out, "a");
            out.extend_from_slice(&[0x4e, 0x65, 0x75, 0x2e]);
            out
        });
        let before = pickle_of(&path);
        let archive = open(&path).unwrap();
        assert_eq!(archive.members[0].name, "state_dict.w");
        let Inner::Pytorch(torch) = &archive.inner else { unreachable!() };
        assert!(torch.refused.is_none());
        // Writing the same tensors back gives the same pickle.
        let same = dir.path().join("same.pt");
        let tensors = [OutTensor {
            name: "state_dict.w".into(),
            dtype: "float32".into(),
            shape: vec![2],
            payload: archive.members[0].payload.clone(),
        }];
        let unchanged = BTreeSet::from(["state_dict.w".to_string()]);
        rewrite_archive(torch, &path, &same, &tensors, &unchanged).unwrap();
        assert_eq!(pickle_of(&same), before);
        // A new name under state_dict goes into that dict.
        let added = dir.path().join("added.pt");
        let more = [
            OutTensor {
                name: "state_dict.w".into(),
                dtype: "float32".into(),
                shape: vec![2],
                payload: archive.members[0].payload.clone(),
            },
            OutTensor {
                name: "state_dict.b".into(),
                dtype: "uint16".into(),
                shape: vec![1],
                payload: Payload::Bytes(vec![7, 0]),
            },
        ];
        rewrite_archive(torch, &path, &added, &more, &unchanged).unwrap();
        let reread = open(&added).unwrap();
        assert_eq!(names(&added), ["state_dict.w", "state_dict.b"]);
        assert_eq!(
            reread.metadata,
            [
                ("epoch".to_string(), "3".to_string()),
                ("loss".into(), "-0.5".into()),
                ("tags.0".into(), "a".into())
            ]
        );
        assert_eq!(values(&added, "state_dict.b"), [7, 0]);
    }

    /// `w` as a v2 rebuild of a 2-element float32 storage `key`.
    fn tensor_pickle(out: &mut Vec<u8>, key: &str) {
        let tensor = fresh_ref(
            &OutTensor {
                name: "w".into(),
                dtype: "float32".into(),
                shape: vec![2],
                payload: Payload::Bytes(vec![0; 8]),
            },
            key.into(),
        )
        .unwrap();
        put_tensor(out, &tensor).unwrap();
    }

    /// A checkpoint from `body`, which pickles the root object using storage
    /// "0" for every tensor.
    fn checkpoint(dir: &Path, body: impl Fn(&mut Vec<u8>)) -> std::path::PathBuf {
        let path = dir.join("c.pt");
        let tensor = OutTensor {
            name: "w".into(),
            dtype: "float32".into(),
            shape: vec![2],
            payload: Payload::Bytes(vec![0; 8]),
        };
        write_archive(&path, &[tensor]).unwrap();
        repickle(&path, |_| {
            let mut out = vec![0x80, 2];
            body(&mut out);
            out.push(0x2e);
            out
        });
        path
    }

    fn rewrite(path: &Path, dest: &Path, keep: &[&str], add: &[&str]) -> Result<Vec<String>> {
        let archive = open(path).unwrap();
        let Inner::Pytorch(torch) = &archive.inner else { unreachable!() };
        let mut tensors: Vec<OutTensor> = archive
            .members
            .iter()
            .filter(|member| keep.contains(&member.name.as_str()))
            .map(|member| OutTensor {
                name: member.name.clone(),
                dtype: member.dtype.clone(),
                shape: member.shape.clone(),
                payload: member.payload.clone(),
            })
            .collect();
        let unchanged = tensors.iter().map(|tensor| tensor.name.clone()).collect();
        tensors.extend(add.iter().map(|name| OutTensor {
            name: name.to_string(),
            dtype: "float32".into(),
            shape: vec![1],
            payload: Payload::Bytes(vec![0; 4]),
        }));
        rewrite_archive(torch, path, dest, &tensors, &unchanged)
    }

    #[test]
    fn int_keys_and_list_items_are_named_and_other_keys_are_not() {
        // PT-012: {'opt': {0: w}, 'list': [w, w], None: w}
        let dir = tempfile::tempdir().unwrap();
        let path = checkpoint(dir.path(), |out| {
            out.extend_from_slice(&[0x7d, 0x28]);
            put_unicode(out, "opt");
            out.extend_from_slice(&[0x7d, 0x4b, 0]);
            tensor_pickle(out, "0");
            out.push(0x73);
            put_unicode(out, "list");
            out.extend_from_slice(&[0x5d, 0x28]);
            tensor_pickle(out, "0");
            tensor_pickle(out, "0");
            out.push(0x65);
            out.push(0x4e);
            tensor_pickle(out, "0");
            out.push(0x75);
        });
        let archive = open(&path).unwrap();
        assert_eq!(names(&path), ["opt.0", "list.0", "list.1"]);
        assert!(archive.notes.iter().any(|note| note.contains("root[None] (float32 [2])")), "{:?}", archive.notes);
        // Deleting a list item would rename the next one.
        let dest = dir.path().join("d.pt");
        let err = rewrite(&path, &dest, &["opt.0", "list.1"], &[]).unwrap_err().to_string();
        assert!(err.contains("list or tuple"), "{err}");
        // The last item too, though it renames nothing (PT-013).
        let err = rewrite(&path, &dest, &["opt.0", "list.0"], &[]).unwrap_err().to_string();
        assert!(err.contains("list or tuple"), "{err}");
        // PT-013: opt.x goes into opt, and the unnamed tensor is kept.
        rewrite(&path, &dest, &["opt.0", "list.0", "list.1"], &["opt.x", "top"]).unwrap();
        assert_eq!(names(&dest), ["opt.0", "opt.x", "list.0", "list.1", "top"]);
        assert!(open(&dest).unwrap().notes.iter().any(|note| note.contains("root[None]")));
        // opt.0 deleted from a dict is fine.
        rewrite(&path, &dest, &["list.0", "list.1"], &[]).unwrap();
        assert_eq!(names(&dest), ["list.0", "list.1"]);
    }

    #[test]
    fn a_list_root_or_an_ambiguous_name_refuses_the_edit() {
        // A list root has no dict to add a new tensor to (PT-013).
        let dir = tempfile::tempdir().unwrap();
        let path = checkpoint(dir.path(), |out| {
            out.extend_from_slice(&[0x5d, 0x28]);
            tensor_pickle(out, "0");
            out.push(0x65);
        });
        let dest = dir.path().join("d.pt");
        let err = rewrite(&path, &dest, &["0"], &["x"]).unwrap_err().to_string();
        assert!(err.contains("PT-013"), "{err}");
        // {'a.b': w, 'a': {'b': w}}: refused as a whole (PT-012).
        let path = checkpoint(dir.path(), |out| {
            out.extend_from_slice(&[0x7d, 0x28]);
            put_unicode(out, "a.b");
            tensor_pickle(out, "0");
            put_unicode(out, "a");
            out.push(0x7d);
            put_unicode(out, "b");
            tensor_pickle(out, "0");
            out.push(0x73);
            out.push(0x75);
        });
        assert!(file(&path).refused.unwrap().contains("PT-012"));
    }

    #[test]
    fn a_set_or_a_foreign_global_refuses_a_rewrite() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.pt");
        let tensor = OutTensor {
            name: "w".into(),
            dtype: "float32".into(),
            shape: vec![2],
            payload: Payload::Bytes(vec![0; 8]),
        };
        write_archive(&path, &[tensor]).unwrap();
        repickle(&path, |pickle| {
            let mut out = pickle[..pickle.len() - 2].to_vec();
            put_unicode(&mut out, "seen");
            out.extend_from_slice(&[0x8f, 0x75, 0x2e]);
            out
        });
        assert_eq!(file(&path).refused.as_deref(), Some("a set"));
    }

    #[test]
    fn a_rewrite_keeps_shared_storage_views_and_metadata_bits() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("generated/test-vectors/small-pt-fp8.pt");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fp8.pt");
        let archive = open(&source).unwrap();
        let Inner::Pytorch(torch) = &archive.inner else { unreachable!() };
        assert!(torch.refused.is_none());
        assert_eq!(data_records(&source), 5);
        // tied_a, tied_b, and row1 are views of one storage, counted once.
        let views: u64 = archive.members.iter().map(|member| member.nbytes).sum();
        assert!(archive.data_bytes < views, "{} {views}", archive.data_bytes);
        assert!(archive.notes.iter().any(|note| note.contains("share 5 storages")), "{:?}", archive.notes);
        let keep: Vec<_> = archive
            .members
            .iter()
            .filter(|member| member.name != "f8e5m2")
            .map(|member| OutTensor {
                name: member.name.clone(),
                dtype: member.dtype.clone(),
                shape: member.shape.clone(),
                payload: member.payload.clone(),
            })
            .collect();
        let unchanged = keep.iter().map(|tensor| tensor.name.clone()).collect();
        rewrite_archive(torch, &source, &path, &keep, &unchanged).unwrap();
        // tied_a, tied_b, and row1 still share one record; f8e5m2's is gone.
        assert_eq!(data_records(&path), 4);
        for tensor in &keep {
            assert_eq!(values(&path, &tensor.name), tensor.payload.read_vec().unwrap(), "{}", tensor.name);
        }
        // Replacing tied_a gives it a record of its own; tied_b keeps the old one.
        let replaced = dir.path().join("replaced.pt");
        let mut changed = keep;
        let index = changed.iter().position(|tensor| tensor.name == "tied_a").unwrap();
        changed[index].payload = Payload::Bytes(vec![0; 16]);
        let unchanged = changed
            .iter()
            .filter(|tensor| tensor.name != "tied_a")
            .map(|tensor| tensor.name.clone())
            .collect();
        rewrite_archive(torch, &source, &replaced, &changed, &unchanged).unwrap();
        assert_eq!(data_records(&replaced), 5);
        assert_eq!(values(&replaced, "tied_a"), vec![0; 16]);
        assert_eq!(values(&replaced, "tied_b"), values(&source, "tied_b"));
    }

    #[test]
    fn the_neg_bit_negates_integers_with_wraparound() {
        // PT-010: int8 -128 stays -128, uint8 1 becomes 255.
        let mut int8 = vec![0x80, 1, 0];
        negate(&mut int8, "int8", false).unwrap();
        assert_eq!(int8, [0x80, 0xff, 0]);
        let mut uint8 = vec![1];
        negate(&mut uint8, "uint8", false).unwrap();
        assert_eq!(uint8, [255]);
        let mut int32 = 5i32.to_le_bytes().to_vec();
        negate(&mut int32, "int32", false).unwrap();
        assert_eq!(i32::from_le_bytes(int32.try_into().unwrap()), -5);
        for dtype in ["bool", "uint16", "float8_e4m3fn"] {
            assert!(negate(&mut vec![0; 2], dtype, false).is_err(), "{dtype}");
        }
    }
}
