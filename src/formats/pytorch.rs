//! PyTorch zip checkpoints (`.pt`, `.pth`, `.bin`, `.ckpt`).
//!
//! Loading uses a restricted unpickler: a global outside the tensor rebuild
//! allowlist is rejected, and no code from the file runs. Legacy pickle streams
//! and the old tar container are recognized and refused. Remove this file and
//! the `pytorch` line in `local/supported-formats.txt` to drop the format.

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

#[derive(Clone)]
struct TensorRef {
    dtype: String,
    shape: Vec<u64>,
    strides: Vec<u64>,
    storage_key: String,
    storage_numel: u64,
    offset_elems: u64,
    itemsize: u64,
}

#[derive(Clone)]
enum Obj {
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    Tuple(Vec<Obj>),
    Dict(Vec<(Obj, Obj)>),
    Global { module: String, name: String },
    Storage { type_name: String, key: String, numel: u64 },
    Tensor(TensorRef),
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
        let name = entry.name();
        if name.ends_with("/data.pkl") || name == "data.pkl" {
            return Ok(true);
        }
    }
    Ok(false)
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

fn swap_elements(bytes: &mut [u8], width: usize) {
    if width <= 1 {
        return;
    }
    for chunk in bytes.chunks_exact_mut(width) {
        chunk.reverse();
    }
}

struct NamedTensor {
    name: String,
    tensor: TensorRef,
}

fn flatten(
    obj: &Obj,
    prefix: &str,
    out: &mut Vec<NamedTensor>,
    meta: &mut Vec<(String, String)>,
    notes: &mut Vec<String>,
) {
    let Obj::Dict(pairs) = obj else {
        notes.push("the pickled object is not a dict of tensors".into());
        return;
    };
    for (key, value) in pairs {
        let Obj::Str(key) = key else {
            notes.push("skipped a pytorch dict key that is not a string".into());
            continue;
        };
        let name = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        match value {
            Obj::Tensor(tensor) => out.push(NamedTensor {
                name,
                tensor: tensor.clone(),
            }),
            Obj::Dict(_) => flatten(value, &name, out, meta, notes),
            Obj::Str(text) => meta.push((name, text.clone())),
            Obj::Int(number) => meta.push((name, number.to_string())),
            Obj::Bool(flag) => meta.push((name, flag.to_string())),
            Obj::None => {}
            _ => notes.push(format!("{name} is not a tensor and was skipped")),
        }
    }
}

fn unpickle(bytes: &[u8]) -> Result<Obj> {
    let mut machine = Machine {
        data: bytes,
        index: 0,
        stack: Vec::new(),
        marks: Vec::new(),
        memo: Vec::new(),
    };
    machine.load()
}

struct Machine<'a> {
    data: &'a [u8],
    index: usize,
    stack: Vec<Obj>,
    marks: Vec<usize>,
    memo: Vec<Obj>,
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
                0x7d => self.stack.push(Obj::Dict(Vec::new())),
                0x29 => self.stack.push(Obj::Tuple(Vec::new())),
                0x4e => self.stack.push(Obj::None),
                0x89 => self.stack.push(Obj::Bool(false)),
                0x88 => self.stack.push(Obj::Bool(true)),
                0x71 => {
                    let index = self.byte()? as usize;
                    self.memo_set(index)?;
                }
                0x72 => {
                    let index = self.u32()? as usize;
                    self.memo_set(index)?;
                }
                0x68 => {
                    let index = self.byte()? as usize;
                    self.memo_get(index)?;
                }
                0x6a => {
                    let index = self.u32()? as usize;
                    self.memo_get(index)?;
                }
                0x94 => {
                    let index = self.memo.len();
                    self.memo_set(index)?;
                }
                0x58 => {
                    let len = self.u32()? as usize;
                    let text = self.utf8(len)?;
                    self.stack.push(Obj::Str(text));
                }
                0x8c => {
                    let len = self.byte()? as usize;
                    let text = self.utf8(len)?;
                    self.stack.push(Obj::Str(text));
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
                0x52 => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    self.stack.push(reduce(callable, args)?);
                }
                0x75 => {
                    let items = self.pop_mark()?;
                    self.set_items(items)?;
                }
                0x95 => {
                    self.index = self
                        .index
                        .checked_add(8)
                        .ok_or_else(|| msg("pytorch pickle frame is truncated"))?;
                    if self.index > self.data.len() {
                        return Err(msg("pytorch pickle frame is truncated"));
                    }
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

    fn memo_set(&mut self, index: usize) -> Result<()> {
        let top = self
            .stack
            .last()
            .cloned()
            .ok_or_else(|| msg("pytorch pickle memo has no value"))?;
        if self.memo.len() <= index {
            self.memo.resize(index + 1, Obj::None);
        }
        self.memo[index] = top;
        Ok(())
    }

    fn memo_get(&mut self, index: usize) -> Result<()> {
        let value = self
            .memo
            .get(index)
            .cloned()
            .ok_or_else(|| msg("pytorch pickle memo index is missing"))?;
        self.stack.push(value);
        Ok(())
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
        let Obj::Dict(pairs) = self
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
    let Obj::Global { name, .. } = &items[1] else {
        return Err(msg("pytorch pickle storage type is not a global"));
    };
    let Obj::Str(key) = &items[2] else {
        return Err(msg("pytorch pickle storage key is not a string"));
    };
    let numel = as_u64(&items[4])?;
    Ok(Obj::Storage {
        type_name: name.clone(),
        key: key.clone(),
        numel,
    })
}

fn reduce(callable: Obj, args: Obj) -> Result<Obj> {
    let Obj::Global { module, name } = callable else {
        return Err(msg("pytorch pickle REDUCE target is not a global"));
    };
    let Obj::Tuple(args) = args else {
        return Err(msg("pytorch pickle REDUCE arguments are not a tuple"));
    };
    if module == "collections" && name == "OrderedDict" {
        return Ok(Obj::Dict(Vec::new()));
    }
    if name == "_rebuild_parameter" || name == "_rebuild_parameter_with_state" {
        return args
            .into_iter()
            .find(|arg| matches!(arg, Obj::Tensor(_)))
            .ok_or_else(|| msg("pytorch parameter rebuild has no tensor"));
    }
    if name == "_rebuild_tensor_v2" || name == "_rebuild_tensor" || name == "_rebuild_tensor_v3" {
        return rebuild(&args);
    }
    Err(msg(format!(
        "pytorch pickle reduce {module}.{name} is not a tensor rebuild"
    )))
}

fn rebuild(args: &[Obj]) -> Result<Obj> {
    if args.len() < 4 {
        return Err(msg("pytorch tensor rebuild is missing arguments"));
    }
    let Obj::Storage {
        type_name,
        key,
        numel,
    } = &args[0]
    else {
        return Err(msg("pytorch tensor storage is not a persistent id"));
    };
    let (mut dtype, mut itemsize) = storage_layout(type_name)?;
    if let Some(Obj::Global { name, .. }) = args.iter().find(|arg| {
        matches!(arg, Obj::Global { module, name } if module == "torch" && !name.ends_with("Storage"))
    }) {
        if let Some((canonical, bits)) = torch_dtype(name) {
            dtype = canonical.into();
            itemsize = bits / 8;
        }
    }
    Ok(Obj::Tensor(TensorRef {
        dtype,
        shape: as_shape(&args[2])?,
        strides: as_shape(&args[3])?,
        storage_key: key.clone(),
        storage_numel: *numel,
        offset_elems: as_u64(&args[1])?,
        itemsize,
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

pub fn write_archive(path: &Path, tensors: &[OutTensor]) -> Result<()> {
    if tensors.is_empty() {
        return Err(msg("a pytorch checkpoint needs at least one tensor"));
    }
    let stem = path
        .file_stem()
        .and_then(|text| text.to_str())
        .filter(|text| !text.is_empty())
        .unwrap_or("archive");
    let mut pickle = Vec::new();
    pickle.extend_from_slice(&[0x80, 2, 0x7d, 0x28]);
    for (index, tensor) in tensors.iter().enumerate() {
        let canonical = dtype::canonical_id(&tensor.dtype)?;
        let (storage, itemsize) = canonical_storage(canonical)?;
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
        put_unicode(&mut pickle, &tensor.name);
        put_global(&mut pickle, "torch._utils", "_rebuild_tensor_v2");
        pickle.push(0x28);
        pickle.push(0x28);
        put_unicode(&mut pickle, "storage");
        put_global(&mut pickle, "torch", storage);
        put_unicode(&mut pickle, &index.to_string());
        put_unicode(&mut pickle, "cpu");
        put_int(&mut pickle, count);
        pickle.push(0x74);
        pickle.push(0x51);
        put_int(&mut pickle, 0);
        put_shape(&mut pickle, &tensor.shape);
        put_shape(&mut pickle, &c_strides(&tensor.shape));
        pickle.push(0x89);
        put_global(&mut pickle, "collections", "OrderedDict");
        pickle.push(0x29);
        pickle.push(0x52);
        pickle.push(0x74);
        pickle.push(0x52);
    }
    pickle.push(0x75);
    pickle.push(0x2e);

    crate::io_util::with_temp_file(path, |tmp| {
        let file = File::create(tmp).map_err(|err| io_err(tmp, err))?;
        let mut writer = ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        let put = |writer: &mut ZipWriter<File>, name: &str, bytes: &[u8]| -> Result<()> {
            writer
                .start_file(name, options)
                .map_err(|err| msg(format!("pytorch: {err}")))?;
            writer.write_all(bytes)?;
            Ok(())
        };
        put(&mut writer, &format!("{stem}/data.pkl"), &pickle)?;
        put(&mut writer, &format!("{stem}/.format_version"), b"1")?;
        put(&mut writer, &format!("{stem}/.storage_alignment"), b"64")?;
        put(&mut writer, &format!("{stem}/byteorder"), b"little")?;
        for (index, tensor) in tensors.iter().enumerate() {
            writer
                .start_file(format!("{stem}/data/{index}"), options)
                .map_err(|err| msg(format!("pytorch: {err}")))?;
            tensor.payload.copy_to(&mut writer)?;
        }
        put(&mut writer, &format!("{stem}/version"), b"3\n")?;
        writer
            .finish()
            .map_err(|err| msg(format!("pytorch: {err}")))?
            .sync_all()?;
        Ok(())
    })
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
    } else {
        out.push(0x4a);
        out.extend_from_slice(&(value as i32).to_le_bytes());
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

// The open() above is unfinished: names are produced by flatten, then members
// are built. The stub at the end of open is replaced by this complete builder,
// which is the function archive::open calls.
pub fn open(path: &Path) -> Result<Archive> {
    let file = File::open(path).map_err(|err| io_err(path, err))?;
    let mut zip = ZipArchive::new(file).map_err(|err| msg(format!("{}: {err}", path.display())))?;
    let mut pkl_name = None;
    let mut names = Vec::new();
    for index in 0..zip.len() {
        let entry = zip
            .by_index(index)
            .map_err(|err| msg(format!("{}: {err}", path.display())))?;
        let name = entry.name().to_string();
        if name.ends_with("/data.pkl") || name == "data.pkl" {
            if pkl_name.is_some() {
                return Err(msg(format!(
                    "{} has more than one data.pkl (PT-001)",
                    path.display()
                )));
            }
            pkl_name = Some(name.clone());
        }
        names.push(name);
    }
    let pkl_name = pkl_name.ok_or_else(|| {
        msg(format!(
            "{} is not a pytorch zip; it has no data.pkl",
            path.display()
        ))
    })?;
    let prefix = pkl_name.trim_end_matches("data.pkl").to_string();
    if names.iter().any(|name| !name.starts_with(&prefix)) {
        return Err(msg(format!(
            "{} mixes zip prefixes (PT-001)",
            path.display()
        )));
    }
    let pickle = read_entry(&mut zip, &pkl_name, path)?;
    let obj = unpickle(&pickle)?;
    let mut found = Vec::new();
    let mut metadata = Vec::new();
    let mut notes = Vec::new();
    flatten(&obj, "", &mut found, &mut metadata, &mut notes);
    if found.is_empty() {
        return Err(msg(format!("{} pickle has no tensors", path.display())));
    }
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
    let mut members = Vec::with_capacity(found.len());
    for named in found {
        let tensor = named.tensor;
        if !is_c_order(&tensor.shape, &tensor.strides) {
            return Err(msg(format!(
                "pytorch tensor {} is strided; tenspile does not gather it",
                named.name
            )));
        }
        let count = numel(&tensor.shape).ok_or_else(|| msg("shape is too large to count"))?;
        let end = tensor
            .offset_elems
            .checked_add(count)
            .ok_or_else(|| msg("pytorch storage offset overflows"))?;
        if tensor.storage_numel < end {
            return Err(msg(format!(
                "pytorch tensor {} storage is smaller than its view (PT-004)",
                named.name
            )));
        }
        let nbytes = count
            .checked_mul(tensor.itemsize)
            .ok_or_else(|| msg("pytorch tensor is too large"))?;
        let entry_name = format!("{prefix}data/{}", tensor.storage_key);
        let stored = zip.by_name(&entry_name).map_err(|_| {
            msg(format!(
                "{} is missing storage {}",
                path.display(),
                tensor.storage_key
            ))
        })?;
        let need = tensor
            .storage_numel
            .checked_mul(tensor.itemsize)
            .ok_or_else(|| msg("pytorch storage is too large"))?;
        if stored.size() < need {
            return Err(msg(format!(
                "pytorch storage {} is shorter than its numel (PT-003)",
                tensor.storage_key
            )));
        }
        let compressed = stored.compression() != CompressionMethod::Stored;
        if compressed {
            notes.push(format!("storage {} is compressed (PT-005)", tensor.storage_key));
        }
        let byte_off = tensor.offset_elems * tensor.itemsize;
        let payload = if compressed || byteorder == "big" {
            let mut raw = Vec::new();
            let mut stored = stored;
            stored.read_to_end(&mut raw).map_err(|err| io_err(path, err))?;
            let end_byte = (byte_off + nbytes) as usize;
            let slice = raw.get(byte_off as usize..end_byte).ok_or_else(|| {
                msg(format!("pytorch storage {} slice is out of range", tensor.storage_key))
            })?;
            let mut bytes = slice.to_vec();
            if byteorder == "big" && tensor.itemsize > 1 {
                swap_elements(&mut bytes, tensor.itemsize as usize);
            }
            Payload::Bytes(bytes)
        } else {
            Payload::Range {
                path: path.to_path_buf(),
                offset: stored.data_start() + byte_off,
                len: nbytes,
            }
        };
        let file_offset = match &payload {
            Payload::Range { offset, .. } => *offset,
            _ => 0,
        };
        members.push(Member {
            name: named.name,
            dtype: tensor.dtype,
            shape: tensor.shape,
            file_offset,
            nbytes,
            numel: Some(count),
            payload,
            shard: None,
            detail: None,
            fortran: false,
        });
    }
    let data_bytes = members
        .iter()
        .fold(0u64, |acc, member| acc.saturating_add(member.nbytes));
    Ok(Archive {
        format: FormatName::Pytorch,
        version,
        path: path.to_path_buf(),
        members,
        metadata,
        data_bytes,
        notes,
        inner: Inner::Pytorch,
    })
}
