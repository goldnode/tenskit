use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::payload::Payload;

use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::error::{io_err, msg, Result};
use super::npy::{self, NpyInfo};

#[derive(Debug, Clone)]
pub struct NpzEntry {
    pub key: String,
    pub zip_name: String,
    pub info: NpyInfo,
    pub compressed: bool,
    /// File offset of a STORED member's npy bytes.
    pub data_start: Option<u64>,
}

/// Arrays in an npz archive, in member-name order, plus notes on the members
/// that are not arrays.
///
/// A member is an array only when it starts with the npy magic, whatever its
/// name; any other member is a blob that `np.load` returns as bytes, so it is
/// skipped with a note (NPZ-007). The key is the member name with one
/// trailing `.npy` removed, and `/` is kept (npz.yaml `naming`). Two members
/// with the same key are refused (NPZ-006). An archive with no members is
/// valid (NPZ-005).
pub fn read_entries(path: &Path) -> Result<(Vec<NpzEntry>, Vec<String>)> {
    let file = File::open(path).map_err(|err| io_err(path, err))?;
    let mut archive =
        ZipArchive::new(file).map_err(|err| msg(format!("{}: {err}", path.display())))?;
    let mut names: Vec<String> = archive
        .file_names()
        .filter(|name| !name.ends_with('/'))
        .map(str::to_string)
        .collect();
    names.sort();
    let mut entries: Vec<NpzEntry> = Vec::with_capacity(names.len());
    let mut notes = Vec::new();
    for name in names {
        let mut entry = archive
            .by_name(&name)
            .map_err(|err| msg(format!("{}: {err}", path.display())))?;
        let declared = entry.size();
        if declared > MAX_MEMBER {
            return Err(msg(format!(
                "{name} declares {declared} bytes, over the 64 GiB member limit (NPZ-004)"
            )));
        }
        let compressed = entry.compression() != CompressionMethod::Stored;
        let data_start = (!compressed).then(|| entry.data_start());
        let mut reader = NpyReader {
            inner: &mut entry,
            pos: 0,
            peeked: Vec::new(),
        };
        if !reader.starts_with_npy_magic()? {
            notes.push(format!(
                "member {name} is not an npy array and was skipped (NPZ-007)"
            ));
            continue;
        }
        let info = npy::read_info_from(Path::new(&name), &mut reader)?;
        if info.native_order {
            notes.push(format!(
                "member {name} descr {} names the native byte order; it is read as little-endian (NPY-013)",
                info.descr
            ));
        }
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        if entries.iter().any(|entry| entry.key == key) {
            return Err(msg(format!(
                "{} has two members named {key} after removing .npy (NPZ-006)",
                path.display()
            )));
        }
        entries.push(NpzEntry {
            key,
            zip_name: name,
            info,
            compressed,
            data_start,
        });
    }
    Ok((entries, notes))
}

/// Members that are not npy arrays (NPZ-007), in member-name order.
pub fn blob_names(path: &Path) -> Result<Vec<String>> {
    let (entries, _) = read_entries(path)?;
    let file = File::open(path).map_err(|err| io_err(path, err))?;
    let archive = ZipArchive::new(file).map_err(|err| msg(format!("{}: {err}", path.display())))?;
    let mut names: Vec<String> = archive
        .file_names()
        .filter(|name| !name.ends_with('/') && !entries.iter().any(|entry| entry.zip_name == *name))
        .map(str::to_string)
        .collect();
    names.sort();
    Ok(names)
}

/// NPZ-004: declared member size checked before inflating.
const MAX_MEMBER: u64 = 64 * 1024 * 1024 * 1024;

/// `Read + Seek` adapter. The npy header reader seeks only forward from the start.
struct NpyReader<'a> {
    inner: &'a mut dyn Read,
    pos: u64,
    /// Bytes read ahead by the magic check and not yet returned.
    peeked: Vec<u8>,
}

impl NpyReader<'_> {
    fn starts_with_npy_magic(&mut self) -> std::io::Result<bool> {
        let mut magic = Vec::with_capacity(6);
        self.inner.take(6).read_to_end(&mut magic)?;
        let found = magic == b"\x93NUMPY";
        self.peeked = magic;
        Ok(found)
    }
}

impl Read for NpyReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = if self.peeked.is_empty() {
            self.inner.read(buf)?
        } else {
            let n = self.peeked.len().min(buf.len());
            buf[..n].copy_from_slice(&self.peeked[..n]);
            self.peeked.drain(..n);
            n
        };
        self.pos += n as u64;
        Ok(n)
    }
}

impl std::io::Seek for NpyReader<'_> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        match pos {
            std::io::SeekFrom::Current(0) => Ok(self.pos),
            std::io::SeekFrom::Start(p) if p == self.pos => Ok(self.pos),
            std::io::SeekFrom::Start(p) if p > self.pos => {
                let skip = p - self.pos;
                std::io::copy(&mut self.inner.take(skip), &mut std::io::sink())?;
                self.pos = p;
                Ok(self.pos)
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "npz entries are read forward only",
            )),
        }
    }
}

pub fn copy_whole_entry(path: &Path, zip_name: &str, out: &mut impl Write) -> Result<()> {
    let file = File::open(path).map_err(|err| io_err(path, err))?;
    let mut archive =
        ZipArchive::new(file).map_err(|err| msg(format!("{}: {err}", path.display())))?;
    let mut entry = archive
        .by_name(zip_name)
        .map_err(|err| msg(format!("{}: {err}", path.display())))?;
    std::io::copy(&mut entry, out)?;
    Ok(())
}

pub fn copy_entry(
    path: &Path,
    zip_name: &str,
    skip: u64,
    len: u64,
    out: &mut impl Write,
) -> Result<()> {
    let file = File::open(path).map_err(|err| io_err(path, err))?;
    let mut archive =
        ZipArchive::new(file).map_err(|err| msg(format!("{}: {err}", path.display())))?;
    let mut entry = archive
        .by_name(zip_name)
        .map_err(|err| msg(format!("{}: {err}", path.display())))?;
    let discarded = std::io::copy(&mut (&mut entry).take(skip), &mut std::io::sink())?;
    if discarded != skip {
        return Err(msg(format!("{zip_name}: short npy header")));
    }
    let copied = std::io::copy(&mut (&mut entry).take(len), out)?;
    if copied != len {
        return Err(msg(format!(
            "{zip_name}: expected {len} payload bytes, read {copied}"
        )));
    }
    Ok(())
}

pub enum NpzMemberBody {
    /// New `.npy` header plus a streamed payload.
    Rebuild {
        descr: String,
        shape: Vec<u64>,
        payload: Payload,
    },
    /// Decompressed `.npy` entry, header included.
    WholeZip { path: PathBuf, entry: String },
    /// An `.npy` file copied in full.
    WholeFile(PathBuf),
    /// A member of another `.npz` copied under its own name, header
    /// included; used for members an edit leaves alone, arrays or not.
    Verbatim { path: PathBuf, entry: String },
}

pub struct NpzMember {
    pub key: String,
    pub body: NpzMemberBody,
}

pub fn write_archive(path: &Path, members: &[NpzMember]) -> Result<()> {
    crate::io_util::check_unique_names(members.iter().map(|member| member.key.as_str()))?;
    crate::io_util::with_temp_file(path, |tmp| {
        let file = crate::io_util::create_new(tmp)?;
        let mut writer = ZipWriter::new(file);
        for member in members {
            if member.key.is_empty()
                || member.key.contains('\0')
                || member.key.starts_with('/')
                || member.key.split('/').any(|part| part == "..")
            {
                return Err(msg(format!("invalid npz member {}", member.key)));
            }
            // np.savez writes ZIP64 records (force_zip64), so members over
            // 4 GiB are written the same way.
            let options = SimpleFileOptions::default()
                .compression_method(CompressionMethod::Stored)
                .large_file(true);
            let name = match &member.body {
                NpzMemberBody::Verbatim { entry, .. } => entry.clone(),
                _ => format!("{}.npy", member.key),
            };
            writer
                .start_file(name, options)
                .map_err(|err| msg(format!("npz: {err}")))?;
            match &member.body {
                NpzMemberBody::Rebuild {
                    descr,
                    shape,
                    payload,
                } => {
                    let header = npy::write_header(descr, shape)?;
                    writer.write_all(&header)?;
                    payload.copy_to(&mut writer)?;
                }
                NpzMemberBody::WholeZip { path, entry } | NpzMemberBody::Verbatim { path, entry } => {
                    copy_whole_entry(path, entry, &mut writer)?;
                }
                NpzMemberBody::WholeFile(source) => {
                    let mut input = File::open(source).map_err(|err| io_err(source, err))?;
                    std::io::copy(&mut input, &mut writer)?;
                }
            }
        }
        writer
            .finish()
            .map_err(|err| msg(format!("npz: {err}")))?
            .sync_all()?;
        Ok(())
    })
}
