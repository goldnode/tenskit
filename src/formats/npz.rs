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
}

pub fn read_entries(path: &Path) -> Result<Vec<NpzEntry>> {
    let file = File::open(path).map_err(|err| io_err(path, err))?;
    let mut archive =
        ZipArchive::new(file).map_err(|err| msg(format!("{}: {err}", path.display())))?;
    let mut names = Vec::new();
    let mut saw_pickle = false;
    for index in 0..archive.len() {
        let entry = archive
            .by_index(index)
            .map_err(|err| msg(format!("{}: {err}", path.display())))?;
        let name = entry.name().to_string();
        if name.ends_with(".pkl") || name.ends_with("/data.pkl") {
            saw_pickle = true;
        }
        if name.ends_with(".npy") && !name.ends_with('/') {
            names.push(name);
        }
    }
    if names.is_empty() {
        if saw_pickle {
            return Err(msg(format!(
                "{} is a PyTorch pickle archive and was not opened",
                path.display()
            )));
        }
        return Err(msg(format!("{} has no .npy members", path.display())));
    }
    names.sort();
    let mut entries = Vec::with_capacity(names.len());
    for name in names {
        let mut entry = archive
            .by_name(&name)
            .map_err(|err| msg(format!("{}: {err}", path.display())))?;
        let declared = entry.size();
        if declared > 64 * 1024 * 1024 * 1024 {
            return Err(msg(format!("{name} declares an implausible size")));
        }
        let compressed = entry.compression() != CompressionMethod::Stored;
        let info = npy::read_info_from(
            Path::new(&name),
            &mut NpyReader {
                inner: &mut entry,
                pos: 0,
            },
        )?;
        let key = name
            .trim_end_matches(".npy")
            .rsplit('/')
            .next()
            .unwrap_or(&name)
            .to_string();
        if key.is_empty() || key.contains("..") {
            return Err(msg(format!("npz member {name} is not a safe name")));
        }
        entries.push(NpzEntry {
            key,
            zip_name: name,
            info,
            compressed,
        });
    }
    Ok(entries)
}

/// `Read + Seek` adapter. The npy header reader seeks only forward from the start.
struct NpyReader<'a> {
    inner: &'a mut dyn Read,
    pos: u64,
}

impl Read for NpyReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
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
}

pub struct NpzMember {
    pub key: String,
    pub body: NpzMemberBody,
}

pub fn write_archive(path: &Path, members: &[NpzMember]) -> Result<()> {
    if members.is_empty() {
        return Err(msg("an npz archive needs at least one array"));
    }
    crate::io_util::with_temp_file(path, |tmp| {
        let file = File::create(tmp).map_err(|err| io_err(tmp, err))?;
        let mut writer = ZipWriter::new(file);
        for member in members {
            if member.key.is_empty() || member.key.contains('\0') || member.key.contains("..") {
                return Err(msg(format!("invalid npz member {}", member.key)));
            }
            let options =
                SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
            writer
                .start_file(format!("{}.npy", member.key), options)
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
                NpzMemberBody::WholeZip { path, entry } => {
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
