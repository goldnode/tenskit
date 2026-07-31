use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use crate::error::{io_err, msg, Result};
use crate::io_util;

/// Where a tensor's bytes live. Copies stream from this source and do not
/// load the tensor into a single buffer.
#[derive(Debug, Clone)]
pub enum Payload {
    Range {
        path: PathBuf,
        offset: u64,
        len: u64,
    },
    Zip {
        path: PathBuf,
        entry: String,
        skip: u64,
        len: u64,
    },
    /// Canonical bytes held in memory. Used when a reader has to byteswap.
    Bytes(Vec<u8>),
}

impl Payload {
    pub fn len(&self) -> u64 {
        match self {
            Self::Range { len, .. } | Self::Zip { len, .. } => *len,
            Self::Bytes(bytes) => bytes.len() as u64,
        }
    }

    pub fn copy_to(&self, out: &mut impl Write) -> Result<()> {
        match self {
            Self::Range { path, offset, len } => io_util::copy_range(path, *offset, *len, out),
            Self::Zip {
                path,
                entry,
                skip,
                len,
            } => {
                #[cfg(format_npz)]
                {
                    crate::formats::npz::copy_entry(path, entry, *skip, *len, out)
                }
                #[cfg(not(format_npz))]
                {
                    let _ = (path, entry, skip, len, out);
                    Err(crate::error::msg("npz support is not compiled in"))
                }
            }
            Self::Bytes(bytes) => {
                out.write_all(bytes)?;
                Ok(())
            }
        }
    }

    pub fn read_vec(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.copy_to(&mut bytes)?;
        Ok(bytes)
    }

    /// A prefix of this payload. The tail, including alignment padding, is dropped.
    pub fn clipped(&self, len: u64) -> Result<Self> {
        if len > self.len() {
            return Err(msg("logical payload is longer than the stored bytes"));
        }
        Ok(match self.clone() {
            Self::Range { path, offset, .. } => Self::Range { path, offset, len },
            Self::Zip {
                path, entry, skip, ..
            } => Self::Zip {
                path,
                entry,
                skip,
                len,
            },
            Self::Bytes(bytes) => {
                let end = usize::try_from(len).map_err(|_| msg("payload is too large"))?;
                Self::Bytes(bytes[..end].to_vec())
            }
        })
    }

    /// Read `buf` from `at` bytes into the payload.
    pub fn read_at(&self, at: u64, buf: &mut [u8]) -> Result<()> {
        let end = at
            .checked_add(buf.len() as u64)
            .ok_or_else(|| msg("read is too large"))?;
        if end > self.len() {
            return Err(msg("read runs past the payload"));
        }
        match self {
            Self::Range { path, offset, .. } => {
                let mut file = File::open(path).map_err(|err| io_err(path, err))?;
                file.seek(SeekFrom::Start(offset + at))
                    .map_err(|err| io_err(path, err))?;
                file.read_exact(buf).map_err(|err| io_err(path, err))?;
                Ok(())
            }
            Self::Bytes(bytes) => {
                let start = usize::try_from(at).map_err(|_| msg("read is too large"))?;
                buf.copy_from_slice(&bytes[start..start + buf.len()]);
                Ok(())
            }
            Self::Zip {
                path, entry, skip, ..
            } => {
                #[cfg(format_npz)]
                {
                    let mut got = Vec::new();
                    crate::formats::npz::copy_entry(
                        path,
                        entry,
                        skip + at,
                        buf.len() as u64,
                        &mut got,
                    )?;
                    if got.len() != buf.len() {
                        return Err(msg(format!(
                            "{entry}: expected {} payload bytes, read {}",
                            buf.len(),
                            got.len()
                        )));
                    }
                    buf.copy_from_slice(&got);
                    Ok(())
                }
                #[cfg(not(format_npz))]
                {
                    let _ = (path, entry, skip, at, buf);
                    Err(msg("npz support is not compiled in"))
                }
            }
        }
    }
}
