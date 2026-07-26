use std::io::Write;
use std::path::PathBuf;

use crate::error::Result;
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
}
