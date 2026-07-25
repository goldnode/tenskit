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
}

impl Payload {
    pub fn len(&self) -> u64 {
        match self {
            Self::Range { len, .. } | Self::Zip { len, .. } => *len,
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
            } => crate::npz::copy_entry(path, entry, *skip, *len, out),
        }
    }
}
