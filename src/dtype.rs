use crate::error::{msg, Result};

include!(concat!(env!("OUT_DIR"), "/spec_tables.rs"));

/// A safetensors dtype name and its width in bits.
///
/// The rows come from `dtypes.yaml` joined with `dtype-mapping.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafeDtype {
    pub name: &'static str,
    pub bits: u32,
    /// NumPy `descr` for a little-endian array, when the mapping has one.
    pub numpy: Option<&'static str>,
    /// Canonical id in `dtypes.yaml`.
    pub canonical: &'static str,
}

fn row_to_safe(row: &SafeRow) -> SafeDtype {
    SafeDtype {
        name: row.name,
        bits: row.bits,
        numpy: row.numpy,
        canonical: row.canonical,
    }
}

pub fn safe_dtype(name: &str) -> Result<SafeDtype> {
    SAFE_ROWS
        .iter()
        .find(|row| row.name == name || row.canonical == name)
        .map(row_to_safe)
        .ok_or_else(|| msg(format!("unknown safetensors dtype {name}")))
}

/// Canonical id for a safetensors name, a ggml type name, or an id that is already canonical.
pub fn canonical_id(name: &str) -> Result<&'static str> {
    if let Some(row) = SAFE_ROWS.iter().find(|row| row.name == name || row.canonical == name) {
        return Ok(row.canonical);
    }
    if let Some(row) = GGML_TYPES.iter().find(|row| row.name == name) {
        return row.canonical.ok_or_else(|| {
            msg(format!(
                "ggml type {name} has no canonical dtype and cannot be stored"
            ))
        });
    }
    if CANON.iter().any(|row| row.id == name) {
        return Ok(CANON.iter().find(|row| row.id == name).unwrap().id);
    }
    Err(msg(format!("unknown dtype {name}")))
}

/// Byte length of a canonical tensor. Block dtypes use the last axis as the block axis.
pub fn canonical_nbytes(id: &str, shape: &[u64]) -> Result<u64> {
    let row = CANON
        .iter()
        .find(|row| row.id == id)
        .ok_or_else(|| msg(format!("unknown canonical dtype {id}")))?;
    let elements = numel(shape).ok_or_else(|| msg("shape is too large to count"))?;
    if row.kind == "block" {
        if shape.is_empty() {
            return Err(msg(format!("block dtype {id} needs a shape")));
        }
        let inner = *shape.last().unwrap();
        let block = u64::from(row.block_elements);
        if block == 0 || inner % block != 0 {
            return Err(msg(format!(
                "block dtype {id} needs the last dimension to be a multiple of {}",
                row.block_elements
            )));
        }
        let blocks = elements / block;
        return blocks
            .checked_mul(u64::from(row.block_bytes))
            .ok_or_else(|| msg("tensor is too large"));
    }
    let bits = elements
        .checked_mul(u64::from(row.bits_num))
        .ok_or_else(|| msg("tensor is too large"))?;
    let denom = u64::from(row.bits_den) * 8;
    Ok(bits.div_ceil(denom))
}

pub fn canon_kind(id: &str) -> Option<&'static str> {
    CANON.iter().find(|row| row.id == id).map(|row| row.kind)
}

pub fn canon_bits(id: &str) -> Option<u32> {
    let row = CANON.iter().find(|row| row.id == id)?;
    if row.bits_den == 1 {
        Some(row.bits_num)
    } else {
        None
    }
}

pub fn numel(shape: &[u64]) -> Option<u64> {
    if shape.is_empty() {
        return Some(1);
    }
    shape
        .iter()
        .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
}

/// Packed byte length. Errors when the bit length is not a whole number of bytes.
pub fn packed_bytes(dtype: SafeDtype, shape: &[u64]) -> Result<u64> {
    let elements = numel(shape).ok_or_else(|| msg("shape is too large to count"))?;
    let bits = elements
        .checked_mul(u64::from(dtype.bits))
        .ok_or_else(|| msg("tensor is too large"))?;
    if bits % 8 != 0 {
        return Err(msg(format!(
            "dtype {} and shape {shape:?} do not fill a whole number of bytes",
            dtype.name
        )));
    }
    Ok(bits / 8)
}

/// Map a simple NumPy `descr` onto a safetensors dtype.
///
/// Big-endian descriptors are refused. Fortran-order is checked by the caller.
pub fn numpy_descr_to_safe(descr: &str) -> Result<SafeDtype> {
    if descr.starts_with('[') || descr.starts_with('{') || descr.starts_with('(') {
        return Err(msg(format!(
            "structured numpy dtype {descr} is not a single tensor"
        )));
    }
    let kind = descr_kind(descr)?;
    if kind == "O" {
        return Err(msg(
            "numpy object dtype is a pickle payload and is not opened",
        ));
    }
    let matched = SAFE_ROWS
        .iter()
        .find(|row| {
            row.numpy
                .is_some_and(|numpy| numpy == descr || numpy_aliases(row.name).contains(&descr))
        })
        .map(row_to_safe);
    matched.ok_or_else(|| {
        msg(format!(
            "numpy descr {descr} has no safetensors dtype; pass --dtype and --format raw"
        ))
    })
}

fn numpy_aliases(safe_name: &str) -> &'static [&'static str] {
    match safe_name {
        "BOOL" => &["?", "|b1", "|?"],
        "U8" => &["|u1", "u1", "<u1"],
        "I8" => &["|i1", "i1", "<i1"],
        _ => &[],
    }
}

fn descr_kind(descr: &str) -> Result<&str> {
    let rest = descr.strip_prefix(['<', '>', '|', '=']).unwrap_or(descr);
    if rest.is_empty() {
        return Err(msg(format!("unreadable numpy descr {descr}")));
    }
    // Kind is the leading letter. Size is the trailing digits.
    let kind_end = rest
        .find(|ch: char| ch.is_ascii_digit())
        .unwrap_or(rest.len());
    Ok(&rest[..kind_end])
}

pub fn format_shape(shape: &[u64]) -> String {
    if shape.is_empty() {
        return "[]".to_string();
    }
    let body = shape
        .iter()
        .map(|dim| dim.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{body}]")
}

pub fn ggml_row(id: u32) -> Option<&'static GgmlRow> {
    GGML_TYPES.iter().find(|row| row.code == id)
}

/// Refuse removed codes and the COUNT sentinel. GGUF-011.
pub fn require_ggml(id: u32) -> Result<&'static GgmlRow> {
    if id == GGML_TYPE_COUNT {
        return Err(msg(
            "ggml type COUNT is a sentinel, not a tensor type",
        ));
    }
    let row = ggml_row(id).ok_or_else(|| msg(format!("unknown ggml type {id}")))?;
    if row.removed {
        return Err(msg(format!(
            "ggml type {} ({id}) was removed from GGUF and is rejected",
            row.name
        )));
    }
    Ok(row)
}

/// GGML tensor type names from `formats/gguf.yaml`.
pub fn ggml_type_name(id: u32) -> String {
    if id == GGML_TYPE_COUNT {
        return "COUNT".to_string();
    }
    ggml_row(id)
        .map(|row| row.name.to_string())
        .unwrap_or_else(|| format!("ggml_type_{id}"))
}

/// Element GGML types that map onto a safetensors dtype.
pub fn ggml_dense(id: u32) -> Option<SafeDtype> {
    let row = ggml_row(id)?;
    if !row.element || row.removed {
        return None;
    }
    safe_dtype(row.name).ok()
}

pub fn is_quant_ggml(id: u32) -> bool {
    ggml_row(id).is_some_and(|row| !row.element && !row.removed)
}

/// On-disk payload size for a ggml tensor. `shape` is file order (`ne[]`).
pub fn ggml_payload_bytes(id: u32, shape: &[u64]) -> Result<u64> {
    let row = require_ggml(id)?;
    if row.element {
        let dtype = safe_dtype(row.name)?;
        return packed_bytes(dtype, shape);
    }
    if shape.is_empty() {
        return Err(msg(format!(
            "ggml type {} needs a dimension for its blocks",
            row.name
        )));
    }
    let inner = shape[0];
    let block = u64::from(row.block_elements);
    if block == 0 || inner % block != 0 {
        return Err(msg(format!(
            "ggml type {} needs ne[0] to be a multiple of {}",
            row.name, row.block_elements
        )));
    }
    let elements = numel(shape).ok_or_else(|| msg("shape is too large to count"))?;
    elements
        .checked_div(block)
        .and_then(|blocks| blocks.checked_mul(u64::from(row.block_bytes)))
        .ok_or_else(|| msg("tensor is too large"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_f4_needs_an_even_nibble_count() {
        let f4 = safe_dtype("F4").unwrap();
        assert!(packed_bytes(f4, &[1]).is_err());
        assert_eq!(packed_bytes(f4, &[2]).unwrap(), 1);
    }

    #[test]
    fn rank_zero_is_one_element() {
        assert_eq!(numel(&[]), Some(1));
        assert_eq!(packed_bytes(safe_dtype("F32").unwrap(), &[]).unwrap(), 4);
    }

    #[test]
    fn registry_comes_from_the_specs() {
        assert_eq!(SPEC_FORMATS.len(), 23);
        let mut ids: Vec<_> = SPEC_FORMATS.iter().map(|row| row.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 23);
        assert!(SPEC_FORMATS.iter().any(|row| row.id == "safetensors" && row.depth == "full"));
        assert!(SPEC_FORMATS.iter().any(|row| row.id == "raw-sidecar" && row.depth == "full"));
        assert_eq!(safe_dtype("F32").unwrap().canonical, "float32");
        assert_eq!(safe_dtype("float32").unwrap().name, "F32");
        assert_eq!(ggml_type_name(40), "NVFP4");
        assert_eq!(ggml_type_name(GGML_TYPE_COUNT), "COUNT");
        assert!(require_ggml(4).is_err());
        assert_eq!(canonical_nbytes("float32", &[4, 64]).unwrap(), 1024);
        assert_eq!(canonical_nbytes("ggml_q8_0", &[4, 64]).unwrap(), 8 * 34);
    }
}
