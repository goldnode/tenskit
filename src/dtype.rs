use crate::error::{msg, Result};

/// A safetensors dtype name and its width in bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafeDtype {
    pub name: &'static str,
    pub bits: u32,
    /// NumPy `descr` for a little-endian array, when one exists.
    pub numpy: Option<&'static str>,
}

const SAFE_DTYPES: &[SafeDtype] = &[
    SafeDtype {
        name: "BOOL",
        bits: 8,
        numpy: Some("|b1"),
    },
    SafeDtype {
        name: "F4",
        bits: 4,
        numpy: None,
    },
    SafeDtype {
        name: "F6_E2M3",
        bits: 6,
        numpy: None,
    },
    SafeDtype {
        name: "F6_E3M2",
        bits: 6,
        numpy: None,
    },
    SafeDtype {
        name: "U8",
        bits: 8,
        numpy: Some("|u1"),
    },
    SafeDtype {
        name: "I8",
        bits: 8,
        numpy: Some("|i1"),
    },
    SafeDtype {
        name: "F8_E5M2",
        bits: 8,
        numpy: None,
    },
    SafeDtype {
        name: "F8_E4M3",
        bits: 8,
        numpy: None,
    },
    SafeDtype {
        name: "F8_E8M0",
        bits: 8,
        numpy: None,
    },
    SafeDtype {
        name: "F8_E4M3FNUZ",
        bits: 8,
        numpy: None,
    },
    SafeDtype {
        name: "F8_E5M2FNUZ",
        bits: 8,
        numpy: None,
    },
    SafeDtype {
        name: "I16",
        bits: 16,
        numpy: Some("<i2"),
    },
    SafeDtype {
        name: "U16",
        bits: 16,
        numpy: Some("<u2"),
    },
    SafeDtype {
        name: "F16",
        bits: 16,
        numpy: Some("<f2"),
    },
    SafeDtype {
        name: "BF16",
        bits: 16,
        numpy: None,
    },
    SafeDtype {
        name: "I32",
        bits: 32,
        numpy: Some("<i4"),
    },
    SafeDtype {
        name: "U32",
        bits: 32,
        numpy: Some("<u4"),
    },
    SafeDtype {
        name: "F32",
        bits: 32,
        numpy: Some("<f4"),
    },
    SafeDtype {
        name: "C64",
        bits: 64,
        numpy: Some("<c8"),
    },
    SafeDtype {
        name: "F64",
        bits: 64,
        numpy: Some("<f8"),
    },
    SafeDtype {
        name: "I64",
        bits: 64,
        numpy: Some("<i8"),
    },
    SafeDtype {
        name: "U64",
        bits: 64,
        numpy: Some("<u8"),
    },
];

pub fn safe_dtype(name: &str) -> Result<SafeDtype> {
    SAFE_DTYPES
        .iter()
        .copied()
        .find(|dtype| dtype.name == name)
        .ok_or_else(|| msg(format!("unknown safetensors dtype {name}")))
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
    let matched = SAFE_DTYPES.iter().copied().find(|dtype| {
        dtype
            .numpy
            .is_some_and(|numpy| numpy == descr || numpy_aliases(dtype.name).contains(&descr))
    });
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

/// GGML tensor type ids from the pinned GGUF spec.
pub fn ggml_type_name(id: u32) -> String {
    let name = match id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        4 => "Q4_2",
        5 => "Q4_3",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        31 => "Q4_0_4_4",
        32 => "Q4_0_4_8",
        33 => "Q4_0_8_8",
        34 => "TQ1_0",
        35 => "TQ2_0",
        36 => "IQ4_NL_4_4",
        37 => "IQ4_NL_4_8",
        38 => "IQ4_NL_8_8",
        39 => "MXFP4",
        40 => "COUNT",
        _ => return format!("ggml_type_{id}"),
    };
    name.to_string()
}

/// Dense GGML types that have a fixed width and a NumPy descr.
pub fn ggml_dense(id: u32) -> Option<SafeDtype> {
    let name = match id {
        0 => "F32",
        1 => "F16",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        30 => "BF16",
        _ => return None,
    };
    safe_dtype(name).ok()
}

pub fn is_quant_ggml(id: u32) -> bool {
    ggml_dense(id).is_none() && id != 40
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
}
