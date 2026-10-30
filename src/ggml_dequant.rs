//! GGML block dequantization from `formats/gguf.yaml` `dtypes[].dequantize`.
//!
//! Q8_1 is marked inferred and is refused. Lookup grids come from
//! `tables/ggml-lookup-tables.json`.

use std::sync::OnceLock;

use serde_json::Value;

use crate::dtype::{self, ggml_row};
use crate::error::{msg, Result};

const TABLES: &str = include_str!("../generated/tables/ggml-lookup-tables.json");

struct Grids {
    kmask_iq2xs: Vec<u32>,
    ksigns_iq2xs: Vec<u32>,
    kvalues_iq4nl: Vec<f32>,
    kvalues_fp4: Vec<f32>,
    iq2xxs_grid: Vec<Vec<f32>>,
    iq2xs_grid: Vec<Vec<f32>>,
    iq2s_grid: Vec<Vec<f32>>,
    iq3xxs_grid: Vec<Vec<f32>>,
    iq3s_grid: Vec<Vec<f32>>,
    iq1s_grid: Vec<Vec<f32>>,
}

fn grids() -> &'static Grids {
    static GRIDS: OnceLock<Grids> = OnceLock::new();
    GRIDS.get_or_init(load_grids)
}

fn load_grids() -> Grids {
    let value: Value = serde_json::from_str(TABLES).expect("ggml-lookup-tables.json");
    let tables = value.get("tables").expect("tables");
    Grids {
        kmask_iq2xs: ints(tables, "kmask_iq2xs"),
        ksigns_iq2xs: ints(tables, "ksigns_iq2xs"),
        kvalues_iq4nl: floats(tables, "kvalues_iq4nl"),
        kvalues_fp4: floats(tables, "kvalues_fp4"),
        iq2xxs_grid: grid_rows(tables, "iq2xxs_grid"),
        iq2xs_grid: grid_rows(tables, "iq2xs_grid"),
        iq2s_grid: grid_rows(tables, "iq2s_grid"),
        iq3xxs_grid: grid_rows(tables, "iq3xxs_grid"),
        iq3s_grid: grid_rows(tables, "iq3s_grid"),
        iq1s_grid: grid_rows(tables, "iq1s_grid"),
    }
}

fn ints(tables: &Value, name: &str) -> Vec<u32> {
    tables[name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("{name} values"))
        .iter()
        .map(|item| item.as_u64().unwrap() as u32)
        .collect()
}

fn floats(tables: &Value, name: &str) -> Vec<f32> {
    tables[name]["values"]
        .as_array()
        .unwrap_or_else(|| panic!("{name} values"))
        .iter()
        .map(|item| item.as_f64().unwrap() as f32)
        .collect()
}

fn grid_rows(tables: &Value, name: &str) -> Vec<Vec<f32>> {
    tables[name]["bytes_le"]
        .as_array()
        .unwrap_or_else(|| panic!("{name} bytes"))
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|item| item.as_i64().unwrap() as f32)
                .collect()
        })
        .collect()
}

fn at<'a>(grid: &'a [Vec<f32>], index: usize, name: &str) -> Result<&'a [f32]> {
    grid.get(index)
        .map(Vec::as_slice)
        .ok_or_else(|| msg(format!("{name} index {index} is outside the lookup table")))
}

/// Dequantize one tensor's logical block bytes to f32, in stored element order.
pub fn dequant_tensor(ggml_type: u32, raw: &[u8]) -> Result<Vec<f32>> {
    if ggml_type == 9 {
        return Err(msg("GGUF Q8_1 dequant is inferred and is refused"));
    }
    let row = ggml_row(ggml_type).ok_or_else(|| msg(format!("unknown ggml type {ggml_type}")))?;
    if row.removed {
        return Err(msg(format!(
            "ggml type {} was removed from GGUF and is rejected",
            row.name
        )));
    }
    if row.element {
        return element_f32(row.canonical.unwrap_or(row.name), raw);
    }
    let block = row.block_bytes as usize;
    let width = row.block_elements as usize;
    if block == 0 || raw.len() % block != 0 {
        return Err(msg(format!(
            "ggml type {} block is {block} bytes and the payload is {} bytes",
            row.name,
            raw.len()
        )));
    }
    let mut out = Vec::with_capacity(raw.len() / block * width);
    for chunk in raw.chunks(block) {
        out.extend(dequant_block(ggml_type, chunk)?);
    }
    Ok(out)
}

fn element_f32(canonical: &str, raw: &[u8]) -> Result<Vec<f32>> {
    match canonical {
        "float32" => Ok(raw
            .chunks_exact(4)
            .map(|item| f32::from_le_bytes([item[0], item[1], item[2], item[3]]))
            .collect()),
        "float16" => Ok(raw
            .chunks_exact(2)
            .map(|item| f16_to_f32(u16::from_le_bytes([item[0], item[1]])))
            .collect()),
        "bfloat16" => Ok(raw
            .chunks_exact(2)
            .map(|item| f32::from_bits(u32::from(u16::from_le_bytes([item[0], item[1]])) << 16))
            .collect()),
        "float64" => Ok(raw
            .chunks_exact(8)
            .map(|item| f64::from_le_bytes(item.try_into().unwrap()) as f32)
            .collect()),
        "int8" => Ok(raw.iter().map(|byte| *byte as i8 as f32).collect()),
        "int16" => Ok(raw
            .chunks_exact(2)
            .map(|item| i16::from_le_bytes([item[0], item[1]]) as f32)
            .collect()),
        "int32" => Ok(raw
            .chunks_exact(4)
            .map(|item| i32::from_le_bytes(item.try_into().unwrap()) as f32)
            .collect()),
        "int64" => Ok(raw
            .chunks_exact(8)
            .map(|item| i64::from_le_bytes(item.try_into().unwrap()) as f32)
            .collect()),
        other => Err(msg(format!(
            "ggml element type {other} is not widened to float32 here"
        ))),
    }
}

fn dequant_block(code: u32, b: &[u8]) -> Result<Vec<f32>> {
    let tables = grids();
    Ok(match code {
        2 => q4_0(b),
        3 => q4_1(b),
        6 => q5_0(b),
        7 => q5_1(b),
        8 => q8_0(b),
        10 => q2_k(b),
        11 => q3_k(b),
        12 => q4_k(b),
        13 => q5_k(b),
        14 => q6_k(b),
        15 => q8_k(b),
        16 => iq2_xxs(b, tables)?,
        17 => iq2_xs(b, tables)?,
        22 => iq2_s(b, tables)?,
        18 => iq3_xxs(b, tables)?,
        21 => iq3_s(b, tables)?,
        19 => iq1_s(b, tables)?,
        29 => iq1_m(b, tables)?,
        20 => iq4_nl(b, tables),
        23 => iq4_xs(b, tables),
        34 => tq1_0(b),
        35 => tq2_0(b),
        39 => mxfp4(b, tables),
        40 => nvfp4(b, tables),
        41 => q1_0(b),
        42 => q2_0(b),
        other => {
            let name = dtype::ggml_type_name(other);
            return Err(msg(format!("ggml type {name} has no dequantizer")));
        }
    })
}

fn f16_at(b: &[u8], offset: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[offset], b[offset + 1]]))
}

fn f32_at(b: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(b[offset..offset + 4].try_into().unwrap())
}

fn u16_at(b: &[u8], offset: usize) -> u32 {
    u16::from_le_bytes([b[offset], b[offset + 1]]) as u32
}

fn u32_at(b: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(b[offset..offset + 4].try_into().unwrap())
}

fn i8_at(b: &[u8], offset: usize) -> f32 {
    b[offset] as i8 as f32
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let man = bits & 0x3ff;
    let out = if exp == 0 {
        if man == 0 {
            (sign as u32) << 31
        } else {
            let mut value = (man as f32) * 2f32.powi(-24);
            if sign == 1 {
                value = -value;
            }
            return value;
        }
    } else if exp == 31 {
        if man == 0 {
            ((sign as u32) << 31) | 0x7f80_0000
        } else {
            0x7fc0_0000 | ((sign as u32) << 31) | ((man as u32) << 13)
        }
    } else {
        let exp = u32::from(exp) + 127 - 15;
        ((sign as u32) << 31) | (exp << 23) | ((man as u32) << 13)
    };
    f32::from_bits(out)
}

/// f32 to IEEE binary16 bits: round to nearest, ties to even, on the exact
/// f32 value (CONV-001). Subnormal results are kept, overflow becomes inf,
/// and a NaN stays a quiet NaN.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man == 0 { 0 } else { 0x200 | (man >> 13) as u16 };
    }
    let exp = exp - 127 + 15;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        // Subnormal result in units of 2^-24.
        let full = man | 0x80_0000;
        let shift = (14 - exp) as u32;
        if shift > 24 {
            return sign;
        }
        return sign | round_shift(full, shift) as u16;
    }
    let packed = ((exp as u32) << 10) | (man >> 13);
    // A carry out of the mantissa raises the exponent, up to inf.
    sign | round_tail(packed, man & 0x1fff, 13) as u16
}

/// f32 to bfloat16 bits with the same rounding (CONV-001). A NaN is quieted
/// so that dropping the low bits cannot turn it into inf.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) as u16) | 0x40;
    }
    round_tail(bits >> 16, bits & 0xffff, 16) as u16
}

/// `value >> shift`, rounded to nearest with ties to even.
fn round_shift(value: u32, shift: u32) -> u32 {
    round_tail(value >> shift, value & ((1 << shift) - 1), shift)
}

/// Round `kept` up when the dropped `tail` (of `width` bits) is above half,
/// or exactly half and `kept` is odd.
fn round_tail(kept: u32, tail: u32, width: u32) -> u32 {
    let half = 1u32 << (width - 1);
    if tail > half || (tail == half && kept & 1 == 1) {
        kept + 1
    } else {
        kept
    }
}

fn e8m0_half(x: u8) -> f32 {
    let bits = if x < 2 {
        0x0020_0000u32 << x
    } else {
        (u32::from(x) - 1) << 23
    };
    f32::from_bits(bits)
}

fn ue4m3_half(x: u8) -> f32 {
    if x == 0 || x == 0x7f {
        return 0.0;
    }
    let exp = (x >> 3) & 15;
    let man = x & 7;
    let mag = if exp == 0 {
        (man as f32) * 2f32.powi(-9)
    } else {
        (1.0 + (man as f32) / 8.0) * 2f32.powi(i32::from(exp) - 7)
    };
    mag * 0.5
}

fn q4_0(b: &[u8]) -> Vec<f32> {
    let d = f16_at(b, 0);
    let mut y = vec![0.0; 32];
    for j in 0..16 {
        y[j] = ((b[2 + j] & 0x0f) as f32 - 8.0) * d;
        y[j + 16] = ((b[2 + j] >> 4) as f32 - 8.0) * d;
    }
    y
}

fn q4_1(b: &[u8]) -> Vec<f32> {
    let d = f16_at(b, 0);
    let m = f16_at(b, 2);
    let mut y = vec![0.0; 32];
    for j in 0..16 {
        y[j] = (b[4 + j] & 0x0f) as f32 * d + m;
        y[j + 16] = (b[4 + j] >> 4) as f32 * d + m;
    }
    y
}

fn q5_0(b: &[u8]) -> Vec<f32> {
    let d = f16_at(b, 0);
    let qh = u32_at(b, 2);
    let mut y = vec![0.0; 32];
    for j in 0..16 {
        let xh0 = ((qh >> j) << 4) & 0x10;
        let xh1 = (qh >> (j + 12)) & 0x10;
        y[j] = (((b[6 + j] & 0x0f) as u32 | xh0) as f32 - 16.0) * d;
        y[j + 16] = (((b[6 + j] >> 4) as u32 | xh1) as f32 - 16.0) * d;
    }
    y
}

fn q5_1(b: &[u8]) -> Vec<f32> {
    let d = f16_at(b, 0);
    let m = f16_at(b, 2);
    let qh = u32_at(b, 4);
    let mut y = vec![0.0; 32];
    for j in 0..16 {
        let xh0 = ((qh >> j) << 4) & 0x10;
        let xh1 = (qh >> (j + 12)) & 0x10;
        y[j] = ((b[8 + j] & 0x0f) as u32 | xh0) as f32 * d + m;
        y[j + 16] = ((b[8 + j] >> 4) as u32 | xh1) as f32 * d + m;
    }
    y
}

fn q8_0(b: &[u8]) -> Vec<f32> {
    let d = f16_at(b, 0);
    (0..32).map(|j| i8_at(b, 2 + j) * d).collect()
}

fn q2_k(b: &[u8]) -> Vec<f32> {
    let sc = &b[0..16];
    let qs = &b[16..80];
    let d = f16_at(b, 80);
    let dmin = f16_at(b, 82);
    let mut y = Vec::with_capacity(256);
    let mut i = 0;
    let mut q = 0;
    for _n in 0..2 {
        let mut shift = 0;
        for _j in 0..4 {
            let s = sc[i];
            i += 1;
            let dl = d * f32::from(s & 0x0f);
            let ml = dmin * f32::from(s >> 4);
            for l in 0..16 {
                y.push(dl * f32::from((qs[q + l] >> shift) & 3) - ml);
            }
            let s = sc[i];
            i += 1;
            let dl = d * f32::from(s & 0x0f);
            let ml = dmin * f32::from(s >> 4);
            for l in 0..16 {
                y.push(dl * f32::from((qs[q + l + 16] >> shift) & 3) - ml);
            }
            shift += 2;
        }
        q += 32;
    }
    y
}

fn q3_k(b: &[u8]) -> Vec<f32> {
    let hm = &b[0..32];
    let qs = &b[32..96];
    let d = f16_at(b, 108);
    let mut a = [u32_at(b, 96), u32_at(b, 100), u32_at(b, 104), 0];
    let km1 = 0x0303_0303u32;
    let km2 = 0x0f0f_0f0fu32;
    let tmp = a[2];
    a[2] = ((a[0] >> 4) & km2) | (((tmp >> 4) & km1) << 4);
    a[3] = ((a[1] >> 4) & km2) | (((tmp >> 6) & km1) << 4);
    a[0] = (a[0] & km2) | (((tmp >> 0) & km1) << 4);
    a[1] = (a[1] & km2) | (((tmp >> 2) & km1) << 4);
    let mut scales = [0i32; 16];
    for (index, word) in a.iter().enumerate() {
        let bytes = word.to_le_bytes();
        for (offset, byte) in bytes.iter().enumerate() {
            let s = i32::from(*byte);
            scales[index * 4 + offset] = if s > 127 { s - 256 } else { s };
        }
    }
    let mut y = Vec::with_capacity(256);
    let mut i = 0;
    let mut q = 0;
    let mut m = 1u8;
    for _n in 0..2 {
        let mut shift = 0;
        for _j in 0..4 {
            let dl = d * (scales[i] - 32) as f32;
            i += 1;
            for l in 0..16 {
                let high = if hm[l] & m == 0 { 4 } else { 0 };
                y.push(dl * ((((qs[q + l] >> shift) & 3) as i32) - high) as f32);
            }
            let dl = d * (scales[i] - 32) as f32;
            i += 1;
            for l in 0..16 {
                let high = if hm[l + 16] & m == 0 { 4 } else { 0 };
                y.push(dl * ((((qs[q + l + 16] >> shift) & 3) as i32) - high) as f32);
            }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
    y
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u32, u32) {
    if j < 4 {
        (u32::from(q[j] & 63), u32::from(q[j + 4] & 63))
    } else {
        (
            u32::from(q[j + 4] & 0x0f) | (u32::from(q[j - 4] >> 6) << 4),
            u32::from(q[j + 4] >> 4) | (u32::from(q[j] >> 6) << 4),
        )
    }
}

fn q4_k(b: &[u8]) -> Vec<f32> {
    let d = f16_at(b, 0);
    let dmin = f16_at(b, 2);
    let sc = &b[4..16];
    let qs = &b[16..144];
    let mut y = Vec::with_capacity(256);
    let mut q = 0;
    for j in 0..4 {
        let (s1, m1) = scale_min_k4(2 * j, sc);
        let (s2, m2) = scale_min_k4(2 * j + 1, sc);
        let d1 = d * s1 as f32;
        let mm1 = dmin * m1 as f32;
        let d2 = d * s2 as f32;
        let mm2 = dmin * m2 as f32;
        for l in 0..32 {
            y.push(d1 * f32::from(qs[q + l] & 0x0f) - mm1);
        }
        for l in 0..32 {
            y.push(d2 * f32::from(qs[q + l] >> 4) - mm2);
        }
        q += 32;
    }
    y
}

fn q5_k(b: &[u8]) -> Vec<f32> {
    let d = f16_at(b, 0);
    let dmin = f16_at(b, 2);
    let sc = &b[4..16];
    let qh = &b[16..48];
    let ql = &b[48..176];
    let mut y = Vec::with_capacity(256);
    let mut q = 0;
    let mut u1 = 1u8;
    let mut u2 = 2u8;
    for j in 0..4 {
        let (s1, m1) = scale_min_k4(2 * j, sc);
        let (s2, m2) = scale_min_k4(2 * j + 1, sc);
        let d1 = d * s1 as f32;
        let mm1 = dmin * m1 as f32;
        let d2 = d * s2 as f32;
        let mm2 = dmin * m2 as f32;
        for l in 0..32 {
            let extra = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
            y.push(d1 * (f32::from(ql[q + l] & 0x0f) + extra) - mm1);
        }
        for l in 0..32 {
            let extra = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
            y.push(d2 * (f32::from(ql[q + l] >> 4) + extra) - mm2);
        }
        q += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
    y
}

fn q6_k(b: &[u8]) -> Vec<f32> {
    let ql = &b[0..128];
    let qh = &b[128..192];
    let d = f16_at(b, 208);
    let mut y = vec![0.0; 256];
    for n in 0..2 {
        let base_y = 128 * n;
        let base_l = 64 * n;
        let base_h = 32 * n;
        let base_s = 8 * n;
        for l in 0..32 {
            let i = l / 16;
            let q1 = i32::from((ql[base_l + l] & 0x0f) | (((qh[base_h + l] >> 0) & 3) << 4)) - 32;
            let q2 =
                i32::from((ql[base_l + l + 32] & 0x0f) | (((qh[base_h + l] >> 2) & 3) << 4)) - 32;
            let q3 = i32::from((ql[base_l + l] >> 4) | (((qh[base_h + l] >> 4) & 3) << 4)) - 32;
            let q4 =
                i32::from((ql[base_l + l + 32] >> 4) | (((qh[base_h + l] >> 6) & 3) << 4)) - 32;
            let s = |offset: usize| i8_at(b, 192 + base_s + i + offset);
            y[base_y + l] = d * s(0) * q1 as f32;
            y[base_y + l + 32] = d * s(2) * q2 as f32;
            y[base_y + l + 64] = d * s(4) * q3 as f32;
            y[base_y + l + 96] = d * s(6) * q4 as f32;
        }
    }
    y
}

fn q8_k(b: &[u8]) -> Vec<f32> {
    let d = f32_at(b, 0);
    (0..256).map(|j| d * i8_at(b, 4 + j)).collect()
}

fn sign_bit(signs: u32, mask: u32) -> f32 {
    if signs & mask != 0 {
        -1.0
    } else {
        1.0
    }
}

fn iq2_xxs(b: &[u8], tables: &Grids) -> Result<Vec<f32>> {
    let d = f16_at(b, 0);
    let mut y = Vec::with_capacity(256);
    for ib in 0..8 {
        let a0 = u32_at(b, 2 + 8 * ib);
        let a1 = u32_at(b, 6 + 8 * ib);
        let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let g = at(
                &tables.iq2xxs_grid,
                ((a0 >> (8 * l)) & 0xff) as usize,
                "iq2xxs_grid",
            )?;
            let s = tables.ksigns_iq2xs[((a1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                y.push(db * g[j] * sign_bit(s, tables.kmask_iq2xs[j]));
            }
        }
    }
    Ok(y)
}

fn iq2_xs(b: &[u8], tables: &Grids) -> Result<Vec<f32>> {
    let d = f16_at(b, 0);
    let sc = &b[66..74];
    let mut y = Vec::with_capacity(256);
    for ib in 0..8 {
        let db = [
            d * (0.5 + f32::from(sc[ib] & 0x0f)) * 0.25,
            d * (0.5 + f32::from(sc[ib] >> 4)) * 0.25,
        ];
        for l in 0..4 {
            let v = u16_at(b, 2 + 2 * (4 * ib + l));
            let g = at(&tables.iq2xs_grid, (v & 511) as usize, "iq2xs_grid")?;
            let s = tables.ksigns_iq2xs[(v >> 9) as usize];
            for j in 0..8 {
                y.push(db[l / 2] * g[j] * sign_bit(s, tables.kmask_iq2xs[j]));
            }
        }
    }
    Ok(y)
}

fn iq2_s(b: &[u8], tables: &Grids) -> Result<Vec<f32>> {
    let d = f16_at(b, 0);
    let qs = &b[2..34];
    let signs = &b[34..66];
    let qh = &b[66..74];
    let sc = &b[74..82];
    let mut y = Vec::with_capacity(256);
    for ib in 0..8 {
        let db = [
            d * (0.5 + f32::from(sc[ib] & 0x0f)) * 0.25,
            d * (0.5 + f32::from(sc[ib] >> 4)) * 0.25,
        ];
        for l in 0..4 {
            let index = u32::from(qs[4 * ib + l]) | ((u32::from(qh[ib]) << (8 - 2 * l)) & 0x300);
            let g = at(&tables.iq2s_grid, index as usize, "iq2s_grid")?;
            let s = u32::from(signs[4 * ib + l]);
            for j in 0..8 {
                y.push(db[l / 2] * g[j] * sign_bit(s, tables.kmask_iq2xs[j]));
            }
        }
    }
    Ok(y)
}

fn iq3_xxs(b: &[u8], tables: &Grids) -> Result<Vec<f32>> {
    let d = f16_at(b, 0);
    let qs = &b[2..66];
    let mut y = Vec::with_capacity(256);
    for ib in 0..8 {
        let a = u32_at(b, 66 + 4 * ib);
        let db = d * (0.5 + (a >> 28) as f32) * 0.5;
        for l in 0..4 {
            let s = tables.ksigns_iq2xs[((a >> (7 * l)) & 127) as usize];
            let g1 = at(
                &tables.iq3xxs_grid,
                qs[8 * ib + 2 * l] as usize,
                "iq3xxs_grid",
            )?;
            let g2 = at(
                &tables.iq3xxs_grid,
                qs[8 * ib + 2 * l + 1] as usize,
                "iq3xxs_grid",
            )?;
            for j in 0..4 {
                y.push(db * g1[j] * sign_bit(s, tables.kmask_iq2xs[j]));
            }
            for j in 0..4 {
                y.push(db * g2[j] * sign_bit(s, tables.kmask_iq2xs[j + 4]));
            }
        }
    }
    Ok(y)
}

fn iq3_s(b: &[u8], tables: &Grids) -> Result<Vec<f32>> {
    let d = f16_at(b, 0);
    let qs = &b[2..66];
    let qh = &b[66..74];
    let signs = &b[74..106];
    let sc = &b[106..110];
    let mut y = Vec::with_capacity(256);
    let mut qi = 0;
    let mut si = 0;
    let mut hi = 0;
    for ib in (0..8).step_by(2) {
        for half in 0..2 {
            let db = d * (1.0 + 2.0 * f32::from((sc[ib / 2] >> (4 * half)) & 0x0f));
            let h = u32::from(qh[hi + half]);
            for l in 0..4 {
                let g1 = at(
                    &tables.iq3s_grid,
                    (u32::from(qs[qi + 2 * l]) | ((h << (8 - 2 * l)) & 256)) as usize,
                    "iq3s_grid",
                )?;
                let g2 = at(
                    &tables.iq3s_grid,
                    (u32::from(qs[qi + 2 * l + 1]) | ((h << (7 - 2 * l)) & 256)) as usize,
                    "iq3s_grid",
                )?;
                let s = u32::from(signs[si + l]);
                for j in 0..4 {
                    y.push(db * g1[j] * sign_bit(s, tables.kmask_iq2xs[j]));
                }
                for j in 0..4 {
                    y.push(db * g2[j] * sign_bit(s, tables.kmask_iq2xs[j + 4]));
                }
            }
            qi += 8;
            si += 4;
        }
        hi += 2;
    }
    Ok(y)
}

fn iq1_s(b: &[u8], tables: &Grids) -> Result<Vec<f32>> {
    let d = f16_at(b, 0);
    let qs = &b[2..34];
    let mut y = Vec::with_capacity(256);
    for ib in 0..8 {
        let h = u16_at(b, 34 + 2 * ib);
        let dl = d * (2.0 * ((h >> 12) & 7) as f32 + 1.0);
        let delta = if h & 0x8000 != 0 { -0.125 } else { 0.125 };
        for l in 0..4 {
            let index = u32::from(qs[4 * ib + l]) | (((h >> (3 * l)) & 7) << 8);
            let g = at(&tables.iq1s_grid, index as usize, "iq1s_grid")?;
            for j in 0..8 {
                y.push(dl * (g[j] + delta));
            }
        }
    }
    Ok(y)
}

fn iq1_m(b: &[u8], tables: &Grids) -> Result<Vec<f32>> {
    let qs = &b[0..32];
    let qh = &b[32..48];
    let sc = [u16_at(b, 48), u16_at(b, 50), u16_at(b, 52), u16_at(b, 54)];
    let bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    let d = f16_to_f32(bits as u16);
    let mut y = Vec::with_capacity(256);
    for ib in 0..8 {
        let dl1 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2))) & 7) as f32 + 1.0);
        let dl2 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) as f32 + 1.0);
        let q = &qs[4 * ib..4 * ib + 4];
        let h = &qh[2 * ib..2 * ib + 2];
        let idx = [
            u32::from(q[0]) | ((u32::from(h[0]) << 8) & 0x700),
            u32::from(q[1]) | ((u32::from(h[0]) << 4) & 0x700),
            u32::from(q[2]) | ((u32::from(h[1]) << 8) & 0x700),
            u32::from(q[3]) | ((u32::from(h[1]) << 4) & 0x700),
        ];
        let dlt = [
            if h[0] & 0x08 != 0 { -0.125 } else { 0.125 },
            if h[0] & 0x80 != 0 { -0.125 } else { 0.125 },
            if h[1] & 0x08 != 0 { -0.125 } else { 0.125 },
            if h[1] & 0x80 != 0 { -0.125 } else { 0.125 },
        ];
        for l in 0..4 {
            let g = at(&tables.iq1s_grid, idx[l] as usize, "iq1s_grid")?;
            let dl = if l < 2 { dl1 } else { dl2 };
            for j in 0..8 {
                y.push(dl * (g[j] + dlt[l]));
            }
        }
    }
    Ok(y)
}

fn iq4_nl(b: &[u8], tables: &Grids) -> Vec<f32> {
    let d = f16_at(b, 0);
    let mut y = vec![0.0; 32];
    for j in 0..16 {
        y[j] = d * tables.kvalues_iq4nl[(b[2 + j] & 0x0f) as usize];
        y[j + 16] = d * tables.kvalues_iq4nl[(b[2 + j] >> 4) as usize];
    }
    y
}

fn iq4_xs(b: &[u8], tables: &Grids) -> Vec<f32> {
    let d = f16_at(b, 0);
    let sh = u16_at(b, 2);
    let sl = &b[4..8];
    let qs = &b[8..136];
    let mut y = vec![0.0; 256];
    for ib in 0..8 {
        let ls = ((u32::from(sl[ib / 2]) >> (4 * (ib % 2))) & 0x0f) | (((sh >> (2 * ib)) & 3) << 4);
        let dl = d * (ls as f32 - 32.0);
        for j in 0..16 {
            y[32 * ib + j] = dl * tables.kvalues_iq4nl[(qs[16 * ib + j] & 0x0f) as usize];
            y[32 * ib + j + 16] = dl * tables.kvalues_iq4nl[(qs[16 * ib + j] >> 4) as usize];
        }
    }
    y
}

fn trit(byte: u8, n: usize) -> f32 {
    let p3 = [1u32, 3, 9, 27, 81, 243];
    let q = (u32::from(byte) * p3[n]) & 0xff;
    ((q * 3) >> 8) as f32 - 1.0
}

fn tq1_0(b: &[u8]) -> Vec<f32> {
    let qs = &b[0..48];
    let qh = &b[48..52];
    let d = f16_at(b, 52);
    let mut y = Vec::with_capacity(256);
    for n in 0..5 {
        for m in 0..32 {
            y.push(trit(qs[m], n) * d);
        }
    }
    for n in 0..5 {
        for m in 0..16 {
            y.push(trit(qs[32 + m], n) * d);
        }
    }
    for n in 0..4 {
        for j in 0..4 {
            y.push(trit(qh[j], n) * d);
        }
    }
    y
}

fn tq2_0(b: &[u8]) -> Vec<f32> {
    let qs = &b[0..64];
    let d = f16_at(b, 64);
    let mut y = Vec::with_capacity(256);
    for j in [0, 32] {
        for l in 0..4 {
            for m in 0..32 {
                y.push((f32::from((qs[j + m] >> (2 * l)) & 3) - 1.0) * d);
            }
        }
    }
    y
}

fn mxfp4(b: &[u8], tables: &Grids) -> Vec<f32> {
    let d = e8m0_half(b[0]);
    let mut y = vec![0.0; 32];
    for j in 0..16 {
        y[j] = tables.kvalues_fp4[(b[1 + j] & 0x0f) as usize] * d;
        y[j + 16] = tables.kvalues_fp4[(b[1 + j] >> 4) as usize] * d;
    }
    y
}

fn nvfp4(b: &[u8], tables: &Grids) -> Vec<f32> {
    let mut y = vec![0.0; 64];
    for s in 0..4 {
        let dd = ue4m3_half(b[s]);
        for j in 0..8 {
            let q = b[4 + 8 * s + j];
            y[16 * s + j] = tables.kvalues_fp4[(q & 0x0f) as usize] * dd;
            y[16 * s + j + 8] = tables.kvalues_fp4[(q >> 4) as usize] * dd;
        }
    }
    y
}

fn q1_0(b: &[u8]) -> Vec<f32> {
    let d = f16_at(b, 0);
    (0..128)
        .map(|j| {
            if (b[2 + j / 8] >> (j % 8)) & 1 != 0 {
                d
            } else {
                -d
            }
        })
        .collect()
}

fn q2_0(b: &[u8]) -> Vec<f32> {
    let d = f16_at(b, 0);
    (0..64)
        .map(|j| (f32::from((b[2 + j / 4] >> ((j % 4) * 2)) & 3) - 1.0) * d)
        .collect()
}

/// Reference `quantize` for Q4_0. `values` length is a multiple of 32.
pub fn quantize_q4_0(values: &[f32]) -> Result<Vec<u8>> {
    quantize_rows(values, 32, quant_q4_0_row)
}

/// Reference `quantize` for Q8_0.
pub fn quantize_q8_0(values: &[f32]) -> Result<Vec<u8>> {
    quantize_rows(values, 32, quant_q8_0_row)
}

fn quantize_rows(values: &[f32], width: usize, row: fn(&[f32]) -> Vec<u8>) -> Result<Vec<u8>> {
    if values.len() % width != 0 {
        return Err(msg(format!(
            "quantize needs a multiple of {width} values, got {}",
            values.len()
        )));
    }
    let mut out = Vec::new();
    for chunk in values.chunks(width) {
        out.extend(row(chunk));
    }
    Ok(out)
}

fn quant_q4_0_row(x: &[f32]) -> Vec<u8> {
    let mut amax = 0.0f32;
    let mut mx = 0.0f32;
    for value in x {
        if value.abs() > amax {
            amax = value.abs();
            mx = *value;
        }
    }
    let d = mx / -8.0;
    let idd = if d == 0.0 { 0.0 } else { 1.0 / d };
    let mut qs = [0u8; 16];
    for j in 0..16 {
        let a = nibble(x[j] * idd + 8.5);
        let c = nibble(x[j + 16] * idd + 8.5);
        qs[j] = a | (c << 4);
    }
    let mut out = Vec::with_capacity(18);
    out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    out.extend_from_slice(&qs);
    out
}

fn quant_q8_0_row(x: &[f32]) -> Vec<u8> {
    let amax = x.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let d = amax / 127.0;
    let idd = if d == 0.0 { 0.0 } else { 1.0 / d };
    let mut out = Vec::with_capacity(34);
    out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    for value in x {
        // CONV-002: roundf of the f32 product, ties away from zero.
        let code = (value * idd).round() as i32;
        out.push(code as u8);
    }
    out
}

fn nibble(value: f32) -> u8 {
    let truncated = value.trunc() as i32;
    truncated.clamp(0, 15) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_dequant_matches_a_hand_built_block() {
        let mut block = vec![0u8; 34];
        block[0] = 0x00;
        block[1] = 0x3c;
        block[2] = 1;
        block[3] = (-2i8) as u8;
        let back = dequant_tensor(8, &block).unwrap();
        assert_eq!(back.len(), 32);
        assert_eq!(back[0], 1.0);
        assert_eq!(back[1], -2.0);
        assert_eq!(back[2], 0.0);
    }

    #[test]
    fn q8_1_is_refused() {
        let err = dequant_tensor(9, &[0u8; 36]).unwrap_err();
        assert!(err.to_string().contains("Q8_1"), "{err}");
    }
}
