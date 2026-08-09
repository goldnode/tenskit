//! Dequantize packs whose formulas are written down, and the Q4_0 / Q8_0
//! reference quantizers. Marlin's tile inverse, EXL2, ModelOpt nibbles, GPTQ
//! 3-bit, and Q8_1 stay refused.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::archive::{self, Archive, Inner, Member};
use crate::cli::{FormatName, DEFAULT_MAX_HEADER};
use crate::dtype::{self, numel};
use crate::error::{msg, Result};
use crate::formats::gguf::{write_dense, DenseTensor, MetaValue};
use crate::ggml_dequant::{self, dequant_tensor};
use crate::payload::Payload;

pub struct Request {
    pub source: PathBuf,
    pub output: PathBuf,
    pub format: Option<FormatName>,
    pub quantize: Option<String>,
    pub force_single: bool,
    pub force: bool,
}

pub fn run(request: &Request) -> Result<Vec<String>> {
    crate::io_util::refuse_existing(&request.output, request.force)?;
    let resolved = archive::resolve_user_path(&request.source, request.force_single)?;
    let archive = archive::open(&resolved.path, None, DEFAULT_MAX_HEADER, None, None)?;
    let config = read_config(resolved.path.parent().unwrap_or(Path::new(".")))?;
    let mut warnings = Vec::new();
    let tensors = dequant_archive(&archive, config.as_ref(), &mut warnings)?;
    let target = output_format(&request.output, request.format)?;
    write_f32(&request.output, target, &tensors)?;
    Ok(warnings)
}

pub(crate) struct FloatTensor {
    pub(crate) name: String,
    pub(crate) shape: Vec<u64>,
    pub(crate) values: Vec<f32>,
}

pub(crate) fn dequant_archive(
    archive: &Archive,
    config: Option<&Value>,
    warnings: &mut Vec<String>,
) -> Result<Vec<FloatTensor>> {
    refuse_unsupported(archive, config)?;
    if archive.members.iter().any(|member| is_block(&member.dtype)) {
        return dequant_ggml(archive);
    }
    let names: Vec<&str> = archive
        .members
        .iter()
        .map(|member| member.name.as_str())
        .collect();
    let quant = quant_config(config);
    let method = quant.as_ref().and_then(|map| map.get("quant_method")).and_then(Value::as_str);
    // GPTQ-007: a gptq config whose version is gemm is AWQ GEMM.
    let gemm_version = quant
        .as_ref()
        .and_then(|map| map.get("version"))
        .and_then(Value::as_str)
        .is_some_and(|version| version.eq_ignore_ascii_case("gemm"));
    match method {
        Some("gptq") if !gemm_version => return dequant_gptq(archive, quant.as_ref()),
        Some("gptq" | "awq") => return dequant_awq(archive, quant.as_ref()),
        Some("bitsandbytes") => return dequant_bnb(archive, quant.as_ref()),
        Some("compressed-tensors") => return dequant_compressed(archive, quant.as_ref(), warnings),
        Some(other) if mlx_config(config).is_none() => {
            return Err(msg(format!("quant_method {other} has no dequant here")))
        }
        _ => {}
    }
    if mlx_config(config).is_some() {
        return dequant_mlx(archive, config);
    }
    // bitsandbytes stores its own state beside each weight (BNB-005).
    if names.iter().any(|name| name.contains(".quant_state.bitsandbytes__")) {
        return dequant_bnb(archive, None);
    }
    // conversion-considerations.md section 3: names alone do not identify a
    // pack, and bits, group size, and zero convention are not in the tensors.
    let guess = if names.iter().any(|name| name.ends_with(".g_idx")) {
        Some("GPTQ")
    } else if names.iter().any(|name| name.ends_with(".qweight")) {
        Some("AWQ or GPTQ")
    } else if names.iter().any(|name| name.ends_with(".absmax")) {
        Some("bitsandbytes")
    } else if names.iter().any(|name| name.ends_with(".weight_packed")) {
        Some("compressed-tensors")
    } else {
        None
    };
    if let Some(guess) = guess {
        return Err(msg(format!(
            "the tensor names look like a {guess} pack, but there is no config.json quantization_config; dequant needs it"
        )));
    }
    float_copy(archive)
}

fn refuse_unsupported(archive: &Archive, config: Option<&Value>) -> Result<()> {
    for member in &archive.members {
        if member.dtype == "Q8_1" || dtype::canonical_id(&member.dtype).ok() == Some("ggml_q8_1") {
            return Err(msg("GGUF Q8_1 dequant is inferred and is refused"));
        }
        if member.name.ends_with(".q_weight") {
            return Err(msg(
                "EXL2-Q1: the q_weight bit shuffle is not specified, so this pack is refused",
            ));
        }
        let modelopt = member.name.ends_with(".weight_scale_2")
            || member.name.ends_with(".weight_global_scale")
            || (member.name.ends_with(".weight_packed")
                && matches!(member.dtype.as_str(), "U8" | "uint8" | "I8"));
        if modelopt {
            return Err(msg(
                "CT-Q1: the ModelOpt NVFP4 nibble order is not specified, so this pack is refused",
            ));
        }
    }
    if let Some(config) = config {
        let quant = config
            .get("quantization_config")
            .or_else(|| config.get("quantization"));
        if let Some(object) = quant.and_then(Value::as_object) {
            let marlin = object.get("checkpoint_format").and_then(Value::as_str) == Some("marlin")
                || object.get("is_marlin_format").and_then(Value::as_bool) == Some(true);
            if marlin {
                return Err(msg(
                    "GPTQ marlin pack is refused: the 16-wide tile inverse is not transcribed",
                ));
            }
            let bits = object
                .get("bits")
                .or_else(|| object.get("w_bit"))
                .and_then(Value::as_u64);
            if bits == Some(3) && object.get("quant_method").and_then(Value::as_str) != Some("compressed-tensors") {
                return Err(msg("GPTQ 3-bit unpack is named but not transcribed"));
            }
        }
    }
    Ok(())
}

fn is_block(dtype: &str) -> bool {
    dtype::canonical_id(dtype)
        .ok()
        .is_some_and(|id| dtype::canon_kind(id) == Some("block"))
}

fn dequant_ggml(archive: &Archive) -> Result<Vec<FloatTensor>> {
    let Inner::Gguf(file) = &archive.inner else {
        return Err(msg("quantized ggml types were found outside a GGUF file"));
    };
    let mut out = Vec::new();
    for member in &archive.members {
        let tensor = file
            .tensors
            .iter()
            .find(|item| item.name == member.name)
            .ok_or_else(|| msg(format!("missing GGUF tensor {}", member.name)))?;
        let raw = member.payload.read_vec()?;
        let values = dequant_tensor(tensor.ggml_type, &raw)?;
        let mut shape = member.shape.clone();
        shape.reverse();
        let expect = numel(&shape).unwrap_or(0) as usize;
        if values.len() != expect {
            return Err(msg(format!(
                "tensor {} dequantized to {} values and the shape needs {expect}",
                member.name,
                values.len()
            )));
        }
        out.push(FloatTensor {
            name: member.name.clone(),
            shape,
            values,
        });
    }
    Ok(out)
}

fn dequant_gptq(archive: &Archive, quant: Option<&Map<String, Value>>) -> Result<Vec<FloatTensor>> {
    let bits = config_bits(quant, "GPTQ")?;
    if bits == 3 {
        return Err(msg("GPTQ 3-bit unpack is named but not transcribed"));
    }
    if !matches!(bits, 2 | 4 | 8) {
        return Err(msg(format!(
            "GPTQ bits {bits} are not the 2, 4, or 8 cuda pack (GPTQ-001)"
        )));
    }
    // GPTQ-006 / GPTQ-008: gptq (or no checkpoint_format) stores zero - 1;
    // gptq_v2 stores the zero itself.
    let zero_offset = match quant
        .and_then(|map| map.get("checkpoint_format"))
        .and_then(Value::as_str)
    {
        None | Some("gptq") => 1,
        Some("gptq_v2") => 0,
        Some(other) => {
            return Err(msg(format!("GPTQ checkpoint_format {other} is not gptq or gptq_v2")))
        }
    };
    let packed = pack_prefixes(archive, "qweight");
    let mut groups: BTreeMap<String, Gptq> = BTreeMap::new();
    let mut plain = Vec::new();
    for member in &archive.members {
        let split = member
            .name
            .rsplit_once('.')
            .filter(|(prefix, suffix)| {
                packed.contains(*prefix)
                    && matches!(*suffix, "qweight" | "qzeros" | "scales" | "g_idx" | "bias")
            });
        let Some((prefix, suffix)) = split else {
            plain.push(member_f32(member)?);
            continue;
        };
        let slot = groups.entry(prefix.to_string()).or_default();
        match suffix {
            "qweight" => slot.qweight = Some(read_i32(member)?),
            "qzeros" => slot.qzeros = Some(read_i32(member)?),
            "scales" => slot.scales = Some(read_f32(member)?),
            "g_idx" => slot.g_idx = Some(read_i32(member)?.0),
            "bias" => slot.bias = Some(member_f32(member)?),
            _ => {}
        }
    }
    let mut out = plain;
    for (prefix, group) in groups {
        let missing = |name: &str| msg(format!("{prefix}.{name} is missing"));
        let qweight = group.qweight.ok_or_else(|| missing("qweight"))?;
        let qzeros = group.qzeros.ok_or_else(|| missing("qzeros"))?;
        let scales = group.scales.ok_or_else(|| missing("scales"))?;
        let g_idx = group.g_idx.ok_or_else(|| missing("g_idx"))?;
        let values = unpack_gptq(
            &qweight.0, &qweight.1, &qzeros.0, &qzeros.1, &scales.0, &scales.1, &g_idx, bits,
            zero_offset,
        )
        .map_err(|err| msg(format!("{prefix}: {err}")))?;
        let out_features = qweight.1[1];
        let in_features = g_idx.len() as u64;
        out.push(FloatTensor {
            name: format!("{prefix}.weight"),
            shape: vec![out_features, in_features],
            values,
        });
        if let Some(mut bias) = group.bias {
            bias.name = format!("{prefix}.bias");
            out.push(bias);
        }
    }
    Ok(out)
}

/// Module prefixes that have a `<prefix>.<suffix>` tensor. A bias or scale
/// belongs to a pack only when its module has the packed weight.
fn pack_prefixes<'a>(archive: &'a Archive, suffix: &str) -> std::collections::BTreeSet<&'a str> {
    archive
        .members
        .iter()
        .filter_map(|member| member.name.strip_suffix(suffix)?.strip_suffix('.'))
        .collect()
}

#[derive(Default)]
struct Gptq {
    qweight: Option<(Vec<i32>, Vec<u64>)>,
    qzeros: Option<(Vec<i32>, Vec<u64>)>,
    scales: Option<(Vec<f32>, Vec<u64>)>,
    g_idx: Option<Vec<i32>>,
    bias: Option<FloatTensor>,
}

/// GPTQ cuda pack, returned as `[out_features, in_features]`.
#[allow(clippy::too_many_arguments)]
pub fn unpack_gptq(
    qweight: &[i32],
    qweight_shape: &[u64],
    qzeros: &[i32],
    qzeros_shape: &[u64],
    scales: &[f32],
    scales_shape: &[u64],
    g_idx: &[i32],
    bits: u32,
    zero_offset: u32,
) -> Result<Vec<f32>> {
    if !matches!(bits, 2 | 4 | 8) {
        return Err(msg(format!("GPTQ bits {bits} are not 2, 4, or 8")));
    }
    if qweight_shape.len() != 2 || qzeros_shape.len() != 2 || scales_shape.len() != 2 {
        return Err(msg("GPTQ qweight, qzeros, and scales are matrices"));
    }
    let rows = qweight_shape[0] as usize;
    let out_f = qweight_shape[1] as usize;
    let pack = (32 / bits) as usize;
    let in_f = rows * pack;
    let zero_rows = qzeros_shape[0] as usize;
    let zero_cols = qzeros_shape[1] as usize;
    // GPTQ-002, GPTQ-003, GPTQ-004.
    if in_f % 32 != 0 {
        return Err(msg("GPTQ in_features is not a multiple of 32 (GPTQ-002)"));
    }
    if zero_cols * pack != out_f {
        return Err(msg("GPTQ qzeros width is not out_features // 32 * bits (GPTQ-003)"));
    }
    if g_idx.len() != in_f {
        return Err(msg("GPTQ g_idx length does not match in_features (GPTQ-004)"));
    }
    if scales_shape[0] as usize != zero_rows || scales_shape[1] as usize != out_f {
        return Err(msg("GPTQ scales are not [groups, out_features]"));
    }
    if qweight.len() != rows * out_f
        || qzeros.len() != zero_rows * zero_cols
        || scales.len() != zero_rows * out_f
    {
        return Err(msg("GPTQ tensor lengths do not match their shapes"));
    }
    if g_idx.iter().any(|g| *g < 0 || *g as usize >= zero_rows) {
        return Err(msg("GPTQ g_idx selects a group that has no scale row (GPTQ-004)"));
    }
    let mask = (1u32 << bits) - 1;
    let mut codes = vec![0u32; in_f * out_f];
    for row in 0..rows {
        for col in 0..out_f {
            let word = qweight[row * out_f + col] as u32;
            for j in 0..pack {
                codes[(row * pack + j) * out_f + col] = (word >> (bits * j as u32)) & mask;
            }
        }
    }
    let mut zeros = vec![0u32; zero_rows * out_f];
    for row in 0..zero_rows {
        for col in 0..zero_cols {
            let word = qzeros[row * zero_cols + col] as u32;
            for j in 0..pack {
                zeros[row * out_f + col * pack + j] =
                    ((word >> (bits * j as u32)) & mask) + zero_offset;
            }
        }
    }
    let mut out = vec![0f32; out_f * in_f];
    for k in 0..in_f {
        let g = g_idx[k] as usize;
        for n in 0..out_f {
            let code = codes[k * out_f + n] as f32;
            let zero = zeros[g * out_f + n] as f32;
            out[n * in_f + k] = (code - zero) * scales[g * out_f + n];
        }
    }
    Ok(out)
}

fn dequant_awq(archive: &Archive, quant: Option<&Map<String, Value>>) -> Result<Vec<FloatTensor>> {
    // AWQ-006: version is case-insensitive and defaults to gemm.
    if let Some(version) = quant.and_then(|map| map.get("version")).and_then(Value::as_str) {
        if !version.eq_ignore_ascii_case("gemm") {
            return Err(msg(format!("AWQ version {version} is not the gemm pack")));
        }
    }
    let bits = config_bits(quant, "AWQ")?;
    if bits != 4 {
        return Err(msg(format!("AWQ w_bit {bits} is not 4 (AWQ-001)")));
    }
    let group = quant
        .and_then(|map| map.get("group_size").or_else(|| map.get("q_group_size")))
        .and_then(Value::as_u64)
        .filter(|group| *group > 0)
        .ok_or_else(|| msg("AWQ dequant needs a positive quantization_config.group_size"))?
        as usize;
    let packed = pack_prefixes(archive, "qweight");
    let mut groups: BTreeMap<String, Awq> = BTreeMap::new();
    let mut plain = Vec::new();
    for member in &archive.members {
        let split = member.name.rsplit_once('.').filter(|(prefix, suffix)| {
            packed.contains(*prefix) && matches!(*suffix, "qweight" | "qzeros" | "scales" | "bias")
        });
        let Some((prefix, suffix)) = split else {
            plain.push(member_f32(member)?);
            continue;
        };
        let slot = groups.entry(prefix.to_string()).or_default();
        match suffix {
            "qweight" => slot.qweight = Some(read_i32(member)?),
            "qzeros" => slot.qzeros = Some(read_i32(member)?),
            "scales" => slot.scales = Some(read_f32(member)?),
            "bias" => slot.bias = Some(member_f32(member)?),
            _ => {}
        }
    }
    let mut out = plain;
    for (prefix, tensors) in groups {
        let missing = |name: &str| msg(format!("{prefix}.{name} is missing"));
        let qweight = tensors.qweight.ok_or_else(|| missing("qweight"))?;
        let qzeros = tensors.qzeros.ok_or_else(|| missing("qzeros"))?;
        let scales = tensors.scales.ok_or_else(|| missing("scales"))?;
        let values = unpack_awq(
            &qweight.0, &qweight.1, &qzeros.0, &qzeros.1, &scales.0, &scales.1, group,
        )
        .map_err(|err| msg(format!("{prefix}: {err}")))?;
        let in_f = qweight.1[0];
        let out_f = qweight.1[1] * 8;
        out.push(FloatTensor {
            name: format!("{prefix}.weight"),
            shape: vec![out_f, in_f],
            values,
        });
        if let Some(mut bias) = tensors.bias {
            bias.name = format!("{prefix}.bias");
            out.push(bias);
        }
    }
    Ok(out)
}

#[derive(Default)]
struct Awq {
    qweight: Option<(Vec<i32>, Vec<u64>)>,
    qzeros: Option<(Vec<i32>, Vec<u64>)>,
    scales: Option<(Vec<f32>, Vec<u64>)>,
    bias: Option<FloatTensor>,
}

/// AWQ GEMM pack, returned as `[out_features, in_features]`, the layout of
/// the float `nn.Linear` weight it replaces.
pub fn unpack_awq(
    qweight: &[i32],
    qweight_shape: &[u64],
    qzeros: &[i32],
    qzeros_shape: &[u64],
    scales: &[f32],
    scales_shape: &[u64],
    group_size: usize,
) -> Result<Vec<f32>> {
    if qweight_shape.len() != 2 || qzeros_shape.len() != 2 || scales_shape.len() != 2 {
        return Err(msg("AWQ qweight, qzeros, and scales are matrices"));
    }
    let order = [0usize, 2, 4, 6, 1, 3, 5, 7];
    let in_f = qweight_shape[0] as usize;
    let packed_out = qweight_shape[1] as usize;
    let out_f = packed_out * 8;
    // AWQ-002 and AWQ-003; qzeros and scales have one row per group.
    if group_size == 0 || in_f % group_size != 0 {
        return Err(msg("AWQ in_features is not a multiple of group_size (AWQ-002)"));
    }
    let groups = in_f / group_size;
    if qzeros_shape != [groups as u64, packed_out as u64]
        || scales_shape != [groups as u64, out_f as u64]
    {
        return Err(msg(
            "AWQ qzeros or scales do not have one row per group (AWQ-003)",
        ));
    }
    if qweight.len() != in_f * packed_out
        || qzeros.len() != groups * packed_out
        || scales.len() != groups * out_f
    {
        return Err(msg("AWQ tensor lengths do not match their shapes"));
    }
    let unpack = |words: &[i32], rows: usize| {
        let mut out = vec![0i32; rows * out_f];
        for row in 0..rows {
            for col in 0..packed_out {
                let word = words[row * packed_out + col] as u32;
                for (i, slot) in order.iter().enumerate() {
                    out[row * out_f + col * 8 + slot] = ((word >> (i * 4)) & 0x0f) as i32;
                }
            }
        }
        out
    };
    let w = unpack(qweight, in_f);
    let z = unpack(qzeros, groups);
    let mut out = vec![0f32; out_f * in_f];
    for k in 0..in_f {
        let g = k / group_size;
        for n in 0..out_f {
            out[n * in_f + k] =
                (w[k * out_f + n] as f32 - z[g * out_f + n] as f32) * scales[g * out_f + n];
        }
    }
    Ok(out)
}

/// bitsandbytes 4-bit. The Hugging Face layout (BNB-005) keeps the state
/// beside each weight: `<m>.weight`, `<m>.weight.absmax`, and
/// `<m>.weight.quant_state.bitsandbytes__nf4` (a uint8 tensor of UTF-8 JSON),
/// plus `nested_absmax` and `nested_quant_map` when the absmax is itself
/// quantized. The logical shape comes from that JSON (BNB-006). Without a
/// quant_state, `<m>.absmax` beside `<m>.weight` is read with the config's
/// quant type and blocksize, and the shape is the unpacked length.
fn dequant_bnb(archive: &Archive, quant: Option<&Map<String, Value>>) -> Result<Vec<FloatTensor>> {
    let members = index(archive);
    let mut used = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for member in &archive.members {
        let Some(module) = member.name.strip_suffix(".weight") else {
            continue;
        };
        let weight = member.name.as_str();
        let state_key = ["nf4", "fp4"]
            .iter()
            .map(|kind| format!("{weight}.quant_state.bitsandbytes__{kind}"))
            .find(|key| members.contains_key(key.as_str()));
        let (state, absmax_name) = if let Some(key) = &state_key {
            let bytes = members[key.as_str()].payload.read_vec()?;
            let state: Value = serde_json::from_slice(&bytes)
                .map_err(|_| msg(format!("{key} is not UTF-8 JSON (BNB-005)")))?;
            used.insert(key.clone());
            (BnbState::from_json(&state, key)?, format!("{weight}.absmax"))
        } else {
            let sibling = format!("{module}.absmax");
            if !members.contains_key(sibling.as_str()) {
                continue;
            }
            let quant = quant.ok_or_else(|| {
                msg(format!(
                    "{weight} has no quant_state; dequant needs config.json quantization_config"
                ))
            })?;
            (BnbState::from_config(quant)?, sibling)
        };
        let absmax_member = members
            .get(absmax_name.as_str())
            .ok_or_else(|| msg(format!("{absmax_name} is missing")))?;
        used.insert(absmax_name.clone());
        // BNB-004: a nested absmax is uint8 codes into nested_quant_map,
        // scaled per nested block, plus nested_offset.
        let absmax = if let Some(nested) = &state.nested {
            let codes = read_u8(absmax_member)?;
            let nested_absmax_name = format!("{weight}.nested_absmax");
            let map_name = format!("{weight}.nested_quant_map");
            let get = |name: &str| {
                members
                    .get(name)
                    .ok_or_else(|| msg(format!("{name} is missing (BNB-004)")))
            };
            let scales = read_f32(get(&nested_absmax_name)?)?.0;
            let map = read_f32(get(&map_name)?)?.0;
            used.insert(nested_absmax_name);
            used.insert(map_name);
            dequant_nested_absmax(&codes, &map, &scales, nested.blocksize, nested.offset)?
        } else {
            read_f32(absmax_member)?.0
        };
        used.insert(format!("{weight}.quant_map"));
        let packed = read_u8(member)?;
        let shape = state.shape.clone().unwrap_or_else(|| vec![packed.len() as u64 * 2]);
        let table = if state.fp4 { &FP4 } else { &NF4 };
        let values = dequant_4bit(&packed, &absmax, state.blocksize, table, &shape)
            .map_err(|err| msg(format!("{weight}: {err}")))?;
        out.push(FloatTensor {
            name: weight.to_string(),
            shape,
            values,
        });
        used.insert(weight.to_string());
    }
    if out.is_empty() {
        return Err(msg("bitsandbytes dequant found no weight with its absmax"));
    }
    for member in &archive.members {
        if !used.contains(&member.name) {
            out.push(member_f32(member)?);
        }
    }
    Ok(out)
}

struct BnbState {
    fp4: bool,
    blocksize: usize,
    shape: Option<Vec<u64>>,
    nested: Option<BnbNested>,
}

struct BnbNested {
    blocksize: usize,
    offset: f32,
}

impl BnbState {
    fn from_json(state: &Value, key: &str) -> Result<BnbState> {
        let field = |name: &str| {
            state
                .get(name)
                .ok_or_else(|| msg(format!("{key} has no {name}")))
        };
        let fp4 = bnb_quant_type(field("quant_type")?.as_str())?;
        let blocksize = bnb_blocksize(field("blocksize")?.as_u64())?;
        let shape = field("shape")?
            .as_array()
            .and_then(|dims| dims.iter().map(Value::as_u64).collect::<Option<Vec<u64>>>())
            .ok_or_else(|| msg(format!("{key} shape is not a list of sizes")))?;
        let nested = match state.get("nested_blocksize") {
            Some(value) => Some(BnbNested {
                blocksize: value
                    .as_u64()
                    .filter(|size| *size > 0)
                    .ok_or_else(|| msg(format!("{key} nested_blocksize is invalid")))?
                    as usize,
                offset: state
                    .get("nested_offset")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| msg(format!("{key} has no nested_offset")))? as f32,
            }),
            None => None,
        };
        Ok(BnbState {
            fp4,
            blocksize,
            shape: Some(shape),
            nested,
        })
    }

    fn from_config(quant: &Map<String, Value>) -> Result<BnbState> {
        if quant.get("bnb_4bit_use_double_quant").and_then(Value::as_bool) == Some(true) {
            return Err(msg(
                "bitsandbytes nested absmax needs the quant_state tensors, which this file does not have",
            ));
        }
        Ok(BnbState {
            fp4: bnb_quant_type(
                quant
                    .get("bnb_4bit_quant_type")
                    .map_or(Some("nf4"), Value::as_str),
            )?,
            blocksize: bnb_blocksize(
                quant
                    .get("bnb_4bit_quant_blocksize")
                    .map_or(Some(64), Value::as_u64),
            )?,
            shape: None,
            nested: None,
        })
    }
}

/// BNB-001: quant_type is nf4 or fp4.
fn bnb_quant_type(value: Option<&str>) -> Result<bool> {
    match value {
        Some("nf4") => Ok(false),
        Some("fp4") => Ok(true),
        other => Err(msg(format!(
            "bitsandbytes quant_type {other:?} is not nf4 or fp4 (BNB-001)"
        ))),
    }
}

/// BNB-001: the 4-bit blocksizes.
fn bnb_blocksize(value: Option<u64>) -> Result<usize> {
    match value {
        Some(size @ (32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096)) => Ok(size as usize),
        other => Err(msg(format!(
            "bitsandbytes blocksize {other:?} is not 32..4096 (BNB-001)"
        ))),
    }
}

/// `dequant_blockwise_8bit(absmax, state2) + offset` (bitsandbytes.yaml read).
fn dequant_nested_absmax(
    codes: &[u8],
    map: &[f32],
    scales: &[f32],
    blocksize: usize,
    offset: f32,
) -> Result<Vec<f32>> {
    if map.len() != 256 || codes.len().div_ceil(blocksize) != scales.len() {
        return Err(msg(
            "bitsandbytes nested_quant_map or nested_absmax does not fit the absmax codes (BNB-004)",
        ));
    }
    Ok(codes
        .iter()
        .enumerate()
        .map(|(index, code)| map[*code as usize] * scales[index / blocksize] + offset)
        .collect())
}

/// Two codes per byte, the even element in the high nibble (BNB-002), each
/// block of `block` values scaled by its absmax.
pub fn dequant_4bit(
    packed: &[u8],
    absmax: &[f32],
    block: usize,
    table: &[f32; 16],
    shape: &[u64],
) -> Result<Vec<f32>> {
    let n = numel(shape).ok_or_else(|| msg("bitsandbytes shape is too large"))? as usize;
    if packed.len() * 2 < n {
        return Err(msg("bitsandbytes weight holds fewer codes than its shape"));
    }
    if block == 0 || absmax.len() != n.div_ceil(block) {
        return Err(msg("bitsandbytes absmax does not have one value per block"));
    }
    let mut values = Vec::with_capacity(n);
    for byte in packed {
        values.push(table[(byte >> 4) as usize]);
        values.push(table[(byte & 0x0f) as usize]);
    }
    values.truncate(n);
    for (index, value) in values.iter_mut().enumerate() {
        *value *= absmax[index / block];
    }
    Ok(values)
}

const NF4: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

const FP4: [f32; 16] = [
    0.0,
    0.0052083334885537624,
    0.6666666865348816,
    1.0,
    0.3333333432674408,
    0.5,
    0.1666666716337204,
    0.25,
    -0.0,
    -0.0052083334885537624,
    -0.6666666865348816,
    -1.0,
    -0.3333333432674408,
    -0.5,
    -0.1666666716337204,
    -0.25,
];

/// compressed-tensors pack-quantized. Each module's scheme comes from
/// config_groups (CT-008), and `w = (q - zp) * (scale / global_scale)` is
/// applied per strategy (CT-006, CT-007).
fn dequant_compressed(
    archive: &Archive,
    quant: Option<&Map<String, Value>>,
    warnings: &mut Vec<String>,
) -> Result<Vec<FloatTensor>> {
    let quant = quant.ok_or_else(|| msg("compressed-tensors dequant needs quantization_config"))?;
    let modules = module_paths(archive);
    let schemes = CtSchemes::from_config(quant, &modules, warnings)?;
    let members = index(archive);
    let mut used = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for member in &archive.members {
        // CT-009: a module is quantized when its scheme parameters are
        // stored, whatever the targets say: weight_packed for pack-quantized,
        // or weight beside weight_scale for int- and float-quantized.
        let (module, packed) = if let Some(module) = member.name.strip_suffix(".weight_packed") {
            (module, true)
        } else if let Some(module) = member.name.strip_suffix(".weight") {
            if !members.contains_key(format!("{module}.weight_scale").as_str()) {
                continue;
            }
            (module, false)
        } else {
            continue;
        };
        let get = |suffix: &str| members.get(format!("{module}.{suffix}").as_str()).copied();
        let (codes, shape, float_codes) = if packed {
            let shape = weight_shape(archive, &member.name)?;
            (Some(read_i32(member)?), shape, false)
        } else {
            let float_codes = dtype::canonical_id(&member.dtype)? != "int8";
            (None, member.shape.clone(), float_codes)
        };
        // The stored tensors can tell schemes apart that class targets cannot
        // (CT-009): the packed width follows num_bits, and only an asymmetric
        // scheme has a zero point.
        let fits = |scheme: &CtScheme| {
            let width_fits = match &codes {
                Some((_, packed_shape)) => {
                    let cols = shape.last().copied().unwrap_or(0);
                    packed_shape.last().copied() == Some((cols * u64::from(scheme.num_bits)).div_ceil(32))
                }
                None => scheme.float == float_codes,
            };
            width_fits && scheme.symmetric == get("weight_zero_point").is_none()
        };
        let scheme = schemes.scheme_for(module, &fits)?;
        let codes = match codes {
            Some((words, packed_shape)) => {
                unpack_pack_quantized(&words, &packed_shape, scheme.num_bits, &shape)?
                    .into_iter()
                    .map(|code| code as f32)
                    .collect()
            }
            None => stored_codes(member, &scheme)?,
        };
        let scale_member = get("weight_scale")
            .ok_or_else(|| msg(format!("{module}.weight_scale is missing")))?;
        let scale_dtype = dtype::canonical_id(&scale_member.dtype)?;
        let (scale, scale_shape) = read_f32(scale_member)?;
        let zero = match get("weight_zero_point") {
            Some(zp) if !scheme.symmetric => Some(read_zero_point(zp, &scheme, &shape, &scale_shape)?),
            Some(_) => {
                return Err(msg(format!(
                    "{module} has a weight_zero_point but its scheme is symmetric"
                )))
            }
            None if !scheme.symmetric => {
                return Err(msg(format!("{module}.weight_zero_point is missing")))
            }
            None => None,
        };
        let global = match get("weight_global_scale") {
            Some(item) => Some(read_f32(item)?.0),
            None => None,
        };
        let values = dequant_ct(&codes, &shape, &scale, &scale_shape, zero.as_deref(), global.as_deref(), &scheme, scale_dtype)
            .map_err(|err| msg(format!("{module}: {err}")))?;
        out.push(FloatTensor {
            name: format!("{module}.weight"),
            shape,
            values,
        });
        for suffix in ["weight", "weight_packed", "weight_shape", "weight_scale", "weight_zero_point", "weight_global_scale"] {
            used.insert(format!("{module}.{suffix}"));
        }
    }
    for member in &archive.members {
        if !used.contains(&member.name) {
            out.push(member_f32(member)?);
        }
    }
    Ok(out)
}

/// Codes of an int- or float-quantized weight, stored unpacked: int8 for an
/// int scheme, an fp8 type for a float one.
fn stored_codes(member: &Member, scheme: &CtScheme) -> Result<Vec<f32>> {
    let canonical = dtype::canonical_id(&member.dtype)?;
    let raw = member.payload.read_vec()?;
    match (canonical, scheme.float) {
        ("int8", false) => Ok(raw.iter().map(|byte| f32::from(*byte as i8)).collect()),
        (_, true) => {
            let format = dtype::MiniFloat::for_dtype(canonical)
                .map_err(|_| msg(format!("{}: {canonical} is not a float-quantized weight type", member.name)))?;
            Ok(raw.iter().map(|byte| format.decode(*byte)).collect())
        }
        _ => Err(msg(format!(
            "{}: {canonical} does not hold the codes of its int scheme",
            member.name
        ))),
    }
}

/// Every module path in the checkpoint: each dotted prefix of a tensor name.
fn module_paths(archive: &Archive) -> std::collections::BTreeSet<String> {
    let mut paths = std::collections::BTreeSet::new();
    for member in &archive.members {
        let mut end = 0;
        while let Some(dot) = member.name[end..].find('.') {
            end += dot;
            paths.insert(member.name[..end].to_string());
            end += 1;
        }
    }
    paths
}

#[derive(Clone, PartialEq)]
pub(crate) struct CtScheme {
    pub(crate) num_bits: u32,
    float: bool,
    symmetric: bool,
    strategy: String,
    group_size: usize,
    block: Option<(usize, usize)>,
}

pub(crate) enum CtTarget {
    Exact(String),
    Regex(String, regex::Regex),
    /// A name that is neither `re:` nor a module path in the checkpoint
    /// (CT-009).
    Class(String),
}

impl CtTarget {
    fn new(target: &str, modules: &std::collections::BTreeSet<String>) -> Result<CtTarget> {
        if let Some(pattern) = target.strip_prefix("re:") {
            // CT-011: re.match is anchored at the start only.
            let compiled = regex::Regex::new(&format!("^(?:{pattern})")).map_err(|err| {
                msg(format!("compressed-tensors target {target} is not a pattern this reader supports (CT-011): {err}"))
            })?;
            return Ok(CtTarget::Regex(target.to_string(), compiled));
        }
        Ok(if modules.contains(target) {
            CtTarget::Exact(target.to_string())
        } else {
            CtTarget::Class(target.to_string())
        })
    }

    fn text(&self) -> &str {
        match self {
            CtTarget::Exact(text) | CtTarget::Regex(text, _) | CtTarget::Class(text) => text,
        }
    }
}

pub(crate) struct CtSchemes {
    /// target -> scheme, in config_groups then targets order; a later group
    /// replaces an earlier one for the same target string (CT-010).
    targets: Vec<(CtTarget, CtScheme)>,
    ignore: Vec<CtTarget>,
}

impl CtSchemes {
    pub(crate) fn from_config(
        quant: &Map<String, Value>,
        modules: &std::collections::BTreeSet<String>,
        warnings: &mut Vec<String>,
    ) -> Result<CtSchemes> {
        let groups = quant
            .get("config_groups")
            .and_then(Value::as_object)
            .ok_or_else(|| msg("compressed-tensors quantization_config has no config_groups"))?;
        let mut targets: Vec<(CtTarget, CtScheme)> = Vec::new();
        // serde_json keeps the file's key order (preserve_order), which is
        // the order CT-010 asks for.
        for (name, group) in groups {
            if group.is_array() {
                return Err(msg(format!(
                    "config_groups {name} is a preset scheme name; preset_name_to_scheme is not in the spec, so it is refused (CT-010)"
                )));
            }
            let weights = group
                .get("weights")
                .and_then(Value::as_object)
                .ok_or_else(|| msg(format!("config_groups {name} has no weights")))?;
            let num_bits = weights
                .get("num_bits")
                .and_then(Value::as_u64)
                .filter(|bits| (1..=8).contains(bits))
                .ok_or_else(|| msg("compressed-tensors num_bits is not in 1..8 (CT-001)"))?
                as u32;
            let float = match weights.get("type").and_then(Value::as_str).unwrap_or("int") {
                "int" => false,
                "float" => true,
                other => return Err(msg(format!("compressed-tensors weights type {other} is not int or float"))),
            };
            let strategy = weights
                .get("strategy")
                .and_then(Value::as_str)
                .unwrap_or("tensor")
                .to_string();
            let group_size = weights.get("group_size").and_then(Value::as_u64).unwrap_or(0) as usize;
            let block = weights
                .get("block_structure")
                .and_then(Value::as_array)
                .and_then(|dims| match dims.as_slice() {
                    [h, w] => Some((h.as_u64()? as usize, w.as_u64()? as usize)),
                    _ => None,
                });
            let scheme = CtScheme {
                num_bits,
                float,
                symmetric: weights.get("symmetric").and_then(Value::as_bool).unwrap_or(true),
                strategy,
                group_size,
                block,
            };
            for target in group
                .get("targets")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                targets.retain(|(existing, _)| existing.text() != target);
                targets.push((CtTarget::new(target, modules)?, scheme.clone()));
            }
        }
        let mut ignore = Vec::new();
        for entry in quant
            .get("ignore")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            let target = CtTarget::new(entry, modules)?;
            if let CtTarget::Class(name) = &target {
                warnings.push(format!(
                    "ignore entry {name} is not a module in the checkpoint; modules with stored scheme parameters are dequantized whatever it says (CT-009)"
                ));
            }
            ignore.push(target);
        }
        Ok(CtSchemes { targets, ignore })
    }

    /// CT-008: exact module names first, then `re:` patterns (sorted), then
    /// class names (sorted). Without the model code a class target matches
    /// every module that has scheme parameters (CT-009); when class targets
    /// of different schemes compete, `fits` (what the stored tensors allow)
    /// must leave one.
    pub(crate) fn scheme_for(&self, module: &str, fits: &dyn Fn(&CtScheme) -> bool) -> Result<CtScheme> {
        for entry in &self.ignore {
            let ignored = match entry {
                CtTarget::Exact(name) => name == module,
                CtTarget::Regex(_, pattern) => pattern.is_match(module),
                CtTarget::Class(_) => false,
            };
            if ignored {
                return Err(msg(format!(
                    "{module} has quantized parameters but ignore entry {} names it (CT-009)",
                    entry.text()
                )));
            }
        }
        let mut exact = None;
        let mut regex: Vec<&(CtTarget, CtScheme)> = Vec::new();
        let mut class: Vec<&(CtTarget, CtScheme)> = Vec::new();
        for entry in &self.targets {
            match &entry.0 {
                CtTarget::Exact(name) if name == module => exact = Some(entry),
                CtTarget::Exact(_) => {}
                CtTarget::Regex(_, pattern) if pattern.is_match(module) => regex.push(entry),
                CtTarget::Regex(..) => {}
                CtTarget::Class(_) => class.push(entry),
            }
        }
        regex.sort_by(|a, b| a.0.text().cmp(b.0.text()));
        if let Some(entry) = exact.or(regex.first().copied()) {
            return Ok(entry.1.clone());
        }
        let mut schemes: Vec<&CtScheme> = Vec::new();
        for (_, scheme) in &class {
            if !schemes.contains(&scheme) {
                schemes.push(scheme);
            }
        }
        if schemes.len() > 1 {
            schemes.retain(|scheme| fits(scheme));
        }
        match schemes.as_slice() {
            [] => Err(msg(format!("no config_groups target matches {module} (CT-008)"))),
            [scheme] => Ok((*scheme).clone()),
            _ => Err(msg(format!(
                "{module}: class targets of different schemes match it and its tensors do not tell them apart (CT-009)"
            ))),
        }
    }
}

/// Unpacked zero points, shaped like the scale (CT-006: asymmetric group and
/// channel zero points are packed on dim 0).
fn read_zero_point(member: &Member, scheme: &CtScheme, shape: &[u64], scale_shape: &[u64]) -> Result<Vec<f32>> {
    let canonical = dtype::canonical_id(&member.dtype)?;
    if canonical != "int32" {
        return Ok(read_f32(member)?.0);
    }
    let packed = read_i32(member)?;
    let rows = shape[..shape.len().saturating_sub(1)].iter().product::<u64>();
    let cols = *scale_shape.last().unwrap_or(&1);
    let codes = unpack_dim0(&packed.0, &packed.1, scheme.num_bits, rows as usize, cols as usize)?;
    Ok(codes.into_iter().map(|code| code as f32).collect())
}

/// pack_to_int32 along dim 0: element (r, c) sits at bit r*bits of column c.
fn unpack_dim0(packed: &[i32], packed_shape: &[u64], bits: u32, rows: usize, cols: usize) -> Result<Vec<i32>> {
    if packed_shape.len() != 2 || packed_shape[1] as usize != cols {
        return Err(msg("compressed-tensors weight_zero_point is not packed on dim 0"));
    }
    let words = packed_shape[0] as usize;
    let mut out = vec![0i32; rows * cols];
    for r in 0..rows {
        let bit = r * bits as usize;
        for c in 0..cols {
            let mut field = 0u64;
            for k in 0..bits as usize {
                let at = bit + k;
                let word = at / 32;
                if word >= words {
                    return Err(msg("compressed-tensors zero point runs past its packed rows"));
                }
                field |= u64::from((packed[word * cols + c] as u32 >> (at % 32)) & 1) << k;
            }
            out[r * cols + c] = field as i32 - (1 << (bits - 1));
        }
    }
    Ok(out)
}

/// compressed-tensors.yaml `dequantize`, computed in the scale's dtype: each
/// step is rounded to bf16 or f16 when the scale is stored that way.
#[allow(clippy::too_many_arguments)]
fn dequant_ct(
    codes: &[f32],
    shape: &[u64],
    scale: &[f32],
    scale_shape: &[u64],
    zero: Option<&[f32]>,
    global: Option<&[f32]>,
    scheme: &CtScheme,
    scale_dtype: &str,
) -> Result<Vec<f32>> {
    let round = |value: f32| match scale_dtype {
        "bfloat16" => f32::from_bits(u32::from(ggml_dequant::f32_to_bf16(value)) << 16),
        "float16" => ggml_dequant::f16_to_f32(ggml_dequant::f32_to_f16(value)),
        _ => value,
    };
    if shape.len() != 2 {
        return Err(msg("compressed-tensors dequant handles [out, in] weights"));
    }
    let (rows, cols) = (shape[0] as usize, shape[1] as usize);
    let global = match global {
        Some([value]) => Some(*value),
        Some(_) => return Err(msg("weight_global_scale is not one value")),
        None => None,
    };
    let scale: Vec<f32> = scale
        .iter()
        .map(|value| global.map_or(*value, |g| round(value / g)))
        .collect();
    // Index of the scale (and zero) for element (r, c).
    let scale_cols = *scale_shape.last().unwrap_or(&1) as usize;
    let pick: Box<dyn Fn(usize, usize) -> usize> = match scheme.strategy.as_str() {
        "tensor" => {
            if scale.len() != 1 {
                return Err(msg("tensor strategy needs one scale"));
            }
            Box::new(|_, _| 0)
        }
        "channel" => {
            if scale.len() != rows {
                return Err(msg("channel strategy needs one scale per row"));
            }
            Box::new(|r, _| r)
        }
        "group" | "tensor_group" => {
            let group = scheme.group_size;
            if group == 0 || (cols >= group && cols % group != 0) {
                return Err(msg("in_features is not divisible by group_size (CT-007)"));
            }
            if scale.len() != rows * cols.div_ceil(group) || scale_cols != cols.div_ceil(group) {
                return Err(msg("group scales are not [out, ceil(in / group_size)]"));
            }
            Box::new(move |r, c| r * scale_cols + c / group)
        }
        "block" => {
            let (bh, bw) = scheme
                .block
                .filter(|(h, w)| *h > 0 && *w > 0)
                .ok_or_else(|| msg("block strategy needs block_structure [h, w]"))?;
            let block_cols = cols.div_ceil(bw);
            if scale.len() != rows.div_ceil(bh) * block_cols {
                return Err(msg("block scales do not cover the padded weight"));
            }
            Box::new(move |r, c| (r / bh) * block_cols + c / bw)
        }
        other => return Err(msg(format!("compressed-tensors strategy {other} is not in CT-007"))),
    };
    if codes.len() != rows * cols {
        return Err(msg("compressed-tensors codes do not match weight_shape"));
    }
    if let Some(zero) = zero {
        if zero.len() != scale.len() {
            return Err(msg("weight_zero_point does not match the scale layout"));
        }
    }
    let mut out = Vec::with_capacity(codes.len());
    for r in 0..rows {
        for c in 0..cols {
            let at = pick(r, c);
            let mut q = round(codes[r * cols + c]);
            if let Some(zero) = zero {
                q = round(q - round(zero[at]));
            }
            out.push(round(q * scale[at]));
        }
    }
    Ok(out)
}

fn weight_shape(archive: &Archive, packed_name: &str) -> Result<Vec<u64>> {
    let shape_name = format!(
        "{}.weight_shape",
        packed_name.trim_end_matches(".weight_packed")
    );
    if let Some(member) = archive
        .members
        .iter()
        .find(|member| member.name == shape_name)
    {
        let raw = member.payload.read_vec()?;
        let width = match dtype::canonical_id(&member.dtype).unwrap_or("int64") {
            "int32" | "uint32" => 4,
            _ => 8,
        };
        let mut shape = Vec::new();
        for chunk in raw.chunks_exact(width) {
            let dim = if width == 4 {
                i32::from_le_bytes(chunk.try_into().unwrap()) as i64
            } else {
                i64::from_le_bytes(chunk.try_into().unwrap())
            };
            shape.push(u64::try_from(dim).map_err(|_| msg(format!("{shape_name} has a negative size")))?);
        }
        if !shape.is_empty() {
            return Ok(shape);
        }
    }
    Err(msg(format!("{packed_name} needs a weight_shape tensor")))
}

pub fn unpack_pack_quantized(
    packed: &[i32],
    packed_shape: &[u64],
    num_bits: u32,
    shape: &[u64],
) -> Result<Vec<i32>> {
    if !(1..=8).contains(&num_bits) {
        return Err(msg(format!("compressed-tensors num_bits {num_bits} is not in 1..8 (CT-001)")));
    }
    let cols = *shape.last().unwrap_or(&0) as usize;
    let rows = numel(&shape[..shape.len().saturating_sub(1)])
        .ok_or_else(|| msg("weight_shape is too large"))? as usize;
    let words = packed_shape.last().copied().unwrap_or(0) as usize;
    // CT-004: weight_shape is the logical shape; the packed rows must match
    // it and the packed width must hold every column.
    let packed_rows = numel(&packed_shape[..packed_shape.len().saturating_sub(1)]).unwrap_or(0) as usize;
    if packed_rows != rows || packed.len() != rows * words || cols * num_bits as usize > words * 32 {
        return Err(msg("weight_packed does not hold weight_shape (CT-004)"));
    }
    let mut codes = vec![0i32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let bit = col * num_bits as usize;
            let word = bit / 32;
            let off = bit % 32;
            let lo_bits = (32 - off).min(num_bits as usize);
            if word >= words {
                return Err(msg("compressed-tensors field runs past weight_packed"));
            }
            let base = row * words + word;
            let mut field = (packed[base] as u32 >> off) & ((1u32 << lo_bits) - 1);
            if (lo_bits as u32) < num_bits {
                let hi_bits = num_bits as usize - lo_bits;
                field |= (packed[base + 1] as u32 & ((1u32 << hi_bits) - 1)) << lo_bits;
            }
            codes[row * cols + col] = field as i32 - (1 << (num_bits - 1));
        }
    }
    Ok(codes)
}

fn dequant_mlx(archive: &Archive, config: Option<&Value>) -> Result<Vec<FloatTensor>> {
    let object =
        mlx_config(config).ok_or_else(|| msg("MLX dequant needs config.json quantization"))?;
    let top = MlxParams::top_level(object)?;
    let members = index(archive);
    let mut consumed = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for member in &archive.members {
        let Some(prefix) = member.name.strip_suffix(".weight") else {
            continue;
        };
        let scales_name = format!("{prefix}.scales");
        let Some(scales_member) = members.get(scales_name.as_str()) else {
            continue;
        };
        let params = match object.get(prefix) {
            None => MlxParams::infer(&top, member, scales_member)
                .map_err(|err| msg(format!("{prefix}: {err}")))?,
            Some(_) => MlxParams::for_module(object, prefix, &top)?
                .ok_or_else(|| msg(format!("{prefix} has scales but quantization.{prefix} is false")))?,
        };
        let (bits, group) = (params.bits, params.group_size);
        let weight = read_u32(member)?;
        let scale_shape = scales_member.shape.clone();
        let values = if params.mode == "affine" {
            let scales = read_f32(scales_member)?;
            let biases_name = format!("{prefix}.biases");
            let biases = members
                .get(biases_name.as_str())
                .ok_or_else(|| msg(format!("{biases_name} is missing (MLX-008)")))?;
            let biases = read_f32(biases)?;
            consumed.insert(biases_name);
            dequant_affine(&weight, &member.shape, &scales.0, &biases.0, group, bits)?
        } else {
            let scale_bytes = fp_scale_bytes(scales_member)?;
            let global_name = format!("{prefix}.global_scale");
            let global = match members.get(global_name.as_str()) {
                Some(item) => {
                    consumed.insert(global_name);
                    Some(read_f32(item)?.0)
                }
                None => None,
            };
            dequant_fp(&weight, &scale_bytes, group, bits, &params.mode, global.as_deref())?
        };
        let scale_cols = *scale_shape.last().unwrap_or(&1);
        out.push(FloatTensor {
            name: member.name.clone(),
            shape: {
                let mut shape = scale_shape;
                if let Some(last) = shape.last_mut() {
                    *last = scale_cols * group as u64;
                }
                shape
            },
            values,
        });
        consumed.insert(member.name.clone());
        consumed.insert(scales_name);
    }
    for member in &archive.members {
        if !consumed.contains(&member.name) {
            out.push(member_f32(member)?);
        }
    }
    Ok(out)
}

struct MlxParams {
    mode: String,
    group_size: usize,
    bits: u32,
}

impl MlxParams {
    /// to_quantized's defaults per mode (MLX-013).
    fn defaults(mode: &str) -> Result<(usize, u32)> {
        Ok(match mode {
            "affine" => (64, 4),
            "mxfp4" => (32, 4),
            "nvfp4" => (16, 4),
            "mxfp8" => (32, 8),
            other => return Err(msg(format!("MLX quantization mode {other} is unknown"))),
        })
    }

    fn from_object(object: &Map<String, Value>) -> Result<MlxParams> {
        let mode = object.get("mode").and_then(Value::as_str).unwrap_or("affine").to_string();
        let (default_group, default_bits) = Self::defaults(&mode)?;
        let params = MlxParams {
            group_size: object
                .get("group_size")
                .and_then(Value::as_u64)
                .map_or(default_group, |value| value as usize),
            bits: object
                .get("bits")
                .and_then(Value::as_u64)
                .map_or(default_bits, |value| value as u32),
            mode,
        };
        params.check()?;
        Ok(params)
    }

    fn top_level(object: &Map<String, Value>) -> Result<MlxParams> {
        Self::from_object(object)
    }

    /// MLX-013: `quantization.<module>` replaces the defaults. true uses the
    /// top level, false leaves the module dense (None), and an object is
    /// filled from its mode's defaults, not from the top level.
    fn for_module(object: &Map<String, Value>, module: &str, top: &MlxParams) -> Result<Option<MlxParams>> {
        match object.get(module) {
            Some(Value::Bool(true)) => Ok(Some(MlxParams {
                mode: top.mode.clone(),
                group_size: top.group_size,
                bits: top.bits,
            })),
            Some(Value::Bool(false)) => Ok(None),
            Some(Value::Object(entry)) => Self::from_object(entry).map(Some),
            _ => Err(msg(format!("quantization.{module} is not true, false, or an object"))),
        }
    }

    /// MLX-016: mlx-lm's infer_quant_config for a module with scales and no
    /// `quantization.<module>` entry. It needs the unquantized input width,
    /// which the files do not hold, but bits * group_size = 32 * W / S does
    /// not. uint8 scales name the mode by that product alone; for affine the
    /// top-level values are used only when their product matches.
    fn infer(top: &MlxParams, weight: &Member, scales: &Member) -> Result<MlxParams> {
        let words = weight.shape.last().copied().unwrap_or(0);
        let groups = scales.shape.last().copied().unwrap_or(0);
        if groups == 0 || (words * 32) % groups != 0 {
            return Err(msg("the weight and scales widths do not give bits * group_size (MLX-016)"));
        }
        let product = words * 32 / groups;
        let params = if dtype::canonical_id(&scales.dtype)? == "uint8" {
            let (mode, bits, group_size) = match product {
                64 => ("nvfp4", 4, 16),
                128 => ("mxfp4", 4, 32),
                256 => ("mxfp8", 8, 32),
                _ => return Err(msg(format!(
                    "uint8 scales with bits * group_size {product} name no mode (MLX-016)"
                ))),
            };
            MlxParams {
                mode: mode.to_string(),
                group_size,
                bits,
            }
        } else {
            if top.bits as u64 * top.group_size as u64 != product {
                return Err(msg(format!(
                    "no quantization entry, and the top-level bits {} and group_size {} do not give the stored bits * group_size {product} (MLX-016)",
                    top.bits, top.group_size
                )));
            }
            MlxParams {
                mode: "affine".to_string(),
                group_size: top.group_size,
                bits: top.bits,
            }
        };
        params.check()?;
        Ok(params)
    }

    /// MLX-006 and MLX-008.
    fn check(&self) -> Result<()> {
        let ok = match self.mode.as_str() {
            "affine" => {
                matches!(self.group_size, 32 | 64 | 128) && matches!(self.bits, 1..=6 | 8)
            }
            "mxfp4" => (self.group_size, self.bits) == (32, 4),
            "nvfp4" => (self.group_size, self.bits) == (16, 4),
            "mxfp8" => (self.group_size, self.bits) == (32, 8),
            _ => false,
        };
        if ok {
            Ok(())
        } else {
            Err(msg(format!(
                "MLX mode {} with group_size {} and bits {} is not allowed (MLX-006, MLX-008)",
                self.mode, self.group_size, self.bits
            )))
        }
    }
}

fn dequant_affine(
    weight: &[u32],
    weight_shape: &[u64],
    scales: &[f32],
    biases: &[f32],
    group: usize,
    bits: u32,
) -> Result<Vec<f32>> {
    let codes = unpack_mlx(weight, weight_shape, bits)?;
    let groups = scales.len();
    if codes.len() != groups * group || biases.len() != groups {
        return Err(msg("MLX affine scales do not match the unpacked weight"));
    }
    let mut out = Vec::with_capacity(codes.len());
    for (index, code) in codes.iter().enumerate() {
        let bucket = index / group;
        out.push(*code as f32 * scales[bucket] + biases[bucket]);
    }
    Ok(out)
}

fn dequant_fp(
    weight: &[u32],
    scales: &[u8],
    group: usize,
    bits: u32,
    mode: &str,
    global: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let bytes: Vec<u8> = weight.iter().flat_map(|word| word.to_le_bytes()).collect();
    let mut values = Vec::new();
    if bits == 4 {
        let lut = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        for byte in bytes {
            values.push(lut[(byte & 0x0f) as usize]);
            values.push(lut[(byte >> 4) as usize]);
        }
    } else if bits == 8 {
        let e4m3 = dtype::MiniFloat::for_dtype("float8_e4m3fn")?;
        for byte in &bytes {
            values.push(e4m3.decode(*byte));
        }
    } else {
        return Err(msg(format!(
            "MLX fp mode {mode} bits {bits} are not 4 or 8"
        )));
    }
    let expect = scales.len() * group;
    if values.len() < expect {
        return Err(msg(
            "MLX fp values are shorter than scales times group_size",
        ));
    }
    values.truncate(expect);
    let decoded: Vec<f32> = if mode == "nvfp4" {
        // MLX-014: one global_scale, or one per expert along axis 0.
        let globals = global.unwrap_or(&[]);
        let experts = globals.len().max(1);
        if scales.len() % experts != 0 {
            return Err(msg("MLX global_scale does not divide the scales into experts (MLX-014)"));
        }
        let per_expert = scales.len() / experts;
        let e4m3 = dtype::MiniFloat::for_dtype("float8_e4m3fn")?;
        scales
            .iter()
            .enumerate()
            .map(|(index, scale)| {
                let factor = globals
                    .get(index / per_expert)
                    .map_or(1.0, |value| value / (448.0 * 6.0));
                e4m3.decode(*scale) * factor
            })
            .collect()
    } else {
        if global.is_some() {
            return Err(msg(format!("MLX mode {mode} has no global_scale")));
        }
        let e8m0 = dtype::MiniFloat::for_dtype("float8_e8m0fnu")?;
        scales.iter().map(|scale| e8m0.decode(*scale)).collect()
    };
    for (index, value) in values.iter_mut().enumerate() {
        *value *= decoded[index / group];
    }
    Ok(values)
}

fn fp_scale_bytes(member: &Member) -> Result<Vec<u8>> {
    let id = dtype::canonical_id(&member.dtype)?;
    if id != "uint8" {
        return Err(msg(format!(
            "{} is {id}; MLX fp scales are uint8",
            member.name
        )));
    }
    member.payload.read_vec()
}

fn unpack_mlx(words: &[u32], shape: &[u64], bits: u32) -> Result<Vec<u32>> {
    if matches!(bits, 3 | 5 | 6) {
        return unpack_bitstream(words, shape, bits);
    }
    unpack_pow2(words, shape, bits)
}

fn unpack_bitstream(words: &[u32], shape: &[u64], bits: u32) -> Result<Vec<u32>> {
    let cols = shape.last().copied().unwrap_or(0) as usize;
    if numel(shape) != Some(words.len() as u64) {
        return Err(msg("MLX weight length does not match its shape"));
    }
    if cols == 0 || (cols * 32) % bits as usize != 0 {
        return Err(msg(
            "MLX bitstream width is not a multiple of the bit width",
        ));
    }
    let rows = numel(&shape[..shape.len().saturating_sub(1)]).unwrap_or(1) as usize;
    let codes_per_row = cols * 32 / bits as usize;
    let mut out = Vec::with_capacity(rows * codes_per_row);
    for row in 0..rows {
        let mut acc = 0u64;
        let mut have = 0u32;
        let mut produced = 0usize;
        for col in 0..cols {
            acc |= u64::from(words[row * cols + col]) << have;
            have += 32;
            while have >= bits && produced < codes_per_row {
                out.push((acc & ((1u64 << bits) - 1)) as u32);
                acc >>= bits;
                have -= bits;
                produced += 1;
            }
        }
        if produced != codes_per_row {
            return Err(msg("MLX bitstream did not fill the row"));
        }
    }
    Ok(out)
}

fn unpack_pow2(words: &[u32], shape: &[u64], bits: u32) -> Result<Vec<u32>> {
    if !matches!(bits, 1 | 2 | 4 | 8) {
        return Err(msg(format!("MLX bits {bits} are not a power-of-two pack")));
    }
    if numel(shape) != Some(words.len() as u64) {
        return Err(msg("MLX weight length does not match its shape"));
    }
    let per = (32 / bits) as usize;
    let cols = shape.last().copied().unwrap_or(0) as usize;
    let rows = numel(&shape[..shape.len().saturating_sub(1)]).unwrap_or(1) as usize;
    let mut out = Vec::with_capacity(rows * cols * per);
    for row in 0..rows {
        for col in 0..cols {
            let word = words[row * cols + col];
            for index in 0..per {
                let shift = index as u32 * bits;
                out.push((word >> shift) & ((1 << bits) - 1));
            }
        }
    }
    Ok(out)
}

fn write_f32(path: &Path, target: FormatName, tensors: &[FloatTensor]) -> Result<()> {
    match target {
        #[cfg(format_gguf)]
        FormatName::Gguf => {
            return Err(msg(
                "dequant output is a float container; pass a .safetensors, .npy, or .npz path. --quantize writes GGUF",
            ));
        }
        _ => {}
    }
    let owned: Vec<crate::formats::safetensors::OutTensor> = tensors
        .iter()
        .map(|tensor| {
            let bytes = tensor
                .values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            crate::formats::safetensors::OutTensor {
                name: tensor.name.clone(),
                dtype: "F32".into(),
                shape: tensor.shape.clone(),
                bytes: Payload::Bytes(bytes),
            }
        })
        .collect();
    match target {
        #[cfg(format_safetensors)]
        FormatName::Safetensors => crate::formats::safetensors::write_file(path, &[], &owned),
        #[cfg(format_npy)]
        FormatName::Npy => {
            if owned.len() != 1 {
                return Err(msg("an npy file holds one tensor"));
            }
            let tensor = &owned[0];
            crate::formats::npy::write_file(path, "<f4", &tensor.shape, |file| {
                tensor.bytes.copy_to(file)
            })
        }
        #[cfg(format_npz)]
        FormatName::Npz => {
            let members = owned
                .iter()
                .map(|tensor| crate::formats::npz::NpzMember {
                    key: tensor.name.clone(),
                    body: crate::formats::npz::NpzMemberBody::Rebuild {
                        descr: "<f4".into(),
                        shape: tensor.shape.clone(),
                        payload: tensor.bytes.clone(),
                    },
                })
                .collect::<Vec<_>>();
            crate::formats::npz::write_archive(path, &members)
        }
        other => Err(msg(format!(
            "tensquant writes safetensors, npy, or npz, not {}",
            other.as_str()
        ))),
    }
}

fn float_copy(archive: &Archive) -> Result<Vec<FloatTensor>> {
    archive
        .members
        .iter()
        .map(|member| {
            let mut tensor = member_f32(member)?;
            tensor.shape = canonical_shape(archive, member);
            Ok(tensor)
        })
        .collect()
}

/// Outermost-first shape. GGUF members carry ggml `ne[]`, innermost first.
fn canonical_shape(archive: &Archive, member: &Member) -> Vec<u64> {
    let mut shape = member.shape.clone();
    if archive.format == FormatName::Gguf {
        shape.reverse();
    }
    shape
}

fn member_f32(member: &Member) -> Result<FloatTensor> {
    let canonical = dtype::canonical_id(&member.dtype)?;
    let raw = member.payload.read_vec()?;
    let values = match canonical {
        "float32" => raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect(),
        "float16" => raw
            .chunks_exact(2)
            .map(|chunk| ggml_dequant::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect(),
        "bfloat16" => raw
            .chunks_exact(2)
            .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16))
            .collect(),
        other => {
            return Err(msg(format!(
                "tensor {} dtype {other} is not a float to dequantize",
                member.name
            )))
        }
    };
    Ok(FloatTensor {
        name: member.name.clone(),
        shape: member.shape.clone(),
        values,
    })
}

fn read_i32(member: &Member) -> Result<(Vec<i32>, Vec<u64>)> {
    let raw = member.payload.read_vec()?;
    Ok((
        raw.chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
            .collect(),
        member.shape.clone(),
    ))
}

fn read_u32(member: &Member) -> Result<Vec<u32>> {
    let raw = member.payload.read_vec()?;
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn read_u8(member: &Member) -> Result<Vec<u8>> {
    member.payload.read_vec()
}

fn read_f32(member: &Member) -> Result<(Vec<f32>, Vec<u64>)> {
    let tensor = member_f32(member)?;
    Ok((tensor.values, member.shape.clone()))
}

fn index(archive: &Archive) -> BTreeMap<&str, &Member> {
    archive
        .members
        .iter()
        .map(|member| (member.name.as_str(), member))
        .collect()
}

/// `quantization_config` with lowercase keys and lowercase quant_method,
/// checkpoint_format, and version values, as AutoGPTQ and AutoAWQ read it
/// (GPTQ-007, AWQ-006).
fn quant_config(config: Option<&Value>) -> Option<Map<String, Value>> {
    let object = config?.get("quantization_config")?.as_object()?;
    Some(
        object
            .iter()
            .map(|(key, value)| {
                let key = key.to_ascii_lowercase();
                let value = match (key.as_str(), value) {
                    ("quant_method" | "checkpoint_format" | "version", Value::String(text)) => {
                        Value::String(text.to_ascii_lowercase())
                    }
                    _ => value.clone(),
                };
                (key, value)
            })
            .collect(),
    )
}

/// Bit width from `bits` or its synonym `w_bit` (GPTQ-007, AWQ-007). Both
/// present and different is refused.
fn config_bits(quant: Option<&Map<String, Value>>, pack: &str) -> Result<u32> {
    let get = |key: &str| quant.and_then(|map| map.get(key)).and_then(Value::as_u64);
    let bits = match (get("bits"), get("w_bit")) {
        (Some(bits), Some(w_bit)) if bits != w_bit => {
            return Err(msg(format!(
                "quantization_config bits {bits} and w_bit {w_bit} disagree (AWQ-007)"
            )))
        }
        (Some(bits), _) | (None, Some(bits)) => bits,
        (None, None) => {
            return Err(msg(format!("{pack} dequant needs quantization_config.bits")))
        }
    };
    u32::try_from(bits).map_err(|_| msg(format!("{pack} bits {bits} are out of range")))
}

fn quant_object(config: Option<&Value>) -> Option<&Map<String, Value>> {
    config
        .and_then(|value| {
            value
                .get("quantization_config")
                .or_else(|| value.get("quantization"))
        })
        .and_then(Value::as_object)
}

fn mlx_config(config: Option<&Value>) -> Option<&Map<String, Value>> {
    let object = quant_object(config)?;
    if object.contains_key("bits")
        && object.contains_key("group_size")
        && !object.contains_key("quant_method")
    {
        Some(object)
    } else {
        config
            .and_then(|value| value.get("quantization"))
            .and_then(Value::as_object)
    }
}

fn read_config(dir: &Path) -> Result<Option<Value>> {
    let path = dir.join("config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path).map_err(|err| crate::error::io_err(&path, err))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|_| msg(format!("{} is not JSON", path.display())))?;
    Ok(Some(value))
}

fn output_format(path: &Path, explicit: Option<FormatName>) -> Result<FormatName> {
    if let Some(format) = explicit {
        return Ok(format);
    }
    let name = path
        .file_name()
        .and_then(|text| text.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if name.ends_with(".safetensors") {
        FormatName::parse("safetensors")
    } else if name.ends_with(".npy") {
        FormatName::parse("npy")
    } else if name.ends_with(".npz") {
        FormatName::parse("npz")
    } else if name.ends_with(".gguf") {
        FormatName::parse("gguf")
    } else {
        Err(msg(
            "pass --format; the output name does not say which container",
        ))
    }
}

pub fn quant_main(args: &[String]) -> Result<i32> {
    match parse_quant(args)? {
        QuantCommand::Help => {
            println!("{}", quant_help());
            Ok(0)
        }
        QuantCommand::Version => {
            println!("{}", crate::version_line("tensquant"));
            Ok(0)
        }
        QuantCommand::Run(request) => {
            let warnings = if request.quantize.is_some() {
                quantize_file(&request)?
            } else {
                run(&request)?
            };
            for warning in warnings {
                eprintln!("tensquant: {warning}");
            }
            Ok(0)
        }
    }
}

fn quantize_file(request: &Request) -> Result<Vec<String>> {
    let kind = request.quantize.as_deref().unwrap_or("");
    let (block_type, file_type) = match kind {
        "q4_0" => (2, 2),
        "q8_0" => (8, 7),
        _ => return Err(msg("--quantize accepts q4_0 or q8_0")),
    };
    if output_format(&request.output, request.format)? != FormatName::Gguf {
        return Err(msg("--quantize writes GGUF; name the output .gguf or pass --format gguf"));
    }
    crate::io_util::refuse_existing(&request.output, request.force)?;
    let resolved = archive::resolve_user_path(&request.source, request.force_single)?;
    let archive = archive::open(&resolved.path, None, DEFAULT_MAX_HEADER, None, None)?;
    let mut dense = Vec::new();
    let mut warnings = Vec::new();
    let mut metadata = vec![
        ("general.quantization_version".into(), MetaValue::U32(2)),
        // llama_ftype MOSTLY_Q4_0 / MOSTLY_Q8_0 (gguf.yaml).
        ("general.file_type".into(), MetaValue::U32(file_type)),
    ];
    for member in &archive.members {
        let canonical = dtype::canonical_id(&member.dtype)?;
        let shape = canonical_shape(&archive, member);
        let mut ne = shape.clone();
        ne.reverse();
        if shape.is_empty() {
            ne = vec![1];
            metadata.push((
                format!("{}.orig_shape", member.name),
                MetaValue::Array {
                    elem: crate::formats::gguf::TYPE_ARRAY_U64,
                    values: Vec::new(),
                },
            ));
        }
        let is_float = matches!(canonical, "float32" | "float16" | "bfloat16");
        // Other element types are copied as they are.
        if !is_float {
            let code = dtype::element_ggml_code(canonical).ok_or_else(|| {
                msg(format!("{} dtype {canonical} has no GGUF type", member.name))
            })?;
            dense.push(DenseTensor {
                name: member.name.clone(),
                ggml_type: code,
                ne,
                payload: member.payload.clone(),
            });
            continue;
        }
        let tensor = member_f32(member)?;
        let (ggml_type, payload) = if keeps_f32(&tensor.name, shape.len()) {
            (0, tensor.values.iter().flat_map(|value| value.to_le_bytes()).collect())
        } else if ne[0] % 32 != 0 {
            // GGUF-024 step 5: a tensor the quantizer cannot take falls back
            // to F16.
            warnings.push(format!(
                "{} innermost size {} is not a multiple of 32; it is written F16 (GGUF-024)",
                tensor.name, ne[0]
            ));
            (
                1,
                tensor
                    .values
                    .iter()
                    .flat_map(|value| ggml_dequant::f32_to_f16(*value).to_le_bytes())
                    .collect(),
            )
        } else {
            let packed = if block_type == 2 {
                ggml_dequant::quantize_q4_0(&tensor.values)?
            } else {
                ggml_dequant::quantize_q8_0(&tensor.values)?
            };
            (block_type, packed)
        };
        dense.push(DenseTensor {
            name: tensor.name,
            ggml_type,
            ne,
            payload: Payload::Bytes(payload),
        });
    }
    write_dense(&request.output, &metadata, &dense)?;
    Ok(warnings)
}

/// GGUF-024 steps 2 and 3: the tensors convert_hf_to_gguf keeps F32. Rank
/// 1 and below, `*_norm.weight`, every name that does not end in `.weight`,
/// `.lora_a`, or `.lora_b`, and a fixed list of roles. The roles are matched
/// by name whatever the architecture.
fn keeps_f32(name: &str, rank: usize) -> bool {
    const ROLES: &[&str] = &[
        "ffn_gate_inp", "ffn_gate_inp_shexp", "position_embd", "token_types", "ssm_conv1d",
        "shortconv.conv", "time_mix_first", "time_mix_w1", "time_mix_w2", "time_mix_decay_w1",
        "time_mix_decay_w2", "time_mix_lerp_fused", "v.position_embd", "a.position_embd",
        "altup_correct_coef", "altup_predict_coef", "ssm_conv1d_q", "ssm_conv1d_k", "ssm_conv1d_v",
        "indexer.proj",
    ];
    if rank <= 1 || name.ends_with("_norm.weight") {
        return true;
    }
    if ![".weight", ".lora_a", ".lora_b"].iter().any(|suffix| name.ends_with(suffix)) {
        return true;
    }
    let Some(role) = name.strip_suffix(".weight") else {
        return false;
    };
    let role = match role.strip_prefix("blk.") {
        Some(rest) => match rest.split_once('.') {
            Some((block, role)) if block.bytes().all(|byte| byte.is_ascii_digit()) => role,
            _ => role,
        },
        None => role,
    };
    if let Some(rest) = role.strip_prefix("posnet.") {
        if let Some((block, norm)) = rest.split_once('.') {
            return block.bytes().all(|byte| byte.is_ascii_digit()) && (norm == "norm1" || norm == "norm2");
        }
    }
    ROLES.contains(&role)
}

enum QuantCommand {
    Help,
    Version,
    Run(Request),
}

fn parse_quant(args: &[String]) -> Result<QuantCommand> {
    let mut output = None;
    let mut format = None;
    let mut quantize = None;
    let mut dequant = false;
    let mut force = false;
    let mut force_next = false;
    let mut source = None;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--help" || arg == "-h" {
            return Ok(QuantCommand::Help);
        }
        if arg == "--version" {
            return Ok(QuantCommand::Version);
        }
        if arg == "--dequant" {
            dequant = true;
            index += 1;
            continue;
        }
        if arg == "--force" {
            force = true;
            index += 1;
            continue;
        }
        if arg == "--force-single-file" {
            force_next = true;
            index += 1;
            continue;
        }
        if arg == "--quantize" || arg == "--format" || arg == "-o" || arg == "--output" {
            index += 1;
            let value = args
                .get(index)
                .ok_or_else(|| msg(format!("{arg} needs a value")))?;
            if arg == "--quantize" {
                quantize = Some(value.clone());
            } else if arg == "--format" {
                format = Some(FormatName::parse(value)?);
            } else {
                output = Some(PathBuf::from(value));
            }
            index += 1;
            continue;
        }
        if arg.starts_with('-') {
            return Err(msg(format!("unknown option {arg}")));
        }
        if source.is_some() {
            return Err(msg("tensquant takes one source path"));
        }
        source = Some((PathBuf::from(arg), force_next));
        force_next = false;
        index += 1;
    }
    if quantize.is_some() && dequant {
        return Err(msg("pass either --dequant or --quantize"));
    }
    if quantize.is_none() && !dequant {
        return Err(msg("pass --dequant or --quantize q4_0|q8_0"));
    }
    let Some((source, force_single)) = source else {
        return Err(msg("usage: tensquant --dequant -o OUT SRC"));
    };
    let Some(output) = output else {
        return Err(msg("tensquant needs -o OUT"));
    };
    Ok(QuantCommand::Run(Request {
        source,
        output,
        format,
        quantize,
        force_single,
        force,
    }))
}

pub fn quant_help() -> &'static str {
    "\
tensquant — dequantize a written pack, or quantize with Q4_0 or Q8_0

Usage:
  tensquant --dequant -o OUT [--format NAME] [--force] [--force-single-file] SRC
  tensquant --quantize q4_0|q8_0 -o OUT.gguf [--force] [--force-single-file] SRC

--dequant writes float32. GGML blocks use the dequantize snippets. Q8_1 is
refused. GPTQ uses the 2, 4, and 8 bit cuda pack, which is also the stored
exllama layout. AWQ uses the gemm nibble order. bitsandbytes uses the NF4
or FP4 codebook. compressed-tensors unpacks pack-quantized codes.
MLX uses quantization in config.json.
GPTQ 3-bit, marlin, EXL2 (EXL2-Q1), and ModelOpt NVFP4 (CT-Q1) stop,
because those inverses are not written out.
--quantize q4_0 and --quantize q8_0 use the reference quantizers. K-quants
and IQ types are not quantized. The result is not the original float.
"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_pack_dequant_matches_the_formula() {
        let bits = 4u32;
        let codes = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let mut word = 0u32;
        for (index, code) in codes.iter().enumerate() {
            word |= code << (bits * index as u32);
        }
        // Four packed rows hold in_features 32 (GPTQ-002); out_features 8.
        let qweight = [word as i32; 32];
        let qzeros = [0i32; 1];
        let scales = [2.0f32; 8];
        let g_idx = vec![0; 32];
        let values = unpack_gptq(
            &qweight,
            &[4, 8],
            &qzeros,
            &[1, 1],
            &scales,
            &[1, 8],
            &g_idx,
            bits,
            1,
        )
        .unwrap();
        assert_eq!(values.len(), 256);
        // Output row 0 is [out=0, in=0..32]; in 0 and 1 are codes 1 and 2.
        assert!((values[0] - (1.0 - 1.0) * 2.0).abs() < 1e-6);
        assert!((values[1] - (2.0 - 1.0) * 2.0).abs() < 1e-6);
    }

    #[test]
    fn quantize_keeps_the_gguf_024_tensors_f32() {
        assert!(super::keeps_f32("blk.0.attn_norm.weight", 1));
        assert!(super::keeps_f32("blk.0.ffn_gate_inp.weight", 2));
        assert!(super::keeps_f32("blk.3.attn_k_norm.weight", 2));
        assert!(super::keeps_f32("blk.0.attn_q.bias", 2));
        assert!(super::keeps_f32("posnet.1.norm2.weight", 2));
        assert!(super::keeps_f32("position_embd.weight", 2));
        assert!(!super::keeps_f32("token_embd.weight", 2));
        assert!(!super::keeps_f32("blk.0.attn_q.weight", 2));
        assert!(!super::keeps_f32("blk.0.attn_q.lora_a", 2));
    }

    #[test]
    fn marlin_exl2_and_modelopt_are_named_refusals() {
        let marlin =
            serde_json::json!({"quantization_config": {"checkpoint_format": "marlin", "bits": 4}});
        let archive = empty_archive();
        let err = refuse_unsupported(&archive, Some(&marlin)).unwrap_err();
        assert!(err.to_string().contains("marlin"), "{err}");

        let mut exl2 = empty_archive();
        exl2.members.push(named("model.q_weight"));
        let err = refuse_unsupported(&exl2, None).unwrap_err();
        assert!(err.to_string().contains("EXL2-Q1"), "{err}");

        let mut modelopt = empty_archive();
        modelopt.members.push(named("layer.weight_scale_2"));
        let err = refuse_unsupported(&modelopt, None).unwrap_err();
        assert!(err.to_string().contains("CT-Q1"), "{err}");

        let three = serde_json::json!({"quantization_config": {"bits": 3}});
        let err = refuse_unsupported(&archive, Some(&three)).unwrap_err();
        assert!(err.to_string().contains("3-bit"), "{err}");
    }

    fn named(name: &str) -> crate::archive::Member {
        crate::archive::Member {
            name: name.into(),
            dtype: "I32".into(),
            shape: vec![1],
            file_offset: 0,
            nbytes: 4,
            numel: Some(1),
            payload: crate::payload::Payload::Bytes(vec![0, 0, 0, 0]),
            shard: None,
            detail: None,
            fortran: false,
        }
    }
}

#[cfg(test)]
fn empty_archive() -> Archive {
    Archive {
        format: FormatName::Safetensors,
        version: "header+blob".into(),
        path: PathBuf::from("model.safetensors"),
        members: Vec::new(),
        metadata: Vec::new(),
        data_bytes: 0,
        notes: Vec::new(),
        inner: Inner::Safe,
    }
}
