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
    if request.output.exists() && !request.force {
        return Err(msg(format!(
            "{} exists; pass --force to overwrite",
            request.output.display()
        )));
    }
    let resolved = archive::resolve_user_path(&request.source, request.force_single)?;
    let archive = archive::open(&resolved.path, None, DEFAULT_MAX_HEADER, None, None)?;
    let config = read_config(resolved.path.parent().unwrap_or(Path::new(".")))?;
    let mut warnings = Vec::new();
    let tensors = if let Some(kind) = &request.quantize {
        quantize_reference(&archive, kind)?
    } else {
        dequant_archive(&archive, config.as_ref(), &mut warnings)?
    };
    let target = output_format(&request.output, request.format)?;
    write_f32(&request.output, target, &tensors)?;
    Ok(warnings)
}

struct FloatTensor {
    name: String,
    shape: Vec<u64>,
    values: Vec<f32>,
}

fn dequant_archive(
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
    if names.iter().any(|name| name.ends_with(".qweight"))
        && names.iter().any(|name| name.ends_with(".g_idx"))
    {
        return dequant_gptq(archive, config);
    }
    if names.iter().any(|name| name.ends_with(".qweight")) {
        return dequant_awq(archive, config);
    }
    if names
        .iter()
        .any(|name| name.ends_with(".absmax") || name.contains("bitsandbytes__"))
    {
        return dequant_bnb(archive, config, warnings);
    }
    if names.iter().any(|name| name.ends_with(".weight_packed")) {
        return dequant_compressed(archive, config, warnings);
    }
    if mlx_config(config).is_some() {
        return dequant_mlx(archive, config);
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
            if bits == Some(3) {
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
        let logical = dtype::ggml_payload_bytes(tensor.ggml_type, &tensor.shape)?;
        let raw = member.payload.clipped(logical)?.read_vec()?;
        let values = dequant_tensor(tensor.ggml_type, &raw)?;
        let mut shape = tensor.shape.clone();
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

fn dequant_gptq(archive: &Archive, config: Option<&Value>) -> Result<Vec<FloatTensor>> {
    let bits = gptq_bits(config)?;
    if bits == 3 {
        return Err(msg("GPTQ 3-bit unpack is named but not transcribed"));
    }
    if !matches!(bits, 2 | 4 | 8) {
        return Err(msg(format!(
            "GPTQ bits {bits} are not the 2, 4, or 8 cuda pack"
        )));
    }
    let mut groups: BTreeMap<String, Gptq> = BTreeMap::new();
    let mut plain = Vec::new();
    for member in &archive.members {
        let Some((prefix, suffix)) = member.name.rsplit_once('.') else {
            plain.push(member_f32(member)?);
            continue;
        };
        if !matches!(suffix, "qweight" | "qzeros" | "scales" | "g_idx" | "bias") {
            plain.push(member_f32(member)?);
            continue;
        }
        let slot = groups.entry(prefix.to_string()).or_default();
        match suffix {
            "qweight" => slot.qweight = Some(read_i32(member)?),
            "qzeros" => slot.qzeros = Some(read_i32(member)?),
            "scales" => slot.scales = Some(read_f32(member)?),
            "g_idx" => slot.g_idx = Some(read_i32(member)?.0),
            "bias" => slot.bias = Some(read_f32(member)?.0),
            _ => {}
        }
    }
    let mut out = plain;
    for (prefix, group) in groups {
        let qweight = group
            .qweight
            .ok_or_else(|| msg(format!("{prefix}.qweight is missing")))?;
        let qzeros = group
            .qzeros
            .ok_or_else(|| msg(format!("{prefix}.qzeros is missing")))?;
        let scales = group
            .scales
            .ok_or_else(|| msg(format!("{prefix}.scales is missing")))?;
        let g_idx = group
            .g_idx
            .ok_or_else(|| msg(format!("{prefix}.g_idx is missing")))?;
        let values = unpack_gptq(
            &qweight.0, &qweight.1, &qzeros.0, &qzeros.1, &scales.0, &scales.1, &g_idx, bits,
        )?;
        let out_features = qweight.1[1];
        let in_features = g_idx.len() as u64;
        out.push(FloatTensor {
            name: format!("{prefix}.weight"),
            shape: vec![out_features, in_features],
            values,
        });
        if let Some(bias) = group.bias {
            out.push(FloatTensor {
                name: format!("{prefix}.bias"),
                shape: vec![bias.len() as u64],
                values: bias,
            });
        }
    }
    Ok(out)
}

#[derive(Default)]
struct Gptq {
    qweight: Option<(Vec<i32>, Vec<u64>)>,
    qzeros: Option<(Vec<i32>, Vec<u64>)>,
    scales: Option<(Vec<f32>, Vec<u64>)>,
    g_idx: Option<Vec<i32>>,
    bias: Option<Vec<f32>>,
}

pub fn unpack_gptq(
    qweight: &[i32],
    qweight_shape: &[u64],
    qzeros: &[i32],
    qzeros_shape: &[u64],
    scales: &[f32],
    scales_shape: &[u64],
    g_idx: &[i32],
    bits: u32,
) -> Result<Vec<f32>> {
    if qweight_shape.len() != 2 || qzeros_shape.len() != 2 || scales_shape.len() != 2 {
        return Err(msg("GPTQ qweight, qzeros, and scales are matrices"));
    }
    let rows = qweight_shape[0] as usize;
    let out_f = qweight_shape[1] as usize;
    let pack = (32 / bits) as usize;
    let in_f = rows * pack;
    if g_idx.len() != in_f {
        return Err(msg("GPTQ g_idx length does not match in_features"));
    }
    let mut codes = vec![0u32; in_f * out_f];
    for row in 0..rows {
        for col in 0..out_f {
            let word = qweight[row * out_f + col] as u32;
            for j in 0..pack {
                let value = (word >> (bits * j as u32)) & ((1 << bits) - 1);
                codes[(row * pack + j) * out_f + col] = value;
            }
        }
    }
    let zero_rows = qzeros_shape[0] as usize;
    let zero_cols = qzeros_shape[1] as usize;
    if zero_cols * pack != out_f {
        return Err(msg("GPTQ qzeros width is not out_features // 32 * bits"));
    }
    let mut zeros = vec![0u32; zero_rows * out_f];
    for row in 0..zero_rows {
        for col in 0..zero_cols {
            let word = qzeros[row * zero_cols + col] as u32;
            for j in 0..pack {
                let value = (word >> (bits * j as u32)) & ((1 << bits) - 1);
                zeros[row * out_f + col * pack + j] = value + 1;
            }
        }
    }
    let scale_cols = scales_shape[1] as usize;
    let mut out = vec![0f32; out_f * in_f];
    for k in 0..in_f {
        let g = g_idx[k] as usize;
        for n in 0..out_f {
            let code = codes[k * out_f + n] as f32;
            let zero = zeros[g * out_f + n] as f32;
            let scale = scales[g * scale_cols + n];
            out[n * in_f + k] = (code - zero) * scale;
        }
    }
    Ok(out)
}

fn dequant_awq(archive: &Archive, config: Option<&Value>) -> Result<Vec<FloatTensor>> {
    if let Some(version) = quant_object(config)
        .and_then(|object| object.get("version"))
        .and_then(Value::as_str)
    {
        if version != "gemm" {
            return Err(msg(format!("AWQ version {version} is not the gemm pack")));
        }
    }
    let group = awq_group(config)?;
    let mut groups: BTreeMap<String, Awq> = BTreeMap::new();
    let mut plain = Vec::new();
    for member in &archive.members {
        let Some((prefix, suffix)) = member.name.rsplit_once('.') else {
            plain.push(member_f32(member)?);
            continue;
        };
        if !matches!(suffix, "qweight" | "qzeros" | "scales" | "bias") {
            plain.push(member_f32(member)?);
            continue;
        }
        let slot = groups.entry(prefix.to_string()).or_default();
        match suffix {
            "qweight" => slot.qweight = Some((read_i32(member)?, member.shape.clone())),
            "qzeros" => slot.qzeros = Some(read_i32(member)?),
            "scales" => slot.scales = Some(read_f32(member)?),
            "bias" => slot.bias = Some(read_f32(member)?.0),
            _ => {}
        }
    }
    let mut out = plain;
    for (prefix, group_tensors) in groups {
        let (qweight, shape) = group_tensors
            .qweight
            .ok_or_else(|| msg(format!("{prefix}.qweight is missing")))?;
        let qzeros = group_tensors
            .qzeros
            .ok_or_else(|| msg(format!("{prefix}.qzeros is missing")))?;
        let scales = group_tensors
            .scales
            .ok_or_else(|| msg(format!("{prefix}.scales is missing")))?;
        let values = unpack_awq(
            &qweight.0, &shape, &qzeros.0, &qzeros.1, &scales.0, &scales.1, group,
        )?;
        let in_f = shape[0];
        let out_f = shape[1] * 8;
        out.push(FloatTensor {
            name: format!("{prefix}.weight"),
            shape: vec![out_f, in_f],
            values,
        });
        if let Some(bias) = group_tensors.bias {
            out.push(FloatTensor {
                name: format!("{prefix}.bias"),
                shape: vec![bias.len() as u64],
                values: bias,
            });
        }
    }
    Ok(out)
}

#[derive(Default)]
struct Awq {
    qweight: Option<((Vec<i32>, Vec<u64>), Vec<u64>)>,
    qzeros: Option<(Vec<i32>, Vec<u64>)>,
    scales: Option<(Vec<f32>, Vec<u64>)>,
    bias: Option<Vec<f32>>,
}

pub fn unpack_awq(
    qweight: &[i32],
    qweight_shape: &[u64],
    qzeros: &[i32],
    qzeros_shape: &[u64],
    scales: &[f32],
    scales_shape: &[u64],
    group_size: usize,
) -> Result<Vec<f32>> {
    let order = [0usize, 2, 4, 6, 1, 3, 5, 7];
    let in_f = qweight_shape[0] as usize;
    let packed_out = qweight_shape[1] as usize;
    let out_f = packed_out * 8;
    let mut w = vec![0i32; in_f * out_f];
    for row in 0..in_f {
        for col in 0..packed_out {
            let word = qweight[row * packed_out + col] as u32;
            for i in 0..8 {
                w[row * out_f + col * 8 + order[i]] = ((word >> (i * 4)) & 0x0f) as i32;
            }
        }
    }
    let zero_rows = qzeros_shape[0] as usize;
    let zero_packed = qzeros_shape[1] as usize;
    let mut z = vec![0i32; zero_rows * out_f];
    for row in 0..zero_rows {
        for col in 0..zero_packed {
            let word = qzeros[row * zero_packed + col] as u32;
            for i in 0..8 {
                z[row * out_f + col * 8 + order[i]] = ((word >> (i * 4)) & 0x0f) as i32;
            }
        }
    }
    let scale_cols = scales_shape[1] as usize;
    let mut out = vec![0f32; out_f * in_f];
    for k in 0..in_f {
        let g = k / group_size;
        for n in 0..out_f {
            out[n * in_f + k] =
                (w[k * out_f + n] as f32 - z[g * out_f + n] as f32) * scales[g * scale_cols + n];
        }
    }
    Ok(out)
}

fn dequant_bnb(
    archive: &Archive,
    config: Option<&Value>,
    warnings: &mut Vec<String>,
) -> Result<Vec<FloatTensor>> {
    let quant = quant_object(config);
    let kind = quant
        .and_then(|object| object.get("bnb_4bit_quant_type"))
        .and_then(Value::as_str)
        .unwrap_or("nf4");
    let block = quant
        .and_then(|object| object.get("bnb_4bit_quant_blocksize"))
        .and_then(Value::as_u64)
        .unwrap_or(64) as usize;
    if quant
        .and_then(|object| object.get("bnb_4bit_use_double_quant"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        return Err(msg(
            "bitsandbytes nested absmax is refused; the packed quant_state map is not restated",
        ));
    }
    let _ = warnings;
    let table: Vec<f32> = if kind == "fp4" {
        FP4.to_vec()
    } else {
        NF4.to_vec()
    };
    let members = index(archive);
    let mut used = Vec::new();
    let mut out = Vec::new();
    for member in &archive.members {
        if !member.name.ends_with(".weight") && !member.name.contains("bitsandbytes__") {
            continue;
        }
        let prefix = member
            .name
            .trim_end_matches(".weight")
            .trim_end_matches(".bitsandbytes__nf4")
            .trim_end_matches(".bitsandbytes__fp4");
        let absmax_name = format!("{prefix}.absmax");
        let Some(absmax) = members.get(absmax_name.as_str()) else {
            continue;
        };
        let packed = read_u8(member)?;
        let scales = read_f32(absmax)?.0;
        let values = dequant_4bit(&packed, &scales, block, &table, member.shape.clone())?;
        out.push(FloatTensor {
            name: format!("{prefix}.weight"),
            shape: member.shape.clone(),
            values,
        });
        used.push(member.name.clone());
        used.push(absmax_name);
    }
    for member in &archive.members {
        if used.iter().any(|name| name == &member.name) || member.name.ends_with(".absmax") {
            continue;
        }
        if member.name.contains("quant_state") || member.name.contains("nested") {
            continue;
        }
        out.push(member_f32(member)?);
    }
    if out.is_empty() {
        return Err(msg("bitsandbytes dequant found no weight and absmax pair"));
    }
    Ok(out)
}

pub fn dequant_4bit(
    packed: &[u8],
    absmax: &[f32],
    block: usize,
    table: &[f32],
    shape: Vec<u64>,
) -> Result<Vec<f32>> {
    let mut codes = Vec::with_capacity(packed.len() * 2);
    for byte in packed {
        codes.push((byte >> 4) as usize);
        codes.push((byte & 0x0f) as usize);
    }
    let n = numel(&shape).unwrap_or(codes.len() as u64) as usize;
    codes.truncate(n);
    let mut values: Vec<f32> = codes.iter().map(|code| table[*code]).collect();
    for (index, scale) in absmax.iter().enumerate() {
        let start = index * block;
        let end = (start + block).min(values.len());
        if start >= values.len() {
            break;
        }
        for value in &mut values[start..end] {
            *value *= scale;
        }
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

fn dequant_compressed(
    archive: &Archive,
    config: Option<&Value>,
    warnings: &mut Vec<String>,
) -> Result<Vec<FloatTensor>> {
    let (bits, symmetric) = compressed_args(config);
    if !symmetric {
        warnings.push("compressed-tensors symmetric is false; zeros stay beside the codes".into());
    }
    let mut out = Vec::new();
    for member in &archive.members {
        if member.name.ends_with(".weight_packed") {
            let packed = read_i32(member)?;
            let shape = weight_shape(archive, &member.name)?;
            let codes = unpack_pack_quantized(&packed.0, &packed.1, bits, &shape)?;
            let values = codes.into_iter().map(|code| code as f32).collect();
            let name = member.name.trim_end_matches("_packed").to_string();
            out.push(FloatTensor {
                name,
                shape,
                values,
            });
        } else if !member.name.ends_with(".weight_shape") {
            out.push(member_f32(member)?);
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
            shape.push(dim as u64);
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
    let cols = *shape.last().unwrap_or(&0) as usize;
    let rows = numel(&shape[..shape.len().saturating_sub(1)]).unwrap_or(1) as usize;
    let words = packed_shape.last().copied().unwrap_or(0) as usize;
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
    let bits = object
        .get("bits")
        .and_then(Value::as_u64)
        .ok_or_else(|| msg("MLX quantization.bits is missing"))? as u32;
    let group = object
        .get("group_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| msg("MLX quantization.group_size is missing"))? as usize;
    let mode = object
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("affine");
    let members = index(archive);
    let mut consumed = Vec::new();
    let mut out = Vec::new();
    for member in &archive.members {
        if !member.name.ends_with(".weight") {
            continue;
        }
        let prefix = member.name.trim_end_matches(".weight");
        let scales_name = format!("{prefix}.scales");
        let Some(scales_member) = members.get(scales_name.as_str()) else {
            continue;
        };
        let weight = read_u32(member)?;
        let scale_shape = scales_member.shape.clone();
        let values = if mode == "affine" || mode.is_empty() {
            let scales = read_f32(scales_member)?;
            let biases_name = format!("{prefix}.biases");
            let biases = members
                .get(biases_name.as_str())
                .ok_or_else(|| msg(format!("{biases_name} is missing")))?;
            let biases = read_f32(biases)?;
            consumed.push(biases_name);
            dequant_affine(&weight, &member.shape, &scales.0, &biases.0, group, bits)?
        } else {
            let scale_bytes = fp_scale_bytes(scales_member)?;
            let global = members
                .get(format!("{prefix}.global_scale").as_str())
                .map(|item| read_f32(item).map(|values| values.0.first().copied().unwrap_or(1.0)))
                .transpose()?;
            dequant_fp(&weight, &scale_bytes, group, bits, mode, global)?
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
        consumed.push(member.name.clone());
        consumed.push(scales_name);
    }
    for member in &archive.members {
        if consumed.iter().any(|name| name == &member.name) {
            continue;
        }
        if member.name.ends_with(".scales") || member.name.ends_with(".biases") {
            continue;
        }
        out.push(member_f32(member)?);
    }
    Ok(out)
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
    global: Option<f32>,
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
        for byte in &bytes {
            values.push(decode_e4m3(*byte));
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
    let decoded: Vec<f32> = if group == 16 {
        let factor = global.map(|value| value / (448.0 * 6.0)).unwrap_or(1.0);
        scales
            .iter()
            .map(|scale| decode_e4m3(*scale) * factor)
            .collect()
    } else {
        scales
            .iter()
            .map(|scale| 2f32.powf(f32::from(*scale) - 127.0))
            .collect()
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

fn decode_e4m3(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0x0f;
    let man = byte & 0x07;
    let mag = if exp == 0 {
        (man as f32) * 2f32.powi(-9)
    } else if exp == 15 {
        return f32::NAN;
    } else {
        (1.0 + (man as f32) / 8.0) * 2f32.powi(i32::from(exp) - 7)
    };
    if sign == 1 {
        -mag
    } else {
        mag
    }
}

fn unpack_mlx(words: &[u32], shape: &[u64], bits: u32) -> Result<Vec<u32>> {
    if matches!(bits, 3 | 5 | 6) {
        return unpack_bitstream(words, shape, bits);
    }
    unpack_pow2(words, shape, bits)
}

fn unpack_bitstream(words: &[u32], shape: &[u64], bits: u32) -> Result<Vec<u32>> {
    let cols = shape.last().copied().unwrap_or(0) as usize;
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
    if bits == 0 || 32 % bits != 0 {
        return Err(msg(format!("MLX bits {bits} are not a power-of-two pack")));
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

fn quantize_reference(_archive: &Archive, _kind: &str) -> Result<Vec<FloatTensor>> {
    Err(msg(
        "reference quantize is written as a GGUF file from float tensors",
    ))
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
    archive.members.iter().map(member_f32).collect()
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

fn gptq_bits(config: Option<&Value>) -> Result<u32> {
    quant_object(config)
        .and_then(|object| object.get("bits").or_else(|| object.get("w_bit")))
        .and_then(Value::as_u64)
        .map(|bits| bits as u32)
        .ok_or_else(|| msg("GPTQ dequant needs quantization_config.bits"))
}

fn awq_group(config: Option<&Value>) -> Result<usize> {
    quant_object(config)
        .and_then(|object| {
            object
                .get("group_size")
                .or_else(|| object.get("q_group_size"))
        })
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| msg("AWQ dequant needs quantization_config.group_size"))
}

fn compressed_args(config: Option<&Value>) -> (u32, bool) {
    let mut bits = 8u32;
    let mut symmetric = true;
    let Some(config) = config else {
        return (bits, symmetric);
    };
    let Some(groups) = config
        .pointer("/quantization_config/config_groups")
        .and_then(Value::as_object)
    else {
        return (bits, symmetric);
    };
    for group in groups.values() {
        let Some(weights) = group.get("weights").and_then(Value::as_object) else {
            continue;
        };
        if let Some(value) = weights.get("num_bits").and_then(Value::as_u64) {
            bits = value as u32;
        }
        if let Some(value) = weights.get("symmetric").and_then(Value::as_bool) {
            symmetric = value;
        }
        break;
    }
    (bits, symmetric)
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
            println!("tensquant {}", env!("CARGO_PKG_VERSION"));
            Ok(0)
        }
        QuantCommand::Run(request) => {
            if request.quantize.is_some() {
                quantize_file(&request)?;
            } else {
                for warning in run(&request)? {
                    eprintln!("tensquant: {warning}");
                }
            }
            Ok(0)
        }
    }
}

fn quantize_file(request: &Request) -> Result<()> {
    let kind = request.quantize.as_deref().unwrap_or("");
    if !matches!(kind, "q4_0" | "q8_0") {
        return Err(msg("--quantize accepts q4_0 or q8_0"));
    }
    if request.output.exists() && !request.force {
        return Err(msg(format!(
            "{} exists; pass --force to overwrite",
            request.output.display()
        )));
    }
    let resolved = archive::resolve_user_path(&request.source, request.force_single)?;
    let archive = archive::open(&resolved.path, None, DEFAULT_MAX_HEADER, None, None)?;
    let mut dense = Vec::new();
    for member in &archive.members {
        let tensor = member_f32(member)?;
        let rows = if tensor.shape.is_empty() {
            return Err(msg(format!("{} needs a shape to quantize", tensor.name)));
        } else {
            let mut ne = tensor.shape.clone();
            ne.reverse();
            ne
        };
        if rows[0] % 32 != 0 {
            return Err(msg(format!(
                "{} innermost size {} is not a multiple of 32",
                tensor.name, rows[0]
            )));
        }
        let packed = if kind == "q4_0" {
            ggml_dequant::quantize_q4_0(&tensor.values)?
        } else {
            ggml_dequant::quantize_q8_0(&tensor.values)?
        };
        dense.push(DenseTensor {
            name: tensor.name,
            ggml_type: if kind == "q4_0" { 2 } else { 8 },
            ne: rows,
            payload: Payload::Bytes(packed),
        });
    }
    let metadata = vec![("general.quantization_version".into(), MetaValue::U32(2))];
    write_dense(&request.output, &metadata, &dense)
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
        let qweight = [word as i32; 8];
        let qzeros = [0i32; 1];
        let scales = [2.0f32; 8];
        let g_idx = vec![0; 8];
        let values = unpack_gptq(
            &qweight,
            &[1, 8],
            &qzeros,
            &[1, 1],
            &scales,
            &[1, 8],
            &g_idx,
            bits,
        )
        .unwrap();
        assert_eq!(values.len(), 64);
        assert!((values[0] - (1.0 - 1.0) * 2.0).abs() < 1e-6);
        assert!((values[1] - (2.0 - 1.0) * 2.0).abs() < 1e-6);
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
