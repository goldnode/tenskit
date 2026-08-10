//! Container conversion. Adapters stay in the format modules. Renames, the
//! LLaMA row permutation, and the Qwen3.5 value changes are explicit passes.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::archive::{self, Archive, Inner};
use crate::cli::{FormatName, DEFAULT_MAX_HEADER};
use crate::dtype::{self, element_ggml_code, numel};
use crate::error::{msg, Result};
use crate::formats::gguf::{write_dense, DenseTensor, MetaValue};
use crate::gguf_names::{self, ConverterRecord};
use crate::payload::Payload;

pub struct Request {
    pub source: PathBuf,
    pub output: PathBuf,
    pub format: Option<FormatName>,
    pub rename: bool,
    pub names_only: bool,
    pub force_single: bool,
    pub force: bool,
}

struct Tensor {
    name: String,
    canonical: String,
    shape: Vec<u64>,
    bytes: Payload,
}

pub fn run(request: &Request) -> Result<Vec<String>> {
    if request.names_only && !request.rename {
        return Err(msg("--names-only applies to --rename"));
    }
    crate::io_util::refuse_existing(&request.output, request.force)?;
    let resolved = archive::resolve_user_path(&request.source, request.force_single)?;
    let archive = archive::open(&resolved.path, None, DEFAULT_MAX_HEADER, None, None)?;
    let target = output_format(&request.output, request.format)?;
    let config_dir = resolved.path.parent().unwrap_or(Path::new("."));
    let config = read_config(config_dir)?;
    let hparams = config.as_ref().map(merge_text_config);
    let mut warnings = Vec::new();
    let mut tensors = load_tensors(&archive)?;
    if request.rename {
        let arch = choose_architecture(&archive, config.as_ref())?;
        let class = config.as_ref().and_then(hf_class);
        let converter = select_converter(&arch, class.as_deref())?;
        let layers = language_layers(hparams.as_ref());
        if !request.names_only {
            apply_values(
                &mut tensors,
                &arch,
                &converter,
                hparams.as_ref(),
                config_dir,
                &mut warnings,
            )?;
        }
        rename_tensors(&mut tensors, &arch, layers, &mut warnings)?;
        if target == FormatName::Gguf {
            if let Some(map) = &hparams {
                warnings.extend(metadata_notes(map));
            }
        }
    }
    for tensor in &mut tensors {
        fit_dtype(tensor, target, &mut warnings)?;
    }
    write_target(
        &request.output,
        target,
        &tensors,
        &archive,
        hparams.as_ref(),
        request.rename,
        &mut warnings,
    )?;
    Ok(warnings)
}

fn output_format(path: &Path, explicit: Option<FormatName>) -> Result<FormatName> {
    if let Some(format) = explicit {
        return Ok(format);
    }
    let name = path
        .file_name()
        .and_then(|text| text.to_str())
        .unwrap_or("");
    let lower = name.to_ascii_lowercase();
    let parsed = if lower.ends_with(".safetensors") {
        Some("safetensors")
    } else if lower.ends_with(".gguf") {
        Some("gguf")
    } else if lower.ends_with(".npy") {
        Some("npy")
    } else if lower.ends_with(".npz") {
        Some("npz")
    } else if lower.ends_with(".pt") || lower.ends_with(".pth") {
        Some("pytorch")
    } else if lower.ends_with(".onnx") {
        Some("onnx")
    } else if lower.ends_with(".rawtensor.json") {
        Some("raw")
    } else {
        None
    };
    parsed
        .map(FormatName::parse)
        .transpose()?
        .ok_or_else(|| msg("pass --format; the output name does not say which container"))
}

fn load_tensors(archive: &Archive) -> Result<Vec<Tensor>> {
    let mut out = Vec::new();
    for member in &archive.members {
        if member.fortran {
            return Err(msg(format!(
                "tensor {} is fortran-order; tensconv does not transpose it",
                member.name
            )));
        }
        let canonical = dtype::canonical_id(&member.dtype)?;
        if dtype::canon_kind(canonical) == Some("block") {
            return Err(msg(format!(
                "tensor {} is quantized; dequantize it with tensquant",
                member.name
            )));
        }
        let mut shape = member.shape.clone();
        #[cfg(format_gguf)]
        if matches!(archive.inner, Inner::Gguf(_)) {
            shape.reverse();
        }
        out.push(Tensor {
            name: member.name.clone(),
            canonical: canonical.to_string(),
            shape,
            bytes: member.payload.clone(),
        });
    }
    Ok(out)
}

fn choose_architecture(archive: &Archive, config: Option<&Value>) -> Result<String> {
    if let Some(arch) = archive
        .metadata
        .iter()
        .find(|(key, _)| key == "general.architecture")
        .map(|(_, value)| value.clone())
        .filter(|value| !value.is_empty())
    {
        return Ok(arch);
    }
    let class = config.and_then(hf_class).ok_or_else(|| {
        msg("--rename needs general.architecture or config.json architectures[0]")
    })?;
    gguf_names::architecture_for_hf_class(&class).ok_or_else(|| {
        msg(format!(
            "config.json architectures {class} is not in the role table"
        ))
    })
}

/// The converter's HF class: `architectures[0]`, or
/// `text_config.architectures[0]` when present, except for the top-level
/// names llama.cpp keeps as they are (conversion-considerations.md section 6,
/// "Choosing the converter class").
pub(crate) fn hf_class(config: &Value) -> Option<String> {
    const KEEP_TOP_LEVEL: &[&str] = &[
        "NEOChatModel",
        "StepVLForConditionalGeneration",
        "Sarashina2VisionForCausalLM",
        "Exaone4_5_ForConditionalGeneration",
        "Step3p7ForConditionalGeneration",
        "KimiK3ForConditionalGeneration",
    ];
    let first = |value: &Value| {
        value
            .get("architectures")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .and_then(Value::as_str)
            .map(str::to_string)
    };
    let top = first(config);
    if top.as_deref().is_some_and(|name| KEEP_TOP_LEVEL.contains(&name)) {
        return top;
    }
    config.get("text_config").and_then(first).or(top)
}

fn select_converter(arch: &str, class: Option<&str>) -> Result<ConverterRecord> {
    let records = gguf_names::converter_records(arch);
    if records.is_empty() {
        return Err(msg(format!("architecture {arch} has no converter record")));
    }
    if let Some(class) = class {
        if let Some(found) = records
            .iter()
            .find(|record| record.hf_architectures.iter().any(|item| item == class))
        {
            return Ok(found.clone());
        }
    }
    // There is no preference order between converter classes: without the
    // HF class from config.json, only an architecture with one converter is
    // decided (conversion-considerations.md section 6).
    if records.len() == 1 {
        return Ok(records.into_iter().next().unwrap());
    }
    let names: Vec<_> = records.iter().map(|record| record.class.as_str()).collect();
    Err(msg(format!(
        "architecture {arch} has more than one converter ({}); put config.json with its architectures beside the weights",
        names.join(", ")
    )))
}

fn apply_values(
    tensors: &mut [Tensor],
    arch: &str,
    converter: &ConverterRecord,
    hparams: Option<&Map<String, Value>>,
    dir: &Path,
    warnings: &mut Vec<String>,
) -> Result<()> {
    if converter.complete {
        for tensor in tensors.iter_mut() {
            apply_branches(tensor, arch, converter, hparams)?;
        }
        return Ok(());
    }
    if converter.undo_permute {
        llama_permute(tensors, hparams)?;
        warnings.push(format!(
            "{}: LLaMA Q/K rows were permuted; other unexpanded arms were left alone",
            converter.class
        ));
        return Ok(());
    }
    if arch == "qwen35" {
        qwen35_values(tensors, hparams, dir)?;
        return Ok(());
    }
    let pending = if converter.unexpanded.is_empty() {
        "the method was not expanded".to_string()
    } else {
        converter.unexpanded.join("; ")
    };
    Err(msg(format!(
        "value changes for {arch} ({}) are not in the spec: {pending}",
        converter.class
    )))
}

fn llama_permute(tensors: &mut [Tensor], hparams: Option<&Map<String, Value>>) -> Result<()> {
    let n_head = hparam_u64(hparams, &["num_attention_heads", "n_head", "n_heads"])
        .ok_or_else(|| msg("LLaMA permute needs num_attention_heads"))?;
    let n_kv = hparam_u64(hparams, &["num_key_value_heads", "n_kv_heads"]).unwrap_or(n_head);
    for tensor in tensors {
        // Exactly these suffixes (section 6); packed GPTQ/AWQ siblings such
        // as q_proj.qweight are not weight rows and are never permuted.
        let name = tensor.name.as_str();
        let heads = if name.ends_with("k_proj.weight") || name.ends_with("k_proj.bias") {
            n_kv
        } else if name.ends_with("q_proj.weight") || name.ends_with("q_proj.bias") {
            n_head
        } else {
            continue;
        };
        let data = tensor.bytes.read_vec()?;
        let width = elem_width(&tensor.canonical)?;
        let permuted = permute_rows(&data, &tensor.shape, width, heads)?;
        tensor.bytes = Payload::Bytes(permuted);
    }
    Ok(())
}

fn permute_rows(data: &[u8], shape: &[u64], width: usize, n_head: u64) -> Result<Vec<u8>> {
    if shape.is_empty() || n_head == 0 {
        return Err(msg("LLaMA permute needs a rank and a head count"));
    }
    let rows = shape[0];
    if rows % (n_head * 2) != 0 {
        return Err(msg(format!(
            "LLaMA permute needs {rows} rows to divide into {n_head} heads"
        )));
    }
    let row_elems = numel(&shape[1..]).unwrap_or(1);
    let row_bytes = (row_elems as usize)
        .checked_mul(width)
        .ok_or_else(|| msg("row is too large"))?;
    if data.len() != rows as usize * row_bytes {
        return Err(msg("LLaMA permute payload does not match the shape"));
    }
    let half = (rows / n_head / 2) as usize;
    let heads = n_head as usize;
    let mut out = vec![0u8; data.len()];
    for head in 0..heads {
        for index in 0..half {
            for pair in 0..2 {
                let source = ((head * 2 + pair) * half + index) * row_bytes;
                let dest = ((head * half + index) * 2 + pair) * row_bytes;
                out[dest..dest + row_bytes].copy_from_slice(&data[source..source + row_bytes]);
            }
        }
    }
    Ok(out)
}

fn qwen35_values(
    tensors: &mut [Tensor],
    hparams: Option<&Map<String, Value>>,
    dir: &Path,
) -> Result<()> {
    let num_k = hparam_u64(hparams, &["linear_num_key_heads"]).unwrap_or(0);
    let num_v = hparam_u64(hparams, &["linear_num_value_heads"]).unwrap_or(0);
    let key_dim = hparam_u64(hparams, &["linear_key_head_dim"]).unwrap_or(0);
    let value_dim = hparam_u64(hparams, &["linear_value_head_dim"]).unwrap_or(0);
    let reorder = num_k > 0 && num_v > 0 && num_k != num_v;
    let manifest = hadamard_names(dir);
    if reorder && num_v % num_k != 0 {
        return Err(msg(
            "linear_num_value_heads is not a multiple of linear_num_key_heads",
        ));
    }
    let per_k = if num_k == 0 { 0 } else { num_v / num_k };
    for tensor in tensors.iter_mut() {
        if reorder && tensor.name.contains("linear_attn.") {
            qwen_reorder(tensor, num_k, per_k, key_dim, value_dim, &manifest)?;
        }
        if tensor.name.contains("linear_attn.") && tensor.name.ends_with(".A_log") {
            neg_exp(tensor)?;
        }
        if norm_plus_one(&tensor.name) {
            add_scalar(tensor, 1.0)?;
        }
        if tensor.name.contains("linear_attn.conv1d") {
            squeeze_all(tensor)?;
        }
    }
    Ok(())
}

fn norm_plus_one(name: &str) -> bool {
    if name.contains("linear_attn.norm.weight") {
        return false;
    }
    name.ends_with("norm.weight")
        || name.ends_with("pre_fc_norm_embedding.weight")
        || name.ends_with("pre_fc_norm_hidden.weight")
}

fn qwen_reorder(
    tensor: &mut Tensor,
    num_k: u64,
    per_k: u64,
    key_dim: u64,
    value_dim: u64,
    manifest: &[String],
) -> Result<()> {
    let name = tensor.name.clone();
    if name.contains(".out_proj.") && hadamard_folds(&name, manifest) {
        return Ok(());
    }
    if name.contains(".conv1d") {
        squeeze_all(tensor)?;
    }
    let data = tensor.bytes.read_vec()?;
    let width = elem_width(&tensor.canonical)?;
    let (axis, start, head_dim) = qwen_axis(&name, &tensor.shape, key_dim, num_k, value_dim)?;
    let idx = v_head_perm(num_k as usize, per_k as usize, head_dim as usize);
    let taken = reorder_slice(&data, &tensor.shape, width, axis, start as usize, &idx)?;
    tensor.bytes = Payload::Bytes(taken);
    Ok(())
}

fn qwen_axis(
    name: &str,
    shape: &[u64],
    key_dim: u64,
    num_k: u64,
    value_dim: u64,
) -> Result<(usize, u64, u64)> {
    if name.contains(".in_proj_qkv.") {
        Ok((0, key_dim * num_k * 2, value_dim))
    } else if name.contains(".in_proj_z.") {
        Ok((0, 0, value_dim))
    } else if name.contains(".in_proj_a.") || name.contains(".in_proj_b.") {
        Ok((0, 0, 1))
    } else if name.contains(".A_log") || name.contains(".dt_bias") || name.contains(".dt_proj") {
        let axis = if shape.len() <= 1 { 0 } else { shape.len() - 1 };
        Ok((axis, 0, 1))
    } else if name.contains(".conv1d") {
        Ok((0, key_dim * num_k * 2, value_dim))
    } else if name.contains(".out_proj.") {
        Ok((1, 0, value_dim))
    } else {
        Err(msg(format!(
            "Qwen3.5 V-head reorder has no axis for {name}"
        )))
    }
}

fn hadamard_names(dir: &Path) -> Vec<String> {
    let path = dir.join("hadamard_packing.json");
    let Ok(bytes) = fs::read(path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return Vec::new();
    };
    value
        .get("tensors")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("name").and_then(Value::as_str).map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn hadamard_folds(name: &str, manifest: &[String]) -> bool {
    manifest
        .iter()
        .any(|entry| name.ends_with(entry.as_str()) || entry.ends_with(name))
}

pub fn v_head_perm(num_k_heads: usize, num_v_per_k: usize, head_dim: usize) -> Vec<usize> {
    let mut idx = Vec::with_capacity(num_k_heads * num_v_per_k * head_dim);
    for head_v in 0..num_v_per_k {
        for head_k in 0..num_k_heads {
            for dim in 0..head_dim {
                idx.push(head_k * (num_v_per_k * head_dim) + head_v * head_dim + dim);
            }
        }
    }
    idx
}

fn reorder_slice(
    data: &[u8],
    shape: &[u64],
    width: usize,
    axis: usize,
    start: usize,
    idx: &[usize],
) -> Result<Vec<u8>> {
    if axis >= shape.len() {
        return Err(msg("V-head reorder axis is outside the shape"));
    }
    let axis_len = shape[axis] as usize;
    if start + idx.len() > axis_len {
        return Err(msg("V-head reorder runs past the axis"));
    }
    let before = numel(&shape[..axis]).unwrap_or(1) as usize;
    let after = numel(&shape[axis + 1..]).unwrap_or(1) as usize;
    let line = after * width;
    let mut out = vec![0u8; data.len()];
    for outer in 0..before {
        for pos in 0..axis_len {
            let source_pos = if pos >= start && pos < start + idx.len() {
                start + idx[pos - start]
            } else {
                pos
            };
            let dest = (outer * axis_len + pos) * line;
            let source = (outer * axis_len + source_pos) * line;
            out[dest..dest + line].copy_from_slice(&data[source..source + line]);
        }
    }
    Ok(out)
}

fn apply_branches(
    tensor: &mut Tensor,
    arch: &str,
    converter: &ConverterRecord,
    hparams: Option<&Map<String, Value>>,
) -> Result<()> {
    let class = hf_arch_from_hparams(hparams);
    for branch in &converter.branches {
        if !when_matches(&branch.when, &tensor.name, class.as_deref())? {
            continue;
        }
        apply_expr(tensor, arch, &branch.assigns, &branch.expr)?;
    }
    Ok(())
}

fn hf_arch_from_hparams(hparams: Option<&Map<String, Value>>) -> Option<String> {
    hparams
        .and_then(|map| map.get("architectures"))
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn when_matches(when: &str, name: &str, hf_arch: Option<&str>) -> Result<bool> {
    let mut hit = false;
    for clause in split_top_or(when) {
        let clause = clause.trim();
        if clause_matches(clause, name, hf_arch)? {
            hit = true;
        }
    }
    Ok(hit)
}

fn split_top_or(when: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut quotes = false;
    let bytes = when.as_bytes();
    let mut index = 0;
    while index + 4 <= bytes.len() {
        if bytes[index] == b'\'' {
            quotes = !quotes;
        }
        if !quotes && when[index..].starts_with(" or ") {
            parts.push(&when[start..index]);
            index += 4;
            start = index;
            continue;
        }
        index += 1;
    }
    parts.push(&when[start..]);
    parts
}

fn clause_matches(clause: &str, name: &str, hf_arch: Option<&str>) -> Result<bool> {
    if let Some(rest) = clause.strip_prefix("name.endswith(") {
        let inner = rest
            .strip_suffix(')')
            .ok_or_else(|| msg(format!("unrecognized condition {clause}")))?;
        return Ok(endswith_any(name, inner));
    }
    if let Some(rest) = clause.strip_prefix("name.startswith(") {
        let literal = quoted(rest.trim().trim_end_matches(')'))?;
        return Ok(name.starts_with(&literal));
    }
    if let Some(literal) = clause.strip_suffix(" in name") {
        let literal = quoted(literal.trim())?;
        return Ok(name.contains(&literal));
    }
    if let Some(rest) = clause.strip_prefix("self.hf_arch ==") {
        let literal = quoted(rest.trim())?;
        return Ok(hf_arch == Some(literal.as_str()));
    }
    Err(msg(format!("unrecognized condition {clause}")))
}

fn endswith_any(name: &str, inner: &str) -> bool {
    let inner = inner.trim();
    if let Some(body) = inner
        .strip_prefix('(')
        .and_then(|text| text.strip_suffix(')'))
    {
        body.split(',').any(|part| {
            quoted(part.trim())
                .ok()
                .is_some_and(|literal| name.ends_with(&literal))
        })
    } else {
        quoted(inner)
            .ok()
            .is_some_and(|literal| name.ends_with(&literal))
    }
}

fn quoted(text: &str) -> Result<String> {
    let text = text.trim().trim_end_matches(')');
    let text = text
        .strip_prefix('\'')
        .and_then(|text| text.strip_suffix('\''))
        .ok_or_else(|| msg(format!("unrecognized condition {text}")))?;
    Ok(text.to_string())
}

fn apply_expr(tensor: &mut Tensor, arch: &str, assigns: &str, expr: &str) -> Result<()> {
    match (assigns, expr) {
        ("data_torch", "data_torch + 1" | "data_torch + 1.0") => add_scalar(tensor, 1.0),
        ("data_torch", "data_torch + 1.0 / 5") => add_scalar(tensor, 0.2),
        ("data_torch", "data_torch + 1.0 / 5 ** 1.5") => add_scalar(tensor, 1.0 / 5f64.powf(1.5)),
        ("data_torch", "-torch.exp(data_torch)") => neg_exp(tensor),
        ("data_torch", "data_torch.unsqueeze(-1)") => unsqueeze(tensor, -1),
        ("data_torch", "data_torch.unsqueeze(0)") => unsqueeze(tensor, 0),
        ("data_torch", "torch.squeeze(data_torch)" | "data_torch.squeeze().contiguous()") => {
            squeeze_all(tensor)
        }
        ("data_torch", "data_torch.squeeze(1)") => squeeze_axis(tensor, 1),
        ("data_torch", "data_torch[1:]") => drop_first(tensor),
        ("data_torch", "data_torch.view(data_torch.shape[0], 16, 16, 3).permute(0, 3, 1, 2)") => {
            clip_permute(tensor)
        }
        ("name", "f'model.{name}'") => {
            tensor.name = format!("model.{}", tensor.name);
            Ok(())
        }
        ("name", "name.replace('mm_projector.linear_', 'mm_projector.proj.linear_', 1)") => {
            tensor.name =
                tensor
                    .name
                    .replacen("mm_projector.linear_", "mm_projector.proj.linear_", 1);
            Ok(())
        }
        ("name", expr) if expr.contains("rpartition") => rpartition_name(tensor, expr),
        ("name", "'backbone.' + name[len('model.'):]") => {
            let rest = tensor
                .name
                .strip_prefix("model.")
                .ok_or_else(|| msg(format!("{} does not start with model.", tensor.name)))?;
            tensor.name = format!("backbone.{rest}");
            Ok(())
        }
        ("name", expr) if expr.contains("e_score_correction.bias") => {
            let rest = tensor
                .name
                .strip_suffix("e_score_correction_bias")
                .ok_or_else(|| msg(format!("{} has no e_score_correction_bias", tensor.name)))?;
            tensor.name = format!("{rest}e_score_correction.bias");
            Ok(())
        }
        ("name", expr) if expr.contains("format_tensor_name") => {
            format_role_name(tensor, arch, expr)
        }
        _ => Err(msg(format!("unrecognized value expression {expr}"))),
    }
}

fn rpartition_name(tensor: &mut Tensor, expr: &str) -> Result<()> {
    let (needle, replacement) = if expr.contains(".dt_bias") {
        (".dt_bias", ".dt_proj.bias")
    } else if expr.contains(".dt_norm_weight") {
        (".dt_norm_weight", ".dt_norm.weight")
    } else if expr.contains(".B_norm_weight") {
        (".B_norm_weight", ".B_norm.weight")
    } else if expr.contains(".C_norm_weight") {
        (".C_norm_weight", ".C_norm.weight")
    } else if expr.contains(".k_weight") {
        (".k_weight", ".k.weight")
    } else if expr.contains(".q_weight") {
        (".q_weight", ".q.weight")
    } else {
        return Err(msg(format!("unrecognized value expression {expr}")));
    };
    let Some(head) = tensor.name.strip_suffix(needle) else {
        return Ok(());
    };
    tensor.name = format!("{head}{replacement}");
    Ok(())
}

fn format_role_name(tensor: &mut Tensor, arch: &str, expr: &str) -> Result<()> {
    let role = if expr.contains("ATTN_POST_NORM") {
        "ATTN_POST_NORM"
    } else if expr.contains("FFN_NORM") {
        "FFN_NORM"
    } else {
        return Err(msg(format!("unrecognized value expression {expr}")));
    };
    let pattern = gguf_names::role_pattern(arch, role)
        .ok_or_else(|| msg(format!("{arch} has no {role} role")))?;
    let bid = tensor
        .name
        .split('.')
        .rev()
        .find(|part| part.bytes().all(|byte| byte.is_ascii_digit()) && !part.is_empty())
        .ok_or_else(|| msg(format!("{} has no block index", tensor.name)))?;
    tensor.name = pattern.replace("{bid}", bid);
    Ok(())
}

fn rename_tensors(
    tensors: &mut [Tensor],
    arch: &str,
    layers: Option<u64>,
    warnings: &mut Vec<String>,
) -> Result<()> {
    for tensor in tensors {
        match gguf_names::map_tensor_name(arch, &tensor.name, layers) {
            Some(mapped) => {
                if mapped != tensor.name {
                    warnings.push(format!("renamed {} -> {mapped}", tensor.name));
                    tensor.name = mapped;
                }
            }
            None => warnings.push(format!(
                "{} is not in the {arch} role table and keeps its name",
                tensor.name
            )),
        }
    }
    Ok(())
}

fn fit_dtype(tensor: &mut Tensor, target: FormatName, warnings: &mut Vec<String>) -> Result<()> {
    if target_accepts(target, &tensor.canonical)? {
        return Ok(());
    }
    let widened = widen_to(tensor, target)?;
    warnings.push(format!(
        "widened {} from {} to {widened}",
        tensor.name, tensor.canonical
    ));
    tensor.canonical = widened;
    Ok(())
}

fn target_accepts(target: FormatName, canonical: &str) -> Result<bool> {
    if dtype::canon_kind(canonical) == Some("block") {
        return Ok(false);
    }
    Ok(match target {
        #[cfg(format_safetensors)]
        FormatName::Safetensors => dtype::safe_dtype(canonical).is_ok(),
        #[cfg(format_npy)]
        FormatName::Npy => dtype::safe_dtype(canonical)
            .ok()
            .is_some_and(|dtype| dtype.numpy.is_some()),
        #[cfg(format_npz)]
        FormatName::Npz => dtype::safe_dtype(canonical)
            .ok()
            .is_some_and(|dtype| dtype.numpy.is_some()),
        #[cfg(format_gguf)]
        FormatName::Gguf => element_ggml_code(canonical).is_some(),
        #[cfg(format_pytorch)]
        FormatName::Pytorch => matches!(
            canonical,
            "float32"
                | "float64"
                | "float16"
                | "bfloat16"
                | "int64"
                | "int32"
                | "int16"
                | "int8"
                | "uint8"
                | "bool"
                | "complex64"
                | "complex128"
        ),
        #[cfg(format_onnx)]
        FormatName::Onnx => matches!(
            canonical,
            "float32"
                | "float16"
                | "float64"
                | "bfloat16"
                | "int8"
                | "int16"
                | "int32"
                | "int64"
                | "uint8"
                | "uint16"
                | "uint32"
                | "uint64"
                | "bool"
                | "complex64"
                | "complex128"
                | "float8_e4m3fn"
                | "float8_e4m3fnuz"
                | "float8_e5m2"
                | "float8_e5m2fnuz"
                | "float8_e8m0fnu"
                | "float4_e2m1fn"
        ),
        #[cfg(format_raw_sidecar)]
        FormatName::Raw => true,
    })
}

fn widen_to(tensor: &mut Tensor, target: FormatName) -> Result<String> {
    let next = match tensor.canonical.as_str() {
        "bfloat16" => "float32",
        "float8_e4m3fn" | "float8_e5m2" | "float8_e4m3fnuz" | "float8_e5m2fnuz"
        | "float4_e2m1fn" => "float16",
        "float8_e8m0fnu" => "float32",
        other => {
            return Err(msg(format!(
                "dtype {other} has no lossless widen for {}",
                target.as_str()
            )))
        }
    };
    if !target_accepts(target, next)? {
        return Err(msg(format!(
            "dtype {} cannot be stored in {}",
            tensor.canonical,
            target.as_str()
        )));
    }
    let raw = tensor.bytes.read_vec()?;
    let widened = match tensor.canonical.as_str() {
        "bfloat16" => bf16_to_f32(&raw)?,
        other => minifloat_widen(&raw, other, next)?,
    };
    tensor.bytes = Payload::Bytes(widened);
    Ok(next.to_string())
}

fn bf16_to_f32(raw: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(raw.len() * 2);
    for chunk in raw.chunks_exact(2) {
        let bits = u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(out)
}

/// Widen fp8, fp4, or E8M0 codes to float16 or float32. Every value is
/// exact in the target (conversion-considerations.md section 2). fp4 holds
/// two codes per byte, low nibble first.
fn minifloat_widen(raw: &[u8], dtype: &str, target: &str) -> Result<Vec<u8>> {
    let format = dtype::MiniFloat::for_dtype(dtype)?;
    let codes: Vec<u8> = if dtype::canon_bits(dtype) == Some(4) {
        raw.iter().flat_map(|byte| [byte & 0x0f, byte >> 4]).collect()
    } else {
        raw.to_vec()
    };
    let mut out = Vec::with_capacity(codes.len() * 4);
    for code in codes {
        let value = format.decode(code);
        if target == "float32" {
            out.extend_from_slice(&value.to_le_bytes());
        } else {
            out.extend_from_slice(&crate::ggml_dequant::f32_to_f16(value).to_le_bytes());
        }
    }
    Ok(out)
}

fn write_target(
    path: &Path,
    target: FormatName,
    tensors: &[Tensor],
    archive: &Archive,
    hparams: Option<&Map<String, Value>>,
    rename: bool,
    warnings: &mut Vec<String>,
) -> Result<()> {
    if tensors.is_empty() {
        return Err(msg("the source has no tensors"));
    }
    let source = source_metadata(archive);
    // Targets without metadata say what they leave behind (metadata-mapping
    // generic_rules: never drop metadata silently).
    let dropped = |warnings: &mut Vec<String>, target: &str| {
        if !source.is_empty() {
            warnings.push(format!(
                "{} metadata keys were not written: {target} has no metadata",
                source.len()
            ));
        }
    };
    match target {
        #[cfg(format_safetensors)]
        FormatName::Safetensors => {
            let converted = safe_out(tensors)?;
            let metadata = string_metadata(archive, &source, warnings);
            crate::formats::safetensors::write_file(path, &metadata, &converted)
        }
        #[cfg(format_npy)]
        FormatName::Npy => {
            if tensors.len() != 1 {
                return Err(msg("an npy file holds one tensor"));
            }
            dropped(warnings, "npy");
            let tensor = &tensors[0];
            let descr = numpy_descr(&tensor.canonical)?;
            crate::formats::npy::write_file(path, descr, &tensor.shape, |file| {
                tensor.bytes.copy_to(file)
            })
        }
        #[cfg(format_npz)]
        FormatName::Npz => {
            dropped(warnings, "npz");
            let members = npz_out(tensors)?;
            crate::formats::npz::write_archive(path, &members)
        }
        #[cfg(format_gguf)]
        FormatName::Gguf => {
            let (mut metadata, dense) = gguf_out(tensors, hparams, rename)?;
            for (key, value) in gguf_metadata(archive, source, warnings) {
                // Keys the writer computes win (CONV-007).
                match metadata.iter().find(|(existing, _)| *existing == key) {
                    Some((_, computed)) if *computed != value => warnings.push(format!(
                        "{key} from the source was {} and is written as {} (CONV-007)",
                        value.text(),
                        computed.text()
                    )),
                    Some(_) => {}
                    None => metadata.push((key, value)),
                }
            }
            write_dense(path, &metadata, &dense)
        }
        #[cfg(format_pytorch)]
        FormatName::Pytorch => {
            dropped(warnings, "the PyTorch writer");
            let converted = pytorch_out(tensors)?;
            crate::formats::pytorch::write_archive(path, &converted)
        }
        #[cfg(format_onnx)]
        FormatName::Onnx => {
            let converted = onnx_out(tensors)?;
            let metadata = string_metadata(archive, &source, warnings);
            crate::formats::onnx::write_archive(path, &metadata, &converted)
        }
        #[cfg(format_raw_sidecar)]
        FormatName::Raw => {
            let converted = raw_out(tensors)?;
            // raw-sidecar keeps every key with its type (metadata: "Converters
            // SHOULD copy source metadata keys verbatim").
            let meta: Map<String, Value> = match &archive.inner {
                Inner::Raw(file) => file.metadata.clone(),
                _ => raw_metadata(&source, warnings),
            };
            crate::formats::rawtensor::write_archive(path, &meta, &converted)
        }
    }
}

/// Source metadata with types: GGUF keys keep theirs, other formats are
/// string maps. `general.alignment` describes the source file's layout and
/// is not carried over.
fn source_metadata(archive: &Archive) -> Vec<(String, MetaValue)> {
    #[cfg(format_gguf)]
    if let Inner::Gguf(file) = &archive.inner {
        return file
            .metadata
            .iter()
            .filter(|field| field.key != "general.alignment")
            .map(|field| (field.key.clone(), field.value.clone()))
            .collect();
    }
    #[cfg(format_raw_sidecar)]
    if let Inner::Raw(file) = &archive.inner {
        // TypedValue keeps its type where GGUF has one; bytes and json go
        // as a STRING holding the CONV-004 envelope (CONV-008).
        return file
            .metadata
            .iter()
            .map(|(key, value)| {
                let typed = MetaValue::from_typed(value)
                    .unwrap_or_else(|| MetaValue::String(retag(value, str::to_ascii_uppercase).to_string()));
                (key.clone(), typed)
            })
            .collect();
    }
    archive
        .metadata
        .iter()
        .map(|(key, value)| (key.clone(), MetaValue::String(value.clone())))
        .collect()
}

#[cfg(format_raw_sidecar)]
/// A TypedValue or envelope with `type` and `elem` spelled by `case`.
fn retag(value: &Value, case: fn(&str) -> String) -> Value {
    let mut value = value.clone();
    if let Some(object) = value.as_object_mut() {
        for field in ["type", "elem"] {
            if let Some(Value::String(name)) = object.get_mut(field) {
                *name = case(name);
            }
        }
    }
    value
}

/// Metadata for a rawtensor manifest from a typed source. Each key keeps
/// its type. `raw-sidecar.metadata.<key>` goes back to `<key>` (CONV-007),
/// and a STRING there holding an envelope becomes that TypedValue again,
/// with T lower-cased. The lower-case spelling written before 2026-10-31
/// is accepted with a warning (CONV-008).
#[cfg(format_raw_sidecar)]
fn raw_metadata(source: &[(String, MetaValue)], warnings: &mut Vec<String>) -> Map<String, Value> {
    let mut out = Map::new();
    for (key, value) in source {
        let Some(raw_key) = key.strip_prefix("raw-sidecar.metadata.") else {
            out.insert(key.clone(), value.typed_value());
            continue;
        };
        let envelope = match value {
            MetaValue::String(text) => serde_json::from_str::<Value>(text)
                .ok()
                .filter(|parsed| parsed.get("type").and_then(Value::as_str).is_some() && parsed.get("value").is_some()),
            _ => None,
        };
        let typed = match envelope {
            Some(parsed) => {
                let tag = parsed["type"].as_str().unwrap_or_default();
                if tag != tag.to_ascii_uppercase() {
                    warnings.push(format!(
                        "{key} has a lower-case envelope type {tag}; it is read as {} (CONV-008)",
                        tag.to_ascii_uppercase()
                    ));
                }
                retag(&parsed, str::to_ascii_lowercase)
            }
            None => value.typed_value(),
        };
        out.insert(raw_key.to_string(), typed);
    }
    out
}

/// `format.id` of the source's spec, the prefix of its keys in GGUF
/// (CONV-008).
fn format_id(format: FormatName) -> &'static str {
    match format {
        FormatName::Raw => "raw-sidecar",
        other => other.as_str(),
    }
}

/// Source metadata as GGUF KVs (conversion-considerations section 16). A
/// GGUF source keeps its keys. A `gguf.<key>` envelope is restored as the
/// typed `<key>` (CONV-007); one that does not parse stays a STRING under
/// `<format id>.metadata.gguf.<key>`. Any other key goes under
/// `<format id>.metadata.<key>` (CONV-008): as a STRING, or for
/// raw-sidecar with its own type when GGUF has it.
fn gguf_metadata(
    archive: &Archive,
    source: Vec<(String, MetaValue)>,
    warnings: &mut Vec<String>,
) -> Vec<(String, MetaValue)> {
    if matches!(archive.inner, Inner::Gguf(_)) {
        return source;
    }
    let prefix = format!("{}.metadata.", format_id(archive.format));
    let mut out = Vec::new();
    for (key, value) in source {
        if let (Some(gguf_key), MetaValue::String(text)) = (key.strip_prefix("gguf."), &value) {
            let restored = serde_json::from_str::<Value>(text)
                .ok()
                .as_ref()
                .and_then(MetaValue::from_typed);
            match restored {
                Some(typed) => out.push((gguf_key.to_string(), typed)),
                None => {
                    warnings.push(format!(
                        "{key} is not a typed GGUF envelope and is kept as {prefix}{key} (CONV-007)"
                    ));
                    out.push((format!("{prefix}{key}"), value));
                }
            }
            continue;
        }
        out.push((format!("{prefix}{key}"), value));
    }
    out
}

/// Metadata for a string map (safetensors `__metadata__`, ONNX
/// `metadata_props`). A GGUF key becomes `gguf.<key>` with the CONV-004
/// envelope. Tokenizer arrays belong in tokenizer.json, not a header, so
/// they are left out and named in a warning (CONV-006).
fn string_metadata(
    archive: &Archive,
    source: &[(String, MetaValue)],
    warnings: &mut Vec<String>,
) -> Vec<(String, String)> {
    if !matches!(archive.format, FormatName::Gguf) {
        return archive.metadata.clone();
    }
    let mut out = Vec::new();
    for (key, value) in source {
        if key.starts_with("tokenizer.ggml.") && matches!(value, MetaValue::Array { .. }) {
            warnings.push(format!(
                "{key} was not written; a tokenizer array belongs in tokenizer.json"
            ));
            continue;
        }
        out.push((format!("gguf.{key}"), value.envelope().to_string()));
    }
    out
}

#[cfg(format_safetensors)]
fn safe_out(tensors: &[Tensor]) -> Result<Vec<crate::formats::safetensors::OutTensor>> {
    tensors
        .iter()
        .map(|tensor| {
            let dtype = dtype::safe_dtype(&tensor.canonical)?;
            Ok(crate::formats::safetensors::OutTensor {
                name: tensor.name.clone(),
                dtype: dtype.name.to_string(),
                shape: tensor.shape.clone(),
                bytes: tensor.bytes.clone(),
            })
        })
        .collect()
}

#[cfg(format_npz)]
fn npz_out(tensors: &[Tensor]) -> Result<Vec<crate::formats::npz::NpzMember>> {
    tensors
        .iter()
        .map(|tensor| {
            Ok(crate::formats::npz::NpzMember {
                key: tensor.name.clone(),
                body: crate::formats::npz::NpzMemberBody::Rebuild {
                    descr: numpy_descr(&tensor.canonical)?.to_string(),
                    shape: tensor.shape.clone(),
                    payload: tensor.bytes.clone(),
                },
            })
        })
        .collect()
}

#[cfg(format_gguf)]
fn gguf_out(
    tensors: &[Tensor],
    hparams: Option<&Map<String, Value>>,
    rename: bool,
) -> Result<(Vec<(String, MetaValue)>, Vec<DenseTensor>)> {
    let mut metadata = Vec::new();
    if rename {
        if let Some(map) = hparams {
            metadata.extend(gguf_hparams(map)?);
        }
        metadata.push(("general.quantization_version".into(), MetaValue::U32(2)));
    }
    let mut dense = Vec::new();
    let mut f16 = 0;
    let mut bf16 = 0;
    let mut other = 0;
    for tensor in tensors {
        let code = element_ggml_code(&tensor.canonical).ok_or_else(|| {
            msg(format!(
                "dtype {} has no GGUF element type",
                tensor.canonical
            ))
        })?;
        match tensor.canonical.as_str() {
            "float32" => {}
            "float16" => f16 += 1,
            "bfloat16" => bf16 += 1,
            _ => other += 1,
        }
        let ne = if tensor.shape.is_empty() {
            // GGUF-018 / GGUF-023: a scalar is stored as [1], and its
            // canonical shape [] is kept in `<name>.orig_shape`.
            metadata.push((
                format!("{}.orig_shape", tensor.name),
                MetaValue::Array {
                    elem: crate::formats::gguf::TYPE_ARRAY_U64,
                    values: Vec::new(),
                },
            ));
            vec![1]
        } else {
            let mut ne = tensor.shape.clone();
            ne.reverse();
            ne
        };
        dense.push(DenseTensor {
            name: tensor.name.clone(),
            ggml_type: code,
            ne,
            payload: tensor.bytes.clone(),
        });
    }
    // llama_ftype names the dominant type (gguf.yaml): ALL_F32 0,
    // MOSTLY_F16 1, MOSTLY_BF16 32. f32 norms beside f16 weights are still
    // MOSTLY_F16. Other mixes are not written.
    let file_type = match (f16, bf16) {
        (0, 0) => Some(0),
        (_, 0) => Some(1),
        (0, _) => Some(32),
        _ => None,
    };
    if let (true, 0, Some(file_type)) = (rename, other, file_type) {
        metadata.push(("general.file_type".into(), MetaValue::U32(file_type)));
    }
    Ok((metadata, dense))
}

#[cfg(format_pytorch)]
fn pytorch_out(tensors: &[Tensor]) -> Result<Vec<crate::formats::pytorch::OutTensor>> {
    Ok(tensors
        .iter()
        .map(|tensor| crate::formats::pytorch::OutTensor {
            name: tensor.name.clone(),
            dtype: tensor.canonical.clone(),
            shape: tensor.shape.clone(),
            payload: tensor.bytes.clone(),
        })
        .collect())
}

#[cfg(format_onnx)]
fn onnx_out(tensors: &[Tensor]) -> Result<Vec<crate::formats::onnx::OutTensor>> {
    Ok(tensors
        .iter()
        .map(|tensor| crate::formats::onnx::OutTensor {
            name: tensor.name.clone(),
            dtype: tensor.canonical.clone(),
            shape: tensor.shape.clone(),
            payload: tensor.bytes.clone(),
        })
        .collect())
}

#[cfg(format_raw_sidecar)]
fn raw_out(tensors: &[Tensor]) -> Result<Vec<crate::formats::rawtensor::OutTensor>> {
    Ok(tensors
        .iter()
        .map(|tensor| crate::formats::rawtensor::OutTensor {
            name: tensor.name.clone(),
            canonical: tensor.canonical.clone(),
            shape: tensor.shape.clone(),
            payload: tensor.bytes.clone(),
            fortran: false,
        })
        .collect())
}

fn numpy_descr(canonical: &str) -> Result<&'static str> {
    dtype::safe_dtype(canonical)?
        .numpy
        .ok_or_else(|| msg(format!("dtype {canonical} has no NumPy descr")))
}

fn gguf_hparams(map: &Map<String, Value>) -> Result<Vec<(String, MetaValue)>> {
    let arch = map
        .get("general.architecture")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            map.get("architectures")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .and_then(Value::as_str)
                .and_then(gguf_names::architecture_for_hf_class)
        })
        .ok_or_else(|| msg("GGUF metadata needs an architecture"))?;
    let mut out = vec![(
        "general.architecture".into(),
        MetaValue::String(arch.clone()),
    )];
    if let Some(name) = map
        .get("_name_or_path")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        out.push(("general.name".into(), MetaValue::String(name.to_string())));
    }
    push_u32(
        &mut out,
        &format!("{arch}.block_count"),
        map,
        &["n_layers", "num_hidden_layers", "n_layer", "num_layers"],
    );
    push_u32(
        &mut out,
        &format!("{arch}.context_length"),
        map,
        &[
            "max_position_embeddings",
            "n_ctx",
            "n_positions",
            "max_length",
            "max_sequence_length",
            "model_max_length",
        ],
    );
    push_u32(
        &mut out,
        &format!("{arch}.embedding_length"),
        map,
        &["hidden_size", "n_embd", "dim"],
    );
    push_u32(
        &mut out,
        &format!("{arch}.feed_forward_length"),
        map,
        &[
            "prefix_dense_intermediate_size",
            "dense_intermediate_size",
            "intermediate_size",
            "n_inner",
            "hidden_dim",
        ],
    );
    push_u32(
        &mut out,
        &format!("{arch}.attention.head_count"),
        map,
        &["num_attention_heads", "n_head", "n_heads"],
    );
    if let Some(heads) = hparam_u64(Some(map), &["num_attention_heads", "n_head", "n_heads"]) {
        let kv = hparam_u64(Some(map), &["num_key_value_heads", "n_kv_heads"]).unwrap_or(heads);
        out.push((
            format!("{arch}.attention.head_count_kv"),
            MetaValue::U32(kv as u32),
        ));
    }
    if let Some(dim) = hparam_u64(Some(map), &["head_dim"]) {
        out.push((
            format!("{arch}.attention.key_length"),
            MetaValue::U32(dim as u32),
        ));
        out.push((
            format!("{arch}.attention.value_length"),
            MetaValue::U32(dim as u32),
        ));
    }
    push_f32(
        &mut out,
        &format!("{arch}.rope.freq_base"),
        map,
        &[
            "global_rope_theta",
            "rope_global_theta",
            "rope_theta_global",
            "rope_theta",
            "rotary_emb_base",
        ],
    );
    push_f32(
        &mut out,
        &format!("{arch}.attention.layer_norm_rms_epsilon"),
        map,
        &["rms_norm_eps", "norm_eps"],
    );
    push_f32(
        &mut out,
        &format!("{arch}.attention.layer_norm_epsilon"),
        map,
        &["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"],
    );
    push_u32(
        &mut out,
        &format!("{arch}.expert_count"),
        map,
        &["num_local_experts", "num_experts", "n_routed_experts"],
    );
    push_u32(
        &mut out,
        &format!("{arch}.expert_used_count"),
        map,
        &[
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k_experts",
        ],
    );
    push_u32(
        &mut out,
        &format!("{arch}.vocab_size"),
        map,
        &["vocab_size"],
    );
    push_u32(
        &mut out,
        &format!("{arch}.attention.sliding_window"),
        map,
        &["sliding_window"],
    );
    Ok(out)
}

fn push_u32(
    out: &mut Vec<(String, MetaValue)>,
    key: &str,
    map: &Map<String, Value>,
    names: &[&str],
) {
    if let Some(value) = hparam_u64(Some(map), names) {
        out.push((key.to_string(), MetaValue::U32(value as u32)));
    }
}

fn push_f32(
    out: &mut Vec<(String, MetaValue)>,
    key: &str,
    map: &Map<String, Value>,
    names: &[&str],
) {
    if let Some(value) = hparam_f64(Some(map), names) {
        out.push((key.to_string(), MetaValue::F32(value as f32)));
    }
}

fn metadata_notes(map: &Map<String, Value>) -> Vec<String> {
    let mut notes = Vec::new();
    if map.get("rope_scaling").is_some() {
        notes.push(
            "rope_scaling was left in config.json; the nested GGUF keys are not copied".into(),
        );
    }
    notes
}

fn read_config(dir: &Path) -> Result<Option<Value>> {
    let path = dir.join("config.json");
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path).map_err(|err| crate::error::io_err(&path, err))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|_| msg(format!("{} is not JSON", path.display())))?;
    if !value.is_object() {
        return Err(msg(format!("{} is not a JSON object", path.display())));
    }
    Ok(Some(value))
}

pub(crate) fn merge_text_config(config: &Value) -> Map<String, Value> {
    let mut merged = config.as_object().cloned().unwrap_or_default();
    if let Some(text) = config.get("text_config").and_then(Value::as_object) {
        for (key, value) in text {
            merged.insert(key.clone(), value.clone());
        }
    }
    merged
}

pub(crate) fn language_layers(hparams: Option<&Map<String, Value>>) -> Option<u64> {
    let map = hparams?;
    let hidden = hparam_u64(
        Some(map),
        &["num_hidden_layers", "n_layers", "n_layer", "num_layers"],
    );
    let nextn = hparam_u64(
        Some(map),
        &["nextn_predict_layers", "num_nextn_predict_layers"],
    );
    match (hidden, nextn) {
        (Some(hidden), Some(next)) => hidden.checked_sub(next),
        (Some(hidden), None) => Some(hidden),
        _ => None,
    }
}

fn hparam_u64(map: Option<&Map<String, Value>>, names: &[&str]) -> Option<u64> {
    let map = map?;
    for name in names {
        if let Some(value) = map.get(*name) {
            if let Some(number) = json_u64(value) {
                return Some(number);
            }
        }
    }
    None
}

fn hparam_f64(map: Option<&Map<String, Value>>, names: &[&str]) -> Option<f64> {
    let map = map?;
    for name in names {
        if let Some(value) = map.get(*name).and_then(Value::as_f64) {
            return Some(value);
        }
        if let Some(value) = map.get(*name).and_then(Value::as_u64) {
            return Some(value as f64);
        }
    }
    None
}

fn json_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|item| u64::try_from(item).ok()))
}

fn elem_width(canonical: &str) -> Result<usize> {
    let bits = dtype::canon_bits(canonical)
        .ok_or_else(|| msg(format!("dtype {canonical} has no element width")))?;
    if bits % 8 != 0 {
        return Err(msg(format!(
            "dtype {canonical} is not a whole number of bytes"
        )));
    }
    Ok((bits / 8) as usize)
}

fn add_scalar(tensor: &mut Tensor, add: f64) -> Result<()> {
    let data = tensor.bytes.read_vec()?;
    let next = map_floats(&tensor.canonical, &data, |value| value + add as f32)?;
    tensor.bytes = Payload::Bytes(next);
    Ok(())
}

fn neg_exp(tensor: &mut Tensor) -> Result<()> {
    let data = tensor.bytes.read_vec()?;
    let next = map_floats(&tensor.canonical, &data, |value| -value.exp())?;
    tensor.bytes = Payload::Bytes(next);
    Ok(())
}

fn map_floats(canonical: &str, data: &[u8], op: impl Fn(f32) -> f32) -> Result<Vec<u8>> {
    match canonical {
        "float32" => {
            let mut out = Vec::with_capacity(data.len());
            for chunk in data.chunks_exact(4) {
                let value = f32::from_le_bytes(chunk.try_into().unwrap());
                out.extend_from_slice(&op(value).to_le_bytes());
            }
            Ok(out)
        }
        "float16" => {
            let mut out = Vec::with_capacity(data.len());
            for chunk in data.chunks_exact(2) {
                let value =
                    crate::ggml_dequant::f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                out.extend_from_slice(&crate::ggml_dequant::f32_to_f16(op(value)).to_le_bytes());
            }
            Ok(out)
        }
        "bfloat16" => {
            let mut out = Vec::with_capacity(data.len());
            for chunk in data.chunks_exact(2) {
                let bits = u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16;
                let value = op(f32::from_bits(bits));
                let stored = crate::ggml_dequant::f32_to_bf16(value);
                out.extend_from_slice(&stored.to_le_bytes());
            }
            Ok(out)
        }
        other => Err(msg(format!("cannot apply a float change to {other}"))),
    }
}

fn unsqueeze(tensor: &mut Tensor, axis: i32) -> Result<()> {
    let rank = tensor.shape.len() as i32;
    let axis = if axis < 0 { rank + 1 + axis } else { axis };
    if axis < 0 || axis as usize > tensor.shape.len() {
        return Err(msg("unsqueeze axis is outside the shape"));
    }
    tensor.shape.insert(axis as usize, 1);
    Ok(())
}

fn squeeze_all(tensor: &mut Tensor) -> Result<()> {
    tensor.shape.retain(|dim| *dim != 1);
    Ok(())
}

fn squeeze_axis(tensor: &mut Tensor, axis: usize) -> Result<()> {
    if tensor.shape.get(axis).copied() != Some(1) {
        return Err(msg(format!(
            "squeeze axis {axis} of {} is not 1",
            tensor.name
        )));
    }
    tensor.shape.remove(axis);
    Ok(())
}

fn drop_first(tensor: &mut Tensor) -> Result<()> {
    if tensor.shape.first().copied().unwrap_or(0) < 1 {
        return Err(msg(format!("{} has no first row to drop", tensor.name)));
    }
    let width = elem_width(&tensor.canonical)?;
    let row = numel(&tensor.shape[1..]).unwrap_or(1) as usize * width;
    let data = tensor.bytes.read_vec()?;
    if data.len() < row {
        return Err(msg(format!("{} is shorter than one row", tensor.name)));
    }
    tensor.bytes = Payload::Bytes(data[row..].to_vec());
    tensor.shape[0] -= 1;
    Ok(())
}

fn clip_permute(tensor: &mut Tensor) -> Result<()> {
    let width = elem_width(&tensor.canonical)?;
    let count = numel(&tensor.shape).unwrap_or(0) as usize;
    if count % (16 * 16 * 3) != 0 {
        return Err(msg(format!(
            "{} cannot be viewed as [n, 16, 16, 3]",
            tensor.name
        )));
    }
    let n = count / (16 * 16 * 3);
    let data = tensor.bytes.read_vec()?;
    let mut out = vec![0u8; data.len()];
    // (n, 16, 16, 3) -> permute (0, 3, 1, 2) = (n, 3, 16, 16)
    for batch in 0..n {
        for h in 0..16 {
            for w in 0..16 {
                for channel in 0..3 {
                    let source = (((batch * 16 + h) * 16 + w) * 3 + channel) * width;
                    let dest = (((batch * 3 + channel) * 16 + h) * 16 + w) * width;
                    out[dest..dest + width].copy_from_slice(&data[source..source + width]);
                }
            }
        }
    }
    tensor.bytes = Payload::Bytes(out);
    tensor.shape = vec![n as u64, 3, 16, 16];
    Ok(())
}

pub fn convert_main(args: &[String]) -> Result<i32> {
    match parse_convert(args)? {
        ConvertCommand::Help => {
            println!("{}", convert_help());
            Ok(0)
        }
        ConvertCommand::Version => {
            println!("{}", crate::version_line("tensconv"));
            Ok(0)
        }
        ConvertCommand::Run(request) => {
            for warning in run(&request)? {
                eprintln!("tensconv: {warning}");
            }
            Ok(0)
        }
    }
}

enum ConvertCommand {
    Help,
    Version,
    Run(Request),
}

fn parse_convert(args: &[String]) -> Result<ConvertCommand> {
    let mut output = None;
    let mut format = None;
    let mut rename = false;
    let mut names_only = false;
    let mut force = false;
    let mut force_next = false;
    let mut source = None;
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--help" || arg == "-h" {
            return Ok(ConvertCommand::Help);
        }
        if arg == "--version" {
            return Ok(ConvertCommand::Version);
        }
        if arg == "--rename" {
            rename = true;
            index += 1;
            continue;
        }
        if arg == "--names-only" {
            names_only = true;
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
        if arg == "--format" || arg == "-o" || arg == "--output" {
            index += 1;
            let value = args
                .get(index)
                .ok_or_else(|| msg(format!("{arg} needs a value")))?;
            if arg == "--format" {
                format = Some(FormatName::parse(value)?);
            } else {
                output = Some(PathBuf::from(value));
            }
            index += 1;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--format=") {
            format = Some(FormatName::parse(value)?);
            index += 1;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--output=") {
            output = Some(PathBuf::from(value));
            index += 1;
            continue;
        }
        if arg.starts_with('-') {
            return Err(msg(format!("unknown option {arg}")));
        }
        if source.is_some() {
            return Err(msg("tensconv takes one source path"));
        }
        source = Some((PathBuf::from(arg), force_next));
        force_next = false;
        index += 1;
    }
    if force_next {
        return Err(msg(
            "--force-single-file must come immediately before a path",
        ));
    }
    let Some((source, force_single)) = source else {
        return Err(msg(convert_usage()));
    };
    let Some(output) = output else {
        return Err(msg("tensconv needs -o OUT"));
    };
    Ok(ConvertCommand::Run(Request {
        source,
        output,
        format,
        rename,
        names_only,
        force_single,
        force,
    }))
}

fn convert_usage() -> &'static str {
    "usage: tensconv [-o OUT] [--format NAME] [--rename] [--names-only] [--force] [--force-single-file] SRC"
}

pub fn convert_help() -> &'static str {
    "\
tensconv — copy tensors into another container

Usage:
  tensconv -o OUT [--format NAME] [--rename] [--names-only] [--force] [--force-single-file] SRC

The source is one container, or a directory of safetensors weights.
Shapes are stored outermost-first. A GGUF ne[] axis is reversed.
A dtype the target cannot store is widened when that widen is exact,
and otherwise the command stops. Narrowing is not done.
--rename maps names through the GGUF role table. The architecture is
general.architecture, or config.json architectures[0]. Value changes
run for a complete converter record, for the LLaMA Q/K permutation,
and for the Qwen3.5 changes. --names-only renames and skips those
changes. Quantized tensors are left for tensquant.
"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v_head_permutation_matches_the_spec_example() {
        assert_eq!(v_head_perm(2, 2, 1), vec![0, 2, 1, 3]);
    }

    #[test]
    fn llama_q_rows_follow_section_6() {
        let values: Vec<f32> = (0..8).map(|index| index as f32).collect();
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let permuted = permute_rows(&bytes, &[8], 4, 2).unwrap();
        let back: Vec<f32> = permuted
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(back, vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
    }

    #[test]
    fn a_gemma_norm_gains_one() {
        let mut tensor = Tensor {
            name: "model.layers.0.input_layernorm.weight".into(),
            canonical: "float32".into(),
            shape: vec![1],
            bytes: Payload::Bytes(1.5f32.to_le_bytes().to_vec()),
        };
        let converter = select_converter("gemma", Some("GemmaForCausalLM")).unwrap();
        assert!(converter.complete);
        apply_branches(&mut tensor, "gemma", &converter, None).unwrap();
        let value = f32::from_le_bytes(tensor.bytes.read_vec().unwrap().try_into().unwrap());
        assert!((value - 2.5).abs() < 1e-6);
    }

    #[cfg(all(format_safetensors, format_gguf))]
    #[test]
    fn a_dense_safetensors_file_round_trips_through_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("in.safetensors");
        let mid = dir.path().join("out.gguf");
        let back = dir.path().join("back.safetensors");
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        crate::formats::safetensors::write_file(
            &src,
            &[],
            &[crate::formats::safetensors::OutTensor {
                name: "weight".into(),
                dtype: "F32".into(),
                shape: vec![2, 3],
                bytes: Payload::Bytes(bytes.clone()),
            }],
        )
        .unwrap();
        run(&Request {
            source: src,
            output: mid.clone(),
            format: None,
            rename: false,
            names_only: false,
            force_single: false,
            force: false,
        })
        .unwrap();
        let gguf = crate::archive::open(&mid, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        assert_eq!(gguf.members[0].shape, vec![3, 2]);
        run(&Request {
            source: mid,
            output: back.clone(),
            format: None,
            rename: false,
            names_only: false,
            force_single: false,
            force: false,
        })
        .unwrap();
        let restored = crate::archive::open(&back, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        assert_eq!(restored.members[0].shape, vec![2, 3]);
        assert_eq!(restored.members[0].payload.read_vec().unwrap(), bytes);
    }
    #[cfg(all(format_safetensors, format_gguf))]
    #[test]
    fn gguf_metadata_survives_a_string_map() {
        use crate::formats::gguf::{DenseTensor, MetaValue};
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("in.gguf");
        let mid = dir.path().join("mid.safetensors");
        let back = dir.path().join("back.gguf");
        let metadata = vec![
            ("demo.count".to_string(), MetaValue::U64(1 << 60)),
            ("demo.eps".to_string(), MetaValue::F32(0.1)),
            ("demo.ids".to_string(), MetaValue::Array { elem: 5, values: vec![] }),
            ("tokenizer.ggml.tokens".to_string(), MetaValue::Array {
                elem: 8,
                values: vec![MetaValue::String("a".into())],
            }),
        ];
        crate::formats::gguf::write_dense(
            &src,
            &metadata,
            &[DenseTensor {
                name: "w".into(),
                ggml_type: 0,
                ne: vec![2],
                payload: Payload::Bytes(vec![0; 8]),
            }],
        )
        .unwrap();
        let request = |source: &Path, output: &Path| Request {
            source: source.to_path_buf(),
            output: output.to_path_buf(),
            format: None,
            rename: false,
            names_only: false,
            force_single: false,
            force: false,
        };
        let warnings = run(&request(&src, &mid)).unwrap();
        assert!(warnings.iter().any(|warning| warning.contains("tokenizer.ggml.tokens")));
        // CONV-004 and CONV-005: upper-case tags, elem, big integers as text,
        // f32 widened to the shortest f64.
        let st = crate::formats::safetensors::read_file(&mid, DEFAULT_MAX_HEADER).unwrap();
        let get = |key: &str| st.metadata.iter().find(|(name, _)| name == key).unwrap().1.clone();
        assert_eq!(get("gguf.demo.count"), r#"{"type":"UINT64","value":"1152921504606846976"}"#);
        assert_eq!(get("gguf.demo.eps"), r#"{"type":"FLOAT32","value":0.10000000149011612}"#);
        assert_eq!(get("gguf.demo.ids"), r#"{"type":"ARRAY","elem":"INT32","value":[]}"#);
        // CONV-007: the typed keys come back.
        run(&request(&mid, &back)).unwrap();
        let archive = crate::archive::open(&back, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let Inner::Gguf(file) = &archive.inner else { panic!() };
        let find = |key: &str| file.metadata.iter().find(|field| field.key == key).map(|field| field.value.clone());
        for (key, value) in &metadata[..3] {
            assert_eq!(find(key).as_ref(), Some(value), "{key}");
        }
    }
    #[cfg(all(format_raw_sidecar, format_gguf))]
    #[test]
    fn raw_sidecar_metadata_round_trips_through_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("in.rawtensor.json");
        let mid = dir.path().join("mid.gguf");
        let back = dir.path().join("back.rawtensor.json");
        let metadata: Map<String, Value> = serde_json::from_str(
            r#"{"n": {"type": "uint32", "value": 7}, "blob": {"type": "bytes", "value": "AAE="}, "cfg": {"type": "json", "value": {"a": [1]}}}"#,
        )
        .unwrap();
        crate::formats::rawtensor::write_archive(
            &src,
            &metadata,
            &[crate::formats::rawtensor::OutTensor {
                name: "w".into(),
                canonical: "float32".into(),
                shape: vec![2],
                payload: Payload::Bytes(vec![0; 8]),
                fortran: false,
            }],
        )
        .unwrap();
        let request = |source: &Path, output: &Path| Request {
            source: source.to_path_buf(),
            output: output.to_path_buf(),
            format: None,
            rename: false,
            names_only: false,
            force_single: false,
            force: false,
        };
        run(&request(&src, &mid)).unwrap();
        // CONV-008: GGUF types kept, bytes and json as upper-case envelopes.
        let gguf = crate::archive::open(&mid, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let Inner::Gguf(file) = &gguf.inner else { panic!() };
        let find = |key: &str| file.metadata.iter().find(|field| field.key == key).unwrap().value.clone();
        assert_eq!(find("raw-sidecar.metadata.n"), MetaValue::U32(7));
        assert_eq!(
            find("raw-sidecar.metadata.blob"),
            MetaValue::String(r#"{"type":"BYTES","value":"AAE="}"#.into())
        );
        // CONV-007: back in rawtensor, the prefix goes and T is lower-cased.
        run(&request(&mid, &back)).unwrap();
        let raw = crate::archive::open(&back, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let Inner::Raw(file) = &raw.inner else { panic!() };
        assert_eq!(file.metadata, metadata);
    }
}
