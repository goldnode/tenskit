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
    if request.output.exists() && !request.force {
        return Err(msg(format!(
            "{} exists; pass --force to overwrite",
            request.output.display()
        )));
    }
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
        let _ = arch;
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
        let mut len = member.nbytes;
        #[cfg(format_gguf)]
        if matches!(archive.inner, Inner::Gguf(_)) {
            shape.reverse();
            if let Inner::Gguf(file) = &archive.inner {
                if let Some(tensor) = file.tensors.iter().find(|item| item.name == member.name) {
                    len = dtype::ggml_payload_bytes(tensor.ggml_type, &tensor.shape)?;
                }
            }
        }
        out.push(Tensor {
            name: member.name.clone(),
            canonical: canonical.to_string(),
            shape,
            bytes: member.payload.clipped(len)?,
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

fn hf_class(config: &Value) -> Option<String> {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(Value::as_str)
        .map(str::to_string)
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
    let preferred = match arch {
        "qwen35" => Some("Qwen3_5TextModel"),
        "llama" => Some("LlamaModel"),
        "gemma" => Some("GemmaModel"),
        "gemma2" => Some("Gemma2Model"),
        _ => None,
    };
    if let Some(name) = preferred {
        if let Some(found) = records.iter().find(|record| record.class == name) {
            return Ok(found.clone());
        }
    }
    if records.len() == 1 {
        return Ok(records.into_iter().next().unwrap());
    }
    let names: Vec<_> = records.iter().map(|record| record.class.as_str()).collect();
    Err(msg(format!(
        "architecture {arch} has more than one converter ({}); name it with config.json architectures",
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
        let heads = if tensor.name.contains("k_proj") {
            n_kv
        } else if tensor.name.contains("q_proj") {
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
        "float8_e4m3fn" => fp8_to_f16(&raw, Fp8::E4M3)?,
        "float8_e5m2" => fp8_to_f16(&raw, Fp8::E5M2)?,
        "float8_e4m3fnuz" => fp8_to_f16(&raw, Fp8::E4M3FNUZ)?,
        "float8_e5m2fnuz" => fp8_to_f16(&raw, Fp8::E5M2FNUZ)?,
        "float4_e2m1fn" => f4_to_f16(&raw)?,
        "float8_e8m0fnu" => e8m0_to_f32(&raw)?,
        _ => unreachable!(),
    };
    tensor.bytes = Payload::Bytes(widened);
    Ok(next.to_string())
}

#[derive(Clone, Copy)]
enum Fp8 {
    E4M3,
    E5M2,
    E4M3FNUZ,
    E5M2FNUZ,
}

fn bf16_to_f32(raw: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(raw.len() * 2);
    for chunk in raw.chunks_exact(2) {
        let bits = u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(out)
}

fn e8m0_to_f32(raw: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(raw.len() * 4);
    for byte in raw {
        let bits = if *byte < 2 {
            0x0020_0000u32 << *byte
        } else {
            (u32::from(*byte) - 1) << 23
        };
        out.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(out)
}

fn f4_to_f16(raw: &[u8]) -> Result<Vec<u8>> {
    let values = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut out = Vec::with_capacity(raw.len() * 4);
    for byte in raw {
        for nibble in [byte & 0x0f, byte >> 4] {
            let sign = nibble & 0x08 != 0;
            let mag = values[(nibble & 0x07) as usize];
            let value = if sign { -mag } else { mag };
            out.extend_from_slice(&crate::ggml_dequant::f32_to_f16(value).to_le_bytes());
        }
    }
    Ok(out)
}

fn fp8_to_f16(raw: &[u8], kind: Fp8) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(raw.len() * 2);
    for byte in raw {
        let value = fp8_to_f32(*byte, kind);
        out.extend_from_slice(&crate::ggml_dequant::f32_to_f16(value).to_le_bytes());
    }
    Ok(out)
}

fn fp8_to_f32(byte: u8, kind: Fp8) -> f32 {
    match kind {
        Fp8::E4M3 => decode_fp8(byte, 4, 3, 7, true, false),
        Fp8::E5M2 => decode_fp8(byte, 5, 2, 15, false, false),
        Fp8::E4M3FNUZ => decode_fp8(byte, 4, 3, 8, true, true),
        Fp8::E5M2FNUZ => decode_fp8(byte, 5, 2, 16, false, true),
    }
}

fn decode_fp8(byte: u8, exp_bits: u32, man_bits: u32, bias: i32, no_inf: bool, fnuz: bool) -> f32 {
    if fnuz && byte == 0x80 {
        return f32::NAN;
    }
    let sign = (byte >> 7) & 1;
    let exp_mask = (1u8 << exp_bits) - 1;
    let man_mask = (1u8 << man_bits) - 1;
    let exp = (byte >> man_bits) & exp_mask;
    let man = byte & man_mask;
    let mag = if exp == 0 {
        (man as f32) * 2f32.powi(1 - bias - man_bits as i32)
    } else if u32::from(exp) == (1u32 << exp_bits) - 1 && !no_inf {
        if man == 0 {
            return if sign == 0 {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
        }
        return f32::NAN;
    } else if no_inf && u32::from(exp) == (1u32 << exp_bits) - 1 {
        return f32::NAN;
    } else {
        (1.0 + (man as f32) / ((1u32 << man_bits) as f32)) * 2f32.powi(i32::from(exp) - bias)
    };
    if sign == 1 {
        -mag
    } else {
        mag
    }
}

fn write_target(
    path: &Path,
    target: FormatName,
    tensors: &[Tensor],
    archive: &Archive,
    hparams: Option<&Map<String, Value>>,
    rename: bool,
) -> Result<()> {
    if tensors.is_empty() {
        return Err(msg("the source has no tensors"));
    }
    match target {
        #[cfg(format_safetensors)]
        FormatName::Safetensors => {
            let converted = safe_out(tensors)?;
            crate::formats::safetensors::write_file(path, &archive.metadata, &converted)
        }
        #[cfg(format_npy)]
        FormatName::Npy => {
            if tensors.len() != 1 {
                return Err(msg("an npy file holds one tensor"));
            }
            let tensor = &tensors[0];
            let descr = numpy_descr(&tensor.canonical)?;
            crate::formats::npy::write_file(path, descr, &tensor.shape, |file| {
                tensor.bytes.copy_to(file)
            })
        }
        #[cfg(format_npz)]
        FormatName::Npz => {
            let members = npz_out(tensors)?;
            crate::formats::npz::write_archive(path, &members)
        }
        #[cfg(format_gguf)]
        FormatName::Gguf => {
            let (metadata, dense) = gguf_out(tensors, hparams, rename)?;
            write_dense(path, &metadata, &dense)
        }
        #[cfg(format_pytorch)]
        FormatName::Pytorch => {
            let converted = pytorch_out(tensors)?;
            crate::formats::pytorch::write_archive(path, &converted)
        }
        #[cfg(format_onnx)]
        FormatName::Onnx => {
            let converted = onnx_out(tensors)?;
            crate::formats::onnx::write_archive(path, &archive.metadata, &converted)
        }
        #[cfg(format_raw_sidecar)]
        FormatName::Raw => {
            let converted = raw_out(tensors)?;
            let meta = Map::new();
            crate::formats::rawtensor::write_archive(path, &meta, &converted)
        }
    }
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
    let mut f32 = 0;
    let mut f16 = 0;
    let mut other = 0;
    for tensor in tensors {
        let code = element_ggml_code(&tensor.canonical).ok_or_else(|| {
            msg(format!(
                "dtype {} has no GGUF element type",
                tensor.canonical
            ))
        })?;
        match tensor.canonical.as_str() {
            "float32" => f32 += 1,
            "float16" => f16 += 1,
            _ => other += 1,
        }
        let ne = if tensor.shape.is_empty() {
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
    if other == 0 {
        let file_type = if f16 > 0 { 1 } else { 0 };
        let _ = (f32, file_type);
        if rename {
            metadata.push((
                "general.file_type".into(),
                MetaValue::U32(if f16 > 0 { 1 } else { 0 }),
            ));
        }
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

fn merge_text_config(config: &Value) -> Map<String, Value> {
    let mut merged = config.as_object().cloned().unwrap_or_default();
    if let Some(text) = config.get("text_config").and_then(Value::as_object) {
        for (key, value) in text {
            merged.insert(key.clone(), value.clone());
        }
    }
    merged
}

fn language_layers(hparams: Option<&Map<String, Value>>) -> Option<u64> {
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
                let stored = (value.to_bits() >> 16) as u16;
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
            println!("tensconv {}", env!("CARGO_PKG_VERSION"));
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
}
