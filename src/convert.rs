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
    /// A value change widened this BF16 tensor to F32 first, as llama.cpp's
    /// prepare_tensors does; it goes back to BF16 unless it keeps F32.
    from_bf16: bool,
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
    if request.rename {
        if let Some(problem) = config.as_ref().and_then(text_config_problem) {
            return Err(msg(problem));
        }
    }
    let hparams = config.as_ref().map(merge_text_config);
    // GGUF-039: hf_arch is the top-level architectures[0], before text_config
    // is merged over it.
    let hf_arch = config
        .as_ref()
        .and_then(|config| config.get("architectures"))
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut warnings = Vec::new();
    #[cfg(format_safetensors)]
    if target == FormatName::Safetensors {
        tied_copy_note(&archive, &mut warnings);
    }
    let mut tensors = load_tensors(&archive)?;
    let mut hparam_keys = None;
    if request.rename {
        let arch = choose_architecture(&archive, config.as_ref())?;
        let class = config.as_ref().and_then(hf_class);
        let converter = select_converter(&arch, class.as_deref())?;
        let layers = language_layers(hparams.as_ref());
        let mut model_tensors = Vec::new();
        if !request.names_only {
            let transcription = hand_transcription(&arch, &converter);
            if transcription == Some(Transcription::Llama) {
                llama_scope(&mut tensors, config.as_ref(), class.as_deref())?;
            }
            if transcription.is_none() {
                check_hooks(&arch, &converter)?;
                apply_filter(&mut tensors, &arch, &converter, hf_arch.as_deref(), &mut warnings)?;
            }
            apply_global_filters(&mut tensors, &converter, &mut warnings)?;
        }
        // converter_attributes model_tensors: the names after index_tensors'
        // filters. ModelBase.prepare_tensors skips the base suffixes later.
        model_tensors.extend(tensors.iter().map(|tensor| tensor.name.clone()));
        skip_base_suffixes(&mut tensors, &mut warnings);
        if !request.names_only {
            extra_tensor_note(&converter, hparams.as_ref(), &mut warnings);
            apply_values(
                &mut tensors,
                &arch,
                &converter,
                hparams.as_ref(),
                hf_arch.as_deref(),
                config_dir,
            )?;
        }
        rename_tensors(&mut tensors, &arch, layers, &mut warnings)?;
        if target == FormatName::Gguf {
            if let Some(map) = &hparams {
                if let Some(config) = &config {
                    warnings.extend(crate::gguf_keys::defaults_notes(config));
                }
                let loaded = config.as_ref().map(loaded_config).unwrap_or_else(|| map.clone());
                hparam_keys = Some(gguf_hparams(&arch, &converter.class, &loaded, config_dir, &model_tensors, &mut warnings)?);
            }
        }
        let gguf_values = target == FormatName::Gguf && !request.names_only;
        if gguf_values {
            keep_f32_roles(&mut tensors, &mut warnings)?;
        }
        narrow_after_values(&mut tensors, gguf_values);
    }
    let mut widened: Vec<((String, String), Vec<String>)> = Vec::new();
    for tensor in &mut tensors {
        if let Some(pair) = fit_dtype(tensor, target)? {
            match widened.iter_mut().find(|(seen, _)| *seen == pair) {
                Some((_, names)) => names.push(tensor.name.clone()),
                None => widened.push((pair, vec![tensor.name.clone()])),
            }
        }
    }
    for ((from, to), names) in widened {
        warnings.push(match names.as_slice() {
            [only] => format!("widened {only} from {from} to {to}"),
            _ => format!(
                "widened {} tensors from {from} to {to}, the first {}",
                names.len(),
                names[0]
            ),
        });
    }
    write_target(
        &request.output,
        target,
        &tensors,
        &archive,
        hparam_keys.as_deref(),
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
        let canonical = dtype::canonical_id(&member.dtype)
            .map_err(|err| msg(format!("tensor {}: {err}", member.name)))?;
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
            from_bf16: false,
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Transcription {
    Llama,
    Qwen35,
}

/// GGUF-030: the two converters the spec transcribes by hand, each only for
/// its own architecture. LlamaModel follows hf_tensor_names_to_gguf and
/// section 6; Qwen3_5TextModel follows qwen35_tensor_names_to_gguf. Every
/// other converter, Qwen3_5MoeTextModel and the other undo_permute classes
/// included, gets the table checks.
fn hand_transcription(arch: &str, converter: &ConverterRecord) -> Option<Transcription> {
    match (arch, converter.class.as_str()) {
        ("llama", "LlamaModel") => Some(Transcription::Llama),
        ("qwen35", "Qwen3_5TextModel") => Some(Transcription::Qwen35),
        _ => None,
    }
}

/// The scope of the LLaMA transcription (hf_tensor_names_to_gguf): a dense
/// checkpoint that is not an EAGLE-3 draft. On that path LlamaModel deletes a
/// `text_model.` segment, and prefixes `model.` when architectures[0] is
/// LlamaModel, before lookup.
fn llama_scope(tensors: &mut [Tensor], config: Option<&Value>, class: Option<&str>) -> Result<()> {
    let draft = config.is_some_and(|config| {
        config.get("draft_vocab_size").is_some()
            && config.get("num_hidden_layers").and_then(Value::as_u64) == Some(1)
    });
    if draft {
        return Err(msg(
            "cannot convert an EAGLE-3 draft with --rename: llama.cpp's LlamaModel converter \
             changes its tensors in ways the spec does not describe (GGUF-030)",
        ));
    }
    if let Some(tensor) = tensors
        .iter()
        .find(|tensor| tensor.name.contains("block_sparse_moe.experts"))
    {
        return Err(msg(format!(
            "cannot convert a mixture-of-experts LLaMA checkpoint with --rename: {} needs the \
             expert merge that llama.cpp's LlamaModel converter does, which the spec does not \
             describe (GGUF-030)",
            tensor.name
        )));
    }
    for tensor in tensors.iter_mut() {
        if tensor.name.contains("text_model.") {
            tensor.name = tensor.name.replace("text_model.", "");
        }
        if class == Some("LlamaModel") {
            tensor.name = format!("model.{}", tensor.name);
        }
    }
    Ok(())
}

/// CONV-010 (conversion-considerations section 18): safetensors cannot
/// alias, so tensors that share one PyTorch storage are written in full, one
/// copy per name, with a warning naming each pair. De-duplicating is the
/// other choice CONV-010 allows; tensconv does not make it.
#[cfg(format_safetensors)]
fn tied_copy_note(archive: &Archive, warnings: &mut Vec<String>) {
    if archive.format != FormatName::Pytorch {
        return;
    }
    let mut seen: std::collections::BTreeMap<(PathBuf, u64, u64), &str> = Default::default();
    for member in &archive.members {
        let Payload::Range { path, offset, len } = &member.payload else {
            continue;
        };
        if *len == 0 {
            continue;
        }
        match seen.get(&(path.clone(), *offset, *len)) {
            Some(first) => warnings.push(format!(
                "{} shares its storage with {first} in the source; safetensors cannot share, so both are \
                 written in full. safetensors' save_model and Hugging Face's save_pretrained \
                 would keep one name instead (CONV-010)",
                member.name
            )),
            None => {
                seen.insert((path.clone(), *offset, *len), member.name.as_str());
            }
        }
    }
}

fn apply_values(
    tensors: &mut [Tensor],
    arch: &str,
    converter: &ConverterRecord,
    hparams: Option<&Map<String, Value>>,
    hf_arch: Option<&str>,
    dir: &Path,
) -> Result<()> {
    if converter.complete {
        for tensor in tensors.iter_mut() {
            apply_branches(tensor, arch, converter, hf_arch)?;
        }
        return Ok(());
    }
    match hand_transcription(arch, converter) {
        Some(Transcription::Llama) => return llama_permute(tensors, hparams),
        Some(Transcription::Qwen35) => return qwen35_values(tensors, hparams, dir),
        None => {}
    }
    let mut text = format!(
        "cannot convert {arch} ({}) with --rename yet. llama.cpp's converter may change \
         some tensor values for this model, and the spec does not yet say which, so the \
         result could load and still answer wrongly. --names-only renames without changing \
         any value, at your own risk.",
        converter.class
    );
    if converter.unexpanded.is_empty() {
        text.push_str(" The spec has not expanded this converter's tensor method.");
    } else {
        text.push_str(" The steps the spec has not expanded, as llama.cpp's Python source:");
        for step in &converter.unexpanded {
            text.push_str("\n  ");
            text.push_str(step);
        }
    }
    Err(msg(text))
}

/// `modify_tensors_base.tensor_loop`: ModelBase.prepare_tensors skips these
/// names before any converter sees them.
fn skip_base_suffixes(tensors: &mut Vec<Tensor>, warnings: &mut Vec<String>) {
    let suffixes = gguf_names::base_drop_suffixes();
    tensors.retain(|tensor| {
        let skip = suffixes.iter().any(|suffix| tensor.name.ends_with(suffix.as_str()));
        if skip {
            warnings.push(format!(
                "left out {}, which llama.cpp's converter skips for every model",
                tensor.name
            ));
        }
        !skip
    });
}

/// GGUF-030: a hook other than generate_extra_tensors changes the tensor set
/// in code the spec does not transcribe.
fn check_hooks(arch: &str, converter: &ConverterRecord) -> Result<()> {
    // A repack row consumes checkpoint tensors, so it counts as another hook (GGUF-035).
    let repack = generated_row(converter).is_some_and(|row| row.kind == "repack");
    let other: Vec<&str> = converter
        .hooks
        .iter()
        .map(String::as_str)
        .filter(|hook| *hook != "generate_extra_tensors" || repack)
        .collect();
    if other.is_empty() {
        return Ok(());
    }
    Err(msg(format!(
        "cannot convert {arch} ({}) with --rename yet. llama.cpp's converter changes the set of \
         tensors for this model in code the spec does not describe ({}), so the result could \
         load and still answer wrongly. --names-only renames without changing any value, at \
         your own risk.",
        converter.class,
        other.join(", ")
    )))
}

/// GGUF-030: the converter's filter_tensors. A complete one drops the source
/// tensors its conditions name; an incomplete one stops the conversion.
fn apply_filter(
    tensors: &mut Vec<Tensor>,
    arch: &str,
    converter: &ConverterRecord,
    hf_arch: Option<&str>,
    warnings: &mut Vec<String>,
) -> Result<()> {
    let Some(filter) = &converter.filter else {
        return Ok(());
    };
    if !filter.complete {
        return Err(msg(format!(
            "cannot convert {arch} ({}) with --rename yet. llama.cpp's converter leaves out some \
             tensors for this model (filter_tensors), and the spec does not yet say which. \
             --names-only renames without changing any value, at your own risk.",
            converter.class
        )));
    }
    let class = hf_arch;
    let mut kept = Vec::with_capacity(tensors.len());
    for tensor in tensors.drain(..) {
        let mut dropped = false;
        for when in &filter.drop_when {
            if when_matches(when, &tensor.name, class)? {
                dropped = true;
                break;
            }
        }
        if dropped {
            warnings.push(format!(
                "left out {}, as llama.cpp's {} converter does",
                tensor.name, converter.class
            ));
        } else {
            kept.push(tensor);
        }
    }
    *tensors = kept;
    Ok(())
}

/// GGUF-030, round 8: after the class's own filters, the global filters that
/// `global_filters_after` names. TextModel's drop test is applied here;
/// ModelBase's rewrites are part of the name lookup.
fn apply_global_filters(
    tensors: &mut Vec<Tensor>,
    converter: &ConverterRecord,
    warnings: &mut Vec<String>,
) -> Result<()> {
    if !converter.global_filters_after.iter().any(|name| name == "TextModel") {
        return Ok(());
    }
    let when = gguf_names::text_model_drop_when()
        .ok_or_else(|| msg("the spec's TextModel filter is not in the expected form"))?;
    let mut kept = Vec::with_capacity(tensors.len());
    for tensor in tensors.drain(..) {
        let local = tensor.name.strip_prefix("model.").unwrap_or(&tensor.name);
        if when_matches(&when, local, None)? {
            warnings.push(format!(
                "left out {}, which llama.cpp's converter drops from a text model",
                tensor.name
            ));
        } else {
            kept.push(tensor);
        }
    }
    *tensors = kept;
    Ok(())
}

/// GGUF-035: the generate_extra_tensors row for this converter, if any.
fn generated_row(converter: &ConverterRecord) -> Option<&'static crate::dtype::GeneratedTensors> {
    let (class, line) = converter.generator.as_ref()?;
    crate::dtype::GENERATED_TENSORS
        .iter()
        .find(|row| row.class == class && row.line == *line)
}

/// GGUF-035's P and R: P is rope_parameters, else rope_scaling, else {}; its
/// rope_type comes from type when it has neither full_attention nor
/// sliding_attention. R is P.full_attention when present, else P.
/// A gguf_generated_tensors condition, read up to its first `;`. None when
/// the form is not one tenspile knows, such as a file beside the weights.
fn generated_condition(when: &str, hparams: Option<&Map<String, Value>>) -> Option<bool> {
    let when = when.split(';').next().unwrap_or(when).trim();
    if when.starts_with("always") {
        return Some(true);
    }
    if when.starts_with("never") {
        return Some(false);
    }
    let (p, r) = crate::gguf_keys::rope_params(hparams);
    let text = |map: &Map<String, Value>, key: &str| map.get(key).and_then(Value::as_str).map(str::to_string);
    let filled = |key: &str| p.get(key).and_then(Value::as_array).is_some_and(|items| !items.is_empty());
    if let Some(rest) = when.strip_prefix("R.rope_type lowercased == ") {
        let want = quoted(rest).ok()?;
        return Some(text(&r, "rope_type").map(|kind| kind.to_lowercase()) == Some(want));
    }
    if let Some(rest) = when.strip_prefix("P.rope_type == ") {
        let want = quoted(rest).ok()?;
        return Some(text(&p, "rope_type") == Some(want));
    }
    match when {
        "P.long_factor is non-empty" => Some(filled("long_factor")),
        "P.long_factor or P.short_factor is non-empty" => Some(filled("long_factor") || filled("short_factor")),
        _ => None,
    }
}

/// GGUF-030 and GGUF-035: tensors that generate_extra_tensors computes are
/// not written. One warning, only when the spec's condition holds, or when
/// tensconv cannot test the condition.
fn extra_tensor_note(
    converter: &ConverterRecord,
    hparams: Option<&Map<String, Value>>,
    warnings: &mut Vec<String>,
) {
    if converter.generator.is_none() {
        return;
    }
    let Some(row) = generated_row(converter) else {
        warnings.push(format!(
            "llama.cpp's {} converter may also compute tensors (generate_extra_tensors), and the \
             spec has no row for it; tensconv does not write them (GGUF-035)",
            converter.class
        ));
        return;
    };
    if row.kind == "none" || row.kind == "repack" {
        return;
    }
    let condition = match generated_condition(row.when, hparams) {
        Some(false) => return,
        Some(true) => String::new(),
        None => format!(" when {}", row.when),
    };
    let tensors = row.tensors.replace('"', "");
    let effect = match row.kind {
        "rope" => "llama.cpp still loads the file, but runs RoPE without them, so its answers can differ",
        "side-file" => "they come from files beside the weights",
        _ => "they are computed from config.json",
    };
    warnings.push(format!(
        "llama.cpp's {} converter would also write {tensors}{condition}. tensconv does not write \
         them; {effect} (GGUF-030, GGUF-035)",
        converter.class
    ));
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
    hf_arch: Option<&str>,
) -> Result<()> {
    for branch in &converter.branches {
        if !when_matches(&branch.when, &tensor.name, hf_arch)? {
            continue;
        }
        apply_expr(tensor, arch, &branch.assigns, &branch.expr)?;
    }
    Ok(())
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
    let clause = clause.trim();
    if let Some(inner) = clause.strip_prefix('(').and_then(|text| text.strip_suffix(')')) {
        if balanced(inner) {
            return clause_matches(inner, name, hf_arch);
        }
    }
    if let Some(rest) = clause.strip_prefix("not ") {
        return Ok(!clause_matches(rest, name, hf_arch)?);
    }
    // filter_tensors on an (name, data) item tests item[0], the name.
    if clause.contains("item[0]") {
        return clause_matches(&clause.replace("item[0]", "name"), name, hf_arch);
    }
    if let Some(rest) = clause.strip_prefix("name ==") {
        return Ok(name == quoted(rest.trim())?);
    }
    if let Some(list) = clause
        .strip_prefix("any((p in name for p in [")
        .and_then(|rest| rest.strip_suffix("]))"))
    {
        let mut hit = false;
        for part in list.split(',') {
            hit |= name.contains(&quoted(part.trim())?);
        }
        return Ok(hit);
    }
    if let Some(rest) = clause.strip_prefix("name.startswith((") {
        let body = rest
            .strip_suffix("))")
            .ok_or_else(|| msg(format!("unrecognized condition {clause}")))?;
        let mut hit = false;
        for part in body.split(',').filter(|part| !part.trim().is_empty()) {
            hit |= name.starts_with(&quoted(part.trim())?);
        }
        return Ok(hit);
    }
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

fn balanced(text: &str) -> bool {
    let mut depth = 0i32;
    for ch in text.chars() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            _ => {}
        }
        if depth < 0 {
            return false;
        }
    }
    depth == 0
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

/// GGUF-029, after GGUF-024 steps 2 and 3: convert_hf_to_gguf writes norms, 1-D tensors,
/// biases, and a few roles as F32, and llama.cpp's CPU backend cannot
/// multiply by a BF16 norm. An F16 or BF16 tensor in those roles is widened
/// to F32, which is exact. Other dtypes are left as they are.
fn keep_f32_roles(tensors: &mut [Tensor], warnings: &mut Vec<String>) -> Result<()> {
    let mut widened = 0;
    for tensor in tensors.iter_mut() {
        if !crate::quant::keeps_f32(&tensor.name, tensor.shape.len()) {
            continue;
        }
        let raw = match tensor.canonical.as_str() {
            "bfloat16" => bf16_to_f32(&tensor.bytes.read_vec()?)?,
            "float16" => tensor
                .bytes
                .read_vec()?
                .chunks_exact(2)
                .flat_map(|pair| {
                    crate::ggml_dequant::f16_to_f32(u16::from_le_bytes([pair[0], pair[1]]))
                        .to_le_bytes()
                })
                .collect(),
            _ => continue,
        };
        tensor.bytes = Payload::Bytes(raw);
        tensor.canonical = "float32".to_string();
        widened += 1;
    }
    if widened > 0 {
        warnings.push(format!(
            "widened {widened} norm, bias, and 1-D tensors to F32 so llama.cpp can run them (GGUF-029)"
        ));
    }
    Ok(())
}

/// Widen a dtype the target cannot store. Returns the (from, to) pair, so
/// one warning can count every tensor widened the same way.
fn fit_dtype(tensor: &mut Tensor, target: FormatName) -> Result<Option<(String, String)>> {
    if target_accepts(target, &tensor.canonical)? {
        return Ok(None);
    }
    let widened = widen_to(tensor, target)?;
    let from = std::mem::replace(&mut tensor.canonical, widened.clone());
    Ok(Some((from, widened)))
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
                | "uint16"
                | "uint32"
                | "uint64"
                | "float8_e4m3fn"
                | "float8_e5m2"
                | "float8_e4m3fnuz"
                | "float8_e5m2fnuz"
                | "float8_e8m0fnu"
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
    hparam_keys: Option<&[(String, MetaValue)]>,
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
            warnings.push(match source.len() {
                1 => format!("1 metadata key was not written: {target} has no metadata"),
                count => format!("{count} metadata keys were not written: {target} has no metadata"),
            });
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
            let (mut metadata, dense) = gguf_out(tensors, hparam_keys, rename)?;
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
            if rename {
                tokenizer_note(&metadata, warnings);
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
pub(crate) fn source_metadata(archive: &Archive) -> Vec<(String, MetaValue)> {
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
pub(crate) fn gguf_metadata(
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
/// GGUF-033 and GGUF-034: llama.cpp needs tokenizer.ggml.model to load a
/// model. Building the tokenizer keys is not transcribed (GGUF-Q5), so a file
/// without them is written and the user is told.
pub(crate) fn tokenizer_note(metadata: &[(String, MetaValue)], warnings: &mut Vec<String>) {
    if metadata.iter().any(|(key, _)| key == "tokenizer.ggml.model") {
        return;
    }
    warnings.push(
        "the output has no tokenizer.ggml.* keys, so llama.cpp will not load it until they are \
         added, for example from another GGUF of the same model (GGUF-033, GGUF-034)"
            .to_string(),
    );
}

fn gguf_out(
    tensors: &[Tensor],
    hparam_keys: Option<&[(String, MetaValue)]>,
    rename: bool,
) -> Result<(Vec<(String, MetaValue)>, Vec<DenseTensor>)> {
    let mut metadata = Vec::new();
    if rename {
        metadata.extend(hparam_keys.unwrap_or_default().iter().cloned());
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
            extra: Map::new(),
        })
        .collect())
}

fn numpy_descr(canonical: &str) -> Result<&'static str> {
    dtype::safe_dtype(canonical)?
        .numpy
        .ok_or_else(|| msg(format!("dtype {canonical} has no NumPy descr")))
}

/// GGUF-037: general.architecture, general.name, and the keys the converter
/// class's set_gguf_parameters chain writes. Listed keys whose guard or value
/// cannot be evaluated from config.json are named in one warning.
fn gguf_hparams(
    arch: &str,
    class: &str,
    map: &Map<String, Value>,
    dir: &Path,
    model_tensors: &[String],
    warnings: &mut Vec<String>,
) -> Result<Vec<(String, MetaValue)>> {
    let mut out = vec![("general.architecture".into(), MetaValue::String(arch.to_string()))];
    if let Some(name) = map
        .get("_name_or_path")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        out.push(("general.name".into(), MetaValue::String(name.to_string())));
    }
    warnings.push(
        "the keys come from config.json with the transformers defaults GGUF-039 and hf-config-defaults.json list; llama.cpp reads \
         transformers' AutoConfig dict, so a key that depends on another default may differ (GGUF-040)"
            .into(),
    );
    let found = crate::gguf_keys::class_keys_with(class, arch, map, dir, model_tensors)?;
    if !found.may_raise.is_empty() {
        warnings.push(format!(
            "llama.cpp's {class} may stop at {}; tensconv cannot tell from config.json whether it would (GGUF-037)",
            found.may_raise.join(", ")
        ));
    }
    if !found.unapplied.is_empty() {
        warnings.push(format!(
            "llama.cpp's {class} changes its config dict at {}, which tensconv cannot evaluate, so keys read after \
             it may differ (GGUF-037)",
            found.unapplied.join(", ")
        ));
    }
    out.extend(found.keys);
    let (sampling, skipped) = crate::gguf_keys::sampling_keys(dir);
    out.extend(sampling);
    warnings.extend(skipped);
    if !found.unevaluated.is_empty() {
        warnings.push(format!(
            "llama.cpp's {class} may also write {}; tensconv cannot evaluate when or what from config.json, so they are left out (GGUF-037)",
            found.unevaluated.join(", ")
        ));
    }
    Ok(out)
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

/// GGUF-039 and GGUF-040: the config dict llama.cpp's rules read, as near as
/// tenspile gets without transformers. The defaults GGUF-039 lists are added
/// to config.json, the sub-config renames are applied, and text_config is
/// merged over the top level, as TextModel.__init__ does.
/// GGUF-039: load_hparams' five tests, in this order. Each sets text_config,
/// so the last present one wins and replaces any text_config already there.
/// The source keys stay in the dict.
fn renamed_text_config(top: &Map<String, Value>) -> Option<Value> {
    let mut chosen = None;
    for name in ["llm_config", "lm_config", "thinker_config", "language_config", "lfm"] {
        let Some(value) = top.get(name) else { continue };
        chosen = Some(if name == "thinker_config" {
            value.get("text_config").cloned().unwrap_or(Value::Null)
        } else {
            value.clone()
        });
    }
    chosen
}

/// The renames that stop llama.cpp's converter: a null source makes
/// text_config null, which TextModel.__init__ cannot merge, and a
/// thinker_config without text_config raises KeyError.
pub(crate) fn text_config_problem(config: &Value) -> Option<String> {
    let top = config.as_object()?;
    if top.get("thinker_config").is_some_and(|thinker| thinker.get("text_config").is_none()) {
        return Some("config.json thinker_config has no text_config; llama.cpp's converter stops there too".into());
    }
    match renamed_text_config(top) {
        Some(Value::Null) => Some(
            "config.json sets text_config to null through a sub-config; llama.cpp's converter stops there too".into(),
        ),
        _ => None,
    }
}

pub(crate) fn merge_text_config(config: &Value) -> Map<String, Value> {
    let top = loaded_config(config);
    let mut merged = top.clone();
    if let Some(text) = top.get("text_config").and_then(Value::as_object) {
        for (key, value) in text {
            merged.insert(key.clone(), value.clone());
        }
    }
    merged
}

/// GGUF-039: the config dict as ModelBase.load_hparams gives it: the listed
/// transformers defaults and the text_config renames applied, and
/// text_config not yet merged (TextModel.__init__ merges it).
pub(crate) fn loaded_config(config: &Value) -> Map<String, Value> {
    let mut top = config.as_object().cloned().unwrap_or_default();
    let _ = crate::gguf_keys::listed_defaults(&mut top);
    if let Some(sub) = renamed_text_config(&top) {
        top.insert("text_config".into(), sub);
    }
    if let Some(sub) = top.get_mut("text_config").and_then(Value::as_object_mut) {
        // GGUF-039: to_dict() writes text_config as the sub-config's own full
        // dict, with its class defaults and the common keys, and those win in the merge.
        let _ = crate::gguf_keys::listed_defaults(sub);
    } else if !top.contains_key("text_config") {
        // A composite type builds one from the top-level keys config.json gives.
        if let Some(built) = config.as_object().and_then(crate::gguf_keys::built_text_config) {
            top.insert("text_config".into(), Value::Object(built));
        }
    }
    top
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

/// GGUF-036: llama.cpp's ModelBase.prepare_tensors turns a tensor that is not
/// F16 or F32 into F32 before modify_tensors changes its values, so `w + 1` on
/// a BF16 norm is exact. An F16 tensor stays F16 for the arithmetic.
fn widen_for_values(tensor: &mut Tensor) -> Result<()> {
    if tensor.canonical == "bfloat16" {
        tensor.bytes = Payload::Bytes(bf16_to_f32(&tensor.bytes.read_vec()?)?);
        tensor.canonical = "float32".to_string();
        tensor.from_bf16 = true;
    }
    Ok(())
}

/// GGUF-036, after the GGUF-029 widen: a BF16 tensor whose values were
/// computed in F32 returns to BF16 unless it is written F32, as llama.cpp's
/// --outtype bf16 rounds it once.
fn narrow_after_values(tensors: &mut [Tensor], keep_f32: bool) {
    for tensor in tensors.iter_mut() {
        if !tensor.from_bf16 || tensor.canonical != "float32" {
            continue;
        }
        if keep_f32 && crate::quant::keeps_f32(&tensor.name, tensor.shape.len()) {
            continue;
        }
        let raw = tensor.bytes.read_vec().unwrap_or_default();
        let narrowed: Vec<u8> = raw
            .chunks_exact(4)
            .flat_map(|chunk| {
                let value = f32::from_le_bytes(chunk.try_into().unwrap());
                crate::ggml_dequant::f32_to_bf16(value).to_le_bytes()
            })
            .collect();
        tensor.bytes = Payload::Bytes(narrowed);
        tensor.canonical = "bfloat16".to_string();
        tensor.from_bf16 = false;
    }
}

fn add_scalar(tensor: &mut Tensor, add: f64) -> Result<()> {
    widen_for_values(tensor)?;
    let data = tensor.bytes.read_vec()?;
    let next = map_floats(&tensor.canonical, &data, |value| value + add as f32)?;
    tensor.bytes = Payload::Bytes(next);
    Ok(())
}

fn neg_exp(tensor: &mut Tensor) -> Result<()> {
    widen_for_values(tensor)?;
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
            outln!("{}", convert_help());
            Ok(0)
        }
        ConvertCommand::Version => {
            outln!("{}", crate::version_line("tensconv"));
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

Examples:
  tensconv -o model.gguf model.safetensors
  tensconv --rename -o model.gguf model/
  tensconv -o weights.npz model.gguf
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
            from_bf16: false,
        };
        let converter = select_converter("gemma", Some("GemmaForCausalLM")).unwrap();
        assert!(converter.complete);
        apply_branches(&mut tensor, "gemma", &converter, None).unwrap();
        let value = f32::from_le_bytes(tensor.bytes.read_vec().unwrap().try_into().unwrap());
        assert!((value - 2.5).abs() < 1e-6);
    }

    #[cfg(all(format_safetensors, format_pytorch))]
    #[test]
    fn a_tied_pytorch_tensor_is_reported_when_written_twice() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("generated/test-vectors/small-pt-fp8.pt");
        let archive = archive::open(&source, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let mut warnings = Vec::new();
        tied_copy_note(&archive, &mut warnings);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].starts_with("tied_b shares its storage with tied_a"), "{warnings:?}");
    }

    #[test]
    fn a_bf16_norm_gains_one_in_f32() {
        // bf16 0x3c01 is 2^-7 * (1 + 1/128); plus 1 is not a bf16 value.
        let small = f32::from_bits(0x3c01_0000);
        let tensor = |name: &str, shape: Vec<u64>| Tensor {
            name: name.into(),
            canonical: "bfloat16".into(),
            shape,
            bytes: Payload::Bytes(vec![0x01, 0x3c]),
            from_bf16: false,
        };
        let converter = select_converter("gemma", Some("GemmaForCausalLM")).unwrap();
        let mut norm = tensor("model.layers.0.input_layernorm.weight", vec![1]);
        apply_branches(&mut norm, "gemma", &converter, None).unwrap();
        norm.name = "blk.0.attn_norm.weight".into();
        let mut other = tensor("blk.0.attn_k.weight", vec![1, 1]);
        add_scalar(&mut other, 1.0).unwrap();
        let mut tensors = vec![norm, other];
        narrow_after_values(&mut tensors, true);
        assert_eq!(tensors[0].canonical, "float32");
        assert_eq!(tensors[0].bytes.read_vec().unwrap(), (small + 1.0).to_le_bytes());
        assert_eq!(tensors[1].canonical, "bfloat16");
        assert_eq!(tensors[1].bytes.read_vec().unwrap(), crate::ggml_dequant::f32_to_bf16(small + 1.0).to_le_bytes());
        // Into safetensors nothing is kept F32, so the norm returns to BF16 too.
        let mut norm = tensor("model.norm.weight", vec![1]);
        add_scalar(&mut norm, 1.0).unwrap();
        let mut tensors = vec![norm];
        narrow_after_values(&mut tensors, false);
        assert_eq!(tensors[0].canonical, "bfloat16");
    }

    #[cfg(all(format_safetensors, format_gguf))]
    #[test]
    fn renamed_norms_and_biases_become_f32_for_llama_cpp() {
        let tensor = |name: &str, canonical: &str, shape: Vec<u64>, bytes: Vec<u8>| Tensor {
            name: name.into(),
            canonical: canonical.into(),
            shape,
            bytes: Payload::Bytes(bytes),
            from_bf16: false,
        };
        // bf16 1.5 is 0x3fc0, f16 -2.0 is 0xc000.
        let mut tensors = vec![
            tensor("blk.0.attn_norm.weight", "bfloat16", vec![2], vec![0xc0, 0x3f, 0xc0, 0x3f]),
            tensor("blk.0.attn_k.bias", "float16", vec![1], vec![0x00, 0xc0]),
            tensor("blk.0.attn_k.weight", "bfloat16", vec![1, 2], vec![0xc0, 0x3f, 0xc0, 0x3f]),
            tensor("blk.0.ffn_norm.weight", "float32", vec![1], 1.0f32.to_le_bytes().to_vec()),
        ];
        let mut warnings = Vec::new();
        keep_f32_roles(&mut tensors, &mut warnings).unwrap();
        assert_eq!(tensors[0].canonical, "float32");
        assert_eq!(tensors[0].bytes.read_vec().unwrap(), [1.5f32.to_le_bytes(), 1.5f32.to_le_bytes()].concat());
        assert_eq!(tensors[1].canonical, "float32");
        assert_eq!(tensors[1].bytes.read_vec().unwrap(), (-2.0f32).to_le_bytes());
        assert_eq!(tensors[2].canonical, "bfloat16");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].starts_with("widened 2 "), "{warnings:?}");
    }

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
                extra: Map::new(),
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

    #[test]
    fn every_drop_condition_in_the_spec_parses() {
        // Every converter's drop tests, its own and its chain's.
        let mut conditions = Vec::new();
        for arch in gguf_names::architecture_ids() {
            for record in gguf_names::converter_records(arch) {
                conditions.extend(record.filter.map(|filter| filter.drop_when).unwrap_or_default());
            }
        }
        assert!(!conditions.is_empty());
        conditions.push(gguf_names::text_model_drop_when().expect("TextModel drop test"));
        for when in &conditions {
            when_matches(when, "model.layers.0.mlp.weight", None)
                .unwrap_or_else(|err| panic!("{when}: {err}"));
        }
        let cases = [
            ("name == 'lm_head.weight'", "lm_head.weight", true),
            ("name == 'lm_head.weight'", "lm_head.weightx", false),
            ("not name.startswith('model.vision.')", "model.vision.x", false),
            ("not name.startswith('model.vision.')", "model.text.x", true),
            ("name.startswith(('lm_head.', 'mtp'))", "mtp.0", true),
            ("item[0].startswith('mtp.')", "mtp.1", true),
            ("any((p in name for p in ['codebook_offsets', 'depth_linear']))", "a.depth_linear.w", true),
            ("not name.startswith('language_model.') or '_mot_gen' in name", "language_model.a_mot_gen", true),
            ("not name.startswith('language_model.') or '_mot_gen' in name", "language_model.a", false),
        ];
        for (when, name, expected) in cases {
            assert_eq!(when_matches(when, name, None).unwrap(), expected, "{when} on {name}");
        }
    }

    #[test]
    fn converters_follow_gguf_030() {
        let record = |class: &str, hooks: &[&str], filter: Option<(bool, &[&str])>| ConverterRecord {
            class: class.into(),
            hf_architectures: Vec::new(),
            complete: true,
            unexpanded: Vec::new(),
            branches: Vec::new(),
            filter: filter.map(|(complete, drop)| gguf_names::TensorFilter {
                complete,
                drop_when: drop.iter().map(|item| item.to_string()).collect(),
            }),
            hooks: hooks.iter().map(|item| item.to_string()).collect(),
            generator: hooks
                .contains(&"generate_extra_tensors")
                .then(|| (class.to_string(), if class == "Phi3MiniModel" { 175 } else { 0 })),
            global_filters_after: Vec::new(),
        };
        assert!(check_hooks("phi3", &record("Phi3MiniModel", &["generate_extra_tensors"], None)).is_ok());
        let err = check_hooks("x", &record("X", &["generate_extra_tensors", "prepare_tensors"], None))
            .unwrap_err()
            .to_string();
        assert!(err.contains("(prepare_tensors)"), "{err}");

        let tensor = |name: &str| Tensor {
            name: name.into(),
            canonical: "float32".into(),
            shape: vec![1],
            bytes: Payload::Bytes(vec![0; 4]),
            from_bf16: false,
        };
        let mut tensors = vec![tensor("lm_head.weight"), tensor("model.norm.weight")];
        let mut warnings = Vec::new();
        let gemma = record("GemmaModel", &[], Some((true, &["name == 'lm_head.weight'"])));
        apply_filter(&mut tensors, "gemma", &gemma, None, &mut warnings).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "model.norm.weight");
        let open = record("LFM2Model", &[], Some((false, &[])));
        assert!(apply_filter(&mut tensors, "lfm2", &open, None, &mut warnings).is_err());

        let mut base = vec![tensor("h.0.attention.masked_bias"), tensor("h.0.attn.c_attn.weight")];
        skip_base_suffixes(&mut base, &mut warnings);
        assert_eq!(base.len(), 1);

        let config = |text: &str| serde_json::from_str::<Value>(text).unwrap().as_object().unwrap().clone();
        let phi = record("Phi3MiniModel", &["generate_extra_tensors"], None);
        let mut notes = Vec::new();
        extra_tensor_note(&phi, Some(&config(r#"{"rope_scaling": null}"#)), &mut notes);
        assert!(notes.is_empty());
        extra_tensor_note(&phi, Some(&config(r#"{"rope_scaling": {"long_factor": [1.0]}}"#)), &mut notes);
        assert_eq!(notes.len(), 1);
        assert!(notes[0].contains("rope_factors_long"), "{notes:?}");
    }

    #[test]
    fn a_gguf_without_a_tokenizer_model_is_reported() {
        let mut warnings = Vec::new();
        tokenizer_note(&[], &mut warnings);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("GGUF-033"));
        let mut warnings = Vec::new();
        tokenizer_note(
            &[("tokenizer.ggml.model".into(), MetaValue::String("gpt2".into()))],
            &mut warnings,
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn the_llama_transcription_keeps_to_its_scope() {
        let record = |class: &str| ConverterRecord {
            class: class.into(),
            hf_architectures: Vec::new(),
            complete: false,
            unexpanded: Vec::new(),
            branches: Vec::new(),
            filter: None,
            hooks: Vec::new(),
            generator: None,
            global_filters_after: Vec::new(),
        };
        assert_eq!(hand_transcription("llama", &record("LlamaModel")), Some(Transcription::Llama));
        assert_eq!(hand_transcription("granite", &record("GraniteModel")), None);
        assert_eq!(hand_transcription("qwen35moe", &record("Qwen3_5MoeTextModel")), None);
        let tensor = |name: &str| Tensor {
            name: name.into(),
            canonical: "float32".into(),
            shape: vec![1],
            bytes: Payload::Bytes(vec![0; 4]),
            from_bf16: false,
        };
        let mut tensors = vec![tensor("text_model.layers.0.mlp.up_proj.weight")];
        llama_scope(&mut tensors, None, Some("LlamaModel")).unwrap();
        assert_eq!(tensors[0].name, "model.layers.0.mlp.up_proj.weight");
        let draft: Value = serde_json::from_str(r#"{"draft_vocab_size": 32000, "num_hidden_layers": 1}"#).unwrap();
        assert!(llama_scope(&mut tensors, Some(&draft), None).is_err());
        let mut moe = vec![tensor("model.layers.0.block_sparse_moe.experts.0.w1.weight")];
        assert!(llama_scope(&mut moe, None, None).is_err());
    }

    #[test]
    fn a_text_model_drops_vision_tensors() {
        let tensor = |name: &str| Tensor {
            name: name.into(),
            canonical: "float32".into(),
            shape: vec![1],
            bytes: Payload::Bytes(vec![0; 4]),
            from_bf16: false,
        };
        let mut converter = gguf_names::converter_records("qwen2")
            .into_iter()
            .find(|record| record.class == "Qwen2Model")
            .unwrap();
        assert!(converter.global_filters_after.iter().any(|name| name == "TextModel"));
        let mut tensors = vec![
            tensor("model.visual.blocks.0.attn.qkv.weight"),
            tensor("model.vision_tower.x.weight"),
            tensor("model.layers.0.mlp.up_proj.weight"),
            tensor("mlp.0.weight"),
            tensor("model.mlp_proj.weight"),
        ];
        let mut warnings = Vec::new();
        apply_global_filters(&mut tensors, &converter, &mut warnings).unwrap();
        let names: Vec<&str> = tensors.iter().map(|tensor| tensor.name.as_str()).collect();
        assert_eq!(names, ["model.layers.0.mlp.up_proj.weight"]);
        converter.global_filters_after.clear();
        let mut kept = vec![tensor("model.visual.x")];
        apply_global_filters(&mut kept, &converter, &mut warnings).unwrap();
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn generated_tensors_follow_gguf_035() {
        assert!(crate::dtype::GENERATED_TENSORS.len() >= 26);
        // Every converter's generator has a row.
        for arch in gguf_names::architecture_ids() {
            for record in gguf_names::converter_records(arch) {
                if record.generator.is_some() {
                    assert!(generated_row(&record).is_some(), "{}", record.class);
                }
            }
        }
        let config = |text: &str| serde_json::from_str::<Value>(text).unwrap().as_object().unwrap().clone();
        let llama = "R.rope_type lowercased == 'llama3'";
        assert_eq!(generated_condition(llama, Some(&config(r#"{"rope_scaling": null}"#))), Some(false));
        assert_eq!(generated_condition(llama, None), Some(false));
        assert_eq!(
            generated_condition(llama, Some(&config(r#"{"rope_scaling": {"type": "Llama3", "factor": 8}}"#))),
            Some(true)
        );
        assert_eq!(
            generated_condition(llama, Some(&config(r#"{"rope_parameters": {"full_attention": {"rope_type": "llama3"}}}"#))),
            Some(true)
        );
        let phi = "P.long_factor is non-empty; also writes the rope_scaling.attn_factor KV";
        assert_eq!(generated_condition(phi, Some(&config(r#"{"rope_scaling": {"long_factor": []}}"#))), Some(false));
        assert_eq!(generated_condition(phi, Some(&config(r#"{"rope_scaling": {"long_factor": [1.0]}}"#))), Some(true));
        assert_eq!(generated_condition("never", None), Some(false));
        assert_eq!(generated_condition("1_Dense/model.safetensors exists", None), None);

        let llama_record = gguf_names::converter_records("llama")
            .into_iter()
            .find(|record| record.class == "LlamaModel")
            .unwrap();
        let mut notes = Vec::new();
        extra_tensor_note(&llama_record, Some(&config(r#"{"rope_scaling": null}"#)), &mut notes);
        assert!(notes.is_empty(), "{notes:?}");
        extra_tensor_note(&llama_record, Some(&config(r#"{"rope_scaling": {"rope_type": "llama3"}}"#)), &mut notes);
        assert!(notes[0].contains("rope_freqs.weight"), "{notes:?}");

        let gpt_oss = gguf_names::converter_records("gpt-oss")
            .into_iter()
            .find(|record| record.class == "GptOssModel");
        if let Some(record) = gpt_oss {
            let err = check_hooks("gpt-oss", &record).unwrap_err().to_string();
            assert!(err.contains("generate_extra_tensors"), "{err}");
        }
    }
}
