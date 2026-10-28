//! tensinfo: what a tensor container holds, for a person at the console.
//! The default is a short overview of the model; the aligned tensor table,
//! the full metadata, and a JSON document are asked for with options.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::archive::{self, Archive, Member};
use crate::cli::{FormatName, DEFAULT_MAX_HEADER};
use crate::error::{msg, Result};

pub struct Request {
    pub file: PathBuf,
    pub force_single: bool,
    pub format: Option<FormatName>,
    pub max_header: u64,
    pub tensors: bool,
    pub fold: bool,
    pub sort: bool,
    pub metadata: bool,
    pub json: bool,
    pub keys: Vec<String>,
}

enum InfoCommand {
    Help,
    Version,
    Run(Request),
}

pub fn info_main(args: &[String]) -> Result<i32> {
    match parse_info(args)? {
        InfoCommand::Help => {
            outln!("{}", info_help());
            Ok(0)
        }
        InfoCommand::Version => {
            outln!("{}", crate::version_line("tensinfo"));
            Ok(0)
        }
        InfoCommand::Run(request) => {
            run(&request)?;
            Ok(0)
        }
    }
}

fn parse_info(args: &[String]) -> Result<InfoCommand> {
    let mut request = Request {
        file: PathBuf::new(),
        force_single: false,
        format: None,
        max_header: DEFAULT_MAX_HEADER,
        tensors: false,
        fold: false,
        sort: false,
        metadata: false,
        json: false,
        keys: Vec::new(),
    };
    let mut file = None;
    let mut force_next = false;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        let mut value = |name: &str| -> Result<String> {
            index += 1;
            args.get(index).cloned().ok_or_else(|| msg(format!("{name} needs a value")))
        };
        match arg {
            "-h" | "--help" => return Ok(InfoCommand::Help),
            "--version" => return Ok(InfoCommand::Version),
            "-t" | "--tensors" => request.tensors = true,
            "--fold" => request.fold = true,
            "--sort" => request.sort = true,
            "-m" | "--metadata" => request.metadata = true,
            "--json" => request.json = true,
            "--force-single-file" => force_next = true,
            "--format" => request.format = Some(FormatName::parse(&value("--format")?)?),
            "--max-header-size" => request.max_header = crate::cli::parse_size(&value("--max-header-size")?)?,
            _ if arg.starts_with('-') && arg.len() > 1 => return Err(msg(format!("unknown option {arg}"))),
            _ if file.is_none() => {
                file = Some(PathBuf::from(arg));
                request.force_single = force_next;
                force_next = false;
            }
            _ => request.keys.push(arg.to_string()),
        }
        index += 1;
    }
    request.file = file.ok_or_else(|| msg("usage: tensinfo [OPTIONS] FILE [KEY ...]"))?;
    if request.json && (request.fold || request.tensors || request.metadata) {
        return Err(msg("--json already holds every tensor and metadata key; it does not take -t, -m, or --fold"));
    }
    if request.fold || request.sort || !request.keys.is_empty() {
        request.tensors = true;
    }
    Ok(InfoCommand::Run(request))
}

pub fn info_help() -> &'static str {
    "\
tensinfo — what a tensor container holds

Usage:
  tensinfo [OPTIONS] FILE [KEY ...]

With no options, a short overview: the format and size, the model type,
the parameter count, the dtype mix, the layer layout, the vocabulary, any
quantization, and whether a tokenizer is present. FILE is a file or a
model directory.

Options:
  -t, --tensors             Add the aligned tensor table. KEY ... limits it to those tensors.
  --fold                    Merge a consecutive numeric name segment in the
                            table: model.layers.[0-29] is one copy per layer.
                            A gap stays on its own line. Implies -t.
  --sort                    Sort the table by name. Name segments split at dots;
                            a segment of digits sorts by its number. Implies -t.
  -m, --metadata            List every metadata key. Long values are shortened.
  --json                    One JSON document: the overview, every metadata key,
                            and every tensor.
  --format NAME             Container to read when the file does not say.
  --max-header-size SIZE    Safetensors header cap (default 100M).
  --force-single-file PATH  Read this .safetensors file even when config.json
                            is in the same directory.
  -h, --help                Show this help.
  --version                 Show the version.

Examples:
  tensinfo model/
  tensinfo -t --fold model.gguf
  tensinfo -m model.gguf
  tensinfo --json model.safetensors
"
}

fn run(request: &Request) -> Result<()> {
    let mut archive = archive::open_user(
        &request.file,
        request.force_single,
        request.format,
        request.max_header,
        None,
        None,
    )?;
    let dir = config_dir(&request.file, &archive);
    if let Some(note) = quantization_note(&dir) {
        if !archive.notes.contains(&note) {
            archive.notes.push(note);
        }
    }
    let mut members = select(&archive, &request.keys)?;
    if request.sort {
        sort_members(&mut members);
    }
    let overview = overview(&archive, &dir);
    if request.json {
        outln!("{}", json_document(&archive, &members, Some(&overview))?);
        return Ok(());
    }
    let width = overview.iter().map(|(label, _)| label.len()).max().unwrap_or(0) + 2;
    for (label, value) in &overview {
        outln!("{:<width$}{value}", format!("{label}:"));
    }
    if request.metadata {
        outln!("");
        if archive.metadata.is_empty() {
            outln!("metadata: none");
        } else {
            outln!("metadata:");
            for (key, value) in &archive.metadata {
                outln!("  {key}={}", short_value(value));
            }
        }
    }
    if request.tensors {
        let text = crate::listing::render_columns(&members, request.fold);
        if !text.is_empty() {
            outln!("");
            outln!("{text}");
        }
    }
    Ok(())
}

/// The directory whose config.json and tokenizer files describe the model:
/// the directory given, or the one holding the file.
fn config_dir(given: &Path, archive: &Archive) -> PathBuf {
    if given.is_dir() {
        return given.to_path_buf();
    }
    archive.path.parent().map(Path::to_path_buf).unwrap_or_default()
}

fn select<'a>(archive: &'a Archive, names: &[String]) -> Result<Vec<&'a Member>> {
    if names.is_empty() {
        return Ok(archive.members.iter().collect());
    }
    names
        .iter()
        .map(|name| {
            archive
                .members
                .iter()
                .find(|member| &member.name == name)
                .ok_or_else(|| msg(format!("no member {name}")))
        })
        .collect()
}

pub(crate) fn sort_members(members: &mut [&Member]) {
    members.sort_by(|left, right| crate::listing::name_order(&left.name, &right.name));
}

/// The overview lines, in order. A line whose source is missing is left out.
fn overview(archive: &Archive, dir: &Path) -> Vec<(&'static str, String)> {
    let meta: BTreeMap<&str, &str> = archive
        .metadata
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect();
    let config = read_config(dir);
    let mut lines = Vec::new();
    // tenspile's version labels (header+blob, index, zip) say how the file is
    // read; GGUF's is the file's own format version.
    let format = match archive.format.as_str() {
        "gguf" => format!("gguf v{}", archive.version),
        other => other.to_string(),
    };
    lines.push((
        "file",
        format!(
            "{} ({format}), {} of tensor data",
            archive.path.display(),
            crate::listing::human_bytes(archive.data_bytes)
        ),
    ));
    let gguf_arch = meta.get("general.architecture").copied();
    if let Some(arch) = gguf_arch {
        let mut text = arch.to_string();
        let name: Vec<&str> = ["general.name", "general.size_label"]
            .iter()
            .filter_map(|key| meta.get(key).copied())
            .collect();
        if !name.is_empty() {
            text.push_str(&format!(" ({})", name.join(", ")));
        }
        lines.push(("model", text));
    } else if let Some(config) = &config {
        let model_type = config.get("model_type").and_then(Value::as_str);
        let class = config
            .get("architectures")
            .and_then(Value::as_array)
            .and_then(|items| items.first())
            .and_then(Value::as_str);
        match (model_type, class) {
            (Some(kind), Some(class)) => lines.push(("model", format!("{kind} ({class})"))),
            (Some(text), None) | (None, Some(text)) => lines.push(("model", text.to_string())),
            (None, None) => {}
        }
    }
    let params: u64 = archive.members.iter().filter_map(|member| member.numel).sum();
    let packed = archive
        .members
        .iter()
        .any(|member| member.detail.as_deref().is_some_and(|detail| detail.contains("pack")));
    lines.push((
        "parameters",
        format!(
            "{} in {} tensors{}",
            crate::listing::human_count(params),
            archive.members.len(),
            if packed { ", counting packed values as stored" } else { "" }
        ),
    ));
    lines.push(("dtypes", dtype_mix(archive, params)));
    let layout = match gguf_arch {
        Some(arch) => {
            let get = |key: &str| meta.get(format!("{arch}.{key}").as_str()).map(|value| value.to_string());
            Layout {
                layers: get("block_count"),
                hidden: get("embedding_length"),
                heads: get("attention.head_count"),
                kv_heads: get("attention.head_count_kv"),
                context: get("context_length"),
                experts: get("expert_count"),
                experts_used: get("expert_used_count"),
            }
        }
        None => match &config {
            Some(config) => {
                // The names hf_config_to_gguf lists for each row.
                let get = |names: &[&str]| crate::gguf_keys::first_present(config, names).map(json_text);
                Layout {
                    layers: get(&["n_layers", "num_hidden_layers", "n_layer", "num_layers"]),
                    hidden: get(&["hidden_size", "n_embd", "dim"]),
                    heads: get(&["num_attention_heads", "n_head", "n_heads"]),
                    kv_heads: get(&["num_key_value_heads", "n_kv_heads"]),
                    context: get(&[
                        "max_position_embeddings",
                        "n_ctx",
                        "n_positions",
                        "max_length",
                        "max_sequence_length",
                        "model_max_length",
                    ]),
                    experts: get(&["num_local_experts", "num_experts", "n_routed_experts"]),
                    experts_used: get(&["num_experts_per_tok", "num_experts_per_token", "top_k_experts"]),
                }
            }
            None => Layout::default(),
        },
    };
    if let Some(text) = layout.text() {
        lines.push(("layers", text));
    }
    if let Some(vocab) = vocabulary(archive, &meta, gguf_arch, config.as_ref()) {
        lines.push(("vocabulary", vocab));
    }
    lines.push(("quantization", quantization(archive, packed)));
    lines.push(("tokenizer", tokenizer(&meta, gguf_arch.is_some(), dir)));
    if let Some(tied) = tied(archive, config.as_ref()) {
        lines.push(("embeddings", tied));
    }
    for note in archive.notes.iter().filter(|note| overview_note(note)) {
        lines.push(("note", note.clone()));
    }
    lines
}

/// Notes that help say what the model is. Those about how the file is laid
/// out, the other files in the directory, or the quantization line's own
/// source stay in `tenspile -i` and `--json`.
fn overview_note(note: &str) -> bool {
    const SKIPPED: [&str; 5] = ["directory ", "alignment ", "data starts at ", "shapes are ggml", "quantization "];
    !SKIPPED.iter().any(|prefix| note.starts_with(prefix))
}

#[derive(Default)]
struct Layout {
    layers: Option<String>,
    hidden: Option<String>,
    heads: Option<String>,
    kv_heads: Option<String>,
    context: Option<String>,
    experts: Option<String>,
    experts_used: Option<String>,
}

impl Layout {
    fn text(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(layers) = &self.layers {
            parts.push(layers.clone());
        }
        if let Some(hidden) = &self.hidden {
            parts.push(format!("hidden {hidden}"));
        }
        if let Some(heads) = &self.heads {
            match self.kv_heads.as_ref().filter(|kv| *kv != heads) {
                Some(kv) => parts.push(format!("heads {heads} ({kv} kv)")),
                None => parts.push(format!("heads {heads}")),
            }
        }
        if let Some(context) = &self.context {
            parts.push(format!("context {context}"));
        }
        if let Some(experts) = &self.experts {
            match &self.experts_used {
                Some(used) => parts.push(format!("experts {experts} ({used} used)")),
                None => parts.push(format!("experts {experts}")),
            }
        }
        (!parts.is_empty()).then(|| parts.join(", "))
    }
}

fn json_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// config.json beside the weights, with text_config merged over the top
/// level so a multimodal model's language part is read.
fn read_config(dir: &Path) -> Option<Map<String, Value>> {
    let text = fs::read_to_string(dir.join("config.json")).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    let mut merged = value.as_object()?.clone();
    if let Some(text) = value.get("text_config").and_then(Value::as_object) {
        for (key, item) in text {
            merged.insert(key.clone(), item.clone());
        }
    }
    Some(merged)
}

/// The dtypes as shares of the parameters, largest first.
fn dtype_mix(archive: &Archive, params: u64) -> String {
    let mut totals: BTreeMap<&str, u64> = BTreeMap::new();
    for member in &archive.members {
        *totals.entry(crate::dtype::display_name(&member.dtype)).or_default() += member.numel.unwrap_or(0);
    }
    if totals.len() == 1 {
        return format!("{} (all)", totals.keys().next().copied().unwrap_or("?"));
    }
    let mut shares: Vec<(&str, u64)> = totals.into_iter().collect();
    shares.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(right.0)));
    shares
        .iter()
        .map(|(dtype, count)| {
            let percent = if params == 0 { 0.0 } else { *count as f64 * 100.0 / params as f64 };
            if percent > 0.0 && percent < 1.0 {
                format!("{dtype} <1%")
            } else {
                format!("{dtype} {percent:.0}%")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn vocabulary(
    archive: &Archive,
    meta: &BTreeMap<&str, &str>,
    gguf_arch: Option<&str>,
    config: Option<&Map<String, Value>>,
) -> Option<String> {
    if let Some(arch) = gguf_arch {
        if let Some(size) = meta.get(format!("{arch}.vocab_size").as_str()) {
            return Some(size.to_string());
        }
        if let Some(tokens) = meta.get("tokenizer.ggml.tokens") {
            if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(tokens) {
                return Some(items.len().to_string());
            }
        }
        // GGUF's token_embd is [vocab, hidden] in canonical order, stored ne[] reversed.
        return archive
            .members
            .iter()
            .find(|member| member.name == "token_embd.weight")
            .and_then(|member| member.shape.last().map(u64::to_string));
    }
    config
        .and_then(|config| config.get("vocab_size"))
        .filter(|value| !value.is_null())
        .map(json_text)
}

/// GGUF block types present, or the pack a Hugging Face directory names.
fn quantization(archive: &Archive, packed: bool) -> String {
    let mut blocks: Vec<&str> = archive
        .members
        .iter()
        .filter(|member| {
            crate::dtype::canonical_id(&member.dtype)
                .ok()
                .and_then(crate::dtype::canon_kind)
                == Some("block")
        })
        .map(|member| member.dtype.as_str())
        .collect();
    blocks.sort_unstable();
    blocks.dedup();
    if !blocks.is_empty() {
        return format!("GGUF blocks {}", blocks.join(", "));
    }
    if let Some(note) = archive.notes.iter().find(|note| note.starts_with("quantization ")) {
        return note.trim_start_matches("quantization ").to_string();
    }
    if packed {
        let pack = archive
            .members
            .iter()
            .find_map(|member| member.detail.as_deref().filter(|detail| detail.contains("pack")))
            .map(|detail| detail.split(';').next().unwrap_or(detail))
            .unwrap_or("a pack");
        return format!("{pack}, without a quantization_config to read it");
    }
    "none".to_string()
}

fn tokenizer(meta: &BTreeMap<&str, &str>, gguf: bool, dir: &Path) -> String {
    if gguf {
        return match meta.get("tokenizer.ggml.model") {
            Some(model) => format!("{model} (tokenizer.ggml.* keys)"),
            None => "none: no tokenizer.ggml.model, so llama.cpp will not load the file".to_string(),
        };
    }
    let found: Vec<&str> = ["tokenizer.json", "tokenizer.model"]
        .into_iter()
        .filter(|name| dir.join(name).is_file())
        .collect();
    if found.is_empty() {
        "none beside the weights".to_string()
    } else {
        format!("{} beside the weights", found.join(" and "))
    }
}

/// Whether the output layer reuses the input embedding.
fn tied(archive: &Archive, config: Option<&Map<String, Value>>) -> Option<String> {
    let has = |name: &str| archive.members.iter().any(|member| member.name == name);
    if has("token_embd.weight") {
        return (!has("output.weight")).then(|| "the output layer shares token_embd".to_string());
    }
    match config.and_then(|config| config.get("tie_word_embeddings")) {
        Some(Value::Bool(true)) => Some("tied input and output (tie_word_embeddings)".to_string()),
        _ => None,
    }
}

pub(crate) fn quantization_note(dir: &Path) -> Option<String> {
    let bytes = fs::read(dir.join("config.json")).ok()?;
    let value: Value = serde_json::from_slice(&bytes).ok()?;
    if let Some(object) = value.get("quantization_config").and_then(Value::as_object) {
        if let Some(method) = object.get("quant_method").and_then(Value::as_str) {
            return Some(quantization_note_text(
                method,
                json_u64(object, &["bits", "w_bit"]),
                json_u64(object, &["group_size", "q_group_size"]),
            ));
        }
    }
    let object = value.get("quantization").and_then(Value::as_object)?;
    if !object.contains_key("bits") && !object.contains_key("group_size") {
        return None;
    }
    Some(quantization_note_text(
        "mlx",
        object.get("bits").and_then(Value::as_u64),
        object.get("group_size").and_then(Value::as_u64),
    ))
}

fn json_u64(object: &Map<String, Value>, keys: &[&str]) -> Option<u64> {
    keys.iter().find_map(|key| object.get(*key).and_then(Value::as_u64))
}

fn quantization_note_text(method: &str, bits: Option<u64>, group: Option<u64>) -> String {
    let mut text = format!("quantization {method}");
    if let Some(bits) = bits {
        text.push_str(&format!(" bits {bits}"));
    }
    if let Some(group) = group {
        text.push_str(&format!(" group {group}"));
    }
    text
}

/// A metadata value for the text listing: long values such as a GGUF
/// vocabulary are cut to their first 120 characters. `--json` prints them whole.
/// A cut array gives its item count, then the length of its full JSON text.
pub(crate) fn short_value(value: &str) -> String {
    const SHOWN: usize = 120;
    let Some((end, _)) = value.char_indices().nth(SHOWN) else {
        return value.to_string();
    };
    let items = value
        .starts_with('[')
        .then(|| serde_json::from_str::<Vec<Value>>(value).ok())
        .flatten();
    match items {
        Some(items) => format!(
            "{}… ({} items; {} bytes as JSON)",
            &value[..end],
            items.len(),
            value.len()
        ),
        None => format!("{}… ({} bytes)", &value[..end], value.len()),
    }
}

/// The JSON document: the archive, its metadata and members, and the
/// overview when there is one.
pub(crate) fn json_document(
    archive: &Archive,
    members: &[&Member],
    overview: Option<&[(&'static str, String)]>,
) -> Result<String> {
    let mut metadata = Map::new();
    for (key, value) in &archive.metadata {
        metadata.insert(key.clone(), Value::String(value.clone()));
    }
    let optional = |value: &Option<String>| match value {
        Some(text) => Value::String(text.clone()),
        None => Value::Null,
    };
    let listed: Vec<Value> = members
        .iter()
        .map(|member| {
            let mut object = Map::new();
            object.insert("name".into(), Value::String(member.name.clone()));
            object.insert(
                "dtype".into(),
                Value::String(crate::dtype::display_name(&member.dtype).to_string()),
            );
            object.insert(
                "shape".into(),
                Value::Array(member.shape.iter().copied().map(Value::from).collect()),
            );
            object.insert("bytes".into(), Value::from(member.nbytes));
            object.insert("offset".into(), Value::from(member.file_offset));
            object.insert("numel".into(), member.numel.map_or(Value::Null, Value::from));
            object.insert("shard".into(), optional(&member.shard));
            object.insert("detail".into(), optional(&member.detail));
            Value::Object(object)
        })
        .collect();
    let mut root = Map::new();
    root.insert("format".into(), Value::String(archive.format.as_str().into()));
    root.insert("version".into(), Value::String(archive.version.clone()));
    root.insert("path".into(), Value::String(archive.path.display().to_string()));
    if let Some(overview) = overview {
        let mut lines = Map::new();
        let params: u64 = archive.members.iter().filter_map(|member| member.numel).sum();
        lines.insert("parameters_exact".into(), Value::from(params));
        // The notes are the document's own "notes" array.
        for (label, value) in overview.iter().filter(|(label, _)| *label != "note") {
            lines.insert((*label).into(), Value::String(value.clone()));
        }
        root.insert("overview".into(), Value::Object(lines));
    }
    root.insert("members".into(), Value::Array(listed));
    root.insert("metadata".into(), Value::Object(metadata));
    root.insert("data_bytes".into(), Value::from(archive.data_bytes));
    root.insert(
        "notes".into(),
        Value::Array(archive.notes.iter().cloned().map(Value::String).collect()),
    );
    Ok(serde_json::to_string_pretty(&Value::Object(root))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cut_array_gives_its_item_count() {
        let long: Vec<String> = (0..200).map(|n| n.to_string()).collect();
        let value = serde_json::to_string(&long).unwrap();
        let shown = short_value(&value);
        assert!(shown.ends_with(&format!("(200 items; {} bytes as JSON)", value.len())), "{shown}");
        let text = "x".repeat(130);
        assert_eq!(short_value(&text), format!("{}… (130 bytes)", "x".repeat(120)));
    }
}
