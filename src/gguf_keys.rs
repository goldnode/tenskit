//! The GGUF hyperparameter keys that tensconv --rename writes from
//! config.json. The base rows transcribe TextModel.set_gguf_parameters
//! (metadata-mapping hf_config_to_gguf).

use serde_json::{Map, Value};

use crate::error::{msg, Result};
use crate::formats::gguf::MetaValue;

/// metadata-mapping hf_config_to_gguf rope_parameters_fill. (1) P is
/// rope_parameters when that key is present, else rope_scaling; null gives {}.
/// (2) Only when P has neither full_attention nor sliding_attention: the local
/// theta keys set P.sliding_attention.rope_theta; rope_theta,
/// partial_rotary_factor, and original_max_position_embeddings are filled
/// from the top level; rope_type is copied from a non-null type. A key already
/// in P is never replaced, and a null found value fills nothing. (3) A nested
/// P is not filled. R is P.full_attention when present, else P.
pub(crate) fn rope_params(hparams: Option<&Map<String, Value>>) -> (Map<String, Value>, Map<String, Value>) {
    let empty = Map::new();
    let map = hparams.unwrap_or(&empty);
    let source = match map.get("rope_parameters") {
        Some(value) => Some(value),
        None => map.get("rope_scaling"),
    };
    let mut p = source.and_then(Value::as_object).cloned().unwrap_or_default();
    if !p.contains_key("full_attention") && !p.contains_key("sliding_attention") {
        if let Some(theta) = first_present(map, LOCAL_ROPE_THETA) {
            let mut sliding = Map::new();
            sliding.insert("rope_theta".into(), theta.clone());
            p.insert("sliding_attention".into(), Value::Object(sliding));
        }
        if !p.contains_key("rope_theta") {
            if let Some(theta) = first_present(map, ROPE_THETA) {
                p.insert("rope_theta".into(), theta.clone());
            }
        }
        if !p.contains_key("rope_type") {
            if let Some(kind) = p.get("type").filter(|kind| !kind.is_null()).cloned() {
                p.insert("rope_type".into(), kind);
            }
        }
        if !p.contains_key("partial_rotary_factor") {
            if let Some(factor) = first_present(map, &["partial_rotary_factor", "rope_pct", "rope_percent"]) {
                p.insert("partial_rotary_factor".into(), factor.clone());
            }
        }
        if !p.contains_key("original_max_position_embeddings") {
            if let Some(original) = map.get("original_max_position_embeddings").filter(|value| !value.is_null()) {
                p.insert("original_max_position_embeddings".into(), original.clone());
            }
        }
    }
    let r = p
        .get("full_attention")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_else(|| p.clone());
    (p, r)
}

static DEFAULTS_JSON: &str = include_str!("../tensdataspecs_readonly/specs/tables/hf-config-defaults.json");

fn defaults_table() -> &'static Value {
    static TABLE: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| serde_json::from_str(DEFAULTS_JSON).expect("hf-config-defaults.json parses"))
}

/// GGUF-039 and GGUF-040: the defaults transformers' to_dict() adds, as
/// tables/hf-config-defaults.json lists them for 14 model_types, with the
/// derived-key formulas GGUF-039 gives. A default applies only when
/// config.json lacks the key. For another model_type only the common keys
/// are added, and the returned note says so.
pub(crate) fn listed_defaults(top: &mut Map<String, Value>) -> Option<String> {
    let table = defaults_table();
    if let Some(dtype) = top.remove("torch_dtype") {
        top.entry("dtype").or_insert(dtype);
    }
    let given = top.clone();
    let model_type = given.get("model_type").and_then(Value::as_str).unwrap_or("").to_string();
    let entry = table["model_types"].get(model_type.as_str());
    if let Some(entry) = entry {
        let derived = entry["derived"].as_object();
        for (key, value) in entry["defaults"].as_object().into_iter().flatten() {
            if !given.contains_key(key) && !derived.is_some_and(|derived| derived.contains_key(key)) {
                top.insert(key.clone(), value.clone());
            }
        }
        derived_keys(&model_type, &given, top, derived);
        if entry["rope_scaling_type_copied_to_rope_type"] == Value::Bool(true) {
            if let Some(scaling) = top.get_mut("rope_scaling").and_then(Value::as_object_mut) {
                if !scaling.contains_key("rope_type") {
                    if let Some(kind) = scaling.get("type").cloned() {
                        scaling.insert("rope_type".into(), kind);
                    }
                }
            }
        }
    }
    for (key, value) in table["common"].as_object().into_iter().flatten() {
        top.entry(key.clone()).or_insert_with(|| value.clone());
    }
    match entry {
        Some(_) => None,
        None => Some(format!(
            "model_type {} is not in tables/hf-config-defaults.json, so only the common transformers defaults were added (GGUF-040)",
            if model_type.is_empty() { "(none)" } else { model_type.as_str() }
        )),
    }
}

/// GGUF-040's notes for a config.json: a model_type, top-level or of the
/// text_config, that the defaults table lacks; and a composite type that
/// transformers gives a text_config built from the top-level keys when
/// config.json has none, which tenspile does not build.
pub(crate) fn defaults_notes(config: &Value) -> Vec<String> {
    let table = defaults_table();
    let mut notes = Vec::new();
    let top_type = config.get("model_type").and_then(Value::as_str).unwrap_or("");
    let text = config.get("text_config").filter(|text| text.is_object());
    let mut unlisted = |model_type: &str| {
        if table["model_types"].get(model_type).is_none() {
            let mut scratch = Map::new();
            scratch.insert("model_type".into(), Value::from(model_type));
            notes.extend(listed_defaults(&mut scratch));
        }
    };
    unlisted(top_type);
    if let Some(text_type) = text.and_then(|text| text.get("model_type")).and_then(Value::as_str) {
        unlisted(text_type);
    }
    let composite = &table["composite_text_config"][top_type];
    if text.is_none() && composite["created_when_absent"] == Value::Bool(true) {
        let text_type = composite["default_text_model_type"].as_str().unwrap_or("?");
        if table["model_types"].get(text_type).is_none() {
            notes.push(format!(
                "model_type {top_type} gets a {text_type} text_config, built from its defaults; {text_type} \
                 is not in tables/hf-config-defaults.json, so its derived keys may differ from llama.cpp's (GGUF-040)"
            ));
        }
    }
    notes
}

/// The composites that build text_config from every top-level key their
/// own __init__ does not take, and the keys each takes (GGUF-039).
fn composite_takes(top_type: &str) -> Option<&'static [&'static str]> {
    const QWEN: &[&str] = &[
        "text_config", "vision_config", "image_token_id", "video_token_id", "vision_start_token_id",
        "vision_end_token_id", "model_type",
    ];
    const GLM: &[&str] = &[
        "text_config", "vision_config", "image_token_id", "video_token_id", "image_start_token_id",
        "image_end_token_id", "video_start_token_id", "video_end_token_id", "model_type",
    ];
    match top_type {
        "qwen2_vl" | "qwen2_5_vl" => Some(QWEN),
        "glm4v" | "glm4v_moe" => Some(GLM),
        _ => None,
    }
}

/// GGUF-039 and GGUF-040: the text_config transformers builds for a
/// composite type when config.json has none. It starts from the top-level
/// keys the composite passes on (every key it does not take, or its
/// copied_top_level_keys), then the text model_type's defaults and derived
/// formulas, then default_text_config for anything still missing.
pub(crate) fn built_text_config(raw: &Map<String, Value>) -> Option<Map<String, Value>> {
    let table = defaults_table();
    let top_type = raw.get("model_type").and_then(Value::as_str)?;
    let composite = &table["composite_text_config"][top_type];
    if composite["created_when_absent"] != Value::Bool(true) {
        return None;
    }
    let text_type = composite["default_text_model_type"].as_str()?;
    let mut built = Map::new();
    match composite_takes(top_type) {
        Some(taken) => {
            for (key, value) in raw {
                if !taken.contains(&key.as_str()) {
                    built.insert(key.clone(), value.clone());
                }
            }
        }
        None => {
            for key in composite["copied_top_level_keys"].as_array().into_iter().flatten().filter_map(Value::as_str) {
                if let Some(value) = raw.get(key) {
                    built.insert(key.to_string(), value.clone());
                }
            }
        }
    }
    built.insert("model_type".into(), Value::from(text_type));
    if table["model_types"].get(text_type).is_some() {
        let _ = listed_defaults(&mut built);
    }
    for (key, value) in composite["default_text_config"].as_object().into_iter().flatten() {
        built.entry(key.clone()).or_insert_with(|| value.clone());
    }
    Some(built)
}

/// GGUF-039's formulas for the keys the table marks derived.
fn derived_keys(
    model_type: &str,
    given: &Map<String, Value>,
    top: &mut Map<String, Value>,
    derived: Option<&Map<String, Value>>,
) {
    let int = |top: &Map<String, Value>, key: &str| top.get(key).and_then(Value::as_u64);
    // The value a derived key has before its formula: as given, else the table's.
    let start = |key: &str| -> Option<Value> {
        given
            .get(key)
            .cloned()
            .or_else(|| derived.and_then(|derived| derived.get(key)).map(|row| row["value_when_all_absent"].clone()))
    };
    let absent_or_null = |key: &str| given.get(key).is_none_or(Value::is_null);
    if model_type == "llama" && absent_or_null("head_dim") {
        if let (Some(hidden), Some(heads)) = (int(top, "hidden_size"), int(top, "num_attention_heads").filter(|heads| *heads > 0)) {
            top.insert("head_dim".into(), Value::from(hidden / heads));
        }
    }
    let kv_when_null = ["llama", "mistral", "mixtral", "qwen2", "qwen3", "phi3", "smollm3", "olmo2", "granite"];
    let kv_when_absent = ["llama", "phi3", "olmo2", "granite"];
    let kv = given.get("num_key_value_heads");
    if (kv == Some(&Value::Null) && kv_when_null.contains(&model_type))
        || (kv.is_none() && kv_when_absent.contains(&model_type))
    {
        if let Some(heads) = top.get("num_attention_heads").cloned() {
            top.insert("num_key_value_heads".into(), heads);
        }
    }
    if matches!(model_type, "qwen2" | "qwen2_moe" | "qwen3" | "qwen3_moe") {
        let window = if top.get("use_sliding_window") == Some(&Value::Bool(true)) {
            given.get("sliding_window").cloned().unwrap_or(Value::from(4096))
        } else {
            Value::Null
        };
        top.insert("sliding_window".into(), window);
    }
    let layers = int(top, "num_hidden_layers").unwrap_or(0);
    let kind = |sliding: bool| Value::from(if sliding { "sliding_attention" } else { "full_attention" });
    let keep_layer_types = given.contains_key("layer_types");
    match model_type {
        "qwen2" | "qwen3" if !keep_layer_types => {
            let windowed = top.get("sliding_window").is_some_and(|value| !value.is_null());
            let from = int(top, "max_window_layers").unwrap_or(0);
            let types = (0..layers).map(|i| kind(windowed && i >= from)).collect();
            top.insert("layer_types".into(), Value::Array(types));
        }
        "gemma2" if !keep_layer_types => {
            let types = (0..layers).map(|i| kind((i + 1) % 2 != 0)).collect();
            top.insert("layer_types".into(), Value::Array(types));
        }
        "gemma3_text" => {
            let pattern = given.get("sliding_window_pattern").and_then(Value::as_u64).unwrap_or(6);
            top.insert("_sliding_window_pattern".into(), Value::from(pattern));
            if !keep_layer_types {
                let types = (0..layers).map(|i| kind(pattern != 0 && (i + 1) % pattern != 0)).collect();
                top.insert("layer_types".into(), Value::Array(types));
            }
            let window = start("sliding_window");
            let window = match (top.get("use_bidirectional_attention"), window.as_ref().and_then(Value::as_u64)) {
                (Some(Value::Bool(true)), Some(window)) => Some(Value::from(window / 2 + 1)),
                _ => window,
            };
            if let Some(window) = window {
                top.insert("sliding_window".into(), window);
            }
        }
        "smollm3" => {
            let interval = int(top, "no_rope_layer_interval").unwrap_or(4).max(1);
            let no_rope: Vec<u64> = match given.get("no_rope_layers").and_then(Value::as_array) {
                Some(items) => items.iter().map(|item| item.as_u64().unwrap_or(0)).collect(),
                None => (0..layers).map(|i| u64::from((i + 1) % interval != 0)).collect(),
            };
            if !given.contains_key("no_rope_layers") {
                top.insert("no_rope_layers".into(), Value::from(no_rope.clone()));
            }
            if !keep_layer_types {
                let windowed = top.get("use_sliding_window") == Some(&Value::Bool(true))
                    && top.get("sliding_window").is_some_and(|value| !value.is_null());
                let types = (0..layers as usize)
                    .map(|i| kind(windowed && no_rope.get(i).copied() == Some(0)))
                    .collect();
                top.insert("layer_types".into(), Value::Array(types));
            }
        }
        _ => {}
    }
}

const ROPE_THETA: &[&str] = &[
    "global_rope_theta",
    "rope_global_theta",
    "rope_theta_global",
    "rope_theta",
    "rotary_emb_base",
];

const LOCAL_ROPE_THETA: &[&str] = &[
    "local_rope_theta",
    "rope_local_theta",
    "rope_theta_local",
    "swa_rope_theta",
    "rope_local_base_freq",
];

const GATING: &[&str] = &[
    "score_function",
    "scoring_func",
    "score_func",
    "moe_router_activation",
    "moe_router_activation_func",
    "expert_selection_fn",
    "router_score_func",
    "router_score_function",
];

/// find_hparam(names): the value of the first listed name that is a key of
/// the dict. A null value counts as present and stops the search
/// (hf_config_to_gguf description).
pub(crate) fn lookup<'a>(map: &'a Map<String, Value>, names: &[&str]) -> Option<&'a Value> {
    names.iter().find_map(|name| map.get(*name))
}

/// find_hparam(names, optional=True) guarded by `is not None`: the lookup's
/// value unless it is null.
pub(crate) fn first_present<'a>(map: &'a Map<String, Value>, names: &[&str]) -> Option<&'a Value> {
    lookup(map, names).filter(|value| !value.is_null())
}

/// Replace a key already written, as GGUFWriter.add_key_value does, or add it.
pub(crate) fn set(out: &mut Vec<(String, MetaValue)>, key: String, value: MetaValue) {
    match out.iter_mut().find(|(seen, _)| *seen == key) {
        Some(slot) => slot.1 = value,
        None => out.push((key, value)),
    }
}

pub(crate) fn as_u32(key: &str, value: &Value) -> Result<MetaValue> {
    value
        .as_u64()
        .and_then(|number| u32::try_from(number).ok())
        .map(MetaValue::U32)
        .ok_or_else(|| msg(format!("config value {value} for {key} is not a u32")))
}

pub(crate) fn as_f32(key: &str, value: &Value) -> Result<MetaValue> {
    value
        .as_f64()
        .map(|number| MetaValue::F32(number as f32))
        .ok_or_else(|| msg(format!("config value {value} for {key} is not a number")))
}

/// The hf_config_to_gguf {arch}.* rows: TextModel.set_gguf_parameters.
pub(crate) fn base_rows(arch: &str, map: &Map<String, Value>, out: &mut Vec<(String, MetaValue)>) -> Result<()> {
    let key = |name: &str| format!("{arch}.{name}");
    let u32_row = |out: &mut Vec<(String, MetaValue)>, name: &str, names: &[&str]| -> Result<()> {
        if let Some(value) = first_present(map, names) {
            set(out, key(name), as_u32(&key(name), value)?);
        }
        Ok(())
    };
    let f32_row = |out: &mut Vec<(String, MetaValue)>, name: &str, names: &[&str]| -> Result<()> {
        if let Some(value) = first_present(map, names) {
            set(out, key(name), as_f32(&key(name), value)?);
        }
        Ok(())
    };
    let blocks = ["n_layers", "num_hidden_layers", "n_layer", "num_layers"];
    find(map, &blocks, "TextModel")?;
    u32_row(out, "block_count", &blocks)?;
    u32_row(
        out,
        "context_length",
        &["max_position_embeddings", "n_ctx", "n_positions", "max_length", "max_sequence_length", "model_max_length"],
    )?;
    u32_row(out, "embedding_length", &["hidden_size", "n_embd", "dim"])?;
    u32_row(
        out,
        "feed_forward_length",
        &["prefix_dense_intermediate_size", "dense_intermediate_size", "intermediate_size", "n_inner", "hidden_dim"],
    )?;
    u32_row(out, "attention.head_count", &["num_attention_heads", "n_head", "n_heads"])?;
    u32_row(out, "attention.head_count_kv", &["num_key_value_heads", "n_kv_heads"])?;
    let causal_off = map.get("is_causal") == Some(&Value::Bool(false))
        || map.get("use_bidirectional_attention") == Some(&Value::Bool(true));
    if causal_off {
        set(out, key("attention.causal"), MetaValue::Bool(false));
    }
    let (p, r) = rope_params(Some(map));
    rope_scaling(arch, &r, out)?;
    if let Some(sections) = p.get("mrope_section") {
        let mut values = Vec::new();
        for item in sections.as_array().into_iter().flatten().take(4) {
            let number = item
                .as_i64()
                .and_then(|number| i32::try_from(number).ok())
                .ok_or_else(|| msg(format!("mrope_section item {item} is not an i32")))?;
            values.push(MetaValue::I32(number));
        }
        while values.len() < 4 {
            values.push(MetaValue::I32(0));
        }
        set(
            out,
            key("rope.dimension_sections"),
            MetaValue::Array { elem: crate::formats::gguf::TYPE_I32, values },
        );
    }
    if let Some(theta) = r.get("rope_theta").filter(|value| !value.is_null()) {
        set(out, key("rope.freq_base"), as_f32(&key("rope.freq_base"), theta)?);
    }
    let swa = p
        .get("sliding_attention")
        .and_then(Value::as_object)
        .and_then(|sliding| sliding.get("rope_theta"))
        .filter(|value| !value.is_null());
    if let Some(theta) = swa {
        set(out, key("rope.freq_base_swa"), as_f32(&key("rope.freq_base_swa"), theta)?);
    }
    f32_row(out, "attention.layer_norm_rms_epsilon", &["rms_norm_eps", "norm_eps"])?;
    f32_row(out, "attention.layer_norm_epsilon", &["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"])?;
    u32_row(out, "expert_count", &["num_local_experts", "num_experts", "n_routed_experts"])?;
    u32_row(out, "expert_used_count", &["num_experts_per_tok", "num_experts_per_token", "top_k_experts"])?;
    u32_row(out, "expert_group_count", &["n_group"])?;
    u32_row(out, "expert_group_used_count", &["topk_group"])?;
    if let Some(func) = first_present(map, GATING) {
        let code = match func.as_str() {
            Some("softmax") => 1,
            Some("sigmoid") => 2,
            Some("sqrtsoftplus") => 4,
            _ => {
                return Err(msg(format!(
                    "expert gating function {func} is not softmax, sigmoid, or sqrtsoftplus; llama.cpp's converter stops there too"
                )))
            }
        };
        set(out, key("expert_gating_func"), MetaValue::U32(code));
    }
    if let Some(dim) = map.get("head_dim").filter(|value| !value.is_null()) {
        set(out, key("attention.key_length"), as_u32(&key("attention.key_length"), dim)?);
        set(out, key("attention.value_length"), as_u32(&key("attention.value_length"), dim)?);
    }
    Ok(())
}

/// The rope.scaling rows, written only when R.rope_type is set.
fn rope_scaling(arch: &str, r: &Map<String, Value>, out: &mut Vec<(String, MetaValue)>) -> Result<()> {
    let key = |name: &str| format!("{arch}.rope.scaling.{name}");
    let Some(kind) = r.get("rope_type").filter(|value| !value.is_null()) else {
        return Ok(());
    };
    let factor = r.get("factor").filter(|value| !value.is_null());
    let kind = kind.as_str().unwrap_or("");
    match (kind, factor) {
        ("linear", Some(factor)) => {
            set(out, key("type"), MetaValue::String("linear".into()));
            set(out, key("factor"), as_f32(&key("factor"), factor)?);
        }
        ("yarn", Some(factor)) => {
            set(out, key("type"), MetaValue::String("yarn".into()));
            set(out, key("factor"), as_f32(&key("factor"), factor)?);
            let original = r
                .get("original_max_position_embeddings")
                .filter(|value| !value.is_null())
                .ok_or_else(|| msg("yarn rope scaling has no original_max_position_embeddings; llama.cpp's converter stops there too"))?;
            set(out, key("original_context_length"), as_u32(&key("original_context_length"), original)?);
            if let Some(truncate) = r.get("truncate").filter(|value| !value.is_null()) {
                let flag = truncate
                    .as_bool()
                    .ok_or_else(|| msg(format!("rope truncate {truncate} is not a bool")))?;
                set(out, key("yarn_truncate"), MetaValue::Bool(flag));
            }
            let optional = [
                ("yarn_ext_factor", r.get("extrapolation_factor")),
                ("yarn_attn_factor", r.get("attention_factor").or_else(|| r.get("attn_factor"))),
                ("yarn_beta_fast", r.get("beta_fast")),
                ("yarn_beta_slow", r.get("beta_slow")),
            ];
            for (name, value) in optional {
                if let Some(value) = value.filter(|value| !value.is_null()) {
                    set(out, key(name), as_f32(&key(name), value)?);
                }
            }
        }
        ("su" | "longrope", _) => set(out, key("type"), MetaValue::String("longrope".into())),
        _ => {}
    }
    Ok(())
}


/// GGUF-038 and metadata-mapping generation_config_to_gguf: the sampler
/// defaults from generation_config.json, each written only when its field is
/// present and not null. A missing or invalid file writes none. A value of
/// the wrong JSON type is left out and named in the returned warning.
pub(crate) fn sampling_keys(dir: &std::path::Path) -> (Vec<(String, MetaValue)>, Option<String>) {
    const ROWS: &[(&str, &str, &[&str])] = &[
        ("sequence", "string", &["sequence"]),
        ("top_k", "i32", &["top_k"]),
        ("top_p", "f32", &["top_p"]),
        ("min_p", "f32", &["min_p"]),
        ("xtc_probability", "f32", &["xtc_probability"]),
        ("xtc_threshold", "f32", &["xtc_threshold"]),
        ("temp", "f32", &["temperature"]),
        ("penalty_last_n", "i32", &["penalty_last_n"]),
        ("penalty_repeat", "f32", &["penalty_repeat", "repetition_penalty"]),
        ("mirostat", "i32", &["mirostat"]),
        ("mirostat_tau", "f32", &["mirostat_tau"]),
        ("mirostat_eta", "f32", &["mirostat_eta"]),
    ];
    let Ok(text) = std::fs::read_to_string(dir.join("generation_config.json")) else {
        return (Vec::new(), None);
    };
    let Ok(Value::Object(config)) = serde_json::from_str::<Value>(&text) else {
        return (Vec::new(), None);
    };
    let mut out = Vec::new();
    let mut skipped = Vec::new();
    for (name, kind, fields) in ROWS {
        let Some((field, value)) = fields
            .iter()
            .find_map(|field| config.get(*field).filter(|value| !value.is_null()).map(|value| (*field, value)))
        else {
            continue;
        };
        let meta = match *kind {
            "string" => value.as_str().map(|text| MetaValue::String(text.into())),
            "i32" => value.as_i64().and_then(|number| i32::try_from(number).ok()).map(MetaValue::I32),
            _ => value.as_f64().map(|number| MetaValue::F32(number as f32)),
        };
        match meta {
            Some(meta) => out.push((format!("general.sampling.{name}"), meta)),
            None => skipped.push(format!("{field} ({value})")),
        }
    }
    let warning = (!skipped.is_empty()).then(|| {
        format!(
            "generation_config.json {} is not the type general.sampling.* needs, so it is left out (GGUF-038)",
            skipped.join(", ")
        )
    });
    (out, warning)
}

// ---------------------------------------------------------------------------
// GGUF-037: the keys a converter class's set_gguf_parameters chain writes.

static CONFIG_KEYS_JSON: &str =
    include_str!("../tensdataspecs_readonly/specs/tables/gguf-config-keys.json");

fn index() -> &'static Value {
    static INDEX: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    INDEX.get_or_init(|| serde_json::from_str(CONFIG_KEYS_JSON).expect("gguf-config-keys.json parses"))
}

/// What a class's chain writes: the keys, and the listed keys whose guard or
/// value could not be evaluated from the config.
pub(crate) struct ClassKeys {
    pub(crate) keys: Vec<(String, MetaValue)>,
    pub(crate) unevaluated: Vec<String>,
    /// hparams_mutations that could not be applied, so the config dict the
    /// keys were read from may differ from llama.cpp's.
    pub(crate) unapplied: Vec<String>,
}

/// GGUF-037: follow `class`'s set_gguf_parameters chain, base first.
/// TextModel is the hf_config_to_gguf rows; a class with a transcription in
/// gguf_class_parameters is that transcription; any other class is its
/// writes in tables/gguf-config-keys.json, evaluated where the guard and the
/// value can be.
#[cfg(test)]
pub(crate) fn class_keys(
    class: &str,
    arch: &str,
    map: &Map<String, Value>,
    dir: &std::path::Path,
) -> Result<ClassKeys> {
    class_keys_with(class, arch, map, dir, &[])
}

/// class_keys, with the checkpoint's tensor names after the filter_tensors
/// chain and the global filters (converter_attributes model_tensors). `map`
/// is the GGUF-039 dict as loaded: renames applied, text_config not yet
/// merged. The converter's hparams_mutations, TextModel's merge among them,
/// are applied before the keys are read.
pub(crate) fn class_keys_with(
    class: &str,
    arch: &str,
    map: &Map<String, Value>,
    dir: &std::path::Path,
    tensors: &[String],
) -> Result<ClassKeys> {
    let mut found = ClassKeys { keys: Vec::new(), unevaluated: Vec::new(), unapplied: Vec::new() };
    let record = index()["converters"].get(class);
    let chain: Vec<String> = match record.and_then(|record| record["set_gguf_parameters"].as_array()) {
        Some(items) => items.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        None => {
            return Err(msg(format!(
                "{class} is not in tables/gguf-config-keys.json, so its GGUF keys are not known (GGUF-037)"
            )))
        }
    };
    let converter = &index()["converters"][class];
    let hf_arch = map
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(Value::as_str)
        .map(str::to_string);
    let mut hparams = apply_mutations(class, map, dir, arch, converter, tensors, hf_arch.as_deref(), &mut found.unapplied)?;
    let sequence = converter["set_gguf_parameters_sequence"]
        .as_str()
        .and_then(|id| index()["sequences"].get(id))
        .and_then(Value::as_array);
    match sequence {
        // A class with a hand transcription follows it (gguf_class_parameters).
        Some(events) if !chain.first().is_some_and(|owner| is_transcribed(owner)) => {
            // Each pop in the sequence is applied by a mutation event; what it
            // removed is collected first so a later read of its return value works.
            let popped = popped_values(events, &hparams);
            let base = Walk { class, dir, arch, converter, tensors, hf_arch, loaded: map, chain: &chain, popped: Some(&popped) };
            base.run(events, &mut hparams, &mut found)?;
        }
        _ => {
            let (p, _) = rope_params(Some(&hparams));
            let mut ctx = Ctx { map: &hparams, p: &p, dir, arch, converter, tensors, hf_arch, loaded: map, overrides: None, popped: None, depth: 0 };
            ctx.chain(&chain, &mut found)?;
        }
    }
    Ok(found)
}

/// The values the sequence's `pop` mutations of top-level keys remove, read
/// from the dict before the walk. A key the walk changes before its pop is
/// not covered.
fn popped_values(events: &[Value], hparams: &Map<String, Value>) -> std::collections::HashMap<String, Value> {
    let mut popped = std::collections::HashMap::new();
    for event in events {
        let Some(site_id) = event["hparams_mutation"].as_str() else { continue };
        let site = &index()["assignment_sites"][site_id];
        if site["op"].as_str() != Some("pop") || site["key"].as_str() != Some("self.hparams") {
            continue;
        }
        let Some(lexed) = lex(&format!("[{}]", site["value"].as_str().unwrap_or(""))) else { continue };
        if let Some(Tok::Str(key)) = lexed.get(1) {
            if let Some(value) = hparams.get(key) {
                popped.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
    }
    popped
}

/// GGUF-037 (round 14): a site that runs and would raise in llama.cpp's
/// converter means llama.cpp writes no file, so tensconv stops too. A site
/// under an `except` guard may be caught there, so it only stays unknown.
fn stop_or_unknown(halt: Halt, class: &str, at: &str, guard: &str) -> Result<()> {
    match halt {
        Halt::Raise(cause) if !guard.contains("except") => Err(msg(format!(
            "llama.cpp's {class} would stop at {at} ({cause}) and write no file, so tensconv stops too (GGUF-037)"
        ))),
        _ => Ok(()),
    }
}

/// Whether the events from here on run: every enclosing call's guard holds
/// (Yes), one does not (No), or one cannot be evaluated (Maybe).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Runs {
    Yes,
    No,
    Maybe,
}

/// The fixed parts of a set_gguf_parameters_sequence walk.
struct Walk<'a> {
    class: &'a str,
    dir: &'a std::path::Path,
    arch: &'a str,
    converter: &'static Value,
    tensors: &'a [String],
    hf_arch: Option<String>,
    loaded: &'a Map<String, Value>,
    chain: &'a [String],
    popped: Option<&'a std::collections::HashMap<String, Value>>,
}

impl Walk<'_> {
    /// GGUF-037 with the round 14 sequence: the converter's
    /// set_gguf_parameters chain flattened into run order. Key writes, hparams
    /// changes, and attribute assignments apply where they run, under the
    /// guards of the calls that lead to them. Calls into TextModel's base,
    /// the pooling helper, and a transcribed class follow their transcriptions.
    fn run(&self, events: &'static [Value], hparams: &mut Map<String, Value>, found: &mut ClassKeys) -> Result<()> {
        let mut overrides: std::collections::HashMap<String, Option<V>> = Default::default();
        let mut method_vars: std::collections::HashMap<String, std::collections::HashMap<String, V>> = Default::default();
        // Each open call: its target and whether its events run.
        let mut stack: Vec<(String, Runs)> = Vec::new();
        // While a call is handled by a transcription, its events are skipped
        // until this stack depth returns.
        let mut skip_to: Option<usize> = None;
        let runs = |stack: &[(String, Runs)]| {
            stack.iter().fold(Runs::Yes, |acc, (_, runs)| match (acc, *runs) {
                (Runs::No, _) | (_, Runs::No) => Runs::No,
                (Runs::Maybe, _) | (_, Runs::Maybe) => Runs::Maybe,
                _ => Runs::Yes,
            })
        };
        for event in events {
            if event.get("return_from").is_some() {
                stack.pop();
                if skip_to.is_some_and(|depth| stack.len() < depth) {
                    skip_to = None;
                }
                continue;
            }
            if let Some(target) = event["call"].as_str() {
                let now = runs(&stack);
                let guard = event["when"].as_str().unwrap_or("always");
                let this = if now == Runs::No || skip_to.is_some() || guard == "always" {
                    Runs::Yes
                } else {
                    let from = event["from"].as_str().unwrap_or("");
                    match self.eval_in(from, hparams, &overrides, &mut method_vars, guard) {
                        Ok(value) if truthy(&value) => Runs::Yes,
                        Ok(_) => Runs::No,
                        Err(_) => Runs::Maybe,
                    }
                };
                stack.push((target.to_string(), this));
                if skip_to.is_some() {
                    continue;
                }
                let now = runs(&stack);
                if now == Runs::No {
                    skip_to = Some(stack.len());
                    continue;
                }
                if self.transcribed_call(target, now, hparams, &overrides, found)? {
                    skip_to = Some(stack.len());
                }
                continue;
            }
            if skip_to.is_some() {
                continue;
            }
            let now = runs(&stack);
            if now == Runs::No {
                continue;
            }
            if let Some(method) = event["write"].as_str() {
                let at = event["index"].as_u64().unwrap_or(0) as usize;
                let write = &index()["methods"][method]["writes"][at];
                self.write(method, write, now, hparams, &overrides, &mut method_vars, found)?;
            } else if let Some(site_id) = event["hparams_mutation"].as_str() {
                let site = &index()["assignment_sites"][site_id];
                let applied = if now == Runs::Yes {
                    apply_mutation(hparams, site, self.dir, self.arch, self.converter, self.tensors, self.hf_arch.as_deref(), self.loaded, self.popped)
                } else {
                    Err(Unknown)
                };
                if let Err(halt) = applied {
                    if now == Runs::Yes {
                        stop_or_unknown(halt, self.class, site_id, site["when"].as_str().unwrap_or(""))?;
                    }
                    found.unapplied.push(format!("{site_id} ({})", site["key"].as_str().unwrap_or("self.hparams")));
                }
            } else if let (Some(name), Some(site_id)) = (event["attribute"].as_str(), event["site"].as_str()) {
                let value = if now == Runs::Yes { self.assignment(name, site_id, hparams, &overrides) } else { Err(Unknown) };
                if let Err(halt) = &value {
                    let guard = index()["assignment_sites"][site_id]["when"].as_str().unwrap_or("");
                    stop_or_unknown(halt.clone(), self.class, site_id, guard)?;
                }
                match value {
                    Ok(Some(value)) => {
                        overrides.insert(name.to_string(), Some(value));
                    }
                    Ok(None) => {}
                    Err(_) => {
                        overrides.insert(name.to_string(), None);
                    }
                }
            }
        }
        // hf_config_to_gguf rope.dimension_count: LlamaModel's rope_dim is
        // assigned twice, so its row states the formula.
        let rope_key = format!("{}.rope.dimension_count", self.arch);
        if self.chain.iter().any(|owner| owner == "LlamaModel") && found.unevaluated.contains(&rope_key) {
            llama_rope_dim(self.arch, hparams, found)?;
        }
        Ok(())
    }

    fn ctx<'b>(
        &'b self,
        hparams: &'b Map<String, Value>,
        p: &'b Map<String, Value>,
        overrides: &'b std::collections::HashMap<String, Option<V>>,
    ) -> Ctx<'b> {
        Ctx {
            map: hparams,
            p,
            dir: self.dir,
            arch: self.arch,
            converter: self.converter,
            tensors: self.tensors,
            hf_arch: self.hf_arch.clone(),
            loaded: self.loaded,
            overrides: Some(overrides),
            popped: self.popped,
            depth: 0,
        }
    }

    /// An expression in `method`, with its locals and the walrus values its
    /// earlier guards set.
    fn eval_in(
        &self,
        method: &str,
        hparams: &Map<String, Value>,
        overrides: &std::collections::HashMap<String, Option<V>>,
        method_vars: &mut std::collections::HashMap<String, std::collections::HashMap<String, V>>,
        text: &str,
    ) -> Ev<V> {
        let (p, _) = rope_params(Some(hparams));
        let ctx = self.ctx(hparams, &p, overrides);
        let locals = index()["methods"][method].get("locals").and_then(Value::as_object);
        let vars = method_vars.remove(method).unwrap_or_default();
        let mut eval = Eval { ctx: &ctx, locals, site_locals: None, vars, pending: Default::default(), depth: 0 };
        let value = eval.expr(text);
        method_vars.insert(method.to_string(), eval.vars);
        value
    }

    /// TextModel's base rows, the pooling helper, or a transcribed class:
    /// these follow their transcriptions instead of their events.
    fn transcribed_call(
        &self,
        target: &str,
        now: Runs,
        hparams: &Map<String, Value>,
        overrides: &std::collections::HashMap<String, Option<V>>,
        found: &mut ClassKeys,
    ) -> Result<bool> {
        let owner = target.strip_suffix(".set_gguf_parameters");
        let handled = target == "TextModel._try_set_pooling_type"
            || owner.is_some_and(|owner| owner == "TextModel" || is_transcribed(owner));
        if !handled {
            return Ok(false);
        }
        let mut scratch = ClassKeys { keys: Vec::new(), unevaluated: Vec::new(), unapplied: Vec::new() };
        let into = if now == Runs::Yes { &mut *found } else { &mut scratch };
        let (p, _) = rope_params(Some(hparams));
        let mut ctx = self.ctx(hparams, &p, overrides);
        if target == "TextModel._try_set_pooling_type" {
            pooling(self.arch, hparams, self.dir, &mut into.keys)?;
        } else if let Some(owner) = owner {
            let from = self.chain.iter().position(|name| name == owner).unwrap_or(self.chain.len());
            ctx.chain(&self.chain[from..], into)?;
        }
        if now == Runs::Maybe {
            for key in scratch.keys.into_iter().map(|(key, _)| key).chain(scratch.unevaluated) {
                if !found.unevaluated.contains(&key) {
                    found.unevaluated.push(key);
                }
            }
        }
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn write(
        &self,
        method: &str,
        write: &'static Value,
        now: Runs,
        hparams: &Map<String, Value>,
        overrides: &std::collections::HashMap<String, Option<V>>,
        method_vars: &mut std::collections::HashMap<String, std::collections::HashMap<String, V>>,
        found: &mut ClassKeys,
    ) -> Result<()> {
        let template = write["key"].as_str().unwrap_or("");
        if template.starts_with("general.") || template.starts_with("tokenizer.") {
            return Ok(());
        }
        let key = template.replace("{arch}", self.arch);
        let guard = write["when"].as_str().unwrap_or("always");
        let value = if now == Runs::Maybe {
            Err(Unknown)
        } else {
            let holds = if guard == "always" {
                Ok(true)
            } else {
                self.eval_in(method, hparams, overrides, method_vars, guard).map(|value| truthy(&value))
            };
            match holds {
                Ok(false) => return Ok(()),
                Ok(true) => write["value"]
                    .as_str()
                    .ok_or(Unknown)
                    .and_then(|text| self.eval_in(method, hparams, overrides, method_vars, text))
                    .and_then(|value| to_meta(&value, write["type"].as_str().unwrap_or(""))),
                Err(halt) => Err(halt),
            }
        };
        if let Err(halt) = &value {
            let at = format!("{method} line {}", write["line"].as_u64().unwrap_or(0));
            stop_or_unknown(halt.clone(), self.class, &at, guard)?;
        }
        match value {
            Ok(Some(value)) => set(&mut found.keys, key, value),
            Ok(None) => {}
            Err(_) => {
                if !found.unevaluated.contains(&key) {
                    found.unevaluated.push(key);
                }
            }
        }
        Ok(())
    }

    /// An attribute assignment inside the sequence: its guard, then `=` or
    /// `+=` onto the attribute's value so far.
    fn assignment(
        &self,
        name: &str,
        site_id: &str,
        hparams: &Map<String, Value>,
        overrides: &std::collections::HashMap<String, Option<V>>,
    ) -> Ev<Option<V>> {
        let site = &index()["assignment_sites"][site_id];
        let (p, _) = rope_params(Some(hparams));
        let ctx = self.ctx(hparams, &p, overrides);
        let mut eval = Eval {
            ctx: &ctx,
            locals: None,
            site_locals: site.get("locals").and_then(Value::as_object),
            vars: Default::default(),
            pending: Default::default(),
            depth: 0,
        };
        let guard = site["when"].as_str().unwrap_or("always");
        if guard != "always" && !truthy(&eval.expr(guard)?) {
            return Ok(None);
        }
        let next = eval.expr(site["value"].as_str().ok_or(Unknown)?)?;
        Ok(Some(match site["op"].as_str() {
            Some("+=") => {
                let current = eval.self_attribute(name)?;
                arith("+", &current, &next)?
            }
            _ => next,
        }))
    }
}

/// converter_attributes hparams_mutations: the changes a converter makes to
/// its config dict before set_gguf_parameters runs (phases init,
/// prepare_tensors, and prepare_metadata before set_gguf_parameters), in run
/// order. A change that cannot be evaluated is named in `unapplied`.
fn apply_mutations(
    class: &str,
    loaded: &Map<String, Value>,
    dir: &std::path::Path,
    arch: &str,
    converter: &'static Value,
    tensors: &[String],
    hf_arch: Option<&str>,
    unapplied: &mut Vec<String>,
) -> Result<Map<String, Value>> {
    let mut hparams = loaded.clone();
    let sites = &index()["assignment_sites"];
    for reference in converter["hparams_mutations"].as_array().into_iter().flatten() {
        let phase = reference["phase"].as_str().unwrap_or("unknown");
        if !matches!(phase, "init" | "prepare_tensors" | "prepare_metadata, before set_gguf_parameters") {
            continue;
        }
        let Some(id) = reference["site"].as_str() else { continue };
        let site = &sites[id];
        match apply_mutation(&mut hparams, site, dir, arch, converter, tensors, hf_arch, loaded, None) {
            Ok(()) => {}
            Err(halt) => {
                stop_or_unknown(halt, class, id, site["when"].as_str().unwrap_or(""))?;
                unapplied.push(format!("{id} ({})", site["key"].as_str().unwrap_or("self.hparams")));
            }
        }
    }
    Ok(hparams)
}

fn apply_mutation(
    hparams: &mut Map<String, Value>,
    site: &'static Value,
    dir: &std::path::Path,
    arch: &str,
    converter: &'static Value,
    tensors: &[String],
    hf_arch: Option<&str>,
    loaded: &Map<String, Value>,
    popped: Option<&std::collections::HashMap<String, Value>>,
) -> Ev<()> {
    let current = hparams.clone();
    let (p, _) = rope_params(Some(&current));
    let ctx = Ctx { map: &current, p: &p, dir, arch, converter, tensors, hf_arch: hf_arch.map(str::to_string), loaded, overrides: None, popped, depth: 0 };
    let mut eval = Eval {
        ctx: &ctx,
        locals: None,
        site_locals: site.get("locals").and_then(Value::as_object),
        vars: Default::default(),
        pending: Default::default(),
        depth: 0,
    };
    // A local passed on as hparams= is the dict itself (constructor_arguments).
    let on_local = site["on"].as_str().is_some_and(|on| on.starts_with("local hparams"));
    if on_local {
        eval.vars.insert("hparams".into(), V::J(Value::Object(current.clone())));
    }
    let op = site["op"].as_str().unwrap_or("");
    let text = site["value"].as_str().unwrap_or("");
    // ModelBase.__init__ loads the dict, or takes the one passed in: both are
    // the dict as it stands here, whatever its guard says.
    if op == "replace" && text.contains("load_hparams") {
        return Ok(());
    }
    let guard = site["when"].as_str().unwrap_or("always");
    if guard != "always" && !truthy(&eval.expr(guard)?) {
        return Ok(());
    }
    if op == "replace" {
        let V::J(Value::Object(next)) = eval.expr(text)? else { return Err(Unknown) };
        *hparams = next;
        return Ok(());
    }
    let target = site["key"].as_str().ok_or(Unknown)?;
    match op {
        "=" | "+=" => {
            let path = hparams_path(&mut eval, target, on_local)?;
            let value = match eval.expr(text)? {
                V::J(value) => value,
                _ => return Err(Unknown),
            };
            let value = if op == "+=" {
                let current = path_get(hparams, &path).ok_or(Unknown)?;
                match arith("+", &V::J(current.clone()), &V::J(value))? {
                    V::J(value) => value,
                    _ => return Err(Unknown),
                }
            } else {
                value
            };
            path_set(hparams, &path, value)
        }
        "del" => {
            let path = hparams_path(&mut eval, target, on_local)?;
            path_remove(hparams, &path)
        }
        "pop" | "setdefault" => {
            let mut path = hparams_path(&mut eval, target, on_local)?;
            let V::J(Value::Array(args)) = eval.expr(&format!("[{text}]"))? else { return Err(Unknown) };
            let key = args.first().and_then(Value::as_str).ok_or(Unknown)?;
            path.push(Value::from(key));
            if op == "pop" {
                let _ = path_remove(hparams, &path);
                Ok(())
            } else {
                if path_get(hparams, &path).is_none() {
                    path_set(hparams, &path, args.get(1).cloned().unwrap_or(Value::Null))?;
                }
                Ok(())
            }
        }
        _ => Err(Unknown),
    }
}

/// The subscripts below self.hparams (or the passed-on local hparams) in a
/// mutation's target, such as `self.hparams['a']['b']`.
fn hparams_path(eval: &mut Eval, target: &str, on_local: bool) -> Ev<Vec<Value>> {
    let tokens = lex(target).ok_or(Unknown)?;
    let mut parser = Parser { tokens, at: 0 };
    let mut tree = parser.test().ok_or(Unknown)?;
    if parser.at != parser.tokens.len() {
        return Err(Unknown);
    }
    let mut path = Vec::new();
    loop {
        match tree {
            E::Index(base, index) => {
                let V::J(index) = eval.eval(&index)? else { return Err(Unknown) };
                path.push(index);
                tree = *base;
            }
            E::Attr(base, name) if name == "hparams" && matches!(*base, E::Name(ref root) if root == "self") => break,
            E::Name(name) if name == "hparams" && on_local => break,
            _ => return Err(Unknown),
        }
    }
    path.reverse();
    Ok(path)
}

fn path_get<'a>(map: &'a Map<String, Value>, path: &[Value]) -> Option<&'a Value> {
    let (first, rest) = path.split_first()?;
    let mut value = map.get(first.as_str()?)?;
    for step in rest {
        value = match (value, step) {
            (Value::Object(map), Value::String(key)) => map.get(key)?,
            (Value::Array(items), Value::Number(index)) => items.get(usize::try_from(index.as_u64()?).ok()?)?,
            _ => return None,
        };
    }
    Some(value)
}

fn path_set(map: &mut Map<String, Value>, path: &[Value], value: Value) -> Ev<()> {
    let (last, parents) = path.split_last().ok_or(Unknown)?;
    let mut container: &mut Map<String, Value> = map;
    for step in parents {
        container = container
            .get_mut(step.as_str().ok_or(Unknown)?)
            .and_then(Value::as_object_mut)
            .ok_or(Unknown)?;
    }
    container.insert(last.as_str().ok_or(Unknown)?.to_string(), value);
    Ok(())
}

fn path_remove(map: &mut Map<String, Value>, path: &[Value]) -> Ev<()> {
    let (last, parents) = path.split_last().ok_or(Unknown)?;
    let mut container: &mut Map<String, Value> = map;
    for step in parents {
        container = container
            .get_mut(step.as_str().ok_or(Unknown)?)
            .and_then(Value::as_object_mut)
            .ok_or(Unknown)?;
    }
    container.shift_remove(last.as_str().ok_or(Unknown)?);
    Ok(())
}

/// find(a, b) in gguf_class_parameters: find_hparam, KeyError when none is present.
/// A null value is found too, but llama.cpp then fails writing it, so it
/// stops the conversion here.
fn find<'a>(map: &'a Map<String, Value>, names: &[&str], class: &str) -> Result<&'a Value> {
    let value = lookup(map, names).ok_or_else(|| {
        msg(format!(
            "config.json has none of {}, which llama.cpp's {class} needs",
            names.join(", ")
        ))
    })?;
    if value.is_null() {
        let name = names.iter().find(|name| map.contains_key(**name)).unwrap_or(&"");
        return Err(msg(format!(
            "config.json {name} is null; llama.cpp's {class} would fail writing it"
        )));
    }
    Ok(value)
}

/// A step of the chain that metadata-mapping gguf_class_parameters
/// transcribes by hand. Returns false for a class it does not cover.
fn is_transcribed(owner: &str) -> bool {
    matches!(owner, "Phi3MiniModel" | "PhiMoeModel" | "Gemma2Model" | "Gemma3Model")
}

fn transcribed(owner: &str, arch: &str, map: &Map<String, Value>, out: &mut Vec<(String, MetaValue)>) -> Result<bool> {
    match owner {
        "Phi3MiniModel" => phi3_mini(owner, arch, map, out)?,
        "PhiMoeModel" => {
            let key = |name: &str| format!("{arch}.{name}");
            let used = find(map, &["num_experts_per_tok", "num_experts_per_token"], owner)?;
            set(out, key("expert_used_count"), as_u32(&key("expert_used_count"), used)?);
            let count = find(map, &["num_local_experts", "num_experts"], owner)?;
            set(out, key("expert_count"), as_u32(&key("expert_count"), count)?);
        }
        "Gemma2Model" => gemma2(owner, arch, map, out)?,
        "Gemma3Model" => gemma3(owner, arch, map, out)?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// gguf_class_parameters Gemma2Model: a replaced base. Every key is written,
/// and a missing config value stops the conversion as it stops llama.cpp's.
fn gemma2(class: &str, arch: &str, map: &Map<String, Value>, out: &mut Vec<(String, MetaValue)>) -> Result<()> {
    let key = |name: &str| format!("{arch}.{name}");
    let u32_rows: [(&str, &[&str]); 6] = [
        ("context_length", &["max_position_embeddings"]),
        ("embedding_length", &["hidden_size"]),
        ("block_count", &["n_layers", "num_hidden_layers", "n_layer", "num_layers"]),
        ("feed_forward_length", &["intermediate_size"]),
        ("attention.head_count", &["num_attention_heads"]),
        ("attention.head_count_kv", &["num_key_value_heads", "num_attention_heads"]),
    ];
    for (name, names) in u32_rows {
        set(out, key(name), as_u32(&key(name), find(map, names, class)?)?);
    }
    let eps = find(map, &["rms_norm_eps"], class)?;
    set(out, key("attention.layer_norm_rms_epsilon"), as_f32(&key("attention.layer_norm_rms_epsilon"), eps)?);
    let head_dim = find(map, &["head_dim"], class)?;
    set(out, key("attention.key_length"), as_u32(&key("attention.key_length"), head_dim)?);
    set(out, key("attention.value_length"), as_u32(&key("attention.value_length"), head_dim)?);
    for name in ["attn_logit_softcapping", "final_logit_softcapping"] {
        set(out, key(name), as_f32(&key(name), find(map, &[name], class)?)?);
    }
    let window = find(map, &["sliding_window"], class)?;
    set(out, key("attention.sliding_window"), as_u32(&key("attention.sliding_window"), window)?);
    Ok(())
}

/// gguf_class_parameters Gemma3Model: written after the base rows, with
/// defaults where the config has no value.
fn gemma3(class: &str, arch: &str, map: &Map<String, Value>, out: &mut Vec<(String, MetaValue)>) -> Result<()> {
    let key = |name: &str| format!("{arch}.{name}");
    // config.get(name, default): a present null is passed on, and fails as a u32 or f32.
    let or = |name: &str, default: Value| map.get(name).cloned().unwrap_or(default);
    let (_, r) = rope_params(Some(map));
    set(out, key("context_length"), as_u32(&key("context_length"), &or("max_position_embeddings", Value::from(131072)))?);
    set(out, key("attention.head_count"), as_u32(&key("attention.head_count"), &or("num_attention_heads", Value::from(8)))?);
    set(
        out,
        key("attention.layer_norm_rms_epsilon"),
        as_f32(&key("attention.layer_norm_rms_epsilon"), &or("rms_norm_eps", Value::from(1e-6)))?,
    );
    let head_dim = or("head_dim", Value::from(256));
    set(out, key("attention.key_length"), as_u32(&key("attention.key_length"), &head_dim)?);
    set(out, key("attention.value_length"), as_u32(&key("attention.value_length"), &head_dim)?);
    let theta = r.get("rope_theta").cloned().unwrap_or(Value::from(1_000_000.0));
    set(out, key("rope.freq_base"), as_f32(&key("rope.freq_base"), &theta)?);
    if map.get("attn_logit_softcapping").is_some_and(|value| !value.is_null()) {
        return Err(msg(format!(
            "config.json sets attn_logit_softcapping, which llama.cpp's {class} asserts is absent"
        )));
    }
    let final_cap = map.get("final_logit_softcapping").filter(|value| {
        !value.is_null() && value.as_f64() != Some(0.0) && value.as_bool() != Some(false)
    });
    if let Some(cap) = final_cap {
        set(out, key("final_logit_softcapping"), as_f32(&key("final_logit_softcapping"), cap)?);
    }
    let pattern_is_one = map.get("sliding_window_pattern").and_then(Value::as_f64) == Some(1.0);
    if !pattern_is_one {
        let window = find(map, &["sliding_window"], class)?;
        set(out, key("attention.sliding_window"), as_u32(&key("attention.sliding_window"), window)?);
    }
    set(out, key("attention.head_count_kv"), as_u32(&key("attention.head_count_kv"), &or("num_key_value_heads", Value::from(4)))?);
    Ok(())
}

/// gguf_class_parameters helpers TextModel._try_set_pooling_type: the pooling
/// type from sentence-transformers' modules.json or embedding_config.json, and
/// the classifier keys from the config dict.
fn pooling(arch: &str, map: &Map<String, Value>, dir: &std::path::Path, out: &mut Vec<(String, MetaValue)>) -> Result<()> {
    let read = |path: std::path::PathBuf| -> Result<Option<Value>> {
        if !path.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path).map_err(|err| crate::error::io_err(&path, err))?;
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|err| msg(format!("{}: {err}", path.display())))
    };
    let pooling_path = read(dir.join("modules.json"))?.and_then(|modules| {
        modules.as_array()?.iter().find_map(|entry| {
            let kind = entry.get("type")?.as_str()?;
            kind.ends_with("Pooling")
                .then(|| entry.get("path").and_then(Value::as_str).map(str::to_string))
                .flatten()
        })
    });
    let key = |name: &str| format!("{arch}.{name}");
    let truthy = |config: &Value, name: &str| config.get(name).is_some_and(|value| truthy(&V::J(value.clone())));
    if let Some(path) = pooling_path {
        let config = read(dir.join(&path).join("config.json"))?.unwrap_or(Value::Null);
        let mode = if truthy(&config, "pooling_mode_mean_tokens") {
            1
        } else if truthy(&config, "pooling_mode_cls_token") {
            2
        } else if truthy(&config, "pooling_mode_lasttoken") {
            3
        } else {
            match config.get("pooling_mode").and_then(Value::as_str) {
                Some("mean") => 1,
                Some("cls") => 2,
                Some("lasttoken") => 3,
                _ => {
                    return Err(msg(format!(
                        "{path}/config.json names no pooling mode llama.cpp's converter supports"
                    )))
                }
            }
        };
        set(out, key("pooling_type"), MetaValue::U32(mode));
    } else if let Some(config) = read(dir.join("embedding_config.json"))? {
        match config.get("pooling") {
            None | Some(Value::Null) => {}
            Some(Value::String(text)) if text == "last_token" => set(out, key("pooling_type"), MetaValue::U32(3)),
            Some(other) => {
                return Err(msg(format!(
                    "embedding_config.json pooling {other} is not last_token, which llama.cpp's converter needs"
                )))
            }
        }
    }
    match map.get("classifier_pooling") {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) if text == "cls" => set(out, key("classifier.pooling_type"), MetaValue::U32(2)),
        Some(Value::String(text)) if text == "mean" => set(out, key("classifier.pooling_type"), MetaValue::U32(1)),
        Some(other) => return Err(msg(format!("classifier_pooling {other} is not cls or mean"))),
    }
    match map.get("classifier_activation") {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) if matches!(text.as_str(), "gelu" | "silu" | "tanh") => {
            set(out, key("classifier.activation"), MetaValue::String(text.clone()))
        }
        Some(other) => return Err(msg(format!("classifier_activation {other} is not gelu, silu, or tanh"))),
    }
    Ok(())
}

/// hf_config_to_gguf rope.dimension_count: LlamaModel uses head_dim, else
/// hidden_size // num_attention_heads. The index cannot evaluate it, because
/// rope_dim is assigned twice there.
fn llama_rope_dim(arch: &str, map: &Map<String, Value>, found: &mut ClassKeys) -> Result<()> {
    let key = format!("{arch}.rope.dimension_count");
    let dim = match map.get("head_dim").filter(|value| !value.is_null()) {
        Some(dim) => as_u32(&key, dim)?,
        None => {
            let hidden = find(map, &["hidden_size"], "LlamaModel")?.as_u64();
            let heads = find(map, &["num_attention_heads"], "LlamaModel")?.as_u64().filter(|heads| *heads > 0);
            match hidden.zip(heads) {
                Some((hidden, heads)) => MetaValue::U32(u32::try_from(hidden / heads).map_err(|_| msg(format!("{key} is too large")))?),
                None => return Err(msg("hidden_size and num_attention_heads must be positive integers for LlamaModel")),
            }
        }
    };
    set(&mut found.keys, key.clone(), dim);
    found.unevaluated.retain(|name| *name != key);
    Ok(())
}

/// gguf_class_parameters Phi3MiniModel: a replaced base, and attn_factor
/// from generate_extra_tensors when P.long_factor is non-empty.
fn phi3_mini(class: &str, arch: &str, map: &Map<String, Value>, out: &mut Vec<(String, MetaValue)>) -> Result<()> {
    let key = |name: &str| format!("{arch}.{name}");
    let (p, r) = rope_params(Some(map));
    let u32_of = |name: &str, value: &Value| as_u32(&key(name), value);
    let context = find(map, &["n_positions", "max_position_embeddings"], class)?;
    set(out, key("context_length"), u32_of("context_length", context)?);
    let original = p
        .get("original_max_position_embeddings")
        .filter(|value| !value.is_null())
        .ok_or_else(|| msg(format!("config.json has no original_max_position_embeddings, which llama.cpp's {class} needs")))?;
    set(out, key("rope.scaling.original_context_length"), u32_of("rope.scaling.original_context_length", original)?);
    let n_embd = find(map, &["hidden_size", "n_embd"], class)?;
    set(out, key("embedding_length"), u32_of("embedding_length", n_embd)?);
    let ff = find(map, &["intermediate_size"], class)?;
    set(out, key("feed_forward_length"), u32_of("feed_forward_length", ff)?);
    let blocks = find(map, &["n_layers", "num_hidden_layers", "n_layer", "num_layers"], class)?;
    set(out, key("block_count"), u32_of("block_count", blocks)?);
    let n_head = find(map, &["num_attention_heads", "n_head"], class)?;
    set(out, key("attention.head_count"), u32_of("attention.head_count", n_head)?);
    let n_kv = find(map, &["num_key_value_heads", "n_head_kv"], class)?;
    set(out, key("attention.head_count_kv"), u32_of("attention.head_count_kv", n_kv)?);
    let eps = find(map, &["rms_norm_eps"], class)?;
    set(out, key("attention.layer_norm_rms_epsilon"), as_f32(&key("attention.layer_norm_rms_epsilon"), eps)?);
    // int(P.get('partial_rotary_factor', 1.0) * n_embd) // n_head
    let partial = p.get("partial_rotary_factor").and_then(Value::as_f64).unwrap_or(1.0);
    let embd = n_embd.as_f64().ok_or_else(|| msg("hidden_size is not a number"))?;
    let heads = n_head.as_u64().filter(|heads| *heads > 0).ok_or_else(|| msg("num_attention_heads is not a positive integer"))?;
    let rope_dim = (partial * embd).trunc() as u64 / heads;
    set(out, key("rope.dimension_count"), MetaValue::U32(rope_dim as u32));
    let theta = r
        .get("rope_theta")
        .filter(|value| !value.is_null())
        .ok_or_else(|| msg(format!("config.json has no rope_theta, which llama.cpp's {class} needs")))?;
    set(out, key("rope.freq_base"), as_f32(&key("rope.freq_base"), theta)?);
    let window = match map.get("sliding_window") {
        None | Some(Value::Null) => MetaValue::U32(0),
        Some(value) => u32_of("attention.sliding_window", value)?,
    };
    set(out, key("attention.sliding_window"), window);
    let long = p.get("long_factor").and_then(Value::as_array).filter(|items| !items.is_empty());
    if let Some(long) = long {
        let scale = context.as_f64().ok_or_else(|| msg("context length is not a number"))?
            / original.as_f64().filter(|value| *value > 0.0).ok_or_else(|| msg("original_max_position_embeddings is not positive"))?;
        let original = original.as_f64().unwrap_or(1.0);
        let kind = p
            .get("rope_type")
            .and_then(Value::as_str)
            .filter(|kind| !kind.is_empty())
            .ok_or_else(|| msg(format!("rope_scaling has long_factor but no type, which llama.cpp's {class} needs")))?
            .to_lowercase();
        let factor = match kind.as_str() {
            "su" | "longrope" if scale > 1.0 => (1.0 + scale.ln() / original.ln()).sqrt(),
            "yarn" if scale > 1.0 => 0.1 * scale.ln() + 1.0,
            "su" | "longrope" | "yarn" => 1.0,
            other => {
                return Err(msg(format!(
                    "rope_scaling type {other} with long_factor is not su, longrope, or yarn; llama.cpp's {class} stops there too"
                )))
            }
        };
        set(out, key("rope.scaling.attn_factor"), MetaValue::F32(factor as f32));
        let short = p.get("short_factor").and_then(Value::as_array);
        let half = (rope_dim / 2) as usize;
        if long.len() != half || short.map(Vec::len) != Some(half) {
            return Err(msg(format!(
                "rope_scaling long_factor and short_factor must each have rope.dimension_count / 2 = {half} items; llama.cpp's {class} stops there too"
            )));
        }
    }
    Ok(())
}

struct Ctx<'a> {
    map: &'a Map<String, Value>,
    p: &'a Map<String, Value>,
    dir: &'a std::path::Path,
    arch: &'a str,
    /// The converter's record: its self_calls map resolves self.<m>() for
    /// this converter, since a method record's resolves_to may say "per converter".
    converter: &'static Value,
    /// converter_attributes model_tensors: the tensor names after filtering.
    tensors: &'a [String],
    hf_arch: Option<String>,
    /// The dict as ModelBase.load_hparams returns it, before any change.
    loaded: &'a Map<String, Value>,
    /// Attributes assigned so far inside the set_gguf_parameters sequence
    /// (None: assigned in a way that cannot be evaluated). Absent outside a
    /// sequence walk.
    overrides: Option<&'a std::collections::HashMap<String, Option<V>>>,
    /// Top-level keys a `pop` mutation removed, with their values, so a later
    /// local such as HunYuan's `saved = self.hparams.pop('num_experts', None)`
    /// reads the value it returned.
    popped: Option<&'a std::collections::HashMap<String, Value>>,
    depth: usize,
}

impl Ctx<'_> {
    fn key(&self, template: &str) -> String {
        template.replace("{arch}", self.arch)
    }

    /// One step of the set_gguf_parameters chain and, through its super()
    /// call, the steps after it. TextModel is the hf_config_to_gguf rows. A
    /// transcribed class extends the rest of its chain.
    fn chain(&mut self, chain: &[String], found: &mut ClassKeys) -> Result<()> {
        let Some((owner, rest)) = chain.split_first() else {
            return Ok(());
        };
        if owner == "TextModel" {
            base_rows(self.arch, self.map, &mut found.keys)?;
            return self.reassigned_block_count(found);
        }
        if is_transcribed(owner) {
            self.chain(rest, found)?;
            transcribed(owner, self.arch, self.map, &mut found.keys)?;
            return Ok(());
        }
        self.method(&format!("{owner}.set_gguf_parameters"), rest, found)?;
        if owner == "LlamaModel" {
            llama_rope_dim(self.arch, self.map, found)?;
        }
        Ok(())
    }

    /// The block_count row writes self.block_count, which 40 converters assign
    /// again in their own __init__ (converter_attributes).
    fn reassigned_block_count(&mut self, found: &mut ClassKeys) -> Result<()> {
        if self.converter["attribute_assignments"].get("block_count").is_none() {
            return Ok(());
        }
        let key = self.key("{arch}.block_count");
        let mut eval = Eval { ctx: self, locals: None, site_locals: None, vars: Default::default(), pending: Default::default(), depth: 0 };
        let value = eval.self_attribute("block_count").and_then(|value| to_meta(&value, "u32"));
        match value {
            Ok(Some(value)) => set(&mut found.keys, key, value),
            _ => {
                found.keys.retain(|(name, _)| *name != key);
                if !found.unevaluated.contains(&key) {
                    found.unevaluated.push(key);
                }
            }
        }
        Ok(())
    }

    /// The writes of one indexed method, the self calls it makes, and its
    /// super() call to the rest of the chain, in line order: a few classes
    /// write keys before they call super().
    fn method(&mut self, name: &str, rest: &[String], found: &mut ClassKeys) -> Result<()> {
        let Some(record) = index()["methods"].get(name) else {
            return Ok(());
        };
        self.depth += 1;
        if self.depth > 16 {
            return Err(msg(format!("{name}: self calls nest too deeply")));
        }
        let locals = record.get("locals").and_then(Value::as_object);
        let mut steps: Vec<(u64, &Value, Step)> = Vec::new();
        for write in record["writes"].as_array().into_iter().flatten() {
            steps.push((write["line"].as_u64().unwrap_or(0), write, Step::Write));
        }
        for call in record["self_calls"].as_array().into_iter().flatten() {
            steps.push((call["line"].as_u64().unwrap_or(0), call, Step::SelfCall));
        }
        if !rest.is_empty() {
            for call in record["super_calls"].as_array().into_iter().flatten() {
                if call["method"].as_str() == Some("set_gguf_parameters") {
                    steps.push((call["line"].as_u64().unwrap_or(0), call, Step::Super));
                }
            }
        }
        steps.sort_by_key(|step| step.0);
        for (_, step, kind) in steps {
            let is_write = kind == Step::Write;
            let mut eval = Eval { ctx: self, locals, site_locals: None, vars: Default::default(), pending: Default::default(), depth: 0 };
            let guard = step["when"].as_str().unwrap_or("always");
            let holds = if guard == "always" { Ok(true) } else { eval.expr(guard).map(|value| truthy(&value)) };
            if is_write {
                let template = step["key"].as_str().unwrap_or("");
                if template.starts_with("general.") || template.starts_with("tokenizer.") {
                    continue;
                }
                let key = self.key(template);
                let value = match holds {
                    Ok(false) => continue,
                    Ok(true) => step["value"]
                        .as_str()
                        .ok_or(Unknown)
                        .and_then(|text| eval.expr(text))
                        .and_then(|value| to_meta(&value, step["type"].as_str().unwrap_or(""))),
                    Err(_) => Err(Unknown),
                };
                match value {
                    Ok(Some(value)) => set(&mut found.keys, key, value),
                    Ok(None) => {}
                    Err(_) => {
                        if !found.unevaluated.contains(&key) {
                            found.unevaluated.push(key);
                        }
                    }
                }
            } else {
                let target = if kind == Step::SelfCall {
                    step["method"]
                        .as_str()
                        .and_then(|name| self.converter["self_calls"].get(name))
                        .and_then(Value::as_str)
                } else {
                    None
                };
                if kind == Step::SelfCall && target.is_none() {
                    continue;
                }
                let run = |this: &mut Self, into: &mut ClassKeys| match target {
                    Some("TextModel._try_set_pooling_type") => pooling(this.arch, this.map, this.dir, &mut into.keys),
                    Some(target) if kind == Step::SelfCall => this.method(target, &[], into),
                    _ => this.chain(rest, into),
                };
                match holds {
                    Ok(false) => {}
                    Ok(true) => run(self, found)?,
                    Err(_) => {
                        // The call may or may not run: none of its keys can be evaluated.
                        let mut skipped = ClassKeys { keys: Vec::new(), unevaluated: Vec::new(), unapplied: Vec::new() };
                        run(self, &mut skipped)?;
                        for key in skipped.keys.into_iter().map(|(key, _)| key).chain(skipped.unevaluated) {
                            if !found.unevaluated.contains(&key) {
                                found.unevaluated.push(key);
                            }
                        }
                    }
                }
            }
        }
        self.depth -= 1;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Step {
    Write,
    SelfCall,
    Super,
}

/// Why an expression has no value: the evaluator cannot decide it from the
/// config, or llama.cpp's converter would raise there (a missing key, a
/// non-optional find_hparam with no name present, a null passed to a key
/// writer). GGUF-037 (round 14): a site that runs and raises stops the
/// conversion; one that cannot be read leaves its key out with a warning.
#[derive(Debug, Clone, PartialEq)]
enum Halt {
    Unknown,
    Raise(String),
}

#[allow(non_upper_case_globals)]
const Unknown: Halt = Halt::Unknown;

impl Halt {
    /// A raise reached only through something unknown may not happen.
    fn unsure(self) -> Halt {
        Halt::Unknown
    }
}

type Ev<T> = std::result::Result<T, Halt>;

#[derive(Debug, Clone, PartialEq)]
enum V {
    J(Value),
    Path(std::path::PathBuf),
    SelfObj,
    Builtin(&'static str),
    Bound(Box<V>, String),
    /// A re.match result: the whole match and each group, None where a
    /// group did not take part.
    Match(Vec<Option<String>>),
}

fn truthy(value: &V) -> bool {
    match value {
        V::J(Value::Null) => false,
        V::J(Value::Bool(flag)) => *flag,
        V::J(Value::Number(number)) => number.as_f64().is_some_and(|number| number != 0.0),
        V::J(Value::String(text)) => !text.is_empty(),
        V::J(Value::Array(items)) => !items.is_empty(),
        V::J(Value::Object(map)) => !map.is_empty(),
        _ => true,
    }
}

/// A Python value as the GGUF type the index gives. `Ok(None)` is an empty
/// list, which add_array leaves out (GGUF-041). Enum constants are not
/// converted.
fn to_meta(value: &V, kind: &str) -> Ev<Option<MetaValue>> {
    use crate::formats::gguf::{TYPE_I32, TYPE_U32, TYPE_U64};
    let V::J(value) = value else { return Err(Unknown) };
    // gguf-py fails packing None (round 10).
    if value.is_null() {
        return Err(Halt::Raise("None passed to the key writer".into()));
    }
    let scalar = match (kind, value) {
        (_, Value::Array(items)) => {
            let fixed = match kind {
                "array[u32]" => Some(TYPE_U32),
                "array[i32]" => Some(TYPE_I32),
                "array[u64]" => Some(TYPE_U64),
                "array" | "u32 or array" | "string or array" => None,
                _ => return Err(Unknown),
            };
            return match fixed {
                // add_key_value with an empty array raises when the file is written.
                Some(_) if items.is_empty() => Err(Unknown),
                Some(elem) => {
                    let values = items
                        .iter()
                        .map(|item| match elem {
                            TYPE_U32 => item.as_u64().and_then(|n| u32::try_from(n).ok()).map(MetaValue::U32),
                            TYPE_I32 => item.as_i64().and_then(|n| i32::try_from(n).ok()).map(MetaValue::I32),
                            _ => item.as_u64().map(MetaValue::U64),
                        })
                        .collect::<Option<Vec<_>>>()
                        .ok_or(Unknown)?;
                    Ok(Some(MetaValue::Array { elem, values }))
                }
                None if items.is_empty() => Ok(None),
                None => inferred_array(items).map(Some),
            };
        }
        ("u32" | "u32 or array", Value::Bool(flag)) if kind == "u32 or array" => Some(MetaValue::U32(u32::from(*flag))),
        ("u32" | "u32 or array", _) => value.as_u64().and_then(|n| u32::try_from(n).ok()).map(MetaValue::U32),
        ("i32", _) => value.as_i64().and_then(|n| i32::try_from(n).ok()).map(MetaValue::I32),
        ("f32", _) => value.as_f64().map(|n| MetaValue::F32(n as f32)),
        ("bool", _) => value.as_bool().map(MetaValue::Bool),
        ("string" | "string or array", _) => value.as_str().map(|text| MetaValue::String(text.into())),
        _ => None,
    };
    scalar.map(Some).ok_or(Unknown)
}

/// GGUF-041: gguf-py types an array by its first element (str STRING, list a
/// nested ARRAY, float FLOAT32, bool BOOL, int INT32), and every element must
/// get the same type.
fn inferred_array(items: &[Value]) -> Ev<MetaValue> {
    use crate::formats::gguf::{TYPE_ARRAY, TYPE_BOOL, TYPE_F32, TYPE_I32, TYPE_STRING};
    let element = |item: &Value| -> Ev<MetaValue> {
        Ok(match item {
            Value::String(text) => MetaValue::String(text.clone()),
            Value::Array(inner) if !inner.is_empty() => inferred_array(inner)?,
            Value::Number(number) if number.is_f64() => MetaValue::F32(number.as_f64().ok_or(Unknown)? as f32),
            Value::Bool(flag) => MetaValue::Bool(*flag),
            Value::Number(number) => MetaValue::I32(number.as_i64().and_then(|n| i32::try_from(n).ok()).ok_or(Unknown)?),
            _ => return Err(Unknown),
        })
    };
    let elem = match items.first().ok_or(Unknown)? {
        Value::String(_) => TYPE_STRING,
        Value::Array(_) => TYPE_ARRAY,
        Value::Number(number) if number.is_f64() => TYPE_F32,
        Value::Bool(_) => TYPE_BOOL,
        Value::Number(_) => TYPE_I32,
        _ => return Err(Unknown),
    };
    let values = items.iter().map(element).collect::<Ev<Vec<_>>>()?;
    if values.iter().any(|value| value.type_id() != elem) {
        return Err(Unknown);
    }
    Ok(MetaValue::Array { elem, values })
}

struct Eval<'c, 'a> {
    ctx: &'c Ctx<'a>,
    locals: Option<&'static Map<String, Value>>,
    /// The locals of an attribute assignment site: each assignment in run
    /// order, with its op and its own guard.
    site_locals: Option<&'static Map<String, Value>>,
    vars: std::collections::HashMap<String, V>,
    /// Site locals being evaluated, so a guard that reads its own local
    /// before it has a value is unknown rather than endless.
    pending: std::collections::HashSet<String>,
    depth: usize,
}

impl Eval<'_, '_> {
    fn expr(&mut self, text: &str) -> Ev<V> {
        let tokens = lex(text).ok_or(Unknown)?;
        let mut parser = Parser { tokens, at: 0 };
        let tree = parser.named().ok_or(Unknown)?;
        if parser.at != parser.tokens.len() {
            return Err(Unknown);
        }
        self.eval(&tree)
    }

    fn name(&mut self, name: &str) -> Ev<V> {
        if let Some(value) = self.vars.get(name) {
            return Ok(value.clone());
        }
        match name {
            "self" => return Ok(V::SelfObj),
            "re" => return Ok(V::Builtin("re")),
            // converter_attributes constructor_arguments: the converter is built
            // as model_class(dir_model, ...) with no hparams argument.
            "kwargs" => return Ok(V::Builtin("kwargs")),
            "args" => return Ok(V::Builtin("args")),
            "ModelBase" => return Ok(V::Builtin("ModelBase")),
            "int" | "float" | "len" | "bool" | "str" | "abs" | "min" | "max" | "any" | "all" | "sum" | "list" | "sorted" | "_set" => {
                return Ok(V::Builtin(match name {
                    "_set" => "_set",
                    "any" => "any",
                    "all" => "all",
                    "sum" => "sum",
                    "list" => "list",
                    "sorted" => "sorted",
                    "int" => "int",
                    "float" => "float",
                    "len" => "len",
                    "bool" => "bool",
                    "str" => "str",
                    "abs" => "abs",
                    "min" => "min",
                    _ => "max",
                }))
            }
            _ => {}
        }
        if let Some(assignments) = self.site_locals.and_then(|locals| locals.get(name)).and_then(Value::as_array) {
            return self.site_local(name, assignments);
        }
        // A local with one assignment in the method is that expression. With
        // more than one, which applies depends on code the index does not give.
        let assigned = self
            .locals
            .and_then(|locals| locals.get(name))
            .and_then(Value::as_array)
            .ok_or(Unknown)?;
        let [only] = assigned.as_slice() else { return Err(Unknown) };
        let text = only.as_str().ok_or(Unknown)?;
        let text = text.split_once(": ").map_or(text, |(_, rest)| rest);
        self.depth += 1;
        if self.depth > 16 {
            return Err(Unknown);
        }
        // The index gives a method local without the guard it is assigned
        // under, so a raise in it may not happen.
        let value = self.expr(text).map_err(Halt::unsure);
        self.depth -= 1;
        let value = value?;
        self.vars.insert(name.to_string(), value.clone());
        Ok(value)
    }

    fn attr(&mut self, base: V, name: &str) -> Ev<V> {
        match (&base, name) {
            (V::SelfObj, "hparams") => Ok(V::J(Value::Object(self.ctx.map.clone()))),
            (V::SelfObj, "rope_parameters") => Ok(V::J(Value::Object(self.ctx.p.clone()))),
            (V::SelfObj, "dir_model") => Ok(V::Path(self.ctx.dir.to_path_buf())),
            // converter_attributes: false for every class chosen from config.json.
            (V::SelfObj, "is_mistral_format") => Ok(V::J(Value::Bool(false))),
            (V::SelfObj, "find_hparam") => Ok(V::Bound(Box::new(base), name.into())),
            (V::SelfObj, _) => self.self_attribute(name),
            (V::J(Value::Object(_)), "get" | "pop") => Ok(V::Bound(Box::new(base), name.into())),
            (V::Builtin("re"), "match") | (V::Match(_), "group") => Ok(V::Bound(Box::new(base), name.into())),
            (V::Builtin("kwargs"), "pop" | "get") | (V::Builtin("ModelBase"), "load_hparams") => {
                Ok(V::Bound(Box::new(base), name.into()))
            }
            (V::Path(_), "is_file" | "exists") => Ok(V::Bound(Box::new(base), name.into())),
            (V::J(Value::String(_)), "endswith" | "startswith" | "lower") => Ok(V::Bound(Box::new(base), name.into())),
            _ => Err(Unknown),
        }
    }

    /// A site local: its assignments run in order, each when its own guard
    /// holds, with `=` replacing and `+=` adding.
    fn site_local(&mut self, name: &str, assignments: &'static [Value]) -> Ev<V> {
        if !self.pending.insert(name.to_string()) {
            return Err(Unknown);
        }
        let mut value: Option<V> = None;
        for assignment in assignments {
            let guard = assignment["when"].as_str().unwrap_or("always");
            let holds = guard == "always" || truthy(&self.expr(guard)?);
            if holds {
                let next = self.expr(assignment["value"].as_str().ok_or(Unknown)?)?;
                let next = match (assignment["op"].as_str(), &value) {
                    (Some("+="), Some(current)) => arith("+", current, &next)?,
                    (Some("+="), None) => return Err(Unknown),
                    _ => next,
                };
                self.vars.insert(name.to_string(), next.clone());
                value = Some(next);
            }
        }
        self.pending.remove(name);
        value.ok_or(Unknown)
    }

    /// converter_attributes: no_mtp and mtp_only are command-line switches,
    /// off by default. Any other self.<attr> runs the converter's
    /// attribute_assignments in order: each site's guard, then `=` or `+=`
    /// of its value, reading its own guarded locals. Each reference's phase
    /// places it in ModelBase.write's run order: sites after the key writes
    /// or never reached are skipped, and a site in set_gguf_parameters or an
    /// unknown helper makes the value unknown unless its guard is false.
    fn self_attribute(&mut self, name: &str) -> Ev<V> {
        if matches!(name, "no_mtp" | "mtp_only") {
            return Ok(V::J(Value::Bool(false)));
        }
        // GGUF-039: hf_arch is the top-level architectures[0] of the loaded
        // dict, before text_config is merged.
        if name == "hf_arch" {
            return self.ctx.hf_arch.clone().map(|arch| V::J(Value::String(arch))).ok_or(Unknown);
        }
        if name == "model_tensors" {
            let names = self.ctx.tensors.iter().map(|name| (name.clone(), Value::Null)).collect();
            return Ok(V::J(Value::Object(names)));
        }
        if let Some(assigned) = self.ctx.overrides.and_then(|overrides| overrides.get(name)) {
            return assigned.clone().ok_or(Unknown);
        }
        let refs = self.ctx.converter["attribute_assignments"].get(name).and_then(Value::as_array);
        let Some(refs) = refs else {
            // No index record (a unit test): block_count is TextModel's lookup.
            if name == "block_count" && self.ctx.converter.is_null() {
                return lookup(self.ctx.map, &["n_layers", "num_hidden_layers", "n_layer", "num_layers"])
                    .map(|value| V::J(value.clone()))
                    .ok_or(Unknown);
            }
            return Err(Unknown);
        };
        if self.depth > 16 {
            return Err(Unknown);
        }
        let sites = &index()["assignment_sites"];
        let mut value: Option<V> = None;
        for reference in refs {
            let site = &sites[reference["site"].as_str().ok_or(Unknown)?];
            let mut eval = Eval {
                ctx: self.ctx,
                locals: None,
                site_locals: site.get("locals").and_then(Value::as_object),
                vars: Default::default(),
                pending: Default::default(),
                depth: self.depth + 1,
            };
            // converter_attributes run_order: a phase after the key writes, or
            // an __init__ the chain never reaches, does not count.
            let phase = reference["phase"].as_str().unwrap_or("unknown");
            if matches!(phase, "never" | "after set_gguf_parameters") {
                continue;
            }
            // In a sequence walk, set_gguf_parameters assignments are events of
            // their own, applied where they run (the overrides above).
            if phase == "set_gguf_parameters" && self.ctx.overrides.is_some() {
                continue;
            }
            let guard = site["when"].as_str().unwrap_or("always");
            let holds = guard == "always" || truthy(&eval.expr(guard)?);
            if !holds {
                continue;
            }
            // In set_gguf_parameters it runs between writes, and an "unknown"
            // helper's callers decide, so its place in the order is not known.
            if !matches!(
                phase,
                "class attribute" | "init" | "prepare_tensors" | "prepare_metadata, before set_gguf_parameters"
            ) {
                return Err(Unknown);
            }
            let next = eval.expr(site["value"].as_str().ok_or(Unknown)?)?;
            value = Some(match (site["op"].as_str(), &value) {
                (Some("+="), Some(current)) => arith("+", current, &next)?,
                (Some("+="), None) => return Err(Unknown),
                _ => next,
            });
        }
        value.ok_or(Unknown)
    }

    fn call(&mut self, func: V, args: Vec<V>, kwargs: Vec<(String, V)>) -> Ev<V> {
        let json = |value: &V| match value {
            V::J(value) => Ok(value.clone()),
            _ => Err(Unknown),
        };
        match func {
            V::Bound(base, method) => match (*base, method.as_str()) {
                // dict.pop reads like get; on self.hparams after the pop
                // mutation ran, it gives the value that pop removed.
                (V::J(Value::Object(map)), "pop") => {
                    let key = args.first().map(json).transpose()?.ok_or(Unknown)?;
                    let key = key.as_str().ok_or(Unknown)?;
                    match map.get(key).cloned().or_else(|| self.ctx.popped.and_then(|popped| popped.get(key)).cloned()) {
                        Some(value) => Ok(V::J(value)),
                        None => args.get(1).cloned().ok_or(Unknown),
                    }
                }
                (V::J(Value::Object(map)), "get") => {
                    let key = args.first().map(json).transpose()?.ok_or(Unknown)?;
                    let key = key.as_str().ok_or(Unknown)?;
                    match map.get(key) {
                        Some(value) => Ok(V::J(value.clone())),
                        None => Ok(args.get(1).cloned().unwrap_or(V::J(Value::Null))),
                    }
                }
                (V::SelfObj, "find_hparam") => {
                    let names = args.first().map(json).transpose()?.ok_or(Unknown)?;
                    let optional = kwargs
                        .iter()
                        .find(|(name, _)| name == "optional")
                        .map(|(_, value)| truthy(value))
                        .unwrap_or(false);
                    for name in names.as_array().ok_or(Unknown)? {
                        if let Some(value) = name.as_str().and_then(|name| self.ctx.map.get(name)) {
                            return Ok(V::J(value.clone()));
                        }
                    }
                    if optional {
                        Ok(V::J(Value::Null))
                    } else {
                        Err(Halt::Raise(format!("find_hparam finds none of {names}")))
                    }
                }
                (V::Path(path), "is_file") => Ok(V::J(Value::Bool(path.is_file()))),
                (V::Path(path), "exists") => Ok(V::J(Value::Bool(path.exists()))),
                (V::J(Value::String(text)), "endswith" | "startswith") => {
                    let affix = args.first().map(json).transpose()?.ok_or(Unknown)?;
                    let affix = affix.as_str().ok_or(Unknown)?;
                    let hit = if method == "endswith" { text.ends_with(affix) } else { text.starts_with(affix) };
                    Ok(V::J(Value::Bool(hit)))
                }
                (V::J(Value::String(text)), "lower") => Ok(V::J(Value::String(text.to_lowercase()))),
                // kwargs.pop('hparams', None) and kwargs.get(...): no keyword is passed.
                (V::Builtin("kwargs"), "pop" | "get") => Ok(args.get(1).cloned().unwrap_or(V::J(Value::Null))),
                // A fresh GGUF-039 dict: renames applied, text_config not yet merged.
                (V::Builtin("ModelBase"), "load_hparams") => Ok(V::J(Value::Object(self.ctx.loaded.clone()))),
                // re.match anchors at the start of the string.
                (V::Builtin("re"), "match") => {
                    let pattern = args.first().map(json).transpose()?.ok_or(Unknown)?;
                    let text = args.get(1).map(json).transpose()?.ok_or(Unknown)?;
                    let (Some(pattern), Some(text)) = (pattern.as_str(), text.as_str()) else { return Err(Unknown) };
                    let compiled = regex::Regex::new(&format!("^(?:{pattern})")).map_err(|_| Unknown)?;
                    Ok(match compiled.captures(text) {
                        Some(caps) => V::Match(caps.iter().map(|group| group.map(|m| m.as_str().to_string())).collect()),
                        None => V::J(Value::Null),
                    })
                }
                (V::Match(groups), "group") => {
                    let index = match args.first() {
                        None => 0,
                        Some(V::J(Value::Number(number))) => number.as_u64().ok_or(Unknown)? as usize,
                        _ => return Err(Unknown),
                    };
                    match groups.get(index).ok_or(Unknown)? {
                        Some(text) => Ok(V::J(Value::String(text.clone()))),
                        None => Ok(V::J(Value::Null)),
                    }
                }
                _ => Err(Unknown),
            },
            V::Builtin(name) => {
                let first = args.first().map(json).transpose()?.ok_or(Unknown)?;
                match name {
                    "int" => match &first {
                        Value::Number(number) if number.is_i64() || number.is_u64() => Ok(V::J(first.clone())),
                        Value::Number(number) => Ok(V::J(Value::from(number.as_f64().ok_or(Unknown)?.trunc() as i64))),
                        Value::Bool(flag) => Ok(V::J(Value::from(i64::from(*flag)))),
                        Value::String(text) => text.trim().parse::<i64>().map(|number| V::J(Value::from(number))).map_err(|_| Unknown),
                        _ => Err(Unknown),
                    },
                    "float" => match &first {
                        Value::Number(number) => Ok(V::J(Value::from(number.as_f64().ok_or(Unknown)?))),
                        Value::String(text) => text.trim().parse::<f64>().map(|number| V::J(Value::from(number))).map_err(|_| Unknown),
                        _ => Err(Unknown),
                    },
                    "len" => match &first {
                        Value::Array(items) => Ok(V::J(Value::from(items.len()))),
                        Value::String(text) => Ok(V::J(Value::from(text.chars().count()))),
                        Value::Object(map) => Ok(V::J(Value::from(map.len()))),
                        _ => Err(Unknown),
                    },
"bool" => Ok(V::J(Value::Bool(truthy(&V::J(first))))),
                    "any" | "all" | "sum" | "list" | "sorted" | "_set" => {
                        let items: Vec<Value> = match first {
                            Value::Array(items) => items,
                            Value::Object(map) => map.keys().map(|key| Value::String(key.clone())).collect(),
                            _ => return Err(Unknown),
                        };
                        match name {
                            "any" => Ok(V::J(Value::Bool(items.iter().any(|item| truthy(&V::J(item.clone())))))),
                            "all" => Ok(V::J(Value::Bool(items.iter().all(|item| truthy(&V::J(item.clone())))))),
                            "sum" => {
                                let mut total = V::J(Value::from(0));
                                for item in items {
                                    total = arith("+", &total, &V::J(item))?;
                                }
                                Ok(total)
                            }
                            "list" => Ok(V::J(Value::Array(items))),
                            "_set" => {
                                let mut unique: Vec<Value> = Vec::new();
                                for item in items {
                                    if !unique.contains(&item) {
                                        unique.push(item);
                                    }
                                }
                                Ok(V::J(Value::Array(unique)))
                            }
                            _ => {
                                let mut items = items;
                                if items.iter().all(Value::is_string) {
                                    items.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
                                } else if items.iter().all(Value::is_number) {
                                    items.sort_by(|a, b| a.as_f64().partial_cmp(&b.as_f64()).unwrap_or(std::cmp::Ordering::Equal));
                                } else {
                                    return Err(Unknown);
                                }
                                Ok(V::J(Value::Array(items)))
                            }
                        }
                    }
                    "str" => match first {
                        Value::String(text) => Ok(V::J(Value::String(text))),
                        _ => Err(Unknown),
                    },
                    "abs" => match first.as_i64() {
                        Some(number) => Ok(V::J(Value::from(number.abs()))),
                        None => Ok(V::J(Value::from(first.as_f64().ok_or(Unknown)?.abs()))),
                    },
                    _ => {
                        // min and max of numbers, given separately or as one list.
                        let items: Vec<Value> = if args.len() == 1 {
                            first.as_array().cloned().ok_or(Unknown)?
                        } else {
                            args.iter().map(json).collect::<Ev<_>>()?
                        };
                        let mut best: Option<Value> = None;
                        for item in items {
                            let number = item.as_f64().ok_or(Unknown)?;
                            let better = match &best {
                                None => true,
                                Some(seen) => {
                                    let seen = seen.as_f64().ok_or(Unknown)?;
                                    if name == "min" { number < seen } else { number > seen }
                                }
                            };
                            if better {
                                best = Some(item);
                            }
                        }
                        best.map(V::J).ok_or(Unknown)
                    }
                }
            }
            _ => Err(Unknown),
        }
    }

    fn eval(&mut self, tree: &E) -> Ev<V> {
        match tree {
            E::Const(value) => Ok(V::J(value.clone())),
            E::Name(name) => self.name(name),
            E::Attr(base, name) => {
                let base = self.eval(base)?;
                self.attr(base, name)
            }
            E::Index(base, index) => {
                let base = self.eval(base)?;
                let index = self.eval(index)?;
                match (base, index) {
                    (V::Builtin("args"), V::J(Value::Number(number))) if number.as_u64() == Some(0) => {
                        Ok(V::Path(self.ctx.dir.to_path_buf()))
                    }
                    (V::J(Value::Object(map)), V::J(Value::String(key))) => {
                        map.get(&key).cloned().map(V::J).ok_or_else(|| Halt::Raise(format!("KeyError '{key}'")))
                    }
                    (V::J(Value::Array(items)), V::J(Value::Number(number))) => {
                        let at = number.as_i64().ok_or(Unknown)?;
                        let at = if at < 0 { items.len() as i64 + at } else { at };
                        usize::try_from(at)
                            .ok()
                            .and_then(|at| items.get(at))
                            .cloned()
                            .map(V::J)
                            .ok_or_else(|| Halt::Raise(format!("IndexError {at}")))
                    }
                    _ => Err(Unknown),
                }
            }
            E::Slice(base, start, end) => {
                let V::J(Value::Array(items)) = self.eval(base)? else { return Err(Unknown) };
                let bound = |this: &mut Self, part: &Option<Box<E>>, default: usize| -> Ev<usize> {
                    match part {
                        None => Ok(default),
                        Some(part) => match this.eval(part)? {
                            V::J(Value::Number(number)) => {
                                let at = number.as_i64().ok_or(Unknown)?;
                                let at = if at < 0 { (items.len() as i64 + at).max(0) } else { at };
                                Ok((at as usize).min(items.len()))
                            }
                            _ => Err(Unknown),
                        },
                    }
                };
                let from = bound(self, start, 0)?;
                let to = bound(self, end, items.len())?;
                Ok(V::J(Value::Array(items.get(from..to.max(from)).unwrap_or(&[]).to_vec())))
            }
            E::Call(func, args, kwargs) => {
                let func = self.eval(func)?;
                let args = args.iter().map(|arg| self.eval(arg)).collect::<Ev<Vec<_>>>()?;
                let kwargs = kwargs
                    .iter()
                    .map(|(name, arg)| Ok((name.clone(), self.eval(arg)?)))
                    .collect::<Ev<Vec<_>>>()?;
                self.call(func, args, kwargs)
            }
            E::Not(inner) => Ok(V::J(Value::Bool(!truthy(&self.eval(inner)?)))),
            E::Neg(inner) => match self.eval(inner)? {
                V::J(Value::Number(number)) => match number.as_i64() {
                    Some(number) => Ok(V::J(Value::from(-number))),
                    None => Ok(V::J(Value::from(-number.as_f64().ok_or(Unknown)?))),
                },
                _ => Err(Unknown),
            },
            E::Bin(op, left, right) => {
                let left = self.eval(left)?;
                let right = self.eval(right)?;
                arith(op, &left, &right)
            }
            E::Cmp(first, rest) => {
                let mut left = self.eval(first)?;
                for (op, right) in rest {
                    let right = self.eval(right)?;
                    if !compare(op, &left, &right)? {
                        return Ok(V::J(Value::Bool(false)));
                    }
                    left = right;
                }
                Ok(V::J(Value::Bool(true)))
            }
            // Three-valued: an operand that decides the result decides it even
            // when an earlier operand is unknown.
            E::And(items) | E::Or(items) => {
                let want = matches!(tree, E::Or(_));
                let mut unknown = false;
                let mut last = V::J(Value::Null);
                for item in items {
                    match self.eval(item) {
                        Ok(value) if truthy(&value) == want => return Ok(value),
                        Ok(value) => last = value,
                        // Python raises here only if no earlier operand ended
                        // the test, which an unknown one may have done.
                        Err(Halt::Raise(cause)) if !unknown => return Err(Halt::Raise(cause)),
                        Err(_) => unknown = true,
                    }
                }
                if unknown {
                    Err(Unknown)
                } else {
                    Ok(last)
                }
            }
            E::IfExp(then, cond, other) => match self.eval(cond) {
                Ok(value) if truthy(&value) => self.eval(then),
                Ok(_) => self.eval(other),
                Err(Halt::Raise(cause)) => Err(Halt::Raise(cause)),
                Err(_) => {
                    let then = self.eval(then).map_err(Halt::unsure)?;
                    if self.eval(other).map_err(Halt::unsure)? == then {
                        Ok(then)
                    } else {
                        Err(Unknown)
                    }
                }
            },
            E::Walrus(name, inner) => {
                let value = self.eval(inner)?;
                self.vars.insert(name.clone(), value.clone());
                Ok(value)
            }
            E::List(items) => Ok(V::J(Value::Array(
                items
                    .iter()
                    .map(|item| match self.eval(item)? {
                        V::J(value) => Ok(value),
                        _ => Err(Unknown),
                    })
                    .collect::<Ev<_>>()?,
            ))),
            E::Dict(items) => {
                let mut map = Map::new();
                for (key, value) in items {
                    let V::J(value) = self.eval(value)? else { return Err(Unknown) };
                    match key {
                        Some(key) => {
                            let V::J(Value::String(key)) = self.eval(key)? else { return Err(Unknown) };
                            map.insert(key, value);
                        }
                        None => {
                            let Value::Object(entries) = value else { return Err(Unknown) };
                            map.extend(entries);
                        }
                    }
                }
                Ok(V::J(Value::Object(map)))
            }
            E::Comp(elt, name, iter, cond) => {
                let items: Vec<Value> = match self.eval(iter)? {
                    V::J(Value::Array(items)) => items,
                    V::J(Value::Object(map)) => map.keys().map(|key| Value::String(key.clone())).collect(),
                    V::J(Value::String(text)) => text.chars().map(|c| Value::String(c.to_string())).collect(),
                    _ => return Err(Unknown),
                };
                let saved = self.vars.remove(name);
                let mut out = Vec::new();
                let mut result = Ok(());
                for item in items {
                    self.vars.insert(name.clone(), V::J(item));
                    let keep = match cond {
                        Some(cond) => self.eval(cond).map(|value| truthy(&value)),
                        None => Ok(true),
                    };
                    match keep.and_then(|keep| if keep { self.eval(elt).map(Some) } else { Ok(None) }) {
                        Ok(Some(V::J(value))) => out.push(value),
                        Ok(Some(_)) => {
                            result = Err(Unknown);
                            break;
                        }
                        Ok(None) => {}
                        Err(err) => {
                            result = Err(err);
                            break;
                        }
                    }
                }
                self.vars.remove(name);
                if let Some(saved) = saved {
                    self.vars.insert(name.clone(), saved);
                }
                result.map(|()| V::J(Value::Array(out)))
            }
            E::BinPath(left, right) => match (self.eval(left)?, self.eval(right)?) {
                (V::Path(path), V::J(Value::String(part))) => Ok(V::Path(path.join(part))),
                (left, right) => arith("/", &left, &right),
            },
        }
    }
}

fn number(value: &V) -> Ev<(Option<i64>, f64)> {
    match value {
        V::J(Value::Number(number)) => Ok((number.as_i64(), number.as_f64().ok_or(Unknown)?)),
        V::J(Value::Bool(flag)) => Ok((Some(i64::from(*flag)), f64::from(u8::from(*flag)))),
        _ => Err(Unknown),
    }
}

fn arith(op: &str, left: &V, right: &V) -> Ev<V> {
    if op == "+" {
        if let (V::J(Value::String(a)), V::J(Value::String(b))) = (left, right) {
            return Ok(V::J(Value::String(format!("{a}{b}"))));
        }
    }
    let (left_int, a) = number(left)?;
    let (right_int, b) = number(right)?;
    let ints = left_int.zip(right_int);
    let value = match (op, ints) {
        ("+", Some((x, y))) => Value::from(x.checked_add(y).ok_or(Unknown)?),
        ("-", Some((x, y))) => Value::from(x.checked_sub(y).ok_or(Unknown)?),
        ("*", Some((x, y))) => Value::from(x.checked_mul(y).ok_or(Unknown)?),
        ("//", Some((x, y))) if y != 0 => Value::from(x.div_euclid(y) - i64::from(x.rem_euclid(y) != 0 && y < 0)),
        ("%", Some((x, y))) if y != 0 => Value::from(x - y * (x as f64 / y as f64).floor() as i64),
        ("+", None) => Value::from(a + b),
        ("-", None) => Value::from(a - b),
        ("*", None) => Value::from(a * b),
        ("/", _) if b != 0.0 => Value::from(a / b),
        ("//", None) if b != 0.0 => Value::from((a / b).floor()),
        _ => return Err(Unknown),
    };
    Ok(V::J(value))
}

fn compare(op: &str, left: &V, right: &V) -> Ev<bool> {
    let (V::J(a), V::J(b)) = (left, right) else { return Err(Unknown) };
    let equal = |a: &Value, b: &Value| match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) if a.is_number() && b.is_number() => x == y,
        _ => a == b,
    };
    match op {
        "==" => Ok(equal(a, b)),
        "!=" => Ok(!equal(a, b)),
        // `is` is used with None, True, and False only.
        "is" => Ok(a == b && (b.is_null() || b.is_boolean())),
        "is not" => Ok(!(a == b && (b.is_null() || b.is_boolean()))),
        "in" | "not in" => {
            let hit = match b {
                Value::Array(items) => items.iter().any(|item| equal(item, a)),
                Value::Object(map) => a.as_str().is_some_and(|key| map.contains_key(key)),
                Value::String(text) => a.as_str().is_some_and(|part| text.contains(part)),
                _ => return Err(Unknown),
            };
            Ok(hit == (op == "in"))
        }
        "<" | ">" | "<=" | ">=" => {
            let (x, y) = (a.as_f64().ok_or(Unknown)?, b.as_f64().ok_or(Unknown)?);
            Ok(match op {
                "<" => x < y,
                ">" => x > y,
                "<=" => x <= y,
                _ => x >= y,
            })
        }
        _ => Err(Unknown),
    }
}

/// The Python expression subset the index uses for guards and values.
#[derive(Debug, Clone)]
enum E {
    Const(Value),
    Name(String),
    Attr(Box<E>, String),
    Index(Box<E>, Box<E>),
    Slice(Box<E>, Option<Box<E>>, Option<Box<E>>),
    Call(Box<E>, Vec<E>, Vec<(String, E)>),
    Not(Box<E>),
    Neg(Box<E>),
    Bin(&'static str, Box<E>, Box<E>),
    BinPath(Box<E>, Box<E>),
    Cmp(Box<E>, Vec<(&'static str, E)>),
    And(Vec<E>),
    Or(Vec<E>),
    IfExp(Box<E>, Box<E>, Box<E>),
    Walrus(String, Box<E>),
    List(Vec<E>),
    /// Pairs, and `**expr` unpacks (no key).
    Dict(Vec<(Option<E>, E)>),
    /// `elt for name in iter [if cond]`, as a list (a generator in a call).
    Comp(Box<E>, String, Box<E>, Option<Box<E>>),
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Name(String),
    Num(Value),
    Str(String),
    Op(&'static str),
}

fn lex(text: &str) -> Option<Vec<Tok>> {
    const OPS: &[&str] = &[
        ":=", "==", "!=", "<=", ">=", "//", "**", "(", ")", "[", "]", "{", "}", ",", ":", ".", "<", ">", "+", "-",
        "*", "/", "%", "&", "|",
    ];
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut at = 0;
    while at < chars.len() {
        let c = chars[at];
        if c.is_whitespace() {
            at += 1;
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = at;
            while at < chars.len() && (chars[at].is_ascii_alphanumeric() || chars[at] == '_') {
                at += 1;
            }
            tokens.push(Tok::Name(chars[start..at].iter().collect()));
        } else if c.is_ascii_digit() || (c == '.' && chars.get(at + 1).is_some_and(char::is_ascii_digit)) {
            let start = at;
            while at < chars.len()
                && (chars[at].is_ascii_alphanumeric() || chars[at] == '.' || chars[at] == '_'
                    || ((chars[at] == '-' || chars[at] == '+') && matches!(chars[at - 1], 'e' | 'E')))
            {
                at += 1;
            }
            let word: String = chars[start..at].iter().filter(|c| **c != '_').collect();
            let value = match word.parse::<i64>() {
                Ok(number) => Value::from(number),
                Err(_) => Value::from(word.parse::<f64>().ok()?),
            };
            tokens.push(Tok::Num(value));
        } else if c == '\'' || c == '"' {
            let mut out = String::new();
            at += 1;
            loop {
                let next = *chars.get(at)?;
                at += 1;
                if next == c {
                    break;
                }
                if next == '\\' {
                    let escaped = *chars.get(at)?;
                    at += 1;
                    // Python keeps the backslash of an escape it does not know, as in '\d'.
                    match escaped {
                        'n' => out.push('\n'),
                        't' => out.push('\t'),
                        '\\' | '\'' | '"' => out.push(escaped),
                        other => {
                            out.push('\\');
                            out.push(other);
                        }
                    }
                } else {
                    out.push(next);
                }
            }
            // Adjacent literals join, as in Python.
            if let Some(Tok::Str(previous)) = tokens.last_mut() {
                previous.push_str(&out);
            } else {
                tokens.push(Tok::Str(out));
            }
        } else {
            let op = OPS.iter().find(|op| {
                let len = op.chars().count();
                chars.get(at..at + len).is_some_and(|slice| slice.iter().copied().eq(op.chars()))
            });
            match op {
                Some(op) => {
                    at += op.chars().count();
                    tokens.push(Tok::Op(op));
                }
                None if c == '=' => {
                    at += 1;
                    tokens.push(Tok::Op("kw="));
                }
                None => return None,
            }
        }
    }
    Some(tokens)
}

struct Parser {
    tokens: Vec<Tok>,
    at: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.at)
    }

    fn eat_op(&mut self, op: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Op(seen)) if *seen == op) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn eat_word(&mut self, word: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Name(seen)) if seen == word) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    /// NAME ':=' test, or test.
    fn named(&mut self) -> Option<E> {
        if let (Some(Tok::Name(name)), Some(Tok::Op(":="))) = (self.tokens.get(self.at), self.tokens.get(self.at + 1)) {
            let name = name.clone();
            self.at += 2;
            return Some(E::Walrus(name, Box::new(self.test()?)));
        }
        self.test()
    }

    fn test(&mut self) -> Option<E> {
        let body = self.or_test()?;
        if self.eat_word("if") {
            let cond = self.or_test()?;
            if !self.eat_word("else") {
                return None;
            }
            let other = self.test()?;
            return Some(E::IfExp(Box::new(body), Box::new(cond), Box::new(other)));
        }
        Some(body)
    }

    fn or_test(&mut self) -> Option<E> {
        let mut items = vec![self.and_test()?];
        while self.eat_word("or") {
            items.push(self.and_test()?);
        }
        Some(if items.len() == 1 { items.pop()? } else { E::Or(items) })
    }

    fn and_test(&mut self) -> Option<E> {
        let mut items = vec![self.not_test()?];
        while self.eat_word("and") {
            items.push(self.not_test()?);
        }
        Some(if items.len() == 1 { items.pop()? } else { E::And(items) })
    }

    fn not_test(&mut self) -> Option<E> {
        if self.eat_word("not") {
            return Some(E::Not(Box::new(self.not_test()?)));
        }
        self.comparison()
    }

    fn comparison(&mut self) -> Option<E> {
        let first = self.arith()?;
        let mut rest = Vec::new();
        loop {
            let op = match self.peek() {
                Some(Tok::Op(op)) if matches!(*op, "==" | "!=" | "<" | ">" | "<=" | ">=") => {
                    let op = *op;
                    self.at += 1;
                    op
                }
                Some(Tok::Name(word)) if word == "in" => {
                    self.at += 1;
                    "in"
                }
                Some(Tok::Name(word)) if word == "not" && matches!(self.tokens.get(self.at + 1), Some(Tok::Name(next)) if next == "in") => {
                    self.at += 2;
                    "not in"
                }
                Some(Tok::Name(word)) if word == "is" => {
                    self.at += 1;
                    if self.eat_word("not") {
                        "is not"
                    } else {
                        "is"
                    }
                }
                _ => break,
            };
            rest.push((op, self.arith()?));
        }
        Some(if rest.is_empty() { first } else { E::Cmp(Box::new(first), rest) })
    }

    fn arith(&mut self) -> Option<E> {
        let mut left = self.term()?;
        loop {
            let op = if self.eat_op("+") {
                "+"
            } else if self.eat_op("-") {
                "-"
            } else {
                return Some(left);
            };
            left = E::Bin(op, Box::new(left), Box::new(self.term()?));
        }
    }

    fn term(&mut self) -> Option<E> {
        let mut left = self.factor()?;
        loop {
            if self.eat_op("//") {
                left = E::Bin("//", Box::new(left), Box::new(self.factor()?));
            } else if self.eat_op("/") {
                left = E::BinPath(Box::new(left), Box::new(self.factor()?));
            } else if self.eat_op("*") {
                left = E::Bin("*", Box::new(left), Box::new(self.factor()?));
            } else if self.eat_op("%") {
                left = E::Bin("%", Box::new(left), Box::new(self.factor()?));
            } else {
                return Some(left);
            }
        }
    }

    fn factor(&mut self) -> Option<E> {
        if self.eat_op("-") {
            return Some(E::Neg(Box::new(self.factor()?)));
        }
        if self.eat_op("+") {
            return self.factor();
        }
        let base = self.atom_expr()?;
        if matches!(self.peek(), Some(Tok::Op("**"))) {
            return None;
        }
        Some(base)
    }

    fn atom_expr(&mut self) -> Option<E> {
        let mut base = self.atom()?;
        loop {
            if self.eat_op(".") {
                let Some(Tok::Name(name)) = self.peek().cloned() else { return None };
                self.at += 1;
                base = E::Attr(Box::new(base), name);
            } else if self.eat_op("(") {
                let mut args = Vec::new();
                let mut kwargs = Vec::new();
                while !self.eat_op(")") {
                    if let Some(kw) = self.keyword_arg() {
                        kwargs.push(kw);
                    } else {
                        let arg = self.named()?;
                        args.push(self.comprehension(arg)?);
                    }
                    if !self.eat_op(",") {
                        if !self.eat_op(")") {
                            return None;
                        }
                        break;
                    }
                }
                base = E::Call(Box::new(base), args, kwargs);
            } else if self.eat_op("[") {
                let start = if matches!(self.peek(), Some(Tok::Op(":"))) { None } else { Some(Box::new(self.test()?)) };
                if self.eat_op(":") {
                    let end = if matches!(self.peek(), Some(Tok::Op("]"))) { None } else { Some(Box::new(self.test()?)) };
                    if !self.eat_op("]") {
                        return None;
                    }
                    base = E::Slice(Box::new(base), start, end);
                } else {
                    if !self.eat_op("]") {
                        return None;
                    }
                    base = E::Index(Box::new(base), start?);
                }
            } else {
                return Some(base);
            }
        }
    }

    /// After the element of a comprehension: `for NAME in iter [if cond]`.
    fn comprehension(&mut self, elt: E) -> Option<E> {
        if !self.eat_word("for") {
            return Some(elt);
        }
        let Some(Tok::Name(name)) = self.peek().cloned() else { return None };
        self.at += 1;
        if !self.eat_word("in") {
            return None;
        }
        let iter = self.or_test()?;
        let cond = if self.eat_word("if") { Some(Box::new(self.or_test()?)) } else { None };
        if matches!(self.peek(), Some(Tok::Name(word)) if word == "for" || word == "if") {
            return None;
        }
        Some(E::Comp(Box::new(elt), name, Box::new(iter), cond))
    }

    /// NAME '=' test inside a call, such as `optional=True`. The lexer gives
    /// a lone '=' as the token `kw=`.
    fn keyword_arg(&mut self) -> Option<(String, E)> {
        if let (Some(Tok::Name(name)), Some(Tok::Op("kw="))) = (self.tokens.get(self.at), self.tokens.get(self.at + 1)) {
            let name = name.clone();
            self.at += 2;
            let value = self.test()?;
            return Some((name, value));
        }
        None
    }

    fn atom(&mut self) -> Option<E> {
        let token = self.peek()?.clone();
        self.at += 1;
        match token {
            Tok::Num(value) => Some(E::Const(value)),
            Tok::Str(text) => Some(E::Const(Value::String(text))),
            Tok::Name(name) => match name.as_str() {
                "None" => Some(E::Const(Value::Null)),
                "True" => Some(E::Const(Value::Bool(true))),
                "False" => Some(E::Const(Value::Bool(false))),
                "not" | "and" | "or" | "in" | "is" | "if" | "else" | "for" | "lambda" => None,
                _ => Some(E::Name(name)),
            },
            Tok::Op("(") => {
                if self.eat_op(")") {
                    return Some(E::List(Vec::new()));
                }
                let first = self.named()?;
                if matches!(self.peek(), Some(Tok::Name(word)) if word == "for") {
                    let comp = self.comprehension(first)?;
                    return self.eat_op(")").then_some(comp);
                }
                if self.eat_op(")") {
                    return Some(first);
                }
                let mut items = vec![first];
                while self.eat_op(",") {
                    if self.eat_op(")") {
                        return Some(E::List(items));
                    }
                    items.push(self.named()?);
                }
                if !self.eat_op(")") {
                    return None;
                }
                Some(E::List(items))
            }
            Tok::Op("[") => {
                let mut items = Vec::new();
                while !self.eat_op("]") {
                    let item = self.named()?;
                    if matches!(self.peek(), Some(Tok::Name(word)) if word == "for") {
                        let comp = self.comprehension(item)?;
                        return self.eat_op("]").then_some(comp);
                    }
                    items.push(item);
                    if !self.eat_op(",") {
                        if !self.eat_op("]") {
                            return None;
                        }
                        break;
                    }
                }
                Some(E::List(items))
            }
            Tok::Op("{") => {
                let mut items = Vec::new();
                while !self.eat_op("}") {
                    if self.eat_op("**") {
                        items.push((None, self.or_test()?));
                        if !self.eat_op(",") {
                            if !self.eat_op("}") {
                                return None;
                            }
                            break;
                        }
                        continue;
                    }
                    let key = self.test()?;
                    // A set comprehension: its items, without repeats.
                    if items.is_empty() && matches!(self.peek(), Some(Tok::Name(word)) if word == "for") {
                        let comp = self.comprehension(key)?;
                        return self.eat_op("}").then(|| E::Call(Box::new(E::Name("_set".into())), vec![comp], Vec::new()));
                    }
                    if !self.eat_op(":") {
                        return None;
                    }
                    items.push((Some(key), self.test()?));
                    if !self.eat_op(",") {
                        if !self.eat_op("}") {
                            return None;
                        }
                        break;
                    }
                }
                Some(E::Dict(items))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(text: &str) -> Map<String, Value> {
        serde_json::from_str::<Value>(text).unwrap().as_object().unwrap().clone()
    }

    fn rows(text: &str) -> Vec<(String, MetaValue)> {
        let mut out = Vec::new();
        base_rows("t", &config(text), &mut out).unwrap();
        out
    }

    fn get<'a>(out: &'a [(String, MetaValue)], key: &str) -> Option<&'a MetaValue> {
        out.iter().find(|(name, _)| name == key).map(|(_, value)| value)
    }

    fn eval_in(config_text: &str, expr: &str) -> Ev<V> {
        let map = config(config_text);
        let (p, _) = rope_params(Some(&map));
        let dir = std::path::Path::new("/nonexistent");
        let ctx = Ctx { map: &map, p: &p, dir, arch: "t", converter: &Value::Null, tensors: &[], hf_arch: None, loaded: &map, overrides: None, popped: None, depth: 0 };
        let mut eval = Eval { ctx: &ctx, locals: None, site_locals: None, vars: Default::default(), pending: Default::default(), depth: 0 };
        eval.expr(expr)
    }

    #[test]
    fn the_evaluator_reads_the_config_and_admits_what_it_cannot() {
        let config = r#"{"a": 3, "b": null, "f": 2.5, "s": "Yarn", "l": [1, 2, 3]}"#;
        let yes = |expr: &str| assert_eq!(eval_in(config, expr), Ok(V::J(Value::Bool(true))), "{expr}");
        yes("self.hparams['a'] == 3");
        yes("self.hparams.get('b') is None");
        yes("self.hparams.get('missing', 7) == 7");
        yes("(x := self.find_hparam(['zz', 'a'], optional=True)) is not None and x // 2 == 1");
        yes("self.find_hparam(['zz'], optional=True) is None");
        yes("int(self.hparams['f'] * 2) == 5");
        yes("self.hparams['s'].lower() in ('yarn', 'linear')");
        yes("len(self.hparams['l'][:2]) == 2 and self.hparams['l'][-1] == 3");
        yes("'a' in self.hparams and 'zz' not in self.hparams");
        yes("not (self.dir_model / 'x.json').is_file()");
        yes("(1 if self.hparams['a'] > 2 else 0) == 1");
        yes("self.is_mistral_format is False");
        // A KeyError, an unknown attribute, or a missing find_hparam is unknown.
        // A missing key or a non-optional find_hparam raises, as in llama.cpp.
        assert_eq!(eval_in(config, "self.hparams['zz']"), Err(Halt::Raise("KeyError 'zz'".into())));
        assert_eq!(eval_in(config, "self.something_else"), Err(Unknown));
        assert!(matches!(eval_in(config, "self.find_hparam(['zz'])"), Err(Halt::Raise(_))));
        // A raise after an unknown operand may not happen.
        assert_eq!(eval_in(config, "self.unknown or self.hparams['zz']"), Err(Unknown));
        assert_eq!(eval_in(config, "gguf.PoolingType.LAST"), Err(Unknown));
        yes("[x * 2 for x in self.hparams['l'] if x > 1] == [4, 6]");
        yes("any(t != 'sparse' for t in ['sparse', 'dense'])");
        yes("not all(t > 1 for t in self.hparams['l'])");
        yes("sum(x for x in self.hparams['l']) == 6");
        yes("sorted(self.hparams['l']) == [1, 2, 3]");
        yes("re.match('lay(\\d+)', 'lay12x').group(1) == '12'");
        yes("re.match('x', 'ax') is None");
        yes("len({x // 2 for x in self.hparams['l']}) == 2");
        assert!(matches!(eval_in(config, "[x for x in self.hparams['zz']]"), Err(Halt::Raise(_))));
        assert_eq!(eval_in(config, "[x for x in l for y in l]"), Err(Unknown));
        // An unknown operand does not hide a decided one.
        assert_eq!(eval_in(config, "self.unknown and False"), Ok(V::J(Value::Bool(false))));
        assert_eq!(eval_in(config, "self.unknown or self.hparams['a']"), Ok(V::J(Value::from(3))));
        assert_eq!(eval_in(config, "self.unknown and True"), Err(Unknown));
    }

    #[test]
    fn a_class_follows_its_chain_from_the_index() {
        let dir = std::path::Path::new("/nonexistent");
        let map = config(r#"{"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2, "vocab_size": 9}"#);
        let llama = class_keys("LlamaModel", "llama", &map, dir).unwrap();
        // LlamaModel adds vocab_size to the base rows (not a Mistral-format checkpoint).
        assert!(llama.keys.iter().any(|(key, value)| key == "llama.vocab_size" && *value == MetaValue::U32(9)));
        assert!(llama.keys.iter().any(|(key, _)| key == "llama.block_count"));
        // rope.dimension_count is hidden_size // num_attention_heads without head_dim.
        let map = config(r#"{"num_hidden_layers": 2, "hidden_size": 576, "num_attention_heads": 9, "vocab_size": 9}"#);
        let llama = class_keys("LlamaModel", "llama", &map, dir).unwrap();
        assert!(llama.keys.iter().any(|(key, value)| key == "llama.rope.dimension_count" && *value == MetaValue::U32(64)));
        assert!(llama.unevaluated.is_empty(), "{:?}", llama.unevaluated);
        let qwen = class_keys("Qwen2Model", "qwen2", &map, dir).unwrap();
        assert!(!qwen.keys.iter().any(|(key, _)| key == "qwen2.vocab_size"));
        // A chain that never reaches TextModel writes none of the base rows.
        // RWKV6Qwen2Model writes a constant context length and no rope keys.
        let map = config(r#"{"num_hidden_layers": 2, "hidden_size": 8, "max_position_embeddings": 32, "rope_theta": 1.0}"#);
        let rwkv = class_keys("RWKV6Qwen2Model", "rwkv6qwen2", &map, dir).unwrap();
        assert!(rwkv.keys.iter().any(|(key, value)| key == "rwkv6qwen2.context_length" && *value == MetaValue::U32(1048576)));
        assert!(!rwkv.keys.iter().any(|(key, _)| key == "rwkv6qwen2.rope.freq_base"));
        assert!(class_keys("NoSuchModel", "x", &map, dir).is_err());
    }

    #[test]
    fn phi3_writes_its_own_keys_and_attn_factor() {
        let dir = std::path::Path::new("/nonexistent");
        let half = "[1.0, 1.0]";
        let text = format!(
            r#"{{"max_position_embeddings": 131072, "original_max_position_embeddings": 4096, "hidden_size": 8,
                "intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "vocab_size": 9,
                "rope_scaling": {{"type": "longrope", "long_factor": {half}, "short_factor": {half}}}}}"#
        );
        let map = config(&text);
        let phi = class_keys("Phi3MiniModel", "phi3", &map, dir).unwrap();
        assert!(phi.unevaluated.is_empty(), "{:?}", phi.unevaluated);
        let get = |key: &str| phi.keys.iter().find(|(name, _)| name == key).map(|(_, value)| value.clone());
        assert_eq!(get("phi3.rope.scaling.original_context_length"), Some(MetaValue::U32(4096)));
        assert_eq!(get("phi3.rope.scaling.attn_factor"), Some(MetaValue::F32(1.190_238_1)));
        assert_eq!(get("phi3.rope.dimension_count"), Some(MetaValue::U32(4)));
        assert_eq!(get("phi3.attention.sliding_window"), Some(MetaValue::U32(0)));
        // The base never runs: no scaling type or factor, no vocab_size.
        assert_eq!(get("phi3.rope.scaling.type"), None);
        assert_eq!(get("phi3.vocab_size"), None);
        // Factor lists of the wrong length stop the conversion, as in llama.cpp.
        let bad = config(&text.replace(r#""short_factor": [1.0, 1.0]"#, r#""short_factor": [1.0]"#));
        assert!(class_keys("Phi3MiniModel", "phi3", &bad, dir).is_err());
        let moe = config(&text.replace("\"vocab_size\": 9", "\"num_local_experts\": 16, \"num_experts_per_tok\": 2"));
        let moe = class_keys("PhiMoeModel", "phimoe", &moe, dir).unwrap();
        assert!(moe.keys.iter().any(|(key, value)| key == "phimoe.expert_count" && *value == MetaValue::U32(16)));
        assert!(moe.keys.iter().any(|(key, _)| key == "phimoe.rope.scaling.attn_factor"));
    }

    #[test]
    fn gemma2_and_gemma3_write_softcapping() {
        let dir = std::path::Path::new("/nonexistent");
        let text = r#"{"max_position_embeddings": 8192, "hidden_size": 8, "num_hidden_layers": 2,
            "intermediate_size": 16, "num_attention_heads": 2, "num_key_value_heads": 1, "rms_norm_eps": 1e-6,
            "head_dim": 4, "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0,
            "sliding_window": 4096, "rope_theta": 10000.0, "vocab_size": 9}"#;
        let gemma2 = class_keys("Gemma2Model", "gemma2", &config(text), dir).unwrap();
        let get = |found: &ClassKeys, key: &str| found.keys.iter().find(|(name, _)| name == key).map(|(_, value)| value.clone());
        assert_eq!(get(&gemma2, "gemma2.attn_logit_softcapping"), Some(MetaValue::F32(50.0)));
        assert_eq!(get(&gemma2, "gemma2.final_logit_softcapping"), Some(MetaValue::F32(30.0)));
        assert_eq!(get(&gemma2, "gemma2.attention.sliding_window"), Some(MetaValue::U32(4096)));
        assert_eq!(get(&gemma2, "gemma2.rope.freq_base"), None);
        assert!(class_keys("Gemma2Model", "gemma2", &config(&text.replace("\"head_dim\": 4,", "")), dir).is_err());

        let text3 = text.replace("\"attn_logit_softcapping\": 50.0, ", "");
        let gemma3 = class_keys("Gemma3Model", "gemma3", &config(&text3), dir).unwrap();
        assert_eq!(get(&gemma3, "gemma3.final_logit_softcapping"), Some(MetaValue::F32(30.0)));
        assert_eq!(get(&gemma3, "gemma3.rope.freq_base"), Some(MetaValue::F32(10000.0)));
        assert_eq!(get(&gemma3, "gemma3.attention.sliding_window"), Some(MetaValue::U32(4096)));
        // The base rows ran first.
        assert_eq!(get(&gemma3, "gemma3.feed_forward_length"), Some(MetaValue::U32(16)));
        let off = text3.replace("\"final_logit_softcapping\": 30.0", "\"final_logit_softcapping\": null, \"sliding_window_pattern\": 1");
        let gemma3 = class_keys("Gemma3Model", "gemma3", &config(&off), dir).unwrap();
        assert_eq!(get(&gemma3, "gemma3.final_logit_softcapping"), None);
        assert_eq!(get(&gemma3, "gemma3.attention.sliding_window"), None);
        assert!(class_keys("Gemma3Model", "gemma3", &config(text), dir).is_err());
    }

    #[test]
    fn sampling_keys_come_from_generation_config() {
        let dir = tempfile::tempdir().unwrap();
        assert!(sampling_keys(dir.path()).0.is_empty());
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"do_sample": false, "repetition_penalty": 1.1, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                "min_p": null, "mirostat": "x", "eos_token_id": [1, 2]}"#,
        )
        .unwrap();
        let (keys, warning) = sampling_keys(dir.path());
        let names: Vec<&str> = keys.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(
            names,
            ["general.sampling.top_k", "general.sampling.top_p", "general.sampling.temp", "general.sampling.penalty_repeat"]
        );
        assert_eq!(keys[0].1, MetaValue::I32(20));
        assert_eq!(keys[3].1, MetaValue::F32(1.1));
        assert!(warning.unwrap().contains("mirostat"));
        std::fs::write(dir.path().join("generation_config.json"), "not json").unwrap();
        assert!(sampling_keys(dir.path()).0.is_empty());
    }

    #[test]
    fn a_write_before_super_is_overwritten_by_the_base() {
        // DistilBertModel writes layer_norm_epsilon 1e-12 and then calls super(),
        // whose base row replaces it when the config gives a value.
        let dir = std::path::Path::new("/nonexistent");
        let get = |text: &str| {
            class_keys("DistilBertModel", "bert", &config(text), dir)
                .unwrap()
                .keys
                .into_iter()
                .find(|(key, _)| key == "bert.attention.layer_norm_epsilon")
                .map(|(_, value)| value)
        };
        assert_eq!(get(r#"{"n_layers": 2, "layer_norm_eps": 1e-5}"#), Some(MetaValue::F32(1e-5)));
        assert_eq!(get(r#"{"n_layers": 2}"#), Some(MetaValue::F32(1e-12)));
    }

    #[test]
    fn pooling_follows_the_helper_transcription() {
        let dir = tempfile::tempdir().unwrap();
        let base = r#"{"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2}"#;
        let get = |found: &ClassKeys, key: &str| found.keys.iter().find(|(name, _)| name == key).map(|(_, value)| value.clone());
        // A plain causal LM writes no pooling_type, and nothing is left unevaluated.
        let plain = class_keys("Qwen2Model", "qwen2", &config(base), dir.path()).unwrap();
        assert_eq!(get(&plain, "qwen2.pooling_type"), None);
        assert!(plain.unevaluated.is_empty(), "{:?}", plain.unevaluated);
        std::fs::write(
            dir.path().join("modules.json"),
            r#"[{"type": "sentence_transformers.models.Transformer", "path": ""},
                {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"}]"#,
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("1_Pooling")).unwrap();
        std::fs::write(dir.path().join("1_Pooling/config.json"), r#"{"pooling_mode_lasttoken": true}"#).unwrap();
        let embed = class_keys("Qwen2Model", "qwen2", &config(base), dir.path()).unwrap();
        assert_eq!(get(&embed, "qwen2.pooling_type"), Some(MetaValue::U32(3)));
        std::fs::write(dir.path().join("1_Pooling/config.json"), r#"{"pooling_mode": "max"}"#).unwrap();
        assert!(class_keys("Qwen2Model", "qwen2", &config(base), dir.path()).is_err());
    }

    #[test]
    fn arrays_are_typed_as_gguf_py_types_them() {
        use crate::formats::gguf::{TYPE_ARRAY, TYPE_F32, TYPE_I32, TYPE_U32};
        let list = |text: &str| V::J(serde_json::from_str(text).unwrap());
        let elem = |value: Ev<Option<MetaValue>>| match value {
            Ok(Some(MetaValue::Array { elem, .. })) => elem,
            other => panic!("{other:?}"),
        };
        // A per-layer list of ints is INT32, not UINT32, unless the writer fixes the type.
        assert_eq!(elem(to_meta(&list("[1, 2]"), "u32 or array")), TYPE_I32);
        assert_eq!(elem(to_meta(&list("[1, 2]"), "array[u32]")), TYPE_U32);
        assert_eq!(elem(to_meta(&list("[0.5, 1.0]"), "array")), TYPE_F32);
        assert_eq!(elem(to_meta(&list("[[1], [2]]"), "array")), TYPE_ARRAY);
        assert_eq!(to_meta(&list("[]"), "array"), Ok(None));
        assert_eq!(to_meta(&list("[]"), "array[u32]"), Err(Unknown));
        assert_eq!(to_meta(&list("[1, 0.5]"), "array"), Err(Unknown));
        assert_eq!(to_meta(&list("true"), "u32 or array"), Ok(Some(MetaValue::U32(1))));
        assert_eq!(to_meta(&list("7"), "u32 or array"), Ok(Some(MetaValue::U32(7))));
    }

    #[test]
    fn self_attributes_come_from_the_converter_record() {
        let dir = std::path::Path::new("/nonexistent");
        let map = config(r#"{"num_hidden_layers": 4, "hidden_size": 8, "num_attention_heads": 2, "logit_scale": 0.5, "sliding_window": 4096, "vocab_size": 10, "head_dim": 4}"#);
        // Cohere2MoeModel adds next-token layers with `+= n_nextn` when there are any.
        let get = |found: &ClassKeys| found.keys.iter().find(|(key, _)| key.ends_with(".block_count")).map(|(_, value)| value.clone());
        let cohere = class_keys("Cohere2MoeModel", "cohere2moe", &map, dir).unwrap();
        assert_eq!(get(&cohere), Some(MetaValue::U32(4)));
        let nextn = config(r#"{"num_hidden_layers": 4, "hidden_size": 8, "num_attention_heads": 2, "num_nextn_predict_layers": 1, "logit_scale": 0.5, "sliding_window": 4096, "vocab_size": 10, "head_dim": 4}"#);
        let cohere = class_keys("Cohere2MoeModel", "cohere2moe", &nextn, dir).unwrap();
        assert_eq!(get(&cohere), Some(MetaValue::U32(5)), "{:?}", cohere.unevaluated);
        // NemotronHPuzzleModel calls GraniteHybridModel.__init__ directly, so
        // NemotronH's MTP step never runs for it.
        let (p, _) = rope_params(Some(&nextn));
        let record = &index()["converters"]["NemotronHPuzzleModel"];
        let ctx = Ctx { map: &nextn, p: &p, dir, arch: "nemotron_h", converter: record, tensors: &[], hf_arch: None, loaded: &nextn, overrides: None, popped: None, depth: 0 };
        let mut eval = Eval { ctx: &ctx, locals: None, site_locals: None, vars: Default::default(), pending: Default::default(), depth: 0 };
        assert_eq!(eval.self_attribute("block_count"), Ok(V::J(Value::from(4))));
        // DeepseekV4DSparkModel counts the layers in the filtered tensor names.
        let record = &index()["converters"]["DeepseekV4DSparkModel"];
        let names = ["layers.0.attn.weight".to_string(), "layers.2.ffn.weight".to_string(), "embed.weight".to_string()];
        let ctx = Ctx { map: &nextn, p: &p, dir, arch: "x", converter: record, tensors: &names, hf_arch: None, loaded: &nextn, overrides: None, popped: None, depth: 0 };
        let mut eval = Eval { ctx: &ctx, locals: None, site_locals: None, vars: Default::default(), pending: Default::default(), depth: 0 };
        assert_eq!(eval.self_attribute("block_count"), Ok(V::J(Value::from(3))));
        // Qwen3.5's MTP mixin: num_hidden_layers, then += mtp_num_hidden_layers.
        let mtp = config(r#"{"num_hidden_layers": 4, "hidden_size": 8, "num_attention_heads": 2, "mtp_num_hidden_layers": 1}"#);
        let (p, _) = rope_params(Some(&mtp));
        let record = &index()["converters"]["Qwen3_5TextModel"];
        let ctx = Ctx { map: &mtp, p: &p, dir, arch: "qwen35", converter: record, tensors: &[], hf_arch: None, loaded: &mtp, overrides: None, popped: None, depth: 0 };
        let mut eval = Eval { ctx: &ctx, locals: None, site_locals: None, vars: Default::default(), pending: Default::default(), depth: 0 };
        assert_eq!(eval.self_attribute("block_count"), Ok(V::J(Value::from(5))));
        // Without a key its writes read, llama.cpp's converter raises, and so does tensconv.
        let err = class_keys("Qwen3_5TextModel", "qwen35", &mtp, dir).err().unwrap().to_string();
        assert!(err.contains("would stop at") && err.contains("KeyError"), "{err}");
        // Qwen2Model keeps TextModel's block_count.
        let qwen = class_keys("Qwen2Model", "qwen2", &map, dir).unwrap();
        assert!(qwen.keys.iter().any(|(key, value)| key == "qwen2.block_count" && *value == MetaValue::U32(4)));
    }

    #[test]
    fn converters_change_the_config_dict_before_the_keys() {
        let dir = std::path::Path::new("/nonexistent");
        let get = |found: &ClassKeys, key: &str| found.keys.iter().find(|(name, _)| name == key).map(|(_, value)| value.clone());
        // TextModel.__init__ merges text_config over the loaded dict.
        let loaded = config(r#"{"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2,
            "text_config": {"hidden_size": 16}}"#);
        let qwen = class_keys("Qwen2Model", "qwen2", &loaded, dir).unwrap();
        assert_eq!(get(&qwen, "qwen2.embedding_length"), Some(MetaValue::U32(16)));
        assert!(qwen.unapplied.is_empty(), "{:?}", qwen.unapplied);
        // CommandR2Model sets max_position_embeddings from model_max_length.
        let command = config(r#"{"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2,
            "max_position_embeddings": 4096, "model_max_length": 131072, "logit_scale": 0.5}"#);
        let command = class_keys("CommandR2Model", "command-r", &command, dir).unwrap();
        assert_eq!(get(&command, "command-r.context_length"), Some(MetaValue::U32(131072)));
        // Dots1Model copies n_routed_experts in __init__: without it llama.cpp
        // raises KeyError, and tensconv stops too.
        let dots = config(r#"{"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2}"#);
        let err = class_keys("Dots1Model", "dots1", &dots, dir).err().unwrap().to_string();
        assert!(err.contains("Dots1Model.__init__:20") && err.contains("KeyError 'n_routed_experts'"), "{err}");
        // NemotronHModel sets num_hidden_layers from layers_block_type in the
        // local it passes on as hparams=.
        let nemotron = config(r#"{"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2,
            "layers_block_type": ["mamba", "attention", "mamba"]}"#);
        let record = &index()["converters"]["NemotronHModel"];
        let mut unapplied = Vec::new();
        let changed = apply_mutations("NemotronHModel", &nemotron, dir, "nemotron_h", record, &[], None, &mut unapplied).unwrap();
        assert_eq!(changed.get("num_hidden_layers"), Some(&Value::from(3)), "{unapplied:?}");
        // With experts and a next-token layer, is_moe (has_moe_params from a
        // fresh load_hparams) is true and the MTP step adds one layer.
        let moe = config(r#"{"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2,
            "layers_block_type": ["mamba", "attention", "mamba"], "num_experts_per_tok": 2, "num_nextn_predict_layers": 1}"#);
        let changed = apply_mutations("NemotronHModel", &moe, dir, "nemotron_h", record, &[], None, &mut unapplied).unwrap();
        let (p, _) = rope_params(Some(&changed));
        let ctx = Ctx { map: &changed, p: &p, dir, arch: "nemotron_h", converter: record, tensors: &[], hf_arch: None, loaded: &moe, overrides: None, popped: None, depth: 0 };
        let mut eval = Eval { ctx: &ctx, locals: None, site_locals: None, vars: Default::default(), pending: Default::default(), depth: 0 };
        assert_eq!(eval.self_attribute("block_count"), Ok(V::J(Value::from(4))));
    }

    #[test]
    fn changes_inside_set_gguf_parameters_apply_where_they_run() {
        // HunYuanModel pops num_experts, calls the base, then puts it back only
        // when it is > 1: the base writes no expert_count.
        let dir = std::path::Path::new("/nonexistent");
        let get = |found: &ClassKeys, key: &str| found.keys.iter().find(|(name, _)| name == key).map(|(_, value)| value.clone());
        let one = config(r#"{"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2, "num_experts": 1}"#);
        let dense = class_keys("HunYuanModel", "hunyuan-dense", &one, dir).unwrap();
        assert_eq!(get(&dense, "hunyuan-dense.expert_count"), None, "{:?}", dense.unapplied);
        assert!(dense.unapplied.is_empty(), "{:?}", dense.unapplied);
    }

    #[test]
    fn every_index_expression_lexes_or_is_unknown() {
        // The evaluator must never panic on the index; unparsable text is unknown.
        let map = config(r#"{"num_hidden_layers": 2}"#);
        let (p, _) = rope_params(Some(&map));
        let ctx = Ctx { map: &map, p: &p, dir: std::path::Path::new("/nonexistent"), arch: "t", converter: &Value::Null, tensors: &[], hf_arch: None, loaded: &map, overrides: None, popped: None, depth: 0 };
        let mut total = 0;
        for record in index()["methods"].as_object().unwrap().values() {
            let locals = record.get("locals").and_then(Value::as_object);
            for write in record["writes"].as_array().into_iter().flatten() {
                for field in ["when", "value"] {
                    if let Some(text) = write[field].as_str() {
                        let mut eval = Eval { ctx: &ctx, locals, site_locals: None, vars: Default::default(), pending: Default::default(), depth: 0 };
                        let _ = eval.expr(text);
                        total += 1;
                    }
                }
            }
        }
        assert!(total > 3000, "{total}");
    }

    #[test]
    fn a_null_stops_the_lookup() {
        // context_length: max_position_embeddings is null, so n_ctx is not tried.
        let out = rows(r#"{"num_layers": 1, "max_position_embeddings": null, "n_ctx": 4096}"#);
        assert!(get(&out, "t.context_length").is_none());
        let mut out = Vec::new();
        let err = base_rows("t", &config(r#"{"n_layers": null, "num_layers": 2}"#), &mut out).unwrap_err();
        assert!(err.to_string().contains("n_layers is null"), "{err}");
    }

    #[test]
    fn p_is_filled_as_rope_parameters_fill_says() {
        let (p, _) = rope_params(Some(&config(
            r#"{"rope_parameters": null, "rope_scaling": {"type": "linear", "factor": 2.0}, "rope_theta": 5.0}"#,
        )));
        // A null rope_parameters gives {} and does not fall back to rope_scaling.
        assert!(!p.contains_key("factor"));
        assert_eq!(p.get("rope_theta"), Some(&Value::from(5.0)));
        let (p, _) = rope_params(Some(&config(
            r#"{"rope_scaling": {"type": "yarn", "rope_type": null}, "rope_local_base_freq": 10.0, "rope_pct": 0.5}"#,
        )));
        // A present rope_type wins, even null; the local theta becomes sliding_attention.
        assert_eq!(p.get("rope_type"), Some(&Value::Null));
        assert_eq!(p["sliding_attention"]["rope_theta"], Value::from(10.0));
        assert_eq!(p.get("partial_rotary_factor"), Some(&Value::from(0.5)));
        let (p, r) = rope_params(Some(&config(
            r#"{"rope_parameters": {"full_attention": {"rope_type": "default"}}, "rope_theta": 5.0, "rope_local_base_freq": 1.0}"#,
        )));
        // A nested P is not filled.
        assert!(!p.contains_key("sliding_attention") && !r.contains_key("rope_theta"));
    }

    #[test]
    fn the_listed_transformers_defaults_are_added() {
        let merged = crate::convert::merge_text_config(&Value::Object(config(
            r#"{"model_type": "llama", "hidden_size": 576, "num_attention_heads": 9, "torch_dtype": "bfloat16"}"#,
        )));
        assert_eq!(merged.get("head_dim"), Some(&Value::from(64)));
        assert_eq!(merged.get("max_length"), Some(&Value::from(20)));
        assert_eq!(merged.get("dtype"), Some(&Value::from("bfloat16")));
        let merged = crate::convert::merge_text_config(&Value::Object(config(
            r#"{"model_type": "qwen2", "use_sliding_window": false, "sliding_window": 32768}"#,
        )));
        assert_eq!(merged.get("sliding_window"), Some(&Value::Null));
        // qwen2's absent num_key_value_heads is the constant 32; llama's copies the heads.
        assert_eq!(merged.get("num_key_value_heads"), Some(&Value::from(32)));
        let merged = crate::convert::merge_text_config(&Value::Object(config(
            r#"{"model_type": "llama", "num_attention_heads": 9, "hidden_size": 576}"#,
        )));
        assert_eq!(merged.get("num_key_value_heads"), Some(&Value::from(9)));
        assert_eq!(merged.get("tie_word_embeddings"), Some(&Value::Bool(false)));
        let merged = crate::convert::merge_text_config(&Value::Object(config(
            r#"{"model_type": "qwen3", "num_hidden_layers": 4, "use_sliding_window": true, "sliding_window": 128, "max_window_layers": 2}"#,
        )));
        assert_eq!(merged.get("head_dim"), Some(&Value::from(128)));
        assert_eq!(merged["layer_types"][1], Value::from("full_attention"));
        assert_eq!(merged["layer_types"][2], Value::from("sliding_attention"));
        let merged = crate::convert::merge_text_config(&Value::Object(config(
            r#"{"model_type": "gemma3_text", "num_hidden_layers": 6, "use_bidirectional_attention": true}"#,
        )));
        assert_eq!(merged.get("sliding_window"), Some(&Value::from(2049)));
        // A text_config gets its own defaults and the common keys, and they win.
        let merged = crate::convert::merge_text_config(&Value::Object(config(
            r#"{"model_type": "gemma3", "architectures": ["Gemma3ForConditionalGeneration"], "eos_token_id": [1, 106],
                "text_config": {"model_type": "gemma3_text", "hidden_size": 8}}"#,
        )));
        assert_eq!(merged.get("eos_token_id"), Some(&Value::from(1)));
        assert_eq!(merged.get("architectures"), Some(&Value::Null));
        assert_eq!(merged.get("model_type"), Some(&Value::from("gemma3_text")));
        assert_eq!(merged["layer_types"][5], Value::from("full_attention"));
        // qwen2_5_vl builds its text_config from the top-level keys it does not take.
        let merged = crate::convert::merge_text_config(&Value::Object(config(
            r#"{"model_type": "qwen2_5_vl", "hidden_size": 896, "num_hidden_layers": 2, "image_token_id": 7,
                "architectures": ["Qwen2_5_VLForConditionalGeneration"]}"#,
        )));
        assert_eq!(merged.get("hidden_size"), Some(&Value::from(896)));
        assert_eq!(merged.get("model_type"), Some(&Value::from("qwen2_5_vl_text")));
        assert_eq!(merged.get("max_window_layers"), Some(&Value::from(80)));
        assert_eq!(merged.get("sliding_window"), Some(&Value::Null));
        assert!(defaults_notes(&Value::Object(config(r#"{"model_type": "qwen2_5_vl"}"#))).iter().all(|note| !note.contains("text_config")));
        // llava takes nothing from the top level: its text_config is llama's defaults.
        let merged = crate::convert::merge_text_config(&Value::Object(config(r#"{"model_type": "llava", "hidden_size": 8}"#)));
        assert_eq!(merged.get("hidden_size"), Some(&Value::from(4096)));
        let merged = crate::convert::merge_text_config(&Value::Object(config(
            r#"{"model_type": "x", "llm_config": {"hidden_size": 7}, "thinker_config": {"text_config": {"hidden_size": 9}}}"#,
        )));
        // thinker_config comes after llm_config, so it is the text_config merged.
        assert_eq!(merged.get("hidden_size"), Some(&Value::from(9)));
        assert!(merged.contains_key("llm_config"));
        let merged = crate::convert::merge_text_config(&Value::Object(config(
            r#"{"model_type": "x", "text_config": {"hidden_size": 1}, "thinker_config": {"text_config": {"hidden_size": 9}}, "lfm": {"hidden_size": 5}}"#,
        )));
        assert_eq!(merged.get("hidden_size"), Some(&Value::from(5)));
        assert!(crate::convert::text_config_problem(&Value::Object(config(r#"{"thinker_config": {}}"#))).is_some());
        assert!(crate::convert::text_config_problem(&Value::Object(config(r#"{"lm_config": null}"#))).is_some());
        assert!(crate::convert::text_config_problem(&Value::Object(config(r#"{"text_config": {}}"#))).is_none());
    }

    #[test]
    fn the_base_rows_follow_text_model() {
        let out = rows(
            r#"{"num_hidden_layers": 2, "num_attention_heads": 8, "vocab_size": 9, "sliding_window": 4,
                "use_sliding_window": false, "rope_theta": 10000.0}"#,
        );
        // No vocab_size, sliding_window, rope.dimension_count, or a defaulted head_count_kv.
        let keys: Vec<&str> = out.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(keys, ["t.block_count", "t.attention.head_count", "t.rope.freq_base"]);

        let out = rows(
            r#"{"num_layers": 1, "rope_scaling": {"type": "yarn", "factor": 4.0, "beta_fast": 32},
                "original_max_position_embeddings": 4096, "is_causal": false, "scoring_func": "sigmoid",
                "head_dim": 64}"#,
        );
        assert!(matches!(get(&out, "t.rope.scaling.type"), Some(MetaValue::String(kind)) if kind == "yarn"));
        // The top-level original_max_position_embeddings is copied into P.
        assert!(matches!(get(&out, "t.rope.scaling.original_context_length"), Some(MetaValue::U32(4096))));
        assert!(matches!(get(&out, "t.rope.scaling.yarn_beta_fast"), Some(MetaValue::F32(value)) if *value == 32.0));
        assert!(matches!(get(&out, "t.attention.causal"), Some(MetaValue::Bool(false))));
        assert!(matches!(get(&out, "t.expert_gating_func"), Some(MetaValue::U32(2))));
        assert!(matches!(get(&out, "t.attention.key_length"), Some(MetaValue::U32(64))));

        let out = rows(r#"{"num_layers": 1, "rope_scaling": {"rope_type": "longrope", "long_factor": [1.0]}}"#);
        assert!(matches!(get(&out, "t.rope.scaling.type"), Some(MetaValue::String(kind)) if kind == "longrope"));
        assert!(get(&out, "t.rope.scaling.factor").is_none());
        let out = rows(r#"{"num_layers": 1, "rope_scaling": {"rope_type": "llama3", "factor": 8.0}}"#);
        assert!(get(&out, "t.rope.scaling.type").is_none());

        let out = rows(r#"{"num_layers": 1, "rope_parameters": {"mrope_section": [16, 24, 24]}}"#);
        let Some(MetaValue::Array { values, .. }) = get(&out, "t.rope.dimension_sections") else { panic!() };
        assert_eq!(values.len(), 4);
        assert!(matches!(values[3], MetaValue::I32(0)));

        let mut out = Vec::new();
        assert!(base_rows("t", &config(r#"{"hidden_size": 8}"#), &mut out).is_err());
        assert!(base_rows("t", &config(r#"{"num_layers": 1, "score_func": "tanh"}"#), &mut out).is_err());
    }
}
