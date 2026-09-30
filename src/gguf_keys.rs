//! The GGUF hyperparameter keys that tensconv --rename writes from
//! config.json. The base rows transcribe TextModel.set_gguf_parameters
//! (metadata-mapping hf_config_to_gguf).

use serde_json::{Map, Value};

use crate::error::{msg, Result};
use crate::formats::gguf::MetaValue;

/// metadata-mapping hf_config_to_gguf: P is config rope_parameters, else
/// rope_scaling, else empty. When P has neither full_attention nor
/// sliding_attention, TextModel.__init__ copies type to rope_type and fills
/// rope_theta, partial_rotary_factor, and original_max_position_embeddings
/// from the top-level config when P lacks them. R is P.full_attention when
/// present, else P.
pub(crate) fn rope_params(hparams: Option<&Map<String, Value>>) -> (Map<String, Value>, Map<String, Value>) {
    let pick = |key: &str| {
        hparams
            .and_then(|map| map.get(key))
            .and_then(Value::as_object)
            .cloned()
    };
    let mut p = pick("rope_parameters").or_else(|| pick("rope_scaling")).unwrap_or_default();
    if !p.contains_key("full_attention") && !p.contains_key("sliding_attention") {
        if !p.contains_key("rope_type") {
            if let Some(kind) = p.get("type").cloned() {
                p.insert("rope_type".into(), kind);
            }
        }
        if let Some(map) = hparams {
            if !p.contains_key("rope_theta") {
                if let Some(theta) = first_present(map, ROPE_THETA) {
                    p.insert("rope_theta".into(), theta.clone());
                }
            }
            for key in ["partial_rotary_factor", "original_max_position_embeddings"] {
                if !p.contains_key(key) {
                    if let Some(value) = map.get(key).filter(|value| !value.is_null()) {
                        p.insert(key.into(), value.clone());
                    }
                }
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

/// find_hparam(names, optional=True): the first name present and not null.
pub(crate) fn first_present<'a>(map: &'a Map<String, Value>, names: &[&str]) -> Option<&'a Value> {
    names
        .iter()
        .find_map(|name| map.get(*name).filter(|value| !value.is_null()))
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
    if first_present(map, &blocks).is_none() {
        return Err(msg(
            "config.json has none of n_layers, num_hidden_layers, n_layer, num_layers; llama.cpp's converter stops there too",
        ));
    }
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
        .filter(|value| !value.is_null())
        .or_else(|| first_present(map, LOCAL_ROPE_THETA));
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
