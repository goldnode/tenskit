//! Hugging Face and GGUF tensor names from `tables/gguf-tensor-roles.json`.
//!
//! Resolution follows conversion-considerations.md section 9. `{bid}` is the
//! block index. `.weight` and `.bias` are tried when the pattern has neither.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use serde_json::Value;

use crate::archive::Archive;

const ROLES_JSON: &str =
    include_str!("../tensdataspecs_readonly/specs/tables/gguf-tensor-roles.json");

struct Catalog {
    arches: Vec<Resolved>,
    groups: Vec<Vec<String>>,
}

struct Resolved {
    id: String,
    pairs: Vec<(String, String)>,
    /// Role id to the GGUF pattern, `{bid}` still in place.
    roles: Vec<(String, String)>,
}

pub struct NameHit {
    pub arches: Vec<String>,
    pub pairs: Vec<(String, String)>,
}

pub fn match_free(
    left: &Archive,
    right: &Archive,
    left_free: &[String],
    right_free: &[String],
) -> NameHit {
    let catalog = catalog();
    let declared = declared_architecture(left).or_else(|| declared_architecture(right));
    let lang_layers = language_layers(left)
        .or_else(|| language_layers(right))
        .or_else(|| nextn_block(left_free))
        .or_else(|| nextn_block(right_free));
    let mut names = Vec::new();
    for name in left_free.iter().chain(right_free) {
        if !is_dropped(name) {
            names.push(name.clone());
        }
    }
    let arches = match declared {
        Some(arch) if catalog.arches.iter().any(|item| item.id == arch) => vec![arch],
        _ => detect(catalog, &names),
    };
    if arches.is_empty() {
        return NameHit {
            arches,
            pairs: Vec::new(),
        };
    }
    let forward = project(catalog, &arches, left_free, right_free, lang_layers, false);
    let backward = project(catalog, &arches, right_free, left_free, lang_layers, true);
    let pairs = if backward.len() > forward.len() {
        backward
    } else {
        forward
    };
    NameHit { arches, pairs }
}

pub fn note(arches: &[String]) -> String {
    let mut arches = arches.to_vec();
    arches.sort();
    arches.dedup();
    match arches.as_slice() {
        [] => "no table matched".to_string(),
        [one] => one.clone(),
        [first, second] => format!("may be {first} or {second}"),
        many => {
            let last = many.last().unwrap();
            let head = many[..many.len() - 1].join(", ");
            format!("may be {head}, or {last}")
        }
    }
}

fn catalog() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(load_catalog)
}

fn load_catalog() -> Catalog {
    let value: Value = serde_json::from_str(ROLES_JSON)
        .unwrap_or_else(|err| panic!("gguf-tensor-roles.json: {err}"));
    let mappings = rows(value.get("mappings"));
    let block_mappings = rows(value.get("block_mappings"));
    let overrides = value
        .get("overrides")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut arches = Vec::new();
    let object = value
        .get("architectures")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("gguf-tensor-roles.json has no architectures"));
    for (id, entry) in object {
        let roles: HashSet<&str> = entry
            .get("roles")
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let mut block = block_mappings.clone();
        if let Some(rows) = overrides.get(id).and_then(Value::as_array) {
            for row in row_slice(rows) {
                if let Some(pos) = block.iter().position(|item| item.tensor == row.tensor) {
                    block[pos] = row;
                } else {
                    block.push(row);
                }
            }
        }
        let mut pairs: HashMap<String, String> = HashMap::new();
        let mut role_gguf: HashMap<String, String> = HashMap::new();
        record(&mut pairs, &mut role_gguf, &mappings, &roles);
        record(&mut pairs, &mut role_gguf, &block, &roles);
        arches.push(Resolved {
            id: id.clone(),
            pairs: pairs.into_iter().collect(),
            roles: role_gguf.into_iter().collect(),
        });
    }
    arches.sort_by(|left, right| left.id.cmp(&right.id));
    let groups = value
        .get("shared_resolutions")
        .and_then(Value::as_array)
        .map(|groups| {
            groups
                .iter()
                .filter_map(|group| {
                    group
                        .get("architectures")
                        .and_then(Value::as_array)
                        .map(|ids| {
                            ids.iter()
                                .filter_map(|id| id.as_str().map(str::to_string))
                                .collect()
                        })
                })
                .collect()
        })
        .unwrap_or_default();
    Catalog { arches, groups }
}

#[derive(Clone)]
struct Row {
    tensor: String,
    gguf: String,
    hf: Vec<String>,
}

fn rows(value: Option<&Value>) -> Vec<Row> {
    value
        .and_then(Value::as_array)
        .map(|items| row_slice(items))
        .unwrap_or_default()
}

fn row_slice(items: &[Value]) -> Vec<Row> {
    items
        .iter()
        .filter_map(Value::as_object)
        .filter_map(|row| {
            Some(Row {
                tensor: row.get("tensor")?.as_str()?.to_string(),
                gguf: row.get("gguf")?.as_str()?.to_string(),
                hf: row
                    .get("hf")?
                    .as_array()?
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect(),
            })
        })
        .collect()
}

fn record(
    pairs: &mut HashMap<String, String>,
    role_gguf: &mut HashMap<String, String>,
    rows: &[Row],
    roles: &HashSet<&str>,
) {
    for row in rows {
        if !roles.contains(row.tensor.as_str()) {
            continue;
        }
        role_gguf.insert(row.tensor.clone(), row.gguf.clone());
        for hf in &row.hf {
            pairs.insert(hf.clone(), row.gguf.clone());
        }
    }
}

/// GGUF pattern for one role of `arch`. `{bid}` is left for the caller to fill.
pub fn role_pattern(arch: &str, role: &str) -> Option<String> {
    catalog()
        .arches
        .iter()
        .find(|item| item.id == arch)
        .and_then(|item| {
            item.roles
                .iter()
                .find(|(tensor, _)| tensor == role)
                .map(|(_, pattern)| pattern.clone())
        })
}

/// Hugging Face tensor name to the GGUF name for one architecture.
pub fn map_tensor_name(arch: &str, name: &str, lang_layers: Option<u64>) -> Option<String> {
    let catalog = catalog();
    let resolved = catalog.arches.iter().find(|item| item.id == arch)?;
    let prepared = prepare_name(name, arch, lang_layers);
    lookup(&resolved.pairs, &prepared)
}

#[derive(Clone, Debug)]
pub struct TensorBranch {
    pub when: String,
    pub assigns: String,
    pub expr: String,
}

#[derive(Clone, Debug)]
pub struct ConverterRecord {
    pub class: String,
    pub hf_architectures: Vec<String>,
    pub undo_permute: bool,
    pub complete: bool,
    pub unexpanded: Vec<String>,
    pub branches: Vec<TensorBranch>,
    /// `filter_tensors`, own or inherited. None when the record has none.
    pub filter: Option<TensorFilter>,
    /// `tensor_hooks` method names: code outside `modify_tensors` that
    /// changes the tensor set (GGUF-030).
    pub hooks: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct TensorFilter {
    pub complete: bool,
    pub drop_when: Vec<String>,
}

/// `modify_tensors_base.tensor_loop.drop_suffixes`: names that
/// `ModelBase.prepare_tensors` skips before `modify_tensors`.
pub fn base_drop_suffixes() -> Vec<String> {
    let value: Value = serde_json::from_str(ROLES_JSON).unwrap_or(Value::Null);
    value
        .pointer("/modify_tensors_base/tensor_loop/drop_suffixes")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

pub fn converter_records(arch: &str) -> Vec<ConverterRecord> {
    let value: Value = serde_json::from_str(ROLES_JSON).unwrap_or(Value::Null);
    let Some(list) = value
        .get("architectures")
        .and_then(|item| item.get(arch))
        .and_then(|item| item.get("converters"))
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    list.iter()
        .filter_map(|item| {
            let class = item.get("class")?.as_str()?.to_string();
            let hf_architectures = item
                .get("hf_architectures")
                .and_then(Value::as_array)
                .map(|ids| {
                    ids.iter()
                        .filter_map(|id| id.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            let modify = item.get("modify_tensors");
            let complete = modify
                .and_then(|item| item.get("complete"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let unexpanded = modify
                .and_then(|item| item.get("unexpanded"))
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            let branches = modify
                .and_then(|item| item.get("branches"))
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| {
                            Some(TensorBranch {
                                when: item.get("when")?.as_str()?.to_string(),
                                assigns: item.get("assigns")?.as_str()?.to_string(),
                                expr: item.get("expr")?.as_str()?.to_string(),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            let filter = item.get("filter_tensors").map(|filter| TensorFilter {
                complete: filter
                    .get("complete")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                drop_when: filter
                    .get("drop_when")
                    .and_then(Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|item| item.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default(),
            });
            let mut hooks: Vec<String> = item
                .get("tensor_hooks")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|hook| hook.get("method")?.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            hooks.dedup();
            Some(ConverterRecord {
                class,
                hf_architectures,
                undo_permute: item.get("undo_permute").and_then(Value::as_bool).unwrap_or(false),
                complete,
                unexpanded,
                branches,
                filter,
                hooks,
            })
        })
        .collect()
}

/// `architectures[0]` to the GGUF architecture id that lists that class.
pub fn architecture_for_hf_class(class: &str) -> Option<String> {
    let value: Value = serde_json::from_str(ROLES_JSON).ok()?;
    let object = value.get("architectures")?.as_object()?;
    for (arch, entry) in object {
        let Some(list) = entry.get("converters").and_then(Value::as_array) else {
            continue;
        };
        for converter in list {
            let hit = converter
                .get("hf_architectures")
                .and_then(Value::as_array)
                .is_some_and(|ids| ids.iter().any(|id| id.as_str() == Some(class)));
            if hit {
                return Some(arch.clone());
            }
        }
    }
    None
}

fn detect(catalog: &Catalog, names: &[String]) -> Vec<String> {
    if names.is_empty() {
        return Vec::new();
    }
    let mut hits: Vec<&Resolved> = catalog
        .arches
        .iter()
        .filter(|arch| names.iter().all(|name| explains(arch, name)))
        .collect();
    if hits.is_empty() {
        return Vec::new();
    }
    let smallest = hits.iter().map(|arch| arch.pairs.len()).min().unwrap();
    hits.retain(|arch| arch.pairs.len() == smallest);
    let mut ids: Vec<String> = hits.iter().map(|arch| arch.id.clone()).collect();
    if let Some(group) = catalog.groups.iter().find(|group| {
        group.len() == ids.len() && group.iter().all(|id| ids.iter().any(|hit| hit == id))
    }) {
        ids = group.clone();
    }
    ids.sort();
    ids
}

fn explains(arch: &Resolved, name: &str) -> bool {
    let prepared = prepare_name(name, &arch.id, Some(0));
    lookup(&arch.pairs, &prepared).is_some() || matches_gguf(&arch.pairs, name)
}

fn project(
    catalog: &Catalog,
    arches: &[String],
    hf_names: &[String],
    gguf_names: &[String],
    lang_layers: Option<u64>,
    flip: bool,
) -> Vec<(String, String)> {
    let gguf_set: HashSet<&str> = gguf_names.iter().map(String::as_str).collect();
    let mut used = HashSet::new();
    let mut pairs = Vec::new();
    for name in hf_names {
        if is_dropped(name) {
            continue;
        }
        let mut mapped = None;
        let mut agree = true;
        for arch_id in arches {
            let Some(arch) = catalog.arches.iter().find(|arch| arch.id == *arch_id) else {
                agree = false;
                break;
            };
            let prepared = prepare_name(name, &arch.id, lang_layers);
            match lookup(&arch.pairs, &prepared) {
                Some(gguf) if gguf_set.contains(gguf.as_str()) => {
                    if mapped.as_ref().is_some_and(|prior| prior != &gguf) {
                        agree = false;
                        break;
                    }
                    mapped = Some(gguf);
                }
                _ => {
                    agree = false;
                    break;
                }
            }
        }
        if agree {
            if let Some(gguf) = mapped {
                if !used.insert(gguf.clone()) {
                    continue;
                }
                if flip {
                    pairs.push((gguf, name.clone()));
                } else {
                    pairs.push((name.clone(), gguf));
                }
            }
        }
    }
    pairs
}

fn lookup(pairs: &[(String, String)], name: &str) -> Option<String> {
    if let Some(found) = best_match(pairs, name) {
        return Some(found);
    }
    for suffix in [".weight", ".bias"] {
        let Some(stem) = name.strip_suffix(suffix) else {
            continue;
        };
        if let Some(found) = best_match(pairs, stem) {
            return Some(format!("{found}{suffix}"));
        }
    }
    None
}

fn best_match(pairs: &[(String, String)], name: &str) -> Option<String> {
    let mut best: Option<(usize, String)> = None;
    for (pattern, gguf) in pairs {
        let Some(bid) = capture_bid(pattern, name) else {
            continue;
        };
        let width = pattern.len();
        if best.as_ref().is_none_or(|(prior, _)| width > *prior) {
            best = Some((width, fill_bid(gguf, &bid)));
        }
    }
    best.map(|(_, gguf)| gguf)
}

fn matches_gguf(pairs: &[(String, String)], name: &str) -> bool {
    pairs.iter().any(|(_, pattern)| {
        gguf_hit(pattern, name)
            || [".weight", ".bias"].iter().any(|suffix| {
                name.strip_suffix(suffix)
                    .is_some_and(|stem| gguf_hit(pattern, stem))
            })
    })
}

fn gguf_hit(pattern: &str, name: &str) -> bool {
    capture_bid(pattern, name).is_some()
}

fn capture_bid(pattern: &str, name: &str) -> Option<String> {
    match pattern.split_once("{bid}") {
        None => (pattern == name).then(String::new),
        Some((prefix, suffix)) => {
            if suffix.contains("{bid}") || !name.starts_with(prefix) || !name.ends_with(suffix) {
                return None;
            }
            let mid_end = name.len() - suffix.len();
            if mid_end < prefix.len() {
                return None;
            }
            let mid = &name[prefix.len()..mid_end];
            is_index(mid).then(|| mid.to_string())
        }
    }
}

fn fill_bid(pattern: &str, bid: &str) -> String {
    pattern.replace("{bid}", bid)
}

fn is_index(text: &str) -> bool {
    !text.is_empty()
        && text.bytes().all(|byte| byte.is_ascii_digit())
        && (text == "0" || !text.starts_with('0'))
}

/// TextModel drops multimodal tensors. The `model.` strip is only for that test.
fn is_dropped(name: &str) -> bool {
    let test = name.strip_prefix("model.").unwrap_or(name);
    const STARTS: &[&str] = &[
        "mlp",
        "vit.",
        "vpm.",
        "siglip2.",
        "conformer.",
        "connector.",
        "merger.",
        "resampler.",
        "sound_encoder.",
        "sound_projection.",
        "speech_embeddings.",
    ];
    if STARTS.iter().any(|prefix| test.starts_with(prefix)) {
        return true;
    }
    const PARTS: &[&str] = &[
        "visual.",
        "vision.",
        "audio.",
        "talker.",
        "vision_",
        "audio_",
        "token2wav.",
        "code2wav.",
        "projector.",
        "pre_mm_projector_norm",
        "image_newline",
        "view_seperator",
        "patch_embed",
        "patch_embedding",
        "patch_merger.",
        "patch_merge_mlp.",
    ];
    PARTS.iter().any(|part| test.contains(part))
}

fn prepare_name(name: &str, arch: &str, lang_layers: Option<u64>) -> String {
    let mut name = name.to_string();
    if name.contains("language_model.") {
        name = name.replace("language_model.", "");
    }
    if name.ends_with("e_score_correction_bias") {
        name = name.replace("e_score_correction_bias", "e_score_correction.bias");
    }
    if arch == "qwen35" {
        name = qwen35_rewrite(&name, lang_layers);
    }
    name
}

/// Checkpoint names from qwen35_tensor_names_to_gguf that TensorNameMap sees
/// only after conversion/qwen.py rewrites them.
fn qwen35_rewrite(name: &str, lang_layers: Option<u64>) -> String {
    let name = if let Some(rest) = name.strip_prefix("model.mtp.") {
        format!("mtp.{rest}")
    } else {
        name.to_string()
    };
    if let Some(stem) = name.strip_suffix(".dt_bias") {
        return format!("{stem}.dt_proj.bias");
    }
    let Some(layers) = lang_layers else {
        return name;
    };
    match name.as_str() {
        "mtp.fc.weight" => return format!("model.layers.{layers}.eh_proj.weight"),
        "mtp.pre_fc_norm_embedding.weight" => {
            return format!("model.layers.{layers}.enorm.weight");
        }
        "mtp.pre_fc_norm_hidden.weight" => return format!("model.layers.{layers}.hnorm.weight"),
        "mtp.norm.weight" => {
            return format!("model.layers.{layers}.shared_head.norm.weight");
        }
        _ => {}
    }
    if let Some(rest) = name.strip_prefix("mtp.layers.") {
        if let Some((index, tail)) = rest.split_once('.') {
            if is_index(index) {
                let layer = layers + index.parse::<u64>().unwrap_or(0);
                return format!("model.layers.{layer}.{tail}");
            }
        }
    }
    name
}

fn declared_architecture(archive: &Archive) -> Option<String> {
    archive
        .metadata
        .iter()
        .find(|(key, _)| key == "general.architecture")
        .map(|(_, value)| value.clone())
        .filter(|value| !value.is_empty())
}

fn nextn_block(names: &[String]) -> Option<u64> {
    names
        .iter()
        .filter_map(|name| {
            let rest = name.strip_prefix("blk.")?;
            let (index, tail) = rest.split_once('.')?;
            (tail.starts_with("nextn.") && is_index(index))
                .then(|| index.parse().ok())
                .flatten()
        })
        .min()
}

fn language_layers(archive: &Archive) -> Option<u64> {
    let block = meta_u64(archive, "qwen35.block_count")?;
    let nextn = meta_u64(archive, "qwen35.nextn_predict_layers")?;
    block.checked_sub(nextn)
}

fn meta_u64(archive: &Archive, key: &str) -> Option<u64> {
    archive
        .metadata
        .iter()
        .find(|(name, _)| name == key)
        .and_then(|(_, value)| value.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapped(arch: &str, name: &str) -> String {
        let catalog = catalog();
        let resolved = catalog
            .arches
            .iter()
            .find(|item| item.id == arch)
            .unwrap_or_else(|| panic!("missing {arch}"));
        let prepared = prepare_name(name, arch, Some(24));
        lookup(&resolved.pairs, &prepared).unwrap_or_else(|| panic!("{arch} missed {name}"))
    }

    #[test]
    fn post_attention_layernorm_depends_on_the_architecture() {
        assert_eq!(
            mapped("llama", "model.layers.3.post_attention_layernorm.weight"),
            "blk.3.ffn_norm.weight"
        );
        assert_eq!(
            mapped(
                "qwen35",
                "model.language_model.layers.3.post_attention_layernorm.weight"
            ),
            "blk.3.post_attention_norm.weight"
        );
        assert_eq!(
            mapped(
                "qwen35",
                "model.language_model.layers.0.linear_attn.dt_bias"
            ),
            "blk.0.ssm_dt.bias"
        );
        assert_eq!(
            mapped("qwen35", "mtp.fc.weight"),
            "blk.24.nextn.eh_proj.weight"
        );
    }

    #[test]
    fn identical_pattern_sets_stay_one_group() {
        let catalog = catalog();
        let group = catalog
            .groups
            .iter()
            .find(|group| group.iter().any(|id| id == "refact"))
            .unwrap();
        let counts: Vec<usize> = group
            .iter()
            .map(|id| {
                catalog
                    .arches
                    .iter()
                    .find(|arch| arch.id == *id)
                    .unwrap()
                    .pairs
                    .len()
            })
            .collect();
        assert!(counts.iter().all(|count| *count == counts[0]), "{counts:?}");
        assert_eq!(counts[0], 308, "shared_resolutions pattern_count");
    }

    #[test]
    fn a_shared_set_is_reported_as_every_architecture() {
        let catalog = catalog();
        let refact = catalog
            .arches
            .iter()
            .find(|arch| arch.id == "refact")
            .unwrap();
        let names: Vec<String> = refact
            .pairs
            .iter()
            .map(|(pattern, _)| pattern.replace("{bid}", "0"))
            .collect();
        let hits = detect(catalog, &names);
        assert!(hits.contains(&"refact".to_string()), "{hits:?}");
        assert!(hits.contains(&"qwen2vl".to_string()), "{hits:?}");
        assert!(hits.len() > 1, "{hits:?}");
    }
}
