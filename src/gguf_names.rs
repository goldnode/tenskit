//! Hugging Face and GGUF tensor names from `tables/gguf-tensor-roles.json`.
//!
//! Resolution follows conversion-considerations.md section 9. `{bid}` is the
//! block index. `.weight` and `.bias` are tried when the pattern has neither.

use std::collections::HashSet;

use crate::archive::Archive;

/// One architecture: the Hugging Face pattern to GGUF pattern pairs and the
/// role to GGUF pattern list, resolved by `build.rs` from `mappings`,
/// `block_mappings`, and this architecture's `overrides` (rows whose tensor
/// is one of its roles, last write winning), and its converter records.
pub(crate) struct ArchTable {
    pub(crate) id: &'static str,
    /// Indices into `PAIRS`.
    pub(crate) pairs: Pairs,
    /// Indices into `ROLE_PATTERNS`: role id to the GGUF pattern, `{bid}`
    /// still in place.
    pub(crate) roles: Pairs,
    pub(crate) converters: &'static [ConverterSpec],
}

/// A list of (key, value) string pairs held as indices into a shared table,
/// so the architectures that share rows store them once.
#[derive(Clone, Copy)]
pub(crate) struct Pairs {
    table: &'static [(&'static str, &'static str)],
    at: &'static [u16],
}

impl Pairs {
    fn iter(self) -> impl Iterator<Item = (&'static str, &'static str)> {
        self.at.iter().map(move |index| self.table[*index as usize])
    }

    fn len(self) -> usize {
        self.at.len()
    }
}

/// A converter record of `architectures.<id>.converters`, as the spec has it.
pub(crate) struct ConverterSpec {
    pub(crate) class: &'static str,
    pub(crate) hf_architectures: &'static [&'static str],
    pub(crate) undo_permute: Option<bool>,
    pub(crate) modify: ModifySpec,
    pub(crate) filter: Option<FilterSpec>,
    pub(crate) global_filters_after: &'static [&'static str],
    pub(crate) tensor_hooks: Option<&'static [HookSpec]>,
}

pub(crate) struct ModifySpec {
    pub(crate) complete: bool,
    pub(crate) unexpanded: Option<&'static [&'static str]>,
    pub(crate) branches: Option<&'static [BranchSpec]>,
}

pub(crate) struct BranchSpec {
    pub(crate) when: &'static str,
    pub(crate) assigns: &'static str,
    pub(crate) expr: &'static str,
}

pub(crate) struct FilterSpec {
    pub(crate) complete: bool,
    pub(crate) drop_when: Option<&'static [&'static str]>,
    /// Each link's `drop_when`.
    pub(crate) chain: Option<&'static [Option<&'static [&'static str]>]>,
}

pub(crate) struct HookSpec {
    pub(crate) method: &'static str,
    pub(crate) class: &'static str,
    pub(crate) line: u64,
}

/// `global_filters.TextModel`.
pub(crate) struct TextModelFilter {
    pub(crate) strip_when: &'static str,
    pub(crate) strip_expr: &'static str,
    pub(crate) drop_when: &'static str,
    pub(crate) complete: Option<bool>,
}

mod roles {
    use super::*;
    include!("../generated/tensor_roles.rs");
}

struct Catalog {
    arches: &'static [ArchTable],
    groups: &'static [&'static [&'static str]],
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
    static CATALOG: Catalog = Catalog { arches: roles::ARCHES, groups: roles::SHARED_GROUPS };
    &CATALOG
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
                .find(|(tensor, _)| *tensor == role)
                .map(|(_, pattern)| pattern.to_string())
        })
}

/// Hugging Face tensor name to the GGUF name for one architecture.
pub fn map_tensor_name(arch: &str, name: &str, lang_layers: Option<u64>) -> Option<String> {
    let catalog = catalog();
    let resolved = catalog.arches.iter().find(|item| item.id == arch)?;
    let prepared = prepare_name(name, arch, lang_layers);
    lookup(resolved.pairs, &prepared)
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
    /// The class and line of the `generate_extra_tensors` hook, which key
    /// its `gguf_generated_tensors` row (GGUF-035).
    pub generator: Option<(String, u64)>,
    /// `global_filters_after`: the global filters that run after the class's
    /// own chain, such as TextModel's drop test.
    pub global_filters_after: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct TensorFilter {
    /// Covers the whole chain (round 8).
    pub complete: bool,
    /// The drop tests of the method and of every method in its `chain`.
    pub drop_when: Vec<String>,
}

/// `global_filters.TextModel`: the drop test, run on the name with a leading
/// `model.` removed for the test only.
pub fn text_model_drop_when() -> Option<String> {
    let filter = &roles::TEXT_MODEL_FILTER;
    if filter.strip_when != "name.startswith('model.')"
        || filter.strip_expr != "name[len('model.'):]"
        || filter.complete != Some(true)
    {
        return None;
    }
    Some(filter.drop_when.to_string())
}

/// `modify_tensors_base.tensor_loop.drop_suffixes`: names that
/// `ModelBase.prepare_tensors` skips before `modify_tensors`.
pub fn base_drop_suffixes() -> Vec<String> {
    roles::BASE_DROP_SUFFIXES.iter().map(|item| item.to_string()).collect()
}

/// Every architecture id in the roles table, sorted.
#[cfg(test)]
pub(crate) fn architecture_ids() -> impl Iterator<Item = &'static str> {
    catalog().arches.iter().map(|arch| arch.id)
}

pub fn converter_records(arch: &str) -> Vec<ConverterRecord> {
    let Some(table) = catalog().arches.iter().find(|item| item.id == arch) else {
        return Vec::new();
    };
    let owned = |items: &[&str]| items.iter().map(|item| item.to_string()).collect::<Vec<String>>();
    table
        .converters
        .iter()
        .map(|item| {
            let filter = item.filter.as_ref().map(|filter| {
                let mut drop_when = owned(filter.drop_when.unwrap_or_default());
                for link in filter.chain.unwrap_or_default() {
                    drop_when.extend(owned(link.unwrap_or_default()));
                }
                TensorFilter { complete: filter.complete, drop_when }
            });
            let hooks_list = item.tensor_hooks.unwrap_or_default();
            let mut hooks: Vec<String> = hooks_list.iter().map(|hook| hook.method.to_string()).collect();
            hooks.dedup();
            let generator = hooks_list
                .iter()
                .find(|hook| hook.method == "generate_extra_tensors")
                .map(|hook| (hook.class.to_string(), hook.line));
            ConverterRecord {
                class: item.class.to_string(),
                hf_architectures: owned(item.hf_architectures),
                undo_permute: item.undo_permute.unwrap_or(false),
                complete: item.modify.complete,
                unexpanded: owned(item.modify.unexpanded.unwrap_or_default()),
                branches: item
                    .modify
                    .branches
                    .unwrap_or_default()
                    .iter()
                    .map(|branch| TensorBranch {
                        when: branch.when.to_string(),
                        assigns: branch.assigns.to_string(),
                        expr: branch.expr.to_string(),
                    })
                    .collect(),
                filter,
                hooks,
                generator,
                global_filters_after: owned(item.global_filters_after),
            }
        })
        .collect()
}

/// `architectures[0]` to the GGUF architecture id that lists that class:
/// the first architecture, in the spec's order, whose converters list it.
pub fn architecture_for_hf_class(class: &str) -> Option<String> {
    let index = roles::HF_CLASS_ARCH;
    index
        .binary_search_by(|(item, _)| (*item).cmp(class))
        .ok()
        .map(|at| index[at].1.to_string())
}

fn detect(catalog: &Catalog, names: &[String]) -> Vec<String> {
    if names.is_empty() {
        return Vec::new();
    }
    let mut hits: Vec<&ArchTable> = catalog
        .arches
        .iter()
        .filter(|arch| names.iter().all(|name| explains(arch, name)))
        .collect();
    if hits.is_empty() {
        return Vec::new();
    }
    let smallest = hits.iter().map(|arch| arch.pairs.len()).min().unwrap();
    hits.retain(|arch| arch.pairs.len() == smallest);
    let mut ids: Vec<String> = hits.iter().map(|arch| arch.id.to_string()).collect();
    if let Some(group) = catalog.groups.iter().find(|group| {
        group.len() == ids.len() && group.iter().all(|id| ids.iter().any(|hit| hit == id))
    }) {
        ids = group.iter().map(|id| id.to_string()).collect();
    }
    ids.sort();
    ids
}

fn explains(arch: &ArchTable, name: &str) -> bool {
    let prepared = prepare_name(name, arch.id, Some(0));
    lookup(arch.pairs, &prepared).is_some() || matches_gguf(arch.pairs, name)
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
            let prepared = prepare_name(name, arch.id, lang_layers);
            match lookup(arch.pairs, &prepared) {
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

fn lookup(pairs: Pairs, name: &str) -> Option<String> {
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

fn best_match(pairs: Pairs, name: &str) -> Option<String> {
    let mut best: Option<(usize, String)> = None;
    for (pattern, gguf) in pairs.iter() {
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

fn matches_gguf(pairs: Pairs, name: &str) -> bool {
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

    use serde_json::{json, Value};
    use std::collections::HashMap;

    fn strs(items: &[&str]) -> Value {
        json!(items)
    }

    /// The converter records, shared groups, TextModel filter, and base drop
    /// suffixes rebuilt from the compiled statics are the spec's, value for
    /// value and in order.
    #[test]
    fn the_generated_roles_are_the_spec_table() {
        let Some(expected) = crate::spec_expected::load("gguf-tensor-roles.json") else { return };
        let spec_arches = expected["architectures"].as_object().unwrap();
        let mut ids: Vec<&str> = spec_arches.keys().map(String::as_str).collect();
        ids.sort();
        assert_eq!(ids, roles::ARCHES.iter().map(|arch| arch.id).collect::<Vec<_>>());
        for arch in roles::ARCHES {
            let rebuilt: Vec<Value> = arch
                .converters
                .iter()
                .map(|c| {
                    let mut modify = serde_json::Map::new();
                    modify.insert("complete".into(), json!(c.modify.complete));
                    if let Some(items) = c.modify.unexpanded {
                        modify.insert("unexpanded".into(), strs(items));
                    }
                    if let Some(items) = c.modify.branches {
                        let branches: Vec<Value> =
                            items.iter().map(|b| json!({"when": b.when, "assigns": b.assigns, "expr": b.expr})).collect();
                        modify.insert("branches".into(), Value::Array(branches));
                    }
                    let filter = c.filter.as_ref().map(|f| json!({"complete": f.complete, "drop_when": f.drop_when, "chain": f.chain}));
                    let hooks = c.tensor_hooks.map(|hooks| {
                        hooks.iter().map(|h| json!({"method": h.method, "class": h.class, "line": h.line})).collect::<Vec<_>>()
                    });
                    json!({
                        "class": c.class,
                        "hf_architectures": strs(c.hf_architectures),
                        "undo_permute": c.undo_permute,
                        "modify_tensors": Value::Object(modify),
                        "filter_tensors": filter,
                        "global_filters_after": strs(c.global_filters_after),
                        "tensor_hooks": hooks,
                    })
                })
                .collect();
            crate::spec_expected::assert_same(
                &Value::Array(rebuilt),
                &spec_arches[arch.id]["converters"],
                &format!("architectures.{}.converters", arch.id),
            );
        }
        let groups: Vec<Value> = roles::SHARED_GROUPS.iter().map(|group| strs(group)).collect();
        crate::spec_expected::assert_same(&Value::Array(groups), &expected["shared_resolutions"], "shared_resolutions");
        let f = &roles::TEXT_MODEL_FILTER;
        let filter = json!({"strip_when": f.strip_when, "strip_expr": f.strip_expr, "drop_when": f.drop_when, "complete": f.complete});
        crate::spec_expected::assert_same(&filter, &expected["text_model_filter"], "text_model_filter");
        crate::spec_expected::assert_same(&strs(roles::BASE_DROP_SUFFIXES), &expected["base_drop_suffixes"], "base_drop_suffixes");
    }

    /// The old run-time loader, on the spec JSON: per architecture, the
    /// pattern pairs and role patterns. The generated catalog must hold the
    /// same entries.
    #[test]
    fn the_generated_catalog_matches_the_old_loader() {
        let Some(expected) = crate::spec_expected::load("gguf-tensor-roles.json") else { return };
        type Row = (String, String, Vec<String>);
        let rows = |value: &Value| -> Vec<Row> {
            value.as_array().unwrap().iter().map(|row| (
                row["tensor"].as_str().unwrap().to_string(),
                row["gguf"].as_str().unwrap().to_string(),
                row["hf"].as_array().unwrap().iter().map(|h| h.as_str().unwrap().to_string()).collect(),
            )).collect()
        };
        let mappings = rows(&expected["mappings"]);
        let block_mappings = rows(&expected["block_mappings"]);
        let mut checked = 0;
        for (id, entry) in expected["architectures"].as_object().unwrap() {
            let roles: std::collections::HashSet<&str> = entry["roles"].as_array().unwrap().iter().map(|r| r.as_str().unwrap()).collect();
            let mut block = block_mappings.clone();
            if let Some(over) = expected["overrides"].get(id) {
                for row in rows(over) {
                    match block.iter().position(|item| item.0 == row.0) {
                        Some(pos) => block[pos] = row,
                        None => block.push(row),
                    }
                }
            }
            let mut pairs: HashMap<String, String> = HashMap::new();
            let mut role_gguf: HashMap<String, String> = HashMap::new();
            for (tensor, gguf, hf) in mappings.iter().chain(&block) {
                if !roles.contains(tensor.as_str()) {
                    continue;
                }
                role_gguf.insert(tensor.clone(), gguf.clone());
                for name in hf {
                    pairs.insert(name.clone(), gguf.clone());
                }
            }
            let arch = roles::ARCHES.iter().find(|arch| arch.id == id).unwrap();
            let generated: HashMap<String, String> = arch.pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            assert_eq!(generated.len(), arch.pairs.len(), "{id}: a pattern is listed twice");
            assert_eq!(generated, pairs, "{id} pairs");
            let generated: HashMap<String, String> = arch.roles.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            assert_eq!(generated.len(), arch.roles.len(), "{id}: a role is listed twice");
            assert_eq!(generated, role_gguf, "{id} roles");
            checked += 1;
        }
        assert_eq!(checked, roles::ARCHES.len());
    }

    /// The class index gives what the old scan gave for every class: the
    /// first architecture in the spec's order whose converters list it.
    #[test]
    fn the_class_index_matches_the_old_scan() {
        let Some(expected) = crate::spec_expected::load("gguf-tensor-roles.json") else { return };
        let mut classes = std::collections::BTreeSet::new();
        let old_scan = |class: &str| -> Option<String> {
            for (arch, entry) in expected["architectures"].as_object().unwrap() {
                for converter in entry["converters"].as_array().unwrap() {
                    if converter["hf_architectures"].as_array().unwrap().iter().any(|id| id.as_str() == Some(class)) {
                        return Some(arch.clone());
                    }
                }
            }
            None
        };
        for entry in expected["architectures"].as_object().unwrap().values() {
            for converter in entry["converters"].as_array().unwrap() {
                for id in converter["hf_architectures"].as_array().unwrap() {
                    classes.insert(id.as_str().unwrap().to_string());
                }
            }
        }
        for class in &classes {
            assert_eq!(architecture_for_hf_class(class), old_scan(class), "{class}");
        }
        assert_eq!(classes.len(), roles::HF_CLASS_ARCH.len());
        assert_eq!(architecture_for_hf_class("NoSuchForCausalLM"), None);
    }

    fn mapped(arch: &str, name: &str) -> String {
        let catalog = catalog();
        let resolved = catalog
            .arches
            .iter()
            .find(|item| item.id == arch)
            .unwrap_or_else(|| panic!("missing {arch}"));
        let prepared = prepare_name(name, arch, Some(24));
        lookup(resolved.pairs, &prepared).unwrap_or_else(|| panic!("{arch} missed {name}"))
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
            .find(|group| group.iter().any(|id| *id == "refact"))
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
