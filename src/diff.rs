//! Compare two tensor containers. Names, canonical dtypes, canonical shapes,
//! and logical tensor bytes. GGUF alignment padding is not part of the value.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::archive::{self, Archive, Inner, Member};
use crate::cli::DEFAULT_MAX_HEADER;
use crate::dtype::{self, format_shape};
use crate::error::{msg, Result};
use crate::payload::Payload;

const CHUNK: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
enum Finding {
    Format {
        left: String,
        right: String,
    },
    OnlyLeft {
        path: String,
        name: String,
    },
    OnlyRight {
        path: String,
        name: String,
    },
    Tensor {
        left_name: String,
        right_name: String,
        left_dtype: String,
        right_dtype: String,
        dtype_differs: bool,
        left_shape: String,
        right_shape: String,
        left_order: String,
        right_order: String,
        values_at: Option<u64>,
        left_len: u64,
        right_len: u64,
    },
    MetaOnly {
        path: String,
        key: String,
    },
    MetaValue {
        key: String,
        left: String,
        right: String,
    },
    Names {
        table: String,
    },
}

impl Finding {
    fn is_difference(&self) -> bool {
        match self {
            Finding::Names { .. } => false,
            Finding::Tensor {
                dtype_differs,
                left_shape,
                right_shape,
                left_order,
                right_order,
                values_at,
                ..
            } => {
                *dtype_differs
                    || left_shape != right_shape
                    || left_order != right_order
                    || values_at.is_some()
            }
            _ => true,
        }
    }
}

struct Side {
    label: String,
    dtype: String,
    shape: Vec<u64>,
    fortran: bool,
    payload: Payload,
}

pub fn compare(
    left: &Path,
    right: &Path,
    json_out: bool,
    max_header: u64,
    match_names: bool,
    fold: bool,
) -> Result<i32> {
    let left_archive = archive::open(left, None, max_header, None, None)?;
    let right_archive = archive::open(right, None, max_header, None, None)?;
    let findings = findings_of(&left_archive, &right_archive, match_names)?;
    let same = !findings.iter().any(Finding::is_difference);
    if json_out {
        println!("{}", render_json(&findings, same));
    } else if !findings.is_empty() {
        println!("{}", render_text(&findings, fold));
    }
    Ok(if same { 0 } else { 1 })
}

fn findings_of(left: &Archive, right: &Archive, match_names: bool) -> Result<Vec<Finding>> {
    let mut findings = Vec::new();
    let left_desc = describe(left);
    let right_desc = describe(right);
    if left_desc != right_desc {
        findings.push(Finding::Format {
            left: format!("{} {left_desc}", left.path.display()),
            right: format!("{} {right_desc}", right.path.display()),
        });
    }
    let left_members = index_members(left)?;
    let right_members = index_members(right)?;
    let (links, note) = link_members(left, right, &left_members, &right_members, match_names);
    if let Some(table) = note {
        findings.push(Finding::Names { table });
    }
    let linked_left: BTreeSet<_> = links.iter().map(|link| link.left.clone()).collect();
    let linked_right: BTreeSet<_> = links.iter().map(|link| link.right.clone()).collect();
    let mut only_left: Vec<_> = left_members
        .keys()
        .filter(|name| !linked_left.contains(*name))
        .cloned()
        .collect();
    let mut only_right: Vec<_> = right_members
        .keys()
        .filter(|name| !linked_right.contains(*name))
        .cloned()
        .collect();
    only_left.sort_by(|left_name, right_name| cmp_name(left_name, right_name));
    only_right.sort_by(|left_name, right_name| cmp_name(left_name, right_name));
    for name in only_left {
        findings.push(Finding::OnlyLeft {
            path: left.path.display().to_string(),
            name,
        });
    }
    for name in only_right {
        findings.push(Finding::OnlyRight {
            path: right.path.display().to_string(),
            name,
        });
    }
    for link in links {
        let left_side = prepare(left, left_members.get(&link.left).unwrap())?;
        let right_side = prepare(right, right_members.get(&link.right).unwrap())?;
        if let Some(finding) = tensor_finding(&link.left, &link.right, &left_side, &right_side)? {
            findings.push(finding);
        }
    }
    push_metadata(&mut findings, left, right);
    Ok(findings)
}

struct Link {
    left: String,
    right: String,
}

/// Exact names first. With `--match-names`, leftover names are paired through
/// the architecture in `general.architecture`, or the smallest role set that
/// explains every remaining name.
fn link_members(
    left_archive: &Archive,
    right_archive: &Archive,
    left: &BTreeMap<String, &Member>,
    right: &BTreeMap<String, &Member>,
    match_names: bool,
) -> (Vec<Link>, Option<String>) {
    let mut links = Vec::new();
    let mut left_free = Vec::new();
    let mut right_free = Vec::new();
    for name in left.keys() {
        if right.contains_key(name) {
            links.push(Link {
                left: name.clone(),
                right: name.clone(),
            });
        } else {
            left_free.push(name.clone());
        }
    }
    for name in right.keys() {
        if !left.contains_key(name) {
            right_free.push(name.clone());
        }
    }
    let note = if match_names {
        let hit =
            crate::gguf_names::match_free(left_archive, right_archive, &left_free, &right_free);
        for (left_name, right_name) in hit.pairs {
            links.push(Link {
                left: left_name,
                right: right_name,
            });
        }
        if left_free.is_empty() && right_free.is_empty() {
            None
        } else {
            Some(crate::gguf_names::note(&hit.arches))
        }
    } else {
        None
    };
    links.sort_by(|left_link, right_link| {
        cmp_name(&left_link.left, &right_link.left)
            .then_with(|| cmp_name(&left_link.right, &right_link.right))
    });
    (links, note)
}

/// Text order, except a digit run compares as an integer. `blk.10` follows `blk.9`.
fn cmp_name(left: &str, right: &str) -> Ordering {
    let mut left_rest = left;
    let mut right_rest = right;
    loop {
        if left_rest.is_empty() || right_rest.is_empty() {
            return left_rest
                .len()
                .cmp(&right_rest.len())
                .then_with(|| left.cmp(right));
        }
        let left_digit = left_rest.as_bytes()[0].is_ascii_digit();
        let right_digit = right_rest.as_bytes()[0].is_ascii_digit();
        if left_digit && right_digit {
            let (left_num, left_next) = take_digits(left_rest);
            let (right_num, right_next) = take_digits(right_rest);
            match cmp_digits(left_num, right_num) {
                Ordering::Equal => {
                    left_rest = left_next;
                    right_rest = right_next;
                }
                order => return order,
            }
        } else {
            let left_char = left_rest.chars().next().unwrap();
            let right_char = right_rest.chars().next().unwrap();
            match left_char.cmp(&right_char) {
                Ordering::Equal => {
                    left_rest = &left_rest[left_char.len_utf8()..];
                    right_rest = &right_rest[right_char.len_utf8()..];
                }
                order => return order,
            }
        }
    }
}

fn take_digits(text: &str) -> (&str, &str) {
    let end = text
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(text.len());
    text.split_at(end)
}

fn cmp_digits(left: &str, right: &str) -> Ordering {
    let left = trim_zeros(left);
    let right = trim_zeros(right);
    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
}

fn trim_zeros(text: &str) -> &str {
    let trimmed = text.trim_start_matches('0');
    if trimmed.is_empty() {
        "0"
    } else {
        trimmed
    }
}

fn describe(archive: &Archive) -> String {
    format!("{} {}", archive.format.as_str(), archive.version)
}

fn index_members(archive: &Archive) -> Result<BTreeMap<String, &Member>> {
    let mut map = BTreeMap::new();
    for member in &archive.members {
        if map.insert(member.name.clone(), member).is_some() {
            return Err(msg(format!(
                "{} has two members named {}",
                archive.path.display(),
                member.name
            )));
        }
    }
    Ok(map)
}

fn prepare(archive: &Archive, member: &Member) -> Result<Side> {
    let mut shape = member.shape.clone();
    #[cfg(format_gguf)]
    if matches!(archive.inner, Inner::Gguf(_)) {
        shape.reverse();
    }
    let len = logical_nbytes(archive, member)?;
    Ok(Side {
        label: member.dtype.clone(),
        dtype: type_key(&member.dtype),
        shape,
        fortran: member.fortran,
        payload: member.payload.clipped(len)?,
    })
}

fn type_key(label: &str) -> String {
    dtype::canonical_id(label)
        .map(str::to_string)
        .unwrap_or_else(|_| label.to_string())
}

fn logical_nbytes(archive: &Archive, member: &Member) -> Result<u64> {
    #[cfg(format_gguf)]
    if let Inner::Gguf(file) = &archive.inner {
        let tensor = file
            .tensors
            .iter()
            .find(|tensor| tensor.name == member.name)
            .ok_or_else(|| msg(format!("missing GGUF tensor {}", member.name)))?;
        let logical = dtype::ggml_payload_bytes(tensor.ggml_type, &tensor.shape)?;
        if logical > member.nbytes {
            return Err(msg(format!(
                "tensor {} is larger than its stored span",
                member.name
            )));
        }
        return Ok(logical);
    }
    Ok(member.nbytes)
}

fn tensor_finding(
    left_name: &str,
    right_name: &str,
    left: &Side,
    right: &Side,
) -> Result<Option<Finding>> {
    let left_order = order_word(left.fortran);
    let right_order = order_word(right.fortran);
    let dtype_differs = left.dtype != right.dtype;
    let left_shape = format_shape(&left.shape);
    let right_shape = format_shape(&right.shape);
    let values = if !dtype_differs && left.shape == right.shape {
        first_mismatch(&left.payload, &right.payload)?
    } else {
        None
    };
    let names_differ = left_name != right_name;
    let content_differs =
        dtype_differs || left_shape != right_shape || left_order != right_order || values.is_some();
    if !names_differ && !content_differs {
        return Ok(None);
    }
    Ok(Some(Finding::Tensor {
        left_name: left_name.to_string(),
        right_name: right_name.to_string(),
        left_dtype: left.label.clone(),
        right_dtype: right.label.clone(),
        dtype_differs,
        left_shape,
        right_shape,
        left_order,
        right_order,
        values_at: values,
        left_len: left.payload.len(),
        right_len: right.payload.len(),
    }))
}

fn tensor_line(finding: &Finding) -> String {
    let Finding::Tensor {
        left_name,
        right_name,
        ..
    } = finding
    else {
        return String::new();
    };
    tensor_line_named(finding, left_name, right_name)
}

fn side_by_side(left: &str, right: &str) -> String {
    if left == right {
        left.to_string()
    } else {
        format!("{left} => {right}")
    }
}

enum NamePart {
    Text(String),
    Index(u64),
}

struct FoldItem {
    index: usize,
    slots: Vec<u64>,
    left: Vec<NamePart>,
    right: Vec<NamePart>,
}

/// Merge tensor lines whose names differ only by a consecutive integer.
fn fold_lines(findings: &[Finding]) -> Vec<String> {
    let mut groups: BTreeMap<String, Vec<FoldItem>> = BTreeMap::new();
    for (index, finding) in findings.iter().enumerate() {
        let Finding::Tensor {
            left_name,
            right_name,
            left_dtype,
            right_dtype,
            dtype_differs,
            left_shape,
            right_shape,
            left_order,
            right_order,
            values_at,
            left_len,
            right_len,
        } = finding
        else {
            continue;
        };
        let left = name_parts(left_name);
        let right = name_parts(right_name);
        let mut slots = indices_of(&left);
        slots.extend(indices_of(&right));
        let key = format!(
            "{} => {} | {dtype_differs} {left_dtype} {right_dtype} {left_shape} {right_shape} {left_order} {right_order} {values_at:?} {left_len} {right_len}",
            name_pattern(&left),
            name_pattern(&right)
        );
        groups.entry(key).or_default().push(FoldItem {
            index,
            slots,
            left,
            right,
        });
    }
    let mut skip = vec![false; findings.len()];
    let mut folded: Vec<Option<String>> = vec![None; findings.len()];
    for mut items in groups.into_values() {
        items.sort_by(|left, right| {
            left.slots
                .cmp(&right.slots)
                .then(left.index.cmp(&right.index))
        });
        let mut start = 0;
        while start < items.len() {
            let mut end = start + 1;
            while end < items.len()
                && continues(&items[start].slots, &items[end].slots, (end - start) as u64)
            {
                end += 1;
            }
            if end - start >= 2 {
                let run = &items[start..end];
                let anchor = run.iter().map(|item| item.index).min().unwrap();
                let mut vary = vec![false; run[0].slots.len()];
                for item in run.iter().skip(1) {
                    for (slot, (&first, &next)) in run[0].slots.iter().zip(&item.slots).enumerate()
                    {
                        if next != first {
                            vary[slot] = true;
                        }
                    }
                }
                let steps = (run.len() - 1) as u64;
                let left_name = fold_name(&run[0].left, &vary, steps);
                let left_slots = indices_of(&run[0].left).len();
                let right_name = fold_name(&run[0].right, &vary[left_slots..], steps);
                folded[anchor] = Some(tensor_line_named(
                    &findings[anchor],
                    &left_name,
                    &right_name,
                ));
                for item in run {
                    if item.index != anchor {
                        skip[item.index] = true;
                    }
                }
            }
            start = end;
        }
    }
    findings
        .iter()
        .enumerate()
        .filter(|(index, _)| !skip[*index])
        .map(|(index, finding)| {
            folded[index]
                .clone()
                .unwrap_or_else(|| render_finding(finding))
        })
        .collect()
}

fn continues(first: &[u64], next: &[u64], step: u64) -> bool {
    if step == 0 || first.len() != next.len() {
        return false;
    }
    let mut moved = false;
    for (&start, &value) in first.iter().zip(next) {
        if value == start {
            continue;
        }
        if value == start.saturating_add(step) {
            moved = true;
            continue;
        }
        return false;
    }
    moved
}

fn fold_name(parts: &[NamePart], vary: &[bool], steps: u64) -> String {
    let mut slot = 0;
    let mut out = Vec::new();
    for part in parts {
        match part {
            NamePart::Text(text) => out.push(text.clone()),
            NamePart::Index(index) => {
                if vary.get(slot).copied().unwrap_or(false) {
                    out.push(format!("[{}-{}]", index, index + steps));
                } else {
                    out.push(index.to_string());
                }
                slot += 1;
            }
        }
    }
    out.join(".")
}

fn name_parts(name: &str) -> Vec<NamePart> {
    name.split('.')
        .map(|segment| match parse_index(segment) {
            Some(index) => NamePart::Index(index),
            None => NamePart::Text(segment.to_string()),
        })
        .collect()
}

fn name_pattern(parts: &[NamePart]) -> String {
    parts
        .iter()
        .map(|part| match part {
            NamePart::Text(text) => text.as_str(),
            NamePart::Index(_) => "{}",
        })
        .collect::<Vec<_>>()
        .join(".")
}

fn indices_of(parts: &[NamePart]) -> Vec<u64> {
    parts
        .iter()
        .filter_map(|part| match part {
            NamePart::Index(index) => Some(*index),
            NamePart::Text(_) => None,
        })
        .collect()
}

fn parse_index(segment: &str) -> Option<u64> {
    if segment.is_empty() || !segment.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    if segment.len() > 1 && segment.starts_with('0') {
        return None;
    }
    segment.parse().ok()
}

fn tensor_line_named(finding: &Finding, left_name: &str, right_name: &str) -> String {
    let Finding::Tensor {
        left_dtype,
        right_dtype,
        dtype_differs,
        left_shape,
        right_shape,
        left_order,
        right_order,
        values_at,
        left_len,
        right_len,
        ..
    } = finding
    else {
        return String::new();
    };
    let names = side_by_side(left_name, right_name);
    let dtype = if *dtype_differs {
        format!("{left_dtype} => {right_dtype}")
    } else {
        left_dtype.clone()
    };
    let shape = side_by_side(left_shape, right_shape);
    let mut line = format!("{names} : {dtype} : {shape}");
    if left_order != right_order {
        line.push_str(&format!(" : {left_order} => {right_order}"));
    }
    if let Some(at) = values_at {
        if left_len == right_len {
            line.push_str(&format!(" : values differ at byte {at}"));
        } else {
            line.push_str(&format!(
                " : values differ at byte {at} ({left_len} vs {right_len} bytes)"
            ));
        }
    }
    line
}

fn order_word(fortran: bool) -> String {
    if fortran {
        "fortran".to_string()
    } else {
        "c-order".to_string()
    }
}

fn first_mismatch(left: &Payload, right: &Payload) -> Result<Option<u64>> {
    let shared = left.len().min(right.len());
    let mut at = 0u64;
    let mut left_buf = vec![0u8; CHUNK];
    let mut right_buf = vec![0u8; CHUNK];
    while at < shared {
        let take = ((shared - at) as usize).min(CHUNK);
        left.read_at(at, &mut left_buf[..take])?;
        right.read_at(at, &mut right_buf[..take])?;
        if let Some(index) = left_buf[..take]
            .iter()
            .zip(&right_buf[..take])
            .position(|(lhs, rhs)| lhs != rhs)
        {
            return Ok(Some(at + index as u64));
        }
        at += take as u64;
    }
    if left.len() != right.len() {
        return Ok(Some(shared));
    }
    Ok(None)
}

fn push_metadata(findings: &mut Vec<Finding>, left: &Archive, right: &Archive) {
    let left_meta = meta_map(&left.metadata);
    let right_meta = meta_map(&right.metadata);
    for key in left_meta.keys() {
        if !right_meta.contains_key(key) {
            findings.push(Finding::MetaOnly {
                path: left.path.display().to_string(),
                key: key.clone(),
            });
        }
    }
    for key in right_meta.keys() {
        if !left_meta.contains_key(key) {
            findings.push(Finding::MetaOnly {
                path: right.path.display().to_string(),
                key: key.clone(),
            });
        }
    }
    for key in left_meta.keys() {
        if let Some(right_value) = right_meta.get(key) {
            let left_value = &left_meta[key];
            if left_value != right_value {
                findings.push(Finding::MetaValue {
                    key: key.clone(),
                    left: left_value.clone(),
                    right: right_value.clone(),
                });
            }
        }
    }
}

fn meta_map(pairs: &[(String, String)]) -> BTreeMap<String, String> {
    pairs.iter().cloned().collect()
}

fn render_text(findings: &[Finding], fold: bool) -> String {
    let lines = if fold {
        fold_lines(findings)
    } else {
        findings.iter().map(render_finding).collect()
    };
    lines.join("\n")
}

fn render_finding(finding: &Finding) -> String {
    match finding {
        Finding::Format { left, right } => format!("format: {left}  vs  {right}"),
        Finding::OnlyLeft { path, name } => format!("only in {path}: {name}"),
        Finding::OnlyRight { path, name } => format!("only in {path}: {name}"),
        Finding::Tensor { .. } => tensor_line(finding),
        Finding::MetaOnly { path, key } => format!("metadata: {key}  only in {path}"),
        Finding::MetaValue { key, left, right } => {
            format!("metadata: {key}  {left}  vs  {right}")
        }
        Finding::Names { table } => format!("names: {table}"),
    }
}

fn render_json(findings: &[Finding], same: bool) -> String {
    let items: Vec<Value> = findings
        .iter()
        .map(|finding| match finding {
            Finding::Format { left, right } => {
                json!({"kind": "format", "left": left, "right": right})
            }
            Finding::OnlyLeft { path, name } => {
                json!({"kind": "only_left", "path": path, "name": name})
            }
            Finding::OnlyRight { path, name } => {
                json!({"kind": "only_right", "path": path, "name": name})
            }
            Finding::Tensor {
                left_name,
                right_name,
                left_dtype,
                right_dtype,
                left_shape,
                right_shape,
                left_order,
                right_order,
                values_at,
                left_len,
                right_len,
                ..
            } => json!({
                "kind": "tensor",
                "left": left_name,
                "right": right_name,
                "left_dtype": left_dtype,
                "right_dtype": right_dtype,
                "left_shape": left_shape,
                "right_shape": right_shape,
                "left_order": left_order,
                "right_order": right_order,
                "values_at": values_at,
                "left_bytes": left_len,
                "right_bytes": right_len
            }),
            Finding::MetaOnly { path, key } => {
                json!({"kind": "metadata_only", "path": path, "key": key})
            }
            Finding::MetaValue { key, left, right } => {
                json!({"kind": "metadata", "key": key, "left": left, "right": right})
            }
            Finding::Names { table } => json!({"kind": "names", "table": table}),
        })
        .collect();
    serde_json::to_string_pretty(&json!({"same": same, "findings": items}))
        .unwrap_or_else(|_| "{}".to_string())
}

pub fn diff_main(args: &[String]) -> Result<i32> {
    match parse_diff(args)? {
        DiffCommand::Help => {
            println!("{}", diff_help());
            Ok(0)
        }
        DiffCommand::Version => {
            println!("{}", crate::version_line("tensdiff"));
            Ok(0)
        }
        DiffCommand::Run {
            json,
            match_names,
            fold,
            left,
            left_force,
            right,
            right_force,
        } => {
            let left = archive::resolve_user_path(&left, left_force)?;
            let right = archive::resolve_user_path(&right, right_force)?;
            compare(
                &left.path,
                &right.path,
                json,
                DEFAULT_MAX_HEADER,
                match_names,
                fold,
            )
        }
    }
}

enum DiffCommand {
    Help,
    Version,
    Run {
        json: bool,
        match_names: bool,
        fold: bool,
        left: PathBuf,
        left_force: bool,
        right: PathBuf,
        right_force: bool,
    },
}

fn parse_diff(args: &[String]) -> Result<DiffCommand> {
    let mut json = false;
    let mut match_names = false;
    let mut fold = false;
    let mut paths: Vec<(PathBuf, bool)> = Vec::new();
    let mut force_next = false;
    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => return Ok(DiffCommand::Help),
            "--version" => return Ok(DiffCommand::Version),
            "--json" => {
                if force_next {
                    return Err(msg(
                        "--force-single-file must come immediately before a path",
                    ));
                }
                json = true;
            }
            "--match-names" => {
                if force_next {
                    return Err(msg(
                        "--force-single-file must come immediately before a path",
                    ));
                }
                match_names = true;
            }
            "--fold" => {
                if force_next {
                    return Err(msg(
                        "--force-single-file must come immediately before a path",
                    ));
                }
                fold = true;
            }
            "--force-single-file" => {
                if force_next {
                    return Err(msg(
                        "--force-single-file must come immediately before a path",
                    ));
                }
                force_next = true;
            }
            other if other.starts_with('-') => {
                return Err(msg(format!("unknown option {other}")));
            }
            other => {
                paths.push((PathBuf::from(other), force_next));
                force_next = false;
            }
        }
    }
    if force_next {
        return Err(msg(
            "--force-single-file must come immediately before a path",
        ));
    }
    if json && fold {
        return Err(msg("--fold does not apply to --json"));
    }
    if paths.len() != 2 {
        return Err(msg(
            "usage: tensdiff [--json] [--match-names] [--fold] [--force-single-file] LEFT RIGHT",
        ));
    }
    let (right, right_force) = paths.pop().unwrap();
    let (left, left_force) = paths.pop().unwrap();
    Ok(DiffCommand::Run {
        json,
        match_names,
        fold,
        left,
        left_force,
        right,
        right_force,
    })
}

pub fn diff_help() -> &'static str {
    "\
tensdiff — compare two tensor containers

Usage:
  tensdiff [--json] [--match-names] [--fold] [--force-single-file] LEFT RIGHT

Exit status:
  0  same names, types, shapes, values, and metadata
  1  a difference was found
  2  a file could not be read

The container format is reported and does not stop the member comparison.
A tensor is one line: name, dtype, and shape. => marks a field that differs.
GGUF shapes are reversed to canonical order. Alignment padding is not compared.
Quantized blocks are compared as stored bytes.
--match-names pairs leftover names through gguf-tensor-roles.json.
general.architecture selects the table. Otherwise the smallest role set that
explains every remaining name is used, and a tie is reported as
\"may be A, B, or C\". Vision tensors are not pairs in a language file.
--fold merges tensor lines that match except for one consecutive number,
so blk.20 and blk.21 become blk.[20-21]. A gap stays on its own line.
--force-single-file applies only to the path written immediately after it.
A .safetensors file beside config.json is refused unless that flag is set.
Pass the directory to read the whole safetensors weight set.
"
}

#[cfg(all(test, format_safetensors))]
mod tests {
    use super::*;
    use crate::formats::safetensors::{self, OutTensor};
    use crate::payload::Payload;

    fn f32_tensor(name: &str, values: &[f32]) -> OutTensor {
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        OutTensor {
            name: name.to_string(),
            dtype: "F32".to_string(),
            shape: vec![values.len() as u64],
            bytes: Payload::Bytes(bytes),
        }
    }

    fn write_pair(dir: &Path, left_values: &[f32], right_values: &[f32]) -> (PathBuf, PathBuf) {
        let left = dir.join("left.safetensors");
        let right = dir.join("right.safetensors");
        safetensors::write_file(
            &left,
            &[("source".into(), "left".into())],
            &[f32_tensor("weight", left_values)],
        )
        .unwrap();
        safetensors::write_file(
            &right,
            &[("source".into(), "right".into())],
            &[f32_tensor("weight", right_values)],
        )
        .unwrap();
        (left, right)
    }

    #[test]
    fn identical_files_have_no_findings() {
        let dir = tempfile::tempdir().unwrap();
        let left = dir.path().join("left.safetensors");
        let right = dir.path().join("right.safetensors");
        let tensor = f32_tensor("weight", &[1.0, 2.0]);
        safetensors::write_file(
            &left,
            &[("source".into(), "same".into())],
            &[tensor.clone()],
        )
        .unwrap();
        safetensors::write_file(&right, &[("source".into(), "same".into())], &[tensor]).unwrap();
        let code = compare(&left, &right, false, DEFAULT_MAX_HEADER, false, false).unwrap();
        assert_eq!(code, 0);
    }

    #[test]
    fn a_changed_byte_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let (left, right) = write_pair(dir.path(), &[1.0, 2.0], &[1.0, 3.0]);
        let left_archive = archive::open(&left, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let right_archive = archive::open(&right, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let text = render_text(
            &findings_of(&left_archive, &right_archive, false).unwrap(),
            false,
        );
        assert!(
            text.contains("weight : F32 : [2] : values differ at byte 6"),
            "{text}"
        );
        assert!(text.contains("metadata: source  left  vs  right"), "{text}");
    }

    #[test]
    fn a_shape_change_keeps_the_shared_name_and_dtype() {
        let dir = tempfile::tempdir().unwrap();
        let left = dir.path().join("left.safetensors");
        let right = dir.path().join("right.safetensors");
        safetensors::write_file(&left, &[], &[f32_tensor("weight", &[1.0, 2.0, 3.0, 4.0])])
            .unwrap();
        let mut wide = f32_tensor("weight", &[1.0, 2.0, 3.0, 4.0]);
        wide.shape = vec![1, 4];
        safetensors::write_file(&right, &[], &[wide]).unwrap();
        let left_archive = archive::open(&left, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let right_archive = archive::open(&right, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let text = render_text(
            &findings_of(&left_archive, &right_archive, false).unwrap(),
            false,
        );
        assert!(text.contains("weight : F32 : [4] => [1, 4]"), "{text}");
        assert!(!text.contains("values differ"), "{text}");
    }

    #[test]
    fn numeric_parts_of_a_name_sort_as_numbers() {
        let dir = tempfile::tempdir().unwrap();
        let left = dir.path().join("left.safetensors");
        let right = dir.path().join("right.safetensors");
        let names = [
            "model.layers.10.mlp.weight",
            "model.layers.2.mlp.weight",
            "model.layers.9.mlp.weight",
        ];
        let mut left_tensors: Vec<_> = names.iter().map(|name| f32_tensor(name, &[1.0])).collect();
        left_tensors.push(f32_tensor("extra.10.weight", &[1.0]));
        left_tensors.push(f32_tensor("extra.9.weight", &[1.0]));
        let right_tensors: Vec<_> = names.iter().map(|name| f32_tensor(name, &[2.0])).collect();
        safetensors::write_file(&left, &[], &left_tensors).unwrap();
        safetensors::write_file(&right, &[], &right_tensors).unwrap();
        let left_archive = archive::open(&left, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let right_archive = archive::open(&right, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let text = render_text(
            &findings_of(&left_archive, &right_archive, false).unwrap(),
            false,
        );
        let at = |name: &str| {
            text.find(name)
                .unwrap_or_else(|| panic!("{name} missing in {text}"))
        };
        assert!(at("extra.9.weight") < at("extra.10.weight"), "{text}");
        assert!(
            at("extra.10.weight") < at("model.layers.2.mlp.weight"),
            "{text}"
        );
        assert!(
            at("model.layers.2.mlp.weight") < at("model.layers.9.mlp.weight"),
            "{text}"
        );
        assert!(
            at("model.layers.9.mlp.weight") < at("model.layers.10.mlp.weight"),
            "{text}"
        );
    }

    #[test]
    fn names_and_dtypes_are_reported() {
        let dir = tempfile::tempdir().unwrap();
        let left = dir.path().join("left.safetensors");
        let right = dir.path().join("right.safetensors");
        safetensors::write_file(
            &left,
            &[],
            &[f32_tensor("keep", &[1.0]), f32_tensor("only_left", &[1.0])],
        )
        .unwrap();
        let mut other = f32_tensor("keep", &[1.0]);
        other.dtype = "F16".into();
        other.shape = vec![2];
        other.bytes = Payload::Bytes(vec![0, 0, 0, 0]);
        safetensors::write_file(&right, &[], &[other, f32_tensor("only_right", &[1.0])]).unwrap();
        let left_archive = archive::open(&left, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let right_archive = archive::open(&right, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let text = render_text(
            &findings_of(&left_archive, &right_archive, false).unwrap(),
            false,
        );
        assert!(
            text.contains("only in") && text.contains("only_left"),
            "{text}"
        );
        assert!(text.contains("only_right"), "{text}");
        assert!(text.contains("keep : F32 => F16 : [1] => [2]"), "{text}");
        assert!(!text.contains("values differ"), "{text}");
    }

    #[cfg(format_gguf)]
    #[test]
    fn gguf_padding_is_not_a_value_difference() {
        let dir = tempfile::tempdir().unwrap();
        let safe = dir.path().join("weight.safetensors");
        let gguf = dir.path().join("weight.gguf");
        safetensors::write_file(&safe, &[], &[f32_tensor("weight", &[1.0, 2.0, 3.0, 4.0])])
            .unwrap();
        crate::formats::gguf::write_fixture(&gguf, "demo", "weight", &[1.0, 2.0, 3.0, 4.0])
            .unwrap();
        let left_archive = archive::open(&safe, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let right_archive = archive::open(&gguf, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let text = render_text(
            &findings_of(&left_archive, &right_archive, false).unwrap(),
            false,
        );
        assert!(text.contains("format:"), "{text}");
        assert!(!text.contains("values differ"), "{text}");
        assert!(!text.contains("=>"), "{text}");
    }

    #[test]
    fn match_names_pairs_the_llama_table_in_either_order() {
        let dir = tempfile::tempdir().unwrap();
        let hf_path = dir.path().join("hf.safetensors");
        let gguf_path = dir.path().join("gguf-names.safetensors");
        let hf = "model.layers.0.self_attn.v_proj.weight";
        let gguf = "blk.0.attn_v.weight";
        safetensors::write_file(
            &hf_path,
            &[("source".into(), "same".into())],
            &[f32_tensor(hf, &[1.0, 2.0])],
        )
        .unwrap();
        safetensors::write_file(
            &gguf_path,
            &[("source".into(), "same".into())],
            &[
                f32_tensor(gguf, &[1.0, 2.0]),
                f32_tensor("extra.weight", &[1.0]),
            ],
        )
        .unwrap();
        let hf_archive = archive::open(&hf_path, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let gguf_archive = archive::open(&gguf_path, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let plain = render_text(
            &findings_of(&hf_archive, &gguf_archive, false).unwrap(),
            false,
        );
        assert!(plain.contains(hf) && plain.contains("only in"), "{plain}");
        assert!(!plain.contains("names:"), "{plain}");
        let text = render_text(
            &findings_of(&hf_archive, &gguf_archive, true).unwrap(),
            false,
        );
        assert!(text.contains("names: no table matched"), "{text}");
        assert!(!text.contains("=>"), "{text}");
        assert!(text.contains("extra.weight"), "{text}");
        assert_eq!(
            compare(&hf_path, &gguf_path, false, DEFAULT_MAX_HEADER, true, false).unwrap(),
            1
        );
        let paired = dir.path().join("paired.safetensors");
        safetensors::write_file(
            &paired,
            &[("source".into(), "same".into())],
            &[f32_tensor(gguf, &[1.0, 2.0])],
        )
        .unwrap();
        let paired_archive = archive::open(&paired, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let paired_text = render_text(
            &findings_of(&hf_archive, &paired_archive, true).unwrap(),
            false,
        );
        assert!(
            paired_text.contains("names:") && !paired_text.contains("no table matched"),
            "{paired_text}"
        );
        assert!(
            paired_text.contains(&format!("{hf} => {gguf} : F32 : [2]")),
            "{paired_text}"
        );
        assert!(!paired_text.contains("values differ"), "{paired_text}");
        let flipped = render_text(
            &findings_of(&paired_archive, &hf_archive, true).unwrap(),
            false,
        );
        assert!(
            flipped.contains(&format!("{gguf} => {hf} : F32 : [2]")),
            "{flipped}"
        );
        assert_eq!(
            compare(&hf_path, &paired, false, DEFAULT_MAX_HEADER, true, false).unwrap(),
            0
        );
        let other = dir.path().join("other.safetensors");
        safetensors::write_file(
            &other,
            &[("source".into(), "same".into())],
            &[f32_tensor("not.in.table", &[1.0])],
        )
        .unwrap();
        let other_archive = archive::open(&other, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let none = render_text(
            &findings_of(&hf_archive, &other_archive, true).unwrap(),
            false,
        );
        assert!(none.contains("names: no table matched"), "{none}");
        assert!(none.contains(hf) && none.contains("not.in.table"), "{none}");
    }

    #[test]
    fn declared_architecture_pairs_a_renamed_qwen_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let hf_path = dir.path().join("qwen-hf.safetensors");
        let gguf_path = dir.path().join("qwen-gguf.safetensors");
        let hf = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        let gguf = "blk.0.attn_qkv.weight";
        safetensors::write_file(
            &hf_path,
            &[("source".into(), "same".into())],
            &[
                f32_tensor(hf, &[1.0, 2.0]),
                f32_tensor("custom.bias", &[3.0]),
            ],
        )
        .unwrap();
        safetensors::write_file(
            &gguf_path,
            &[
                ("source".into(), "same".into()),
                ("general.architecture".into(), "qwen35".into()),
            ],
            &[f32_tensor(gguf, &[1.0, 2.0])],
        )
        .unwrap();
        let hf_archive = archive::open(&hf_path, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let gguf_archive = archive::open(&gguf_path, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let text = render_text(
            &findings_of(&hf_archive, &gguf_archive, true).unwrap(),
            false,
        );
        assert!(text.contains("names: qwen35"), "{text}");
        assert!(
            text.contains(&format!("{hf} => {gguf} : F32 : [2]")),
            "{text}"
        );
        assert!(text.contains("custom.bias"), "{text}");
        assert!(!text.contains("values differ"), "{text}");
    }

    fn paired(left: &str, right: &str, shape: &str) -> Finding {
        Finding::Tensor {
            left_name: left.to_string(),
            right_name: right.to_string(),
            left_dtype: "F32".to_string(),
            right_dtype: "BF16".to_string(),
            dtype_differs: true,
            left_shape: shape.to_string(),
            right_shape: shape.to_string(),
            left_order: "c-order".to_string(),
            right_order: "c-order".to_string(),
            values_at: None,
            left_len: 0,
            right_len: 0,
        }
    }

    #[test]
    fn fold_merges_a_consecutive_number_and_keeps_a_gap() {
        let findings = vec![
            paired(
                "blk.20.attn_norm.weight",
                "model.language_model.layers.20.input_layernorm.weight",
                "[1024]",
            ),
            paired(
                "blk.21.attn_norm.weight",
                "model.language_model.layers.21.input_layernorm.weight",
                "[1024]",
            ),
            paired(
                "blk.22.attn_norm.weight",
                "model.language_model.layers.21.input_layernorm.weight",
                "[1024]",
            ),
            paired(
                "blk.24.attn_norm.weight",
                "model.language_model.layers.24.input_layernorm.weight",
                "[1024]",
            ),
            paired(
                "blk.30.ssm_conv1d.weight",
                "model.language_model.layers.30.linear_attn.conv1d.weight",
                "[6144, 4]",
            ),
            paired(
                "blk.40.attn_norm.weight",
                "model.language_model.layers.7.input_layernorm.weight",
                "[1024]",
            ),
            paired(
                "blk.41.attn_norm.weight",
                "model.language_model.layers.7.input_layernorm.weight",
                "[1024]",
            ),
        ];
        let text = render_text(&findings, true);
        assert!(
            text.contains("blk.[20-21].attn_norm.weight => model.language_model.layers.[20-21].input_layernorm.weight : F32 => BF16 : [1024]"),
            "{text}"
        );
        assert!(
            !text.contains("blk.20.attn_norm.weight =>")
                && !text.contains("blk.21.attn_norm.weight =>"),
            "{text}"
        );
        assert!(
            text.contains("blk.22.attn_norm.weight => model.language_model.layers.21.input_layernorm.weight : F32 => BF16 : [1024]"),
            "{text}"
        );
        assert!(
            text.contains("blk.24.attn_norm.weight => model.language_model.layers.24.input_layernorm.weight : F32 => BF16 : [1024]"),
            "{text}"
        );
        assert!(text.contains("blk.30.ssm_conv1d.weight =>"), "{text}");
        assert!(
            text.contains("blk.[40-41].attn_norm.weight => model.language_model.layers.7.input_layernorm.weight : F32 => BF16 : [1024]"),
            "{text}"
        );
        let plain = render_text(&findings, false);
        assert!(plain.contains("blk.20.attn_norm.weight =>"), "{plain}");
        assert!(!plain.contains("[20-21]"), "{plain}");
    }
}
