//! Compare two tensor containers. Names, canonical dtypes, canonical shapes,
//! and logical tensor bytes. GGUF alignment padding is not part of the value.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::archive::{self, Archive, Inner, Member};
use crate::cli::{Cli, DEFAULT_MAX_HEADER};
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
    Dtype {
        name: String,
        left: String,
        right: String,
    },
    Shape {
        name: String,
        left: String,
        right: String,
    },
    Order {
        name: String,
        left: String,
        right: String,
    },
    Values {
        name: String,
        at: u64,
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
    Pair {
        left: String,
        right: String,
    },
}

impl Finding {
    fn is_difference(&self) -> bool {
        !matches!(self, Finding::Names { .. } | Finding::Pair { .. })
    }
}

struct Side {
    label: String,
    dtype: String,
    shape: Vec<u64>,
    fortran: bool,
    payload: Payload,
}

pub fn from_cli(cli: &Cli) -> Result<i32> {
    let left = cli
        .file
        .as_deref()
        .ok_or_else(|| msg("diff needs two archive paths"))?;
    let right = cli
        .operands
        .first()
        .ok_or_else(|| msg("diff needs two archive paths"))?;
    if cli.operands.len() != 1 {
        return Err(msg("diff needs two archive paths"));
    }
    compare(
        left,
        Path::new(right),
        cli.json,
        cli.max_header,
        cli.match_names,
    )
}

pub fn compare(
    left: &Path,
    right: &Path,
    json_out: bool,
    max_header: u64,
    match_names: bool,
) -> Result<i32> {
    let left_archive = archive::open(left, None, max_header, None, None)?;
    let right_archive = archive::open(right, None, max_header, None, None)?;
    let findings = findings_of(&left_archive, &right_archive, match_names)?;
    let same = !findings.iter().any(Finding::is_difference);
    if json_out {
        println!("{}", render_json(&findings, same));
    } else if !findings.is_empty() {
        println!("{}", render_text(&findings));
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
    for name in left_members.keys() {
        if !linked_left.contains(name) {
            findings.push(Finding::OnlyLeft {
                path: left.path.display().to_string(),
                name: name.clone(),
            });
        }
    }
    for name in right_members.keys() {
        if !linked_right.contains(name) {
            findings.push(Finding::OnlyRight {
                path: right.path.display().to_string(),
                name: name.clone(),
            });
        }
    }
    for link in links {
        if link.left != link.right {
            findings.push(Finding::Pair {
                left: link.left.clone(),
                right: link.right.clone(),
            });
        }
        let label = if link.left == link.right {
            link.left.clone()
        } else {
            format!("{}  =  {}", link.left, link.right)
        };
        let left_side = prepare(left, left_members.get(&link.left).unwrap())?;
        let right_side = prepare(right, right_members.get(&link.right).unwrap())?;
        push_member(&mut findings, &label, &left_side, &right_side)?;
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
    if !match_names {
        return (links, None);
    }
    let hit = crate::gguf_names::match_free(left_archive, right_archive, &left_free, &right_free);
    let note = if left_free.is_empty() && right_free.is_empty() {
        None
    } else {
        Some(crate::gguf_names::note(&hit.arches))
    };
    if hit.pairs.is_empty() {
        return (links, note);
    }
    for (left_name, right_name) in hit.pairs {
        links.push(Link {
            left: left_name,
            right: right_name,
        });
    }
    links.sort_by(|left_link, right_link| {
        left_link
            .left
            .cmp(&right_link.left)
            .then(left_link.right.cmp(&right_link.right))
    });
    (links, note)
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

fn push_member(findings: &mut Vec<Finding>, name: &str, left: &Side, right: &Side) -> Result<()> {
    if left.fortran != right.fortran {
        findings.push(Finding::Order {
            name: name.to_string(),
            left: order_word(left.fortran),
            right: order_word(right.fortran),
        });
    }
    let same_type = left.dtype == right.dtype;
    if !same_type {
        findings.push(Finding::Dtype {
            name: name.to_string(),
            left: left.label.clone(),
            right: right.label.clone(),
        });
    }
    let same_shape = left.shape == right.shape;
    if !same_shape {
        findings.push(Finding::Shape {
            name: name.to_string(),
            left: format_shape(&left.shape),
            right: format_shape(&right.shape),
        });
    }
    if same_type && same_shape {
        if let Some(at) = first_mismatch(&left.payload, &right.payload)? {
            findings.push(Finding::Values {
                name: name.to_string(),
                at,
                left_len: left.payload.len(),
                right_len: right.payload.len(),
            });
        }
    }
    Ok(())
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

fn render_text(findings: &[Finding]) -> String {
    findings
        .iter()
        .map(|finding| match finding {
            Finding::Format { left, right } => format!("format: {left}  vs  {right}"),
            Finding::OnlyLeft { path, name } => format!("only in {path}: {name}"),
            Finding::OnlyRight { path, name } => format!("only in {path}: {name}"),
            Finding::Dtype { name, left, right } => format!("dtype: {name}  {left}  vs  {right}"),
            Finding::Shape { name, left, right } => format!("shape: {name}  {left}  vs  {right}"),
            Finding::Order { name, left, right } => format!("order: {name}  {left}  vs  {right}"),
            Finding::Values {
                name,
                at,
                left_len,
                right_len,
            } => {
                if left_len == right_len {
                    format!("values: {name}  differ at byte {at}")
                } else {
                    format!("values: {name}  differ at byte {at} ({left_len} vs {right_len} bytes)")
                }
            }
            Finding::MetaOnly { path, key } => format!("metadata: {key}  only in {path}"),
            Finding::MetaValue { key, left, right } => {
                format!("metadata: {key}  {left}  vs  {right}")
            }
            Finding::Names { table } => format!("names: {table}"),
            Finding::Pair { left, right } => format!("{left}  =  {right}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
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
            Finding::Dtype { name, left, right } => {
                json!({"kind": "dtype", "name": name, "left": left, "right": right})
            }
            Finding::Shape { name, left, right } => {
                json!({"kind": "shape", "name": name, "left": left, "right": right})
            }
            Finding::Order { name, left, right } => {
                json!({"kind": "order", "name": name, "left": left, "right": right})
            }
            Finding::Values {
                name,
                at,
                left_len,
                right_len,
            } => json!({
                "kind": "values",
                "name": name,
                "at": at,
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
            Finding::Pair { left, right } => {
                json!({"kind": "pair", "left": left, "right": right})
            }
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
            println!("tensdiff {}", env!("CARGO_PKG_VERSION"));
            Ok(0)
        }
        DiffCommand::Run {
            json,
            match_names,
            left,
            right,
        } => compare(&left, &right, json, DEFAULT_MAX_HEADER, match_names),
    }
}

enum DiffCommand {
    Help,
    Version,
    Run {
        json: bool,
        match_names: bool,
        left: PathBuf,
        right: PathBuf,
    },
}

fn parse_diff(args: &[String]) -> Result<DiffCommand> {
    let mut json = false;
    let mut match_names = false;
    let mut paths = Vec::new();
    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => return Ok(DiffCommand::Help),
            "--version" => return Ok(DiffCommand::Version),
            "--json" => json = true,
            "--match-names" => match_names = true,
            other if other.starts_with('-') => {
                return Err(msg(format!("unknown option {other}")));
            }
            other => paths.push(PathBuf::from(other)),
        }
    }
    if paths.len() != 2 {
        return Err(msg("usage: tensdiff [--json] [--match-names] LEFT RIGHT"));
    }
    Ok(DiffCommand::Run {
        json,
        match_names,
        left: paths.remove(0),
        right: paths.remove(0),
    })
}

pub fn diff_help() -> &'static str {
    "\
tensdiff — compare two tensor containers

Usage:
  tensdiff [--json] [--match-names] LEFT RIGHT

Exit status:
  0  same names, types, shapes, values, and metadata
  1  a difference was found
  2  a file could not be read

The container format is reported and does not stop the member comparison.
GGUF shapes are reversed to canonical order. Alignment padding is not compared.
Quantized blocks are compared as stored bytes.
--match-names pairs leftover names through gguf-tensor-roles.json.
general.architecture selects the table. Otherwise the smallest role set that
explains every remaining name is used, and a tie is reported as
\"may be A, B, or C\". Vision tensors are not pairs in a language file.
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
        let code = compare(&left, &right, false, DEFAULT_MAX_HEADER, false).unwrap();
        assert_eq!(code, 0);
    }

    #[test]
    fn a_changed_byte_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let (left, right) = write_pair(dir.path(), &[1.0, 2.0], &[1.0, 3.0]);
        let left_archive = archive::open(&left, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let right_archive = archive::open(&right, None, DEFAULT_MAX_HEADER, None, None).unwrap();
        let text = render_text(&findings_of(&left_archive, &right_archive, false).unwrap());
        assert!(text.contains("values: weight  differ at byte 6"), "{text}");
        assert!(text.contains("metadata: source  left  vs  right"), "{text}");
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
        let text = render_text(&findings_of(&left_archive, &right_archive, false).unwrap());
        assert!(
            text.contains("only in") && text.contains("only_left"),
            "{text}"
        );
        assert!(text.contains("only_right"), "{text}");
        assert!(text.contains("dtype: keep  F32  vs  F16"), "{text}");
        assert!(!text.contains("values: keep"), "{text}");
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
        let text = render_text(&findings_of(&left_archive, &right_archive, false).unwrap());
        assert!(text.contains("format:"), "{text}");
        assert!(!text.contains("values:"), "{text}");
        assert!(!text.contains("dtype:"), "{text}");
        assert!(!text.contains("shape:"), "{text}");
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
        let plain = render_text(&findings_of(&hf_archive, &gguf_archive, false).unwrap());
        assert!(plain.contains(hf) && plain.contains("only in"), "{plain}");
        assert!(!plain.contains("names:"), "{plain}");
        let text = render_text(&findings_of(&hf_archive, &gguf_archive, true).unwrap());
        assert!(text.contains("names: no table matched"), "{text}");
        assert!(!text.contains(" = "), "{text}");
        assert!(text.contains("extra.weight"), "{text}");
        assert_eq!(
            compare(&hf_path, &gguf_path, false, DEFAULT_MAX_HEADER, true).unwrap(),
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
        let paired_text = render_text(&findings_of(&hf_archive, &paired_archive, true).unwrap());
        assert!(
            paired_text.contains("names:") && !paired_text.contains("no table matched"),
            "{paired_text}"
        );
        assert!(
            paired_text.contains(&format!("{hf}  =  {gguf}")),
            "{paired_text}"
        );
        assert!(!paired_text.contains("values:"), "{paired_text}");
        let flipped = render_text(&findings_of(&paired_archive, &hf_archive, true).unwrap());
        assert!(flipped.contains(&format!("{gguf}  =  {hf}")), "{flipped}");
        assert_eq!(
            compare(&hf_path, &paired, false, DEFAULT_MAX_HEADER, true).unwrap(),
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
        let none = render_text(&findings_of(&hf_archive, &other_archive, true).unwrap());
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
        let text = render_text(&findings_of(&hf_archive, &gguf_archive, true).unwrap());
        assert!(text.contains("names: qwen35"), "{text}");
        assert!(text.contains(&format!("{hf}  =  {gguf}")), "{text}");
        assert!(text.contains("custom.bias"), "{text}");
        assert!(!text.contains("values:"), "{text}");
    }
}
