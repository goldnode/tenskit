//! Reads tensdataspecs_readonly and fails the build if the dtype registry,
//! the generated mapping, and the format ids disagree. The Rust tables are
//! emitted from those files so they are not a second copy.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let specs = manifest.join("tensdataspecs_readonly/specs");
    println!("cargo:rerun-if-changed={}", specs.display());

    let yaml_dtypes = parse_dtype_registry(&specs.join("mappings/dtypes.yaml"));
    let mapping = parse_mapping(&specs.join("mappings/dtype-mapping.json"));
    check_registry(&yaml_dtypes, &mapping);

    let formats = parse_formats(&specs.join("formats"));
    let ggml = parse_gguf_types(&specs.join("formats/gguf.yaml"));
    check_gguf(&ggml, &mapping, &yaml_dtypes);

    let safe = safetensors_rows(&yaml_dtypes, &mapping);
    let name_tables = parse_name_tables(&specs.join("mappings/metadata-mapping.yaml"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(
        out_dir.join("spec_tables.rs"),
        render(&yaml_dtypes, &formats, &ggml, &safe, &name_tables),
    )
    .unwrap();

    for row in &formats {
        println!(
            "cargo::rustc-check-cfg=cfg(format_{})",
            row.id.replace('-', "_")
        );
    }
    let enabled = enabled_format_ids(&manifest, &formats);
    for id in &enabled {
        println!("cargo:rustc-cfg=format_{}", id.replace('-', "_"));
    }
    fs::write(
        out_dir.join("enabled_formats.rs"),
        render_enabled(&manifest.join("src/formats"), &enabled),
    )
    .unwrap();
}

/// Ids turned on by `local/supported-formats.txt`. Each one must be a spec id
/// with a source file under `src/formats/`.
fn enabled_format_ids(manifest: &Path, formats: &[FormatRow]) -> Vec<String> {
    let path = manifest.join("local/supported-formats.txt");
    println!("cargo:rerun-if-changed={}", path.display());
    let text =
        fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let known: BTreeSet<&str> = formats.iter().map(|row| row.id.as_str()).collect();
    let mut ids = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.ends_with(':') {
            continue;
        }
        if !known.contains(line) {
            panic!(
                "{}:{}: {line} is not a format id in tensdataspecs_readonly/specs/formats",
                path.display(),
                line_no + 1
            );
        }
        if ids.iter().any(|id: &String| id == line) {
            panic!("{} lists {line} twice", path.display());
        }
        let source = manifest.join("src/formats").join(module_file(line));
        if !source.is_file() {
            panic!(
                "{} enables {line}, but {} is missing",
                path.display(),
                source.display()
            );
        }
        ids.push(line.to_string());
    }
    if ids.iter().any(|id| id == "npz") && !ids.iter().any(|id| id == "npy") {
        panic!("npz is enabled and npy is not; the npz reader is a zip of npy members");
    }
    ids
}

fn module_file(id: &str) -> String {
    match id {
        "raw-sidecar" => "rawtensor.rs".to_string(),
        "compressed-tensors" => "compressed_tensors.rs".to_string(),
        other => format!("{other}.rs"),
    }
}

fn module_name(id: &str) -> String {
    match id {
        "raw-sidecar" => "rawtensor".to_string(),
        "compressed-tensors" => "compressed_tensors".to_string(),
        other => other.replace('-', "_"),
    }
}

fn render_enabled(dir: &Path, ids: &[String]) -> String {
    let mut out = String::from("// Generated from local/supported-formats.txt. Do not edit.\n");
    let list = ids
        .iter()
        .map(|id| format!("    \"{id}\""))
        .collect::<Vec<_>>()
        .join(",\n");
    out.push_str(&format!(
        "#[allow(dead_code)]\npub static ENABLED_IDS: &[&str] = &[\n{list},\n];\n\n"
    ));
    for id in ids {
        let source = dir.join(module_file(id));
        out.push_str(&format!(
            "#[path = \"{}\"]\npub mod {};\n\n",
            source.display(),
            module_name(id)
        ));
    }
    out
}

#[derive(Clone)]
struct Canon {
    id: String,
    kind: String,
    bits_num: u32,
    bits_den: u32,
    block_elements: u32,
    block_bytes: u32,
}

#[derive(Clone)]
struct MapDtype {
    id: String,
    kind: String,
    bits: f64,
    safetensors: Vec<String>,
    npy: Vec<String>,
    gguf: Vec<String>,
}

#[derive(Clone)]
struct FormatRow {
    id: String,
    depth: String,
    category: String,
    extensions: Vec<String>,
}

#[derive(Clone)]
struct GgmlType {
    code: u32,
    name: String,
    canonical: String,
    element: bool,
    removed: bool,
    bits: u32,
    block_elements: u32,
    block_bytes: u32,
}

fn parse_dtype_registry(path: &Path) -> Vec<Canon> {
    let text =
        fs::read_to_string(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let mut rows = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("- {") && !trimmed.starts_with("-{") {
            continue;
        }
        let body = trimmed.trim_start_matches('-').trim();
        let fields = parse_flow_map(body);
        let get = |key: &str| {
            fields
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, value)| value.clone())
        };
        let id = get("id").unwrap_or_else(|| panic!("dtype row without id in {}", path.display()));
        let kind = get("kind").unwrap_or_else(|| panic!("{id} has no kind"));
        let bits = get("bits").unwrap_or_else(|| panic!("{id} has no bits"));
        let (bits_num, bits_den) = rational(&bits);
        let block_elements = get("block_elements")
            .map(|value| value.parse().unwrap())
            .unwrap_or(0);
        let block_bytes = get("block_bytes")
            .map(|value| value.parse().unwrap())
            .unwrap_or(0);
        rows.push(Canon {
            id,
            kind,
            bits_num,
            bits_den,
            block_elements,
            block_bytes,
        });
    }
    if rows.is_empty() {
        panic!("{} has no dtype rows", path.display());
    }
    rows
}

fn parse_mapping(path: &Path) -> Vec<MapDtype> {
    let text =
        fs::read_to_string(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    let rows = value["dtypes"].as_array().expect("dtypes array");
    rows.iter()
        .map(|row| {
            let formats = &row["formats"];
            MapDtype {
                id: row["id"].as_str().unwrap().to_string(),
                kind: row["kind"].as_str().unwrap().to_string(),
                bits: row["bits"].as_f64().unwrap(),
                safetensors: string_list(&formats["safetensors"]),
                npy: string_list(&formats["npy"]),
                gguf: string_list(&formats["gguf"]),
            }
        })
        .collect()
}

fn string_list(value: &serde_json::Value) -> Vec<String> {
    value
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .filter_map(|item| item.as_str().map(str::to_string))
        .collect()
}

fn check_registry(yaml: &[Canon], mapping: &[MapDtype]) {
    let yaml_ids: BTreeSet<_> = yaml.iter().map(|row| row.id.as_str()).collect();
    let map_ids: BTreeSet<_> = mapping.iter().map(|row| row.id.as_str()).collect();
    if yaml_ids != map_ids {
        let missing: Vec<_> = yaml_ids.difference(&map_ids).collect();
        let extra: Vec<_> = map_ids.difference(&yaml_ids).collect();
        panic!(
            "dtypes.yaml and dtype-mapping.json disagree. missing from mapping: {missing:?}. extra in mapping: {extra:?}"
        );
    }
    for row in yaml {
        let mapped = mapping.iter().find(|item| item.id == row.id).unwrap();
        if mapped.kind != row.kind {
            panic!(
                "{} kind is {} in yaml and {} in the mapping",
                row.id, row.kind, mapped.kind
            );
        }
        let bits = row.bits_num as f64 / row.bits_den as f64;
        if (bits - mapped.bits).abs() > 1e-6 {
            panic!(
                "{} bits are {bits} in yaml and {} in the mapping",
                row.id, mapped.bits
            );
        }
    }
}

fn parse_formats(dir: &Path) -> Vec<FormatRow> {
    let mut rows = Vec::new();
    let mut files: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("yaml"))
        .collect();
    files.sort();
    for path in files {
        let text = fs::read_to_string(&path).unwrap();
        let mut in_format = false;
        let mut id = None;
        let mut depth = None;
        let mut category = None;
        let mut extensions = Vec::new();
        for line in text.lines() {
            if !in_format {
                if line == "format:" {
                    in_format = true;
                }
                continue;
            }
            if !line.starts_with(' ') && !line.is_empty() {
                break;
            }
            let trimmed = line.trim();
            if let Some(value) = trimmed.strip_prefix("id:") {
                id = Some(unquote(value.trim()));
            } else if let Some(value) = trimmed.strip_prefix("depth:") {
                depth = Some(unquote(value.trim()));
            } else if let Some(value) = trimmed.strip_prefix("category:") {
                category = Some(unquote(value.trim()));
            } else if let Some(value) = trimmed.strip_prefix("extensions:") {
                extensions = parse_flow_list(value.trim());
            }
        }
        let id = id.unwrap_or_else(|| panic!("{} has no format.id", path.display()));
        let depth = depth.unwrap_or_else(|| panic!("{id} has no depth"));
        let category = category.unwrap_or_else(|| panic!("{id} has no category"));
        rows.push(FormatRow {
            id,
            depth,
            category,
            extensions,
        });
    }
    let mut seen = BTreeSet::new();
    for row in &rows {
        if !seen.insert(row.id.as_str()) {
            panic!("duplicate format id {}", row.id);
        }
    }
    rows
}

fn parse_gguf_types(path: &Path) -> (u32, Vec<GgmlType>) {
    let text = fs::read_to_string(path).unwrap();
    let count = text
        .lines()
        .find_map(|line| {
            let trimmed = line.trim();
            if trimmed.contains("name: COUNT") && trimmed.contains("value:") {
                let value = trimmed.split("value:").nth(1)?;
                let number: String = value
                    .trim()
                    .chars()
                    .take_while(|ch| ch.is_ascii_digit())
                    .collect();
                number.parse().ok()
            } else {
                None
            }
        })
        .unwrap_or_else(|| panic!("gguf.yaml has no GGML_TYPE_COUNT"));

    let start = text
        .find("\ndtypes:\n")
        .map(|index| index + 1)
        .unwrap_or_else(|| panic!("gguf.yaml has no dtypes section"));
    let section = &text[start..];
    let end = section
        .find("\ntensor_descriptor:")
        .unwrap_or_else(|| panic!("gguf.yaml dtypes section does not end"));
    let section = &section[..end];

    let mut types = Vec::new();
    let lines: Vec<&str> = section.lines().collect();
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index].trim();
        if let Some(body) = line.strip_prefix("- {") {
            let fields = parse_flow_map(&format!("{{{body}"));
            types.push(ggml_from_fields(&fields));
            index += 1;
            continue;
        }
        if line.starts_with("- code:") {
            let mut fields = Vec::new();
            let code = line.trim_start_matches("- code:").trim().to_string();
            fields.push(("code".to_string(), code));
            index += 1;
            while index < lines.len() {
                let next = lines[index];
                if next.trim_start().starts_with("- ") && next.starts_with("  - ") {
                    break;
                }
                let trimmed = next.trim();
                if let Some((key, value)) = trimmed.split_once(':') {
                    if matches!(
                        key,
                        "name"
                            | "canonical"
                            | "storage"
                            | "status"
                            | "block_elements"
                            | "block_bytes"
                            | "bits"
                    ) && !value.trim().is_empty()
                        && !value.trim().starts_with('|')
                    {
                        fields.push((key.to_string(), value.trim().trim_matches('"').to_string()));
                    }
                }
                index += 1;
            }
            types.push(ggml_from_fields(&fields));
            continue;
        }
        index += 1;
    }
    if types.is_empty() {
        panic!("gguf.yaml produced no ggml types");
    }
    (count, types)
}

fn ggml_from_fields(fields: &[(String, String)]) -> GgmlType {
    let get = |key: &str| {
        fields
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.clone())
    };
    let code: u32 = get("code").unwrap().parse().unwrap();
    let name = get("name").unwrap();
    let canonical = get("canonical").unwrap_or_else(|| "none".into());
    let storage = get("storage").unwrap_or_else(|| "element".into());
    let status = get("status").unwrap_or_else(|| "active".into());
    let bits_text = get("bits").unwrap_or_else(|| "0".into());
    let (num, den) = rational(&bits_text);
    let bits = if den == 1 { num } else { 0 };
    GgmlType {
        code,
        name,
        canonical,
        element: storage == "element",
        removed: status == "removed",
        bits,
        block_elements: get("block_elements")
            .map(|value| value.parse().unwrap())
            .unwrap_or(0),
        block_bytes: get("block_bytes")
            .map(|value| value.parse().unwrap())
            .unwrap_or(0),
    }
}

fn check_gguf(ggml: &(u32, Vec<GgmlType>), mapping: &[MapDtype], yaml: &[Canon]) {
    let (count, types) = ggml;
    if *count == 0 {
        panic!("GGML_TYPE_COUNT is 0");
    }
    let by_code: BTreeMap<u32, &GgmlType> = types.iter().map(|row| (row.code, row)).collect();
    for mapped in mapping {
        for entry in &mapped.gguf {
            let mut parts = entry.split_whitespace();
            let code: u32 = parts
                .next()
                .unwrap_or_else(|| panic!("bad gguf mapping entry {entry}"))
                .parse()
                .unwrap_or_else(|_| panic!("bad gguf code in {entry}"));
            let name = parts
                .next()
                .unwrap_or_else(|| panic!("bad gguf name in {entry}"));
            let row = by_code
                .get(&code)
                .unwrap_or_else(|| panic!("dtype-mapping gguf code {code} is not in gguf.yaml"));
            if row.name != name {
                panic!(
                    "gguf code {code} is {} in gguf.yaml and {name} in dtype-mapping.json",
                    row.name
                );
            }
            if row.canonical != mapped.id {
                panic!(
                    "gguf {} canonical is {} in gguf.yaml and {} in the mapping",
                    row.name, row.canonical, mapped.id
                );
            }
        }
    }
    for row in types {
        if row.removed {
            continue;
        }
        if row.canonical == "none" {
            continue;
        }
        if !yaml.iter().any(|canon| canon.id == row.canonical) {
            panic!(
                "gguf type {} canonical {} is not in dtypes.yaml",
                row.name, row.canonical
            );
        }
        let mapped = mapping
            .iter()
            .find(|item| item.id == row.canonical)
            .unwrap();
        let needle = format!("{} {}", row.code, row.name);
        if !mapped.gguf.iter().any(|entry| entry == &needle) {
            panic!("gguf type {needle} is missing from dtype-mapping.json");
        }
    }
}

struct SafeRow {
    name: String,
    bits: u32,
    numpy: Option<String>,
    canonical: String,
}

fn safetensors_rows(yaml: &[Canon], mapping: &[MapDtype]) -> Vec<SafeRow> {
    let mut rows = Vec::new();
    for canon in yaml {
        let mapped = mapping.iter().find(|item| item.id == canon.id).unwrap();
        let Some(name) = mapped.safetensors.first() else {
            continue;
        };
        if canon.bits_den != 1 {
            panic!("safetensors dtype {name} has fractional bits");
        }
        let numpy = mapped.npy.iter().find(|code| is_numpy_descr(code)).cloned();
        rows.push(SafeRow {
            name: name.clone(),
            bits: canon.bits_num,
            numpy,
            canonical: canon.id.clone(),
        });
    }
    rows.sort_by(|left, right| left.name.cmp(&right.name));
    rows
}

fn is_numpy_descr(code: &str) -> bool {
    let rest = code.trim_start_matches(['<', '>', '|', '=']);
    rest.len() >= 2
        && rest
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_alphabetic())
        && rest.chars().skip(1).all(|ch| ch.is_ascii_digit())
}

struct NameTable {
    id: String,
    patterns: Vec<(String, String)>,
}

/// Hugging Face <-> GGUF tensor names from metadata-mapping.yaml.
/// A `(dropped)` GGUF name is not a pair.
fn parse_name_tables(path: &Path) -> Vec<NameTable> {
    let text =
        fs::read_to_string(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    let mut tables = Vec::new();
    let mut current: Option<NameTable> = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if !line.starts_with(' ') && !line.starts_with('\t') && trimmed.ends_with(':') {
            if let Some(table) = current.take() {
                if !table.patterns.is_empty() {
                    tables.push(table);
                }
            }
            current = Some(NameTable {
                id: trimmed.trim_end_matches(':').to_string(),
                patterns: Vec::new(),
            });
            continue;
        }
        if !trimmed.starts_with("- {") || !trimmed.contains("hf:") || !trimmed.contains("gguf:") {
            continue;
        }
        let Some(table) = current.as_mut() else {
            continue;
        };
        let fields = parse_flow_map(trimmed.trim_start_matches('-').trim());
        let hf = fields
            .iter()
            .find(|(key, _)| key == "hf")
            .map(|(_, value)| value.clone());
        let gguf = fields
            .iter()
            .find(|(key, _)| key == "gguf")
            .map(|(_, value)| value.clone());
        let (Some(hf), Some(gguf)) = (hf, gguf) else {
            continue;
        };
        if gguf == "(dropped)" || hf.is_empty() || gguf.is_empty() {
            continue;
        }
        table.patterns.push((hf, gguf));
    }
    if let Some(table) = current {
        if !table.patterns.is_empty() {
            tables.push(table);
        }
    }
    if tables.is_empty() {
        panic!("{} has no tensor name tables", path.display());
    }
    tables
}

fn render(
    canon: &[Canon],
    formats: &[FormatRow],
    ggml: &(u32, Vec<GgmlType>),
    safe: &[SafeRow],
    name_tables: &[NameTable],
) -> String {
    let mut out = String::from("// Generated from tensdataspecs_readonly. Do not edit.\n\n");
    out.push_str(&format!("pub const GGML_TYPE_COUNT: u32 = {};\n\n", ggml.0));
    out.push_str("pub struct CanonRow {\n    pub id: &'static str,\n    pub kind: &'static str,\n    pub bits_num: u32,\n    pub bits_den: u32,\n    pub block_elements: u32,\n    pub block_bytes: u32,\n}\n\n");
    out.push_str("pub static CANON: &[CanonRow] = &[\n");
    for row in canon {
        out.push_str(&format!(
            "    CanonRow {{ id: {}, kind: {}, bits_num: {}, bits_den: {}, block_elements: {}, block_bytes: {} }},\n",
            rust_str(&row.id),
            rust_str(&row.kind),
            row.bits_num,
            row.bits_den,
            row.block_elements,
            row.block_bytes
        ));
    }
    out.push_str("];\n\n");
    out.push_str("#[allow(dead_code)]\npub struct FormatRow {\n    pub id: &'static str,\n    pub depth: &'static str,\n    pub category: &'static str,\n    pub extensions: &'static [&'static str],\n}\n\n");
    out.push_str("#[allow(dead_code)]\npub static SPEC_FORMATS: &[FormatRow] = &[\n");
    for row in formats {
        let extensions = row
            .extensions
            .iter()
            .map(|ext| rust_str(ext))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!(
            "    FormatRow {{ id: {}, depth: {}, category: {}, extensions: &[{extensions}] }},\n",
            rust_str(&row.id),
            rust_str(&row.depth),
            rust_str(&row.category)
        ));
    }
    out.push_str("];\n\n");
    out.push_str("#[allow(dead_code)]\npub struct GgmlRow {\n    pub code: u32,\n    pub name: &'static str,\n    pub canonical: Option<&'static str>,\n    pub element: bool,\n    pub removed: bool,\n    pub bits: u32,\n    pub block_elements: u32,\n    pub block_bytes: u32,\n}\n\n");
    out.push_str("pub static GGML_TYPES: &[GgmlRow] = &[\n");
    for row in &ggml.1 {
        let canonical = if row.canonical == "none" {
            "None".to_string()
        } else {
            format!("Some({})", rust_str(&row.canonical))
        };
        out.push_str(&format!(
            "    GgmlRow {{ code: {}, name: {}, canonical: {canonical}, element: {}, removed: {}, bits: {}, block_elements: {}, block_bytes: {} }},\n",
            row.code,
            rust_str(&row.name),
            row.element,
            row.removed,
            row.bits,
            row.block_elements,
            row.block_bytes
        ));
    }
    out.push_str("];\n\n");
    out.push_str("pub struct SafeRow {\n    pub name: &'static str,\n    pub bits: u32,\n    pub numpy: Option<&'static str>,\n    pub canonical: &'static str,\n}\n\n");
    out.push_str("pub static SAFE_ROWS: &[SafeRow] = &[\n");
    for row in safe {
        let numpy = match &row.numpy {
            Some(code) => format!("Some({})", rust_str(code)),
            None => "None".to_string(),
        };
        out.push_str(&format!(
            "    SafeRow {{ name: {}, bits: {}, numpy: {numpy}, canonical: {} }},\n",
            rust_str(&row.name),
            row.bits,
            rust_str(&row.canonical)
        ));
    }
    out.push_str("];\n\n");
    out.push_str(
        "#[allow(dead_code)]\npub struct NamePattern {\n    pub hf: &'static str,\n    pub gguf: &'static str,\n}\n\n",
    );
    out.push_str("#[allow(dead_code)]\npub struct NameTable {\n    pub id: &'static str,\n    pub patterns: &'static [NamePattern],\n}\n\n");
    out.push_str("#[allow(dead_code)]\npub static NAME_TABLES: &[NameTable] = &[\n");
    for table in name_tables {
        out.push_str(&format!(
            "    NameTable {{ id: {}, patterns: &[\n",
            rust_str(&table.id)
        ));
        for (hf, gguf) in &table.patterns {
            out.push_str(&format!(
                "        NamePattern {{ hf: {}, gguf: {} }},\n",
                rust_str(hf),
                rust_str(gguf)
            ));
        }
        out.push_str("    ] },\n");
    }
    out.push_str("];\n");
    out
}

fn parse_flow_map(input: &str) -> Vec<(String, String)> {
    let text = input.trim();
    let text = text
        .strip_prefix('{')
        .unwrap_or_else(|| panic!("not a flow map: {input}"))
        .strip_suffix('}')
        .unwrap_or_else(|| panic!("not a flow map: {input}"));
    let bytes = text.as_bytes();
    let mut index = 0;
    let mut fields = Vec::new();
    while index < bytes.len() {
        while index < bytes.len() && (bytes[index] == b' ' || bytes[index] == b',') {
            index += 1;
        }
        if index >= bytes.len() || bytes[index] == b'}' {
            break;
        }
        let key_start = index;
        while index < bytes.len() && bytes[index] != b':' {
            index += 1;
        }
        if index >= bytes.len() {
            break;
        }
        let key = text[key_start..index].trim().to_string();
        index += 1;
        while index < bytes.len() && bytes[index] == b' ' {
            index += 1;
        }
        let value = if index < bytes.len() && (bytes[index] == b'"' || bytes[index] == b'\'') {
            let quote = bytes[index];
            index += 1;
            let start = index;
            while index < bytes.len() && bytes[index] != quote {
                index += 1;
            }
            let value = text[start..index].to_string();
            index += 1;
            value
        } else if index < bytes.len() && bytes[index] == b'[' {
            let start = index;
            while index < bytes.len() && bytes[index] != b']' {
                index += 1;
            }
            index += 1;
            text[start..index].to_string()
        } else {
            let start = index;
            while index < bytes.len() && bytes[index] != b',' {
                index += 1;
            }
            text[start..index].trim().to_string()
        };
        if !key.is_empty() {
            fields.push((key, value));
        }
    }
    fields
}

fn parse_flow_list(input: &str) -> Vec<String> {
    let text = input.trim();
    if text.is_empty() || text == "[]" {
        return Vec::new();
    }
    let inner = text
        .strip_prefix('[')
        .and_then(|text| text.strip_suffix(']'))
        .unwrap_or_else(|| panic!("not a flow list: {input}"));
    if inner.trim().is_empty() {
        return Vec::new();
    }
    inner
        .split(',')
        .map(|item| unquote(item.trim()))
        .filter(|item| !item.is_empty())
        .collect()
}

fn unquote(text: &str) -> String {
    text.trim_matches(['"', '\'']).to_string()
}

fn rational(text: &str) -> (u32, u32) {
    let text = text.trim();
    if let Some((whole, frac)) = text.split_once('.') {
        let den = 10u32.pow(frac.len() as u32);
        let num: u32 = format!("{whole}{frac}")
            .parse()
            .unwrap_or_else(|_| panic!("bad bits {text}"));
        let divisor = gcd(num, den);
        (num / divisor, den / divisor)
    } else {
        (
            text.parse().unwrap_or_else(|_| panic!("bad bits {text}")),
            1,
        )
    }
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left.max(1)
}

fn rust_str(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
}
