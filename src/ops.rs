use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::archive::{self, Archive, Inner, Member};
use crate::cli::{Cli, FormatName, Op};
use crate::detect::{self, Detected};
use crate::dtype::{self, format_shape};
use crate::error::{io_err, msg, Result};
use crate::io_util::{self, relative_child, safe_member_path};
use crate::payload::Payload;

#[cfg(format_gguf)]
use crate::formats::gguf::{self, MetaValue};
#[cfg(format_npy)]
use crate::formats::npy;
#[cfg(format_npz)]
use crate::formats::npz::{self, NpzMember, NpzMemberBody};
#[cfg(format_safetensors)]
use crate::formats::safetensors::{self, OutTensor};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Modify {
    Append,
    Update,
    Delete,
}

#[derive(Clone)]
struct Planned {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    payload: Payload,
    /// Set when the source is an `.npy` file that can be stored whole.
    npy_file: Option<PathBuf>,
    fortran: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pack {
    Npy,
    Safe,
    Raw,
}

enum ExtractJob {
    #[cfg(format_npy)]
    Npy { path: PathBuf, tensor: Planned },
    #[cfg(format_raw_sidecar)]
    Raw {
        path: PathBuf,
        members: Vec<Planned>,
    },
    #[cfg(format_npz)]
    Npz {
        path: PathBuf,
        members: Vec<Planned>,
    },
    #[cfg(format_safetensors)]
    Safe {
        path: PathBuf,
        members: Vec<Planned>,
    },
    #[cfg(format_pytorch)]
    Pytorch {
        path: PathBuf,
        members: Vec<Planned>,
    },
    #[cfg(format_onnx)]
    Onnx {
        path: PathBuf,
        members: Vec<Planned>,
    },
}

pub fn execute(cli: Cli) -> Result<i32> {
    if cli.directory != Path::new(".") && cli.op != Op::Extract {
        return Err(msg("-C applies to extract"));
    }
    if cli.output.is_some() && matches!(cli.op, Op::List | Op::Info) {
        return Err(msg("-o does not apply to list or info"));
    }
    if cli.json && (cli.columns || cli.fold) {
        return Err(msg("--columns and --fold do not apply to --json"));
    }
    if cli.columns && !matches!(cli.op, Op::List | Op::Info) {
        return Err(msg("--columns applies to list or info"));
    }
    if cli.fold && !matches!(cli.op, Op::List | Op::Info) {
        return Err(msg("--fold applies to list or info"));
    }
    match cli.op {
        Op::List => list(&cli)?,
        Op::Info => info(&cli)?,
        Op::Extract => extract(&cli)?,
        Op::Create => create(&cli)?,
        Op::Append => modify(&cli, Modify::Append)?,
        Op::Update => modify(&cli, Modify::Update)?,
        Op::Delete => modify(&cli, Modify::Delete)?,
    }
    Ok(0)
}

fn list(cli: &Cli) -> Result<()> {
    let archive = open_source(cli)?;
    let mut selected = select(&archive, &cli.operands)?;
    sort_members(cli, &mut selected);
    if cli.json {
        outln!("{}", json_document(&archive, &selected)?);
        return Ok(());
    }
    print_members(&selected, member_style(cli));
    Ok(())
}

fn info(cli: &Cli) -> Result<()> {
    let mut archive = open_source(cli)?;
    if let Some(path) = cli.file.as_ref().filter(|path| path.is_dir()) {
        if let Some(note) = quantization_note(path) {
            archive.notes.push(note);
        }
    }
    let mut members: Vec<&Member> = archive.members.iter().collect();
    sort_members(cli, &mut members);
    if cli.json {
        outln!("{}", json_document(&archive, &members)?);
        return Ok(());
    }
    outln!("format: {}", archive.format.as_str());
    outln!("version: {}", archive.version);
    outln!("path: {}", archive.path.display());
    outln!("members: {}", archive.members.len());
    outln!("data_bytes: {}", archive.data_bytes);
    if archive.metadata.is_empty() {
        outln!("metadata: none");
    } else {
        outln!("metadata:");
        for (key, value) in &archive.metadata {
            outln!("  {key}={}", short_value(value));
        }
    }
    outln!("dtypes:");
    let mut totals: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    for member in &archive.members {
        let slot = totals.entry(crate::dtype::display_name(&member.dtype)).or_insert((0, 0));
        slot.0 += 1;
        slot.1 = slot.1.saturating_add(member.numel.unwrap_or(0));
    }
    if totals.is_empty() {
        outln!("  none");
    } else {
        for (dtype, (count, params)) in totals {
            outln!("  {dtype}\t{count}\t{params}");
        }
    }
    for note in &archive.notes {
        outln!("note: {note}");
    }
    if cli.verbose || cli.columns || cli.fold {
        print_members(&members, member_style(cli));
    }
    Ok(())
}

enum MemberStyle {
    Names,
    Tabs,
    Columns { fold: bool },
}

fn quantization_note(dir: &Path) -> Option<String> {
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
    keys.iter()
        .find_map(|key| object.get(*key).and_then(Value::as_u64))
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

fn sort_members(cli: &Cli, members: &mut [&Member]) {
    if cli.sort {
        members.sort_by(|left, right| crate::listing::name_order(&left.name, &right.name));
    }
}

fn member_style(cli: &Cli) -> MemberStyle {
    if cli.fold || cli.columns {
        MemberStyle::Columns { fold: cli.fold }
    } else if cli.verbose {
        MemberStyle::Tabs
    } else {
        MemberStyle::Names
    }
}

fn print_members(members: &[&Member], style: MemberStyle) {
    match style {
        MemberStyle::Names => {
            for member in members {
                outln!("{}", member.name);
            }
            return;
        }
        MemberStyle::Columns { fold } => {
            let text = crate::listing::render_columns(members, fold);
            if !text.is_empty() {
                outln!("{text}");
            }
            return;
        }
        MemberStyle::Tabs => {}
    }
    outln!("name\tdtype\tshape\tbytes\toffset\tparams\tshard\tdetail");
    for member in members {
        let params = member
            .numel
            .map(|count| count.to_string())
            .unwrap_or_else(|| "-".to_string());
        let shard = member.shard.as_deref().unwrap_or("-");
        let detail = member.detail.as_deref().unwrap_or("-");
        outln!(
            "{}\t{}\t{}\t{}\t{}\t{params}\t{shard}\t{detail}",
            member.name,
            crate::dtype::display_name(&member.dtype),
            format_shape(&member.shape),
            member.nbytes,
            member.file_offset
        );
    }
}

/// A metadata value for the text listing: long values such as a GGUF
/// vocabulary are cut to their first 120 characters. `--json` prints them whole.
/// A cut array gives its item count, then the length of its full JSON text.
fn short_value(value: &str) -> String {
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

fn json_document(archive: &Archive, members: &[&Member]) -> Result<String> {
    let mut metadata = Map::new();
    for (key, value) in &archive.metadata {
        metadata.insert(key.clone(), Value::String(value.clone()));
    }
    let mut listed = Vec::with_capacity(members.len());
    for member in members {
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
        object.insert(
            "numel".into(),
            match member.numel {
                Some(count) => Value::from(count),
                None => Value::Null,
            },
        );
        object.insert(
            "shard".into(),
            match &member.shard {
                Some(shard) => Value::String(shard.clone()),
                None => Value::Null,
            },
        );
        object.insert(
            "detail".into(),
            match &member.detail {
                Some(detail) => Value::String(detail.clone()),
                None => Value::Null,
            },
        );
        listed.push(Value::Object(object));
    }
    let mut root = Map::new();
    root.insert(
        "format".into(),
        Value::String(archive.format.as_str().into()),
    );
    root.insert("version".into(), Value::String(archive.version.clone()));
    root.insert(
        "path".into(),
        Value::String(archive.path.display().to_string()),
    );
    root.insert("members".into(), Value::Array(listed));
    root.insert("metadata".into(), Value::Object(metadata));
    root.insert("data_bytes".into(), Value::from(archive.data_bytes));
    root.insert(
        "notes".into(),
        Value::Array(archive.notes.iter().cloned().map(Value::String).collect()),
    );
    Ok(serde_json::to_string_pretty(&Value::Object(root))?)
}

fn extract(cli: &Cli) -> Result<()> {
    #[cfg(format_gguf)]
    if cli.format == Some(FormatName::Gguf) {
        return Err(msg(
            "extract cannot write GGUF; choose npy, raw, npz, safetensors, pytorch, or onnx",
        ));
    }
    let resolved = archive::resolve_user_path(cli.file.as_ref().unwrap(), cli.file_force_single)?;
    let archive = open_extract_source(&resolved.path, cli)?;
    let selected = select(&archive, &cli.operands)?;
    let jobs = extract_jobs(cli, &archive, &selected)?;
    if cli.dry_run {
        outln!("dry-run: extract {}", archive.path.display());
        for job in &jobs {
            outln!("{}", job_path(job).display());
        }
        return Ok(());
    }
    for job in &jobs {
        refuse_overwrite(job_path(job), None, cli.force)?;
    }
    for job in &jobs {
        ensure_parent(job_path(job))?;
        match job {
            #[cfg(format_npy)]
            ExtractJob::Npy { path, tensor } => write_npy(path, tensor)?,
            #[cfg(format_raw_sidecar)]
            ExtractJob::Raw { path, members } => {
                let tensors = members
                    .iter()
                    .map(|item| {
                        crate::formats::rawtensor::to_out(
                            &item.name,
                            &item.dtype,
                            &item.shape,
                            &item.payload,
                            item.fortran,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                crate::formats::rawtensor::write_archive(path, &Map::new(), &tensors)?;
            }
            #[cfg(format_npz)]
            ExtractJob::Npz { path, members } => {
                let stored = members.iter().map(npz_member).collect::<Result<Vec<_>>>()?;
                npz::write_archive(path, &stored)?;
            }
            #[cfg(format_safetensors)]
            ExtractJob::Safe { path, members } => {
                let tensors = members.iter().map(as_out).collect::<Result<Vec<_>>>()?;
                safetensors::write_file(path, &[], &tensors)?;
            }
            #[cfg(format_pytorch)]
            ExtractJob::Pytorch { path, members } => write_pytorch(path, members)?,
            #[cfg(format_onnx)]
            ExtractJob::Onnx { path, members } => write_onnx(path, &[], members)?,
        }
    }
    Ok(())
}

fn job_path(job: &ExtractJob) -> &Path {
    match job {
        #[cfg(format_npy)]
        ExtractJob::Npy { path, .. } => path,
        #[cfg(format_raw_sidecar)]
        ExtractJob::Raw { path, .. } => path,
        #[cfg(format_npz)]
        ExtractJob::Npz { path, .. } => path,
        #[cfg(format_safetensors)]
        ExtractJob::Safe { path, .. } => path,
        #[cfg(format_pytorch)]
        ExtractJob::Pytorch { path, .. } => path,
        #[cfg(format_onnx)]
        ExtractJob::Onnx { path, .. } => path,
    }
}

fn extract_jobs(cli: &Cli, archive: &Archive, selected: &[&Member]) -> Result<Vec<ExtractJob>> {
    if let Some(format) = combined_format(cli) {
        let path = combined_path(cli, selected, format)?;
        let pack = match format {
            #[cfg(format_npz)]
            FormatName::Npz => Pack::Npy,
            #[cfg(format_raw_sidecar)]
            FormatName::Raw => Pack::Raw,
            #[allow(unreachable_patterns)]
            _ => Pack::Safe,
        };
        let members = selected
            .iter()
            .map(|member| planned_export(archive, member, pack))
            .collect::<Result<Vec<_>>>()?;
        let job = match format {
            #[cfg(format_npz)]
            FormatName::Npz => ExtractJob::Npz { path, members },
            #[cfg(format_raw_sidecar)]
            FormatName::Raw => ExtractJob::Raw { path, members },
            #[cfg(format_pytorch)]
            FormatName::Pytorch => ExtractJob::Pytorch { path, members },
            #[cfg(format_onnx)]
            FormatName::Onnx => ExtractJob::Onnx { path, members },
            #[allow(unreachable_patterns)]
            _ => ExtractJob::Safe { path, members },
        };
        return Ok(vec![job]);
    }
    if selected.len() > 1 && cli.output.is_some() {
        return Err(msg(
            "several members need --format npz, safetensors, or raw, or -o with a .npz, .safetensors, or .rawtensor.json name; omit -o to write one file each",
        ));
    }
    let mut jobs = Vec::with_capacity(selected.len());
    let mut seen = BTreeSet::new();
    for member in selected {
        let path = if selected.len() == 1 {
            if let Some(output) = &cli.output {
                output.clone()
            } else {
                per_file_path(&cli.directory, member, single_encoding(cli, None, member)?)?
            }
        } else {
            per_file_path(&cli.directory, member, single_encoding(cli, None, member)?)?
        };
        if !seen.insert(path.clone()) {
            return Err(msg(format!("two members would write {}", path.display())));
        }
        let encoding = single_encoding(cli, Some(&path), member)?;
        let job = match encoding {
            #[cfg(format_npy)]
            FormatName::Npy => ExtractJob::Npy {
                path,
                tensor: planned_export(archive, member, Pack::Npy)?,
            },
            #[cfg(format_raw_sidecar)]
            FormatName::Raw => ExtractJob::Raw {
                path,
                members: vec![planned_export(archive, member, Pack::Raw)?],
            },
            other => {
                return Err(msg(format!(
                    "extract encoding {} needs one output archive",
                    other.as_str()
                )));
            }
        };
        jobs.push(job);
    }
    Ok(jobs)
}

fn combined_format(cli: &Cli) -> Option<FormatName> {
    match cli.format {
        #[cfg(format_npz)]
        Some(FormatName::Npz) => cli.format,
        #[cfg(format_safetensors)]
        Some(FormatName::Safetensors) => cli.format,
        #[cfg(format_raw_sidecar)]
        Some(FormatName::Raw) => cli.format,
        #[cfg(format_pytorch)]
        Some(FormatName::Pytorch) => cli.format,
        #[cfg(format_onnx)]
        Some(FormatName::Onnx) => cli.format,
        _ => match cli.output.as_deref() {
            #[cfg(format_npz)]
            Some(path) if ends_with(path, ".npz") => Some(FormatName::Npz),
            #[cfg(format_safetensors)]
            Some(path) if ends_with(path, ".safetensors") => Some(FormatName::Safetensors),
            #[cfg(format_raw_sidecar)]
            Some(path) if ends_with(path, ".rawtensor.json") => Some(FormatName::Raw),
            #[cfg(format_pytorch)]
            Some(path) if ends_with(path, ".pt") || ends_with(path, ".pth") => {
                Some(FormatName::Pytorch)
            }
            #[cfg(format_onnx)]
            Some(path) if ends_with(path, ".onnx") => Some(FormatName::Onnx),
            _ => None,
        },
    }
}

fn combined_path(cli: &Cli, selected: &[&Member], format: FormatName) -> Result<PathBuf> {
    if let Some(output) = &cli.output {
        return Ok(output.clone());
    }
    if selected.len() == 1 {
        let ext = match format {
            #[cfg(format_npz)]
            FormatName::Npz => "npz",
            #[cfg(format_raw_sidecar)]
            FormatName::Raw => "rawtensor.json",
            #[cfg(format_pytorch)]
            FormatName::Pytorch => "pt",
            #[cfg(format_onnx)]
            FormatName::Onnx => "onnx",
            #[allow(unreachable_patterns)]
            _ => "safetensors",
        };
        return per_file_path(&cli.directory, selected[0], ext_format(ext));
    }
    Err(msg(format!(
        "extracting several members as {} needs -o",
        format.as_str()
    )))
}

fn ext_format(ext: &str) -> FormatName {
    match ext {
        #[cfg(format_npz)]
        "npz" => FormatName::Npz,
        #[cfg(format_safetensors)]
        "safetensors" => FormatName::Safetensors,
        #[cfg(format_raw_sidecar)]
        "raw" | "rawtensor.json" => FormatName::Raw,
        #[cfg(format_pytorch)]
        "pt" => FormatName::Pytorch,
        #[cfg(format_onnx)]
        "onnx" => FormatName::Onnx,
        #[cfg(format_npy)]
        _ => FormatName::Npy,
        #[cfg(not(format_npy))]
        _ => unreachable!("extract extension {ext} needs npy"),
    }
}

fn per_file_path(dir: &Path, member: &Member, encoding: FormatName) -> Result<PathBuf> {
    let ext = match encoding {
        #[cfg(format_raw_sidecar)]
        FormatName::Raw => "rawtensor.json",
        #[cfg(format_npz)]
        FormatName::Npz => "npz",
        #[cfg(format_safetensors)]
        FormatName::Safetensors => "safetensors",
        #[cfg(format_pytorch)]
        FormatName::Pytorch => "pt",
        #[cfg(format_onnx)]
        FormatName::Onnx => "onnx",
        #[allow(unreachable_patterns)]
        _ => "npy",
    };
    safe_member_path(dir, &member.name, ext)
}

fn single_encoding(cli: &Cli, path: Option<&Path>, member: &Member) -> Result<FormatName> {
    match cli.format {
        #[cfg(format_raw_sidecar)]
        Some(FormatName::Raw) => return Ok(FormatName::Raw),
        #[cfg(format_npy)]
        Some(FormatName::Npy) => return Ok(FormatName::Npy),
        Some(other) => {
            return Err(msg(format!(
                "--format {} is an archive, not a single-member extract",
                other.as_str()
            )));
        }
        None => {}
    }
    if let Some(path) = path {
        #[cfg(format_raw_sidecar)]
        if ends_with(path, ".rawtensor.json") {
            return Ok(FormatName::Raw);
        }
        #[cfg(format_npy)]
        if ends_with(path, ".npy") {
            return Ok(FormatName::Npy);
        }
    }
    #[cfg(format_npy)]
    if numpy_descr(&member.dtype).is_ok() {
        return Ok(FormatName::Npy);
    }
    #[cfg(format_raw_sidecar)]
    {
        let _ = member;
        return Ok(FormatName::Raw);
    }
    #[allow(unreachable_code)]
    Err(msg(format!(
        "tensor {} has no compiled single-file encoding",
        member.name
    )))
}

fn planned_export(archive: &Archive, member: &Member, pack: Pack) -> Result<Planned> {
    // A fortran-order array goes to .npy as stored, with fortran_order set.
    if member.fortran && !matches!(pack, Pack::Npy) {
        return Err(msg(format!(
            "member {} is fortran-order; tenspile does not transpose it, and rawtensor bytes are C order (RAW-009)",
            member.name
        )));
    }
    if matches!(pack, Pack::Npy) {
        numpy_descr(&member.dtype)?;
    }
    let mut shape = member.shape.clone();
    #[cfg(format_gguf)]
    if matches!(archive.inner, Inner::Gguf(_)) {
        shape.reverse();
    }
    Ok(Planned {
        name: member.name.clone(),
        dtype: member.dtype.clone(),
        shape,
        payload: clip(&member.payload, logical_len(archive, member, pack)?)?,
        npy_file: None,
        fortran: member.fortran,
    })
}

fn logical_len(archive: &Archive, member: &Member, pack: Pack) -> Result<u64> {
    #[cfg(not(format_gguf))]
    {
        let _ = (archive, pack);
        return Ok(member.nbytes);
    }
    #[cfg(format_gguf)]
    let Inner::Gguf(file) = &archive.inner
    else {
        return Ok(member.nbytes);
    };
    let tensor = file
        .tensors
        .iter()
        .find(|tensor| tensor.name == member.name)
        .ok_or_else(|| msg(format!("missing GGUF tensor {}", member.name)))?;
    if dtype::is_quant_ggml(tensor.ggml_type) && pack != Pack::Raw {
        return Err(msg(format!(
            "tensor {} is a quantized span; extract it with --format raw",
            member.name
        )));
    }
    let logical = dtype::ggml_payload_bytes(tensor.ggml_type, &tensor.shape)?;
    if logical > member.nbytes {
        return Err(msg(format!(
            "tensor {} is larger than its stored span",
            member.name
        )));
    }
    Ok(logical)
}

fn clip(payload: &Payload, len: u64) -> Result<Payload> {
    if len > payload.len() {
        return Err(msg("export is longer than the stored bytes"));
    }
    Ok(match payload.clone() {
        Payload::Range { path, offset, .. } => Payload::Range { path, offset, len },
        Payload::Zip {
            path, entry, skip, ..
        } => Payload::Zip {
            path,
            entry,
            skip,
            len,
        },
        Payload::Bytes(bytes) => {
            let end = usize::try_from(len).map_err(|_| msg("export is too large"))?;
            Payload::Bytes(bytes[..end].to_vec())
        }
    })
}

fn create(cli: &Cli) -> Result<()> {
    if cli.output.is_some() {
        return Err(msg(
            "the archive to create is -f or the first argument; -o is for update and delete",
        ));
    }
    let dest = cli.file.as_ref().unwrap();
    let format = dest_format(cli, dest)?;
    #[cfg(format_gguf)]
    if format == FormatName::Gguf {
        return Err(msg("creating a GGUF file is not supported"));
    }
    let incoming = load_incoming(cli)?;
    let metadata = string_updates(cli)?;
    #[allow(unreachable_patterns)]
    match format {
        #[cfg(format_safetensors)]
        FormatName::Safetensors => {}
        #[cfg(format_npy)]
        FormatName::Npy => {
            if incoming.len() != 1 {
                return Err(msg("an npy file holds one array; use .npz for several"));
            }
            if !metadata.is_empty() {
                return Err(msg("npy files have no string metadata"));
            }
        }
        #[cfg(format_npz)]
        FormatName::Npz => {
            if incoming.is_empty() {
                return Err(msg("writing an npz archive with no arrays is not supported"));
            }
            if !metadata.is_empty() {
                return Err(msg("npz files have no string metadata"));
            }
        }
        #[cfg(format_raw_sidecar)]
        FormatName::Raw => {}
        #[cfg(format_pytorch)]
        FormatName::Pytorch => {
            if incoming.is_empty() {
                return Err(msg("writing a pytorch checkpoint with no tensors is not supported"));
            }
            if !metadata.is_empty() {
                return Err(msg(
                    "a pytorch state_dict written here has no string metadata",
                ));
            }
        }
        #[cfg(format_onnx)]
        FormatName::Onnx => {
            if incoming.is_empty() {
                return Err(msg("writing an ONNX file with no initializers is not supported"));
            }
        }
        #[cfg(format_gguf)]
        FormatName::Gguf => unreachable!("gguf create is refused above"),
        _ => return Err(msg("that format is not compiled in")),
    }
    if dest.exists() && !cli.force {
        return Err(msg(format!(
            "{} exists; pass --force to overwrite",
            dest.display()
        )));
    }
    if cli.dry_run {
        outln!("dry-run: create {}", dest.display());
        print_plan(&incoming, &metadata);
        return Ok(());
    }
    ensure_parent(dest)?;
    #[allow(unreachable_patterns)]
    match format {
        #[cfg(format_safetensors)]
        FormatName::Safetensors => {
            let tensors = incoming.iter().map(as_out).collect::<Result<Vec<_>>>()?;
            safetensors::write_file(dest, &metadata, &tensors)?;
        }
        #[cfg(format_npy)]
        FormatName::Npy => write_npy(dest, &incoming[0])?,
        #[cfg(format_npz)]
        FormatName::Npz => {
            let members = incoming
                .iter()
                .map(npz_member)
                .collect::<Result<Vec<_>>>()?;
            npz::write_archive(dest, &members)?;
        }
        #[cfg(format_raw_sidecar)]
        FormatName::Raw => write_rawtensor(dest, &metadata, &incoming)?,
        #[cfg(format_pytorch)]
        FormatName::Pytorch => write_pytorch(dest, &incoming)?,
        #[cfg(format_onnx)]
        FormatName::Onnx => write_onnx(dest, &metadata, &incoming)?,
        #[cfg(format_gguf)]
        FormatName::Gguf => unreachable!("gguf create is refused above"),
        _ => return Err(msg("that format is not compiled in")),
    }
    Ok(())
}

fn modify(cli: &Cli, mode: Modify) -> Result<()> {
    let requested = cli.file.as_ref().unwrap();
    let resolved = archive::resolve_user_path(requested, cli.file_force_single)?;
    let source = &resolved.path;
    let dest = cli.output.clone().unwrap_or_else(|| source.clone());
    let archive = archive::open(
        source,
        cli.format,
        cli.max_header,
        cli.dtype.as_deref(),
        cli.shape.as_deref(),
    )?;
    #[cfg(format_safetensors)]
    if matches!(archive.inner, Inner::Index(_)) && !same_parent(source, &dest) {
        return Err(msg(
            "-o for a shard index must stay in the same directory as the index",
        ));
    }
    if mode == Modify::Update {
        report_changed_types(cli, &archive, &dest);
    }
    #[allow(unreachable_patterns)]
    match &archive.inner {
        #[cfg(format_gguf)]
        Inner::Gguf(_) => edit_gguf(cli, &archive, mode, &dest),
        #[cfg(format_safetensors)]
        Inner::Index(_) => edit_index(cli, &archive, mode, &dest),
        #[cfg(format_safetensors)]
        Inner::Safe => edit_safe(cli, &archive, mode, &dest),
        #[cfg(format_npy)]
        Inner::Npy => edit_single_file(cli, &archive, mode, &dest, false),
        #[cfg(format_npz)]
        Inner::Npz => edit_npz(cli, &archive, mode, &dest),
        #[cfg(format_raw_sidecar)]
        Inner::Raw(_) => edit_raw(cli, &archive, mode, &dest),
        #[cfg(format_pytorch)]
        Inner::Pytorch(file) => edit_pytorch(cli, &archive, mode, &dest, file),
        #[cfg(format_onnx)]
        Inner::Onnx => edit_onnx(cli, &archive, mode, &dest),
        _ => Err(msg("this format cannot be edited in this build")),
    }
}

/// An update writes the new tensor as it is, so a different dtype or shape
/// changes the file's tensor. Say so; it is often a widen such as BF16 to F32
/// from an .npy file.
fn report_changed_types(cli: &Cli, archive: &Archive, dest: &Path) {
    if cli.operands.is_empty() {
        return;
    }
    let Ok(incoming) = load_incoming(cli) else {
        return;
    };
    for item in &incoming {
        let Some(member) = archive.members.iter().find(|member| member.name == item.name) else {
            continue;
        };
        let old = dtype::canonical_id(&member.dtype).ok();
        if old != dtype::canonical_id(&item.dtype).ok() || old.is_none() && member.dtype != item.dtype {
            eprintln!(
                "tenspile: {}: {} changes dtype from {} to {}",
                dest.display(),
                item.name,
                dtype::display_name(&member.dtype),
                dtype::display_name(&item.dtype)
            );
        }
        let mut shape = member.shape.clone();
        #[cfg(format_gguf)]
        if matches!(archive.inner, Inner::Gguf(_)) {
            shape.reverse();
        }
        if shape != item.shape {
            eprintln!(
                "tenspile: {}: {} changes shape from {} to {}",
                dest.display(),
                item.name,
                format_shape(&shape),
                format_shape(&item.shape)
            );
        }
    }
}

#[cfg(format_gguf)]
fn edit_gguf(cli: &Cli, archive: &Archive, mode: Modify, dest: &Path) -> Result<()> {
    if mode == Modify::Delete || !cli.operands.is_empty() {
        return Err(msg(
            "GGUF tensors cannot be added, replaced, or deleted; a metadata update copies the tensor blob",
        ));
    }
    let updates = gguf_updates(cli)?;
    if updates.is_empty() {
        return Err(msg("GGUF update needs --metadata or --metadata-file"));
    }
    let Inner::Gguf(file) = &archive.inner else {
        return Err(msg("internal error: expected a GGUF archive"));
    };
    if cli.dry_run {
        outln!("dry-run: update metadata in {}", dest.display());
        for (key, value) in &updates {
            outln!("metadata {key}={}", value.display());
        }
        return Ok(());
    }
    refuse_overwrite(dest, Some(&archive.path), cli.force)?;
    ensure_parent(dest)?;
    gguf::rewrite_metadata(file, dest, &updates)
}

#[cfg(format_safetensors)]
fn edit_safe(cli: &Cli, archive: &Archive, mode: Modify, dest: &Path) -> Result<()> {
    let updates = string_updates(cli)?;
    let metadata = merge_strings(&archive.metadata, &updates);
    let planned = plan_members(cli, &archive.members, mode, &updates)?;
    if cli.dry_run {
        outln!("dry-run: write {}", dest.display());
        print_plan(&planned, &metadata);
        return Ok(());
    }
    refuse_overwrite(dest, Some(&archive.path), cli.force)?;
    ensure_parent(dest)?;
    let tensors = planned.iter().map(as_out).collect::<Result<Vec<_>>>()?;
    safetensors::write_file(dest, &metadata, &tensors)
}

#[cfg(format_npz)]
fn edit_npz(cli: &Cli, archive: &Archive, mode: Modify, dest: &Path) -> Result<()> {
    let updates = string_updates(cli)?;
    if !updates.is_empty() {
        return Err(msg("npz files have no string metadata"));
    }
    let planned = plan_members(cli, &archive.members, mode, &updates)?;
    if planned.is_empty() {
        return Err(msg("writing an npz archive with no arrays is not supported"));
    }
    if cli.dry_run {
        outln!("dry-run: write {}", dest.display());
        print_plan(&planned, &[]);
        return Ok(());
    }
    refuse_overwrite(dest, Some(&archive.path), cli.force)?;
    ensure_parent(dest)?;
    // Arrays the edit leaves alone, and members that are not arrays, are
    // copied as stored, header and compression included (NPZ-008, NPZ-009).
    let (entries, _) = npz::read_entries(&archive.path)?;
    let mut members = Vec::with_capacity(planned.len());
    for item in &planned {
        let original = archive.members.iter().find(|member| member.name == item.name);
        let entry = entries.iter().find(|entry| entry.key == item.name);
        match (original, entry) {
            (Some(member), Some(entry)) if same_payload(&member.payload, &item.payload) => members.push(NpzMember {
                key: item.name.clone(),
                body: NpzMemberBody::Verbatim {
                    path: archive.path.clone(),
                    entry: entry.zip_name.clone(),
                },
            }),
            _ => members.push(npz_member(item)?),
        }
    }
    for name in npz::blob_names(&archive.path)? {
        members.push(NpzMember {
            key: name.clone(),
            body: NpzMemberBody::Verbatim {
                path: archive.path.clone(),
                entry: name,
            },
        });
    }
    // A replaced array keeps its member's compression, and a new one
    // follows the archive: deflated when every member is (NPZ-009).
    let all_deflated = npz::all_deflated(&archive.path)?;
    npz::write_archive_with(dest, &members, &|key| {
        all_deflated || entries.iter().any(|entry| entry.key == key && entry.compressed)
    })
}

#[cfg(format_pytorch)]
fn write_pytorch(dest: &Path, incoming: &[Planned]) -> Result<()> {
    let tensors = incoming
        .iter()
        .map(|item| crate::formats::pytorch::OutTensor {
            name: item.name.clone(),
            dtype: item.dtype.clone(),
            shape: item.shape.clone(),
            payload: item.payload.clone(),
        })
        .collect::<Vec<_>>();
    crate::formats::pytorch::write_archive(dest, &tensors)
}

#[cfg(format_onnx)]
fn write_onnx(dest: &Path, metadata: &[(String, String)], incoming: &[Planned]) -> Result<()> {
    let tensors = incoming
        .iter()
        .map(|item| crate::formats::onnx::OutTensor {
            name: item.name.clone(),
            dtype: item.dtype.clone(),
            shape: item.shape.clone(),
            payload: item.payload.clone(),
        })
        .collect::<Vec<_>>();
    crate::formats::onnx::write_archive(dest, metadata, &tensors)
}

#[cfg(format_pytorch)]
fn edit_pytorch(
    cli: &Cli,
    archive: &Archive,
    mode: Modify,
    dest: &Path,
    file: &crate::formats::pytorch::TorchFile,
) -> Result<()> {
    let updates = string_updates(cli)?;
    if !updates.is_empty() {
        return Err(msg(
            "a pytorch checkpoint has no string metadata that tenspile writes",
        ));
    }
    if let Some(reason) = &file.refused {
        return Err(msg(format!(
            "{} holds {reason}, which tenspile cannot write back (PT-011); use tensconv to write a new state_dict on purpose",
            archive.path.display()
        )));
    }
    let planned = plan_members(cli, &archive.members, mode, &updates)?;
    if planned.is_empty() {
        return Err(msg("writing a pytorch checkpoint with no tensors is not supported"));
    }
    if cli.dry_run {
        outln!("dry-run: write {}", dest.display());
        print_plan(&planned, &[]);
        return Ok(());
    }
    refuse_overwrite(dest, Some(&archive.path), cli.force)?;
    ensure_parent(dest)?;
    // Tensors that still hold the stored values keep their storage, view,
    // and conj and neg bits; the pickled structure around them is kept.
    let unchanged = unchanged_names(archive, &planned);
    let tensors = planned
        .iter()
        .map(|item| crate::formats::pytorch::OutTensor {
            name: item.name.clone(),
            dtype: item.dtype.clone(),
            shape: item.shape.clone(),
            payload: item.payload.clone(),
        })
        .collect::<Vec<_>>();
    let notes = crate::formats::pytorch::rewrite_archive(file, &archive.path, dest, &tensors, &unchanged)?;
    for note in notes {
        eprintln!("tenspile: {}: {note}", dest.display());
    }
    Ok(())
}

/// Names whose planned tensor is the stored one: same dtype, shape, and
/// bytes source.
#[cfg(any(format_onnx, format_pytorch))]
fn unchanged_names(archive: &Archive, planned: &[Planned]) -> std::collections::BTreeSet<String> {
    planned
        .iter()
        .filter(|item| {
            archive.members.iter().any(|member| {
                member.name == item.name
                    && member.dtype == item.dtype
                    && member.shape == item.shape
                    && same_payload(&member.payload, &item.payload)
            })
        })
        .map(|item| item.name.clone())
        .collect()
}

#[cfg(format_onnx)]
fn edit_onnx(cli: &Cli, archive: &Archive, mode: Modify, dest: &Path) -> Result<()> {
    let updates = string_updates(cli)?;
    let metadata = merge_strings(&archive.metadata, &updates);
    let planned = plan_members(cli, &archive.members, mode, &updates)?;
    if planned.is_empty() {
        return Err(msg("writing an ONNX file with no initializers is not supported"));
    }
    if cli.dry_run {
        outln!("dry-run: write {}", dest.display());
        print_plan(&planned, &metadata);
        return Ok(());
    }
    refuse_overwrite(dest, Some(&archive.path), cli.force)?;
    ensure_parent(dest)?;
    // Initializers that still hold the stored bytes are copied as stored,
    // and the rest of the model (graph nodes, inputs, outputs) is kept.
    let unchanged = unchanged_names(archive, &planned);
    let tensors = planned
        .iter()
        .map(|item| crate::formats::onnx::OutTensor {
            name: item.name.clone(),
            dtype: item.dtype.clone(),
            shape: item.shape.clone(),
            payload: item.payload.clone(),
        })
        .collect::<Vec<_>>();
    let notes = crate::formats::onnx::rewrite_archive(&archive.path, dest, &metadata, &tensors, &unchanged)?;
    for note in notes {
        eprintln!("tenspile: {}: {note}", dest.display());
    }
    Ok(())
}

#[cfg(any(format_onnx, format_npz, format_raw_sidecar, format_pytorch))]
fn same_payload(left: &Payload, right: &Payload) -> bool {
    match (left, right) {
        (
            Payload::Range { path, offset, len },
            Payload::Range { path: other_path, offset: other_offset, len: other_len },
        ) => path == other_path && offset == other_offset && len == other_len,
        (Payload::Bytes(left), Payload::Bytes(right)) => left == right,
        (
            Payload::Zip { path, entry, skip, len },
            Payload::Zip { path: other_path, entry: other_entry, skip: other_skip, len: other_len },
        ) => path == other_path && entry == other_entry && skip == other_skip && len == other_len,
        _ => false,
    }
}

#[cfg(format_raw_sidecar)]
fn write_rawtensor(dest: &Path, metadata: &[(String, String)], incoming: &[Planned]) -> Result<()> {
    let typed = crate::formats::rawtensor::string_metadata(metadata)?;
    let tensors = incoming
        .iter()
        .map(|item| {
            crate::formats::rawtensor::to_out(
                &item.name,
                &item.dtype,
                &item.shape,
                &item.payload,
                item.fortran,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    crate::formats::rawtensor::write_archive(dest, &typed, &tensors)
}

#[cfg(format_raw_sidecar)]
fn edit_raw(cli: &Cli, archive: &Archive, mode: Modify, dest: &Path) -> Result<()> {
    let Inner::Raw(file) = &archive.inner else {
        return Err(msg("internal error: expected a rawtensor archive"));
    };
    let updates = string_updates(cli)?;
    let metadata = crate::formats::rawtensor::merge_metadata(&file.metadata, &updates)?;
    let planned = plan_members(cli, &archive.members, mode, &updates)?;
    if cli.dry_run {
        outln!("dry-run: write {}", dest.display());
        let shown = crate::formats::rawtensor::display_metadata(&metadata);
        print_plan(&planned, &shown);
        return Ok(());
    }
    refuse_overwrite(dest, Some(&archive.path), cli.force)?;
    let data_name = crate::formats::rawtensor::data_file_name(dest)?;
    let data_path = dest
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(data_name);
    if data_path != file_beside(&archive.path) {
        refuse_overwrite(&data_path, None, cli.force)?;
    }
    ensure_parent(dest)?;
    // A new or changed tensor gets a sha256 when every input tensor had one,
    // so the file stays fully checkable (RAW-012).
    let hashed = !file.tensors.is_empty()
        && file.tensors.iter().all(|record| record.extra.contains_key("sha256"));
    let mut replaced = Vec::new();
    let mut added = Vec::new();
    let tensors = planned
        .iter()
        .map(|item| {
            let mut out = crate::formats::rawtensor::to_out(
                &item.name,
                &item.dtype,
                &item.shape,
                &item.payload,
                item.fortran,
            )?;
            // A tensor the edit leaves alone keeps attrs, native, sha256,
            // and any other entry field (RAW-011).
            if let Some(record) = file.tensors.iter().find(|record| {
                record.name == item.name
                    && record.dtype == item.dtype
                    && record.shape == item.shape
                    && same_payload(&record.payload, &item.payload)
            }) {
                out.extra = record.extra.clone();
                return Ok(out);
            }
            if file.tensors.iter().any(|record| record.name == item.name) {
                replaced.push(item.name.as_str());
            } else {
                added.push(item.name.as_str());
            }
            if hashed {
                let digest = crate::sha256::hex(&item.payload.read_vec()?);
                out.extra.insert("sha256".into(), Value::String(digest));
            }
            Ok(out)
        })
        .collect::<Result<Vec<_>>>()?;
    let deleted: Vec<&str> = file
        .tensors
        .iter()
        .filter(|record| !planned.iter().any(|item| item.name == record.name))
        .map(|record| record.name.as_str())
        .collect();
    // `source` stays as it was; the edit is appended to `history` (RAW-010).
    let mut action = Vec::new();
    for (verb, names) in [("replaced", &replaced), ("added", &added), ("deleted", &deleted)] {
        if !names.is_empty() {
            action.push(format!("{verb} {}", names.join(", ")));
        }
    }
    if !updates.is_empty() {
        let keys: Vec<&str> = updates.iter().map(|(key, _)| key.as_str()).collect();
        action.push(format!("set metadata {}", keys.join(", ")));
    }
    let mut extra = file.extra.clone();
    let entry = serde_json::json!({
        "converter": concat!("tenspile ", env!("CARGO_PKG_VERSION")),
        "action": if action.is_empty() { "rewrote the manifest".to_string() } else { action.join("; ") },
    });
    match extra.get_mut("history") {
        Some(Value::Array(history)) => history.push(entry),
        Some(_) => {
            return Err(msg(format!(
                "{} history is not an array (RAW-010)",
                archive.path.display()
            )))
        }
        None => {
            extra.insert("history".into(), Value::Array(vec![entry]));
        }
    }
    crate::formats::rawtensor::write_archive_with(dest, &metadata, &extra, &tensors)
}

#[cfg(format_raw_sidecar)]
fn file_beside(manifest: &Path) -> PathBuf {
    crate::formats::rawtensor::data_file_name(manifest)
        .map(|name| {
            manifest
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(name)
        })
        .unwrap_or_else(|_| manifest.to_path_buf())
}

#[cfg(any(format_npy, format_raw_sidecar))]
fn edit_single_file(
    cli: &Cli,
    archive: &Archive,
    mode: Modify,
    dest: &Path,
    raw: bool,
) -> Result<()> {
    let updates = string_updates(cli)?;
    if !updates.is_empty() {
        return Err(msg(if raw {
            "raw files have no metadata"
        } else {
            "npy files have no string metadata"
        }));
    }
    let kind = if raw { "raw file" } else { "npy file" };
    match mode {
        Modify::Append => {
            return Err(msg(format!(
                "a {kind} holds one tensor; append into an npz or safetensors file"
            )));
        }
        Modify::Delete => {
            return Err(msg(format!(
                "a {kind} holds one tensor and cannot be emptied"
            )));
        }
        Modify::Update => {}
    }
    let incoming = load_incoming(cli)?;
    if incoming.len() != 1 {
        return Err(msg(format!("a {kind} holds one tensor")));
    }
    if cli.dry_run {
        outln!("dry-run: write {}", dest.display());
        print_plan(&incoming, &[]);
        return Ok(());
    }
    refuse_overwrite(dest, Some(&archive.path), cli.force)?;
    ensure_parent(dest)?;
    if raw {
        io_util::atomic_write(dest, |file| incoming[0].payload.copy_to(file))
    } else {
        write_npy(dest, &incoming[0])
    }
}

#[cfg(format_safetensors)]
fn edit_index(cli: &Cli, archive: &Archive, mode: Modify, dest: &Path) -> Result<()> {
    let Inner::Index(index) = &archive.inner else {
        return Err(msg("internal error: expected a shard index"));
    };
    let updates = string_updates(cli)?;
    let incoming = if mode == Modify::Delete {
        Vec::new()
    } else {
        load_incoming(cli)?
    };
    if mode != Modify::Delete && incoming.is_empty() && updates.is_empty() {
        return Err(msg("nothing to write; pass member files or --metadata"));
    }
    for item in &incoming {
        validate_safe(item)?;
    }
    let delete_names = if mode == Modify::Delete {
        operand_names(cli)?
    } else {
        Vec::new()
    };

    let mut map = index.weight_map.clone();
    let mut rewritten = BTreeSet::new();
    match mode {
        Modify::Delete => {
            for name in &delete_names {
                let position = map
                    .iter()
                    .position(|(existing, _)| existing == name)
                    .ok_or_else(|| msg(format!("no tensor {name}")))?;
                let shard = map[position].1.clone();
                map.remove(position);
                rewritten.insert(shard);
            }
        }
        Modify::Append => {
            if !incoming.is_empty() {
                let last = last_shard(&map)?;
                for item in &incoming {
                    if map.iter().any(|(name, _)| name == &item.name) {
                        return Err(msg(format!(
                            "tensor {} already exists; use -u to replace it",
                            item.name
                        )));
                    }
                    map.push((item.name.clone(), last.clone()));
                    rewritten.insert(last.clone());
                }
            }
        }
        Modify::Update => {
            for item in &incoming {
                if let Some(position) = map.iter().position(|(name, _)| name == &item.name) {
                    rewritten.insert(map[position].1.clone());
                } else {
                    let last = last_shard(&map)?;
                    map.push((item.name.clone(), last.clone()));
                    rewritten.insert(last);
                }
            }
        }
    }

    let live: BTreeSet<String> = map.iter().map(|(_, shard)| shard.clone()).collect();
    let mut dropped = Vec::new();
    for shard in shard_names(&index.weight_map) {
        if !live.contains(&shard) {
            dropped.push(shard.clone());
            rewritten.remove(&shard);
        }
    }

    if cli.dry_run {
        outln!("dry-run: update {}", dest.display());
        for shard in &rewritten {
            outln!("rewrite {shard}");
        }
        for shard in &dropped {
            outln!("drop {shard}");
        }
        for (key, value) in &updates {
            outln!("metadata {key}={value}");
        }
        return Ok(());
    }

    // The shards are shared by every index in the directory. Rewriting them
    // for a new index name would leave the original index listing tensors
    // its shards no longer hold, so only a metadata change may go to -o.
    if !rewritten.is_empty() && !same_path(dest, &archive.path) {
        return Err(msg(
            "-o with a tensor change on a shard index would rewrite the shards the original index lists; edit the index in place, or copy the directory first",
        ));
    }
    refuse_overwrite(dest, Some(&archive.path), cli.force)?;
    ensure_parent(dest)?;
    // Shards first, then the index, then unreferenced shard files. A crash can
    // leave a shard ahead of the index; the index is never pointed at a shard
    // that has not been written.
    for shard in &rewritten {
        let shard_path = relative_child(&index.directory, shard)?;
        let old = safetensors::read_file(&shard_path, cli.max_header)?;
        let tensors = shard_tensors(shard, &archive.members, &map, &incoming)?;
        safetensors::write_file(&shard_path, &old.metadata, &tensors)?;
    }
    let mut metadata = index.metadata.clone();
    for (key, value) in &updates {
        metadata.insert(key.clone(), Value::String(value.clone()));
    }
    let mut total = 0u64;
    for (name, _) in &map {
        let nbytes = nbytes_of(name, &archive.members, &incoming)?;
        total = total
            .checked_add(nbytes)
            .ok_or_else(|| msg("index total_size overflows"))?;
    }
    metadata.insert("total_size".into(), Value::from(total));
    write_index(dest, &metadata, &map)?;
    if same_path(dest, &archive.path) {
        for shard in &dropped {
            let path = relative_child(&index.directory, shard)?;
            fs::remove_file(&path).map_err(|err| io_err(&path, err))?;
        }
    }
    Ok(())
}

fn last_shard(map: &[(String, String)]) -> Result<String> {
    map.iter()
        .map(|(_, shard)| shard.clone())
        .max()
        .ok_or_else(|| msg("this index has no shards to receive a tensor"))
}

fn shard_names(map: &[(String, String)]) -> BTreeSet<String> {
    map.iter().map(|(_, shard)| shard.clone()).collect()
}

fn shard_tensors(
    shard: &str,
    members: &[Member],
    map: &[(String, String)],
    incoming: &[Planned],
) -> Result<Vec<OutTensor>> {
    let mut existing: Vec<&Member> = members
        .iter()
        .filter(|member| member.shard.as_deref() == Some(shard))
        .collect();
    existing.sort_by_key(|member| member.file_offset);
    let mut tensors = Vec::new();
    for member in existing {
        let kept = map
            .iter()
            .any(|(name, owner)| name == &member.name && owner == shard);
        if !kept {
            continue;
        }
        if let Some(item) = incoming.iter().find(|item| item.name == member.name) {
            tensors.push(as_out(item)?);
        } else {
            tensors.push(as_out(&from_member(member))?);
        }
    }
    for (name, owner) in map {
        if owner != shard || tensors.iter().any(|tensor| &tensor.name == name) {
            continue;
        }
        let item = incoming
            .iter()
            .find(|item| &item.name == name)
            .ok_or_else(|| msg(format!("new tensor {name} is missing")))?;
        tensors.push(as_out(item)?);
    }
    Ok(tensors)
}

fn nbytes_of(name: &str, members: &[Member], incoming: &[Planned]) -> Result<u64> {
    if let Some(item) = incoming.iter().find(|item| item.name == name) {
        return Ok(item.payload.len());
    }
    members
        .iter()
        .find(|member| member.name == name)
        .map(|member| member.nbytes)
        .ok_or_else(|| msg(format!("missing tensor {name}")))
}

fn write_index(
    path: &Path,
    metadata: &Map<String, Value>,
    weight_map: &[(String, String)],
) -> Result<()> {
    let mut weights = Map::new();
    for (name, shard) in weight_map {
        weights.insert(name.clone(), Value::String(shard.clone()));
    }
    let mut root = Map::new();
    root.insert("metadata".into(), Value::Object(metadata.clone()));
    root.insert("weight_map".into(), Value::Object(weights));
    let mut bytes = serde_json::to_vec_pretty(&Value::Object(root))?;
    bytes.push(b'\n');
    io_util::atomic_write(path, |file| {
        file.write_all(&bytes)?;
        Ok(())
    })
}

fn plan_members(
    cli: &Cli,
    members: &[Member],
    mode: Modify,
    updates: &[(String, String)],
) -> Result<Vec<Planned>> {
    match mode {
        Modify::Delete => assemble(members, &[], &operand_names(cli)?, Modify::Delete),
        Modify::Append | Modify::Update => {
            let incoming = load_incoming(cli)?;
            if incoming.is_empty() && updates.is_empty() {
                return Err(msg("nothing to write; pass member files or --metadata"));
            }
            assemble(members, &incoming, &[], mode)
        }
    }
}

fn assemble(
    members: &[Member],
    incoming: &[Planned],
    delete_names: &[String],
    mode: Modify,
) -> Result<Vec<Planned>> {
    let mut ordered: Vec<&Member> = members.iter().collect();
    ordered.sort_by_key(|member| (member.file_offset, member.name.clone()));
    match mode {
        Modify::Append => {
            for item in incoming {
                if ordered.iter().any(|member| member.name == item.name) {
                    return Err(msg(format!(
                        "tensor {} already exists; use -u to replace it",
                        item.name
                    )));
                }
            }
            let mut planned = ordered
                .iter()
                .map(|member| from_member(member))
                .collect::<Vec<_>>();
            planned.extend(incoming.iter().cloned());
            Ok(planned)
        }
        Modify::Update => {
            let mut planned = Vec::with_capacity(ordered.len() + incoming.len());
            for member in &ordered {
                if let Some(item) = incoming.iter().find(|item| item.name == member.name) {
                    planned.push(item.clone());
                } else {
                    planned.push(from_member(member));
                }
            }
            for item in incoming {
                if !ordered.iter().any(|member| member.name == item.name) {
                    planned.push(item.clone());
                }
            }
            Ok(planned)
        }
        Modify::Delete => {
            if delete_names.is_empty() {
                return Err(msg("name the members to delete"));
            }
            for name in delete_names {
                if !ordered.iter().any(|member| &member.name == name) {
                    return Err(msg(format!("no tensor {name}")));
                }
            }
            Ok(ordered
                .into_iter()
                .filter(|member| !delete_names.iter().any(|name| name == &member.name))
                .map(from_member)
                .collect())
        }
    }
}

fn from_member(member: &Member) -> Planned {
    Planned {
        name: member.name.clone(),
        dtype: member.dtype.clone(),
        shape: member.shape.clone(),
        payload: member.payload.clone(),
        npy_file: None,
        fortran: member.fortran,
    }
}

fn print_plan(planned: &[Planned], metadata: &[(String, String)]) {
    for item in planned {
        outln!(
            "{} {} {} {}",
            item.name,
            item.dtype,
            format_shape(&item.shape),
            item.payload.len()
        );
    }
    for (key, value) in metadata {
        outln!("metadata {key}={value}");
    }
}

fn open_source(cli: &Cli) -> Result<Archive> {
    let path = cli.file.as_ref().unwrap();
    if cli.op == Op::Extract {
        let resolved = archive::resolve_user_path(path, cli.file_force_single)?;
        return open_extract_source(&resolved.path, cli);
    }
    archive::open_user(
        path,
        cli.file_force_single,
        cli.format,
        cli.max_header,
        cli.dtype.as_deref(),
        cli.shape.as_deref(),
    )
}

fn open_extract_source(path: &Path, cli: &Cli) -> Result<Archive> {
    match detect::sniff(path, cli.max_header) {
        Ok(kind) => {
            detect::refuse_unsafe(kind, path)?;
            archive::open(path, None, cli.max_header, None, None)
        }
        Err(err) => Err(err),
    }
}

fn select<'a>(archive: &'a Archive, names: &[String]) -> Result<Vec<&'a Member>> {
    if names.is_empty() {
        return Ok(archive.members.iter().collect());
    }
    let mut chosen = Vec::with_capacity(names.len());
    for name in names {
        if name.contains('=') {
            return Err(msg(format!(
                "member selector {name} should be a name, not KEY=PATH"
            )));
        }
        let member = archive
            .members
            .iter()
            .find(|member| &member.name == name)
            .ok_or_else(|| msg(format!("no member {name}")))?;
        chosen.push(member);
    }
    Ok(chosen)
}

fn dest_format(cli: &Cli, dest: &Path) -> Result<FormatName> {
    if let Some(format) = cli.format {
        return Ok(format);
    }
    let name = dest
        .file_name()
        .and_then(|text| text.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if name.ends_with(".safetensors.index.json") {
        return Err(msg(
            "creating a shard index is not supported; edit an existing index instead",
        ));
    }
    #[cfg(format_safetensors)]
    if name.ends_with(".safetensors") {
        return Ok(FormatName::Safetensors);
    }
    #[cfg(format_gguf)]
    if name.ends_with(".gguf") {
        return Ok(FormatName::Gguf);
    }
    #[cfg(format_npy)]
    if name.ends_with(".npy") {
        return Ok(FormatName::Npy);
    }
    #[cfg(format_npz)]
    if name.ends_with(".npz") {
        return Ok(FormatName::Npz);
    }
    #[cfg(format_raw_sidecar)]
    if name.ends_with(".rawtensor.json") {
        return Ok(FormatName::Raw);
    }
    #[cfg(format_pytorch)]
    if name.ends_with(".pt") || name.ends_with(".pth") || name.ends_with(".ckpt") {
        return Ok(FormatName::Pytorch);
    }
    #[cfg(format_onnx)]
    if name.ends_with(".onnx") {
        return Ok(FormatName::Onnx);
    }
    Err(msg(format!(
        "cannot tell the format of {}; pass --format",
        dest.display()
    )))
}

struct Operand {
    key: Option<String>,
    source: String,
}

enum Source {
    File(PathBuf),
    Member { archive: PathBuf, member: String },
}

fn load_incoming(cli: &Cli) -> Result<Vec<Planned>> {
    let mut incoming = Vec::new();
    for (index, operand) in cli.operands.iter().enumerate() {
        let force_single = cli.operand_force.get(index).copied().unwrap_or(false);
        let spec = split_operand(operand)?;
        match resolve_source(&spec.source)? {
            Source::File(path) => incoming.extend(load_file(&path, spec.key.as_deref(), cli)?),
            Source::Member { archive, member } => {
                let key = spec.key.as_deref().ok_or_else(|| {
                    msg(format!(
                        "copying a member needs KEY={}:{}",
                        archive.display(),
                        member
                    ))
                })?;
                incoming.push(load_member(&archive, &member, key, cli, force_single)?);
            }
        }
    }
    let mut seen = BTreeSet::new();
    for item in &incoming {
        check_name(&item.name)?;
        if !seen.insert(item.name.clone()) {
            return Err(msg(format!("duplicate tensor {}", item.name)));
        }
    }
    Ok(incoming)
}

fn split_operand(text: &str) -> Result<Operand> {
    match text.split_once('=') {
        Some((key, source)) => {
            if key.is_empty() {
                return Err(msg("tensor key before '=' is empty"));
            }
            if source.is_empty() {
                return Err(msg(format!("tensor {key} is missing a source path")));
            }
            Ok(Operand {
                key: Some(key.to_string()),
                source: source.to_string(),
            })
        }
        None => Ok(Operand {
            key: None,
            source: text.to_string(),
        }),
    }
}

/// A value that names an existing file is that file. Otherwise a single colon
/// splits `archive:member` when the left side is a file.
fn resolve_source(text: &str) -> Result<Source> {
    let path = Path::new(text);
    if path.is_file() {
        return Ok(Source::File(path.to_path_buf()));
    }
    if let Some((archive, member)) = text.rsplit_once(':') {
        if !member.is_empty() && !archive.is_empty() && Path::new(archive).is_file() {
            return Ok(Source::Member {
                archive: PathBuf::from(archive),
                member: member.to_string(),
            });
        }
    }
    Err(msg(format!("cannot find {text}")))
}

fn load_file(path: &Path, key: Option<&str>, cli: &Cli) -> Result<Vec<Planned>> {
    match detect::sniff(path, cli.max_header) {
        Ok(kind) => {
            detect::refuse_unsafe(kind, path)?;
            match kind {
                Detected::Npy => Ok(vec![load_npy(path, key, cli)?]),
                Detected::Zip => {
                    if key.is_some() {
                        return Err(msg(format!(
                            "{} is an npz archive; pass it without a key, or copy one array with KEY={}:MEMBER",
                            path.display(),
                            path.display()
                        )));
                    }
                    load_npz_all(path, cli)
                }
                Detected::Safetensors
                | Detected::Gguf
                | Detected::Index
                | Detected::RawTensor
                | Detected::Pytorch
                | Detected::Onnx => {
                    // A file holding one tensor, such as one tenspile -x wrote
                    // as rawtensor, needs no :MEMBER.
                    let archive = archive::open(path, None, cli.max_header, None, None)?;
                    if let [only] = archive.members.as_slice() {
                        let name = key.unwrap_or(&only.name).to_string();
                        return Ok(vec![load_member(path, &only.name, &name, cli, true)?]);
                    }
                    Err(msg(format!(
                        "{} holds {} tensors; copy one with KEY={}:MEMBER",
                        path.display(),
                        archive.members.len(),
                        path.display()
                    )))
                }
                Detected::Pickle | Detected::LegacyGgml => unreachable!("refused above"),
            }
        }
        Err(err) => {
            if cli.dtype.is_some() || cli.shape.is_some() {
                Ok(vec![load_raw(path, key, cli)?])
            } else {
                Err(err)
            }
        }
    }
}

fn load_npy(path: &Path, key: Option<&str>, cli: &Cli) -> Result<Planned> {
    let info = npy::read_info(path)?;
    if info.fortran_order {
        return Err(msg(format!(
            "{} is fortran-order; tenspile does not transpose it",
            path.display()
        )));
    }
    let dtype = info.dtype.ok_or_else(|| {
        msg(format!(
            "{} descr {} has no safetensors dtype",
            path.display(),
            info.descr
        ))
    })?;
    let name = match key {
        Some(name) => name.to_string(),
        None => file_stem(path),
    };
    check_declared(cli, &name, dtype.name, &info.shape)?;
    Ok(Planned {
        name,
        dtype: dtype.name.to_string(),
        payload: info.canonical_payload(Payload::Range {
            path: path.to_path_buf(),
            offset: info.data_offset,
            len: info.nbytes,
        })?,
        // A big-endian file is not copied whole; its swapped bytes are.
        npy_file: (!info.big_endian).then(|| path.to_path_buf()),
        shape: info.shape,
        fortran: false,
    })
}

fn load_npz_all(path: &Path, cli: &Cli) -> Result<Vec<Planned>> {
    let (entries, notes) = npz::read_entries(path)?;
    for note in notes {
        eprintln!("tenspile: {}: {note}", path.display());
    }
    let mut planned = Vec::with_capacity(entries.len());
    for entry in entries {
        if entry.info.fortran_order {
            return Err(msg(format!(
                "{} member {} is fortran-order; tenspile does not transpose it",
                path.display(),
                entry.key
            )));
        }
        let dtype = entry.info.dtype.ok_or_else(|| {
            msg(format!(
                "{} member {} descr {} has no safetensors dtype",
                path.display(),
                entry.key,
                entry.info.descr
            ))
        })?;
        check_declared(cli, &entry.key, dtype.name, &entry.info.shape)?;
        planned.push(Planned {
            name: entry.key,
            dtype: dtype.name.to_string(),
            payload: entry.info.canonical_payload(Payload::Zip {
                path: path.to_path_buf(),
                entry: entry.zip_name.clone(),
                skip: entry.info.data_offset,
                len: entry.info.nbytes,
            })?,
            shape: entry.info.shape,
            npy_file: None,
            fortran: false,
        });
    }
    Ok(planned)
}

fn load_raw(path: &Path, key: Option<&str>, cli: &Cli) -> Result<Planned> {
    let dtype_name = cli
        .dtype
        .as_deref()
        .ok_or_else(|| msg(format!("{} needs --dtype", path.display())))?;
    let shape = cli
        .shape
        .as_deref()
        .ok_or_else(|| msg(format!("{} needs --shape", path.display())))?;
    let dtype = dtype::safe_dtype(dtype_name)?;
    let nbytes = dtype::packed_bytes(dtype, shape)?;
    let file_len = fs::metadata(path).map_err(|err| io_err(path, err))?.len();
    if file_len != nbytes {
        return Err(msg(format!(
            "{} is {file_len} bytes; dtype {} shape {} needs {nbytes}",
            path.display(),
            dtype.name,
            format_shape(shape)
        )));
    }
    Ok(Planned {
        name: key.map(str::to_string).unwrap_or_else(|| file_stem(path)),
        dtype: dtype.name.to_string(),
        shape: shape.to_vec(),
        payload: Payload::Range {
            path: path.to_path_buf(),
            offset: 0,
            len: nbytes,
        },
        npy_file: None,
        fortran: false,
    })
}

fn load_member(
    path: &Path,
    member_name: &str,
    key: &str,
    cli: &Cli,
    force_single: bool,
) -> Result<Planned> {
    let resolved = archive::resolve_user_path(path, force_single)?;
    let archive = archive::open(&resolved.path, None, cli.max_header, None, None)?;
    let member = archive
        .members
        .iter()
        .find(|member| member.name == member_name)
        .ok_or_else(|| msg(format!("{} has no member {member_name}", path.display())))?;
    if member.fortran {
        return Err(msg(format!(
            "{} member {member_name} is fortran-order; tenspile does not transpose it",
            path.display()
        )));
    }
    let mut shape = member.shape.clone();
    let mut payload = member.payload.clone();
    #[cfg(format_gguf)]
    if matches!(archive.inner, Inner::Gguf(_)) {
        shape.reverse();
        let logical = logical_len(&archive, member, Pack::Safe)
            .or_else(|_| logical_len(&archive, member, Pack::Raw))?;
        payload = clip(&payload, logical)?;
    }
    check_declared(cli, key, &member.dtype, &shape)?;
    Ok(Planned {
        name: key.to_string(),
        dtype: member.dtype.clone(),
        shape,
        payload,
        npy_file: None,
        fortran: member.fortran,
    })
}

fn check_declared(cli: &Cli, name: &str, dtype: &str, shape: &[u64]) -> Result<()> {
    if let Some(want) = &cli.dtype {
        let same = want == dtype
            || dtype::canonical_id(want).ok() == dtype::canonical_id(dtype).ok()
                && dtype::canonical_id(want).is_ok();
        if !same {
            return Err(msg(format!(
                "--dtype {want} does not match tensor {name} ({dtype})"
            )));
        }
    }
    if let Some(want) = &cli.shape {
        if want.as_slice() != shape {
            return Err(msg(format!(
                "--shape {} does not match tensor {name} ({})",
                format_shape(want),
                format_shape(shape)
            )));
        }
    }
    Ok(())
}

fn check_name(name: &str) -> Result<()> {
    if name.is_empty() || name.contains('\0') {
        return Err(msg(format!("invalid tensor name {name:?}")));
    }
    if name == safetensors::METADATA_KEY {
        return Err(msg("__metadata__ is reserved"));
    }
    Ok(())
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|text| text.to_str())
        .unwrap_or("tensor")
        .to_string()
}

fn operand_names(cli: &Cli) -> Result<Vec<String>> {
    if cli.operands.is_empty() {
        return Err(msg("name the members to delete"));
    }
    for name in &cli.operands {
        if name.contains('=') {
            return Err(msg(format!("delete takes member names, not {name}")));
        }
        check_name(name)?;
    }
    Ok(cli.operands.clone())
}

fn string_updates(cli: &Cli) -> Result<Vec<(String, String)>> {
    let mut pairs = Vec::new();
    if let Some(path) = &cli.metadata_file {
        let text = fs::read_to_string(path).map_err(|err| io_err(path, err))?;
        let value: Value = serde_json::from_str(&text)?;
        let object = value
            .as_object()
            .ok_or_else(|| msg("metadata file must be a JSON object"))?;
        for (key, entry) in object {
            let text = entry.as_str().ok_or_else(|| {
                msg(format!(
                    "metadata {key} is not a string; safetensors metadata is text"
                ))
            })?;
            upsert_string(&mut pairs, key, text);
        }
    }
    for (key, value) in &cli.metadata {
        upsert_string(&mut pairs, key, value);
    }
    Ok(pairs)
}

fn upsert_string(pairs: &mut Vec<(String, String)>, key: &str, value: &str) {
    if let Some((_, slot)) = pairs.iter_mut().find(|(existing, _)| existing == key) {
        *slot = value.to_string();
    } else {
        pairs.push((key.to_string(), value.to_string()));
    }
}

fn merge_strings(base: &[(String, String)], updates: &[(String, String)]) -> Vec<(String, String)> {
    let mut merged = base.to_vec();
    for (key, value) in updates {
        upsert_string(&mut merged, key, value);
    }
    merged
}

fn gguf_updates(cli: &Cli) -> Result<Vec<(String, MetaValue)>> {
    let mut pairs = Vec::new();
    if let Some(path) = &cli.metadata_file {
        let text = fs::read_to_string(path).map_err(|err| io_err(path, err))?;
        let value: Value = serde_json::from_str(&text)?;
        let object = value
            .as_object()
            .ok_or_else(|| msg("metadata file must be a JSON object"))?;
        for (key, entry) in object {
            let parsed = if let Some(text) = entry.as_str() {
                MetaValue::String(text.to_string())
            } else {
                let wrapped = serde_json::json!({ key: entry });
                gguf::parse_typed_metadata(&wrapped)?
                    .into_iter()
                    .next()
                    .map(|(_, value)| value)
                    .ok_or_else(|| msg(format!("metadata {key} is empty")))?
            };
            upsert_meta(&mut pairs, key, parsed);
        }
    }
    for (key, value) in &cli.metadata {
        upsert_meta(&mut pairs, key, MetaValue::String(value.clone()));
    }
    Ok(pairs)
}

fn upsert_meta(pairs: &mut Vec<(String, MetaValue)>, key: &str, value: MetaValue) {
    if let Some((_, slot)) = pairs.iter_mut().find(|(existing, _)| existing == key) {
        *slot = value;
    } else {
        pairs.push((key.to_string(), value));
    }
}

fn validate_safe(item: &Planned) -> Result<()> {
    let dtype = dtype::safe_dtype(&item.dtype)?;
    let expected = dtype::packed_bytes(dtype, &item.shape)?;
    if expected != item.payload.len() {
        return Err(msg(format!(
            "tensor {} is {} bytes and dtype {} needs {expected}",
            item.name,
            item.payload.len(),
            item.dtype
        )));
    }
    Ok(())
}

#[cfg(format_safetensors)]
fn as_out(item: &Planned) -> Result<OutTensor> {
    let dtype = dtype::safe_dtype(&item.dtype)?;
    let expected = dtype::packed_bytes(dtype, &item.shape)?;
    if expected != item.payload.len() {
        return Err(msg(format!(
            "tensor {} is {} bytes and dtype {} needs {expected}",
            item.name,
            item.payload.len(),
            dtype.name
        )));
    }
    Ok(OutTensor {
        name: item.name.clone(),
        dtype: dtype.name.to_string(),
        shape: item.shape.clone(),
        bytes: item.payload.clone(),
    })
}

fn numpy_descr(dtype_name: &str) -> Result<String> {
    let dtype = dtype::safe_dtype(dtype_name)?;
    dtype.numpy.map(str::to_string).ok_or_else(|| {
        msg(format!(
            "dtype {dtype_name} has no .npy encoding; use --format raw"
        ))
    })
}

#[cfg(format_npy)]
fn write_npy(path: &Path, item: &Planned) -> Result<()> {
    let descr = numpy_descr(&item.dtype)?;
    npy::write_file_ordered(path, &descr, &item.shape, item.fortran, |file| item.payload.copy_to(file))
}

#[cfg(format_npz)]
fn npz_member(item: &Planned) -> Result<NpzMember> {
    validate_safe(item)?;
    let body = if let Some(path) = &item.npy_file {
        NpzMemberBody::WholeFile(path.clone())
    } else if let Payload::Zip { path, entry, .. } = &item.payload {
        NpzMemberBody::WholeZip {
            path: path.clone(),
            entry: entry.clone(),
        }
    } else {
        NpzMemberBody::Rebuild {
            descr: numpy_descr(&item.dtype)?,
            shape: item.shape.clone(),
            payload: item.payload.clone(),
        }
    };
    Ok(NpzMember {
        key: item.name.clone(),
        body,
    })
}

/// Refuse to replace an existing path without --force. A rawtensor manifest
/// also writes its `.bin`, so that file is checked too. A dangling symlink
/// counts as existing.
fn refuse_overwrite(dest: &Path, source: Option<&Path>, force: bool) -> Result<()> {
    if source.is_some_and(|source| same_path(dest, source)) {
        return Ok(());
    }
    crate::io_util::refuse_existing(dest, force)
}

fn same_path(left: &Path, right: &Path) -> bool {
    if left == right {
        return true;
    }
    match (fs::canonicalize(left), fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn same_parent(left: &Path, right: &Path) -> bool {
    parent_of(left) == parent_of(right)
}

fn parent_of(path: &Path) -> PathBuf {
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() && !dir.exists() {
            fs::create_dir_all(dir).map_err(|err| io_err(dir, err))?;
        }
    }
    Ok(())
}

fn ends_with(path: &Path, suffix: &str) -> bool {
    path.file_name()
        .and_then(|text| text.to_str())
        .is_some_and(|name| name.to_ascii_lowercase().ends_with(suffix))
}

#[cfg(test)]
mod tests {
    use super::short_value;

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
