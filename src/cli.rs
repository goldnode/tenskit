use std::path::PathBuf;

use crate::error::{msg, Result};

pub const DEFAULT_MAX_HEADER: u64 = 100 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Create,
    List,
    Extract,
    Append,
    Update,
    Delete,
    Info,
    Diff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatName {
    #[cfg(format_safetensors)]
    Safetensors,
    #[cfg(format_gguf)]
    Gguf,
    #[cfg(format_npy)]
    Npy,
    #[cfg(format_npz)]
    Npz,
    #[cfg(format_raw_sidecar)]
    Raw,
    #[cfg(format_pytorch)]
    Pytorch,
    #[cfg(format_onnx)]
    Onnx,
}

impl FormatName {
    pub fn parse(text: &str) -> Result<Self> {
        match text {
            #[cfg(format_safetensors)]
            "safetensors" | "safetensor" => Ok(Self::Safetensors),
            #[cfg(format_gguf)]
            "gguf" => Ok(Self::Gguf),
            #[cfg(format_npy)]
            "npy" => Ok(Self::Npy),
            #[cfg(format_npz)]
            "npz" => Ok(Self::Npz),
            #[cfg(format_raw_sidecar)]
            "raw" | "rawtensor" => Ok(Self::Raw),
            #[cfg(format_pytorch)]
            "pytorch" | "pt" => Ok(Self::Pytorch),
            #[cfg(format_onnx)]
            "onnx" => Ok(Self::Onnx),
            _ => Err(msg(format!(
                "unknown format {text}; this build accepts {}",
                enabled_format_names()
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            #[cfg(format_safetensors)]
            Self::Safetensors => "safetensors",
            #[cfg(format_gguf)]
            Self::Gguf => "gguf",
            #[cfg(format_npy)]
            Self::Npy => "npy",
            #[cfg(format_npz)]
            Self::Npz => "npz",
            #[cfg(format_raw_sidecar)]
            Self::Raw => "raw",
            #[cfg(format_pytorch)]
            Self::Pytorch => "pytorch",
            #[cfg(format_onnx)]
            Self::Onnx => "onnx",
        }
    }
}

fn enabled_format_names() -> String {
    let mut names = Vec::new();
    #[cfg(format_safetensors)]
    names.push("safetensors");
    #[cfg(format_gguf)]
    names.push("gguf");
    #[cfg(format_npy)]
    names.push("npy");
    #[cfg(format_npz)]
    names.push("npz");
    #[cfg(format_raw_sidecar)]
    names.push("raw");
    #[cfg(format_pytorch)]
    names.push("pytorch");
    #[cfg(format_onnx)]
    names.push("onnx");
    if names.is_empty() {
        "no container formats".to_string()
    } else {
        names.join(", ")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cli {
    pub op: Op,
    pub file: Option<PathBuf>,
    pub verbose: bool,
    pub columns: bool,
    pub fold: bool,
    pub directory: PathBuf,
    pub output: Option<PathBuf>,
    pub format: Option<FormatName>,
    pub json: bool,
    pub match_names: bool,
    pub dtype: Option<String>,
    pub shape: Option<Vec<u64>>,
    pub metadata: Vec<(String, String)>,
    pub metadata_file: Option<PathBuf>,
    pub dry_run: bool,
    pub force: bool,
    pub max_header: u64,
    pub operands: Vec<String>,
}

pub fn parse(args: &[String]) -> Result<Parsed> {
    let mut op: Option<Op> = None;
    let mut file: Option<PathBuf> = None;
    let mut verbose = false;
    let mut columns = false;
    let mut fold = false;
    let mut directory = PathBuf::from(".");
    let mut output: Option<PathBuf> = None;
    let mut format: Option<FormatName> = None;
    let mut json = false;
    let mut match_names = false;
    let mut dtype: Option<String> = None;
    let mut shape: Option<String> = None;
    let mut metadata = Vec::new();
    let mut metadata_file: Option<PathBuf> = None;
    let mut dry_run = false;
    let mut force = false;
    let mut max_header = DEFAULT_MAX_HEADER;
    let mut operands = Vec::new();

    let mut index = 0;
    let mut ended = false;
    while index < args.len() {
        let arg = &args[index];
        if ended || arg == "-" || !arg.starts_with('-') {
            operands.push(arg.clone());
            index += 1;
            continue;
        }
        if arg == "--" {
            ended = true;
            index += 1;
            continue;
        }
        if arg == "--help" || arg == "-h" {
            return Ok(Parsed::Help);
        }
        if arg == "--version" {
            return Ok(Parsed::Version);
        }
        if let Some(rest) = arg.strip_prefix("--") {
            let (name, inline) = match rest.split_once('=') {
                Some((name, value)) => (name, Some(value.to_string())),
                None => (rest, None),
            };
            let mut take = |label: &str| -> Result<String> {
                if let Some(value) = inline.clone() {
                    Ok(value)
                } else {
                    index += 1;
                    args.get(index)
                        .cloned()
                        .ok_or_else(|| msg(format!("--{label} needs a value")))
                }
            };
            match name {
                "file" => file = Some(PathBuf::from(take("file")?)),
                "directory" => directory = PathBuf::from(take("directory")?),
                "output" => output = Some(PathBuf::from(take("output")?)),
                "format" => format = Some(FormatName::parse(&take("format")?)?),
                "dtype" => dtype = Some(take("dtype")?),
                "shape" => shape = Some(take("shape")?),
                "metadata" => {
                    let pair = take("metadata")?;
                    metadata.push(split_metadata(&pair)?);
                }
                "metadata-file" => metadata_file = Some(PathBuf::from(take("metadata-file")?)),
                "max-header-size" => {
                    max_header = parse_size(&take("max-header-size")?)?;
                }
                "json" => json = true,
                "match-names" => match_names = true,
                "verbose" => verbose = true,
                "columns" => columns = true,
                "fold" => fold = true,
                "dry-run" => dry_run = true,
                "force" => force = true,
                "delete" => set_op(&mut op, Op::Delete)?,
                "create" => set_op(&mut op, Op::Create)?,
                "list" => set_op(&mut op, Op::List)?,
                "extract" => set_op(&mut op, Op::Extract)?,
                "append" => set_op(&mut op, Op::Append)?,
                "update" => set_op(&mut op, Op::Update)?,
                "info" => set_op(&mut op, Op::Info)?,
                "diff" => set_op(&mut op, Op::Diff)?,
                _ => return Err(msg(format!("unknown option --{name}"))),
            }
            index += 1;
            continue;
        }

        let cluster = &arg[1..];
        if cluster.is_empty() {
            return Err(msg("missing short option"));
        }
        let chars: Vec<char> = cluster.chars().collect();
        let mut cursor = 0;
        while cursor < chars.len() {
            let ch = chars[cursor];
            let rest: String = chars[cursor + 1..].iter().collect();
            let needs_value = matches!(ch, 'f' | 'C' | 'o');
            if needs_value {
                let value = if rest.is_empty() {
                    index += 1;
                    args.get(index)
                        .cloned()
                        .ok_or_else(|| msg(format!("-{ch} needs a value")))?
                } else {
                    rest
                };
                match ch {
                    'f' => file = Some(PathBuf::from(value)),
                    'C' => directory = PathBuf::from(value),
                    'o' => output = Some(PathBuf::from(value)),
                    _ => unreachable!(),
                }
                break;
            }
            match ch {
                'c' => set_op(&mut op, Op::Create)?,
                't' => set_op(&mut op, Op::List)?,
                'x' => set_op(&mut op, Op::Extract)?,
                'r' => set_op(&mut op, Op::Append)?,
                'u' => set_op(&mut op, Op::Update)?,
                'i' => set_op(&mut op, Op::Info)?,
                'v' => verbose = true,
                'n' => dry_run = true,
                'd' => set_op(&mut op, Op::Diff)?,
                'h' => return Ok(Parsed::Help),
                _ => return Err(msg(format!("unknown option -{ch}"))),
            }
            cursor += 1;
        }
        index += 1;
    }

    let op =
        op.ok_or_else(|| msg("choose one operation: -c, -t, -x, -r, -u, -i, -d, or --delete"))?;
    if op == Op::Diff {
        let (left, right) = if file.is_none() && operands.len() == 2 {
            let right = operands.pop().unwrap();
            let left = operands.pop().unwrap();
            (PathBuf::from(left), right)
        } else if file.is_some() && operands.len() == 1 {
            (file.clone().unwrap(), operands[0].clone())
        } else {
            return Err(msg("diff needs two archive paths"));
        };
        file = Some(left);
        operands = vec![right];
    } else if file.is_none() {
        if operands.is_empty() {
            return Err(msg(
                "missing archive path; pass -f FILE or the path as the first argument",
            ));
        }
        file = Some(PathBuf::from(operands.remove(0)));
    }
    let shape = match shape {
        Some(text) => Some(parse_shape(&text)?),
        None => None,
    };
    Ok(Parsed::Run(Cli {
        op,
        file,
        verbose,
        columns,
        fold,
        directory,
        output,
        format,
        json,
        match_names,
        dtype,
        shape,
        metadata,
        metadata_file,
        dry_run,
        force,
        max_header,
        operands,
    }))
}

fn set_op(slot: &mut Option<Op>, op: Op) -> Result<()> {
    if let Some(existing) = *slot {
        if existing != op {
            return Err(msg(
                "pass only one of -c, -t, -x, -r, -u, -i, -d, or --delete",
            ));
        }
    }
    *slot = Some(op);
    Ok(())
}

fn split_metadata(pair: &str) -> Result<(String, String)> {
    let (key, value) = pair
        .split_once('=')
        .ok_or_else(|| msg("--metadata needs KEY=VALUE"))?;
    if key.is_empty() {
        return Err(msg("--metadata key is empty"));
    }
    Ok((key.to_string(), value.to_string()))
}

pub fn parse_shape(text: &str) -> Result<Vec<u64>> {
    let text = text.trim();
    if text.is_empty() || text == "[]" {
        return Ok(Vec::new());
    }
    let text = text.trim_start_matches('[').trim_end_matches(']').trim();
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let mut shape = Vec::new();
    for part in text.split([',', 'x', 'X']) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let dim: u64 = part
            .parse()
            .map_err(|_| msg(format!("invalid shape dimension {part}")))?;
        shape.push(dim);
    }
    Ok(shape)
}

fn parse_size(text: &str) -> Result<u64> {
    let text = text.trim();
    let (number, factor) = if let Some(rest) = text.strip_suffix(['K', 'k']) {
        (rest, 1024u64)
    } else if let Some(rest) = text.strip_suffix(['M', 'm']) {
        (rest, 1024 * 1024)
    } else if let Some(rest) = text.strip_suffix(['G', 'g']) {
        (rest, 1024 * 1024 * 1024)
    } else {
        (text, 1)
    };
    let value: u64 = number
        .trim()
        .parse()
        .map_err(|_| msg(format!("invalid size {text}")))?;
    value
        .checked_mul(factor)
        .ok_or_else(|| msg(format!("size {text} overflows")))
}

#[derive(Debug)]
pub enum Parsed {
    Help,
    Version,
    Run(Cli),
}

pub fn help_text() -> &'static str {
    "\
tenspile — inspect and edit tensor containers

Usage:
  tenspile -c [-f] ARCHIVE [KEY=FILE ...]
  tenspile -t [-v] [--columns] [--fold] [-f] ARCHIVE [KEY ...]
  tenspile -x [-f] ARCHIVE [KEY ...]
  tenspile -r [-f] ARCHIVE KEY=FILE ...
  tenspile -u [-f] ARCHIVE KEY=FILE ...
  tenspile --delete [-f] ARCHIVE KEY ...
  tenspile -i [-f] ARCHIVE
  tenspile -d ARCHIVE ARCHIVE

Operations (exactly one):
  -c, --create     Create an archive. Refuses to overwrite unless --force.
  -t, --list       List member names. -v adds dtype, shape, bytes, offset, and parameters.
  -x, --extract    Extract members. No names means all of them.
  -r, --append     Add members. Errors if a tensor key already exists.
  -u, --update     Replace a tensor key, or add it if it is missing.
  --delete         Remove tensor keys and write the archive again.
  -i, --info       Summarize the archive from its header.
  -d, --diff       Compare two archives. Exit 1 when they differ.
                   --match-names pairs tensors through the GGUF role table.

Options:
  -f, --file FILE           Archive path. If omitted, the first argument is the archive.
  -v, --verbose             Longer listing. Fields are separated by tabs.
  --columns                 Aligned listing for the console, with headings. Implies -v.
                            Bytes and parameter counts are scaled. Shard and
                            detail columns appear only when a member has them.
  --fold                    With --columns, merge a consecutive run of one
                            numeric name segment when the tensors match.
                            model.layers.[3-5] is one copy per layer. A
                            mismatch stays on its own line. Implies --columns.
  -C, --directory DIR       Directory for extracted files (default: .).
  -o, --output FILE         Write the new archive here instead of replacing -f.
  --format NAME             Container to read or write. This build's names are
                            printed when the name is unknown. Which formats are
                            compiled in is local/supported-formats.txt.
                            raw is rawtensor/1 (a .rawtensor.json manifest plus a .bin).
  --json                    Machine-readable listing or info.
  --match-names             On diff, pair tensors through the GGUF role
                            table. A GGUF general.architecture selects it.
                            Otherwise the smallest matching role set is used,
                            and a tie is printed as \"may be A, B, or C\".
  --dtype NAME              Dtype for a headerless input file (safetensors name or canonical id).
  --shape LIST              Shape for a headerless input, for example 2,2 or [2, 2].
  --metadata KEY=VALUE      Set a text metadata entry.
  --metadata-file FILE      JSON object of metadata. GGUF values are typed.
  --max-header-size SIZE    Safetensors header cap (default 100M).
  --force                   Overwrite an existing file on create or extract.
  -n, --dry-run             Print the plan and do not write.
  -h, --help                Show this help.
  --version                 Show the version.

Members:
  A bare name selects a member.
  KEY=PATH adds or replaces from a file. A bare .npy uses its file stem as the key.
  KEY=ARCHIVE:MEMBER copies one member out of another container.
  A bare .npz on create, append, or update imports every array in the archive.

Examples:
  tenspile -tvf model.safetensors
  tenspile -t --columns --fold -f model.safetensors
  tenspile -i model.gguf
  tenspile -xf model.safetensors -C out embed
  tenspile -cf out.safetensors embed=embed.npy --metadata format=pt
  tenspile -uf model.safetensors norm=new_norm.npy
  tenspile --delete -f model.safetensors optimizer.step
  tenspile -d left.safetensors right.gguf
  tenspile -d --match-names model.safetensors model.gguf
"
}

pub fn version_text() -> String {
    format!("tenspile {}", env!("CARGO_PKG_VERSION"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|item| (*item).to_string()).collect()
    }

    #[test]
    fn clustered_list_flags_take_the_archive() {
        let parsed = parse(&args(&["-tvf", "model.safetensors"])).unwrap();
        let Parsed::Run(cli) = parsed else {
            panic!("expected a command");
        };
        assert_eq!(cli.op, Op::List);
        assert!(cli.verbose);
        assert_eq!(cli.file.unwrap(), PathBuf::from("model.safetensors"));
    }

    #[test]
    fn glued_file_argument() {
        let parsed = parse(&args(&["-cfout.safetensors", "a.npy"])).unwrap();
        let Parsed::Run(cli) = parsed else {
            panic!("expected a command");
        };
        assert_eq!(cli.op, Op::Create);
        assert_eq!(cli.file.unwrap(), PathBuf::from("out.safetensors"));
        assert_eq!(cli.operands, vec!["a.npy".to_string()]);
    }

    #[test]
    fn two_operations_are_rejected() {
        let err = parse(&args(&["-ct", "file"])).unwrap_err();
        assert!(err.to_string().contains("only one"));
    }
}
