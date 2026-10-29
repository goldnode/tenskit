//! tensinfo's console table: `-t` aligns the fields, and `--fold` collapses
//! consecutive identical layers into a range. tenspile's `-tv` stays
//! tab-separated and does not use this module.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::archive::Member;
use crate::dtype::format_shape;

struct Row {
    /// Earliest source member. Orders the table and supplies the stats.
    order: usize,
    /// Dotted parent. Empty when the name is a single segment.
    header: String,
    leaf: String,
    dtype: String,
    shape: String,
    bytes: String,
    /// `-` when the row stands for more than one offset.
    offset: String,
    params: String,
    shard: String,
    detail: String,
}

/// `--sort` order, segment by segment at each `.`. An all-digit segment sorts
/// by its number and before any other segment; other segments sort by bytes.
/// layers.2 sorts before layers.10, and a tie such as 2 and 02 falls back to bytes.
pub fn name_order(left: &str, right: &str) -> Ordering {
    let mut left_parts = left.split('.');
    let mut right_parts = right.split('.');
    loop {
        match (left_parts.next(), right_parts.next()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(a), Some(b)) => {
                let order = match (digits(a), digits(b)) {
                    (true, true) => {
                        let short_a = a.trim_start_matches('0');
                        let short_b = b.trim_start_matches('0');
                        short_a.len().cmp(&short_b.len()).then_with(|| short_a.cmp(short_b))
                    }
                    (true, false) => Ordering::Less,
                    (false, true) => Ordering::Greater,
                    (false, false) => Ordering::Equal,
                }
                .then_with(|| a.cmp(b));
                if order != Ordering::Equal {
                    return order;
                }
            }
        }
    }
}

fn digits(segment: &str) -> bool {
    !segment.is_empty() && segment.bytes().all(|byte| byte.is_ascii_digit())
}

/// Aligned listing. `fold` merges a consecutive run of one numeric name segment
/// when dtype, shape, size, shard, and detail all match. Bytes and params are
/// one copy, not the sum of the run.
pub fn render_columns(members: &[&Member], fold: bool) -> String {
    let rows = if fold {
        fold_rows(members)
    } else {
        members
            .iter()
            .enumerate()
            .map(|(order, member)| {
                row_from(order, &member.name, member, member.file_offset.to_string())
            })
            .collect()
    };
    layout(&rows)
}

fn row_from(order: usize, name: &str, member: &Member, offset: String) -> Row {
    let (header, leaf) = split_header(name);
    Row {
        order,
        header,
        leaf,
        dtype: crate::dtype::display_name(&member.dtype).to_string(),
        shape: format_shape(&member.shape),
        bytes: human_bytes(member.nbytes),
        offset,
        params: match member.numel {
            Some(count) => human_count(count),
            None => "-".to_string(),
        },
        shard: member.shard.clone().unwrap_or_default(),
        detail: member.detail.clone().unwrap_or_default(),
    }
}

/// Three or more segments keep the last two as the leaf (`q_proj.weight`).
/// Two segments keep the last one. A single segment is the whole line.
fn split_header(name: &str) -> (String, String) {
    let segs: Vec<&str> = name.split('.').collect();
    if segs.len() >= 3 {
        let leaf = segs[segs.len() - 2..].join(".");
        (segs[..segs.len() - 2].join("."), leaf)
    } else if segs.len() == 2 {
        (segs[0].to_string(), segs[1].to_string())
    } else {
        (String::new(), name.to_string())
    }
}

fn layout(rows: &[Row]) -> String {
    if rows.is_empty() {
        return String::new();
    }
    let mut ordered: Vec<&Row> = rows.iter().collect();
    ordered.sort_by_key(|row| row.order);
    let show_shard = ordered.iter().any(|row| !row.shard.is_empty());
    let show_detail = ordered.iter().any(|row| !row.detail.is_empty());
    let name_width = ordered
        .iter()
        .map(|row| {
            if row.header.is_empty() {
                row.leaf.chars().count()
            } else {
                2 + row.leaf.chars().count()
            }
        })
        .max()
        .unwrap_or(0)
        .max("name".len());
    let dtype_width = column_width(ordered.iter().map(|row| row.dtype.as_str()), "dtype");
    let shape_width = column_width(ordered.iter().map(|row| row.shape.as_str()), "shape");
    let bytes_width = column_width(ordered.iter().map(|row| row.bytes.as_str()), "bytes");
    let offset_width = column_width(ordered.iter().map(|row| row.offset.as_str()), "offset");
    let params_width = column_width(ordered.iter().map(|row| row.params.as_str()), "params");
    let shard_width = column_width(ordered.iter().map(|row| row.shard.as_str()), "shard");

    let mut lines = Vec::new();
    let mut head = vec![
        pad_end("name", name_width),
        pad_end("dtype", dtype_width),
        pad_end("shape", shape_width),
        pad_start("bytes", bytes_width),
        pad_start("offset", offset_width),
        pad_start("params", params_width),
    ];
    if show_shard {
        head.push(pad_end("shard", shard_width));
    }
    if show_detail {
        head.push("detail".to_string());
    }
    lines.push(head.join("  "));
    let mut previous_header: Option<&str> = None;
    for row in ordered {
        if previous_header != Some(row.header.as_str()) {
            if !row.header.is_empty() {
                lines.push(format!("{}.", row.header));
            }
            previous_header = Some(row.header.as_str());
        }
        let name = if row.header.is_empty() {
            row.leaf.clone()
        } else {
            format!("  {}", row.leaf)
        };
        let mut cells = vec![
            pad_end(&name, name_width),
            pad_end(&row.dtype, dtype_width),
            pad_end(&row.shape, shape_width),
            pad_start(&row.bytes, bytes_width),
            pad_start(&row.offset, offset_width),
            pad_start(&row.params, params_width),
        ];
        if show_shard {
            cells.push(pad_end(&row.shard, shard_width));
        }
        if show_detail {
            cells.push(row.detail.clone());
        }
        while cells.last().is_some_and(|cell| cell.trim().is_empty()) {
            cells.pop();
        }
        lines.push(cells.join("  "));
    }
    lines.join("\n")
}

fn width<'a>(values: impl Iterator<Item = &'a str>) -> usize {
    values.map(|value| value.chars().count()).max().unwrap_or(0)
}

fn column_width<'a>(values: impl Iterator<Item = &'a str>, label: &str) -> usize {
    width(values).max(label.chars().count())
}

fn pad_end(text: &str, width: usize) -> String {
    let extra = width.saturating_sub(text.chars().count());
    format!("{text}{}", " ".repeat(extra))
}

fn pad_start(text: &str, width: usize) -> String {
    let extra = width.saturating_sub(text.chars().count());
    format!("{}{text}", " ".repeat(extra))
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Sig {
    prefix: String,
    suffix: String,
    dtype: String,
    shape: Vec<u64>,
    nbytes: u64,
    numel: Option<u64>,
    shard: Option<String>,
    detail: Option<String>,
}

struct Piece {
    order: usize,
    index: u64,
    sig: Sig,
    member_index: usize,
}

fn fold_rows(members: &[&Member]) -> Vec<Row> {
    let Some(depth) = best_depth(members) else {
        return members
            .iter()
            .enumerate()
            .map(|(order, member)| {
                row_from(order, &member.name, member, member.file_offset.to_string())
            })
            .collect();
    };
    let mut folded_orders = vec![false; members.len()];
    let mut groups: BTreeMap<Sig, Vec<Piece>> = BTreeMap::new();
    for (order, member) in members.iter().enumerate() {
        let Some(piece) = piece_at(member, order, depth) else {
            continue;
        };
        groups.entry(piece.sig.clone()).or_default().push(piece);
    }
    let mut rows = Vec::new();
    for (sig, mut pieces) in groups {
        pieces.sort_by_key(|piece| (piece.index, piece.order));
        for run in runs(&pieces) {
            if run.len() < 2 {
                continue;
            }
            for piece in run {
                folded_orders[piece.order] = true;
            }
            let first = &members[run[0].member_index];
            let token = format!("[{}-{}]", run[0].index, run[run.len() - 1].index);
            let name = display_name(&sig.prefix, &token, &sig.suffix);
            rows.push(row_from(run[0].order, &name, first, "-".to_string()));
        }
    }
    for (order, member) in members.iter().enumerate() {
        if !folded_orders[order] {
            rows.push(row_from(
                order,
                &member.name,
                member,
                member.file_offset.to_string(),
            ));
        }
    }
    rows
}

fn best_depth(members: &[&Member]) -> Option<usize> {
    let max_len = members
        .iter()
        .map(|member| member.name.split('.').count())
        .max()?;
    let mut best: Option<(usize, usize)> = None;
    for depth in 0..max_len {
        let score = depth_score(members, depth);
        if score < 2 {
            continue;
        }
        let replace = match best {
            None => true,
            Some((best_score, best_depth)) => {
                score > best_score || (score == best_score && depth < best_depth)
            }
        };
        if replace {
            best = Some((score, depth));
        }
    }
    best.map(|(_, depth)| depth)
}

fn depth_score(members: &[&Member], depth: usize) -> usize {
    let mut groups: BTreeMap<Sig, Vec<u64>> = BTreeMap::new();
    for (order, member) in members.iter().enumerate() {
        let Some(piece) = piece_at(member, order, depth) else {
            continue;
        };
        groups.entry(piece.sig).or_default().push(piece.index);
    }
    let mut covered = 0;
    for mut indices in groups.into_values() {
        indices.sort_unstable();
        covered += run_cover(&indices);
    }
    covered
}

fn run_cover(indices: &[u64]) -> usize {
    if indices.is_empty() {
        return 0;
    }
    let mut covered = 0;
    let mut run = 1u64;
    for pair in indices.windows(2) {
        if pair[1] == pair[0].saturating_add(1) {
            run += 1;
        } else {
            if run >= 2 {
                covered += run as usize;
            }
            run = 1;
        }
    }
    if run >= 2 {
        covered += run as usize;
    }
    covered
}

fn piece_at(member: &Member, order: usize, depth: usize) -> Option<Piece> {
    let segs: Vec<&str> = member.name.split('.').collect();
    if segs.len() <= depth {
        return None;
    }
    let index = parse_index(segs[depth])?;
    let prefix = segs[..depth].join(".");
    let suffix = segs[depth + 1..].join(".");
    Some(Piece {
        order,
        index,
        member_index: order,
        sig: Sig {
            prefix,
            suffix,
            dtype: crate::dtype::display_name(&member.dtype).to_string(),
            shape: member.shape.clone(),
            nbytes: member.nbytes,
            numel: member.numel,
            shard: member.shard.clone(),
            detail: member.detail.clone(),
        },
    })
}

fn runs<'a>(pieces: &'a [Piece]) -> Vec<&'a [Piece]> {
    let mut out = Vec::new();
    let mut start = 0;
    for index in 1..=pieces.len() {
        let split = index == pieces.len()
            || pieces[index].index != pieces[index - 1].index.saturating_add(1);
        if split {
            out.push(&pieces[start..index]);
            start = index;
        }
    }
    out
}

fn display_name(prefix: &str, token: &str, suffix: &str) -> String {
    let mut parts = Vec::new();
    if !prefix.is_empty() {
        parts.push(prefix);
    }
    parts.push(token);
    if !suffix.is_empty() {
        parts.push(suffix);
    }
    parts.join(".")
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

/// 1024-based, so a checkpoint size is `64 MiB` rather than a raw count.
pub(crate) fn human_bytes(nbytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut scale = 1u128;
    let mut unit = 0;
    while unit + 1 < UNITS.len() && u128::from(nbytes) >= scale * 1024 {
        scale *= 1024;
        unit += 1;
    }
    if unit == 0 {
        return format!("{nbytes} B");
    }
    let tenths = (u128::from(nbytes) * 10 + scale / 2) / scale;
    let whole = tenths / 10;
    let frac = tenths % 10;
    if frac == 0 {
        format!("{whole} {}", UNITS[unit])
    } else {
        format!("{whole}.{frac} {}", UNITS[unit])
    }
}

/// Thousands are decimal: 16_777_216 parameters is `16.8M`.
pub(crate) fn human_count(count: u64) -> String {
    const STEPS: [(u64, &str); 4] = [
        (1_000_000_000_000, "T"),
        (1_000_000_000, "G"),
        (1_000_000, "M"),
        (1_000, "K"),
    ];
    for (div, suffix) in STEPS {
        if count >= div {
            let div = u128::from(div);
            let tenths = (u128::from(count) * 10 + div / 2) / div;
            let whole = tenths / 10;
            let frac = tenths % 10;
            if frac == 0 {
                return format!("{whole}{suffix}");
            }
            return format!("{whole}.{frac}{suffix}");
        }
    }
    count.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::Payload;

    fn member(name: &str, dtype: &str, shape: &[u64], nbytes: u64, offset: u64) -> Member {
        let numel = if shape.is_empty() {
            Some(1)
        } else {
            Some(
                shape
                    .iter()
                    .copied()
                    .fold(1u64, |acc, dim| acc.saturating_mul(dim)),
            )
        };
        Member {
            name: name.to_string(),
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
            file_offset: offset,
            nbytes,
            numel,
            payload: Payload::Range {
                path: std::path::PathBuf::from("x"),
                offset,
                len: nbytes,
            },
            shard: None,
            detail: None,
            fortran: false,
        }
    }

    #[test]
    fn bytes_and_params_use_the_listed_units() {
        assert_eq!(human_bytes(8), "8 B");
        assert_eq!(human_bytes(64 * 1024 * 1024), "64 MiB");
        assert_eq!(human_bytes(172 * 1024 * 1024), "172 MiB");
        assert_eq!(human_count(16_777_216), "16.8M");
        assert_eq!(human_count(4_194_304), "4.2M");
        assert_eq!(human_count(2), "2");
    }

    #[test]
    fn columns_group_a_shared_parent_and_keep_the_offset() {
        let q = member(
            "model.layers.0.self_attn.q_proj.weight",
            "F32",
            &[4096, 4096],
            64 * 1024 * 1024,
            104,
        );
        let k = member(
            "model.layers.0.self_attn.k_proj.weight",
            "F32",
            &[4096, 1024],
            16 * 1024 * 1024,
            200,
        );
        let rows = [&q, &k];
        let text = render_columns(&rows, false);
        assert_eq!(
            text,
            "\
name             dtype  shape          bytes  offset  params
model.layers.0.self_attn.
  q_proj.weight  F32    [4096, 4096]  64 MiB     104   16.8M
  k_proj.weight  F32    [4096, 1024]  16 MiB     200    4.2M"
        );
    }

    #[test]
    fn fold_collapses_a_consecutive_run_and_keeps_a_mismatch() {
        let mut owned = Vec::new();
        for layer in 3..=5 {
            owned.push(member(
                &format!("model.layers.{layer}.self_attn.q_proj.weight"),
                "F32",
                &[4096, 4096],
                64 * 1024 * 1024,
                layer * 10,
            ));
            owned.push(member(
                &format!("model.layers.{layer}.self_attn.k_proj.weight"),
                "F32",
                &[4096, 1024],
                16 * 1024 * 1024,
                layer * 10 + 1,
            ));
        }
        owned.push(member(
            "model.layers.4.self_attn.v_proj.weight",
            "F16",
            &[4096, 1024],
            8 * 1024 * 1024,
            99,
        ));
        let refs: Vec<&Member> = owned.iter().collect();
        let text = render_columns(&refs, true);
        assert!(text.starts_with("name"), "{text}");
        assert!(text.contains("model.layers.[3-5].self_attn."), "{text}");
        assert!(
            text.contains("  q_proj.weight  F32    [4096, 4096]  64 MiB       -   16.8M"),
            "{text}"
        );
        assert!(
            text.contains("  k_proj.weight  F32    [4096, 1024]  16 MiB       -    4.2M"),
            "{text}"
        );
        assert!(text.contains("model.layers.4.self_attn."), "{text}");
        assert!(
            text.contains("  v_proj.weight  F16    [4096, 1024]   8 MiB      99    4.2M"),
            "{text}"
        );
        assert!(!text.contains("model.layers.3.self_attn."));
    }

    #[test]
    fn fold_does_not_bridge_a_gap_or_a_second_index() {
        let a = member("model.layers.3.mlp.w", "F32", &[2], 8, 1);
        let b = member("model.layers.5.mlp.w", "F32", &[2], 8, 2);
        let refs = [&a, &b];
        let text = render_columns(&refs, true);
        assert!(!text.contains("layers.["));
        assert!(text.contains("model.layers.3."));
        assert!(text.contains("model.layers.5."));
        assert!(text.contains("mlp.w"));

        let mut experts = Vec::new();
        for layer in 0..2 {
            for expert in 0..2 {
                experts.push(member(
                    &format!("model.layers.{layer}.experts.{expert}.w"),
                    "F32",
                    &[2],
                    8,
                    experts.len() as u64,
                ));
            }
        }
        let refs: Vec<&Member> = experts.iter().collect();
        let text = render_columns(&refs, true);
        assert!(text.contains("model.layers.[0-1].experts."));
        assert!(!text.contains("experts.[0-1]"));
    }

    #[test]
    fn numeric_segments_sort_by_number() {
        let mut names = vec![
            "model.layers.10.mlp.weight",
            "model.norm.weight",
            "model.layers.2.mlp.weight",
            "model.layers.2.attn.weight",
            "model.embed.weight",
            "model.layers.02.mlp.weight",
            "model.layers.1a.weight",
            "model.layers.b.weight",
            "model",
        ];
        names.sort_by(|a, b| name_order(a, b));
        assert_eq!(
            names,
            [
                "model",
                "model.embed.weight",
                "model.layers.02.mlp.weight",
                "model.layers.2.attn.weight",
                "model.layers.2.mlp.weight",
                "model.layers.10.mlp.weight",
                "model.layers.1a.weight",
                "model.layers.b.weight",
                "model.norm.weight",
            ]
        );
    }
}
