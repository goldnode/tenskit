//! A small regular-expression matcher for compressed-tensors `re:` targets
//! (CT-008), with Python `re.match` semantics: anchored at the start, not at
//! the end. It covers literals, `.`, escapes (`\d`, `\w`, `\s`, and escaped
//! punctuation), bracket classes, `*`, `+`, `?`, and `^`/`$`. Groups,
//! alternation, and counted repeats are refused rather than guessed at.

use crate::error::{msg, Result};

#[derive(Debug)]
enum Atom {
    Any,
    Char(char),
    Class { ranges: Vec<(char, char)>, negated: bool },
    End,
}

#[derive(Debug)]
struct Piece {
    atom: Atom,
    min: usize,
    max: usize,
}

fn class_escape(ch: char) -> Option<Vec<(char, char)>> {
    Some(match ch {
        'd' => vec![('0', '9')],
        'w' => vec![('a', 'z'), ('A', 'Z'), ('0', '9'), ('_', '_')],
        's' => vec![(' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r')],
        _ => return None,
    })
}

fn parse(pattern: &str) -> Result<Vec<Piece>> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut pieces = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        let ch = chars[index];
        index += 1;
        let atom = match ch {
            '.' => Atom::Any,
            '^' if pieces.is_empty() => continue,
            '$' => Atom::End,
            '\\' => {
                let next = *chars
                    .get(index)
                    .ok_or_else(|| msg(format!("pattern {pattern} ends with a backslash")))?;
                index += 1;
                match class_escape(next) {
                    Some(ranges) => Atom::Class { ranges, negated: false },
                    None if next.is_ascii_alphanumeric() => {
                        return Err(msg(format!("pattern escape \\{next} is not supported")))
                    }
                    None => Atom::Char(next),
                }
            }
            '[' => {
                let negated = chars.get(index) == Some(&'^');
                if negated {
                    index += 1;
                }
                let mut ranges = Vec::new();
                let mut first = true;
                loop {
                    let ch = *chars
                        .get(index)
                        .ok_or_else(|| msg(format!("pattern {pattern} has an open bracket")))?;
                    index += 1;
                    if ch == ']' && !first {
                        break;
                    }
                    first = false;
                    let low = if ch == '\\' {
                        let next = *chars.get(index).ok_or_else(|| msg("bad escape in pattern"))?;
                        index += 1;
                        if let Some(more) = class_escape(next) {
                            ranges.extend(more);
                            continue;
                        }
                        next
                    } else {
                        ch
                    };
                    if chars.get(index) == Some(&'-') && chars.get(index + 1).is_some_and(|c| *c != ']') {
                        let high = chars[index + 1];
                        index += 2;
                        ranges.push((low, high));
                    } else {
                        ranges.push((low, low));
                    }
                }
                Atom::Class { ranges, negated }
            }
            '(' | ')' | '|' | '{' | '}' => {
                return Err(msg(format!(
                    "pattern {pattern} uses {ch}, which this matcher does not support"
                )))
            }
            '*' | '+' | '?' => {
                return Err(msg(format!("pattern {pattern} has a repeat with nothing before it")))
            }
            other => Atom::Char(other),
        };
        let (min, max) = match chars.get(index) {
            Some('*') => (0, usize::MAX),
            Some('+') => (1, usize::MAX),
            Some('?') => (0, 1),
            _ => (1, 1),
        };
        if (min, max) != (1, 1) {
            index += 1;
            if chars.get(index) == Some(&'?') {
                // Lazy repeats match the same strings under re.match.
                index += 1;
            }
        }
        pieces.push(Piece { atom, min, max });
    }
    Ok(pieces)
}

fn atom_matches(atom: &Atom, ch: char) -> bool {
    match atom {
        Atom::Any => ch != '\n',
        Atom::Char(want) => *want == ch,
        Atom::Class { ranges, negated } => {
            ranges.iter().any(|(low, high)| (*low..=*high).contains(&ch)) != *negated
        }
        Atom::End => false,
    }
}

fn match_here(pieces: &[Piece], text: &[char]) -> bool {
    let Some((piece, rest)) = pieces.split_first() else {
        return true;
    };
    if let Atom::End = piece.atom {
        return text.is_empty() && match_here(rest, text);
    }
    let mut count = 0;
    while count < piece.max && count < text.len() && atom_matches(&piece.atom, text[count]) {
        count += 1;
    }
    // Greedy, then back off.
    loop {
        if count >= piece.min && match_here(rest, &text[count..]) {
            return true;
        }
        if count == 0 || count <= piece.min {
            return false;
        }
        count -= 1;
    }
}

/// Python `re.match(pattern, text) is not None`.
pub fn matches_prefix(pattern: &str, text: &str) -> Result<bool> {
    let pieces = parse(pattern)?;
    let chars: Vec<char> = text.chars().collect();
    Ok(match_here(&pieces, &chars))
}

#[cfg(test)]
mod tests {
    use super::matches_prefix;

    #[test]
    fn match_is_anchored_at_the_start_only() {
        assert!(matches_prefix(".*lm_head", "model.lm_head").unwrap());
        assert!(matches_prefix("model", "model.layers.0").unwrap());
        assert!(!matches_prefix("layers", "model.layers.0").unwrap());
        assert!(matches_prefix(r"model\.layers\.\d+\.mlp\..*", "model.layers.12.mlp.up_proj").unwrap());
        assert!(!matches_prefix(r"model\.layers\.\d+$", "model.layers.12.mlp").unwrap());
        assert!(matches_prefix("[^.]+_proj", "q_proj").unwrap());
        assert!(matches_prefix("(a|b)", "a").is_err());
    }
}
