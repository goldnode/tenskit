//! The round-trip check for the generated tables. `generated/spec-expected/`
//! holds the JSON each generated Rust table stands for. These tests rebuild
//! that JSON from the compiled statics and require it to be identical, so a
//! table that lost, reordered, or changed a value fails.

use serde_json::Value;

/// The expected JSON for `file`.
pub(crate) fn load(file: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("generated/spec-expected").join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

/// The first path where two values differ, for a readable failure.
pub(crate) fn first_difference(left: &Value, right: &Value, at: &str) -> Option<String> {
    match (left, right) {
        (Value::Object(a), Value::Object(b)) => {
            let ka: Vec<&String> = a.keys().collect();
            let kb: Vec<&String> = b.keys().collect();
            if ka != kb {
                return Some(format!("{at}: keys {ka:?} vs {kb:?}"));
            }
            a.iter().find_map(|(key, value)| first_difference(value, &b[key], &format!("{at}.{key}")))
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!("{at}: {} items vs {}", a.len(), b.len()));
            }
            a.iter().zip(b).enumerate().find_map(|(i, (x, y))| first_difference(x, y, &format!("{at}[{i}]")))
        }
        // serde_json compares numbers by kind and value: 1, -1, and 1.0 differ.
        _ if left == right => None,
        _ => Some(format!("{at}: {left} vs {right}")),
    }
}

/// Fail with the first difference between the rebuilt and expected JSON.
pub(crate) fn assert_same(rebuilt: &Value, expected: &Value, file: &str) {
    if let Some(diff) = first_difference(rebuilt, expected, file) {
        panic!("the generated table differs from the spec at {diff}");
    }
}
