//! One module per format. `build.rs` includes only the ids listed in
//! `local/supported-formats.txt` (recorded in `generated/formats.txt` for a
//! clone without it), so deleting a line there drops that format from the
//! build. Deleting the matching file under `src/formats/` then
//! removes it from the source tree as well.

include!("../../generated/enabled_formats.rs");

#[cfg(test)]
mod tests {
    #[test]
    fn the_formats_list_is_what_gets_compiled() {
        let text = include_str!("../../generated/formats.txt");
        let ids: Vec<&str> = text.lines().filter_map(|line| line.strip_prefix("+ ")).collect();
        assert_eq!(ids, super::ENABLED_IDS);
        // local/supported-formats.txt is not in every clone; when it is, the
        // generated list must follow it.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("local/supported-formats.txt");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let mut ids = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.ends_with(':') {
                continue;
            }
            ids.push(line);
        }
        assert_eq!(ids, super::ENABLED_IDS);
    }
}
