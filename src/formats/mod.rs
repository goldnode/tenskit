//! One module per format. `build.rs` includes only the ids listed in
//! `local/supported-formats.txt`, so deleting a line there drops that format
//! from the build. Deleting the matching file under `src/formats/` then
//! removes it from the source tree as well.

include!(concat!(env!("OUT_DIR"), "/enabled_formats.rs"));

#[cfg(test)]
mod tests {
    #[test]
    fn the_supported_formats_file_is_what_gets_compiled() {
        let text = include_str!("../../local/supported-formats.txt");
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
