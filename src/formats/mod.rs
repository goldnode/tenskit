//! One module per format. The formats compiled in are the `+` lines of
//! `generated/formats.txt`. `generated/enabled_formats.rs` declares their
//! modules, and `build.rs` sets a `format_<id>` cfg for each.

include!("../../generated/enabled_formats.rs");

#[cfg(test)]
mod tests {
    #[test]
    fn the_formats_list_is_what_gets_compiled() {
        let text = include_str!("../../generated/formats.txt");
        let ids: Vec<&str> = text.lines().filter_map(|line| line.strip_prefix("+ ")).collect();
        assert_eq!(ids, super::ENABLED_IDS);
    }

    /// build.rs keeps its own list for the `format_<id>` cfgs; it must name
    /// the same formats, in the same order, with `-` written as `_`.
    #[test]
    fn build_rs_enables_the_same_formats() {
        let build = include_str!("../../build.rs");
        let start = build.find("const ENABLED_FORMATS").expect("build.rs has ENABLED_FORMATS");
        let list = &build[start..];
        let list = &list[list.find("= &[").unwrap() + 4..list.find("];").unwrap()];
        let cfgs: Vec<&str> = list.split(',').map(|item| item.trim().trim_matches('"')).filter(|item| !item.is_empty()).collect();
        let expected: Vec<String> = super::ENABLED_IDS.iter().map(|id| id.replace('-', "_")).collect();
        assert_eq!(cfgs, expected);
    }
}
