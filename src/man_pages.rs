#[test]
fn the_build_copies_pages_into_man1() {
    let dir = std::path::Path::new(env!("TENSPILE_MAN_DIR"));
    for name in ["tenspile.1", "tensdiff.1", "tensconv.1", "tensquant.1", "tensinfo.1"] {
        let page = dir.join(name);
        let text = std::fs::read_to_string(&page).unwrap_or_else(|err| panic!("{}: {err}", page.display()));
        let footer = concat!("\"tenskit ", env!("CARGO_PKG_VERSION"), "\"");
        assert!(text.lines().next().unwrap().contains(footer), "{} lacks {footer}", page.display());
    }
}

#[test]
fn man_pages_list_every_long_option_from_help() {
    check(crate::cli::help_text(), include_str!("../man/tenspile.1"));
    check(crate::diff::diff_help(), include_str!("../man/tensdiff.1"));
    check(
        crate::convert::convert_help(),
        include_str!("../man/tensconv.1"),
    );
    check(crate::quant::quant_help(), include_str!("../man/tensquant.1"));
    check(crate::info::info_help(), include_str!("../man/tensinfo.1"));
}

fn check(help: &str, page: &str) {
    let flat: String = page.chars().filter(|ch| *ch != '\\').collect();
    for option in long_options(help) {
        assert!(
            flat.contains(&option),
            "{option} is in the help and missing from the man page"
        );
    }
}

fn long_options(help: &str) -> Vec<String> {
    let bytes = help.as_bytes();
    let mut found = Vec::new();
    let mut index = 0;
    while index + 2 < bytes.len() {
        if bytes[index] == b'-' && bytes[index + 1] == b'-' && bytes[index + 2].is_ascii_alphanumeric()
        {
            let start = index;
            index += 2;
            while index < bytes.len() && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'-')
            {
                index += 1;
            }
            let option = help[start..index].to_string();
            if !found.contains(&option) {
                found.push(option);
            }
        } else {
            index += 1;
        }
    }
    found
}
