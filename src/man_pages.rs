#[test]
fn man_pages_list_every_long_option_from_help() {
    check(crate::cli::help_text(), include_str!("../man/tenspile.1"));
    check(crate::diff::diff_help(), include_str!("../man/tensdiff.1"));
    check(
        crate::convert::convert_help(),
        include_str!("../man/tensconv.1"),
    );
    check(crate::quant::quant_help(), include_str!("../man/tensquant.1"));
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
