#![forbid(unsafe_code)]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match tenspile::run(&args) {
        Ok(0) => {}
        Ok(code) => std::process::exit(code),
        Err(err) => {
            eprintln!("tenspile: {err}");
            std::process::exit(if diff_args(&args) { 2 } else { 1 });
        }
    }
}

fn diff_args(args: &[String]) -> bool {
    args.iter().any(|arg| {
        arg == "--diff"
            || arg == "-d"
            || (arg.starts_with('-')
                && !arg.starts_with("--")
                && arg.chars().skip(1).any(|ch| ch == 'd'))
    })
}
