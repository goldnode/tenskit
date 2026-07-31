#![forbid(unsafe_code)]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match tenspile::diff_main(&args) {
        Ok(code) => std::process::exit(code),
        Err(err) => {
            eprintln!("tensdiff: {err}");
            std::process::exit(2);
        }
    }
}
