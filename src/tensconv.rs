#![forbid(unsafe_code)]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match tenspile::convert_main(&args) {
        Ok(code) => std::process::exit(code),
        Err(err) => {
            eprintln!("tensconv: {err}");
            std::process::exit(1);
        }
    }
}
