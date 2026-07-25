#![forbid(unsafe_code)]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(err) = tenspile::run(&args) {
        eprintln!("tenspile: {err}");
        std::process::exit(1);
    }
}
