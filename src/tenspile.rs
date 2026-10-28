#![forbid(unsafe_code)]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match tenskit::run(&args) {
        Ok(0) => {}
        Ok(code) => std::process::exit(code),
        Err(err) => {
            eprintln!("tenspile: {err}");
            std::process::exit(1);
        }
    }
}
