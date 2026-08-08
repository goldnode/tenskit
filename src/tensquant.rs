#![forbid(unsafe_code)]

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match tenspile::quant_main(&args) {
        Ok(code) => std::process::exit(code),
        Err(err) => {
            eprintln!("tensquant: {err}");
            std::process::exit(1);
        }
    }
}
