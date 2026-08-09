#![forbid(unsafe_code)]

mod archive;
mod cli;
mod convert;
mod detect;
mod diff;
mod dtype;
mod error;
mod formats;
mod ggml_dequant;
mod gguf_names;
mod io_util;
mod listing;
mod ops;
mod payload;
mod quant;
mod sha256;

#[cfg(test)]
mod man_pages;

#[cfg(test)]
mod spec_audit;

pub use error::{Error, Result};

/// First line is the command and version. Second line is the author.
pub fn version_line(command: &str) -> String {
    format!(
        "{command} {}\nNick Reynolds (GitHub and X: goldnode)",
        env!("CARGO_PKG_VERSION")
    )
}

pub fn run(args: &[String]) -> Result<i32> {
    match cli::parse(args)? {
        cli::Parsed::Help => {
            println!("{}", cli::help_text());
            Ok(0)
        }
        cli::Parsed::Version => {
            println!("{}", cli::version_text());
            Ok(0)
        }
        cli::Parsed::Run(command) => ops::execute(command),
    }
}

pub fn diff_main(args: &[String]) -> Result<i32> {
    diff::diff_main(args)
}

pub fn convert_main(args: &[String]) -> Result<i32> {
    convert::convert_main(args)
}

pub fn quant_main(args: &[String]) -> Result<i32> {
    quant::quant_main(args)
}

/// Small GGUF v3 writer used by tests. Creating GGUF is not a CLI operation.
#[doc(hidden)]
#[cfg(format_gguf)]
pub fn write_gguf_fixture(
    path: &std::path::Path,
    name: &str,
    tensor_name: &str,
    f32_values: &[f32],
) -> Result<()> {
    formats::gguf::write_fixture(path, name, tensor_name, f32_values)
}
