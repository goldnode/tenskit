#![forbid(unsafe_code)]

/// `outln!` for command output. A closed pipe, as with `| head`, ends the
/// process quietly with status 141, as a shell reports a command that SIGPIPE stopped.
macro_rules! outln {
    ($($arg:tt)*) => {
        $crate::io_util::out_line(format_args!($($arg)*))
    };
}

mod archive;
mod cli;
mod convert;
mod detect;
mod diff;
mod dtype;
mod error;
mod formats;
mod ggml_dequant;
mod gguf_keys;
mod gguf_names;
mod info;
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

#[cfg(test)]
mod spec_expected;

pub use error::{Error, Result};

/// First line is the command and version. Second line is the author.
pub fn version_line(command: &str) -> String {
    format!(
        "{command} {} (tenskit.org)\nNick Reynolds [goldnode] (nicks.net)",
        env!("CARGO_PKG_VERSION")
    )
}

pub fn run(args: &[String]) -> Result<i32> {
    match cli::parse(args)? {
        cli::Parsed::Help => {
            outln!("{}", cli::help_text());
            Ok(0)
        }
        cli::Parsed::Version => {
            outln!("{}", cli::version_text());
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

pub fn info_main(args: &[String]) -> Result<i32> {
    info::info_main(args)
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
