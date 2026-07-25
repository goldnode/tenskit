#![forbid(unsafe_code)]

mod archive;
mod cli;
mod detect;
mod dtype;
mod error;
mod gguf;
mod io_util;
mod npy;
mod npz;
mod ops;
mod payload;
mod safetensors;

pub use error::{Error, Result};

pub fn run(args: &[String]) -> Result<()> {
    match cli::parse(args)? {
        cli::Parsed::Help => {
            println!("{}", cli::help_text());
            Ok(())
        }
        cli::Parsed::Version => {
            println!("{}", cli::version_text());
            Ok(())
        }
        cli::Parsed::Run(command) => ops::execute(command),
    }
}

/// Small GGUF v3 writer used by tests. Creating GGUF is not a CLI operation.
#[doc(hidden)]
pub fn write_gguf_fixture(
    path: &std::path::Path,
    name: &str,
    tensor_name: &str,
    f32_values: &[f32],
) -> Result<()> {
    gguf::write_fixture(path, name, tensor_name, f32_values)
}
