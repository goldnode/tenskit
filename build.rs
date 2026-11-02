use std::env;
use std::fs;
use std::path::PathBuf;

const ENABLED_FORMATS: &[&str] = &[
    "safetensors",
    "gguf",
    "npz",
    "pytorch",
    "onnx",
    "npy",
    "mlx",
    "gptq",
    "awq",
    "bitsandbytes",
    "compressed_tensors",
    "raw_sidecar",
];

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    for format in ENABLED_FORMATS {
        println!("cargo:rustc-check-cfg=cfg(format_{format})");
        println!("cargo:rustc-cfg=format_{format}");
    }

    install_man_pages(&manifest);
}

fn install_man_pages(manifest: &PathBuf) {
    let source = manifest.join("man");
    println!("cargo:rerun-if-changed={}", source.display());
    println!(
        "cargo:rerun-if-changed={}",
        manifest.join("Cargo.toml").display()
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let profile_dir = out_dir
        .parent()
        .and_then(|path| path.parent())
        .and_then(|path| path.parent())
        .unwrap_or_else(|| panic!("OUT_DIR is not inside target/<profile>/build"));
    let dest = profile_dir.join("share").join("man").join("man1");
    fs::create_dir_all(&dest)
        .unwrap_or_else(|err| panic!("create {}: {err}", dest.display()));

    let entries =
        fs::read_dir(&source).unwrap_or_else(|err| panic!("read {}: {err}", source.display()));
    let mut names = Vec::new();
    for entry in entries {
        let path = entry
            .unwrap_or_else(|err| panic!("read {}: {err}", source.display()))
            .path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("1") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path
            .file_name()
            .unwrap_or_else(|| panic!("{} has no file name", path.display()));
        let page = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        if !page.contains("@VERSION@") {
            panic!("{} has no @VERSION@ in its .TH footer", path.display());
        }
        let page = page.replace("@VERSION@", &env::var("CARGO_PKG_VERSION").unwrap());
        let target = dest.join(name);
        fs::write(&target, page.as_bytes())
            .unwrap_or_else(|err| panic!("write {}: {err}", target.display()));
        names.push(name.to_os_string());
    }
    if names.is_empty() {
        panic!("{} has no section 1 pages", source.display());
    }

    let entries =
        fs::read_dir(&dest).unwrap_or_else(|err| panic!("read {}: {err}", dest.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|err| panic!("read {}: {err}", dest.display()))
            .path();
        let name = path
            .file_name()
            .unwrap_or_else(|| panic!("{} has no file name", path.display()));
        if !names.iter().any(|kept| kept == name) {
            fs::remove_file(&path)
                .unwrap_or_else(|err| panic!("remove {}: {err}", path.display()));
        }
    }

    println!("cargo:rustc-env=TENSPILE_MAN_DIR={}", dest.display());
}
