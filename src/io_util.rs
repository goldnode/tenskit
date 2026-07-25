use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::error::{io_err, msg, Result};

pub fn copy_range(path: &Path, offset: u64, len: u64, out: &mut impl Write) -> Result<()> {
    let mut file = File::open(path).map_err(|err| io_err(path, err))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|err| io_err(path, err))?;
    let copied = io::copy(&mut file.take(len), out)?;
    if copied != len {
        return Err(msg(format!(
            "{}: expected {len} bytes at offset {offset}, read {copied}",
            path.display()
        )));
    }
    Ok(())
}

#[allow(dead_code)]
pub fn read_range(path: &Path, offset: u64, len: u64) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    copy_range(path, offset, len, &mut buf)?;
    Ok(buf)
}

pub fn atomic_write(dest: &Path, body: impl FnOnce(&mut File) -> Result<()>) -> Result<()> {
    with_temp_file(dest, |tmp| {
        let mut file = File::create(tmp).map_err(|err| io_err(tmp, err))?;
        body(&mut file)?;
        file.sync_all().map_err(|err| io_err(tmp, err))?;
        Ok(())
    })
}

pub fn with_temp_file(dest: &Path, body: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    let parent = dest.parent().filter(|dir| !dir.as_os_str().is_empty());
    if let Some(dir) = parent {
        if !dir.is_dir() {
            return Err(msg(format!("directory {} does not exist", dir.display())));
        }
    }
    let tmp = tmp_path(dest);
    let result = body(&tmp);
    if let Err(err) = result {
        let _ = fs::remove_file(&tmp);
        return Err(err);
    }
    fs::rename(&tmp, dest).map_err(|err| io_err(dest, err))?;
    Ok(())
}

fn tmp_path(dest: &Path) -> PathBuf {
    let name = dest
        .file_name()
        .and_then(|text| text.to_str())
        .unwrap_or("archive");
    let tmp_name = format!(
        ".{name}.tenspile-tmp-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    );
    match dest.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(tmp_name),
        _ => PathBuf::from(tmp_name),
    }
}

/// Reject absolute paths and `..` so an index cannot point outside its directory.
pub fn relative_child(base: &Path, name: &str) -> Result<PathBuf> {
    let path = Path::new(name);
    if path.is_absolute() {
        return Err(msg(format!("path {name} is absolute")));
    }
    if path
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(msg(format!("path {name} leaves its directory")));
    }
    Ok(base.join(path))
}

pub fn safe_member_path(dir: &Path, key: &str, extension: &str) -> Result<PathBuf> {
    if key.is_empty() || key.contains('\0') {
        return Err(msg(format!("invalid member name {key:?}")));
    }
    let relative = Path::new(key);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(msg(format!(
            "member name {key} is not a safe relative path"
        )));
    }
    let mut path = dir.join(relative);
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext == extension => {}
        _ => {
            path.set_extension(extension);
        }
    }
    // `set_extension` replaces an existing extension. Keys often contain dots
    // (`model.layers.0.weight`) and must keep them.
    if path.file_name().and_then(|name| name.to_str()) != Some(&format!("{key}.{extension}"))
        && !key.ends_with(&format!(".{extension}"))
    {
        path = dir.join(format!("{key}.{extension}"));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn member_paths_keep_dots_in_the_key() {
        let path = safe_member_path(Path::new("out"), "model.layers.0.weight", "npy").unwrap();
        assert_eq!(path, PathBuf::from("out/model.layers.0.weight.npy"));
    }

    #[test]
    fn parent_segments_are_rejected() {
        assert!(relative_child(Path::new("."), "../secret").is_err());
    }
}
