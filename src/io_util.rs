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
        let mut file = create_new(tmp)?;
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
    let tmp = temp_beside(dest);
    let result = body(&tmp).and_then(|()| {
        // An in-place rewrite keeps the original file's permissions.
        if let Ok(meta) = fs::metadata(dest) {
            fs::set_permissions(&tmp, meta.permissions()).map_err(|err| io_err(&tmp, err))?;
        }
        fs::rename(&tmp, dest).map_err(|err| io_err(dest, err))
    });
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Open a temp file that must not exist yet, so a planted file or symlink at
/// that name is not followed or truncated.
pub fn create_new(path: &Path) -> Result<File> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| io_err(path, err))
}

/// Refuse to replace `dest` without `force`. For a rawtensor manifest the
/// data file it writes beside itself is checked as well. A dangling symlink
/// counts as existing.
pub fn refuse_existing(dest: &Path, force: bool) -> Result<()> {
    if force {
        return Ok(());
    }
    let mut paths = vec![dest.to_path_buf()];
    #[cfg(format_raw_sidecar)]
    if let Ok(data) = crate::formats::rawtensor::data_path_for(dest) {
        paths.push(data);
    }
    for path in paths {
        if fs::symlink_metadata(&path).is_ok() {
            return Err(msg(format!(
                "{} exists; pass --force to overwrite",
                path.display()
            )));
        }
    }
    Ok(())
}

/// Refuse a writer's tensor list when two tensors share a name.
pub fn check_unique_names<'a>(names: impl Iterator<Item = &'a str>) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(crate::error::msg(format!("tensor name {name} appears twice")));
        }
    }
    Ok(())
}

pub fn temp_beside(dest: &Path) -> PathBuf {
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
