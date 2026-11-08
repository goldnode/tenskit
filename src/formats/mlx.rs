//! MLX does not define its own container. It stores tensors in npy, npz, or
//! safetensors, and its quant files are ordinary tensors plus a config.
//!
//! `<V2` in npy is a void descr. The npy spec says that is ambiguous, so the
//! dtype is left unchanged and `-i` records that MLX would call it bfloat16.

pub fn void_note(descr: &str) -> Option<&'static str> {
    if descr.contains('V') {
        Some("MLX would read a <V2 descr as bfloat16; the numpy descr is unchanged")
    } else {
        None
    }
}

/// Quantized MLX weights are a uint32 `weight` plus `scales` and maybe `biases`.
/// Group size is not in the tensor file.
pub fn quant_detail(name: &str, peers: &[&str]) -> Option<&'static str> {
    let (prefix, suffix) = name.rsplit_once('.')?;
    if !matches!(suffix, "weight" | "scales" | "biases") {
        return None;
    }
    let has = |suffix: &str| {
        let needle = format!("{prefix}.{suffix}");
        peers.iter().any(|peer| *peer == needle)
    };
    if has("weight") && has("scales") {
        Some("MLX quant names; group size is not stored in the tensor file")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn void_descrs_are_noted_and_not_renamed() {
        assert!(void_note("<V2").unwrap().contains("bfloat16"));
        assert!(void_note("<f4").is_none());
    }
}
