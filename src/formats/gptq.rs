//! GPTQ names inside a safetensors or pytorch container.
//!
//! The packed integers stay packed. Dequant is a separate conversion, and this
//! tool does not run it.

/// Note for a tensor that belongs to a GPTQ group (`qweight` plus `g_idx`).
pub fn detail(name: &str, peers: &[&str]) -> Option<&'static str> {
    let (prefix, suffix) = name.rsplit_once('.')?;
    if !matches!(suffix, "qweight" | "qzeros" | "scales" | "g_idx") {
        return None;
    }
    let has = |suffix: &str| {
        let needle = format!("{prefix}.{suffix}");
        peers.iter().any(|peer| *peer == needle)
    };
    if has("qweight") && has("g_idx") {
        Some("GPTQ pack; bytes stay packed")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_qweight_with_g_idx_is_gptq() {
        let peers = ["gptq.qweight", "gptq.qzeros", "gptq.scales", "gptq.g_idx"];
        assert!(detail("gptq.qweight", &peers).unwrap().contains("GPTQ"));
        assert!(detail("other.weight", &peers).is_none());
    }
}
