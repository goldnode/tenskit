//! AWQ GEMM names inside a container.
//!
//! Bytes stay packed. A group that also has `g_idx` is GPTQ, not this pack.

pub fn detail(name: &str, peers: &[&str]) -> Option<&'static str> {
    let (prefix, suffix) = name.rsplit_once('.')?;
    if !matches!(suffix, "qweight" | "qzeros" | "scales") {
        return None;
    }
    let has = |suffix: &str| {
        let needle = format!("{prefix}.{suffix}");
        peers.iter().any(|peer| *peer == needle)
    };
    if has("g_idx") {
        return None;
    }
    if has("qweight") && has("qzeros") && has("scales") {
        Some("AWQ GEMM pack; bytes stay packed")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_names_are_awq_until_g_idx_appears() {
        let peers = ["awq.qweight", "awq.qzeros", "awq.scales"];
        assert!(detail("awq.qweight", &peers).unwrap().contains("AWQ"));
        let gptq = ["gptq.qweight", "gptq.qzeros", "gptq.scales", "gptq.g_idx"];
        assert!(detail("gptq.qweight", &gptq).is_none());
    }
}
