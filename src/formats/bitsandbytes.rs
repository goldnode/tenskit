//! bitsandbytes 4-bit quant state stored as tensors.
//!
//! The packed codes stay packed.

pub fn detail(name: &str, peers: &[&str]) -> Option<&'static str> {
    if name.contains("bitsandbytes__") {
        return Some("bitsandbytes pack; bytes stay packed");
    }
    // BNB-005: `<W>` and `<W>.<quant_state key>` beside
    // `<W>.quant_state.bitsandbytes__nf4` (or `__fp4`).
    let has_state = |weight: &str| {
        peers.iter().any(|peer| {
            peer.strip_prefix(weight)
                .and_then(|rest| rest.strip_prefix(".quant_state.bitsandbytes__"))
                .is_some()
        })
    };
    if has_state(name) {
        return Some("bitsandbytes 4-bit pack; bytes stay packed");
    }
    if let Some((weight, key)) = name.rsplit_once('.') {
        let state_key = matches!(key, "absmax" | "quant_map" | "nested_absmax" | "nested_quant_map");
        if state_key && has_state(weight) {
            return Some("bitsandbytes 4-bit pack; bytes stay packed");
        }
    }
    let (prefix, suffix) = name.rsplit_once('.')?;
    if !matches!(suffix, "weight" | "absmax" | "quant_state") {
        return None;
    }
    let has = |suffix: &str| {
        let needle = format!("{prefix}.{suffix}");
        peers.iter().any(|peer| *peer == needle)
    };
    if has("weight") && has("absmax") {
        Some("bitsandbytes 4-bit pack; bytes stay packed")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_and_absmax_mark_the_pack() {
        let peers = ["bnb.weight", "bnb.absmax"];
        assert!(detail("bnb.weight", &peers).unwrap().contains("bitsandbytes"));
        assert!(detail("plain.weight", &["plain.weight"]).is_none());
        // The Hugging Face layout names the state after the weight.
        let hf = ["l.weight", "l.weight.absmax", "l.weight.quant_map", "l.weight.quant_state.bitsandbytes__nf4", "l.bias"];
        for name in &hf[..4] {
            assert!(detail(name, &hf).is_some(), "{name}");
        }
        assert!(detail("l.bias", &hf).is_none());
    }
}
