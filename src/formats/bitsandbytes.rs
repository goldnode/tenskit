//! bitsandbytes 4-bit quant state stored as tensors.
//!
//! The packed codes stay packed. Remove this file and the `bitsandbytes` line
//! in `local/supported-formats.txt` to drop the format.

pub fn detail(name: &str, peers: &[&str]) -> Option<&'static str> {
    if name.contains("bitsandbytes__") {
        return Some("bitsandbytes pack; bytes stay packed");
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
    }
}
