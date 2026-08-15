//! compressed-tensors pack names.
//!
//! `weight_packed` stays a packed integer tensor. ModelOpt nibble order is an
//! open question in the spec and is not invented here. Remove this file and
//! the `compressed-tensors` line in `local/supported-formats.txt` to drop the
//! format.

/// `weight_packed`, and the tensors stored beside it for the same module
/// (compressed-tensors.yaml `hierarchical_names`).
pub fn detail(name: &str, peers: &[&str]) -> Option<&'static str> {
    const COMPANIONS: &[&str] = &["weight_shape", "weight_scale", "weight_zero_point", "weight_global_scale"];
    let (module, suffix) = match name.rsplit_once('.') {
        Some((module, suffix)) => (Some(module), suffix),
        None => (None, name),
    };
    let packed = match module {
        Some(module) => format!("{module}.weight_packed"),
        None => "weight_packed".to_string(),
    };
    let is_pack = suffix == "weight_packed"
        || (COMPANIONS.contains(&suffix) && peers.iter().any(|peer| *peer == packed));
    is_pack.then_some("compressed-tensors pack; bytes stay packed")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_packed_is_the_pack() {
        let peers = ["ct.weight_packed", "ct.weight_scale", "ct.weight_shape", "other.weight_scale"];
        assert!(detail("ct.weight_packed", &peers).unwrap().contains("compressed-tensors"));
        assert!(detail("ct.weight_scale", &peers).is_some());
        assert!(detail("ct.weight_shape", &peers).is_some());
        assert!(detail("other.weight_scale", &peers).is_none());
        assert!(detail("ct.weight", &peers).is_none());
    }
}
