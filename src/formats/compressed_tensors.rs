//! compressed-tensors pack names.
//!
//! `weight_packed` stays a packed integer tensor. ModelOpt nibble order is an
//! open question in the spec and is not invented here. Remove this file and
//! the `compressed-tensors` line in `local/supported-formats.txt` to drop the
//! format.

pub fn detail(name: &str) -> Option<&'static str> {
    if name.rsplit_once('.').is_some_and(|(_, suffix)| suffix == "weight_packed")
        || name == "weight_packed"
    {
        Some("compressed-tensors pack; bytes stay packed")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_packed_is_the_pack() {
        assert!(detail("ct.weight_packed").unwrap().contains("compressed-tensors"));
        assert!(detail("ct.weight").is_none());
    }
}
