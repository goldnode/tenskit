//! Compare the readers with the fixtures in tensdataspecs_readonly.

use std::fs;
use std::path::PathBuf;

use serde_json::Value;

use crate::archive::{self, Inner};
use crate::cli::FormatName;
use crate::dtype::{self, canonical_id};
use crate::sha256;

fn vectors() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tensdataspecs_readonly/specs/test-vectors")
}

fn expected() -> Value {
    let path = vectors().join("expected.json");
    serde_json::from_slice(&fs::read(&path).unwrap()).unwrap()
}

fn tensor(doc: &Value, name: &str) -> (String, Vec<u64>, String) {
    let row = &doc["tensors"][name];
    let dtype = row["dtype"].as_str().unwrap().to_string();
    let shape = row["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|dim| dim.as_u64().unwrap())
        .collect();
    let digest = row["sha256"].as_str().unwrap().to_string();
    (dtype, shape, digest)
}

fn hash_member(member: &crate::archive::Member) -> String {
    sha256::hex(&member.payload.read_vec().unwrap())
}

#[test]
fn safetensors_fixture_matches_canonical_bytes() {
    let doc = expected();
    let archive = archive::open(
        &vectors().join("small.safetensors"),
        None,
        crate::cli::DEFAULT_MAX_HEADER,
        None,
        None,
    )
    .unwrap();
    assert_eq!(archive.format, FormatName::Safetensors);
    for name in ["w_f32", "b_f16", "e_bf16", "i_i32", "s_f32"] {
        let (dtype, shape, digest) = tensor(&doc, name);
        let member = archive
            .members
            .iter()
            .find(|member| member.name == name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(canonical_id(&member.dtype).unwrap(), dtype);
        assert_eq!(member.shape, shape);
        assert_eq!(hash_member(member), digest);
    }
}

#[test]
fn npy_fixture_matches_canonical_bytes() {
    let doc = expected();
    let archive = archive::open(
        &vectors().join("w_f32.npy"),
        None,
        crate::cli::DEFAULT_MAX_HEADER,
        None,
        None,
    )
    .unwrap();
    let (dtype, shape, digest) = tensor(&doc, "w_f32");
    let member = &archive.members[0];
    assert_eq!(member.name, "w_f32");
    assert_eq!(canonical_id(&member.dtype).unwrap(), dtype);
    assert_eq!(member.shape, shape);
    assert_eq!(hash_member(member), digest);
}

#[test]
fn npz_fixture_matches_canonical_bytes() {
    let doc = expected();
    let archive = archive::open(
        &vectors().join("small.npz"),
        None,
        crate::cli::DEFAULT_MAX_HEADER,
        None,
        None,
    )
    .unwrap();
    for name in ["w_f32", "b_f16", "i_i32", "s_f32"] {
        let (dtype, shape, digest) = tensor(&doc, name);
        let member = archive
            .members
            .iter()
            .find(|member| member.name == name)
            .unwrap();
        assert_eq!(canonical_id(&member.dtype).unwrap(), dtype, "{name}");
        assert_eq!(member.shape, shape, "{name}");
        assert_eq!(hash_member(member), digest, "{name}");
    }
    // npy.yaml: bfloat16 is stored as void `<V2` and is not a canonical dtype by itself.
    let bf16 = archive
        .members
        .iter()
        .find(|member| member.name == "e_bf16")
        .unwrap();
    assert!(
        bf16.dtype.contains('V'),
        "expected a void descr, got {}",
        bf16.dtype
    );
    let (_, _, digest) = tensor(&doc, "e_bf16");
    assert_eq!(hash_member(bf16), digest);
}

#[test]
fn gguf_fixture_matches_canonical_bytes() {
    let doc = expected();
    let archive = archive::open(
        &vectors().join("small.gguf"),
        None,
        crate::cli::DEFAULT_MAX_HEADER,
        None,
        None,
    )
    .unwrap();
    assert!(archive.notes.iter().any(|note| note.contains("ne[]")));
    assert!(archive
        .members
        .iter()
        .all(|member| member.name != "s_f32"));
    let Inner::Gguf(file) = &archive.inner else {
        panic!("expected gguf");
    };
    for name in ["w_f32", "b_f16", "e_bf16", "i_i32"] {
        let (dtype, shape, digest) = tensor(&doc, name);
        let member = archive
            .members
            .iter()
            .find(|member| member.name == name)
            .unwrap();
        let mut canonical_shape = member.shape.clone();
        canonical_shape.reverse();
        assert_eq!(canonical_shape, shape, "{name} file shape {:?}", member.shape);
        assert_eq!(canonical_id(&member.dtype).unwrap(), dtype);
        let tensor = file
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap();
        let logical = dtype::ggml_payload_bytes(tensor.ggml_type, &tensor.shape).unwrap();
        let bytes = crate::io_util::read_range(&archive.path, member.file_offset, logical).unwrap();
        assert_eq!(sha256::hex(&bytes), digest, "{name}");
    }
    let quant = file
        .tensors
        .iter()
        .find(|tensor| tensor.name == "w_q8_0")
        .unwrap();
    let logical = dtype::ggml_payload_bytes(quant.ggml_type, &quant.shape).unwrap();
    let bytes = crate::io_util::read_range(&archive.path, quant.file_offset, logical).unwrap();
    let blocks = doc["tensors"]["w_q8_0 (gguf, dequantized to float32)"]["raw_blocks_sha256"]
        .as_str()
        .unwrap();
    assert_eq!(sha256::hex(&bytes), blocks);
}

#[test]
fn rawtensor_fixture_matches_canonical_bytes() {
    let doc = expected();
    let archive = archive::open(
        &vectors().join("small.rawtensor.json"),
        None,
        crate::cli::DEFAULT_MAX_HEADER,
        None,
        None,
    )
    .unwrap();
    assert_eq!(archive.version, "1");
    for name in ["w_f32", "b_f16", "e_bf16", "i_i32", "s_f32"] {
        let (dtype, shape, digest) = tensor(&doc, name);
        let member = archive
            .members
            .iter()
            .find(|member| member.name == name)
            .unwrap();
        assert_eq!(member.dtype, dtype);
        assert_eq!(member.shape, shape);
        assert_eq!(hash_member(member), digest);
    }
    let listed = crate::run(&[
        "-t".into(),
        "--json".into(),
        "-f".into(),
        vectors().join("small.safetensors").display().to_string(),
    ]);
    assert!(listed.is_ok());
}

#[test]
fn edge_fixtures_follow_the_spec() {
    let edge = vectors().join("edge");
    let unpadded = archive::open(
        &edge.join("safetensors-unpadded.bin"),
        None,
        crate::cli::DEFAULT_MAX_HEADER,
        None,
        None,
    )
    .unwrap();
    assert!(unpadded.notes.iter().any(|note| note.contains("ST-016")));
    assert_eq!(unpadded.members[0].name, "w");
    assert_eq!(unpadded.members[0].shape, vec![1]);

    let mispad = archive::open(
        &edge.join("safetensors-v030-mispad.bin"),
        None,
        crate::cli::DEFAULT_MAX_HEADER,
        None,
        None,
    )
    .unwrap();
    assert!(mispad.notes.iter().any(|note| note.contains("ST-016")));

    let npy = archive::open(
        &edge.join("npy-align16.npy"),
        None,
        crate::cli::DEFAULT_MAX_HEADER,
        None,
        None,
    )
    .unwrap();
    assert!(npy.notes.iter().any(|note| note.contains("NPY-005")));

    let big = crate::run(&[
        "-i".into(),
        "-f".into(),
        edge.join("gguf-v3-be.gguf").display().to_string(),
    ]);
    assert!(big.unwrap_err().to_string().contains("big-endian"));
    let v1 = crate::run(&[
        "-i".into(),
        "-f".into(),
        edge.join("gguf-v1-header.gguf").display().to_string(),
    ]);
    assert!(v1.unwrap_err().to_string().contains("version 1"));
}

#[cfg(format_pytorch)]
#[test]
fn pytorch_fixture_matches_canonical_bytes() {
    let doc = expected();
    let archive = archive::open(
        &vectors().join("small.pt"),
        None,
        crate::cli::DEFAULT_MAX_HEADER,
        None,
        None,
    )
    .unwrap();
    assert_eq!(archive.format, FormatName::Pytorch);
    for name in ["w_f32", "b_f16", "e_bf16", "i_i32", "s_f32"] {
        let (dtype, shape, digest) = tensor(&doc, name);
        let member = archive
            .members
            .iter()
            .find(|member| member.name == name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(canonical_id(&member.dtype).unwrap(), dtype, "{name}");
        assert_eq!(member.shape, shape, "{name}");
        assert_eq!(hash_member(member), digest, "{name}");
    }
}

#[cfg(format_onnx)]
#[test]
fn onnx_fixture_matches_canonical_bytes() {
    let doc = expected();
    let archive = archive::open(
        &vectors().join("small.onnx"),
        None,
        crate::cli::DEFAULT_MAX_HEADER,
        None,
        None,
    )
    .unwrap();
    assert_eq!(archive.format, FormatName::Onnx);
    for name in ["w_f32", "b_f16", "e_bf16", "i_i32", "s_f32"] {
        let (dtype, shape, digest) = tensor(&doc, name);
        let member = archive
            .members
            .iter()
            .find(|member| member.name == name)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert_eq!(canonical_id(&member.dtype).unwrap(), dtype, "{name}");
        assert_eq!(member.shape, shape, "{name}");
        assert_eq!(hash_member(member), digest, "{name}");
    }
}

#[test]
fn quant_pack_safetensors_match_canonical_bytes() {
    let doc = expected();
    let packs = [
        (
            "small-gptq.safetensors",
            ["gptq.qweight", "gptq.qzeros", "gptq.scales", "gptq.g_idx"]
                .as_slice(),
        ),
        (
            "small-awq.safetensors",
            ["awq.qweight", "awq.qzeros", "awq.scales"].as_slice(),
        ),
        (
            "small-bnb.safetensors",
            ["bnb.weight", "bnb.absmax"].as_slice(),
        ),
        (
            "small-ct.safetensors",
            ["ct.weight_packed"].as_slice(),
        ),
    ];
    for (file, names) in packs {
        let archive = archive::open(
            &vectors().join(file),
            None,
            crate::cli::DEFAULT_MAX_HEADER,
            None,
            None,
        )
        .unwrap();
        for name in names {
            let (dtype, shape, digest) = tensor(&doc, name);
            let member = archive
                .members
                .iter()
                .find(|member| member.name == *name)
                .unwrap_or_else(|| panic!("{file} missing {name}"));
            assert_eq!(canonical_id(&member.dtype).unwrap(), dtype, "{name}");
            assert_eq!(member.shape, shape, "{name}");
            assert_eq!(hash_member(member), digest, "{name}");
            let detail = member.detail.as_deref().unwrap_or("");
            if file.contains("gptq") {
                assert!(detail.contains("GPTQ"), "{name} detail {detail}");
            } else if file.contains("awq") {
                assert!(detail.contains("AWQ"), "{name} detail {detail}");
            } else if file.contains("bnb") {
                assert!(detail.contains("bitsandbytes"), "{name} detail {detail}");
            } else if file.contains("ct") {
                assert!(detail.contains("compressed-tensors"), "{name} detail {detail}");
            }
        }
    }
}
