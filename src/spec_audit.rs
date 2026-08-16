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

/// Open `file` and compare each named tensor with `expected.json`, where the
/// expected row is the file's tensor name with `prefix` in front.
fn check_review_fixture(file: &str, prefix: &str, names: &[&str]) -> crate::archive::Archive {
    let doc = expected();
    let archive = archive::open(
        &vectors().join(file),
        None,
        crate::cli::DEFAULT_MAX_HEADER,
        None,
        None,
    )
    .unwrap_or_else(|err| panic!("{file}: {err}"));
    for name in names {
        let (dtype, shape, digest) = tensor(&doc, &format!("{prefix}{name}"));
        let member = archive
            .members
            .iter()
            .find(|member| member.name == *name)
            .unwrap_or_else(|| panic!("{file} missing {name}"));
        assert_eq!(canonical_id(&member.dtype).unwrap(), dtype, "{name}");
        assert_eq!(member.shape, shape, "{name}");
        assert_eq!(hash_member(member), digest, "{name}");
    }
    archive
}

#[cfg(format_pytorch)]
#[test]
fn pytorch_state_dict_with_build_and_untyped_storage_reads() {
    let archive = check_review_fixture(
        "small-state-dict.pt",
        "sd.",
        &["u16", "u32", "u64", "lin.weight", "lin.bias"],
    );
    assert_eq!(archive.format, FormatName::Pytorch);
}

#[cfg(format_onnx)]
#[test]
fn onnx_typed_fields_are_packed_varints() {
    check_review_fixture(
        "small-typed.onnx",
        "onnx.",
        &["t_f32", "t_f16", "t_i8", "t_i32", "t_i64", "t_u32"],
    );
}

#[test]
fn safetensors_header_of_640_bytes_is_not_a_pickle() {
    let path = vectors().join("edge/safetensors-n640.bin");
    let kind = crate::detect::sniff(&path, crate::cli::DEFAULT_MAX_HEADER).unwrap();
    assert_eq!(kind, crate::detect::Detected::Safetensors);
    archive::open(&path, None, crate::cli::DEFAULT_MAX_HEADER, None, None).unwrap();
}

#[cfg(format_gguf)]
#[test]
fn gguf_without_tensors_or_padding_reads() {
    let path = vectors().join("edge/gguf-v3-no-tensors.gguf");
    let archive = archive::open(&path, None, crate::cli::DEFAULT_MAX_HEADER, None, None).unwrap();
    assert_eq!(archive.format, FormatName::Gguf);
    assert!(archive.members.is_empty());
}

fn hex_bits(text: &str) -> u32 {
    u32::from_str_radix(text.trim_start_matches("0x"), 16).unwrap()
}

#[test]
fn float_narrowing_rounds_to_nearest_even() {
    let doc: Value = serde_json::from_slice(&fs::read(vectors().join("rounding.json")).unwrap()).unwrap();
    for (dtype, cast) in [
        ("float16", crate::ggml_dequant::f32_to_f16 as fn(f32) -> u16),
        ("bfloat16", crate::ggml_dequant::f32_to_bf16 as fn(f32) -> u16),
    ] {
        for case in doc["casts"][dtype].as_array().unwrap() {
            let input = f32::from_bits(hex_bits(case["in_f32_bits"].as_str().unwrap()));
            let got = cast(input);
            let note = case["note"].as_str().unwrap_or("");
            if input.is_nan() {
                // CONV-001 leaves the NaN payload open; the result must be a NaN.
                let back = if dtype == "float16" {
                    crate::ggml_dequant::f16_to_f32(got)
                } else {
                    f32::from_bits(u32::from(got) << 16)
                };
                assert!(back.is_nan(), "{dtype} {note}: {got:#06x}");
            } else {
                let expect = hex_bits(case["out_bits"].as_str().unwrap()) as u16;
                assert_eq!(got, expect, "{dtype} {note}: {got:#06x}");
            }
        }
    }
}

#[test]
fn q8_0_quantize_uses_roundf() {
    let doc: Value = serde_json::from_slice(&fs::read(vectors().join("rounding.json")).unwrap()).unwrap();
    let block = &doc["q8_0"];
    let input: Vec<f32> = block["input_f32_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|bits| f32::from_bits(hex_bits(bits.as_str().unwrap())))
        .collect();
    let bytes = crate::ggml_dequant::quantize_q8_0(&input).unwrap();
    let hex: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    assert_eq!(hex, block["block_hex"].as_str().unwrap());
}

/// Dequantize `file` with `config` and compare one output tensor with the
/// `dequantized` entry of expected.json. `transpose` swaps a 2-D result
/// whose reference is stored [in, out].
fn check_dequant(file: &str, config: Value, tensor: &str, entry: &str, transpose: bool) {
    let doc = expected();
    let row = &doc["dequantized"][entry];
    let archive = archive::open(
        &vectors().join(file),
        None,
        crate::cli::DEFAULT_MAX_HEADER,
        None,
        None,
    )
    .unwrap();
    let mut warnings = Vec::new();
    let tensors =
        crate::quant::dequant_archive(&archive, Some(&config), &mut warnings)
            .unwrap_or_else(|err| panic!("{file}: {err}"));
    let found = tensors
        .iter()
        .find(|item| item.name == tensor)
        .unwrap_or_else(|| panic!("{file}: no {tensor}"));
    let mut shape = found.shape.clone();
    let mut values = found.values.clone();
    if transpose {
        let (rows, cols) = (shape[0] as usize, shape[1] as usize);
        values = (0..cols * rows)
            .map(|index| found.values[(index % rows) * cols + index / rows])
            .collect();
        shape.reverse();
    }
    let expect_shape: Vec<u64> = row["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|dim| dim.as_u64().unwrap())
        .collect();
    assert_eq!(shape, expect_shape, "{file}");
    let bytes: Vec<u8> = match row["dtype"].as_str().unwrap() {
        "float32" => values.iter().flat_map(|value| value.to_le_bytes()).collect(),
        "bfloat16" => values
            .iter()
            .flat_map(|value| crate::ggml_dequant::f32_to_bf16(*value).to_le_bytes())
            .collect(),
        other => panic!("{other}"),
    };
    assert_eq!(sha256::hex(&bytes), row["sha256"].as_str().unwrap(), "{file}");
}

#[test]
fn gptq_dequant_matches_the_reference() {
    check_dequant(
        "small-gptq.safetensors",
        serde_json::json!({"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 32}}),
        "gptq.weight",
        "small-gptq.safetensors",
        false,
    );
}

#[test]
fn awq_dequant_matches_the_reference() {
    check_dequant(
        "small-awq.safetensors",
        serde_json::json!({"quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 64, "version": "GEMM"}}),
        "awq.weight",
        "small-awq.safetensors",
        false,
    );
}

#[test]
fn bitsandbytes_dequant_matches_the_reference() {
    check_dequant(
        "small-bnb.safetensors",
        serde_json::json!({"quantization_config": {"quant_method": "bitsandbytes", "bnb_4bit_quant_type": "nf4", "bnb_4bit_quant_blocksize": 64}}),
        "bnb.weight",
        "small-bnb.safetensors",
        false,
    );
    // The Hugging Face layout carries its own quant_state (BNB-005, BNB-006).
    check_dequant(
        "small-bnb-hf.safetensors",
        serde_json::json!({"quantization_config": {"quant_method": "bitsandbytes"}}),
        "layer.weight",
        "small-bnb-hf.safetensors layer.weight",
        false,
    );
}

#[test]
fn compressed_tensors_group_dequant_matches_the_reference() {
    check_dequant(
        "small-ct-group.safetensors",
        serde_json::json!({"quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {"group_0": {
                "targets": ["re:ct$"],
                "weights": {"num_bits": 4, "type": "int", "symmetric": true, "strategy": "group", "group_size": 8}
            }}
        }}),
        "ct.weight",
        "small-ct-group.safetensors ct.weight",
        false,
    );
}

#[cfg(format_npy)]
#[test]
fn big_endian_npy_reads_as_canonical_little_endian() {
    let dir = tempfile::tempdir().unwrap();
    // complex64 swaps each f32 part, so (1+2j) stays (1+2j) (SCHEMA.md byteswap).
    let path = dir.path().join("c.npy");
    let mut bytes = crate::formats::npy::write_header(">c8", &[1]).unwrap();
    bytes.extend_from_slice(&1.0f32.to_be_bytes());
    bytes.extend_from_slice(&2.0f32.to_be_bytes());
    fs::write(&path, &bytes).unwrap();
    let archive = archive::open(&path, None, crate::cli::DEFAULT_MAX_HEADER, None, None).unwrap();
    let mut expect = 1.0f32.to_le_bytes().to_vec();
    expect.extend_from_slice(&2.0f32.to_le_bytes());
    assert_eq!(archive.members[0].payload.read_vec().unwrap(), expect);
}

#[test]
fn compressed_tensors_asymmetric_dequant_matches_the_reference() {
    // CT-006: the zero point is packed on dim 0.
    check_dequant(
        "small-ct-asym.safetensors",
        serde_json::json!({"quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {"group_0": {
                "targets": ["Linear"],
                "weights": {"num_bits": 4, "type": "int", "symmetric": false, "strategy": "group", "group_size": 8}
            }}
        }}),
        "ct.weight",
        "small-ct-asym.safetensors ct.weight",
        false,
    );
}

#[test]
fn bitsandbytes_nested_absmax_dequant_matches_the_reference() {
    // BNB-004: compress_statistics=True.
    check_dequant(
        "small-bnb-nested.safetensors",
        serde_json::json!({"quantization_config": {"quant_method": "bitsandbytes"}}),
        "layer.weight",
        "small-bnb-nested.safetensors layer.weight",
        false,
    );
}

#[test]
fn gptq_v2_dequant_matches_the_reference() {
    // GPTQ-008: v2 zeros are the true zero point.
    check_dequant(
        "small-gptq-v2.safetensors",
        serde_json::json!({"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 32, "checkpoint_format": "gptq_v2"}}),
        "gptq.weight",
        "small-gptq-v2.safetensors",
        false,
    );
}

#[test]
fn mlx_nvfp4_per_expert_dequant_matches_the_reference() {
    // MLX-014: global_scale[e] belongs to expert e.
    let name = "model.layers.0.mlp.switch_mlp.gate_proj.weight";
    let entry = format!("small-mlx-nvfp4-experts.safetensors {name}");
    let config = expected()["dequantized"][&entry]["config"].clone();
    check_dequant("small-mlx-nvfp4-experts.safetensors", config, name, &entry, false);
    // MLX-016: without the per-module entry the mode is inferred from the
    // uint8 scales (bits * group_size = 32 * 4 / 2 = 64, nvfp4).
    let inferred = serde_json::json!({"quantization": {"group_size": 64, "bits": 4}});
    check_dequant("small-mlx-nvfp4-experts.safetensors", inferred, name, &entry, false);
}

#[test]
fn gguf_orig_shape_restores_the_canonical_shape() {
    // GGUF-023: an empty orig_shape is a scalar; [2, 3] stored as ne [6].
    let path = vectors().join("edge/gguf-orig-shape.gguf");
    let archive = archive::open(&path, None, crate::cli::DEFAULT_MAX_HEADER, None, None).unwrap();
    let shapes: Vec<(&str, Vec<u64>)> = archive
        .members
        .iter()
        .map(|member| (member.name.as_str(), member.shape.clone()))
        .collect();
    assert_eq!(shapes, vec![("s_f32", vec![]), ("v", vec![3, 2])]);
    assert!(!archive.metadata.iter().any(|(key, _)| key.ends_with(".orig_shape")));
}

#[test]
fn compressed_tensors_targets_follow_ct_009_to_ct_011() {
    let modules: std::collections::BTreeSet<String> =
        ["model", "model.layers", "model.layers.0", "model.layers.0.self_attn", "model.layers.0.self_attn.q_proj", "lm_head"]
            .iter()
            .map(|path| path.to_string())
            .collect();
    let scheme = |bits: u64| serde_json::json!({"num_bits": bits, "type": "int", "symmetric": true, "strategy": "channel"});
    let quant = |groups: Value, ignore: Value| {
        serde_json::json!({"config_groups": groups, "ignore": ignore}).as_object().unwrap().clone()
    };
    let mut warnings = Vec::new();
    // CT-011: groups and alternation, anchored at the start only.
    let schemes = crate::quant::CtSchemes::from_config(
        &quant(
            serde_json::json!({"group_0": {"targets": [r"re:.*self_attn\.(q|k|v|o)_proj"], "weights": scheme(4)}}),
            serde_json::json!(["lm_head"]),
        ),
        &modules,
        &mut warnings,
    )
    .unwrap();
    assert!(schemes.scheme_for("model.layers.0.self_attn.q_proj", &|_| true).is_ok());
    assert!(schemes.scheme_for("model.layers.0.mlp.up_proj", &|_| true).is_err());
    assert!(warnings.is_empty());
    // Lookahead is past what the regex crate compiles: refused, with the pattern.
    let err = crate::quant::CtSchemes::from_config(
        &quant(
            serde_json::json!({"group_0": {"targets": [r"re:^(?!.*mtp\.).+$"], "weights": scheme(4)}}),
            serde_json::json!([]),
        ),
        &modules,
        &mut warnings,
    )
    .err()
    .unwrap();
    assert!(err.to_string().contains("CT-011") && err.to_string().contains("mtp"));
    // CT-009: class targets of two schemes; only the tensors can decide.
    let two = crate::quant::CtSchemes::from_config(
        &quant(
            serde_json::json!({
                "group_0": {"targets": ["Linear"], "weights": scheme(4)},
                "group_1": {"targets": ["MoELinear"], "weights": scheme(8)}
            }),
            serde_json::json!(["Embedding"]),
        ),
        &modules,
        &mut warnings,
    )
    .unwrap();
    assert!(warnings.iter().any(|warning| warning.contains("Embedding") && warning.contains("CT-009")));
    let module = "model.layers.0.self_attn.q_proj";
    assert!(two.scheme_for(module, &|_| true).is_err());
    let eight = two
        .scheme_for(module, &|scheme| scheme.num_bits == 8)
        .unwrap();
    assert_eq!(eight.num_bits, 8);
}

#[cfg(format_npy)]
#[test]
fn native_order_npy_reads_as_little_endian_with_a_note() {
    // NPY-013: '=' is read as '<' and reported.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("n.npy");
    let mut bytes = crate::formats::npy::write_header("=f4", &[1]).unwrap();
    bytes.extend_from_slice(&1.5f32.to_le_bytes());
    fs::write(&path, &bytes).unwrap();
    let archive = archive::open(&path, None, crate::cli::DEFAULT_MAX_HEADER, None, None).unwrap();
    assert!(archive.notes.iter().any(|note| note.contains("NPY-013")));
    assert_eq!(archive.members[0].payload.read_vec().unwrap(), 1.5f32.to_le_bytes());
}

#[test]
fn compressed_tensors_naive_formats_dequant_match_the_reference() {
    // int-quantized: int8 codes, channel scale.
    check_dequant(
        "small-ct-int8.safetensors",
        serde_json::json!({"quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "config_groups": {"group_0": {
                "targets": ["Linear"],
                "weights": {"num_bits": 8, "type": "int", "symmetric": true, "strategy": "channel"}
            }}
        }}),
        "ct.weight",
        "small-ct-int8.safetensors ct.weight",
        false,
    );
    // float-quantized: float8_e4m3fn codes, block [4, 8] with partial edge
    // blocks, here reached through the FP8_BLOCK-style preset fields.
    check_dequant(
        "small-ct-fp8-block.safetensors",
        serde_json::json!({"quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {"FP8_BLOCK": {
                "targets": ["Linear"],
                "weights": {"num_bits": 8, "type": "float", "symmetric": true, "strategy": "block", "block_structure": [4, 8]}
            }}
        }}),
        "ct.weight",
        "small-ct-fp8-block.safetensors ct.weight",
        false,
    );
}

#[test]
fn compressed_tensors_preset_names_expand_only_for_target_lists() {
    // CT-012.
    let modules: std::collections::BTreeSet<String> = ["layer"].iter().map(|path| path.to_string()).collect();
    let mut warnings = Vec::new();
    let config = |groups: Value| serde_json::json!({"config_groups": groups}).as_object().unwrap().clone();
    let schemes = crate::quant::CtSchemes::from_config(
        &config(serde_json::json!({"w4a16": ["Linear"], "fp8_dynamic": ["re:lm_head"], "UNQUANTIZED": ["re:skip"]})),
        &modules,
        &mut warnings,
    )
    .unwrap();
    let group = schemes.scheme_for("layer", &|_| true).unwrap();
    assert_eq!((group.num_bits, group.float, group.strategy.as_str(), group.group_size), (4, false, "group", 128));
    let fp8 = schemes.scheme_for("lm_head", &|_| true).unwrap();
    assert_eq!((fp8.num_bits, fp8.float, fp8.strategy.as_str()), (8, true, "channel"));
    let unknown = crate::quant::CtSchemes::from_config(
        &config(serde_json::json!({"W9A16": ["Linear"]})),
        &modules,
        &mut warnings,
    );
    assert!(unknown.err().unwrap().to_string().contains("CT-012"));
}

#[cfg(all(format_gguf, format_npz))]
#[test]
fn dequant_copies_integer_tensors_unchanged() {
    // float32 cannot hold every int32, so a plain integer tensor is copied.
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.npz");
    crate::quant::run(&crate::quant::Request {
        source: vectors().join("small.gguf"),
        output: out.clone(),
        format: None,
        quantize: None,
        force_single: false,
        force: false,
    })
    .unwrap();
    let source = archive::open(&vectors().join("small.gguf"), None, crate::cli::DEFAULT_MAX_HEADER, None, None).unwrap();
    let written = archive::open(&out, None, crate::cli::DEFAULT_MAX_HEADER, None, None).unwrap();
    let find = |archive: &crate::archive::Archive, name: &str| {
        archive.members.iter().find(|member| member.name == name).unwrap().clone()
    };
    let (before, after) = (find(&source, "i_i32"), find(&written, "i_i32"));
    assert_eq!(after.dtype, "I32");
    assert_eq!(after.shape, vec![3, 2]);
    assert_eq!(after.payload.read_vec().unwrap(), before.payload.read_vec().unwrap());
    assert_eq!(find(&written, "w_q8_0").dtype, "F32");
}

#[cfg(format_npz)]
#[test]
fn an_npz_edit_keeps_members_it_does_not_touch() {
    use std::io::{Read, Write};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pack.npz");
    fs::copy(vectors().join("small.npz"), &path).unwrap();
    {
        let file = fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
        let mut writer = zip::ZipWriter::new_append(file).unwrap();
        writer.start_file("readme.txt", zip::write::SimpleFileOptions::default()).unwrap();
        writer.write_all(b"hello").unwrap();
        writer.finish().unwrap();
    }
    let read_member = |path: &std::path::Path, name: &str| {
        let mut archive = zip::ZipArchive::new(fs::File::open(path).unwrap()).unwrap();
        let mut bytes = Vec::new();
        archive.by_name(name).unwrap().read_to_end(&mut bytes).unwrap();
        bytes
    };
    let original_bf16 = read_member(&path, "e_bf16.npy");
    // e_bf16 is a <V2 array with no safetensors dtype; it is left alone.
    crate::run(&["--delete".into(), "-f".into(), path.display().to_string(), "s_f32".into()]).unwrap();
    assert_eq!(read_member(&path, "e_bf16.npy"), original_bf16);
    assert_eq!(read_member(&path, "readme.txt"), b"hello");
    let archive = archive::open(&path, None, crate::cli::DEFAULT_MAX_HEADER, None, None).unwrap();
    assert!(!archive.members.iter().any(|member| member.name == "s_f32"));
}
