use super::*;

#[test]
fn glm53_flash_pinned_config_rejects_contract_and_same_cardinality_mutations() {
    config();
    let mut value: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    value["model_type"] = serde_json::json!("deepseek_v3");
    assert!(Glm53FlashHfConfig::from_slice(&serde_json::to_vec(&value).unwrap()).is_err());

    let mut value: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    value["text_config"]["layer_types"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(Glm53FlashHfConfig::from_slice(&serde_json::to_vec(&value).unwrap()).is_err());

    let mut value: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    value["quantization_config"]["fmt"] = serde_json::json!("e5m2");
    assert!(Glm53FlashHfConfig::from_slice(&serde_json::to_vec(&value).unwrap()).is_err());

    let mut value: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    value["text_config"]["rms_norm_eps"] = serde_json::json!(1.0e-6);
    let error = Glm53FlashHfConfig::from_slice(&serde_json::to_vec(&value).unwrap())
        .unwrap_err()
        .to_string();
    assert!(error.contains("rms_norm_eps must be 1e-5"), "{error}");

    let mut value: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    let exclusions = value["quantization_config"]["modules_to_not_convert"]
        .as_array_mut()
        .unwrap();
    let original_len = exclusions.len();
    exclusions[0] = serde_json::json!("same.cardinality.digest.mutation");
    assert_eq!(exclusions.len(), original_len);
    assert_eq!(
        exclusions
            .iter()
            .filter_map(serde_json::Value::as_str)
            .collect::<BTreeSet<_>>()
            .len(),
        original_len
    );
    let error = Glm53FlashHfConfig::from_slice(&serde_json::to_vec(&value).unwrap())
        .unwrap_err()
        .to_string();
    assert!(error.contains("exclusion inventory mismatch"), "{error}");

    let mut value: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    let quantization = value["quantization_config"].as_object_mut().unwrap();
    let field_count = quantization.len();
    quantization.remove("fmt").unwrap();
    quantization.insert(
        "unknown_quantization_field".to_string(),
        serde_json::json!("e4m3"),
    );
    assert_eq!(quantization.len(), field_count);
    let error = Glm53FlashHfConfig::from_slice(&serde_json::to_vec(&value).unwrap())
        .unwrap_err()
        .to_string();
    assert!(error.contains("unknown field"), "{error}");
}

#[test]
fn glm53_flash_indexer_schedule_comes_from_the_validated_layer_list() {
    use Glm53FlashIndexerKind::{Full, Shared};

    let uniform_kind = |layers: &[Glm53FlashIndexerKind]| {
        Glm53FlashIndexerSchedule::uniform(layers).map(Glm53FlashIndexerSchedule::kind)
    };
    assert_eq!(uniform_kind(&[Full; 45]), Some(Full));
    assert_eq!(uniform_kind(&[Shared; 45]), Some(Shared));
    let mut mixed = [Full; 45];
    mixed[3] = Shared;
    assert_eq!(uniform_kind(&mixed), None);
    assert_eq!(uniform_kind(&[]), None);

    assert_eq!(config().validate().unwrap().kind(), Full);

    // The published contract pins every row to `full`, so a changed row is rejected, not mapped.
    for changed_layers in [3..4, 0..45] {
        let mut value: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
        let layers = value["text_config"]["indexer_types"]
            .as_array_mut()
            .unwrap();
        for layer in changed_layers.clone() {
            layers[layer] = serde_json::json!("shared");
        }
        let error =
            Glm53FlashHfConfig::from_slice(&serde_json::to_vec(&value).unwrap()).unwrap_err();
        assert!(
            matches!(
                &error,
                Glm53FlashMetadataError::InvalidConfig(message)
                    if message == "published indexer schedule mismatch"
            ),
            "layers {changed_layers:?}: {error:?}"
        );
    }
}
