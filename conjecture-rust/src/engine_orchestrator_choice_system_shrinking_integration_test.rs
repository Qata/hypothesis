//! Basic functionality tests for Choice System Shrinking Integration

use crate::choice::{ChoiceNode, ChoiceType, ChoiceValue, Constraints, IntegerConstraints};
use crate::data::{ConjectureResult, Status};
use crate::engine_orchestrator_choice_system_shrinking_integration::{
    ChoiceSystemShrinkingIntegration, ShrinkingIntegrationConfig
};
use std::time::Duration;

#[test]
fn test_shrinking_integration_basic_functionality() {
    // Create a basic configuration
    let config = ShrinkingIntegrationConfig {
        max_shrinking_attempts: 10,
        shrinking_timeout: Duration::from_secs(5),
        enable_advanced_patterns: true,
        enable_multi_strategy: true,
        quality_improvement_threshold: 0.001,
        max_concurrent_strategies: 2,
        debug_logging: false,
        use_hex_notation: false,
    };

    // Create the shrinking integration engine
    let shrinking_integration = ChoiceSystemShrinkingIntegration::new(config);
    
    // Verify the engine was created successfully
    assert!(!shrinking_integration.config.debug_logging);
    assert_eq!(shrinking_integration.config.max_shrinking_attempts, 10);
}

#[test]
fn test_conjecture_result_to_choice_conversion() {
    // Create a simple ConjectureResult with some choice nodes
    let mut nodes = Vec::new();
    
    // Add an integer choice
    nodes.push(ChoiceNode {
        choice_type: ChoiceType::Integer,
        value: ChoiceValue::Integer(42),
        constraints: Some(Box::new(Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
        }))),
        metadata: None,
    });
    
    let result = ConjectureResult {
        nodes,
        status: Status::Interesting,
        output: vec![42u8],
        tags: std::collections::HashSet::new(),
        target_observations: vec![],
        extra_information: ExtraInformation::new(),
    };

    let config = ShrinkingIntegrationConfig::default();
    let mut shrinking_integration = ChoiceSystemShrinkingIntegration::new(config);
    
    // Test the conversion
    let conversion_result = shrinking_integration.convert_conjecture_result_to_choices(&result);
    assert!(conversion_result.is_ok());
    
    let choices = conversion_result.unwrap();
    assert_eq!(choices.len(), 1);
}

#[test]
fn test_shrinking_integration_configuration() {
    let config = ShrinkingIntegrationConfig::default();
    
    // Verify default configuration values
    assert_eq!(config.max_shrinking_attempts, 500);
    assert_eq!(config.shrinking_timeout, Duration::from_secs(300));
    assert!(config.enable_advanced_patterns);
    assert!(config.enable_multi_strategy);
    assert_eq!(config.quality_improvement_threshold, 0.001);
    assert_eq!(config.max_concurrent_strategies, 4);
    assert!(config.debug_logging);
    assert!(config.use_hex_notation);
}