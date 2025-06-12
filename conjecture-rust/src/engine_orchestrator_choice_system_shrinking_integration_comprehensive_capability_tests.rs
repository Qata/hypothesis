//! Comprehensive tests for EngineOrchestrator Choice System Shrinking Integration capability
//!
//! This module provides comprehensive integration tests that validate the complete
//! Choice System Shrinking Integration capability works correctly. These tests focus on the
//! complete capability's behavior, not individual functions, and validate the entire capability
//! works correctly through integration with the sophisticated ChoiceShrinker implementation.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::choice::{ChoiceNode, ChoiceType, ChoiceValue, Constraints, IntegerConstraints, BooleanConstraints, FloatConstraints, StringConstraints, BytesConstraints};
use crate::data::{ConjectureData, ConjectureResult, Status};
use crate::engine_orchestrator::{
    EngineOrchestrator, OrchestratorConfig, ExecutionPhase, ExitReason, OrchestrationError
};
use crate::engine_orchestrator_choice_system_shrinking_integration::{
    ChoiceSystemShrinkingIntegration, ShrinkingIntegrationConfig, ShrinkingIntegrationResult,
    ConversionError
};

/// Test configuration for shrinking integration capability tests
const TEST_SHRINKING_TIMEOUT: Duration = Duration::from_secs(10);
const TEST_MAX_SHRINKING_ATTEMPTS: usize = 50;
const TEST_QUALITY_THRESHOLD: f64 = 0.1;

/// Create a test example with complex choice structure for shrinking tests
fn create_complex_test_example() -> ConjectureResult {
    let nodes = vec![
        ChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(100),
            constraints: Constraints::Integer(IntegerConstraints::default()),
            was_forced: false,
            index: Some(0),
        },
        ChoiceNode {
            choice_type: ChoiceType::Boolean,
            value: ChoiceValue::Boolean(true),
            constraints: Constraints::Boolean(BooleanConstraints::default()),
            was_forced: false,
            index: Some(1),
        },
        ChoiceNode {
            choice_type: ChoiceType::Float,
            value: ChoiceValue::Float(3.14159),
            constraints: Constraints::Float(FloatConstraints::default()),
            was_forced: false,
            index: Some(2),
        },
        ChoiceNode {
            choice_type: ChoiceType::String,
            value: ChoiceValue::String("hello_world_test_string".to_string()),
            constraints: Constraints::String(StringConstraints::default()),
            was_forced: false,
            index: Some(3),
        },
        ChoiceNode {
            choice_type: ChoiceType::Bytes,
            value: ChoiceValue::Bytes(vec![1, 2, 3, 4, 5, 6, 7, 8]),
            constraints: Constraints::Bytes(BytesConstraints::default()),
            was_forced: false,
            index: Some(4),
        },
    ];

    ConjectureResult {
        status: Status::Interesting,
        nodes,
        length: 5,
        events: HashMap::new(),
        buffer: Vec::new(),
        examples: Vec::new(),
        interesting_origin: Some("test_complex".to_string()),
        output: Vec::new(),
        extra_information: crate::data::ExtraInformation::new(),
        expected_exception: None,
        expected_traceback: None,
        has_discards: false,
        target_observations: HashMap::new(),
        tags: HashSet::new(),
        spans: Vec::new(),
        arg_slices: Vec::new(),
        slice_comments: HashMap::new(),
        misaligned_at: None,
        cannot_proceed_scope: None,
    }
}

/// Create a minimal test example for baseline shrinking tests
fn create_minimal_test_example() -> ConjectureResult {
    let nodes = vec![
        ChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(1),
            constraints: Constraints::Integer(IntegerConstraints::default()),
            was_forced: false,
            index: Some(0),
        },
    ];

    ConjectureResult {
        status: Status::Interesting,
        nodes,
        length: 1,
        events: HashMap::new(),
        buffer: Vec::new(),
        examples: Vec::new(),
        interesting_origin: Some("test_minimal".to_string()),
        output: Vec::new(),
        extra_information: crate::data::ExtraInformation::new(),
        expected_exception: None,
        expected_traceback: None,
        has_discards: false,
        target_observations: HashMap::new(),
        tags: HashSet::new(),
        spans: Vec::new(),
        arg_slices: Vec::new(),
        slice_comments: HashMap::new(),
        misaligned_at: None,
        cannot_proceed_scope: None,
    }
}

/// Create test orchestrator with shrinking integration capability
fn create_test_orchestrator_with_shrinking() -> EngineOrchestrator {
    let test_function = Box::new(|data: &mut ConjectureData| -> Result<(), OrchestrationError> {
        // Mock test function that produces interesting examples
        data.status = Status::Interesting;
        Ok(())
    });

    let mut config = OrchestratorConfig::default();
    config.debug_logging = true;
    config.max_examples = 10;
    config.ignore_limits = false;

    EngineOrchestrator::new(test_function, config)
}

/// Test the complete choice system shrinking integration capability end-to-end
#[test]
fn test_choice_system_shrinking_integration_complete_capability() {
    println!("Testing complete Choice System Shrinking Integration capability");

    // Create test configuration for shrinking integration
    let shrinking_config = ShrinkingIntegrationConfig {
        max_shrinking_attempts: TEST_MAX_SHRINKING_ATTEMPTS,
        shrinking_timeout: TEST_SHRINKING_TIMEOUT,
        enable_advanced_patterns: true,
        enable_multi_strategy: true,
        quality_improvement_threshold: TEST_QUALITY_THRESHOLD,
        max_concurrent_strategies: 2,
        debug_logging: true,
        use_hex_notation: true,
    };

    // Create shrinking integration engine
    let mut shrinking_integration = ChoiceSystemShrinkingIntegration::new(shrinking_config);

    // Create test orchestrator
    let mut orchestrator = create_test_orchestrator_with_shrinking();

    // Create test examples to shrink
    let mut interesting_examples = HashMap::new();
    interesting_examples.insert("complex_example".to_string(), create_complex_test_example());
    interesting_examples.insert("minimal_example".to_string(), create_minimal_test_example());

    // Apply sophisticated shrinking integration
    let start_time = Instant::now();
    let shrinking_results = shrinking_integration.integrate_shrinking(&mut orchestrator, &interesting_examples);
    let integration_duration = start_time.elapsed();

    // Validate shrinking integration results
    assert!(shrinking_results.is_ok(), "Shrinking integration should succeed");
    let results = shrinking_results.unwrap();

    // Verify all examples were processed
    assert_eq!(results.len(), 2, "Should process all interesting examples");
    assert!(results.contains_key("complex_example"), "Should process complex example");
    assert!(results.contains_key("minimal_example"), "Should process minimal example");

    // Validate complex example shrinking
    let complex_result = &results["complex_example"];
    assert!(complex_result.success, "Complex example shrinking should succeed");
    assert!(complex_result.final_size <= complex_result.original_size, "Should not increase size");
    assert!(!complex_result.successful_strategies.is_empty(), "Should use successful strategies");
    assert!(complex_result.shrinking_duration < TEST_SHRINKING_TIMEOUT, "Should complete within timeout");

    // Validate minimal example handling (may not shrink further)
    let minimal_result = &results["minimal_example"];
    assert_eq!(minimal_result.original_size, 1, "Minimal example should have size 1");

    // Validate integration metrics
    let metrics = shrinking_integration.get_metrics();
    assert_eq!(metrics.total_attempts, 2, "Should track all attempts");
    assert!(metrics.successful_integrations > 0, "Should have successful integrations");
    assert!(integration_duration < Duration::from_secs(30), "Integration should complete quickly");

    println!("✓ Choice System Shrinking Integration complete capability test passed");
    println!("  - Processed {} examples in {:?}", results.len(), integration_duration);
    println!("  - Complex example: {} -> {} nodes ({:.1}% reduction)", 
             complex_result.original_size, complex_result.final_size, complex_result.reduction_percentage());
    println!("  - Successful strategies: {:?}", complex_result.successful_strategies);
}

/// Test shrinking integration with sophisticated ConjectureResult conversion
#[test]
fn test_shrinking_integration_conjecture_result_conversion() {
    println!("Testing shrinking integration ConjectureResult conversion capability");

    let shrinking_config = ShrinkingIntegrationConfig {
        max_shrinking_attempts: 10,
        shrinking_timeout: Duration::from_secs(5),
        enable_advanced_patterns: true,
        enable_multi_strategy: false, // Focus on conversion testing
        quality_improvement_threshold: 0.05,
        max_concurrent_strategies: 1,
        debug_logging: true,
        use_hex_notation: true,
    };

    let mut shrinking_integration = ChoiceSystemShrinkingIntegration::new(shrinking_config);
    let mut orchestrator = create_test_orchestrator_with_shrinking();

    // Create example with diverse choice types to test conversion
    let complex_example = create_complex_test_example();
    let original_node_count = complex_example.nodes.len();

    let mut test_examples = HashMap::new();
    test_examples.insert("conversion_test".to_string(), complex_example);

    // Test conversion through shrinking integration
    let conversion_result = shrinking_integration.integrate_shrinking(&mut orchestrator, &test_examples);
    assert!(conversion_result.is_ok(), "Conversion through shrinking should succeed");

    let results = conversion_result.unwrap();
    let conversion_test_result = &results["conversion_test"];

    // Validate conversion preserved choice types
    if let Some(ref shrunk_result) = conversion_test_result.shrunk_result {
        assert!(!shrunk_result.nodes.is_empty(), "Conversion should preserve some nodes");
        
        // Verify choice type preservation
        for node in &shrunk_result.nodes {
            match (&node.choice_type, &node.value) {
                (ChoiceType::Integer, ChoiceValue::Integer(_)) => {},
                (ChoiceType::Boolean, ChoiceValue::Boolean(_)) => {},
                (ChoiceType::Float, ChoiceValue::Float(_)) => {},
                (ChoiceType::String, ChoiceValue::String(_)) => {},
                (ChoiceType::Bytes, ChoiceValue::Bytes(_)) => {},
                _ => panic!("Conversion should preserve choice type consistency"),
            }
        }
    }

    // Validate conversion metrics
    let metrics = shrinking_integration.get_metrics();
    assert!(metrics.conversion_time > Duration::ZERO, "Should track conversion time");
    assert_eq!(metrics.conversion_errors, 0, "Should have no conversion errors for valid input");

    println!("✓ Shrinking integration ConjectureResult conversion test passed");
    println!("  - Original nodes: {}", original_node_count);
    println!("  - Conversion time: {:?}", metrics.conversion_time);
    println!("  - Conversion errors: {}", metrics.conversion_errors);
}

/// Test shrinking integration with deadline and timeout management
#[test]
fn test_shrinking_integration_deadline_management() {
    println!("Testing shrinking integration deadline management capability");

    // Configure short timeout to test deadline management
    let short_timeout_config = ShrinkingIntegrationConfig {
        max_shrinking_attempts: 100, // High attempts to trigger timeout
        shrinking_timeout: Duration::from_millis(100), // Very short timeout
        enable_advanced_patterns: true,
        enable_multi_strategy: true,
        quality_improvement_threshold: 0.001, // Low threshold for more work
        max_concurrent_strategies: 4,
        debug_logging: true,
        use_hex_notation: true,
    };

    let mut shrinking_integration = ChoiceSystemShrinkingIntegration::new(short_timeout_config);
    let mut orchestrator = create_test_orchestrator_with_shrinking();

    // Create large complex example to trigger timeout
    let mut large_example = create_complex_test_example();
    // Add many more nodes to make shrinking take longer
    for i in 0..20 {
        large_example.nodes.push(ChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(i * 10),
            constraints: Constraints::Integer(IntegerConstraints::default()),
            was_forced: false,
            index: Some(large_example.nodes.len()),
        });
    }
    large_example.length = large_example.nodes.len();

    let mut test_examples = HashMap::new();
    test_examples.insert("timeout_test".to_string(), large_example);

    // Test deadline management
    let start_time = Instant::now();
    let deadline_result = shrinking_integration.integrate_shrinking(&mut orchestrator, &test_examples);
    let actual_duration = start_time.elapsed();

    // Validate deadline was respected (allowing some margin for overhead)
    assert!(deadline_result.is_ok(), "Deadline management should not cause hard failures");
    let results = deadline_result.unwrap();
    
    let timeout_test_result = &results["timeout_test"];
    // Should complete within reasonable time despite large input
    assert!(actual_duration < Duration::from_secs(5), "Should respect timeout constraints");
    
    // May succeed or fail due to timeout, but should track duration
    assert!(timeout_test_result.shrinking_duration <= Duration::from_secs(2), 
            "Should track reasonable shrinking duration");

    println!("✓ Shrinking integration deadline management test passed");
    println!("  - Actual duration: {:?}", actual_duration);
    println!("  - Shrinking duration: {:?}", timeout_test_result.shrinking_duration);
    println!("  - Success: {}", timeout_test_result.success);
}

/// Test shrinking integration error handling and recovery
#[test]
fn test_shrinking_integration_error_handling() {
    println!("Testing shrinking integration error handling capability");

    let error_config = ShrinkingIntegrationConfig {
        max_shrinking_attempts: 5,
        shrinking_timeout: Duration::from_secs(2),
        enable_advanced_patterns: true,
        enable_multi_strategy: true,
        quality_improvement_threshold: 0.1,
        max_concurrent_strategies: 2,
        debug_logging: true,
        use_hex_notation: true,
    };

    let mut shrinking_integration = ChoiceSystemShrinkingIntegration::new(error_config);
    let mut orchestrator = create_test_orchestrator_with_shrinking();

    // Create invalid example to test error handling
    let mut invalid_example = ConjectureResult {
        status: Status::Interesting,
        nodes: vec![
            ChoiceNode {
                choice_type: ChoiceType::Integer,
                value: ChoiceValue::Boolean(true), // Type mismatch to trigger error
                constraints: Constraints::Integer(IntegerConstraints::default()),
                was_forced: false,
                index: Some(0),
            },
        ],
        length: 1,
        events: HashMap::new(),
        buffer: Vec::new(),
        examples: Vec::new(),
        interesting_origin: Some("test_invalid".to_string()),
        output: Vec::new(),
        extra_information: crate::data::ExtraInformation::new(),
        expected_exception: None,
        expected_traceback: None,
        has_discards: false,
        target_observations: HashMap::new(),
        tags: HashSet::new(),
        spans: Vec::new(),
        arg_slices: Vec::new(),
        slice_comments: HashMap::new(),
        misaligned_at: None,
        cannot_proceed_scope: None,
    };

    let mut test_examples = HashMap::new();
    test_examples.insert("error_test".to_string(), invalid_example);
    test_examples.insert("valid_test".to_string(), create_minimal_test_example());

    // Test error handling and recovery
    let error_result = shrinking_integration.integrate_shrinking(&mut orchestrator, &test_examples);

    // Should handle errors gracefully and continue with other examples
    assert!(error_result.is_ok(), "Should handle individual example errors gracefully");
    let results = error_result.unwrap();

    // Validate error handling
    assert_eq!(results.len(), 2, "Should process all examples despite errors");
    
    let error_test_result = &results["error_test"];
    assert!(!error_test_result.success, "Invalid example should fail gracefully");
    assert!(!error_test_result.errors.is_empty(), "Should track errors");
    
    let valid_test_result = &results["valid_test"];
    // Valid example should succeed despite other errors
    assert!(valid_test_result.success || valid_test_result.errors.is_empty(), 
            "Valid examples should not be affected by other errors");

    // Validate error tracking in metrics
    let metrics = shrinking_integration.get_metrics();
    assert_eq!(metrics.total_attempts, 2, "Should track all attempts including failed ones");

    println!("✓ Shrinking integration error handling test passed");
    println!("  - Total attempts: {}", metrics.total_attempts);
    println!("  - Error test success: {}", error_test_result.success);
    println!("  - Error count: {}", error_test_result.errors.len());
    println!("  - Valid test success: {}", valid_test_result.success);
}

/// Test shrinking integration with orchestrator integration and lifecycle management
#[test]
fn test_shrinking_integration_orchestrator_lifecycle() {
    println!("Testing shrinking integration orchestrator lifecycle capability");

    let lifecycle_config = ShrinkingIntegrationConfig {
        max_shrinking_attempts: 20,
        shrinking_timeout: Duration::from_secs(5),
        enable_advanced_patterns: true,
        enable_multi_strategy: true,
        quality_improvement_threshold: 0.05,
        max_concurrent_strategies: 3,
        debug_logging: true,
        use_hex_notation: true,
    };

    let mut shrinking_integration = ChoiceSystemShrinkingIntegration::new(lifecycle_config);

    // Create orchestrator and put it in shrinking phase
    let mut orchestrator = create_test_orchestrator_with_shrinking();
    
    // Simulate finding interesting examples during generation
    let mut interesting_examples = HashMap::new();
    interesting_examples.insert("lifecycle_test_1".to_string(), create_complex_test_example());
    interesting_examples.insert("lifecycle_test_2".to_string(), create_minimal_test_example());

    // Track orchestrator state before shrinking
    let initial_call_count = orchestrator.call_count();
    let initial_phase = orchestrator.current_phase();

    // Apply shrinking integration with lifecycle management
    let lifecycle_result = shrinking_integration.integrate_shrinking(&mut orchestrator, &interesting_examples);
    assert!(lifecycle_result.is_ok(), "Lifecycle integration should succeed");

    let results = lifecycle_result.unwrap();

    // Validate orchestrator state preservation
    assert_eq!(orchestrator.call_count(), initial_call_count, "Call count should be preserved");
    
    // Validate shrinking results with lifecycle considerations
    assert_eq!(results.len(), 2, "Should process all examples through lifecycle");
    
    for (example_key, result) in &results {
        // Each result should have proper lifecycle tracking
        assert!(result.shrinking_duration > Duration::ZERO, 
                "Should track duration for example: {}", example_key);
        
        if result.success {
            assert!(result.shrinks_performed > 0 || result.original_size == result.final_size,
                    "Successful results should show work performed or no improvement possible");
        }
    }

    // Validate integration with orchestrator metrics
    let metrics = shrinking_integration.get_metrics();
    assert!(metrics.total_attempts >= 2, "Should track orchestrator integration attempts");
    assert!(metrics.shrinking_time > Duration::ZERO, "Should track total shrinking time");

    println!("✓ Shrinking integration orchestrator lifecycle test passed");
    println!("  - Initial phase: {:?}", initial_phase);
    println!("  - Results processed: {}", results.len());
    println!("  - Total shrinking time: {:?}", metrics.shrinking_time);
    println!("  - Successful integrations: {}", metrics.successful_integrations);
}

/// Test shrinking integration multi-strategy coordination
#[test]
fn test_shrinking_integration_multi_strategy_coordination() {
    println!("Testing shrinking integration multi-strategy coordination capability");

    let multi_strategy_config = ShrinkingIntegrationConfig {
        max_shrinking_attempts: 30,
        shrinking_timeout: Duration::from_secs(8),
        enable_advanced_patterns: true,
        enable_multi_strategy: true,
        quality_improvement_threshold: 0.01, // Low threshold to enable more strategies
        max_concurrent_strategies: 4, // Enable multiple strategies
        debug_logging: true,
        use_hex_notation: true,
    };

    let mut shrinking_integration = ChoiceSystemShrinkingIntegration::new(multi_strategy_config);
    let mut orchestrator = create_test_orchestrator_with_shrinking();

    // Create complex example suitable for multi-strategy shrinking
    let mut multi_strategy_example = create_complex_test_example();
    // Add additional complexity to benefit from multiple strategies
    for i in 0..10 {
        multi_strategy_example.nodes.push(ChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(i + 1000),
            constraints: Constraints::Integer(IntegerConstraints::default()),
            was_forced: false,
            index: Some(multi_strategy_example.nodes.len()),
        });
    }
    multi_strategy_example.length = multi_strategy_example.nodes.len();

    let mut test_examples = HashMap::new();
    test_examples.insert("multi_strategy_test".to_string(), multi_strategy_example);

    // Apply multi-strategy shrinking
    let strategy_result = shrinking_integration.integrate_shrinking(&mut orchestrator, &test_examples);
    assert!(strategy_result.is_ok(), "Multi-strategy shrinking should succeed");

    let results = strategy_result.unwrap();
    let multi_strategy_result = &results["multi_strategy_test"];

    // Validate multi-strategy coordination
    assert!(multi_strategy_result.success, "Multi-strategy shrinking should succeed");
    assert!(multi_strategy_result.shrinks_performed > 0, "Should perform shrinks");
    
    // Should use multiple strategies for complex examples
    if multi_strategy_result.successful_strategies.len() > 1 {
        println!("✓ Multiple strategies used: {:?}", multi_strategy_result.successful_strategies);
    } else {
        println!("! Single strategy sufficient for this example: {:?}", 
                 multi_strategy_result.successful_strategies);
    }

    // Validate quality improvement from multi-strategy approach
    assert!(multi_strategy_result.quality_improvement >= 0.0, 
            "Should track quality improvement");
    
    // Validate reduction effectiveness
    let reduction = multi_strategy_result.reduction_percentage();
    assert!(reduction >= 0.0, "Reduction percentage should be non-negative");
    
    if reduction > 0.0 {
        println!("✓ Achieved {:.1}% reduction through multi-strategy shrinking", reduction);
    } else {
        println!("! No size reduction achieved (example may be minimal)");
    }

    println!("✓ Shrinking integration multi-strategy coordination test passed");
    println!("  - Original size: {}", multi_strategy_result.original_size);
    println!("  - Final size: {}", multi_strategy_result.final_size);
    println!("  - Strategies used: {:?}", multi_strategy_result.successful_strategies);
    println!("  - Quality improvement: {:.3}", multi_strategy_result.quality_improvement);
}

/// Integration test for Choice System Shrinking Integration capability with complex scenarios
#[test]
fn test_choice_system_shrinking_integration_complex_scenarios() {
    println!("Testing Choice System Shrinking Integration with complex scenarios");
    
    // Create advanced shrinking configuration
    let complex_config = ShrinkingIntegrationConfig {
        max_shrinking_attempts: 25,
        shrinking_timeout: Duration::from_secs(6),
        enable_advanced_patterns: true,
        enable_multi_strategy: true,
        quality_improvement_threshold: 0.05,
        max_concurrent_strategies: 3,
        debug_logging: true,
        use_hex_notation: true,
    };

    // Create shrinking integration engine
    let mut shrinking_integration = ChoiceSystemShrinkingIntegration::new(complex_config);
    let mut orchestrator = create_test_orchestrator_with_shrinking();

    // Create test examples with different complexity levels
    let mut complex_examples = HashMap::new();
    complex_examples.insert("scenario_complex".to_string(), create_complex_test_example());
    complex_examples.insert("scenario_minimal".to_string(), create_minimal_test_example());

    // Apply shrinking integration
    let scenario_result = shrinking_integration.integrate_shrinking(&mut orchestrator, &complex_examples);
    assert!(scenario_result.is_ok(), "Complex scenario shrinking integration should succeed");

    let results = scenario_result.unwrap();

    // Validate complex scenario results
    assert_eq!(results.len(), 2, "Should process all complex scenario examples");
    
    let complex_result = &results["scenario_complex"];
    assert!(complex_result.success, "Complex scenario should succeed");
    
    let minimal_result = &results["scenario_minimal"];
    assert_eq!(minimal_result.original_size, 1, "Minimal scenario should have correct size");

    // Validate metrics
    let metrics = shrinking_integration.get_metrics();
    assert_eq!(metrics.total_attempts, 2, "Should track all scenario attempts");

    println!("✓ Choice System Shrinking Integration complex scenarios test passed");
    println!("  - Complex scenario examples processed: {}", results.len());
    println!("  - Total attempts: {}", metrics.total_attempts);
    println!("  - Successful integrations: {}", metrics.successful_integrations);
}

/// Comprehensive integration test validating the complete shrinking capability behavior
#[test]
fn test_complete_shrinking_capability_integration() {
    println!("Testing complete Choice System Shrinking Integration capability integration");

    // Create comprehensive test scenario
    let comprehensive_config = ShrinkingIntegrationConfig {
        max_shrinking_attempts: 40,
        shrinking_timeout: Duration::from_secs(10),
        enable_advanced_patterns: true,
        enable_multi_strategy: true,
        quality_improvement_threshold: 0.02,
        max_concurrent_strategies: 4,
        debug_logging: true,
        use_hex_notation: true,
    };

    let mut shrinking_integration = ChoiceSystemShrinkingIntegration::new(comprehensive_config);
    let mut orchestrator = create_test_orchestrator_with_shrinking();

    // Create diverse set of test examples
    let mut comprehensive_examples = HashMap::new();
    
    // Complex example with all choice types
    comprehensive_examples.insert("comprehensive_complex".to_string(), create_complex_test_example());
    
    // Minimal example
    comprehensive_examples.insert("comprehensive_minimal".to_string(), create_minimal_test_example());
    
    // Large example with many nodes
    let mut large_example = create_complex_test_example();
    for i in 0..25 {
        large_example.nodes.push(ChoiceNode {
            choice_type: if i % 2 == 0 { ChoiceType::Integer } else { ChoiceType::Boolean },
            value: if i % 2 == 0 { 
                ChoiceValue::Integer(i * 3) 
            } else { 
                ChoiceValue::Boolean(i % 3 == 0) 
            },
            constraints: if i % 2 == 0 { 
                Constraints::Integer(IntegerConstraints::default()) 
            } else { 
                Constraints::Boolean(BooleanConstraints::default()) 
            },
            was_forced: false,
            index: Some(large_example.nodes.len()),
        });
    }
    large_example.length = large_example.nodes.len();
    comprehensive_examples.insert("comprehensive_large".to_string(), large_example);

    // Apply comprehensive shrinking integration
    let start_time = Instant::now();
    let comprehensive_result = shrinking_integration.integrate_shrinking(
        &mut orchestrator, &comprehensive_examples
    );
    let total_duration = start_time.elapsed();

    // Validate comprehensive integration
    assert!(comprehensive_result.is_ok(), "Comprehensive shrinking integration should succeed");
    let results = comprehensive_result.unwrap();

    // Validate all examples processed
    assert_eq!(results.len(), 3, "Should process all comprehensive examples");
    assert!(results.contains_key("comprehensive_complex"), "Should process complex example");
    assert!(results.contains_key("comprehensive_minimal"), "Should process minimal example");
    assert!(results.contains_key("comprehensive_large"), "Should process large example");

    // Validate shrinking effectiveness across examples
    let mut total_original_size = 0;
    let mut total_final_size = 0;
    let mut successful_shrinks = 0;
    let mut total_strategies_used = HashSet::new();

    for (example_key, result) in &results {
        total_original_size += result.original_size;
        total_final_size += result.final_size;
        
        if result.success && result.final_size < result.original_size {
            successful_shrinks += 1;
        }
        
        for strategy in &result.successful_strategies {
            total_strategies_used.insert(strategy.clone());
        }
        
        // Validate individual result properties
        assert!(result.shrinking_duration < Duration::from_secs(15), 
                "Example {} should complete within reasonable time", example_key);
        assert!(result.original_size > 0, "Example {} should have positive original size", example_key);
        assert!(result.final_size <= result.original_size, 
                "Example {} should not increase in size", example_key);
    }

    // Validate overall shrinking effectiveness
    assert!(successful_shrinks > 0, "Should successfully shrink at least some examples");
    assert!(!total_strategies_used.is_empty(), "Should use shrinking strategies");
    
    let overall_reduction = if total_original_size > 0 {
        ((total_original_size - total_final_size) as f64 / total_original_size as f64) * 100.0
    } else {
        0.0
    };

    // Validate comprehensive metrics
    let final_metrics = shrinking_integration.get_metrics();
    assert_eq!(final_metrics.total_attempts, 3, "Should track all comprehensive attempts");
    assert!(final_metrics.successful_integrations > 0, "Should have successful integrations");
    assert!(final_metrics.shrinking_time > Duration::ZERO, "Should track shrinking time");
    assert!(total_duration < Duration::from_secs(30), "Comprehensive test should complete reasonably");

    println!("✓ Complete Choice System Shrinking Integration capability test passed");
    println!("  - Examples processed: {}", results.len());
    println!("  - Total duration: {:?}", total_duration);
    println!("  - Successful shrinks: {}/{}", successful_shrinks, results.len());
    println!("  - Overall size reduction: {:.1}%", overall_reduction);
    println!("  - Strategies used: {:?}", total_strategies_used);
    println!("  - Final metrics: successful={}, errors={}, avg_quality={:.3}", 
             final_metrics.successful_integrations, 
             final_metrics.conversion_errors,
             final_metrics.average_quality_improvement);
}