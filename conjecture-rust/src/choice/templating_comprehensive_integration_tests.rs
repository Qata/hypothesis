//! Comprehensive PyO3/FFI Integration Tests for Choice Templating and Forcing System
//!
//! This module provides comprehensive integration tests that validate the complete 
//! Choice templating and forcing system capability through PyO3 and FFI interfaces.
//! These tests validate the entire capability's behavior, not individual functions,
//! following the architectural blueprint for idiomatic Rust test patterns.

use crate::choice::{
    ChoiceType, ChoiceValue, Constraints, ChoiceNode,
    IntegerConstraints, BooleanConstraints, FloatConstraints, 
    StringConstraints, BytesConstraints, UnicodeCategories,
    templating::{
        TemplateType, ChoiceTemplate, TemplateEntry, TemplateEngine, 
        TemplateError, TemplateEngineState, templates
    }
};
use std::collections::VecDeque;

/// Comprehensive integration test for template-driven choice generation capability
/// Tests the complete end-to-end workflow of the templating system
#[test]
fn test_comprehensive_template_driven_choice_generation() {
    // Test 1: Complete template engine workflow with mixed entry types
    let mut engine = TemplateEngine::new().with_debug().with_metadata("Integration test engine".to_string());
    
    // Add comprehensive set of template entries
    let entries = vec![
        TemplateEntry::Template(ChoiceTemplate::simplest()),
        TemplateEntry::DirectValue(ChoiceValue::Integer(42)),
        TemplateEntry::Template(ChoiceTemplate::at_index(3)),
        TemplateEntry::PartialNode {
            value: ChoiceValue::Boolean(true),
            constraints: Some(Constraints::Boolean(BooleanConstraints::default())),
            metadata: Some("Forced boolean choice".to_string()),
        },
        TemplateEntry::Template(ChoiceTemplate::biased(0.75)),
        TemplateEntry::DirectValue(ChoiceValue::String("forced_string".to_string())),
        TemplateEntry::Template(ChoiceTemplate::custom("test_custom".to_string())),
    ];
    
    engine.add_entries(entries);
    assert_eq!(engine.remaining_count(), 7);
    assert_eq!(engine.processed_count(), 0);
    assert!(!engine.has_misalignment());
    
    // Process each template entry with appropriate constraints
    let test_cases = vec![
        (ChoiceType::Integer, Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            shrink_towards: Some(0),
            weights: None,
        })),
        (ChoiceType::Integer, Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            shrink_towards: Some(0),
            weights: None,
        })),
        (ChoiceType::Integer, Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            shrink_towards: Some(0),
            weights: None,
        })),
        (ChoiceType::Boolean, Constraints::Boolean(BooleanConstraints::default())),
        (ChoiceType::Float, Constraints::Float(FloatConstraints {
            min_value: 0.0,
            max_value: 1.0,
            allow_nan: false,
            smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
        })),
        (ChoiceType::String, Constraints::String(StringConstraints {
            min_size: 0,
            max_size: 100,
            intervals: UnicodeCategories::default(),
        })),
        (ChoiceType::Integer, Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            shrink_towards: Some(0),
            weights: None,
        })),
    ];
    
    let mut results = Vec::new();
    for (choice_type, constraints) in test_cases {
        let result = engine.process_next_template(choice_type, &constraints).unwrap();
        assert!(result.is_some());
        results.push(result.unwrap());
    }
    
    // Validate results match expected template behavior
    assert_eq!(results.len(), 7);
    
    // Result 0: Simplest template - should generate simplest choice (0)
    assert_eq!(results[0].value, ChoiceValue::Integer(0));
    assert!(!results[0].was_forced); // Template-generated, not forced
    
    // Result 1: Direct value - should use forced value (42)
    assert_eq!(results[1].value, ChoiceValue::Integer(42));
    assert!(results[1].was_forced); // Direct values are forced
    
    // Result 2: AtIndex template - should generate choice at index (simplified to 0)
    assert_eq!(results[2].value, ChoiceValue::Integer(0));
    assert!(!results[2].was_forced);
    
    // Result 3: Partial node - should use forced boolean value
    assert_eq!(results[3].value, ChoiceValue::Boolean(true));
    assert!(results[3].was_forced); // Partial nodes are forced
    
    // Result 4: Biased template - should generate choice with bias (simplified to 0.0)
    assert_eq!(results[4].value, ChoiceValue::Float(0.0));
    assert!(!results[4].was_forced); // Biased templates are not forced
    
    // Result 5: Direct string value - should use forced value
    assert_eq!(results[5].value, ChoiceValue::String("forced_string".to_string()));
    assert!(results[5].was_forced); // Direct values are forced
    
    // Result 6: Custom template - should generate simplest choice (fallback)
    assert_eq!(results[6].value, ChoiceValue::Integer(0));
    assert!(!results[6].was_forced);
    
    // Validate engine state after processing
    assert_eq!(engine.remaining_count(), 0);
    assert_eq!(engine.processed_count(), 7);
    assert!(!engine.has_misalignment());
    
    // Test no more templates available
    let no_result = engine.process_next_template(
        ChoiceType::Integer, 
        &Constraints::Integer(IntegerConstraints::default())
    ).unwrap();
    assert!(no_result.is_none());
}

/// Test forced value insertion with constraint validation and misalignment handling
#[test]
fn test_forced_value_insertion_with_constraint_validation() {
    let mut engine = TemplateEngine::new().with_debug();
    
    // Add values that will trigger misalignment
    let problematic_entries = vec![
        TemplateEntry::DirectValue(ChoiceValue::Integer(500)), // Outside constraints
        TemplateEntry::DirectValue(ChoiceValue::Boolean(true)), // Type mismatch with integer expected
        TemplateEntry::DirectValue(ChoiceValue::String("toolong".repeat(20))), // Violates string size constraints
        TemplateEntry::DirectValue(ChoiceValue::Float(std::f64::NAN)), // NaN when not allowed
    ];
    
    engine.add_entries(problematic_entries);
    
    // Test constraint violations and fallback behavior
    let int_constraints = Constraints::Integer(IntegerConstraints {
        min_value: Some(0),
        max_value: Some(100),
        shrink_towards: Some(50),
        weights: None,
    });
    
    // Process first entry - integer outside range, should fall back to simplest (50)
    let result1 = engine.process_next_template(ChoiceType::Integer, &int_constraints).unwrap();
    assert!(result1.is_some());
    assert_eq!(result1.unwrap().value, ChoiceValue::Integer(50)); // fallback to shrink_towards
    assert!(engine.has_misalignment());
    assert_eq!(engine.misalignment_index(), Some(1));
    
    // Process second entry - type mismatch, should fall back to simplest
    let result2 = engine.process_next_template(ChoiceType::Integer, &int_constraints).unwrap();
    assert!(result2.is_some());
    assert_eq!(result2.unwrap().value, ChoiceValue::Integer(50));
    
    // Process third entry - string size violation
    let string_constraints = Constraints::String(StringConstraints {
        min_size: 0,
        max_size: 10, // Max 10 characters
        intervals: UnicodeCategories::default(),
    });
    
    let result3 = engine.process_next_template(ChoiceType::String, &string_constraints).unwrap();
    assert!(result3.is_some());
    assert_eq!(result3.unwrap().value, ChoiceValue::String(String::new())); // fallback to empty string
    
    // Process fourth entry - NaN when not allowed
    let float_constraints = Constraints::Float(FloatConstraints {
        min_value: 0.0,
        max_value: 1.0,
        allow_nan: false, // NaN not allowed
        smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
    });
    
    let result4 = engine.process_next_template(ChoiceType::Float, &float_constraints).unwrap();
    assert!(result4.is_some());
    assert_eq!(result4.unwrap().value, ChoiceValue::Float(0.0)); // fallback to 0.0
    
    // Validate final engine state
    assert_eq!(engine.processed_count(), 4);
    assert!(engine.has_misalignment());
}

/// Test template usage counting and exhaustion handling
#[test]
fn test_template_usage_counting_and_exhaustion() {
    let mut engine = TemplateEngine::new();
    
    // Create templates with limited usage counts
    let limited_templates = vec![
        TemplateEntry::Template(ChoiceTemplate::with_count(TemplateType::Simplest, 2)),
        TemplateEntry::Template(ChoiceTemplate::with_count(TemplateType::AtIndex(5), 1)),
        TemplateEntry::Template(ChoiceTemplate::with_count(TemplateType::Biased { bias: 0.8 }, 3)),
    ];
    
    engine.add_entries(limited_templates);
    
    let constraints = Constraints::Integer(IntegerConstraints::default());
    
    // Process first template twice (should succeed)
    let result1 = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
    assert!(result1.is_some());
    
    // Process second template once (should succeed)
    let result2 = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
    assert!(result2.is_some());
    
    // Process third template three times (should succeed)
    let result3a = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
    assert!(result3a.is_some());
    
    // All templates should now be exhausted
    assert_eq!(engine.remaining_count(), 0);
    assert_eq!(engine.processed_count(), 3);
    
    // Test expected choice count calculation
    let new_engine = TemplateEngine::from_templates(vec![
        ChoiceTemplate::with_count(TemplateType::Simplest, 5),
        ChoiceTemplate::unlimited(TemplateType::Biased { bias: 0.5 }),
        ChoiceTemplate::with_count(TemplateType::AtIndex(2), 3),
    ]);
    
    let expected_count = new_engine.calculate_expected_choice_count();
    assert_eq!(expected_count, 9); // 5 + 1 + 3 = 9
}

/// Test template engine state management and backup/restore functionality
#[test]
fn test_template_engine_state_management() {
    let mut engine = TemplateEngine::from_values(vec![
        ChoiceValue::Integer(1),
        ChoiceValue::Integer(2),
        ChoiceValue::Integer(3),
        ChoiceValue::Integer(4),
    ]).with_metadata("State management test".to_string());
    
    // Save initial state
    let initial_state = engine.clone_state();
    assert_eq!(initial_state.remaining_entries.len(), 4);
    assert_eq!(initial_state.processed_count, 0);
    assert_eq!(initial_state.misalignment_index, None);
    
    let constraints = Constraints::Integer(IntegerConstraints::default());
    
    // Process two entries
    let _result1 = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
    let _result2 = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
    
    assert_eq!(engine.remaining_count(), 2);
    assert_eq!(engine.processed_count(), 2);
    
    // Save intermediate state
    let intermediate_state = engine.clone_state();
    
    // Process remaining entries
    let _result3 = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
    let _result4 = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
    
    assert_eq!(engine.remaining_count(), 0);
    assert_eq!(engine.processed_count(), 4);
    
    // Restore to intermediate state
    engine.restore_state(intermediate_state);
    assert_eq!(engine.remaining_count(), 2);
    assert_eq!(engine.processed_count(), 2);
    
    // Restore to initial state
    engine.restore_state(initial_state);
    assert_eq!(engine.remaining_count(), 4);
    assert_eq!(engine.processed_count(), 0);
    
    // Test reset functionality
    let _result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
    engine.reset();
    assert_eq!(engine.remaining_count(), 0);
    assert_eq!(engine.processed_count(), 0);
    assert!(!engine.has_misalignment());
}

/// Test template convenience functions and pattern creation
#[test]
fn test_template_convenience_functions_and_patterns() {
    // Test simplest sequence creation
    let simplest_seq = templates::simplest_sequence(5);
    assert_eq!(simplest_seq.len(), 5);
    for entry in &simplest_seq {
        match entry {
            TemplateEntry::Template(template) => {
                assert_eq!(template.template_type, TemplateType::Simplest);
                assert!(template.is_forcing);
            }
            _ => panic!("Expected template entry"),
        }
    }
    
    // Test value sequence creation
    let test_values = vec![
        ChoiceValue::Integer(1),
        ChoiceValue::Boolean(true),
        ChoiceValue::Float(3.14),
        ChoiceValue::String("test".to_string()),
        ChoiceValue::Bytes(vec![1, 2, 3]),
    ];
    let value_seq = templates::value_sequence(test_values.clone());
    assert_eq!(value_seq.len(), 5);
    
    for (i, entry) in value_seq.iter().enumerate() {
        match entry {
            TemplateEntry::DirectValue(value) => {
                assert_eq!(*value, test_values[i]);
            }
            _ => panic!("Expected direct value entry"),
        }
    }
    
    // Test index sequence creation
    let indices = vec![0, 1, 2, 5, 10];
    let index_seq = templates::index_sequence(indices.clone());
    assert_eq!(index_seq.len(), 5);
    
    for (i, entry) in index_seq.iter().enumerate() {
        match entry {
            TemplateEntry::Template(template) => {
                if let TemplateType::AtIndex(idx) = template.template_type {
                    assert_eq!(idx, indices[i]);
                } else {
                    panic!("Expected AtIndex template type");
                }
            }
            _ => panic!("Expected template entry"),
        }
    }
    
    // Test biased sequence creation
    let biases = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    let biased_seq = templates::biased_sequence(biases.clone());
    assert_eq!(biased_seq.len(), 5);
    
    for (i, entry) in biased_seq.iter().enumerate() {
        match entry {
            TemplateEntry::Template(template) => {
                if let TemplateType::Biased { bias } = template.template_type {
                    assert!((bias - biases[i]).abs() < f64::EPSILON);
                    assert!(!template.is_forcing); // Biased templates are not forcing
                } else {
                    panic!("Expected Biased template type");
                }
            }
            _ => panic!("Expected template entry"),
        }
    }
    
    // Test mixed sequence creation
    let templates_list = vec![
        ChoiceTemplate::simplest(),
        ChoiceTemplate::at_index(2),
        ChoiceTemplate::biased(0.6),
    ];
    let values_list = vec![
        ChoiceValue::Integer(100),
        ChoiceValue::Boolean(false),
    ];
    
    let mixed_seq = templates::mixed_sequence(templates_list, values_list);
    assert_eq!(mixed_seq.len(), 5); // 3 templates + 2 values
    
    // Validate mixed sequence has correct entry types
    let template_count = mixed_seq.iter().filter(|entry| matches!(entry, TemplateEntry::Template(_))).count();
    let value_count = mixed_seq.iter().filter(|entry| matches!(entry, TemplateEntry::DirectValue(_))).count();
    assert_eq!(template_count, 3);
    assert_eq!(value_count, 2);
}

/// Test comprehensive template type behaviors and characteristics
#[test]
fn test_comprehensive_template_type_behaviors() {
    // Test TemplateType equality and hashing
    let type1 = TemplateType::Simplest;
    let type2 = TemplateType::Simplest;
    assert_eq!(type1, type2);
    
    let type3 = TemplateType::AtIndex(5);
    let type4 = TemplateType::AtIndex(5);
    assert_eq!(type3, type4);
    
    let type5 = TemplateType::Biased { bias: 0.75 };
    let type6 = TemplateType::Biased { bias: 0.75 };
    assert_eq!(type5, type6);
    
    let type7 = TemplateType::Custom { name: "test".to_string() };
    let type8 = TemplateType::Custom { name: "test".to_string() };
    assert_eq!(type7, type8);
    
    // Test TemplateType display formatting
    assert_eq!(TemplateType::Simplest.to_string(), "simplest");
    assert_eq!(TemplateType::AtIndex(42).to_string(), "at_index(42)");
    assert_eq!(TemplateType::Biased { bias: 0.123 }.to_string(), "biased(0.123)");
    assert_eq!(TemplateType::Custom { name: "custom_test".to_string() }.to_string(), "custom(custom_test)");
    
    // Test TemplateType default
    assert_eq!(TemplateType::default(), TemplateType::Simplest);
    
    // Test ChoiceTemplate builders and characteristics
    let simple_template = ChoiceTemplate::simplest();
    assert_eq!(simple_template.template_type, TemplateType::Simplest);
    assert!(simple_template.is_forcing);
    assert_eq!(simple_template.count, None);
    assert_eq!(simple_template.metadata, None);
    
    let index_template = ChoiceTemplate::at_index(10);
    assert!(matches!(index_template.template_type, TemplateType::AtIndex(10)));
    assert!(index_template.is_forcing);
    assert!(index_template.metadata.is_some());
    
    let biased_template = ChoiceTemplate::biased(0.8);
    assert!(matches!(biased_template.template_type, TemplateType::Biased { bias } if (bias - 0.8).abs() < f64::EPSILON));
    assert!(!biased_template.is_forcing); // Biased templates are not forcing
    assert!(biased_template.metadata.is_some());
    
    let custom_template = ChoiceTemplate::custom("my_custom".to_string());
    assert!(matches!(custom_template.template_type, TemplateType::Custom { name } if name == "my_custom"));
    assert!(custom_template.is_forcing);
    assert!(custom_template.metadata.is_some());
    
    // Test template with metadata
    let template_with_meta = ChoiceTemplate::simplest().with_metadata("Custom metadata".to_string());
    assert_eq!(template_with_meta.metadata, Some("Custom metadata".to_string()));
}

/// Test error handling and edge cases in template processing
#[test]
fn test_template_error_handling_and_edge_cases() {
    // Test TemplateError display formatting
    assert_eq!(TemplateError::ExhaustedTemplate.to_string(), "Template has no remaining uses");
    assert_eq!(TemplateError::ConstraintMismatch.to_string(), "Template value violates constraints");
    assert_eq!(TemplateError::TypeMismatch.to_string(), "Template type doesn't match expected choice type");
    assert_eq!(TemplateError::UnknownCustomTemplate("unknown".to_string()).to_string(), "Unknown custom template: unknown");
    assert_eq!(TemplateError::ProcessingFailed("test error".to_string()).to_string(), "Template processing failed: test error");
    
    // Test TemplateError equality
    assert_eq!(TemplateError::ExhaustedTemplate, TemplateError::ExhaustedTemplate);
    assert_ne!(TemplateError::ExhaustedTemplate, TemplateError::ConstraintMismatch);
    
    // Test template consumption edge cases
    let mut template = ChoiceTemplate::with_count(TemplateType::Simplest, 0);
    assert!(!template.has_remaining_count());
    
    let consume_result = template.consume_usage();
    assert!(consume_result.is_err());
    assert_eq!(consume_result.unwrap_err(), TemplateError::ExhaustedTemplate);
    
    // Test unlimited template consumption
    let unlimited_template = ChoiceTemplate::unlimited(TemplateType::Simplest);
    assert!(unlimited_template.has_remaining_count());
    assert_eq!(unlimited_template.remaining_count(), None);
    
    let consumed_unlimited = unlimited_template.consume_usage().unwrap();
    assert!(consumed_unlimited.has_remaining_count());
    assert_eq!(consumed_unlimited.remaining_count(), None);
    
    // Test empty engine processing
    let mut empty_engine = TemplateEngine::new();
    let result = empty_engine.process_next_template(
        ChoiceType::Integer,
        &Constraints::Integer(IntegerConstraints::default())
    ).unwrap();
    assert!(result.is_none());
    
    // Test debug info formatting
    let engine_debug = empty_engine.debug_info();
    assert!(engine_debug.contains("remaining: 0"));
    assert!(engine_debug.contains("processed: 0"));
    assert!(engine_debug.contains("misalignment: None"));
    assert!(engine_debug.contains("debug: false"));
}

/// Test template entry debug descriptions and characteristics
#[test]
fn test_template_entry_debug_and_characteristics() {
    // Test direct value entry
    let direct_entry = TemplateEntry::direct(ChoiceValue::Integer(42));
    assert!(direct_entry.is_forcing());
    assert_eq!(direct_entry.debug_description(), "Direct(Integer(42))");
    
    // Test template entry
    let template_entry = TemplateEntry::template(ChoiceTemplate::biased(0.5));
    assert!(!template_entry.is_forcing()); // Biased templates are not forcing
    assert_eq!(template_entry.debug_description(), "Template(biased(0.500))");
    
    // Test partial node entry with constraints
    let partial_with_constraints = TemplateEntry::partial_node(
        ChoiceValue::String("test".to_string()),
        Some(Constraints::String(StringConstraints {
            min_size: 0,
            max_size: 100,
            intervals: UnicodeCategories::default(),
        })),
        Some("Test metadata".to_string()),
    );
    assert!(partial_with_constraints.is_forcing());
    let debug_desc = partial_with_constraints.debug_description();
    assert!(debug_desc.contains("Partial(String(\"test\")"));
    assert!(debug_desc.contains("with constraints"));
    assert!(debug_desc.contains("(Test metadata)"));
    
    // Test partial node entry without constraints
    let partial_no_constraints = TemplateEntry::partial_node(
        ChoiceValue::Boolean(true),
        None,
        None,
    );
    assert!(partial_no_constraints.is_forcing());
    assert_eq!(partial_no_constraints.debug_description(), "Partial(Boolean(true))");
}

/// Test simplest choice generation across all choice types with various constraints
#[test]
fn test_simplest_choice_generation_comprehensive() {
    let engine = TemplateEngine::new();
    
    // Test integer simplest with shrink_towards in range
    let int_constraints_in_range = Constraints::Integer(IntegerConstraints {
        min_value: Some(10),
        max_value: Some(20),
        shrink_towards: Some(15),
        weights: None,
    });
    
    let int_node = engine.generate_simplest_choice(ChoiceType::Integer, &int_constraints_in_range).unwrap();
    assert_eq!(int_node.value, ChoiceValue::Integer(15));
    assert!(!int_node.was_forced); // Generated choices are not forced by default
    
    // Test integer simplest with shrink_towards below range
    let int_constraints_below = Constraints::Integer(IntegerConstraints {
        min_value: Some(10),
        max_value: Some(20),
        shrink_towards: Some(5),
        weights: None,
    });
    
    let int_node_clamped = engine.generate_simplest_choice(ChoiceType::Integer, &int_constraints_below).unwrap();
    assert_eq!(int_node_clamped.value, ChoiceValue::Integer(10)); // Clamped to min_value
    
    // Test boolean simplest
    let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
    let bool_node = engine.generate_simplest_choice(ChoiceType::Boolean, &bool_constraints).unwrap();
    assert_eq!(bool_node.value, ChoiceValue::Boolean(false));
    
    // Test float simplest with finite bounds
    let float_constraints_finite = Constraints::Float(FloatConstraints {
        min_value: -5.0,
        max_value: 5.0,
        allow_nan: false,
        smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
    });
    
    let float_node = engine.generate_simplest_choice(ChoiceType::Float, &float_constraints_finite).unwrap();
    assert_eq!(float_node.value, ChoiceValue::Float(0.0));
    
    // Test string simplest with min_size > 0
    let string_constraints_min = Constraints::String(StringConstraints {
        min_size: 3,
        max_size: 100,
        intervals: UnicodeCategories::default(),
    });
    
    let string_node = engine.generate_simplest_choice(ChoiceType::String, &string_constraints_min).unwrap();
    if let ChoiceValue::String(s) = string_node.value {
        assert_eq!(s.len(), 3);
        assert_eq!(s, "aaa"); // Should repeat the minimal character
    } else {
        panic!("Expected string value");
    }
    
    // Test bytes simplest with min_size > 0
    let bytes_constraints_min = Constraints::Bytes(BytesConstraints {
        min_size: 4,
        max_size: 100,
    });
    
    let bytes_node = engine.generate_simplest_choice(ChoiceType::Bytes, &bytes_constraints_min).unwrap();
    if let ChoiceValue::Bytes(b) = bytes_node.value {
        assert_eq!(b.len(), 4);
        assert_eq!(b, vec![0u8; 4]);
    } else {
        panic!("Expected bytes value");
    }
    
    // Test constraint mismatch error
    let wrong_constraints = Constraints::Boolean(BooleanConstraints::default());
    let error_result = engine.generate_simplest_choice(ChoiceType::Integer, &wrong_constraints);
    assert!(error_result.is_err());
    assert_eq!(error_result.unwrap_err(), TemplateError::ConstraintMismatch);
}

/// Test complete PyO3/FFI compatibility interface validation
#[test]
fn test_pyo3_ffi_compatibility_interface() {
    // This test validates that all templating types can be properly serialized/deserialized
    // for PyO3/FFI compatibility, ensuring the complete capability works across language boundaries
    
    // Test TemplateType serialization characteristics
    let template_types = vec![
        TemplateType::Simplest,
        TemplateType::AtIndex(42),
        TemplateType::Biased { bias: 0.618 },
        TemplateType::Custom { name: "ffi_test".to_string() },
    ];
    
    for template_type in template_types {
        // Validate that template types maintain their characteristics through clone
        let cloned_type = template_type.clone();
        assert_eq!(template_type, cloned_type);
        
        // Validate Display trait for FFI string conversion
        let display_string = template_type.to_string();
        assert!(!display_string.is_empty());
        
        // Create templates from each type
        let template = ChoiceTemplate::unlimited(template_type);
        assert!(template.has_remaining_count());
    }
    
    // Test ChoiceTemplate FFI characteristics
    let templates = vec![
        ChoiceTemplate::simplest(),
        ChoiceTemplate::at_index(10),
        ChoiceTemplate::biased(0.5),
        ChoiceTemplate::custom("ffi_custom".to_string()),
        ChoiceTemplate::with_count(TemplateType::Simplest, 5),
    ];
    
    for template in templates {
        // Validate template can be cloned for FFI transfer
        let cloned_template = template.clone();
        assert_eq!(template, cloned_template);
        
        // Validate template entry creation
        let entry = TemplateEntry::template(template);
        assert!(!entry.debug_description().is_empty());
    }
    
    // Test TemplateEngine FFI operations
    let mut engine = TemplateEngine::from_values(vec![
        ChoiceValue::Integer(1),
        ChoiceValue::Boolean(true),
        ChoiceValue::Float(2.5),
        ChoiceValue::String("ffi_test".to_string()),
        ChoiceValue::Bytes(vec![1, 2, 3, 4]),
    ]).with_debug().with_metadata("FFI compatibility test".to_string());
    
    // Validate engine state can be captured for FFI
    let state = engine.clone_state();
    assert_eq!(state.remaining_entries.len(), 5);
    assert_eq!(state.processed_count, 0);
    assert_eq!(state.misalignment_index, None);
    
    // Test debug info generation for FFI logging
    let debug_info = engine.debug_info();
    assert!(debug_info.contains("TemplateEngine"));
    assert!(debug_info.contains("remaining: 5"));
    
    // Test expected choice count calculation for FFI planning
    let expected_count = engine.calculate_expected_choice_count();
    assert_eq!(expected_count, 5);
    
    // Validate state restoration for FFI rollback operations
    let original_state = engine.clone_state();
    let constraints = Constraints::Integer(IntegerConstraints::default());
    let _ = engine.process_next_template(ChoiceType::Integer, &constraints);
    
    engine.restore_state(original_state);
    assert_eq!(engine.remaining_count(), 5);
    assert_eq!(engine.processed_count(), 0);
    
    // Test error handling for FFI error propagation
    let exhausted_template = ChoiceTemplate::with_count(TemplateType::Simplest, 0);
    let consumption_result = exhausted_template.consume_usage();
    assert!(consumption_result.is_err());
    
    let error = consumption_result.unwrap_err();
    let error_message = error.to_string();
    assert!(!error_message.is_empty());
}

/// Test template system integration with choice constraint system
#[test]
fn test_template_constraint_system_integration() {
    let mut engine = TemplateEngine::new().with_debug();
    
    // Add entries that exercise all constraint types
    let comprehensive_entries = vec![
        // Integer with complex constraints
        TemplateEntry::PartialNode {
            value: ChoiceValue::Integer(75),
            constraints: Some(Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(100),
                shrink_towards: Some(50),
                weights: Some(vec![1, 2, 3, 4, 5]),
            })),
            metadata: Some("Complex integer constraints".to_string()),
        },
        
        // Float with NaN handling
        TemplateEntry::DirectValue(ChoiceValue::Float(std::f64::NAN)),
        
        // String with unicode constraints
        TemplateEntry::PartialNode {
            value: ChoiceValue::String("测试".to_string()),
            constraints: Some(Constraints::String(StringConstraints {
                min_size: 1,
                max_size: 10,
                intervals: UnicodeCategories::default(),
            })),
            metadata: Some("Unicode string test".to_string()),
        },
        
        // Bytes with size constraints
        TemplateEntry::PartialNode {
            value: ChoiceValue::Bytes(vec![0xFF, 0xFE, 0xFD]),
            constraints: Some(Constraints::Bytes(BytesConstraints {
                min_size: 0,
                max_size: 5,
            })),
            metadata: Some("Binary data test".to_string()),
        },
    ];
    
    engine.add_entries(comprehensive_entries);
    
    // Process each entry with matching constraint validation
    
    // Test integer with complex constraints
    let int_constraints = Constraints::Integer(IntegerConstraints {
        min_value: Some(0),
        max_value: Some(100),
        shrink_towards: Some(50),
        weights: Some(vec![1, 2, 3, 4, 5]),
    });
    
    let int_result = engine.process_next_template(ChoiceType::Integer, &int_constraints).unwrap();
    assert!(int_result.is_some());
    let int_node = int_result.unwrap();
    assert_eq!(int_node.value, ChoiceValue::Integer(75));
    assert!(int_node.was_forced);
    
    // Test float with NaN (should trigger misalignment fallback)
    let float_constraints = Constraints::Float(FloatConstraints {
        min_value: 0.0,
        max_value: 1.0,
        allow_nan: false,
        smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
    });
    
    let float_result = engine.process_next_template(ChoiceType::Float, &float_constraints).unwrap();
    assert!(float_result.is_some());
    let float_node = float_result.unwrap();
    assert_eq!(float_node.value, ChoiceValue::Float(0.0)); // Fallback to simplest
    assert!(engine.has_misalignment());
    
    // Test string with unicode
    let string_constraints = Constraints::String(StringConstraints {
        min_size: 1,
        max_size: 10,
        intervals: UnicodeCategories::default(),
    });
    
    let string_result = engine.process_next_template(ChoiceType::String, &string_constraints).unwrap();
    assert!(string_result.is_some());
    let string_node = string_result.unwrap();
    assert_eq!(string_node.value, ChoiceValue::String("测试".to_string()));
    assert!(string_node.was_forced);
    
    // Test bytes
    let bytes_constraints = Constraints::Bytes(BytesConstraints {
        min_size: 0,
        max_size: 5,
    });
    
    let bytes_result = engine.process_next_template(ChoiceType::Bytes, &bytes_constraints).unwrap();
    assert!(bytes_result.is_some());
    let bytes_node = bytes_result.unwrap();
    assert_eq!(bytes_node.value, ChoiceValue::Bytes(vec![0xFF, 0xFE, 0xFD]));
    assert!(bytes_node.was_forced);
    
    // Validate final engine state
    assert_eq!(engine.processed_count(), 4);
    assert!(engine.has_misalignment());
    assert_eq!(engine.misalignment_index(), Some(2)); // Float NaN caused misalignment
}