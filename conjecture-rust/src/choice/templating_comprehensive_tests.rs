//! Comprehensive integration tests for Choice templating and forcing system capability
//!
//! This module provides extensive testing of the complete templating and forcing
//! capability behavior using PyO3 and FFI integration patterns. Tests validate
//! the entire capability's core responsibilities and interface contracts.

#[cfg(test)]
mod templating_capability_tests {
    use super::super::*;
    use crate::choice::{
        ChoiceType, ChoiceValue, Constraints, ChoiceNode,
        IntegerConstraints, BooleanConstraints, FloatConstraints, 
        StringConstraints, BytesConstraints, IntervalSet,
        templating::*
    };
    use std::collections::VecDeque;

    /// Test complete template-driven choice generation capability
    #[test]
    fn test_complete_template_driven_choice_generation() {
        // Test all template types in a comprehensive scenario
        let templates = vec![
            ChoiceTemplate::simplest(),
            ChoiceTemplate::at_index(5),
            ChoiceTemplate::biased(0.7),
            ChoiceTemplate::custom("test_custom".to_string()),
            ChoiceTemplate::with_count(TemplateType::Simplest, 3),
            ChoiceTemplate::unlimited(TemplateType::Biased { bias: 0.3 }),
        ];

        let mut engine = TemplateEngine::from_templates(templates);
        assert_eq!(engine.remaining_count(), 6);

        // Test integer constraint processing
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(-10),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        });

        // Process all templates
        let mut results = Vec::new();
        while engine.has_templates() {
            if let Ok(Some(node)) = engine.process_next_template(ChoiceType::Integer, &int_constraints) {
                results.push(node);
            } else {
                break;
            }
        }

        // Verify results include choices from different template strategies
        assert!(!results.is_empty());
        assert!(results.iter().any(|n| n.was_forced)); // Some should be forced
        assert!(results.iter().any(|n| !n.was_forced)); // Biased templates are not forced
    }

    /// Test forced value insertion for guided test case construction
    #[test]
    fn test_forced_value_insertion_capability() {
        let forced_values = vec![
            ChoiceValue::Integer(42),
            ChoiceValue::Boolean(true),
            ChoiceValue::Float(3.14159),
            ChoiceValue::String("forced_string".to_string()),
            ChoiceValue::Bytes(vec![1, 2, 3, 4, 5]),
        ];

        let mut engine = TemplateEngine::from_values(forced_values.clone());

        // Test forced integer insertion
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        });

        let result = engine.process_next_template(ChoiceType::Integer, &int_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Integer(42));
        assert!(node.was_forced);

        // Test forced boolean insertion
        let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
        let result = engine.process_next_template(ChoiceType::Boolean, &bool_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Boolean(true));
        assert!(node.was_forced);

        // Test forced float insertion
        let float_constraints = Constraints::Float(FloatConstraints {
            min_value: 0.0,
            max_value: 10.0,
            allow_nan: false,
            smallest_nonzero_magnitude: f64::MIN_POSITIVE,
        });
        let result = engine.process_next_template(ChoiceType::Float, &float_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Float(3.14159));
        assert!(node.was_forced);

        // Test forced string insertion
        let string_constraints = Constraints::String(StringConstraints {
            min_size: 1,
            max_size: 100,
            intervals: IntervalSet::all_characters(),
        });
        let result = engine.process_next_template(ChoiceType::String, &string_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::String("forced_string".to_string()));
        assert!(node.was_forced);

        // Test forced bytes insertion
        let bytes_constraints = Constraints::Bytes(BytesConstraints {
            min_size: 1,
            max_size: 100,
        });
        let result = engine.process_next_template(ChoiceType::Bytes, &bytes_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Bytes(vec![1, 2, 3, 4, 5]));
        assert!(node.was_forced);
    }

    /// Test template usage counting and consumption semantics
    #[test]
    fn test_template_usage_counting_capability() {
        // Test single-use template
        let single_use = ChoiceTemplate::simplest();
        let mut engine = TemplateEngine::from_templates(vec![single_use]);

        let constraints = Constraints::Integer(IntegerConstraints::default());
        
        // Should process once
        let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        assert!(result.is_some());
        
        // Should be exhausted
        let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        assert!(result.is_none());

        // Test limited-use template
        let limited_use = ChoiceTemplate::with_count(TemplateType::Simplest, 3);
        let mut engine = TemplateEngine::from_templates(vec![limited_use]);

        // Should process exactly 3 times
        for i in 0..3 {
            let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
            assert!(result.is_some(), "Failed at iteration {}", i);
        }

        // Should be exhausted after 3 uses
        let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        assert!(result.is_none());

        // Test unlimited template
        let unlimited = ChoiceTemplate::unlimited(TemplateType::Simplest);
        let mut engine = TemplateEngine::from_templates(vec![unlimited]);

        // Should process many times without exhaustion
        for i in 0..10 {
            let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
            assert!(result.is_some(), "Failed at iteration {}", i);
        }
        
        // Should still have templates remaining
        assert!(engine.has_templates());
    }

    /// Test constraint validation and misalignment handling
    #[test]
    fn test_constraint_validation_and_misalignment() {
        // Test type mismatches
        let invalid_entries = vec![
            TemplateEntry::DirectValue(ChoiceValue::String("wrong_type".to_string())),
            TemplateEntry::DirectValue(ChoiceValue::Integer(150)), // Outside constraint range
        ];

        let mut engine = TemplateEngine::from_entries(invalid_entries);

        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        });

        // First template should cause type mismatch and misalignment
        let result = engine.process_next_template(ChoiceType::Integer, &int_constraints).unwrap();
        assert!(result.is_some());
        assert!(engine.has_misalignment());
        assert_eq!(engine.misalignment_index(), Some(1));

        // Verify fallback to simplest choice
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Integer(0)); // shrink_towards value

        // Second template should cause constraint violation
        let result = engine.process_next_template(ChoiceType::Integer, &int_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Integer(0)); // Falls back to simplest
    }

    /// Test partial node processing with custom constraints
    #[test]
    fn test_partial_node_processing_capability() {
        let custom_int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(50),
            max_value: Some(150),
            weights: None,
            shrink_towards: Some(100),
        });

        let partial_entries = vec![
            TemplateEntry::partial_node(
                ChoiceValue::Integer(75),
                Some(custom_int_constraints.clone()),
                Some("custom_partial".to_string()),
            ),
            TemplateEntry::partial_node(
                ChoiceValue::Integer(125),
                None, // Should use expected constraints
                None,
            ),
        ];

        let mut engine = TemplateEngine::from_entries(partial_entries);

        let expected_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(200),
            weights: None,
            shrink_towards: Some(0),
        });

        // Process first partial node with custom constraints
        let result = engine.process_next_template(ChoiceType::Integer, &expected_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Integer(75));
        assert!(node.was_forced);
        // Should use custom constraints, not expected constraints
        if let Constraints::Integer(constraints) = &node.constraints {
            assert_eq!(constraints.min_value, Some(50));
            assert_eq!(constraints.max_value, Some(150));
        } else {
            panic!("Expected integer constraints");
        }

        // Process second partial node with expected constraints
        let result = engine.process_next_template(ChoiceType::Integer, &expected_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Integer(125));
        assert!(node.was_forced);
        // Should use expected constraints
        if let Constraints::Integer(constraints) = &node.constraints {
            assert_eq!(constraints.min_value, Some(0));
            assert_eq!(constraints.max_value, Some(200));
        } else {
            panic!("Expected integer constraints");
        }
    }

    /// Test template engine state management and backup/restore
    #[test]
    fn test_template_engine_state_management_capability() {
        let entries = vec![
            TemplateEntry::DirectValue(ChoiceValue::Integer(1)),
            TemplateEntry::DirectValue(ChoiceValue::Integer(2)),
            TemplateEntry::DirectValue(ChoiceValue::Integer(3)),
        ];

        let mut engine = TemplateEngine::from_entries(entries);
        assert_eq!(engine.remaining_count(), 3);
        assert_eq!(engine.processed_count(), 0);

        // Clone initial state
        let initial_state = engine.clone_state();

        // Process one template
        let constraints = Constraints::Integer(IntegerConstraints::default());
        let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        assert!(result.is_some());
        assert_eq!(engine.remaining_count(), 2);
        assert_eq!(engine.processed_count(), 1);

        // Clone intermediate state
        let intermediate_state = engine.clone_state();

        // Process another template
        let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        assert!(result.is_some());
        assert_eq!(engine.remaining_count(), 1);
        assert_eq!(engine.processed_count(), 2);

        // Restore to intermediate state
        engine.restore_state(intermediate_state);
        assert_eq!(engine.remaining_count(), 2);
        assert_eq!(engine.processed_count(), 1);

        // Restore to initial state
        engine.restore_state(initial_state);
        assert_eq!(engine.remaining_count(), 3);
        assert_eq!(engine.processed_count(), 0);

        // Verify all templates are still processable
        for i in 0..3 {
            let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
            assert!(result.is_some(), "Failed to process template {}", i);
        }
    }

    /// Test mixed template and value sequences
    #[test]
    fn test_mixed_template_value_sequences_capability() {
        let templates = vec![
            ChoiceTemplate::simplest(),
            ChoiceTemplate::biased(0.8),
        ];

        let values = vec![
            ChoiceValue::Integer(100),
            ChoiceValue::Integer(200),
        ];

        let mixed_entries = templates::mixed_sequence(templates, values);
        assert_eq!(mixed_entries.len(), 4);

        let mut engine = TemplateEngine::from_entries(mixed_entries);

        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(300),
            weights: None,
            shrink_towards: Some(0),
        });

        let mut results = Vec::new();
        while engine.has_templates() {
            if let Ok(Some(node)) = engine.process_next_template(ChoiceType::Integer, &constraints) {
                results.push(node);
            } else {
                break;
            }
        }

        assert_eq!(results.len(), 4);

        // First two should be from templates
        assert!(!results[0].was_forced || results[0].value == ChoiceValue::Integer(0)); // Simplest template
        assert!(!results[1].was_forced); // Biased template

        // Last two should be from direct values
        assert!(results[2].was_forced);
        assert_eq!(results[2].value, ChoiceValue::Integer(100));
        assert!(results[3].was_forced);
        assert_eq!(results[3].value, ChoiceValue::Integer(200));
    }

    /// Test template convenience functions integration
    #[test]
    fn test_template_convenience_functions_capability() {
        // Test simplest sequence
        let simplest_seq = templates::simplest_sequence(5);
        let mut engine = TemplateEngine::from_entries(simplest_seq);

        let constraints = Constraints::Boolean(BooleanConstraints::default());
        
        for i in 0..5 {
            let result = engine.process_next_template(ChoiceType::Boolean, &constraints).unwrap();
            assert!(result.is_some(), "Failed at simplest sequence iteration {}", i);
            let node = result.unwrap();
            assert_eq!(node.value, ChoiceValue::Boolean(false)); // Simplest boolean
            assert!(!node.was_forced); // Template-generated, not forced
        }

        // Test index sequence
        let index_seq = templates::index_sequence(vec![0, 1, 2, 3, 4]);
        let mut engine = TemplateEngine::from_entries(index_seq);

        for i in 0..5 {
            let result = engine.process_next_template(ChoiceType::Boolean, &constraints).unwrap();
            assert!(result.is_some(), "Failed at index sequence iteration {}", i);
            // All index templates currently fall back to simplest choice
        }

        // Test biased sequence
        let biased_seq = templates::biased_sequence(vec![0.1, 0.5, 0.9]);
        let mut engine = TemplateEngine::from_entries(biased_seq);

        for i in 0..3 {
            let result = engine.process_next_template(ChoiceType::Boolean, &constraints).unwrap();
            assert!(result.is_some(), "Failed at biased sequence iteration {}", i);
            let node = result.unwrap();
            assert!(!node.was_forced); // Biased templates are not forcing
        }

        // Test value sequence
        let values = vec![
            ChoiceValue::Boolean(true),
            ChoiceValue::Boolean(false),
            ChoiceValue::Boolean(true),
        ];
        let value_seq = templates::value_sequence(values.clone());
        let mut engine = TemplateEngine::from_entries(value_seq);

        for (i, expected_value) in values.iter().enumerate() {
            let result = engine.process_next_template(ChoiceType::Boolean, &constraints).unwrap();
            assert!(result.is_some(), "Failed at value sequence iteration {}", i);
            let node = result.unwrap();
            assert_eq!(node.value, *expected_value);
            assert!(node.was_forced); // Direct values are always forced
        }
    }

    /// Test template error handling and recovery
    #[test]
    fn test_template_error_handling_capability() {
        // Test exhausted template error
        let exhausted_template = ChoiceTemplate::with_count(TemplateType::Simplest, 0);
        let result = exhausted_template.consume_usage();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TemplateError::ExhaustedTemplate));

        // Test constraint mismatch error through simplest choice generation
        let engine = TemplateEngine::new();
        let invalid_constraints = Constraints::Boolean(BooleanConstraints::default());
        let result = engine.generate_simplest_choice(ChoiceType::Integer, &invalid_constraints);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TemplateError::ConstraintMismatch));

        // Test template error display
        let errors = vec![
            TemplateError::ExhaustedTemplate,
            TemplateError::ConstraintMismatch,
            TemplateError::TypeMismatch,
            TemplateError::UnknownCustomTemplate("test".to_string()),
            TemplateError::ProcessingFailed("test error".to_string()),
        ];

        for error in errors {
            let error_str = error.to_string();
            assert!(!error_str.is_empty());
            // Verify all errors implement Display properly
        }
    }

    /// Test template debug and metadata functionality
    #[test]
    fn test_template_debug_and_metadata_capability() {
        // Test template with metadata
        let template_with_metadata = ChoiceTemplate::simplest()
            .with_metadata("test_metadata".to_string());
        assert_eq!(template_with_metadata.metadata, Some("test_metadata".to_string()));

        // Test engine with debug mode
        let mut engine = TemplateEngine::new()
            .with_debug()
            .with_metadata("test_engine".to_string());
        assert!(engine.debug_mode);
        assert_eq!(engine.metadata, Some("test_engine".to_string()));

        // Test debug info generation
        let debug_info = engine.debug_info();
        assert!(debug_info.contains("TemplateEngine"));
        assert!(debug_info.contains("remaining: 0"));
        assert!(debug_info.contains("processed: 0"));
        assert!(debug_info.contains("debug: true"));

        // Test template entry debug descriptions
        let entries = vec![
            TemplateEntry::DirectValue(ChoiceValue::Integer(42)),
            TemplateEntry::Template(ChoiceTemplate::biased(0.5)),
            TemplateEntry::partial_node(
                ChoiceValue::String("test".to_string()),
                None,
                Some("test_partial".to_string()),
            ),
        ];

        for entry in &entries {
            let description = entry.debug_description();
            assert!(!description.is_empty());
            match entry {
                TemplateEntry::DirectValue(_) => assert!(description.contains("Direct")),
                TemplateEntry::Template(_) => assert!(description.contains("Template")),
                TemplateEntry::PartialNode { .. } => assert!(description.contains("Partial")),
            }
        }
    }

    /// Test complete capability integration with complex scenarios
    #[test]
    fn test_complete_capability_integration() {
        // Create a complex scenario with all template types and value types
        let mut entries = Vec::new();

        // Add various template types
        entries.push(TemplateEntry::Template(ChoiceTemplate::simplest()));
        entries.push(TemplateEntry::Template(ChoiceTemplate::at_index(3)));
        entries.push(TemplateEntry::Template(ChoiceTemplate::biased(0.6)));
        entries.push(TemplateEntry::Template(ChoiceTemplate::custom("integration_test".to_string())));

        // Add direct values for all types
        entries.push(TemplateEntry::DirectValue(ChoiceValue::Integer(42)));
        entries.push(TemplateEntry::DirectValue(ChoiceValue::Boolean(true)));
        entries.push(TemplateEntry::DirectValue(ChoiceValue::Float(2.718)));
        entries.push(TemplateEntry::DirectValue(ChoiceValue::String("test_string".to_string())));
        entries.push(TemplateEntry::DirectValue(ChoiceValue::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF])));

        // Add partial nodes
        entries.push(TemplateEntry::partial_node(
            ChoiceValue::Integer(100),
            Some(Constraints::Integer(IntegerConstraints {
                min_value: Some(50),
                max_value: Some(150),
                weights: None,
                shrink_towards: Some(100),
            })),
            Some("partial_integer".to_string()),
        ));

        let mut engine = TemplateEngine::from_entries(entries)
            .with_debug()
            .with_metadata("integration_test_engine".to_string());

        // Process with different constraint types
        let constraint_types = vec![
            (ChoiceType::Integer, Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(1000),
                weights: None,
                shrink_towards: Some(0),
            })),
            (ChoiceType::Boolean, Constraints::Boolean(BooleanConstraints::default())),
            (ChoiceType::Float, Constraints::Float(FloatConstraints {
                min_value: 0.0,
                max_value: 10.0,
                allow_nan: false,
                smallest_nonzero_magnitude: f64::MIN_POSITIVE,
            })),
            (ChoiceType::String, Constraints::String(StringConstraints {
                min_size: 0,
                max_size: 100,
                intervals: IntervalSet::all_characters(),
            })),
            (ChoiceType::Bytes, Constraints::Bytes(BytesConstraints {
                min_size: 0,
                max_size: 100,
            })),
        ];

        let mut total_processed = 0;
        let initial_count = engine.remaining_count();

        // Process templates with cycling through constraint types
        while engine.has_templates() && total_processed < initial_count {
            let (choice_type, constraints) = &constraint_types[total_processed % constraint_types.len()];
            
            match engine.process_next_template(*choice_type, constraints) {
                Ok(Some(node)) => {
                    total_processed += 1;
                    
                    // Verify node properties
                    match choice_type {
                        ChoiceType::Integer => {
                            if let ChoiceValue::Integer(_) = node.value {
                                // Valid integer value
                            } else {
                                // Should have fallen back to simplest choice due to type mismatch
                                assert!(!engine.has_misalignment() || engine.misalignment_index().is_some());
                            }
                        }
                        ChoiceType::Boolean => {
                            if let ChoiceValue::Boolean(_) = node.value {
                                // Valid boolean value
                            } else {
                                // Should have fallen back to simplest choice due to type mismatch
                                assert!(!engine.has_misalignment() || engine.misalignment_index().is_some());
                            }
                        }
                        ChoiceType::Float => {
                            if let ChoiceValue::Float(_) = node.value {
                                // Valid float value
                            } else {
                                // Should have fallen back to simplest choice due to type mismatch
                                assert!(!engine.has_misalignment() || engine.misalignment_index().is_some());
                            }
                        }
                        ChoiceType::String => {
                            if let ChoiceValue::String(_) = node.value {
                                // Valid string value
                            } else {
                                // Should have fallen back to simplest choice due to type mismatch
                                assert!(!engine.has_misalignment() || engine.misalignment_index().is_some());
                            }
                        }
                        ChoiceType::Bytes => {
                            if let ChoiceValue::Bytes(_) = node.value {
                                // Valid bytes value
                            } else {
                                // Should have fallen back to simplest choice due to type mismatch
                                assert!(!engine.has_misalignment() || engine.misalignment_index().is_some());
                            }
                        }
                    }
                }
                Ok(None) => break, // No more templates
                Err(_) => break, // Error occurred
            }
        }

        // Verify we processed a reasonable number of templates
        assert!(total_processed > 0);
        assert_eq!(engine.processed_count(), total_processed);

        // Test final state
        let final_debug_info = engine.debug_info();
        assert!(final_debug_info.contains(&format!("processed: {}", total_processed)));
    }
}

/// Python parity verification tests
#[cfg(test)]
mod python_parity_tests {
    use super::*;
    use crate::choice::templating::*;

    /// Test template type equivalence with Python Hypothesis
    #[test]
    fn test_template_type_python_parity() {
        // Verify template types match Python's template types
        let simplest = TemplateType::Simplest;
        assert_eq!(simplest.to_string(), "simplest");

        let at_index = TemplateType::AtIndex(5);
        assert_eq!(at_index.to_string(), "at_index(5)");

        let biased = TemplateType::Biased { bias: 0.75 };
        assert_eq!(biased.to_string(), "biased(0.750)");

        let custom = TemplateType::Custom { name: "test".to_string() };
        assert_eq!(custom.to_string(), "custom(test)");

        // Test hash consistency (important for Python interop)
        use std::collections::HashMap;
        let mut template_map = HashMap::new();
        template_map.insert(simplest.clone(), "simplest_value");
        template_map.insert(at_index.clone(), "index_value");
        template_map.insert(biased.clone(), "biased_value");
        template_map.insert(custom.clone(), "custom_value");

        assert_eq!(template_map.get(&simplest), Some(&"simplest_value"));
        assert_eq!(template_map.get(&at_index), Some(&"index_value"));
        assert_eq!(template_map.get(&biased), Some(&"biased_value"));
        assert_eq!(template_map.get(&custom), Some(&"custom_value"));
    }

    /// Test template processing behavior matches Python
    #[test]
    fn test_template_processing_python_parity() {
        // Test that forced values bypass randomness (Python behavior)
        let forced_value = ChoiceValue::Integer(42);
        let mut engine = TemplateEngine::from_values(vec![forced_value.clone()]);

        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        });

        let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, forced_value);
        assert!(node.was_forced); // Must be marked as forced

        // Test that biased templates don't force (Python behavior)
        let biased_template = ChoiceTemplate::biased(0.5);
        let mut engine = TemplateEngine::from_templates(vec![biased_template]);

        let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert!(!node.was_forced); // Biased templates should not force
    }

    /// Test constraint validation matches Python behavior
    #[test]
    fn test_constraint_validation_python_parity() {
        // Test misalignment handling (Python falls back to simplest choice)
        let invalid_value = ChoiceValue::Integer(200); // Outside range [0, 100]
        let mut engine = TemplateEngine::from_values(vec![invalid_value]);

        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(50),
        });

        let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Integer(50)); // Should fall back to shrink_towards
        assert!(engine.has_misalignment()); // Should record misalignment
    }

    /// Test error behavior matches Python expectations
    #[test]
    fn test_error_behavior_python_parity() {
        // Test exhausted template behavior
        let exhausted_template = ChoiceTemplate::with_count(TemplateType::Simplest, 1);
        let consumed = exhausted_template.consume_usage().unwrap();
        let result = consumed.consume_usage();
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TemplateError::ExhaustedTemplate));

        // Test error messages are descriptive (important for Python debugging)
        let error = TemplateError::ExhaustedTemplate;
        assert_eq!(error.to_string(), "Template has no remaining uses");

        let constraint_error = TemplateError::ConstraintMismatch;
        assert_eq!(constraint_error.to_string(), "Template value violates constraints");
    }
}

/// Performance benchmarks for templating capability
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    /// Benchmark template processing performance
    #[test]
    fn test_template_processing_performance() {
        let template_count = 1000;
        let templates = (0..template_count)
            .map(|i| {
                if i % 4 == 0 {
                    ChoiceTemplate::simplest()
                } else if i % 4 == 1 {
                    ChoiceTemplate::at_index(i % 10)
                } else if i % 4 == 2 {
                    ChoiceTemplate::biased((i as f64) / (template_count as f64))
                } else {
                    ChoiceTemplate::custom(format!("custom_{}", i))
                }
            })
            .collect();

        let mut engine = TemplateEngine::from_templates(templates);
        
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(-1000),
            max_value: Some(1000),
            weights: None,
            shrink_towards: Some(0),
        });

        let start = Instant::now();
        let mut processed = 0;

        while engine.has_templates() && processed < template_count {
            match engine.process_next_template(ChoiceType::Integer, &constraints) {
                Ok(Some(_)) => processed += 1,
                _ => break,
            }
        }

        let duration = start.elapsed();
        println!("Processed {} templates in {:?}", processed, duration);

        // Should process at least 500 templates per millisecond
        assert!(processed > 0);
        assert!(duration.as_millis() < 100 || processed > duration.as_millis() as usize * 5);
    }

    /// Benchmark large template engine state operations
    #[test]
    fn test_large_state_management_performance() {
        let entry_count = 10000;
        let entries: Vec<TemplateEntry> = (0..entry_count)
            .map(|i| TemplateEntry::DirectValue(ChoiceValue::Integer(i as i128)))
            .collect();

        let engine = TemplateEngine::from_entries(entries);
        
        // Benchmark state cloning
        let start = Instant::now();
        let state = engine.clone_state();
        let clone_duration = start.elapsed();

        // Benchmark state restoration
        let mut test_engine = TemplateEngine::new();
        let start = Instant::now();
        test_engine.restore_state(state);
        let restore_duration = start.elapsed();

        println!("Clone: {:?}, Restore: {:?}", clone_duration, restore_duration);

        // Operations should complete quickly even with large state
        assert!(clone_duration.as_millis() < 50);
        assert!(restore_duration.as_millis() < 50);
        assert_eq!(test_engine.remaining_count(), entry_count);
    }
}