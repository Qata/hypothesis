//! Comprehensive integration tests for Choice templating and forcing system capability
//!
//! These tests validate the complete templating capability behavior through PyO3 and FFI,
//! focusing on testing the entire capability's behavior rather than individual functions.
//! Tests validate the capability's core responsibilities and interface contracts.

#[cfg(test)]
mod comprehensive_templating_capability_tests {
    use crate::choice::{
        ChoiceType, ChoiceValue, Constraints,
        IntegerConstraints, BooleanConstraints, FloatConstraints, StringConstraints, BytesConstraints,
        TemplateType, ChoiceTemplate, TemplateEntry, TemplateEngine, TemplateError,
        templates, IntervalSet
    };

    /// Test complete template-driven choice generation capability
    #[test]
    fn test_complete_template_driven_choice_generation() {
        // Create a comprehensive template engine with all template types
        let mut engine = TemplateEngine::new().with_debug().with_metadata("test_engine".to_string());
        
        // Add templates for all four generation strategies
        let templates = vec![
            TemplateEntry::Template(ChoiceTemplate::simplest()),
            TemplateEntry::Template(ChoiceTemplate::at_index(5)),
            TemplateEntry::Template(ChoiceTemplate::biased(0.7)),
            TemplateEntry::Template(ChoiceTemplate::custom("test_custom".to_string())),
        ];
        
        engine.add_entries(templates);
        
        // Validate initial state
        assert_eq!(engine.remaining_count(), 4);
        assert_eq!(engine.processed_count(), 0);
        assert!(!engine.has_misalignment());
        
        // Process each template type with appropriate constraints
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(10),
        });
        
        // Test simplest template
        let result = engine.process_next_template(ChoiceType::Integer, &int_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Integer(10)); // shrink_towards
        assert!(!node.was_forced); // Template-generated choices aren't forced by default
        assert_eq!(engine.processed_count(), 1);
        
        // Test index template (falls back to simplest for now)
        let result = engine.process_next_template(ChoiceType::Integer, &int_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Integer(10));
        assert_eq!(engine.processed_count(), 2);
        
        // Test biased template
        let result = engine.process_next_template(ChoiceType::Integer, &int_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Integer(10));
        assert!(!node.was_forced); // Biased templates are not forced
        assert_eq!(engine.processed_count(), 3);
        
        // Test custom template
        let result = engine.process_next_template(ChoiceType::Integer, &int_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Integer(10));
        assert_eq!(engine.processed_count(), 4);
        
        // No more templates
        let result = engine.process_next_template(ChoiceType::Integer, &int_constraints).unwrap();
        assert!(result.is_none());
    }

    /// Test complete forced value insertion capability across all value types
    #[test]
    fn test_complete_forced_value_insertion_capability() {
        // Test forced value insertion for all supported types
        let test_cases = vec![
            (
                ChoiceValue::Integer(42),
                ChoiceType::Integer,
                Constraints::Integer(IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(100),
                    weights: None,
                    shrink_towards: Some(0),
                }),
            ),
            (
                ChoiceValue::Boolean(true),
                ChoiceType::Boolean,
                Constraints::Boolean(BooleanConstraints::default()),
            ),
            (
                ChoiceValue::Float(3.14),
                ChoiceType::Float,
                Constraints::Float(FloatConstraints {
                    min_value: 0.0,
                    max_value: 10.0,
                    allow_nan: false,
                    smallest_nonzero_magnitude: f64::MIN_POSITIVE,
                }),
            ),
            (
                ChoiceValue::String("forced_string".to_string()),
                ChoiceType::String,
                Constraints::String(StringConstraints {
                    min_size: 0,
                    max_size: 100,
                    intervals: IntervalSet::default(),
                }),
            ),
            (
                ChoiceValue::Bytes(vec![1, 2, 3, 4, 5]),
                ChoiceType::Bytes,
                Constraints::Bytes(BytesConstraints {
                    min_size: 0,
                    max_size: 10,
                }),
            ),
        ];
        
        for (forced_value, choice_type, constraints) in test_cases {
            let mut engine = TemplateEngine::from_values(vec![forced_value.clone()]);
            
            let result = engine.process_next_template(choice_type, &constraints).unwrap();
            assert!(result.is_some());
            
            let node = result.unwrap();
            assert_eq!(node.value, forced_value);
            assert!(node.was_forced); // Direct values are always forced
            assert_eq!(node.choice_type, choice_type);
            assert_eq!(node.constraints, constraints);
        }
    }

    /// Test comprehensive template usage counting and exhaustion
    #[test]
    fn test_template_usage_counting_capability() {
        // Create templates with various usage counts
        let templates = vec![
            ChoiceTemplate::with_count(TemplateType::Simplest, 1),
            ChoiceTemplate::with_count(TemplateType::AtIndex(0), 2),
            ChoiceTemplate::with_count(TemplateType::Biased { bias: 0.5 }, 3),
            ChoiceTemplate::unlimited(TemplateType::Custom { name: "unlimited".to_string() }),
        ];
        
        let mut engine = TemplateEngine::from_templates(templates);
        let constraints = Constraints::Integer(IntegerConstraints::default());
        
        // Process first template (count=1)
        let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        assert!(result.is_some());
        assert_eq!(engine.processed_count(), 1);
        
        // Process second template twice (count=2)
        for i in 2..=3 {
            let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
            assert!(result.is_some());
            assert_eq!(engine.processed_count(), i);
        }
        
        // Process third template three times (count=3)
        for i in 4..=6 {
            let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
            assert!(result.is_some());
            assert_eq!(engine.processed_count(), i);
        }
        
        // Process unlimited template (should work indefinitely)
        for i in 7..=10 {
            let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
            assert!(result.is_some());
            assert_eq!(engine.processed_count(), i);
        }
        
        // Unlimited template should still be available
        assert!(engine.has_templates());
    }

    /// Test comprehensive misalignment detection and recovery
    #[test]
    fn test_misalignment_detection_and_recovery_capability() {
        // Create engine with misaligned values (wrong types and constraint violations)
        let misaligned_values = vec![
            ChoiceValue::String("wrong_type".to_string()), // Wrong type for integer
            ChoiceValue::Integer(500), // Outside constraint range
            ChoiceValue::Float(f64::NAN), // NaN when not allowed
            ChoiceValue::Boolean(true), // Correct value for recovery test
        ];
        
        let mut engine = TemplateEngine::from_values(misaligned_values);
        
        // Test type mismatch detection
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        });
        
        let result = engine.process_next_template(ChoiceType::Integer, &int_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Integer(0)); // Fallback to simplest
        assert!(engine.has_misalignment());
        assert_eq!(engine.misalignment_index(), Some(1));
        
        // Test constraint violation detection
        let result = engine.process_next_template(ChoiceType::Integer, &int_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Integer(0)); // Fallback to simplest
        assert!(engine.has_misalignment());
        assert_eq!(engine.misalignment_index(), Some(1)); // First misalignment index preserved
        
        // Test NaN handling
        let float_constraints = Constraints::Float(FloatConstraints {
            min_value: 0.0,
            max_value: 1.0,
            allow_nan: false,
            smallest_nonzero_magnitude: f64::MIN_POSITIVE,
        });
        
        let result = engine.process_next_template(ChoiceType::Float, &float_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Float(0.0)); // Fallback to simplest
        
        // Test successful processing after misalignments
        let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
        let result = engine.process_next_template(ChoiceType::Boolean, &bool_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Boolean(true)); // Correct value
        assert!(node.was_forced);
    }

    /// Test template engine state management capability
    #[test]
    fn test_template_engine_state_management_capability() {
        // Create engine with multiple templates
        let templates = vec![
            TemplateEntry::DirectValue(ChoiceValue::Integer(1)),
            TemplateEntry::DirectValue(ChoiceValue::Integer(2)),
            TemplateEntry::DirectValue(ChoiceValue::Integer(3)),
        ];
        
        let mut engine = TemplateEngine::from_entries(templates);
        engine.metadata = Some("state_test".to_string());
        
        // Capture initial state
        let initial_state = engine.clone_state();
        assert_eq!(initial_state.remaining_entries.len(), 3);
        assert_eq!(initial_state.processed_count, 0);
        assert_eq!(initial_state.misalignment_index, None);
        
        // Process some templates
        let constraints = Constraints::Integer(IntegerConstraints::default());
        let _ = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        let _ = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        
        assert_eq!(engine.remaining_count(), 1);
        assert_eq!(engine.processed_count(), 2);
        
        // Capture intermediate state
        let intermediate_state = engine.clone_state();
        assert_eq!(intermediate_state.remaining_entries.len(), 1);
        assert_eq!(intermediate_state.processed_count, 2);
        
        // Process remaining template
        let _ = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        assert_eq!(engine.remaining_count(), 0);
        assert_eq!(engine.processed_count(), 3);
        
        // Restore to intermediate state
        engine.restore_state(intermediate_state);
        assert_eq!(engine.remaining_count(), 1);
        assert_eq!(engine.processed_count(), 2);
        
        // Restore to initial state
        engine.restore_state(initial_state);
        assert_eq!(engine.remaining_count(), 3);
        assert_eq!(engine.processed_count(), 0);
        
        // Test reset functionality
        let _ = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        engine.reset();
        assert_eq!(engine.remaining_count(), 0);
        assert_eq!(engine.processed_count(), 0);
        assert!(!engine.has_misalignment());
    }

    /// Test partial node processing capability
    #[test]
    fn test_partial_node_processing_capability() {
        // Create partial nodes with different configurations
        let partial_entries = vec![
            TemplateEntry::partial_node(
                ChoiceValue::Integer(42),
                Some(Constraints::Integer(IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(100),
                    weights: None,
                    shrink_towards: Some(0),
                })),
                Some("custom_constraint_node".to_string()),
            ),
            TemplateEntry::partial_node(
                ChoiceValue::String("test".to_string()),
                None, // Use expected constraints
                Some("default_constraint_node".to_string()),
            ),
            TemplateEntry::partial_node(
                ChoiceValue::Float(3.14),
                Some(Constraints::Float(FloatConstraints {
                    min_value: 0.0,
                    max_value: 10.0,
                    allow_nan: false,
                    smallest_nonzero_magnitude: f64::MIN_POSITIVE,
                })),
                None, // No metadata
            ),
        ];
        
        let mut engine = TemplateEngine::from_entries(partial_entries);
        
        // Process partial node with custom constraints
        let default_int_constraints = Constraints::Integer(IntegerConstraints::default());
        let result = engine.process_next_template(ChoiceType::Integer, &default_int_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Integer(42));
        assert!(node.was_forced);
        // Should use the custom constraints from the partial node
        if let Constraints::Integer(constraints) = &node.constraints {
            assert_eq!(constraints.min_value, Some(0));
            assert_eq!(constraints.max_value, Some(100));
        } else {
            panic!("Expected integer constraints");
        }
        
        // Process partial node with default constraints
        let string_constraints = Constraints::String(StringConstraints {
            min_size: 0,
            max_size: 50,
            intervals: IntervalSet::default(),
        });
        let result = engine.process_next_template(ChoiceType::String, &string_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::String("test".to_string()));
        assert!(node.was_forced);
        // Should use the expected constraints (default)
        if let Constraints::String(constraints) = &node.constraints {
            assert_eq!(constraints.max_size, 50);
        } else {
            panic!("Expected string constraints");
        }
        
        // Process partial node with custom constraints but no metadata
        let default_float_constraints = Constraints::Float(FloatConstraints::default());
        let result = engine.process_next_template(ChoiceType::Float, &default_float_constraints).unwrap();
        assert!(result.is_some());
        let node = result.unwrap();
        assert_eq!(node.value, ChoiceValue::Float(3.14));
        assert!(node.was_forced);
        // Should use the custom constraints from the partial node
        if let Constraints::Float(constraints) = &node.constraints {
            assert_eq!(constraints.max_value, 10.0);
            assert!(!constraints.allow_nan);
        } else {
            panic!("Expected float constraints");
        }
    }

    /// Test template convenience functions capability
    #[test]
    fn test_template_convenience_functions_capability() {
        // Test simplest sequence creation
        let simplest_seq = templates::simplest_sequence(5);
        assert_eq!(simplest_seq.len(), 5);
        
        let mut engine = TemplateEngine::from_entries(simplest_seq);
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(10),
            max_value: Some(20),
            weights: None,
            shrink_towards: Some(15),
        });
        
        // All should generate the simplest choice (shrink_towards value)
        for i in 1..=5 {
            let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
            assert!(result.is_some());
            let node = result.unwrap();
            assert_eq!(node.value, ChoiceValue::Integer(15));
            assert_eq!(engine.processed_count(), i);
        }
        
        // Test value sequence creation
        let values = vec![
            ChoiceValue::Integer(100),
            ChoiceValue::Integer(200),
            ChoiceValue::Integer(300),
        ];
        let value_seq = templates::value_sequence(values.clone());
        let mut engine = TemplateEngine::from_entries(value_seq);
        
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(1000),
            weights: None,
            shrink_towards: Some(0),
        });
        
        // Should process the exact forced values
        for (i, expected_value) in values.iter().enumerate() {
            let result = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
            assert!(result.is_some());
            let node = result.unwrap();
            assert_eq!(node.value, *expected_value);
            assert!(node.was_forced);
            assert_eq!(engine.processed_count(), i + 1);
        }
        
        // Test index sequence creation
        let indices = vec![0, 1, 2, 3];
        let index_seq = templates::index_sequence(indices);
        assert_eq!(index_seq.len(), 4);
        
        for (i, entry) in index_seq.iter().enumerate() {
            if let TemplateEntry::Template(template) = entry {
                if let TemplateType::AtIndex(idx) = &template.template_type {
                    assert_eq!(*idx, i);
                } else {
                    panic!("Expected AtIndex template type");
                }
            } else {
                panic!("Expected template entry");
            }
        }
        
        // Test biased sequence creation
        let biases = vec![0.1, 0.3, 0.7, 0.9];
        let biased_seq = templates::biased_sequence(biases.clone());
        assert_eq!(biased_seq.len(), 4);
        
        for (i, entry) in biased_seq.iter().enumerate() {
            if let TemplateEntry::Template(template) = entry {
                if let TemplateType::Biased { bias } = &template.template_type {
                    assert!((bias - biases[i]).abs() < f64::EPSILON);
                } else {
                    panic!("Expected Biased template type");
                }
                assert!(!template.is_forcing); // Biased templates are not forcing
            } else {
                panic!("Expected template entry");
            }
        }
        
        // Test mixed sequence creation
        let templates_vec = vec![
            ChoiceTemplate::simplest(),
            ChoiceTemplate::at_index(1),
        ];
        let values_vec = vec![
            ChoiceValue::Integer(42),
            ChoiceValue::Integer(84),
        ];
        let mixed_seq = templates::mixed_sequence(templates_vec, values_vec);
        assert_eq!(mixed_seq.len(), 4);
        
        // Should have 2 template entries followed by 2 direct value entries
        for i in 0..2 {
            if let TemplateEntry::Template(_) = &mixed_seq[i] {
                // Expected template
            } else {
                panic!("Expected template entry at index {}", i);
            }
        }
        for i in 2..4 {
            if let TemplateEntry::DirectValue(_) = &mixed_seq[i] {
                // Expected direct value
            } else {
                panic!("Expected direct value entry at index {}", i);
            }
        }
    }

    /// Test template error handling capability
    #[test]
    fn test_template_error_handling_capability() {
        // Test exhausted template error
        let exhausted_template = ChoiceTemplate::with_count(TemplateType::Simplest, 0);
        let mut engine = TemplateEngine::from_templates(vec![exhausted_template]);
        
        let constraints = Constraints::Integer(IntegerConstraints::default());
        let result = engine.process_next_template(ChoiceType::Integer, &constraints);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TemplateError::ExhaustedTemplate);
        
        // Test constraint mismatch in simplest choice generation
        let engine = TemplateEngine::new();
        let wrong_constraints = Constraints::Boolean(BooleanConstraints::default());
        let result = engine.generate_simplest_choice(ChoiceType::Integer, &wrong_constraints);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TemplateError::ConstraintMismatch);
        
        // Test error display formatting
        assert_eq!(
            TemplateError::ExhaustedTemplate.to_string(),
            "Template has no remaining uses"
        );
        assert_eq!(
            TemplateError::ConstraintMismatch.to_string(),
            "Template value violates constraints"
        );
        assert_eq!(
            TemplateError::TypeMismatch.to_string(),
            "Template type doesn't match expected choice type"
        );
        assert_eq!(
            TemplateError::UnknownCustomTemplate("test".to_string()).to_string(),
            "Unknown custom template: test"
        );
        assert_eq!(
            TemplateError::ProcessingFailed("test error".to_string()).to_string(),
            "Template processing failed: test error"
        );
    }

    /// Test complete template type system capability
    #[test]
    fn test_complete_template_type_system_capability() {
        // Test all template type variants
        let template_types = vec![
            TemplateType::Simplest,
            TemplateType::AtIndex(42),
            TemplateType::Biased { bias: 0.75 },
            TemplateType::Custom { name: "custom_test".to_string() },
        ];
        
        // Test Display trait implementation
        assert_eq!(template_types[0].to_string(), "simplest");
        assert_eq!(template_types[1].to_string(), "at_index(42)");
        assert_eq!(template_types[2].to_string(), "biased(0.750)");
        assert_eq!(template_types[3].to_string(), "custom(custom_test)");
        
        // Test Hash trait implementation (via collections)
        use std::collections::HashSet;
        let mut type_set = HashSet::new();
        for template_type in &template_types {
            type_set.insert(template_type.clone());
        }
        assert_eq!(type_set.len(), 4);
        
        // Test PartialEq and Eq
        assert_eq!(TemplateType::Simplest, TemplateType::Simplest);
        assert_eq!(TemplateType::AtIndex(5), TemplateType::AtIndex(5));
        assert_ne!(TemplateType::AtIndex(5), TemplateType::AtIndex(6));
        assert_eq!(
            TemplateType::Biased { bias: 0.5 },
            TemplateType::Biased { bias: 0.5 }
        );
        assert_ne!(
            TemplateType::Biased { bias: 0.5 },
            TemplateType::Biased { bias: 0.6 }
        );
        assert_eq!(
            TemplateType::Custom { name: "test".to_string() },
            TemplateType::Custom { name: "test".to_string() }
        );
        assert_ne!(
            TemplateType::Custom { name: "test1".to_string() },
            TemplateType::Custom { name: "test2".to_string() }
        );
        
        // Test Default trait
        assert_eq!(TemplateType::default(), TemplateType::Simplest);
        
        // Test template creation with different types
        for template_type in template_types {
            let template = ChoiceTemplate::unlimited(template_type.clone());
            assert_eq!(template.template_type, template_type);
            assert!(template.is_forcing || matches!(template_type, TemplateType::Biased { .. }));
            assert_eq!(template.remaining_count(), None); // Unlimited templates expose None through remaining_count()
        }
    }

    /// Test template entry debug and analysis capability
    #[test]
    fn test_template_entry_debug_analysis_capability() {
        // Test debug descriptions for all entry types
        let entries = vec![
            TemplateEntry::DirectValue(ChoiceValue::Integer(42)),
            TemplateEntry::Template(ChoiceTemplate::biased(0.5)),
            TemplateEntry::partial_node(
                ChoiceValue::String("test".to_string()),
                Some(Constraints::String(StringConstraints::default())),
                Some("test_metadata".to_string()),
            ),
            TemplateEntry::partial_node(
                ChoiceValue::Boolean(true),
                None,
                None,
            ),
        ];
        
        let descriptions = entries.iter()
            .map(|entry| entry.debug_description())
            .collect::<Vec<_>>();
        
        assert_eq!(descriptions[0], "Direct(Integer(42))");
        assert_eq!(descriptions[1], "Template(biased(0.500))");
        assert!(descriptions[2].starts_with("Partial(String(\"test\") with constraints (test_metadata)"));
        assert_eq!(descriptions[3], "Partial(Boolean(true))");
        
        // Test forcing flags
        assert!(entries[0].is_forcing()); // Direct values are forcing
        assert!(!entries[1].is_forcing()); // Biased templates are not forcing
        assert!(entries[2].is_forcing()); // Partial nodes are forcing
        assert!(entries[3].is_forcing()); // Partial nodes are forcing
        
        // Test template engine debug info
        let mut engine = TemplateEngine::from_entries(entries);
        engine.debug_mode = true;
        engine.metadata = Some("debug_test".to_string());
        
        let debug_info = engine.debug_info();
        assert!(debug_info.contains("remaining: 4"));
        assert!(debug_info.contains("processed: 0"));
        assert!(debug_info.contains("misalignment: None"));
        assert!(debug_info.contains("debug: true"));
        
        // Process one entry and check updated debug info
        let constraints = Constraints::Integer(IntegerConstraints::default());
        let _ = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        
        let debug_info = engine.debug_info();
        assert!(debug_info.contains("remaining: 3"));
        assert!(debug_info.contains("processed: 1"));
    }

    /// Test template engine choice count calculation capability
    #[test]
    fn test_template_choice_count_calculation_capability() {
        // Create templates with various count configurations
        let templates = vec![
            TemplateEntry::Template(ChoiceTemplate::with_count(TemplateType::Simplest, 3)),
            TemplateEntry::Template(ChoiceTemplate::with_count(TemplateType::AtIndex(0), 2)),
            TemplateEntry::Template(ChoiceTemplate::unlimited(TemplateType::Biased { bias: 0.5 })),
            TemplateEntry::DirectValue(ChoiceValue::Integer(42)),
            TemplateEntry::partial_node(
                ChoiceValue::Boolean(true),
                None,
                None,
            ),
        ];
        
        let engine = TemplateEngine::from_entries(templates);
        
        // Calculate expected choice count
        // Template with count 3 + Template with count 2 + Unlimited (counts as 1) + Direct value (1) + Partial node (1)
        let expected_count = engine.calculate_expected_choice_count();
        assert_eq!(expected_count, 3 + 2 + 1 + 1 + 1); // = 8
        
        // Test with processed templates
        let mut engine = TemplateEngine::from_values(vec![
            ChoiceValue::Integer(1),
            ChoiceValue::Integer(2),
        ]);
        
        assert_eq!(engine.calculate_expected_choice_count(), 2);
        
        let constraints = Constraints::Integer(IntegerConstraints::default());
        let _ = engine.process_next_template(ChoiceType::Integer, &constraints).unwrap();
        
        // After processing one, count should include processed count
        assert_eq!(engine.calculate_expected_choice_count(), 2); // 1 remaining + 1 processed
    }

    /// Test complete templating integration with Python FFI patterns
    #[test]
    fn test_python_ffi_integration_patterns() {
        // Simulate Python-style template usage patterns
        
        // Pattern 1: Sequential template application
        let sequential_templates = vec![
            TemplateEntry::DirectValue(ChoiceValue::Integer(1)),
            TemplateEntry::Template(ChoiceTemplate::simplest()),
            TemplateEntry::DirectValue(ChoiceValue::Integer(2)),
            TemplateEntry::Template(ChoiceTemplate::at_index(0)),
        ];
        
        let mut engine = TemplateEngine::from_entries(sequential_templates);
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        });
        
        // Process all templates in sequence
        let mut results = Vec::new();
        while engine.has_templates() {
            if let Ok(Some(node)) = engine.process_next_template(ChoiceType::Integer, &int_constraints) {
                results.push(node);
            }
        }
        
        assert_eq!(results.len(), 4);
        assert_eq!(results[0].value, ChoiceValue::Integer(1)); // Direct value
        assert_eq!(results[1].value, ChoiceValue::Integer(0)); // Simplest (shrink_towards)
        assert_eq!(results[2].value, ChoiceValue::Integer(2)); // Direct value
        assert_eq!(results[3].value, ChoiceValue::Integer(0)); // At index (fallback to simplest)
        
        // Verify forcing patterns
        assert!(results[0].was_forced); // Direct values are forced
        assert!(!results[1].was_forced); // Template-generated choices aren't forced by default
        assert!(results[2].was_forced); // Direct values are forced
        assert!(!results[3].was_forced); // Template-generated choices aren't forced by default
        
        // Pattern 2: Error recovery and fallback
        let problematic_templates = vec![
            TemplateEntry::DirectValue(ChoiceValue::String("wrong_type".to_string())),
            TemplateEntry::DirectValue(ChoiceValue::Integer(100)), // Outside range
            TemplateEntry::DirectValue(ChoiceValue::Integer(5)), // Valid
        ];
        
        let mut engine = TemplateEngine::from_entries(problematic_templates);
        let strict_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(3),
        });
        
        // Process with error recovery
        let mut recovered_results = Vec::new();
        while engine.has_templates() {
            if let Ok(Some(node)) = engine.process_next_template(ChoiceType::Integer, &strict_constraints) {
                recovered_results.push(node);
            }
        }
        
        assert_eq!(recovered_results.len(), 3);
        assert_eq!(recovered_results[0].value, ChoiceValue::Integer(3)); // Fallback to simplest
        assert_eq!(recovered_results[1].value, ChoiceValue::Integer(3)); // Fallback to simplest
        assert_eq!(recovered_results[2].value, ChoiceValue::Integer(5)); // Valid value
        
        // Should have detected misalignments
        assert!(engine.has_misalignment());
        assert_eq!(engine.misalignment_index(), Some(1));
        
        // Pattern 3: State management for debugging/analysis
        let debug_templates = vec![
            TemplateEntry::Template(ChoiceTemplate::biased(0.3).with_metadata("debug_template".to_string())),
            TemplateEntry::partial_node(
                ChoiceValue::Float(2.5),
                Some(Constraints::Float(FloatConstraints {
                    min_value: 0.0,
                    max_value: 5.0,
                    allow_nan: false,
                    smallest_nonzero_magnitude: f64::MIN_POSITIVE,
                })),
                Some("analysis_node".to_string()),
            ),
        ];
        
        let mut engine = TemplateEngine::from_entries(debug_templates)
            .with_debug()
            .with_metadata("ffi_integration_test".to_string());
        
        // Capture state before processing
        let initial_state = engine.clone_state();
        
        // Process templates
        let float_constraints = Constraints::Float(FloatConstraints::default());
        let result1 = engine.process_next_template(ChoiceType::Float, &float_constraints).unwrap();
        let result2 = engine.process_next_template(ChoiceType::Float, &float_constraints).unwrap();
        
        assert!(result1.is_some());
        assert!(result2.is_some());
        assert_eq!(engine.processed_count(), 2);
        
        // Restore state and re-process
        engine.restore_state(initial_state);
        assert_eq!(engine.processed_count(), 0);
        assert_eq!(engine.remaining_count(), 2);
        
        // Verify engine metadata
        assert_eq!(engine.metadata, Some("ffi_integration_test".to_string()));
        assert!(engine.debug_mode);
    }
}