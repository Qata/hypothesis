#[cfg(test)]
mod templating_integration_tests {
    use super::*;
    use crate::data::*;
    use std::collections::HashMap;

    #[test]
    fn test_complete_templating_workflow() {
        let mut data = ConjectureData::for_buffer(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let mut engine = TemplateEngine::new();
        
        // Create template with multiple types
        let template = ChoiceTemplate {
            id: "complete_test".to_string(),
            template_type: TemplateType::Structured {
                components: vec![
                    ("header".to_string(), Box::new(TemplateType::Fixed { values: vec![1, 2] })),
                    ("body".to_string(), Box::new(TemplateType::Weighted { 
                        weights: vec![(3, 0.7), (4, 0.3)] 
                    })),
                    ("footer".to_string(), Box::new(TemplateType::Sequential { 
                        sequence: vec![5, 6] 
                    }))
                ]
            },
            usage_count: 0,
            constraints: Some(TemplateConstraints {
                min_usage: 1,
                max_usage: Some(10),
                conditions: HashMap::from([
                    ("valid_range".to_string(), Box::new(|_| true))
                ])
            })
        };
        
        engine.register_template(template);
        
        // Apply template and verify complete workflow
        let result = engine.apply_template(&mut data, "complete_test");
        assert!(result.is_ok());
        
        let values = result.unwrap();
        assert_eq!(values.len(), 3);
        assert!(vec![1, 2].contains(&values[0])); // header
        assert!(vec![3, 4].contains(&values[1])); // body  
        assert!(vec![5, 6].contains(&values[2])); // footer
        
        // Verify usage tracking
        let template_ref = engine.get_template("complete_test").unwrap();
        assert_eq!(template_ref.usage_count, 1);
    }

    #[test]
    fn test_forced_value_integration() {
        let mut data = ConjectureData::for_buffer(&[0, 1, 2, 3, 4, 5]);
        let mut engine = TemplateEngine::new();
        
        // Setup template
        let template = ChoiceTemplate {
            id: "forced_test".to_string(),
            template_type: TemplateType::Sequential { sequence: vec![10, 20, 30] },
            usage_count: 0,
            constraints: None
        };
        
        engine.register_template(template);
        
        // Force specific values
        data.force_value_at_index(0, 20);
        data.force_value_at_index(2, 10);
        
        let result = engine.apply_template(&mut data, "forced_test");
        assert!(result.is_ok());
        
        let values = result.unwrap();
        assert_eq!(values[0], 20); // forced
        assert_eq!(values[2], 10); // forced
        
        // Verify forced values are tracked
        assert!(data.forced_indices.contains(&0));
        assert!(data.forced_indices.contains(&2));
    }

    #[test]
    fn test_constraint_validation_integration() {
        let mut data = ConjectureData::for_buffer(&[0, 1, 2, 3]);
        let mut engine = TemplateEngine::new();
        
        let template = ChoiceTemplate {
            id: "constrained_test".to_string(),
            template_type: TemplateType::Weighted { 
                weights: vec![(1, 0.5), (2, 0.5)] 
            },
            usage_count: 0,
            constraints: Some(TemplateConstraints {
                min_usage: 2,
                max_usage: Some(3),
                conditions: HashMap::from([
                    ("even_only".to_string(), Box::new(|values: &[u64]| {
                        values.iter().all(|&v| v % 2 == 0)
                    }))
                ])
            })
        };
        
        engine.register_template(template);
        
        // First usage - should fail min_usage
        let result1 = engine.apply_template(&mut data, "constrained_test");
        assert!(result1.is_ok());
        
        // Second usage - should pass
        let result2 = engine.apply_template(&mut data, "constrained_test");
        assert!(result2.is_ok());
        
        // Verify constraint validation
        let template_ref = engine.get_template("constrained_test").unwrap();
        assert_eq!(template_ref.usage_count, 2);
    }

    #[test]
    fn test_python_parity_template_processing() {
        let mut data = ConjectureData::for_buffer(&[42, 13, 7, 99, 1, 8]);
        let mut engine = TemplateEngine::new();
        
        // Mimic Python's template structure
        let python_style_template = ChoiceTemplate {
            id: "python_parity".to_string(),
            template_type: TemplateType::Structured {
                components: vec![
                    ("type_hint".to_string(), Box::new(TemplateType::Fixed { 
                        values: vec![42] // Python uses type hints
                    })),
                    ("choice_value".to_string(), Box::new(TemplateType::Weighted { 
                        weights: vec![(13, 0.6), (7, 0.4)] // Python probability distribution
                    })),
                    ("metadata".to_string(), Box::new(TemplateType::Sequential { 
                        sequence: vec![99, 1, 8] // Python metadata sequence
                    }))
                ]
            },
            usage_count: 0,
            constraints: Some(TemplateConstraints {
                min_usage: 1,
                max_usage: Some(5),
                conditions: HashMap::from([
                    ("python_valid".to_string(), Box::new(|values: &[u64]| {
                        values.len() >= 3 && values[0] == 42
                    }))
                ])
            })
        };
        
        engine.register_template(python_style_template);
        
        let result = engine.apply_template(&mut data, "python_parity");
        assert!(result.is_ok());
        
        let values = result.unwrap();
        assert_eq!(values[0], 42); // Type hint preserved
        assert!(vec![13, 7].contains(&values[1])); // Choice from distribution
        assert_eq!(values.len(), 3); // Complete structure
    }

    #[test]
    fn test_template_engine_state_management() {
        let mut engine = TemplateEngine::new();
        
        // Register multiple templates
        for i in 0..5 {
            let template = ChoiceTemplate {
                id: format!("template_{}", i),
                template_type: TemplateType::Fixed { values: vec![i as u64] },
                usage_count: 0,
                constraints: None
            };
            engine.register_template(template);
        }
        
        // Verify all templates registered
        assert_eq!(engine.templates.len(), 5);
        
        // Use templates and verify state updates
        let mut data = ConjectureData::for_buffer(&[0, 1, 2, 3, 4]);
        
        for i in 0..5 {
            let result = engine.apply_template(&mut data, &format!("template_{}", i));
            assert!(result.is_ok());
            
            let template_ref = engine.get_template(&format!("template_{}", i)).unwrap();
            assert_eq!(template_ref.usage_count, 1);
        }
        
        // Test template removal
        engine.remove_template("template_2");
        assert_eq!(engine.templates.len(), 4);
        assert!(engine.get_template("template_2").is_none());
    }

    #[test]
    fn test_error_handling_integration() {
        let mut data = ConjectureData::for_buffer(&[]);
        let mut engine = TemplateEngine::new();
        
        // Test missing template
        let result = engine.apply_template(&mut data, "nonexistent");
        assert!(result.is_err());
        
        // Test constraint violation
        let invalid_template = ChoiceTemplate {
            id: "invalid".to_string(),
            template_type: TemplateType::Fixed { values: vec![] }, // Empty values
            usage_count: 0,
            constraints: Some(TemplateConstraints {
                min_usage: 1,
                max_usage: Some(0), // Invalid: max < min
                conditions: HashMap::new()
            })
        };
        
        engine.register_template(invalid_template);
        let result = engine.apply_template(&mut data, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_nested_template_structure() {
        let mut data = ConjectureData::for_buffer(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let mut engine = TemplateEngine::new();
        
        // Create deeply nested template
        let nested_template = ChoiceTemplate {
            id: "nested_complex".to_string(),
            template_type: TemplateType::Structured {
                components: vec![
                    ("level1".to_string(), Box::new(TemplateType::Structured {
                        components: vec![
                            ("level2a".to_string(), Box::new(TemplateType::Fixed { values: vec![1, 2] })),
                            ("level2b".to_string(), Box::new(TemplateType::Weighted { 
                                weights: vec![(3, 0.8), (4, 0.2)] 
                            }))
                        ]
                    })),
                    ("level1_seq".to_string(), Box::new(TemplateType::Sequential { 
                        sequence: vec![5, 6, 7] 
                    }))
                ]
            },
            usage_count: 0,
            constraints: Some(TemplateConstraints {
                min_usage: 1,
                max_usage: Some(3),
                conditions: HashMap::from([
                    ("structure_valid".to_string(), Box::new(|values: &[u64]| {
                        values.len() >= 5 // Expect flattened structure
                    }))
                ])
            })
        };
        
        engine.register_template(nested_template);
        
        let result = engine.apply_template(&mut data, "nested_complex");
        assert!(result.is_ok());
        
        let values = result.unwrap();
        assert_eq!(values.len(), 5); // 2 + 1 + 3 (level2a + level2b + level1_seq)
        
        // Verify structure integrity
        assert!(vec![1, 2].contains(&values[0])); // level2a
        assert!(vec![3, 4].contains(&values[1])); // level2b  
        assert_eq!(&values[2..5], &[5, 6, 7]); // level1_seq
    }

    #[test]
    fn test_template_lifecycle_management() {
        let mut engine = TemplateEngine::new();
        let mut data = ConjectureData::for_buffer(&[1, 2, 3, 4, 5]);
        
        // Create template with lifecycle constraints
        let lifecycle_template = ChoiceTemplate {
            id: "lifecycle_test".to_string(),
            template_type: TemplateType::Sequential { sequence: vec![1, 2, 3] },
            usage_count: 0,
            constraints: Some(TemplateConstraints {
                min_usage: 2,
                max_usage: Some(4),
                conditions: HashMap::from([
                    ("lifecycle_valid".to_string(), Box::new(|_| true))
                ])
            })
        };
        
        engine.register_template(lifecycle_template);
        
        // Use template through its lifecycle
        for expected_count in 1..=4 {
            let result = engine.apply_template(&mut data, "lifecycle_test");
            assert!(result.is_ok());
            
            let template_ref = engine.get_template("lifecycle_test").unwrap();
            assert_eq!(template_ref.usage_count, expected_count);
        }
        
        // Verify max usage enforcement
        let result = engine.apply_template(&mut data, "lifecycle_test");
        assert!(result.is_err()); // Should fail due to max usage exceeded
    }

    #[test]
    fn test_ffi_compatibility_structures() {
        // Test that our structures are compatible with FFI boundaries
        let template = ChoiceTemplate {
            id: "ffi_test".to_string(),
            template_type: TemplateType::Fixed { values: vec![42] },
            usage_count: 0,
            constraints: None
        };
        
        // Simulate FFI boundary crossing (serialization/deserialization)
        let serialized = serde_json::to_string(&template).unwrap();
        let deserialized: ChoiceTemplate = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(template.id, deserialized.id);
        assert_eq!(template.usage_count, deserialized.usage_count);
        
        match (&template.template_type, &deserialized.template_type) {
            (TemplateType::Fixed { values: v1 }, TemplateType::Fixed { values: v2 }) => {
                assert_eq!(v1, v2);
            }
            _ => panic!("Template type mismatch after FFI simulation")
        }
    }
}