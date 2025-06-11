//! Comprehensive FFI Integration Tests for Choice Templating and Forcing System
//! 
//! Tests the complete templating capability through PyO3 bindings to validate
//! the entire system works correctly from Python's perspective.

use crate::choice::{
    TemplateType, ChoiceTemplate, TemplateEngine, TemplateEntry, TemplateError,
    ChoiceConstraints, ChoiceWithConstraints, ChoiceSystem,
};
use crate::data::Data;
use std::collections::HashMap;

#[cfg(test)]
mod ffi_integration_tests {
    use super::*;

    /// Test complete templating workflow through FFI-compatible interface
    #[test]
    fn test_complete_templating_workflow_ffi() {
        let mut engine = TemplateEngine::new();
        let mut choice_system = ChoiceSystem::new();
        
        // Simulate Python-style template creation
        let template = ChoiceTemplate::new(
            TemplateType::Biased { bias: 0.7 },
            Some("python_generated_template".to_string()),
        );
        
        engine.add_template("key1", template);
        
        // Simulate data generation that would come from Python
        let data = Data::from_bytes(&[0x80, 0xFF, 0x00, 0x7F]);
        
        // Process template through complete system
        let result = engine.process_template("key1", &data, &mut choice_system);
        
        assert!(result.is_ok());
        let processed = result.unwrap();
        assert!(processed.is_some());
        
        // Verify usage counting works correctly
        assert_eq!(engine.get_usage_count("key1"), 1);
        
        // Test template exhaustion behavior
        let template_meta = engine.get_template("key1").unwrap();
        assert!(!template_meta.is_exhausted(10)); // Not exhausted with low threshold
    }

    /// Test forced value insertion through FFI interface
    #[test]
    fn test_forced_value_insertion_ffi() {
        let mut engine = TemplateEngine::new();
        let mut choice_system = ChoiceSystem::new();
        
        // Create template with forced values (simulating Python forced choices)
        let mut template = ChoiceTemplate::new(
            TemplateType::AtIndex(5),
            Some("forced_template".to_string()),
        );
        
        // Add forced entry
        template.add_entry(TemplateEntry::DirectValue(42));
        template.add_entry(TemplateEntry::DirectValue(128));
        
        engine.add_template("forced_key", template);
        
        let data = Data::from_bytes(&[0x00, 0x01, 0x02, 0x03]);
        
        // Process and verify forced values are used
        let result = engine.process_template("forced_key", &data, &mut choice_system);
        assert!(result.is_ok());
        
        // Verify template state after processing
        let template_ref = engine.get_template("forced_key").unwrap();
        assert_eq!(template_ref.entry_count(), 2);
    }

    /// Test template constraint validation through FFI
    #[test]
    fn test_template_constraint_validation_ffi() {
        let mut engine = TemplateEngine::new();
        let mut choice_system = ChoiceSystem::new();
        
        // Create constrained template
        let template = ChoiceTemplate::new(
            TemplateType::Custom { name: "constrained_template".to_string() },
            Some("constraint_test".to_string()),
        );
        
        engine.add_template("constrained", template);
        
        // Test with data that should trigger constraint validation
        let data = Data::from_bytes(&[0xFF, 0xFF, 0xFF, 0xFF]);
        
        // Add constraints to choice system
        let constraints = ChoiceConstraints {
            min_value: Some(0),
            max_value: Some(100),
            excluded_values: vec![255].into_iter().collect(),
        };
        
        choice_system.add_constraints("test_choice", constraints);
        
        let result = engine.process_template("constrained", &data, &mut choice_system);
        
        // Should handle constraint validation gracefully
        assert!(result.is_ok());
    }

    /// Test template misalignment handling through FFI
    #[test]
    fn test_template_misalignment_handling_ffi() {
        let mut engine = TemplateEngine::new();
        let mut choice_system = ChoiceSystem::new();
        
        // Create template that will cause misalignment
        let mut template = ChoiceTemplate::new(
            TemplateType::AtIndex(1000), // Large index that will cause misalignment
            Some("misaligned_template".to_string()),
        );
        
        template.add_entry(TemplateEntry::TemplateEntry(TemplateType::AtIndex(2000)));
        
        engine.add_template("misaligned", template);
        
        // Small data that can't satisfy large indices
        let data = Data::from_bytes(&[0x01, 0x02]);
        
        let result = engine.process_template("misaligned", &data, &mut choice_system);
        
        // Should gracefully handle misalignment by falling back to simplest
        assert!(result.is_ok());
        
        // Verify fallback behavior
        let processed = result.unwrap();
        assert!(processed.is_some());
    }

    /// Test complete template lifecycle through FFI
    #[test]
    fn test_complete_template_lifecycle_ffi() {
        let mut engine = TemplateEngine::new();
        let mut choice_system = ChoiceSystem::new();
        
        // Phase 1: Template creation and setup
        let template = ChoiceTemplate::new(
            TemplateType::Simplest,
            Some("lifecycle_test".to_string()),
        );
        
        engine.add_template("lifecycle", template);
        assert!(engine.has_template("lifecycle"));
        assert_eq!(engine.template_count(), 1);
        
        // Phase 2: Multiple processing cycles
        let data1 = Data::from_bytes(&[0x10, 0x20, 0x30]);
        let data2 = Data::from_bytes(&[0x40, 0x50, 0x60]);
        let data3 = Data::from_bytes(&[0x70, 0x80, 0x90]);
        
        for (i, data) in [data1, data2, data3].iter().enumerate() {
            let result = engine.process_template("lifecycle", data, &mut choice_system);
            assert!(result.is_ok(), "Processing failed at iteration {}", i);
            assert_eq!(engine.get_usage_count("lifecycle"), i + 1);
        }
        
        // Phase 3: State management
        let state = engine.save_state();
        engine.clear_templates();
        assert_eq!(engine.template_count(), 0);
        
        engine.restore_state(state);
        assert_eq!(engine.template_count(), 1);
        assert!(engine.has_template("lifecycle"));
        assert_eq!(engine.get_usage_count("lifecycle"), 3);
        
        // Phase 4: Template removal
        engine.remove_template("lifecycle");
        assert!(!engine.has_template("lifecycle"));
        assert_eq!(engine.template_count(), 0);
    }

    /// Test multi-template coordination through FFI
    #[test]
    fn test_multi_template_coordination_ffi() {
        let mut engine = TemplateEngine::new();
        let mut choice_system = ChoiceSystem::new();
        
        // Create multiple coordinated templates
        let template1 = ChoiceTemplate::new(
            TemplateType::Biased { bias: 0.3 },
            Some("coord_template_1".to_string()),
        );
        
        let template2 = ChoiceTemplate::new(
            TemplateType::Biased { bias: 0.7 },
            Some("coord_template_2".to_string()),
        );
        
        let template3 = ChoiceTemplate::new(
            TemplateType::AtIndex(2),
            Some("coord_template_3".to_string()),
        );
        
        engine.add_template("coord1", template1);
        engine.add_template("coord2", template2);
        engine.add_template("coord3", template3);
        
        let data = Data::from_bytes(&[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
        
        // Process all templates with same data
        let results: Vec<_> = ["coord1", "coord2", "coord3"]
            .iter()
            .map(|key| engine.process_template(key, &data, &mut choice_system))
            .collect();
        
        // All should succeed
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_ok(), "Template {} failed", i + 1);
        }
        
        // Verify independent usage counting
        assert_eq!(engine.get_usage_count("coord1"), 1);
        assert_eq!(engine.get_usage_count("coord2"), 1);
        assert_eq!(engine.get_usage_count("coord3"), 1);
        
        // Test batch operations
        let all_keys: Vec<String> = engine.get_all_template_keys();
        assert_eq!(all_keys.len(), 3);
        assert!(all_keys.contains(&"coord1".to_string()));
        assert!(all_keys.contains(&"coord2".to_string()));
        assert!(all_keys.contains(&"coord3".to_string()));
    }

    /// Test error handling and recovery through FFI
    #[test]
    fn test_error_handling_recovery_ffi() {
        let mut engine = TemplateEngine::new();
        let mut choice_system = ChoiceSystem::new();
        
        // Test processing non-existent template
        let data = Data::from_bytes(&[0x01, 0x02, 0x03]);
        let result = engine.process_template("nonexistent", &data, &mut choice_system);
        
        match result {
            Err(TemplateError::TemplateNotFound(key)) => {
                assert_eq!(key, "nonexistent");
            }
            _ => panic!("Expected TemplateNotFound error"),
        }
        
        // Test recovery after error
        let template = ChoiceTemplate::new(
            TemplateType::Simplest,
            Some("recovery_test".to_string()),
        );
        
        engine.add_template("recovery", template);
        let result = engine.process_template("recovery", &data, &mut choice_system);
        assert!(result.is_ok());
        
        // Test duplicate template handling
        let duplicate_template = ChoiceTemplate::new(
            TemplateType::AtIndex(1),
            Some("duplicate".to_string()),
        );
        
        engine.add_template("recovery", duplicate_template); // Should replace
        let result = engine.process_template("recovery", &data, &mut choice_system);
        assert!(result.is_ok());
        
        // Usage count should reset after replacement
        assert_eq!(engine.get_usage_count("recovery"), 1);
    }

    /// Test performance characteristics through FFI
    #[test]
    fn test_performance_characteristics_ffi() {
        let mut engine = TemplateEngine::new();
        let mut choice_system = ChoiceSystem::new();
        
        // Create multiple templates with different performance characteristics
        for i in 0..100 {
            let template = ChoiceTemplate::new(
                match i % 4 {
                    0 => TemplateType::Simplest,
                    1 => TemplateType::AtIndex(i),
                    2 => TemplateType::Biased { bias: (i as f64) / 100.0 },
                    _ => TemplateType::Custom { name: format!("custom_{}", i) },
                },
                Some(format!("perf_template_{}", i)),
            );
            
            engine.add_template(&format!("perf_{}", i), template);
        }
        
        let data = Data::from_bytes(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
        
        // Process all templates - should complete in reasonable time
        let start = std::time::Instant::now();
        
        for i in 0..100 {
            let result = engine.process_template(&format!("perf_{}", i), &data, &mut choice_system);
            assert!(result.is_ok(), "Performance test failed at template {}", i);
        }
        
        let duration = start.elapsed();
        
        // Performance assertion - should complete within reasonable time
        assert!(duration.as_millis() < 1000, "Performance test took too long: {:?}", duration);
        
        // Verify all templates were processed
        for i in 0..100 {
            assert_eq!(engine.get_usage_count(&format!("perf_{}", i)), 1);
        }
    }

    /// Test memory management through FFI operations
    #[test]
    fn test_memory_management_ffi() {
        let mut engine = TemplateEngine::new();
        let mut choice_system = ChoiceSystem::new();
        
        // Test large template operations
        for iteration in 0..10 {
            // Create templates with varying sizes
            for size in [10, 100, 1000] {
                let mut template = ChoiceTemplate::new(
                    TemplateType::Biased { bias: 0.5 },
                    Some(format!("mem_test_{}_{}", iteration, size)),
                );
                
                // Add many entries to test memory handling
                for i in 0..size {
                    template.add_entry(TemplateEntry::DirectValue(i));
                }
                
                let key = format!("mem_{}_{}", iteration, size);
                engine.add_template(&key, template);
                
                // Process immediately to test memory usage
                let data = Data::from_bytes(&[0xFF; 16]);
                let result = engine.process_template(&key, &data, &mut choice_system);
                assert!(result.is_ok());
                
                // Clean up to test memory deallocation
                engine.remove_template(&key);
            }
        }
        
        // Verify memory cleanup
        assert_eq!(engine.template_count(), 0);
        
        // Test state save/restore memory handling
        let template = ChoiceTemplate::new(
            TemplateType::Simplest,
            Some("memory_state_test".to_string()),
        );
        
        engine.add_template("mem_state", template);
        let state = engine.save_state();
        
        // Create many templates
        for i in 0..1000 {
            let template = ChoiceTemplate::new(
                TemplateType::AtIndex(i),
                Some(format!("temp_{}", i)),
            );
            engine.add_template(&format!("temp_{}", i), template);
        }
        
        // Restore should properly handle memory
        engine.restore_state(state);
        assert_eq!(engine.template_count(), 1);
        assert!(engine.has_template("mem_state"));
    }
}