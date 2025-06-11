//! Integration tests for Choice templating and forcing system capability
//!
//! This module provides comprehensive integration tests that validate the complete
//! Choice templating and forcing system capability behavior. Tests focus on the
//! capability's core responsibilities: template-driven choice generation and
//! forced value insertion for guided test case construction.

use crate::choice::{
    templating::*,
    ChoiceType, ChoiceValue, Constraints, ChoiceNode,
    IntegerConstraints, BooleanConstraints, FloatConstraints, StringConstraints, BytesConstraints,
    StringAlphabet, BytesMask
};
use std::collections::VecDeque;

/// Integration test suite for validating the complete templating capability
pub struct TemplatingCapabilityIntegrationTests {
    engine: TemplateEngine,
    test_results: Vec<String>,
}

impl TemplatingCapabilityIntegrationTests {
    pub fn new() -> Self {
        Self {
            engine: TemplateEngine::new().with_debug(),
            test_results: Vec::new(),
        }
    }

    /// Run complete capability validation with all core functionality
    pub fn run_complete_validation(&mut self) -> Result<Vec<String>, String> {
        self.test_results.clear();

        // Test 1: Complete template-driven choice generation capability
        self.validate_template_driven_generation()?;

        // Test 2: Complete forced value insertion capability  
        self.validate_forced_value_insertion()?;

        // Test 3: Complete template usage management capability
        self.validate_template_usage_management()?;

        // Test 4: Complete misalignment handling capability
        self.validate_misalignment_handling()?;

        // Test 5: Complete state management capability
        self.validate_state_management()?;

        // Test 6: Complete constraint validation capability
        self.validate_constraint_validation()?;

        // Test 7: Complete template type coverage capability
        self.validate_template_type_coverage()?;

        // Test 8: Complete error handling capability
        self.validate_error_handling()?;

        Ok(self.test_results.clone())
    }

    /// Test 1: Validate complete template-driven choice generation capability
    fn validate_template_driven_generation(&mut self) -> Result<(), String> {
        self.engine.reset();
        
        // Create comprehensive template patterns
        let templates = vec![
            ChoiceTemplate::simplest(),
            ChoiceTemplate::at_index(5),
            ChoiceTemplate::biased(0.75),
            ChoiceTemplate::custom("test_custom".to_string()),
            ChoiceTemplate::unlimited(TemplateType::Simplest),
        ];

        for template in templates {
            self.engine.add_entry(TemplateEntry::Template(template));
        }

        // Test with different choice types and constraints
        let test_cases = vec![
            (ChoiceType::Integer, Constraints::Integer(IntegerConstraints {
                min_value: Some(-100),
                max_value: Some(100),
                weights: None,
                shrink_towards: Some(0),
            })),
            (ChoiceType::Boolean, Constraints::Boolean(BooleanConstraints::default())),
            (ChoiceType::Float, Constraints::Float(FloatConstraints {
                min_value: -10.0,
                max_value: 10.0,
                allow_nan: false,
                smallest_nonzero_magnitude: f64::MIN_POSITIVE,
            })),
            (ChoiceType::String, Constraints::String(StringConstraints {
                min_size: 0,
                max_size: 100,
                intervals: StringAlphabet::ascii(),
            })),
            (ChoiceType::Bytes, Constraints::Bytes(BytesConstraints {
                min_size: 0,
                max_size: 100,
                mask: BytesMask::all(),
            })),
        ];

        let mut successful_generations = 0;
        for (choice_type, constraints) in test_cases {
            // Process multiple templates for each type
            for template_num in 1..=4 { // Process 4 templates per type
                match self.engine.process_next_template(choice_type, &constraints) {
                    Ok(Some(node)) => {
                        successful_generations += 1;
                        // Validate node structure
                        if !self.validate_choice_node(&node, choice_type, &constraints) {
                            return Err(format!("Invalid choice node generated for {} template {}", 
                                self.choice_type_name(choice_type), template_num));
                        }
                    },
                    Ok(None) => break, // No more templates for this type
                    Err(e) => return Err(format!("Template processing failed: {}", e)),
                }
            }
        }

        self.test_results.push(format!("Template-driven generation: {} successful generations", successful_generations));
        Ok(())
    }

    /// Test 2: Validate complete forced value insertion capability
    fn validate_forced_value_insertion(&mut self) -> Result<(), String> {
        self.engine.reset();

        // Test comprehensive forced value patterns
        let forced_values = vec![
            // Integer values with different ranges
            (ChoiceValue::Integer(-42), ChoiceType::Integer, "negative integer"),
            (ChoiceValue::Integer(0), ChoiceType::Integer, "zero integer"),
            (ChoiceValue::Integer(42), ChoiceType::Integer, "positive integer"),
            (ChoiceValue::Integer(i128::MAX), ChoiceType::Integer, "max integer"),
            
            // Boolean values
            (ChoiceValue::Boolean(true), ChoiceType::Boolean, "true boolean"),
            (ChoiceValue::Boolean(false), ChoiceType::Boolean, "false boolean"),
            
            // Float values with special cases
            (ChoiceValue::Float(0.0), ChoiceType::Float, "zero float"),
            (ChoiceValue::Float(-3.14159), ChoiceType::Float, "negative float"),
            (ChoiceValue::Float(2.71828), ChoiceType::Float, "positive float"),
            (ChoiceValue::Float(f64::MIN_POSITIVE), ChoiceType::Float, "min positive float"),
            
            // String values with different patterns
            (ChoiceValue::String("".to_string()), ChoiceType::String, "empty string"),
            (ChoiceValue::String("a".to_string()), ChoiceType::String, "single char string"),
            (ChoiceValue::String("Hello, World!".to_string()), ChoiceType::String, "multi-char string"),
            (ChoiceValue::String("🦀🔥".to_string()), ChoiceType::String, "unicode string"),
            
            // Bytes values with different patterns
            (ChoiceValue::Bytes(vec![]), ChoiceType::Bytes, "empty bytes"),
            (ChoiceValue::Bytes(vec![0]), ChoiceType::Bytes, "single byte"),
            (ChoiceValue::Bytes(vec![1, 2, 3, 4, 5]), ChoiceType::Bytes, "multi-byte sequence"),
            (ChoiceValue::Bytes(vec![255; 10]), ChoiceType::Bytes, "repeated bytes"),
        ];

        for (value, choice_type, description) in forced_values {
            self.engine.add_entry(TemplateEntry::DirectValue(value.clone()));
            
            let constraints = self.create_permissive_constraints(choice_type);
            
            match self.engine.process_next_template(choice_type, &constraints) {
                Ok(Some(node)) => {
                    // Validate forced value was correctly inserted
                    if node.value != value {
                        return Err(format!("Forced value mismatch for {}: expected {:?}, got {:?}", 
                            description, value, node.value));
                    }
                    if !node.was_forced {
                        return Err(format!("Forced value not marked as forced: {}", description));
                    }
                    self.test_results.push(format!("Forced value insertion: {} successful", description));
                },
                Ok(None) => return Err(format!("No choice generated for forced value: {}", description)),
                Err(e) => return Err(format!("Forced value processing failed for {}: {}", description, e)),
            }
        }

        Ok(())
    }

    /// Test 3: Validate complete template usage management capability
    fn validate_template_usage_management(&mut self) -> Result<(), String> {
        self.engine.reset();

        // Test different usage count patterns
        let usage_patterns = vec![
            (1, "single-use template"),
            (3, "limited-use template"),
            (10, "multi-use template"),
            (usize::MAX, "unlimited template"),
        ];

        for (count, description) in usage_patterns {
            let template = if count == usize::MAX {
                ChoiceTemplate::unlimited(TemplateType::Simplest)
            } else {
                ChoiceTemplate::with_count(TemplateType::Simplest, count)
            };

            self.engine.add_entry(TemplateEntry::Template(template));
            
            let constraints = Constraints::Integer(IntegerConstraints::default());
            let expected_uses = if count == usize::MAX { 5 } else { count }; // Test up to 5 uses for unlimited
            
            let mut successful_uses = 0;
            for use_num in 1..=expected_uses {
                match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
                    Ok(Some(_)) => {
                        successful_uses += 1;
                        if count != usize::MAX && use_num == count {
                            // Template should be exhausted now
                            match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
                                Ok(None) => {
                                    self.test_results.push(format!("Usage management: {} exhausted correctly after {} uses", 
                                        description, count));
                                    break;
                                },
                                Ok(Some(_)) => return Err(format!("Template not exhausted after {} uses: {}", count, description)),
                                Err(_) => return Err(format!("Unexpected error after template exhaustion: {}", description)),
                            }
                        }
                    },
                    Ok(None) => {
                        if use_num <= count {
                            return Err(format!("Template exhausted prematurely at use {}: {}", use_num, description));
                        }
                        break;
                    },
                    Err(e) => return Err(format!("Template usage failed at use {}: {} - {}", use_num, description, e)),
                }
            }

            if count == usize::MAX {
                self.test_results.push(format!("Usage management: {} processed {} uses successfully", description, successful_uses));
            }
        }

        Ok(())
    }

    /// Test 4: Validate complete misalignment handling capability  
    fn validate_misalignment_handling(&mut self) -> Result<(), String> {
        self.engine.reset();

        // Test different misalignment scenarios
        let misalignment_tests = vec![
            // Value outside integer constraints
            (
                TemplateEntry::DirectValue(ChoiceValue::Integer(1000)),
                ChoiceType::Integer,
                Constraints::Integer(IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(100),
                    weights: None,
                    shrink_towards: Some(50),
                }),
                "integer value outside range"
            ),
            // String too long for constraints
            (
                TemplateEntry::DirectValue(ChoiceValue::String("this string is way too long".to_string())),
                ChoiceType::String,
                Constraints::String(StringConstraints {
                    min_size: 0,
                    max_size: 5,
                    intervals: StringAlphabet::ascii(),
                }),
                "string exceeds max length"
            ),
            // Bytes too large for constraints  
            (
                TemplateEntry::DirectValue(ChoiceValue::Bytes(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
                ChoiceType::Bytes,
                Constraints::Bytes(BytesConstraints {
                    min_size: 0,
                    max_size: 3,
                    mask: BytesMask::all(),
                }),
                "bytes exceed max size"
            ),
            // Float outside range
            (
                TemplateEntry::DirectValue(ChoiceValue::Float(100.0)),
                ChoiceType::Float,
                Constraints::Float(FloatConstraints {
                    min_value: -1.0,
                    max_value: 1.0,
                    allow_nan: false,
                    smallest_nonzero_magnitude: f64::MIN_POSITIVE,
                }),
                "float outside range"
            ),
        ];

        for (entry, choice_type, constraints, description) in misalignment_tests {
            self.engine.reset();
            self.engine.add_entry(entry);

            match self.engine.process_next_template(choice_type, &constraints) {
                Ok(Some(node)) => {
                    // Should have fallen back to valid choice
                    if !self.validate_choice_node(&node, choice_type, &constraints) {
                        return Err(format!("Fallback choice invalid for: {}", description));
                    }
                    if !self.engine.has_misalignment() {
                        return Err(format!("Misalignment not detected for: {}", description));
                    }
                    self.test_results.push(format!("Misalignment handling: {} handled correctly", description));
                },
                Ok(None) => return Err(format!("No fallback choice generated for: {}", description)),
                Err(e) => return Err(format!("Misalignment handling failed for {}: {}", description, e)),
            }
        }

        Ok(())
    }

    /// Test 5: Validate complete state management capability
    fn validate_state_management(&mut self) -> Result<(), String> {
        self.engine.reset();

        // Set up initial state
        let initial_entries = vec![
            TemplateEntry::DirectValue(ChoiceValue::Integer(1)),
            TemplateEntry::DirectValue(ChoiceValue::Integer(2)),
            TemplateEntry::DirectValue(ChoiceValue::Integer(3)),
        ];

        for entry in initial_entries {
            self.engine.add_entry(entry);
        }

        // Save initial state
        let initial_state = self.engine.clone_state();
        assert_eq!(initial_state.remaining_entries.len(), 3);
        assert_eq!(initial_state.processed_count, 0);

        // Process some templates
        let constraints = Constraints::Integer(IntegerConstraints::default());
        let _ = self.engine.process_next_template(ChoiceType::Integer, &constraints);
        let _ = self.engine.process_next_template(ChoiceType::Integer, &constraints);

        // Verify state changes
        assert_eq!(self.engine.remaining_count(), 1);
        assert_eq!(self.engine.processed_count(), 2);

        // Save intermediate state
        let intermediate_state = self.engine.clone_state();

        // Process remaining template
        let _ = self.engine.process_next_template(ChoiceType::Integer, &constraints);
        assert_eq!(self.engine.remaining_count(), 0);
        assert_eq!(self.engine.processed_count(), 3);

        // Restore to intermediate state
        self.engine.restore_state(intermediate_state.clone());
        assert_eq!(self.engine.remaining_count(), 1);
        assert_eq!(self.engine.processed_count(), 2);

        // Restore to initial state
        self.engine.restore_state(initial_state);
        assert_eq!(self.engine.remaining_count(), 3);
        assert_eq!(self.engine.processed_count(), 0);

        self.test_results.push("State management: backup and restore working correctly".to_string());
        Ok(())
    }

    /// Test 6: Validate complete constraint validation capability
    fn validate_constraint_validation(&mut self) -> Result<(), String> {
        self.engine.reset();

        // Test constraint validation for different types
        let validation_tests = vec![
            // Integer constraint validation
            (
                ChoiceValue::Integer(50),
                ChoiceType::Integer,
                Constraints::Integer(IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(100),
                    weights: None,
                    shrink_towards: Some(0),
                }),
                true,
                "valid integer within range"
            ),
            (
                ChoiceValue::Integer(-10),
                ChoiceType::Integer,
                Constraints::Integer(IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(100),
                    weights: None,
                    shrink_towards: Some(0),
                }),
                false,
                "invalid integer below range"
            ),
            // String constraint validation
            (
                ChoiceValue::String("hello".to_string()),
                ChoiceType::String,
                Constraints::String(StringConstraints {
                    min_size: 3,
                    max_size: 10,
                    intervals: StringAlphabet::ascii(),
                }),
                true,
                "valid string within size range"
            ),
            (
                ChoiceValue::String("hi".to_string()),
                ChoiceType::String,
                Constraints::String(StringConstraints {
                    min_size: 3,
                    max_size: 10,
                    intervals: StringAlphabet::ascii(),
                }),
                false,
                "invalid string below min size"
            ),
            // Float constraint validation  
            (
                ChoiceValue::Float(0.5),
                ChoiceType::Float,
                Constraints::Float(FloatConstraints {
                    min_value: 0.0,
                    max_value: 1.0,
                    allow_nan: false,
                    smallest_nonzero_magnitude: f64::MIN_POSITIVE,
                }),
                true,
                "valid float within range"
            ),
            (
                ChoiceValue::Float(f64::NAN),
                ChoiceType::Float,
                Constraints::Float(FloatConstraints {
                    min_value: 0.0,
                    max_value: 1.0,
                    allow_nan: false,
                    smallest_nonzero_magnitude: f64::MIN_POSITIVE,
                }),
                false,
                "invalid NaN when not allowed"
            ),
        ];

        for (value, choice_type, constraints, should_be_valid, description) in validation_tests {
            self.engine.reset();
            self.engine.add_entry(TemplateEntry::DirectValue(value));

            match self.engine.process_next_template(choice_type, &constraints) {
                Ok(Some(node)) => {
                    let is_valid = self.validate_choice_node(&node, choice_type, &constraints);
                    if should_be_valid {
                        if is_valid && !self.engine.has_misalignment() {
                            self.test_results.push(format!("Constraint validation: {} passed", description));
                        } else {
                            return Err(format!("Valid constraint incorrectly rejected: {}", description));
                        }
                    } else {
                        if self.engine.has_misalignment() {
                            self.test_results.push(format!("Constraint validation: {} correctly rejected", description));
                        } else {
                            return Err(format!("Invalid constraint incorrectly accepted: {}", description));
                        }
                    }
                },
                Ok(None) => return Err(format!("No choice generated for constraint test: {}", description)),
                Err(e) => return Err(format!("Constraint validation failed: {} - {}", description, e)),
            }
        }

        Ok(())
    }

    /// Test 7: Validate complete template type coverage capability
    fn validate_template_type_coverage(&mut self) -> Result<(), String> {
        self.engine.reset();

        // Test all template types comprehensively
        let template_type_tests = vec![
            (TemplateType::Simplest, "simplest template type"),
            (TemplateType::AtIndex(7), "at-index template type"),
            (TemplateType::Biased { bias: 0.3 }, "biased template type"),
            (TemplateType::Custom { name: "test_custom".to_string() }, "custom template type"),
        ];

        for (template_type, description) in template_type_tests {
            let template = ChoiceTemplate::unlimited(template_type.clone());
            self.engine.add_entry(TemplateEntry::Template(template));

            let constraints = Constraints::Integer(IntegerConstraints::default());
            
            match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
                Ok(Some(node)) => {
                    // Validate that choice was generated according to template type
                    if !self.validate_choice_node(&node, ChoiceType::Integer, &constraints) {
                        return Err(format!("Invalid choice from {}", description));
                    }
                    
                    // Check forcing behavior based on template type
                    match template_type {
                        TemplateType::Biased { .. } => {
                            if node.was_forced {
                                return Err(format!("Biased template incorrectly marked as forced"));
                            }
                        },
                        _ => {
                            // Other template types may or may not be forced depending on implementation
                        }
                    }
                    
                    self.test_results.push(format!("Template type coverage: {} working correctly", description));
                },
                Ok(None) => return Err(format!("No choice generated for {}", description)),
                Err(e) => return Err(format!("Template type failed: {} - {}", description, e)),
            }
        }

        Ok(())
    }

    /// Test 8: Validate complete error handling capability
    fn validate_error_handling(&mut self) -> Result<(), String> {
        self.engine.reset();

        // Test exhausted template error handling
        let exhausted_template = ChoiceTemplate::with_count(TemplateType::Simplest, 1);
        self.engine.add_entry(TemplateEntry::Template(exhausted_template));
        
        let constraints = Constraints::Integer(IntegerConstraints::default());
        
        // Use the template once
        match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
            Ok(Some(_)) => {},
            _ => return Err("Failed to use template once".to_string()),
        }
        
        // Try to use exhausted template (should return None, not error)
        match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
            Ok(None) => self.test_results.push("Error handling: exhausted template handled correctly".to_string()),
            Ok(Some(_)) => return Err("Exhausted template incorrectly generated choice".to_string()),
            Err(e) => return Err(format!("Unexpected error from exhausted template: {}", e)),
        }

        // Test constraint mismatch handling
        self.engine.reset();
        match self.engine.generate_simplest_choice(ChoiceType::Integer, &Constraints::Boolean(BooleanConstraints::default())) {
            Err(TemplateError::ConstraintMismatch) => {
                self.test_results.push("Error handling: constraint mismatch detected correctly".to_string());
            },
            Ok(_) => return Err("Constraint mismatch not detected".to_string()),
            Err(e) => return Err(format!("Unexpected error for constraint mismatch: {}", e)),
        }

        Ok(())
    }

    // Helper methods

    fn validate_choice_node(&self, node: &ChoiceNode, expected_type: ChoiceType, constraints: &Constraints) -> bool {
        // Validate choice type matches
        if node.choice_type != expected_type {
            return false;
        }

        // Validate value matches type
        match (&node.value, expected_type) {
            (ChoiceValue::Integer(_), ChoiceType::Integer) => {},
            (ChoiceValue::Boolean(_), ChoiceType::Boolean) => {},
            (ChoiceValue::Float(_), ChoiceType::Float) => {},
            (ChoiceValue::String(_), ChoiceType::String) => {},
            (ChoiceValue::Bytes(_), ChoiceType::Bytes) => {},
            _ => return false,
        }

        // Validate value satisfies constraints
        self.value_satisfies_constraints(&node.value, constraints)
    }

    fn value_satisfies_constraints(&self, value: &ChoiceValue, constraints: &Constraints) -> bool {
        match (value, constraints) {
            (ChoiceValue::Integer(val), Constraints::Integer(c)) => {
                let min_ok = c.min_value.map_or(true, |min| *val >= min);
                let max_ok = c.max_value.map_or(true, |max| *val <= max);
                min_ok && max_ok
            },
            (ChoiceValue::Boolean(_), Constraints::Boolean(_)) => true,
            (ChoiceValue::Float(val), Constraints::Float(c)) => {
                if val.is_nan() {
                    c.allow_nan
                } else {
                    *val >= c.min_value && *val <= c.max_value
                }
            },
            (ChoiceValue::String(val), Constraints::String(c)) => {
                val.len() >= c.min_size && val.len() <= c.max_size
            },
            (ChoiceValue::Bytes(val), Constraints::Bytes(c)) => {
                val.len() >= c.min_size && val.len() <= c.max_size
            },
            _ => false,
        }
    }

    fn create_permissive_constraints(&self, choice_type: ChoiceType) -> Constraints {
        match choice_type {
            ChoiceType::Integer => Constraints::Integer(IntegerConstraints {
                min_value: Some(i128::MIN / 2),
                max_value: Some(i128::MAX / 2),
                weights: None,
                shrink_towards: Some(0),
            }),
            ChoiceType::Boolean => Constraints::Boolean(BooleanConstraints::default()),
            ChoiceType::Float => Constraints::Float(FloatConstraints {
                min_value: f64::NEG_INFINITY,
                max_value: f64::INFINITY,
                allow_nan: true,
                smallest_nonzero_magnitude: f64::MIN_POSITIVE,
            }),
            ChoiceType::String => Constraints::String(StringConstraints {
                min_size: 0,
                max_size: 10000,
                intervals: StringAlphabet::ascii(),
            }),
            ChoiceType::Bytes => Constraints::Bytes(BytesConstraints {
                min_size: 0,
                max_size: 10000,
                mask: BytesMask::all(),
            }),
        }
    }

    fn choice_type_name(&self, choice_type: ChoiceType) -> &'static str {
        match choice_type {
            ChoiceType::Integer => "Integer",
            ChoiceType::Boolean => "Boolean",
            ChoiceType::Float => "Float",
            ChoiceType::String => "String",
            ChoiceType::Bytes => "Bytes",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_capability_validation() {
        let mut test_suite = TemplatingCapabilityIntegrationTests::new();
        
        let results = test_suite.run_complete_validation();
        assert!(results.is_ok(), "Capability validation failed: {:?}", results.err());
        
        let test_results = results.unwrap();
        assert!(!test_results.is_empty(), "No test results generated");
        
        // Verify all major capability areas were tested
        let expected_capabilities = vec![
            "Template-driven generation",
            "Forced value insertion", 
            "Usage management",
            "Misalignment handling",
            "State management",
            "Constraint validation",
            "Template type coverage",
            "Error handling",
        ];
        
        for capability in expected_capabilities {
            let found = test_results.iter().any(|result| result.contains(capability));
            assert!(found, "Missing test results for capability: {}", capability);
        }
        
        println!("Capability validation completed successfully:");
        for result in test_results {
            println!("  ✓ {}", result);
        }
    }

    #[test]
    fn test_template_driven_generation_only() {
        let mut test_suite = TemplatingCapabilityIntegrationTests::new();
        
        let result = test_suite.validate_template_driven_generation();
        assert!(result.is_ok(), "Template-driven generation test failed: {:?}", result.err());
        
        assert!(!test_suite.test_results.is_empty());
        assert!(test_suite.test_results[0].contains("Template-driven generation"));
    }

    #[test]
    fn test_forced_value_insertion_only() {
        let mut test_suite = TemplatingCapabilityIntegrationTests::new();
        
        let result = test_suite.validate_forced_value_insertion();
        assert!(result.is_ok(), "Forced value insertion test failed: {:?}", result.err());
        
        // Should have results for all forced value types tested
        let forced_value_results: Vec<_> = test_suite.test_results.iter()
            .filter(|r| r.contains("Forced value insertion"))
            .collect();
        assert!(forced_value_results.len() >= 16); // At least 16 different forced value patterns
    }

    #[test]
    fn test_misalignment_handling_only() {
        let mut test_suite = TemplatingCapabilityIntegrationTests::new();
        
        let result = test_suite.validate_misalignment_handling();
        assert!(result.is_ok(), "Misalignment handling test failed: {:?}", result.err());
        
        let misalignment_results: Vec<_> = test_suite.test_results.iter()
            .filter(|r| r.contains("Misalignment handling"))
            .collect();
        assert!(misalignment_results.len() >= 4); // At least 4 different misalignment scenarios
    }

    #[test]
    fn test_state_management_only() {
        let mut test_suite = TemplatingCapabilityIntegrationTests::new();
        
        let result = test_suite.validate_state_management();
        assert!(result.is_ok(), "State management test failed: {:?}", result.err());
        
        assert!(test_suite.test_results.iter().any(|r| r.contains("State management")));
    }

    #[test]
    fn test_template_type_coverage_only() {
        let mut test_suite = TemplatingCapabilityIntegrationTests::new();
        
        let result = test_suite.validate_template_type_coverage();
        assert!(result.is_ok(), "Template type coverage test failed: {:?}", result.err());
        
        let coverage_results: Vec<_> = test_suite.test_results.iter()
            .filter(|r| r.contains("Template type coverage"))
            .collect();
        assert!(coverage_results.len() >= 4); // All 4 template types should be covered
    }

    #[test]
    fn test_error_handling_only() {
        let mut test_suite = TemplatingCapabilityIntegrationTests::new();
        
        let result = test_suite.validate_error_handling();
        assert!(result.is_ok(), "Error handling test failed: {:?}", result.err());
        
        let error_results: Vec<_> = test_suite.test_results.iter()
            .filter(|r| r.contains("Error handling"))
            .collect();
        assert!(error_results.len() >= 2); // At least 2 error handling scenarios
    }
}