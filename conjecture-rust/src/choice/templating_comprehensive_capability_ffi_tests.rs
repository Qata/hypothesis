//! Comprehensive FFI integration tests for Choice templating and forcing system capability
//!
//! This module provides comprehensive PyO3 and FFI integration tests that validate
//! the complete Choice templating and forcing system capability. Tests focus on
//! validating the entire capability's behavior and interface contracts rather than
//! individual functions, following the architectural blueprint patterns.

use crate::choice::{
    templating::*,
    ChoiceType, ChoiceValue, Constraints, ChoiceNode,
    IntegerConstraints, BooleanConstraints, FloatConstraints, StringConstraints, BytesConstraints,
    StringAlphabet, BytesMask
};
use std::collections::VecDeque;

/// FFI-compatible wrapper for the Choice templating and forcing system capability
/// This provides the interface that would be exposed through PyO3 bindings
#[derive(Debug, Clone)]
pub struct TemplatingCapabilityFFI {
    template_engine: TemplateEngine,
    debug_enabled: bool,
}

impl TemplatingCapabilityFFI {
    /// Create new templating capability instance
    pub fn new() -> Self {
        Self {
            template_engine: TemplateEngine::new(),
            debug_enabled: false,
        }
    }

    /// Enable debug mode for capability testing
    pub fn enable_debug(&mut self) {
        self.debug_enabled = true;
        self.template_engine = self.template_engine.clone().with_debug();
    }

    /// Add template-driven choice generation pattern
    pub fn add_template_pattern(&mut self, template_type: i32, bias: f64, count: Option<usize>) -> Result<(), String> {
        let template_type = match template_type {
            0 => TemplateType::Simplest,
            1 => TemplateType::AtIndex(count.unwrap_or(0)),
            2 => TemplateType::Biased { bias },
            3 => TemplateType::Custom { name: format!("custom_{}", count.unwrap_or(0)) },
            _ => return Err("Invalid template type".to_string()),
        };

        let template = match count {
            Some(c) => ChoiceTemplate::with_count(template_type, c),
            None => ChoiceTemplate::unlimited(template_type),
        };

        self.template_engine.add_entry(TemplateEntry::Template(template));
        Ok(())
    }

    /// Add forced value insertion for guided test construction
    pub fn add_forced_value(&mut self, value_type: i32, value_data: Vec<u8>) -> Result<(), String> {
        let choice_value = match value_type {
            0 => { // Integer
                if value_data.len() >= 16 {
                    let mut bytes = [0u8; 16];
                    bytes.copy_from_slice(&value_data[0..16]);
                    ChoiceValue::Integer(i128::from_le_bytes(bytes))
                } else {
                    return Err("Invalid integer data".to_string());
                }
            },
            1 => { // Boolean
                if !value_data.is_empty() {
                    ChoiceValue::Boolean(value_data[0] != 0)
                } else {
                    return Err("Invalid boolean data".to_string());
                }
            },
            2 => { // Float
                if value_data.len() >= 8 {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&value_data[0..8]);
                    ChoiceValue::Float(f64::from_le_bytes(bytes))
                } else {
                    return Err("Invalid float data".to_string());
                }
            },
            3 => { // String
                match String::from_utf8(value_data) {
                    Ok(s) => ChoiceValue::String(s),
                    Err(_) => return Err("Invalid string data".to_string()),
                }
            },
            4 => { // Bytes
                ChoiceValue::Bytes(value_data)
            },
            _ => return Err("Invalid value type".to_string()),
        };

        self.template_engine.add_entry(TemplateEntry::DirectValue(choice_value));
        Ok(())
    }

    /// Process template-driven choice generation with complete capability validation
    pub fn process_templated_choice(&mut self, choice_type: i32, constraints_data: Vec<u8>) -> Result<Vec<u8>, String> {
        let choice_type = match choice_type {
            0 => ChoiceType::Integer,
            1 => ChoiceType::Boolean,
            2 => ChoiceType::Float,
            3 => ChoiceType::String,
            4 => ChoiceType::Bytes,
            _ => return Err("Invalid choice type".to_string()),
        };

        let constraints = self.decode_constraints(choice_type, constraints_data)?;
        
        match self.template_engine.process_next_template(choice_type, &constraints) {
            Ok(Some(node)) => self.encode_choice_node(node),
            Ok(None) => Ok(vec![]), // No more templates
            Err(e) => Err(format!("Template processing error: {}", e)),
        }
    }

    /// Validate complete templating capability with comprehensive test patterns
    pub fn validate_capability(&mut self) -> Result<Vec<String>, String> {
        let mut validation_results = Vec::new();

        // Test 1: Template-driven choice generation
        self.template_engine.reset();
        self.template_engine.add_entries(vec![
            TemplateEntry::Template(ChoiceTemplate::simplest()),
            TemplateEntry::Template(ChoiceTemplate::at_index(5)),
            TemplateEntry::Template(ChoiceTemplate::biased(0.7)),
        ]);

        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        });

        for i in 0..3 {
            match self.template_engine.process_next_template(ChoiceType::Integer, &int_constraints) {
                Ok(Some(_)) => validation_results.push(format!("Template {} processed successfully", i + 1)),
                Ok(None) => return Err("Unexpected end of templates".to_string()),
                Err(e) => return Err(format!("Template processing failed: {}", e)),
            }
        }

        // Test 2: Forced value insertion
        self.template_engine.reset();
        self.template_engine.add_entries(vec![
            TemplateEntry::DirectValue(ChoiceValue::Integer(42)),
            TemplateEntry::DirectValue(ChoiceValue::Boolean(true)),
            TemplateEntry::DirectValue(ChoiceValue::Float(3.14)),
        ]);

        let forced_values = vec![
            (ChoiceType::Integer, &int_constraints),
            (ChoiceType::Boolean, &Constraints::Boolean(BooleanConstraints::default())),
            (ChoiceType::Float, &Constraints::Float(FloatConstraints {
                min_value: 0.0,
                max_value: 10.0,
                allow_nan: false,
                smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
            })),
        ];

        for (i, (choice_type, constraints)) in forced_values.iter().enumerate() {
            match self.template_engine.process_next_template(*choice_type, constraints) {
                Ok(Some(node)) => {
                    if node.was_forced {
                        validation_results.push(format!("Forced value {} inserted successfully", i + 1));
                    } else {
                        return Err("Value was not marked as forced".to_string());
                    }
                },
                Ok(None) => return Err("Unexpected end of forced values".to_string()),
                Err(e) => return Err(format!("Forced value processing failed: {}", e)),
            }
        }

        // Test 3: Misalignment handling and fallback
        self.template_engine.reset();
        self.template_engine.add_entry(TemplateEntry::DirectValue(ChoiceValue::Integer(200))); // Outside constraints

        match self.template_engine.process_next_template(ChoiceType::Integer, &int_constraints) {
            Ok(Some(node)) => {
                if self.template_engine.has_misalignment() {
                    validation_results.push("Misalignment detected and handled correctly".to_string());
                    if let ChoiceValue::Integer(val) = node.value {
                        if val == 0 { // Should fall back to shrink_towards value
                            validation_results.push("Fallback to simplest choice successful".to_string());
                        }
                    }
                } else {
                    return Err("Misalignment not detected".to_string());
                }
            },
            Ok(None) => return Err("Template processing returned None".to_string()),
            Err(e) => return Err(format!("Misalignment test failed: {}", e)),
        }

        // Test 4: Template usage counting and exhaustion
        self.template_engine.reset();
        let limited_template = ChoiceTemplate::with_count(TemplateType::Simplest, 2);
        self.template_engine.add_entry(TemplateEntry::Template(limited_template));

        for i in 0..2 {
            match self.template_engine.process_next_template(ChoiceType::Integer, &int_constraints) {
                Ok(Some(_)) => validation_results.push(format!("Limited template use {} successful", i + 1)),
                Ok(None) => return Err("Template exhausted prematurely".to_string()),
                Err(e) => return Err(format!("Limited template failed: {}", e)),
            }
        }

        // Verify template is exhausted
        match self.template_engine.process_next_template(ChoiceType::Integer, &int_constraints) {
            Ok(None) => validation_results.push("Template exhaustion handled correctly".to_string()),
            Ok(Some(_)) => return Err("Template not properly exhausted".to_string()),
            Err(_) => return Err("Template exhaustion error handling failed".to_string()),
        }

        // Test 5: Complex template patterns with state management
        self.template_engine.reset();
        let state_backup = self.template_engine.clone_state();
        
        self.template_engine.add_entries(vec![
            TemplateEntry::Template(ChoiceTemplate::unlimited(TemplateType::Simplest)),
            TemplateEntry::DirectValue(ChoiceValue::Integer(99)),
        ]);

        // Process some templates
        let _ = self.template_engine.process_next_template(ChoiceType::Integer, &int_constraints);
        let _ = self.template_engine.process_next_template(ChoiceType::Integer, &int_constraints);

        // Restore state
        self.template_engine.restore_state(state_backup);
        
        if self.template_engine.processed_count() == 0 && self.template_engine.remaining_count() == 0 {
            validation_results.push("State management working correctly".to_string());
        } else {
            return Err("State restoration failed".to_string());
        }

        Ok(validation_results)
    }

    /// Get comprehensive capability status and metrics
    pub fn get_capability_status(&self) -> Vec<String> {
        vec![
            format!("Remaining templates: {}", self.template_engine.remaining_count()),
            format!("Processed count: {}", self.template_engine.processed_count()),
            format!("Has misalignment: {}", self.template_engine.has_misalignment()),
            format!("Debug enabled: {}", self.debug_enabled),
            format!("Engine state: {}", self.template_engine.debug_info()),
        ]
    }

    // Helper methods for FFI data encoding/decoding

    fn decode_constraints(&self, choice_type: ChoiceType, data: Vec<u8>) -> Result<Constraints, String> {
        match choice_type {
            ChoiceType::Integer => {
                if data.len() >= 32 {
                    let min_bytes = &data[0..16];
                    let max_bytes = &data[16..32];
                    let min_value = i128::from_le_bytes(min_bytes.try_into().unwrap());
                    let max_value = i128::from_le_bytes(max_bytes.try_into().unwrap());
                    Ok(Constraints::Integer(IntegerConstraints {
                        min_value: Some(min_value),
                        max_value: Some(max_value),
                        weights: None,
                        shrink_towards: Some(0),
                    }))
                } else {
                    Ok(Constraints::Integer(IntegerConstraints::default()))
                }
            },
            ChoiceType::Boolean => Ok(Constraints::Boolean(BooleanConstraints::default())),
            ChoiceType::Float => {
                if data.len() >= 16 {
                    let min_bytes = &data[0..8];
                    let max_bytes = &data[8..16];
                    let min_value = f64::from_le_bytes(min_bytes.try_into().unwrap());
                    let max_value = f64::from_le_bytes(max_bytes.try_into().unwrap());
                    Ok(Constraints::Float(FloatConstraints {
                        min_value,
                        max_value,
                        allow_nan: false,
                        smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
                    }))
                } else {
                    Ok(Constraints::Float(FloatConstraints::default()))
                }
            },
            ChoiceType::String => {
                let min_size = if data.len() >= 8 {
                    usize::from_le_bytes(data[0..8].try_into().unwrap())
                } else {
                    0
                };
                let max_size = if data.len() >= 16 {
                    usize::from_le_bytes(data[8..16].try_into().unwrap())
                } else {
                    1000
                };
                Ok(Constraints::String(StringConstraints {
                    min_size,
                    max_size,
                    intervals: StringAlphabet::ascii(),
                }))
            },
            ChoiceType::Bytes => {
                let min_size = if data.len() >= 8 {
                    usize::from_le_bytes(data[0..8].try_into().unwrap())
                } else {
                    0
                };
                let max_size = if data.len() >= 16 {
                    usize::from_le_bytes(data[8..16].try_into().unwrap())
                } else {
                    1000
                };
                Ok(Constraints::Bytes(BytesConstraints {
                    min_size,
                    max_size,
                    mask: BytesMask::all(),
                }))
            },
        }
    }

    fn encode_choice_node(&self, node: ChoiceNode) -> Result<Vec<u8>, String> {
        let mut result = Vec::new();
        
        // Encode choice type (1 byte)
        result.push(match node.choice_type {
            ChoiceType::Integer => 0,
            ChoiceType::Boolean => 1,
            ChoiceType::Float => 2,
            ChoiceType::String => 3,
            ChoiceType::Bytes => 4,
        });

        // Encode was_forced flag (1 byte)
        result.push(if node.was_forced { 1 } else { 0 });

        // Encode value
        match node.value {
            ChoiceValue::Integer(val) => {
                result.extend_from_slice(&val.to_le_bytes());
            },
            ChoiceValue::Boolean(val) => {
                result.push(if val { 1 } else { 0 });
            },
            ChoiceValue::Float(val) => {
                result.extend_from_slice(&val.to_le_bytes());
            },
            ChoiceValue::String(val) => {
                let bytes = val.into_bytes();
                result.extend_from_slice(&bytes.len().to_le_bytes());
                result.extend_from_slice(&bytes);
            },
            ChoiceValue::Bytes(val) => {
                result.extend_from_slice(&val.len().to_le_bytes());
                result.extend_from_slice(&val);
            },
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_capability_creation() {
        let capability = TemplatingCapabilityFFI::new();
        assert!(!capability.debug_enabled);
        assert_eq!(capability.template_engine.remaining_count(), 0);
    }

    #[test]
    fn test_ffi_template_pattern_addition() {
        let mut capability = TemplatingCapabilityFFI::new();
        
        // Add simplest template pattern
        let result = capability.add_template_pattern(0, 0.0, None);
        assert!(result.is_ok());
        assert_eq!(capability.template_engine.remaining_count(), 1);

        // Add index template pattern
        let result = capability.add_template_pattern(1, 0.0, Some(5));
        assert!(result.is_ok());
        assert_eq!(capability.template_engine.remaining_count(), 2);

        // Add biased template pattern
        let result = capability.add_template_pattern(2, 0.7, None);
        assert!(result.is_ok());
        assert_eq!(capability.template_engine.remaining_count(), 3);

        // Add custom template pattern
        let result = capability.add_template_pattern(3, 0.0, Some(1));
        assert!(result.is_ok());
        assert_eq!(capability.template_engine.remaining_count(), 4);

        // Invalid template type
        let result = capability.add_template_pattern(99, 0.0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ffi_forced_value_insertion() {
        let mut capability = TemplatingCapabilityFFI::new();

        // Add integer forced value
        let int_bytes = 42i128.to_le_bytes().to_vec();
        let result = capability.add_forced_value(0, int_bytes);
        assert!(result.is_ok());
        assert_eq!(capability.template_engine.remaining_count(), 1);

        // Add boolean forced value
        let result = capability.add_forced_value(1, vec![1]);
        assert!(result.is_ok());
        assert_eq!(capability.template_engine.remaining_count(), 2);

        // Add float forced value
        let float_bytes = 3.14f64.to_le_bytes().to_vec();
        let result = capability.add_forced_value(2, float_bytes);
        assert!(result.is_ok());
        assert_eq!(capability.template_engine.remaining_count(), 3);

        // Add string forced value
        let result = capability.add_forced_value(3, b"test".to_vec());
        assert!(result.is_ok());
        assert_eq!(capability.template_engine.remaining_count(), 4);

        // Add bytes forced value
        let result = capability.add_forced_value(4, vec![1, 2, 3, 4]);
        assert!(result.is_ok());
        assert_eq!(capability.template_engine.remaining_count(), 5);

        // Invalid value type
        let result = capability.add_forced_value(99, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ffi_templated_choice_processing() {
        let mut capability = TemplatingCapabilityFFI::new();

        // Add a forced integer value
        let int_bytes = 42i128.to_le_bytes().to_vec();
        capability.add_forced_value(0, int_bytes).unwrap();

        // Create constraints for integer processing
        let mut constraints_data = Vec::new();
        constraints_data.extend_from_slice(&0i128.to_le_bytes()); // min_value
        constraints_data.extend_from_slice(&100i128.to_le_bytes()); // max_value

        // Process the templated choice
        let result = capability.process_templated_choice(0, constraints_data);
        assert!(result.is_ok());
        
        let encoded_node = result.unwrap();
        assert!(!encoded_node.is_empty());
        
        // Verify choice type is encoded correctly (first byte should be 0 for Integer)
        assert_eq!(encoded_node[0], 0);
        
        // Verify was_forced flag is set (second byte should be 1)
        assert_eq!(encoded_node[1], 1);
    }

    #[test]
    fn test_ffi_capability_validation_comprehensive() {
        let mut capability = TemplatingCapabilityFFI::new();
        capability.enable_debug();
        
        let validation_results = capability.validate_capability();
        assert!(validation_results.is_ok());
        
        let results = validation_results.unwrap();
        assert!(!results.is_empty());
        
        // Check that all major validation points passed
        let expected_validations = vec![
            "Template 1 processed successfully",
            "Template 2 processed successfully", 
            "Template 3 processed successfully",
            "Forced value 1 inserted successfully",
            "Forced value 2 inserted successfully",
            "Forced value 3 inserted successfully",
            "Misalignment detected and handled correctly",
            "Fallback to simplest choice successful",
            "Limited template use 1 successful",
            "Limited template use 2 successful",
            "Template exhaustion handled correctly",
            "State management working correctly",
        ];
        
        for expected in expected_validations {
            assert!(results.contains(&expected.to_string()), 
                "Missing validation: {}", expected);
        }
    }

    #[test]
    fn test_ffi_capability_status_reporting() {
        let mut capability = TemplatingCapabilityFFI::new();
        capability.enable_debug();
        
        capability.add_template_pattern(0, 0.0, None).unwrap();
        capability.add_forced_value(0, 42i128.to_le_bytes().to_vec()).unwrap();
        
        let status = capability.get_capability_status();
        assert_eq!(status.len(), 5);
        
        assert!(status[0].contains("Remaining templates: 2"));
        assert!(status[1].contains("Processed count: 0"));
        assert!(status[2].contains("Has misalignment: false"));
        assert!(status[3].contains("Debug enabled: true"));
        assert!(status[4].contains("Engine state:"));
    }

    #[test]
    fn test_ffi_constraint_encoding_decoding() {
        let capability = TemplatingCapabilityFFI::new();

        // Test integer constraints
        let mut int_data = Vec::new();
        int_data.extend_from_slice(&(-10i128).to_le_bytes());
        int_data.extend_from_slice(&10i128.to_le_bytes());
        
        let constraints = capability.decode_constraints(ChoiceType::Integer, int_data).unwrap();
        if let Constraints::Integer(int_constraints) = constraints {
            assert_eq!(int_constraints.min_value, Some(-10));
            assert_eq!(int_constraints.max_value, Some(10));
        } else {
            panic!("Expected integer constraints");
        }

        // Test float constraints
        let mut float_data = Vec::new();
        float_data.extend_from_slice(&(-1.0f64).to_le_bytes());
        float_data.extend_from_slice(&1.0f64.to_le_bytes());
        
        let constraints = capability.decode_constraints(ChoiceType::Float, float_data).unwrap();
        if let Constraints::Float(float_constraints) = constraints {
            assert_eq!(float_constraints.min_value, -1.0);
            assert_eq!(float_constraints.max_value, 1.0);
        } else {
            panic!("Expected float constraints");
        }

        // Test string constraints
        let mut string_data = Vec::new();
        string_data.extend_from_slice(&5usize.to_le_bytes()); // min_size
        string_data.extend_from_slice(&20usize.to_le_bytes()); // max_size
        
        let constraints = capability.decode_constraints(ChoiceType::String, string_data).unwrap();
        if let Constraints::String(string_constraints) = constraints {
            assert_eq!(string_constraints.min_size, 5);
            assert_eq!(string_constraints.max_size, 20);
        } else {
            panic!("Expected string constraints");
        }
    }

    #[test]
    fn test_ffi_choice_node_encoding() {
        let capability = TemplatingCapabilityFFI::new();

        // Test integer node encoding
        let int_node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            true,
        );
        
        let encoded = capability.encode_choice_node(int_node).unwrap();
        assert!(!encoded.is_empty());
        assert_eq!(encoded[0], 0); // Integer type
        assert_eq!(encoded[1], 1); // was_forced = true
        
        // Extract and verify the integer value
        let value_bytes = &encoded[2..18];
        let decoded_value = i128::from_le_bytes(value_bytes.try_into().unwrap());
        assert_eq!(decoded_value, 42);

        // Test boolean node encoding
        let bool_node = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints::default()),
            false,
        );
        
        let encoded = capability.encode_choice_node(bool_node).unwrap();
        assert_eq!(encoded[0], 1); // Boolean type
        assert_eq!(encoded[1], 0); // was_forced = false
        assert_eq!(encoded[2], 1); // boolean value = true
    }

    #[test]
    fn test_ffi_complete_workflow_integration() {
        let mut capability = TemplatingCapabilityFFI::new();
        capability.enable_debug();

        // Step 1: Add comprehensive template patterns
        capability.add_template_pattern(0, 0.0, None).unwrap(); // Simplest
        capability.add_template_pattern(1, 0.0, Some(3)).unwrap(); // At index 3
        capability.add_template_pattern(2, 0.8, None).unwrap(); // Biased
        
        // Step 2: Add forced values
        capability.add_forced_value(0, 100i128.to_le_bytes().to_vec()).unwrap();
        capability.add_forced_value(1, vec![1]).unwrap(); // true

        // Step 3: Process all templates with different constraints
        let mut constraints_data = Vec::new();
        constraints_data.extend_from_slice(&0i128.to_le_bytes());
        constraints_data.extend_from_slice(&1000i128.to_le_bytes());

        // Process integer templates
        for i in 0..4 {
            let result = capability.process_templated_choice(0, constraints_data.clone());
            assert!(result.is_ok(), "Failed to process template {}", i + 1);
            let encoded = result.unwrap();
            if !encoded.is_empty() {
                assert_eq!(encoded[0], 0); // Should be integer type
            }
        }

        // Process boolean template
        let result = capability.process_templated_choice(1, vec![]);
        assert!(result.is_ok());
        let encoded = result.unwrap();
        if !encoded.is_empty() {
            assert_eq!(encoded[0], 1); // Should be boolean type
            assert_eq!(encoded[1], 1); // Should be forced
        }

        // Step 4: Verify no more templates remain
        let result = capability.process_templated_choice(0, constraints_data);
        assert!(result.is_ok());
        let encoded = result.unwrap();
        assert!(encoded.is_empty(), "Expected no more templates");

        // Step 5: Check final status
        let status = capability.get_capability_status();
        assert!(status[0].contains("Remaining templates: 0"));
        assert!(status[1].contains("Processed count: 5"));
    }

    #[test]
    fn test_ffi_error_handling_comprehensive() {
        let mut capability = TemplatingCapabilityFFI::new();

        // Test invalid template type
        let result = capability.add_template_pattern(99, 0.0, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid template type"));

        // Test invalid value type for forced value
        let result = capability.add_forced_value(99, vec![]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid value type"));

        // Test invalid choice type for processing
        let result = capability.process_templated_choice(99, vec![]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid choice type"));

        // Test processing with no templates
        let result = capability.process_templated_choice(0, vec![]);
        assert!(result.is_ok());
        let encoded = result.unwrap();
        assert!(encoded.is_empty()); // Should return empty for no templates

        // Test insufficient data for values
        let result = capability.add_forced_value(0, vec![1, 2]); // Insufficient for i128
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid integer data"));

        // Test invalid string data
        let result = capability.add_forced_value(3, vec![0xFF, 0xFF, 0xFF, 0xFF]); // Invalid UTF-8
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid string data"));
    }
}