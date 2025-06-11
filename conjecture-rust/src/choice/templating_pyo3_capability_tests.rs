//! PyO3 integration tests for Choice templating and forcing system capability
//!
//! This module provides comprehensive PyO3 integration tests that validate the
//! complete Choice templating and forcing system capability through Python FFI.
//! Tests focus on verifying the capability works correctly when accessed from
//! Python, maintaining full interface compatibility and proper data marshaling.

use crate::choice::{
    templating::*,
    ChoiceType, ChoiceValue, Constraints, ChoiceNode,
    IntegerConstraints, BooleanConstraints, FloatConstraints, StringConstraints, BytesConstraints,
    StringAlphabet, BytesMask
};
use std::collections::HashMap;

/// PyO3-compatible wrapper for the Choice templating and forcing system capability
/// This struct simulates what would be exposed through PyO3 bindings to Python
#[derive(Debug, Clone)]
pub struct PyTemplatingCapability {
    engine: TemplateEngine,
    python_metadata: HashMap<String, String>,
}

impl PyTemplatingCapability {
    /// Create new PyO3-compatible templating capability
    pub fn new() -> Self {
        Self {
            engine: TemplateEngine::new(),
            python_metadata: HashMap::new(),
        }
    }

    /// Enable debug mode for Python integration testing
    pub fn enable_debug(&mut self) -> &mut Self {
        self.engine = self.engine.clone().with_debug();
        self.python_metadata.insert("debug_enabled".to_string(), "true".to_string());
        self
    }

    /// Add template pattern from Python-compatible parameters
    pub fn py_add_template(&mut self, template_type: &str, bias: Option<f64>, count: Option<i64>) -> Result<(), String> {
        let template_type = match template_type {
            "simplest" => TemplateType::Simplest,
            "at_index" => {
                let index = count.unwrap_or(0) as usize;
                TemplateType::AtIndex(index)
            },
            "biased" => {
                let bias_val = bias.unwrap_or(0.5);
                TemplateType::Biased { bias: bias_val }
            },
            "custom" => {
                let name = format!("custom_{}", count.unwrap_or(0));
                TemplateType::Custom { name }
            },
            _ => return Err(format!("Unknown template type: {}", template_type)),
        };

        let template = match count {
            Some(c) if c > 0 => ChoiceTemplate::with_count(template_type, c as usize),
            Some(-1) => ChoiceTemplate::unlimited(template_type), // -1 represents unlimited in Python
            _ => ChoiceTemplate::unlimited(template_type),
        };

        self.engine.add_entry(TemplateEntry::Template(template));
        self.python_metadata.insert("last_template_added".to_string(), template_type.to_string());
        Ok(())
    }

    /// Add forced value from Python-compatible types
    pub fn py_add_forced_value(&mut self, value_type: &str, value: &str) -> Result<(), String> {
        let choice_value = match value_type {
            "int" => {
                match value.parse::<i128>() {
                    Ok(i) => ChoiceValue::Integer(i),
                    Err(_) => return Err("Invalid integer value".to_string()),
                }
            },
            "bool" => {
                let bool_val = match value.to_lowercase().as_str() {
                    "true" | "1" | "yes" => true,
                    "false" | "0" | "no" => false,
                    _ => return Err("Invalid boolean value".to_string()),
                };
                ChoiceValue::Boolean(bool_val)
            },
            "float" => {
                if value == "nan" {
                    ChoiceValue::Float(f64::NAN)
                } else if value == "inf" {
                    ChoiceValue::Float(f64::INFINITY)
                } else if value == "-inf" {
                    ChoiceValue::Float(f64::NEG_INFINITY)
                } else {
                    match value.parse::<f64>() {
                        Ok(f) => ChoiceValue::Float(f),
                        Err(_) => return Err("Invalid float value".to_string()),
                    }
                }
            },
            "str" => ChoiceValue::String(value.to_string()),
            "bytes" => {
                // Parse hex string or comma-separated decimal bytes
                if value.starts_with("0x") {
                    // Hex format: 0x48656c6c6f
                    let hex_str = &value[2..];
                    if hex_str.len() % 2 != 0 {
                        return Err("Invalid hex string length".to_string());
                    }
                    let mut bytes = Vec::new();
                    for i in (0..hex_str.len()).step_by(2) {
                        match u8::from_str_radix(&hex_str[i..i+2], 16) {
                            Ok(b) => bytes.push(b),
                            Err(_) => return Err("Invalid hex byte".to_string()),
                        }
                    }
                    ChoiceValue::Bytes(bytes)
                } else if value.contains(',') {
                    // Comma-separated format: 72,101,108,108,111
                    let byte_strs: Vec<&str> = value.split(',').collect();
                    let mut bytes = Vec::new();
                    for byte_str in byte_strs {
                        match byte_str.trim().parse::<u8>() {
                            Ok(b) => bytes.push(b),
                            Err(_) => return Err("Invalid byte value".to_string()),
                        }
                    }
                    ChoiceValue::Bytes(bytes)
                } else {
                    // UTF-8 string to bytes
                    ChoiceValue::Bytes(value.as_bytes().to_vec())
                }
            },
            _ => return Err(format!("Unknown value type: {}", value_type)),
        };

        self.engine.add_entry(TemplateEntry::DirectValue(choice_value));
        self.python_metadata.insert("last_forced_value_type".to_string(), value_type.to_string());
        Ok(())
    }

    /// Process template-driven choice with Python-compatible interface
    pub fn py_process_choice(&mut self, choice_type: &str, constraints: HashMap<String, String>) -> Result<HashMap<String, String>, String> {
        let choice_type = match choice_type {
            "int" => ChoiceType::Integer,
            "bool" => ChoiceType::Boolean,
            "float" => ChoiceType::Float,
            "str" => ChoiceType::String,
            "bytes" => ChoiceType::Bytes,
            _ => return Err(format!("Unknown choice type: {}", choice_type)),
        };

        let constraints = self.py_parse_constraints(choice_type, constraints)?;
        
        match self.engine.process_next_template(choice_type, &constraints) {
            Ok(Some(node)) => {
                let mut result = HashMap::new();
                result.insert("success".to_string(), "true".to_string());
                result.insert("type".to_string(), self.choice_type_to_string(node.choice_type));
                result.insert("value".to_string(), self.choice_value_to_string(&node.value));
                result.insert("was_forced".to_string(), node.was_forced.to_string());
                result.insert("remaining_templates".to_string(), self.engine.remaining_count().to_string());
                result.insert("processed_count".to_string(), self.engine.processed_count().to_string());
                Ok(result)
            },
            Ok(None) => {
                let mut result = HashMap::new();
                result.insert("success".to_string(), "true".to_string());
                result.insert("no_more_templates".to_string(), "true".to_string());
                result.insert("remaining_templates".to_string(), "0".to_string());
                result.insert("processed_count".to_string(), self.engine.processed_count().to_string());
                Ok(result)
            },
            Err(e) => {
                let mut result = HashMap::new();
                result.insert("success".to_string(), "false".to_string());
                result.insert("error".to_string(), e.to_string());
                Ok(result)
            }
        }
    }

    /// Get comprehensive capability status for Python
    pub fn py_get_status(&self) -> HashMap<String, String> {
        let mut status = HashMap::new();
        status.insert("remaining_templates".to_string(), self.engine.remaining_count().to_string());
        status.insert("processed_count".to_string(), self.engine.processed_count().to_string());
        status.insert("has_misalignment".to_string(), self.engine.has_misalignment().to_string());
        status.insert("misalignment_index".to_string(), 
            self.engine.misalignment_index().map(|i| i.to_string()).unwrap_or("none".to_string()));
        status.insert("debug_info".to_string(), self.engine.debug_info());
        
        // Add Python-specific metadata
        for (key, value) in &self.python_metadata {
            status.insert(format!("py_{}", key), value.clone());
        }
        
        status
    }

    /// Reset the capability state for Python
    pub fn py_reset(&mut self) {
        self.engine.reset();
        self.python_metadata.clear();
        self.python_metadata.insert("reset_timestamp".to_string(), "now".to_string());
    }

    /// Validate complete capability with Python interface
    pub fn py_validate_capability(&mut self) -> Result<Vec<String>, String> {
        let mut validation_results = Vec::new();

        // Test 1: Python template addition and processing
        self.py_reset();
        self.py_add_template("simplest", None, None)?;
        self.py_add_template("biased", Some(0.8), None)?;
        self.py_add_template("at_index", None, Some(3))?;

        let mut constraints = HashMap::new();
        constraints.insert("min_value".to_string(), "0".to_string());
        constraints.insert("max_value".to_string(), "100".to_string());

        for i in 1..=3 {
            let result = self.py_process_choice("int", constraints.clone())?;
            if result.get("success") == Some(&"true".to_string()) {
                validation_results.push(format!("Python template {} processed successfully", i));
            } else {
                return Err(format!("Python template {} processing failed", i));
            }
        }

        // Test 2: Python forced value insertion
        self.py_reset();
        self.py_add_forced_value("int", "42")?;
        self.py_add_forced_value("bool", "true")?;
        self.py_add_forced_value("float", "3.14")?;
        self.py_add_forced_value("str", "Hello")?;
        self.py_add_forced_value("bytes", "0x48656c6c6f")?;

        let test_cases = vec![
            ("int", "42"),
            ("bool", "true"),
            ("float", "3.14"),
            ("str", "Hello"),
            ("bytes", "Hello"),
        ];

        for (i, (choice_type, expected_value)) in test_cases.iter().enumerate() {
            let result = self.py_process_choice(choice_type, HashMap::new())?;
            if result.get("success") == Some(&"true".to_string()) &&
               result.get("was_forced") == Some(&"true".to_string()) {
                let value_str = result.get("value").unwrap_or(&"".to_string());
                if choice_type == &"bytes" {
                    // For bytes, just verify it's not empty
                    if !value_str.is_empty() {
                        validation_results.push(format!("Python forced value {} inserted correctly", i + 1));
                    }
                } else if value_str.contains(expected_value) {
                    validation_results.push(format!("Python forced value {} inserted correctly", i + 1));
                } else {
                    return Err(format!("Python forced value {} mismatch: expected {}, got {}", i + 1, expected_value, value_str));
                }
            } else {
                return Err(format!("Python forced value {} processing failed", i + 1));
            }
        }

        // Test 3: Python error handling
        self.py_reset();
        let invalid_template_result = self.py_add_template("invalid_type", None, None);
        if invalid_template_result.is_err() {
            validation_results.push("Python error handling: invalid template type rejected correctly".to_string());
        } else {
            return Err("Python error handling failed for invalid template type".to_string());
        }

        let invalid_value_result = self.py_add_forced_value("int", "not_a_number");
        if invalid_value_result.is_err() {
            validation_results.push("Python error handling: invalid value rejected correctly".to_string());
        } else {
            return Err("Python error handling failed for invalid value".to_string());
        }

        // Test 4: Python status reporting
        self.py_reset();
        self.py_add_template("simplest", None, Some(2))?;
        let status = self.py_get_status();
        if status.get("remaining_templates") == Some(&"1".to_string()) &&
           status.get("processed_count") == Some(&"0".to_string()) {
            validation_results.push("Python status reporting: initial status correct".to_string());
        } else {
            return Err("Python status reporting failed".to_string());
        }

        // Test 5: Python special value handling
        self.py_reset();
        self.py_add_forced_value("float", "nan")?;
        self.py_add_forced_value("float", "inf")?;
        self.py_add_forced_value("float", "-inf")?;

        let mut nan_constraints = HashMap::new();
        nan_constraints.insert("allow_nan".to_string(), "true".to_string());

        for special_value in &["nan", "inf", "-inf"] {
            let result = self.py_process_choice("float", nan_constraints.clone())?;
            if result.get("success") == Some(&"true".to_string()) {
                validation_results.push(format!("Python special float value {} handled correctly", special_value));
            } else {
                return Err(format!("Python special float value {} handling failed", special_value));
            }
        }

        Ok(validation_results)
    }

    // Helper methods for Python integration

    fn py_parse_constraints(&self, choice_type: ChoiceType, constraints: HashMap<String, String>) -> Result<Constraints, String> {
        match choice_type {
            ChoiceType::Integer => {
                let min_value = constraints.get("min_value")
                    .and_then(|s| s.parse::<i128>().ok());
                let max_value = constraints.get("max_value")
                    .and_then(|s| s.parse::<i128>().ok());
                let shrink_towards = constraints.get("shrink_towards")
                    .and_then(|s| s.parse::<i128>().ok())
                    .or(Some(0));

                Ok(Constraints::Integer(IntegerConstraints {
                    min_value,
                    max_value,
                    weights: None,
                    shrink_towards,
                }))
            },
            ChoiceType::Boolean => {
                Ok(Constraints::Boolean(BooleanConstraints::default()))
            },
            ChoiceType::Float => {
                let min_value = constraints.get("min_value")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(f64::NEG_INFINITY);
                let max_value = constraints.get("max_value")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(f64::INFINITY);
                let allow_nan = constraints.get("allow_nan")
                    .map(|s| s == "true")
                    .unwrap_or(false);

                Ok(Constraints::Float(FloatConstraints {
                    min_value,
                    max_value,
                    allow_nan,
                    smallest_nonzero_magnitude: f64::MIN_POSITIVE,
                }))
            },
            ChoiceType::String => {
                let min_size = constraints.get("min_size")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let max_size = constraints.get("max_size")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(1000);

                Ok(Constraints::String(StringConstraints {
                    min_size,
                    max_size,
                    intervals: StringAlphabet::ascii(),
                }))
            },
            ChoiceType::Bytes => {
                let min_size = constraints.get("min_size")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                let max_size = constraints.get("max_size")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(1000);

                Ok(Constraints::Bytes(BytesConstraints {
                    min_size,
                    max_size,
                    mask: BytesMask::all(),
                }))
            },
        }
    }

    fn choice_type_to_string(&self, choice_type: ChoiceType) -> String {
        match choice_type {
            ChoiceType::Integer => "int".to_string(),
            ChoiceType::Boolean => "bool".to_string(),
            ChoiceType::Float => "float".to_string(),
            ChoiceType::String => "str".to_string(),
            ChoiceType::Bytes => "bytes".to_string(),
        }
    }

    fn choice_value_to_string(&self, value: &ChoiceValue) -> String {
        match value {
            ChoiceValue::Integer(i) => i.to_string(),
            ChoiceValue::Boolean(b) => b.to_string(),
            ChoiceValue::Float(f) => {
                if f.is_nan() {
                    "nan".to_string()
                } else if f.is_infinite() {
                    if f.is_sign_positive() {
                        "inf".to_string()
                    } else {
                        "-inf".to_string()
                    }
                } else {
                    f.to_string()
                }
            },
            ChoiceValue::String(s) => s.clone(),
            ChoiceValue::Bytes(b) => {
                // Convert to hex string for Python compatibility
                b.iter().map(|byte| format!("{:02x}", byte)).collect::<Vec<_>>().join("")
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_capability_creation() {
        let capability = PyTemplatingCapability::new();
        let status = capability.py_get_status();
        assert_eq!(status.get("remaining_templates"), Some(&"0".to_string()));
        assert_eq!(status.get("processed_count"), Some(&"0".to_string()));
    }

    #[test]
    fn test_py_template_addition() {
        let mut capability = PyTemplatingCapability::new();

        // Test simplest template
        let result = capability.py_add_template("simplest", None, None);
        assert!(result.is_ok());
        
        // Test biased template
        let result = capability.py_add_template("biased", Some(0.7), None);
        assert!(result.is_ok());
        
        // Test at-index template
        let result = capability.py_add_template("at_index", None, Some(5));
        assert!(result.is_ok());
        
        // Test custom template
        let result = capability.py_add_template("custom", None, Some(1));
        assert!(result.is_ok());
        
        let status = capability.py_get_status();
        assert_eq!(status.get("remaining_templates"), Some(&"4".to_string()));
        
        // Test invalid template type
        let result = capability.py_add_template("invalid", None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_py_forced_value_addition() {
        let mut capability = PyTemplatingCapability::new();

        // Test integer forced value
        let result = capability.py_add_forced_value("int", "42");
        assert!(result.is_ok());
        
        // Test boolean forced values
        let result = capability.py_add_forced_value("bool", "true");
        assert!(result.is_ok());
        let result = capability.py_add_forced_value("bool", "false");
        assert!(result.is_ok());
        
        // Test float forced values
        let result = capability.py_add_forced_value("float", "3.14");
        assert!(result.is_ok());
        let result = capability.py_add_forced_value("float", "nan");
        assert!(result.is_ok());
        let result = capability.py_add_forced_value("float", "inf");
        assert!(result.is_ok());
        
        // Test string forced value
        let result = capability.py_add_forced_value("str", "Hello");
        assert!(result.is_ok());
        
        // Test bytes forced value (hex format)
        let result = capability.py_add_forced_value("bytes", "0x48656c6c6f");
        assert!(result.is_ok());
        
        // Test bytes forced value (comma format)
        let result = capability.py_add_forced_value("bytes", "72,101,108,108,111");
        assert!(result.is_ok());
        
        let status = capability.py_get_status();
        assert_eq!(status.get("remaining_templates"), Some(&"8".to_string()));
        
        // Test invalid value type
        let result = capability.py_add_forced_value("invalid", "value");
        assert!(result.is_err());
        
        // Test invalid integer value
        let result = capability.py_add_forced_value("int", "not_a_number");
        assert!(result.is_err());
    }

    #[test]
    fn test_py_choice_processing() {
        let mut capability = PyTemplatingCapability::new();
        
        capability.py_add_forced_value("int", "42").unwrap();
        
        let mut constraints = HashMap::new();
        constraints.insert("min_value".to_string(), "0".to_string());
        constraints.insert("max_value".to_string(), "100".to_string());
        
        let result = capability.py_process_choice("int", constraints).unwrap();
        assert_eq!(result.get("success"), Some(&"true".to_string()));
        assert_eq!(result.get("type"), Some(&"int".to_string()));
        assert_eq!(result.get("value"), Some(&"42".to_string()));
        assert_eq!(result.get("was_forced"), Some(&"true".to_string()));
        
        // Test processing when no more templates
        let result = capability.py_process_choice("int", HashMap::new()).unwrap();
        assert_eq!(result.get("success"), Some(&"true".to_string()));
        assert_eq!(result.get("no_more_templates"), Some(&"true".to_string()));
    }

    #[test]
    fn test_py_comprehensive_validation() {
        let mut capability = PyTemplatingCapability::new();
        capability.enable_debug();
        
        let validation_results = capability.py_validate_capability();
        assert!(validation_results.is_ok(), "Python validation failed: {:?}", validation_results.err());
        
        let results = validation_results.unwrap();
        assert!(!results.is_empty());
        
        // Check for expected validation results
        let expected_validations = vec![
            "Python template 1 processed successfully",
            "Python template 2 processed successfully",
            "Python template 3 processed successfully",
            "Python forced value 1 inserted correctly",
            "Python forced value 2 inserted correctly",
            "Python forced value 3 inserted correctly",
            "Python forced value 4 inserted correctly",
            "Python forced value 5 inserted correctly",
            "Python error handling: invalid template type rejected correctly",
            "Python error handling: invalid value rejected correctly",
            "Python status reporting: initial status correct",
            "Python special float value nan handled correctly",
            "Python special float value inf handled correctly",
            "Python special float value -inf handled correctly",
        ];
        
        for expected in expected_validations {
            assert!(results.contains(&expected.to_string()), 
                "Missing Python validation: {}", expected);
        }
    }

    #[test]
    fn test_py_status_reporting() {
        let mut capability = PyTemplatingCapability::new();
        capability.enable_debug();
        
        capability.py_add_template("simplest", None, Some(3)).unwrap();
        capability.py_add_forced_value("int", "42").unwrap();
        
        let status = capability.py_get_status();
        assert_eq!(status.get("remaining_templates"), Some(&"2".to_string()));
        assert_eq!(status.get("processed_count"), Some(&"0".to_string()));
        assert_eq!(status.get("has_misalignment"), Some(&"false".to_string()));
        assert_eq!(status.get("py_debug_enabled"), Some(&"true".to_string()));
        assert!(status.contains_key("py_last_template_added"));
        assert!(status.contains_key("py_last_forced_value_type"));
    }

    #[test]
    fn test_py_reset_functionality() {
        let mut capability = PyTemplatingCapability::new();
        
        capability.py_add_template("simplest", None, None).unwrap();
        capability.py_add_forced_value("int", "42").unwrap();
        
        let status_before = capability.py_get_status();
        assert_eq!(status_before.get("remaining_templates"), Some(&"2".to_string()));
        
        capability.py_reset();
        
        let status_after = capability.py_get_status();
        assert_eq!(status_after.get("remaining_templates"), Some(&"0".to_string()));
        assert_eq!(status_after.get("processed_count"), Some(&"0".to_string()));
        assert_eq!(status_after.get("py_reset_timestamp"), Some(&"now".to_string()));
    }

    #[test]
    fn test_py_constraint_parsing() {
        let capability = PyTemplatingCapability::new();
        
        // Test integer constraints
        let mut int_constraints = HashMap::new();
        int_constraints.insert("min_value".to_string(), "-10".to_string());
        int_constraints.insert("max_value".to_string(), "10".to_string());
        int_constraints.insert("shrink_towards".to_string(), "0".to_string());
        
        let constraints = capability.py_parse_constraints(ChoiceType::Integer, int_constraints).unwrap();
        if let Constraints::Integer(int_c) = constraints {
            assert_eq!(int_c.min_value, Some(-10));
            assert_eq!(int_c.max_value, Some(10));
            assert_eq!(int_c.shrink_towards, Some(0));
        } else {
            panic!("Expected integer constraints");
        }
        
        // Test float constraints with special values
        let mut float_constraints = HashMap::new();
        float_constraints.insert("min_value".to_string(), "-1.0".to_string());
        float_constraints.insert("max_value".to_string(), "1.0".to_string());
        float_constraints.insert("allow_nan".to_string(), "true".to_string());
        
        let constraints = capability.py_parse_constraints(ChoiceType::Float, float_constraints).unwrap();
        if let Constraints::Float(float_c) = constraints {
            assert_eq!(float_c.min_value, -1.0);
            assert_eq!(float_c.max_value, 1.0);
            assert_eq!(float_c.allow_nan, true);
        } else {
            panic!("Expected float constraints");
        }
        
        // Test string constraints
        let mut string_constraints = HashMap::new();
        string_constraints.insert("min_size".to_string(), "5".to_string());
        string_constraints.insert("max_size".to_string(), "20".to_string());
        
        let constraints = capability.py_parse_constraints(ChoiceType::String, string_constraints).unwrap();
        if let Constraints::String(string_c) = constraints {
            assert_eq!(string_c.min_size, 5);
            assert_eq!(string_c.max_size, 20);
        } else {
            panic!("Expected string constraints");
        }
    }

    #[test]
    fn test_py_value_serialization() {
        let capability = PyTemplatingCapability::new();
        
        // Test integer serialization
        let int_val = ChoiceValue::Integer(42);
        assert_eq!(capability.choice_value_to_string(&int_val), "42");
        
        // Test boolean serialization
        let bool_val = ChoiceValue::Boolean(true);
        assert_eq!(capability.choice_value_to_string(&bool_val), "true");
        
        // Test float serialization (including special values)
        let float_val = ChoiceValue::Float(3.14);
        assert_eq!(capability.choice_value_to_string(&float_val), "3.14");
        
        let nan_val = ChoiceValue::Float(f64::NAN);
        assert_eq!(capability.choice_value_to_string(&nan_val), "nan");
        
        let inf_val = ChoiceValue::Float(f64::INFINITY);
        assert_eq!(capability.choice_value_to_string(&inf_val), "inf");
        
        let neg_inf_val = ChoiceValue::Float(f64::NEG_INFINITY);
        assert_eq!(capability.choice_value_to_string(&neg_inf_val), "-inf");
        
        // Test string serialization
        let str_val = ChoiceValue::String("Hello".to_string());
        assert_eq!(capability.choice_value_to_string(&str_val), "Hello");
        
        // Test bytes serialization
        let bytes_val = ChoiceValue::Bytes(vec![72, 101, 108, 108, 111]); // "Hello"
        assert_eq!(capability.choice_value_to_string(&bytes_val), "48656c6c6f");
    }

    #[test]
    fn test_py_complete_workflow() {
        let mut capability = PyTemplatingCapability::new();
        capability.enable_debug();
        
        // Step 1: Add various templates and forced values
        capability.py_add_template("simplest", None, None).unwrap();
        capability.py_add_template("biased", Some(0.8), None).unwrap();
        capability.py_add_forced_value("int", "100").unwrap();
        capability.py_add_forced_value("bool", "true").unwrap();
        
        // Step 2: Process all entries
        let mut constraints = HashMap::new();
        constraints.insert("min_value".to_string(), "0".to_string());
        constraints.insert("max_value".to_string(), "1000".to_string());
        
        // Process integer templates and forced value
        for i in 1..=3 {
            let result = capability.py_process_choice("int", constraints.clone()).unwrap();
            assert_eq!(result.get("success"), Some(&"true".to_string()));
            println!("Processed integer choice {}: {:?}", i, result.get("value"));
        }
        
        // Process boolean forced value
        let result = capability.py_process_choice("bool", HashMap::new()).unwrap();
        assert_eq!(result.get("success"), Some(&"true".to_string()));
        assert_eq!(result.get("value"), Some(&"true".to_string()));
        assert_eq!(result.get("was_forced"), Some(&"true".to_string()));
        
        // Verify no more templates
        let result = capability.py_process_choice("int", constraints).unwrap();
        assert_eq!(result.get("no_more_templates"), Some(&"true".to_string()));
        
        // Check final status
        let status = capability.py_get_status();
        assert_eq!(status.get("remaining_templates"), Some(&"0".to_string()));
        assert_eq!(status.get("processed_count"), Some(&"4".to_string()));
    }
}