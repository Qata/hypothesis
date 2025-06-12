//! Comprehensive tests for forced value insertion capability
//!
//! This module provides exhaustive testing of the forced value insertion capability
//! within the Choice templating and forcing system. Tests focus on validating that
//! forced values are correctly inserted, properly validated, and appropriately marked
//! during guided test case construction.

use crate::choice::{
    templating::*,
    ChoiceType, ChoiceValue, Constraints, ChoiceNode,
    IntegerConstraints, BooleanConstraints, FloatConstraints, StringConstraints, BytesConstraints,
    IntervalSet
};

/// Comprehensive test suite for forced value insertion capability
pub struct ForcedValueInsertionTests {
    engine: TemplateEngine,
    test_results: Vec<String>,
}

impl ForcedValueInsertionTests {
    pub fn new() -> Self {
        Self {
            engine: TemplateEngine::new().with_debug(),
            test_results: Vec::new(),
        }
    }

    /// Run complete validation of forced value insertion capability
    pub fn run_comprehensive_validation(&mut self) -> Result<Vec<String>, String> {
        self.test_results.clear();

        // Test 1: Basic forced value insertion for all types
        self.test_basic_forced_value_insertion()?;

        // Test 2: Edge case forced values
        self.test_edge_case_forced_values()?;

        // Test 3: Forced value constraint validation
        self.test_forced_value_constraint_validation()?;

        // Test 4: Forced value priority and ordering
        self.test_forced_value_priority_ordering()?;

        // Test 5: Forced value state preservation
        self.test_forced_value_state_preservation()?;

        // Test 6: Mixed forced and template values
        self.test_mixed_forced_template_values()?;

        // Test 7: Forced value error scenarios
        self.test_forced_value_error_scenarios()?;

        // Test 8: Forced value metadata and debugging
        self.test_forced_value_metadata_debugging()?;

        Ok(self.test_results.clone())
    }

    /// Test 1: Basic forced value insertion for all supported types
    fn test_basic_forced_value_insertion(&mut self) -> Result<(), String> {
        self.engine.reset();

        // Test comprehensive integer forced values
        let integer_values = vec![
            (ChoiceValue::Integer(0), "zero"),
            (ChoiceValue::Integer(42), "positive"),
            (ChoiceValue::Integer(-42), "negative"),
            (ChoiceValue::Integer(i128::MAX), "maximum"),
            (ChoiceValue::Integer(i128::MIN), "minimum"),
            (ChoiceValue::Integer(1), "one"),
            (ChoiceValue::Integer(-1), "negative one"),
        ];

        for (value, description) in integer_values {
            self.engine.reset();
            self.engine.add_entry(TemplateEntry::DirectValue(value.clone()));
            
            let constraints = Constraints::Integer(IntegerConstraints {
                min_value: Some(i128::MIN),
                max_value: Some(i128::MAX),
                weights: None,
                shrink_towards: Some(0),
            });

            match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
                Ok(Some(node)) => {
                    self.validate_forced_value_insertion(&node, &value, true)?;
                    self.test_results.push(format!("Basic integer forced value: {} inserted correctly", description));
                },
                Ok(None) => return Err(format!("No choice generated for integer forced value: {}", description)),
                Err(e) => return Err(format!("Integer forced value failed {}: {}", description, e)),
            }
        }

        // Test boolean forced values
        let boolean_values = vec![
            (ChoiceValue::Boolean(true), "true"),
            (ChoiceValue::Boolean(false), "false"),
        ];

        for (value, description) in boolean_values {
            self.engine.reset();
            self.engine.add_entry(TemplateEntry::DirectValue(value.clone()));
            
            let constraints = Constraints::Boolean(BooleanConstraints::default());

            match self.engine.process_next_template(ChoiceType::Boolean, &constraints) {
                Ok(Some(node)) => {
                    self.validate_forced_value_insertion(&node, &value, true)?;
                    self.test_results.push(format!("Basic boolean forced value: {} inserted correctly", description));
                },
                Ok(None) => return Err(format!("No choice generated for boolean forced value: {}", description)),
                Err(e) => return Err(format!("Boolean forced value failed {}: {}", description, e)),
            }
        }

        // Test float forced values
        let float_values = vec![
            (ChoiceValue::Float(0.0), "zero"),
            (ChoiceValue::Float(3.14159), "pi"),
            (ChoiceValue::Float(-2.71828), "negative e"),
            (ChoiceValue::Float(f64::MIN_POSITIVE), "min positive"),
            (ChoiceValue::Float(f64::MAX), "maximum"),
            (ChoiceValue::Float(f64::INFINITY), "infinity"),
            (ChoiceValue::Float(f64::NEG_INFINITY), "negative infinity"),
        ];

        for (value, description) in float_values {
            self.engine.reset();
            self.engine.add_entry(TemplateEntry::DirectValue(value.clone()));
            
            let constraints = Constraints::Float(FloatConstraints {
                min_value: f64::NEG_INFINITY,
                max_value: f64::INFINITY,
                allow_nan: true,
                smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
            });

            match self.engine.process_next_template(ChoiceType::Float, &constraints) {
                Ok(Some(node)) => {
                    self.validate_forced_value_insertion(&node, &value, true)?;
                    self.test_results.push(format!("Basic float forced value: {} inserted correctly", description));
                },
                Ok(None) => return Err(format!("No choice generated for float forced value: {}", description)),
                Err(e) => return Err(format!("Float forced value failed {}: {}", description, e)),
            }
        }

        // Test string forced values
        let string_values = vec![
            (ChoiceValue::String("".to_string()), "empty"),
            (ChoiceValue::String("a".to_string()), "single char"),
            (ChoiceValue::String("Hello".to_string()), "simple text"),
            (ChoiceValue::String("Hello, World!".to_string()), "punctuated text"),
            (ChoiceValue::String("🦀 Rust 🔥".to_string()), "unicode"),
            (ChoiceValue::String("line1\nline2\tindented".to_string()), "multiline"),
            (ChoiceValue::String(" ".repeat(100), "long spaces"),
        ];

        for (value, description) in string_values {
            self.engine.reset();
            self.engine.add_entry(TemplateEntry::DirectValue(value.clone()));
            
            let constraints = Constraints::String(StringConstraints {
                min_size: 0,
                max_size: 1000,
                intervals: IntervalSet::default(),
            });

            match self.engine.process_next_template(ChoiceType::String, &constraints) {
                Ok(Some(node)) => {
                    self.validate_forced_value_insertion(&node, &value, true)?;
                    self.test_results.push(format!("Basic string forced value: {} inserted correctly", description));
                },
                Ok(None) => return Err(format!("No choice generated for string forced value: {}", description)),
                Err(e) => return Err(format!("String forced value failed {}: {}", description, e)),
            }
        }

        // Test bytes forced values
        let bytes_values = vec![
            (ChoiceValue::Bytes(vec![]), "empty"),
            (ChoiceValue::Bytes(vec![0]), "single zero"),
            (ChoiceValue::Bytes(vec![255]), "single max"),
            (ChoiceValue::Bytes(vec![1, 2, 3, 4, 5]), "sequence"),
            (ChoiceValue::Bytes(vec![0; 100]), "zero filled"),
            (ChoiceValue::Bytes(vec![255; 50]), "max filled"),
            (ChoiceValue::Bytes((0..=255).collect()), "full range"),
        ];

        for (value, description) in bytes_values {
            self.engine.reset();
            self.engine.add_entry(TemplateEntry::DirectValue(value.clone()));
            
            let constraints = Constraints::Bytes(BytesConstraints {
                min_size: 0,
                max_size: 1000,
            });

            match self.engine.process_next_template(ChoiceType::Bytes, &constraints) {
                Ok(Some(node)) => {
                    self.validate_forced_value_insertion(&node, &value, true)?;
                    self.test_results.push(format!("Basic bytes forced value: {} inserted correctly", description));
                },
                Ok(None) => return Err(format!("No choice generated for bytes forced value: {}", description)),
                Err(e) => return Err(format!("Bytes forced value failed {}: {}", description, e)),
            }
        }

        Ok(())
    }

    /// Test 2: Edge case forced values that test boundary conditions
    fn test_edge_case_forced_values(&mut self) -> Result<(), String> {
        // Test NaN float forced value
        self.engine.reset();
        self.engine.add_entry(TemplateEntry::DirectValue(ChoiceValue::Float(f64::NAN)));
        
        let nan_constraints = Constraints::Float(FloatConstraints {
            min_value: f64::NEG_INFINITY,
            max_value: f64::INFINITY,
            allow_nan: true,
            smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
        });

        match self.engine.process_next_template(ChoiceType::Float, &nan_constraints) {
            Ok(Some(node)) => {
                if let ChoiceValue::Float(val) = node.value {
                    if !val.is_nan() {
                        return Err("NaN forced value was not preserved".to_string());
                    }
                    if !node.was_forced {
                        return Err("NaN forced value not marked as forced".to_string());
                    }
                    self.test_results.push("Edge case: NaN float forced value handled correctly".to_string());
                } else {
                    return Err("NaN forced value returned wrong type".to_string());
                }
            },
            Ok(None) => return Err("No choice generated for NaN forced value".to_string()),
            Err(e) => return Err(format!("NaN forced value failed: {}", e)),
        }

        // Test empty string with min size constraint (should cause misalignment)
        self.engine.reset();
        self.engine.add_entry(TemplateEntry::DirectValue(ChoiceValue::String("".to_string())));
        
        let min_size_constraints = Constraints::String(StringConstraints {
            min_size: 5,
            max_size: 100,
            intervals: IntervalSet::default(),
        });

        match self.engine.process_next_template(ChoiceType::String, &min_size_constraints) {
            Ok(Some(node)) => {
                if self.engine.has_misalignment() {
                    // Should have fallen back to valid string
                    if let ChoiceValue::String(s) = &node.value {
                        if s.len() >= 5 {
                            self.test_results.push("Edge case: empty string misalignment handled correctly".to_string());
                        } else {
                            return Err("Misalignment fallback string too short".to_string());
                        }
                    } else {
                        return Err("Misalignment fallback returned wrong type".to_string());
                    }
                } else {
                    return Err("Empty string misalignment not detected".to_string());
                }
            },
            Ok(None) => return Err("No choice generated for misaligned empty string".to_string()),
            Err(e) => return Err(format!("Empty string misalignment test failed: {}", e)),
        }

        // Test very large integer forced value
        self.engine.reset();
        let large_int = i128::MAX - 1000;
        self.engine.add_entry(TemplateEntry::DirectValue(ChoiceValue::Integer(large_int)));
        
        let large_int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(i128::MIN),
            max_value: Some(i128::MAX),
            weights: None,
            shrink_towards: Some(0),
        });

        match self.engine.process_next_template(ChoiceType::Integer, &large_int_constraints) {
            Ok(Some(node)) => {
                if let ChoiceValue::Integer(val) = node.value {
                    if val == large_int && node.was_forced {
                        self.test_results.push("Edge case: large integer forced value handled correctly".to_string());
                    } else {
                        return Err("Large integer forced value not preserved correctly".to_string());
                    }
                } else {
                    return Err("Large integer forced value returned wrong type".to_string());
                }
            },
            Ok(None) => return Err("No choice generated for large integer forced value".to_string()),
            Err(e) => return Err(format!("Large integer forced value failed: {}", e)),
        }

        Ok(())
    }

    /// Test 3: Forced value constraint validation and misalignment handling
    fn test_forced_value_constraint_validation(&mut self) -> Result<(), String> {
        // Test integer forced value outside constraints
        self.engine.reset();
        self.engine.add_entry(TemplateEntry::DirectValue(ChoiceValue::Integer(1000)));
        
        let tight_int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(50),
        });

        match self.engine.process_next_template(ChoiceType::Integer, &tight_int_constraints) {
            Ok(Some(node)) => {
                if self.engine.has_misalignment() {
                    // Should fall back to valid value within constraints
                    if let ChoiceValue::Integer(val) = node.value {
                        if val >= 0 && val <= 100 {
                            self.test_results.push("Constraint validation: integer outside range handled correctly".to_string());
                        } else {
                            return Err("Fallback integer value outside constraints".to_string());
                        }
                    } else {
                        return Err("Integer constraint validation returned wrong type".to_string());
                    }
                } else {
                    return Err("Integer constraint violation not detected".to_string());
                }
            },
            Ok(None) => return Err("No choice generated for constrained integer".to_string()),
            Err(e) => return Err(format!("Integer constraint validation failed: {}", e)),
        }

        // Test string forced value too long for constraints
        self.engine.reset();
        self.engine.add_entry(TemplateEntry::DirectValue(ChoiceValue::String("This string is way too long".to_string())));
        
        let short_string_constraints = Constraints::String(StringConstraints {
            min_size: 0,
            max_size: 5,
            intervals: IntervalSet::default(),
        });

        match self.engine.process_next_template(ChoiceType::String, &short_string_constraints) {
            Ok(Some(node)) => {
                if self.engine.has_misalignment() {
                    if let ChoiceValue::String(s) = &node.value {
                        if s.len() <= 5 {
                            self.test_results.push("Constraint validation: string too long handled correctly".to_string());
                        } else {
                            return Err("Fallback string still too long".to_string());
                        }
                    } else {
                        return Err("String constraint validation returned wrong type".to_string());
                    }
                } else {
                    return Err("String length constraint violation not detected".to_string());
                }
            },
            Ok(None) => return Err("No choice generated for constrained string".to_string()),
            Err(e) => return Err(format!("String constraint validation failed: {}", e)),
        }

        // Test float forced value outside range
        self.engine.reset();
        self.engine.add_entry(TemplateEntry::DirectValue(ChoiceValue::Float(100.0)));
        
        let narrow_float_constraints = Constraints::Float(FloatConstraints {
            min_value: -1.0,
            max_value: 1.0,
            allow_nan: false,
            smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
        });

        match self.engine.process_next_template(ChoiceType::Float, &narrow_float_constraints) {
            Ok(Some(node)) => {
                if self.engine.has_misalignment() {
                    if let ChoiceValue::Float(val) = node.value {
                        if val >= -1.0 && val <= 1.0 {
                            self.test_results.push("Constraint validation: float outside range handled correctly".to_string());
                        } else {
                            return Err("Fallback float value outside constraints".to_string());
                        }
                    } else {
                        return Err("Float constraint validation returned wrong type".to_string());
                    }
                } else {
                    return Err("Float constraint violation not detected".to_string());
                }
            },
            Ok(None) => return Err("No choice generated for constrained float".to_string()),
            Err(e) => return Err(format!("Float constraint validation failed: {}", e)),
        }

        Ok(())
    }

    /// Test 4: Forced value priority and ordering in template queues
    fn test_forced_value_priority_ordering(&mut self) -> Result<(), String> {
        self.engine.reset();

        // Add sequence of forced values in specific order
        let forced_sequence = vec![
            ChoiceValue::Integer(1),
            ChoiceValue::Integer(2),
            ChoiceValue::Integer(3),
            ChoiceValue::Integer(4),
            ChoiceValue::Integer(5),
        ];

        for value in &forced_sequence {
            self.engine.add_entry(TemplateEntry::DirectValue(value.clone()));
        }

        let constraints = Constraints::Integer(IntegerConstraints::default());
        let mut extracted_values = Vec::new();

        // Process all forced values and verify ordering
        for expected_index in 0..forced_sequence.len() {
            match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
                Ok(Some(node)) => {
                    if let ChoiceValue::Integer(val) = node.value {
                        extracted_values.push(val);
                        if !node.was_forced {
                            return Err(format!("Forced value {} not marked as forced", val));
                        }
                    } else {
                        return Err("Forced value sequence returned wrong type".to_string());
                    }
                },
                Ok(None) => return Err(format!("Forced value sequence ended prematurely at index {}", expected_index)),
                Err(e) => return Err(format!("Forced value sequence processing failed: {}", e)),
            }
        }

        // Verify no more values remain
        match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
            Ok(None) => {},
            Ok(Some(_)) => return Err("Extra forced values found after sequence".to_string()),
            Err(e) => return Err(format!("Error checking end of forced sequence: {}", e)),
        }

        // Verify ordering preserved
        let expected_ints: Vec<i128> = forced_sequence.iter().map(|v| {
            if let ChoiceValue::Integer(i) = v { *i } else { panic!("Expected integer") }
        }).collect();

        if extracted_values == expected_ints {
            self.test_results.push("Priority ordering: forced value sequence preserved correctly".to_string());
        } else {
            return Err(format!("Forced value ordering not preserved: expected {:?}, got {:?}", 
                expected_ints, extracted_values));
        }

        Ok(())
    }

    /// Test 5: Forced value state preservation during engine operations
    fn test_forced_value_state_preservation(&mut self) -> Result<(), String> {
        self.engine.reset();

        // Set up initial forced values
        let initial_values = vec![
            ChoiceValue::Integer(10),
            ChoiceValue::Integer(20),
            ChoiceValue::Integer(30),
        ];

        for value in &initial_values {
            self.engine.add_entry(TemplateEntry::DirectValue(value.clone()));
        }

        // Save initial state
        let initial_state = self.engine.clone_state();
        assert_eq!(initial_state.remaining_entries.len(), 3);

        // Process one forced value
        let constraints = Constraints::Integer(IntegerConstraints::default());
        match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
            Ok(Some(node)) => {
                if let ChoiceValue::Integer(val) = node.value {
                    if val != 10 {
                        return Err("First forced value not processed correctly".to_string());
                    }
                } else {
                    return Err("First forced value wrong type".to_string());
                }
            },
            _ => return Err("Failed to process first forced value".to_string()),
        }

        assert_eq!(self.engine.remaining_count(), 2);
        assert_eq!(self.engine.processed_count(), 1);

        // Save intermediate state
        let intermediate_state = self.engine.clone_state();

        // Process remaining values
        let _ = self.engine.process_next_template(ChoiceType::Integer, &constraints);
        let _ = self.engine.process_next_template(ChoiceType::Integer, &constraints);

        assert_eq!(self.engine.remaining_count(), 0);
        assert_eq!(self.engine.processed_count(), 3);

        // Restore to intermediate state
        self.engine.restore_state(intermediate_state);
        assert_eq!(self.engine.remaining_count(), 2);
        assert_eq!(self.engine.processed_count(), 1);

        // Verify next value is correct
        match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
            Ok(Some(node)) => {
                if let ChoiceValue::Integer(val) = node.value {
                    if val == 20 {
                        self.test_results.push("State preservation: intermediate state restored correctly".to_string());
                    } else {
                        return Err("Intermediate state restoration failed - wrong value".to_string());
                    }
                } else {
                    return Err("Intermediate state restoration wrong type".to_string());
                }
            },
            _ => return Err("Failed to process after intermediate restoration".to_string()),
        }

        // Restore to initial state
        self.engine.restore_state(initial_state);
        assert_eq!(self.engine.remaining_count(), 3);
        assert_eq!(self.engine.processed_count(), 0);

        // Verify first value again
        match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
            Ok(Some(node)) => {
                if let ChoiceValue::Integer(val) = node.value {
                    if val == 10 {
                        self.test_results.push("State preservation: initial state restored correctly".to_string());
                    } else {
                        return Err("Initial state restoration failed - wrong value".to_string());
                    }
                } else {
                    return Err("Initial state restoration wrong type".to_string());
                }
            },
            _ => return Err("Failed to process after initial restoration".to_string()),
        }

        Ok(())
    }

    /// Test 6: Mixed forced values and template values processing
    fn test_mixed_forced_template_values(&mut self) -> Result<(), String> {
        self.engine.reset();

        // Create mixed sequence: forced value, template, forced value, template
        self.engine.add_entry(TemplateEntry::DirectValue(ChoiceValue::Integer(100)));
        self.engine.add_entry(TemplateEntry::Template(ChoiceTemplate::simplest()));
        self.engine.add_entry(TemplateEntry::DirectValue(ChoiceValue::Integer(200)));
        self.engine.add_entry(TemplateEntry::Template(ChoiceTemplate::at_index(5)));

        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(1000),
            weights: None,
            shrink_towards: Some(0),
        });

        // Process first entry (forced value)
        match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
            Ok(Some(node)) => {
                if let ChoiceValue::Integer(val) = node.value {
                    if val == 100 && node.was_forced {
                        self.test_results.push("Mixed sequence: first forced value processed correctly".to_string());
                    } else {
                        return Err("First forced value in mixed sequence not correct".to_string());
                    }
                } else {
                    return Err("First forced value wrong type".to_string());
                }
            },
            _ => return Err("Failed to process first mixed entry".to_string()),
        }

        // Process second entry (template)
        match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
            Ok(Some(node)) => {
                if let ChoiceValue::Integer(val) = node.value {
                    // Template should generate simplest choice (shrink_towards = 0)
                    if val == 0 && !node.was_forced {
                        self.test_results.push("Mixed sequence: first template processed correctly".to_string());
                    } else {
                        return Err("First template in mixed sequence not correct".to_string());
                    }
                } else {
                    return Err("First template wrong type".to_string());
                }
            },
            _ => return Err("Failed to process first template in mixed sequence".to_string()),
        }

        // Process third entry (forced value)
        match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
            Ok(Some(node)) => {
                if let ChoiceValue::Integer(val) = node.value {
                    if val == 200 && node.was_forced {
                        self.test_results.push("Mixed sequence: second forced value processed correctly".to_string());
                    } else {
                        return Err("Second forced value in mixed sequence not correct".to_string());
                    }
                } else {
                    return Err("Second forced value wrong type".to_string());
                }
            },
            _ => return Err("Failed to process second forced value in mixed sequence".to_string()),
        }

        // Process fourth entry (template)
        match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
            Ok(Some(node)) => {
                if let ChoiceValue::Integer(_val) = node.value {
                    // At-index template may generate different values, but should not be forced
                    if !node.was_forced {
                        self.test_results.push("Mixed sequence: second template processed correctly".to_string());
                    } else {
                        return Err("Second template incorrectly marked as forced".to_string());
                    }
                } else {
                    return Err("Second template wrong type".to_string());
                }
            },
            _ => return Err("Failed to process second template in mixed sequence".to_string()),
        }

        // Verify no more entries
        match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
            Ok(None) => {
                self.test_results.push("Mixed sequence: all entries processed, sequence complete".to_string());
            },
            Ok(Some(_)) => return Err("Extra entries found after mixed sequence".to_string()),
            Err(e) => return Err(format!("Error checking end of mixed sequence: {}", e)),
        }

        Ok(())
    }

    /// Test 7: Forced value error scenarios and recovery
    fn test_forced_value_error_scenarios(&mut self) -> Result<(), String> {
        // Test type mismatch scenario
        self.engine.reset();
        self.engine.add_entry(TemplateEntry::DirectValue(ChoiceValue::Integer(42)));
        
        let bool_constraints = Constraints::Boolean(BooleanConstraints::default());

        match self.engine.process_next_template(ChoiceType::Boolean, &bool_constraints) {
            Ok(Some(node)) => {
                if self.engine.has_misalignment() {
                    // Should fall back to valid boolean
                    if let ChoiceValue::Boolean(_) = node.value {
                        self.test_results.push("Error scenarios: type mismatch handled with fallback".to_string());
                    } else {
                        return Err("Type mismatch fallback returned wrong type".to_string());
                    }
                } else {
                    return Err("Type mismatch not detected".to_string());
                }
            },
            Ok(None) => return Err("No choice generated for type mismatch".to_string()),
            Err(e) => return Err(format!("Type mismatch handling failed: {}", e)),
        }

        // Test NaN with disallowed NaN constraint
        self.engine.reset();
        self.engine.add_entry(TemplateEntry::DirectValue(ChoiceValue::Float(f64::NAN)));
        
        let no_nan_constraints = Constraints::Float(FloatConstraints {
            min_value: -10.0,
            max_value: 10.0,
            allow_nan: false,
            smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
        });

        match self.engine.process_next_template(ChoiceType::Float, &no_nan_constraints) {
            Ok(Some(node)) => {
                if self.engine.has_misalignment() {
                    if let ChoiceValue::Float(val) = node.value {
                        if !val.is_nan() && val >= -10.0 && val <= 10.0 {
                            self.test_results.push("Error scenarios: NaN constraint violation handled correctly".to_string());
                        } else {
                            return Err("NaN fallback value invalid".to_string());
                        }
                    } else {
                        return Err("NaN fallback returned wrong type".to_string());
                    }
                } else {
                    return Err("NaN constraint violation not detected".to_string());
                }
            },
            Ok(None) => return Err("No choice generated for NaN constraint violation".to_string()),
            Err(e) => return Err(format!("NaN constraint handling failed: {}", e)),
        }

        Ok(())
    }

    /// Test 8: Forced value metadata and debugging capabilities
    fn test_forced_value_metadata_debugging(&mut self) -> Result<(), String> {
        self.engine.reset();

        // Test partial node with metadata
        let partial_entry = TemplateEntry::partial_node(
            ChoiceValue::Integer(42),
            Some(Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(100),
                weights: None,
                shrink_towards: Some(0),
            })),
            Some("Test partial node with metadata".to_string()),
        );

        self.engine.add_entry(partial_entry.clone());

        // Verify debug description
        let debug_desc = partial_entry.debug_description();
        if !debug_desc.contains("Partial") || !debug_desc.contains("with constraints") {
            return Err("Partial node debug description incomplete".to_string());
        }

        let constraints = Constraints::Integer(IntegerConstraints::default());
        match self.engine.process_next_template(ChoiceType::Integer, &constraints) {
            Ok(Some(node)) => {
                if let ChoiceValue::Integer(val) = node.value {
                    if val == 42 && node.was_forced {
                        self.test_results.push("Metadata debugging: partial node with metadata processed correctly".to_string());
                    } else {
                        return Err("Partial node with metadata not processed correctly".to_string());
                    }
                } else {
                    return Err("Partial node returned wrong type".to_string());
                }
            },
            _ => return Err("Failed to process partial node with metadata".to_string()),
        }

        // Test engine debug info
        let debug_info = self.engine.debug_info();
        if !debug_info.contains("TemplateEngine") {
            return Err("Engine debug info incomplete".to_string());
        }

        self.test_results.push("Metadata debugging: engine debug info available".to_string());

        // Test forced entry type checking
        let direct_entry = TemplateEntry::direct(ChoiceValue::String("test".to_string()));
        if !direct_entry.is_forcing() {
            return Err("Direct entry not marked as forcing".to_string());
        }

        let direct_debug = direct_entry.debug_description();
        if !direct_debug.contains("Direct") {
            return Err("Direct entry debug description incomplete".to_string());
        }

        self.test_results.push("Metadata debugging: entry debug descriptions working correctly".to_string());

        Ok(())
    }

    // Helper methods

    fn validate_forced_value_insertion(&self, node: &ChoiceNode, expected_value: &ChoiceValue, should_be_forced: bool) -> Result<(), String> {
        // Validate value matches exactly
        if node.value != *expected_value {
            return Err(format!("Forced value mismatch: expected {:?}, got {:?}", expected_value, node.value));
        }

        // Validate forcing flag
        if node.was_forced != should_be_forced {
            return Err(format!("Forced flag mismatch: expected {}, got {}", should_be_forced, node.was_forced));
        }

        // Validate type consistency
        let value_type = match &node.value {
            ChoiceValue::Integer(_) => ChoiceType::Integer,
            ChoiceValue::Boolean(_) => ChoiceType::Boolean,
            ChoiceValue::Float(_) => ChoiceType::Float,
            ChoiceValue::String(_) => ChoiceType::String,
            ChoiceValue::Bytes(_) => ChoiceType::Bytes,
        };

        if node.choice_type != value_type {
            return Err(format!("Type mismatch: node type {:?}, value type {:?}", node.choice_type, value_type));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_forced_value_validation() {
        let mut test_suite = ForcedValueInsertionTests::new();
        
        let results = test_suite.run_comprehensive_validation();
        assert!(results.is_ok(), "Forced value validation failed: {:?}", results.err());
        
        let test_results = results.unwrap();
        assert!(!test_results.is_empty(), "No test results generated");
        
        // Verify all major test areas completed
        let expected_test_areas = vec![
            "Basic integer forced value",
            "Basic boolean forced value", 
            "Basic float forced value",
            "Basic string forced value",
            "Basic bytes forced value",
            "Edge case:",
            "Constraint validation:",
            "Priority ordering:",
            "State preservation:",
            "Mixed sequence:",
            "Error scenarios:",
            "Metadata debugging:",
        ];
        
        for area in expected_test_areas {
            let found = test_results.iter().any(|result| result.contains(area));
            assert!(found, "Missing test results for area: {}", area);
        }
        
        println!("Forced value insertion capability validation completed:");
        for result in test_results {
            println!("  ✓ {}", result);
        }
    }

    #[test]
    fn test_basic_forced_values_only() {
        let mut test_suite = ForcedValueInsertionTests::new();
        
        let result = test_suite.test_basic_forced_value_insertion();
        assert!(result.is_ok(), "Basic forced value test failed: {:?}", result.err());
        
        // Should have results for all basic types
        let basic_results: Vec<_> = test_suite.test_results.iter()
            .filter(|r| r.contains("Basic") && r.contains("forced value"))
            .collect();
        assert!(basic_results.len() >= 20); // At least 20 different basic forced values
    }

    #[test]
    fn test_edge_cases_only() {
        let mut test_suite = ForcedValueInsertionTests::new();
        
        let result = test_suite.test_edge_case_forced_values();
        assert!(result.is_ok(), "Edge case test failed: {:?}", result.err());
        
        let edge_results: Vec<_> = test_suite.test_results.iter()
            .filter(|r| r.contains("Edge case"))
            .collect();
        assert!(edge_results.len() >= 3); // At least 3 edge cases tested
    }

    #[test]
    fn test_constraint_validation_only() {
        let mut test_suite = ForcedValueInsertionTests::new();
        
        let result = test_suite.test_forced_value_constraint_validation();
        assert!(result.is_ok(), "Constraint validation test failed: {:?}", result.err());
        
        let constraint_results: Vec<_> = test_suite.test_results.iter()
            .filter(|r| r.contains("Constraint validation"))
            .collect();
        assert!(constraint_results.len() >= 3); // At least 3 constraint validation tests
    }

    #[test]
    fn test_mixed_forced_template_only() {
        let mut test_suite = ForcedValueInsertionTests::new();
        
        let result = test_suite.test_mixed_forced_template_values();
        assert!(result.is_ok(), "Mixed forced/template test failed: {:?}", result.err());
        
        let mixed_results: Vec<_> = test_suite.test_results.iter()
            .filter(|r| r.contains("Mixed sequence"))
            .collect();
        assert!(mixed_results.len() >= 4); // At least 4 mixed sequence validations
    }

    #[test]
    fn test_error_scenarios_only() {
        let mut test_suite = ForcedValueInsertionTests::new();
        
        let result = test_suite.test_forced_value_error_scenarios();
        assert!(result.is_ok(), "Error scenarios test failed: {:?}", result.err());
        
        let error_results: Vec<_> = test_suite.test_results.iter()
            .filter(|r| r.contains("Error scenarios"))
            .collect();
        assert!(error_results.len() >= 2); // At least 2 error scenarios tested
    }
}