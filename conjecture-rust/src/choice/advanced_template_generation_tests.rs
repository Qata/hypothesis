//! Comprehensive tests for advanced template generation capabilities
//! 
//! Tests the newly implemented index-based, biased, and custom choice generation
//! functionality to ensure proper Python Hypothesis parity.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::choice::{ChoiceType, ChoiceValue, Constraints};
    use crate::choice::{IntegerConstraints, FloatConstraints, BooleanConstraints, StringConstraints, BytesConstraints, IntervalSet};

    fn create_test_engine() -> TemplateEngine {
        TemplateEngine::new()
    }

    #[test]
    fn test_generate_at_index_integers() {
        let engine = create_test_engine();
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(-10),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        });

        // Test index 0 should give shrink_towards value (0)
        let choice_0 = engine.generate_at_index(0, ChoiceType::Integer, &constraints).unwrap();
        if let ChoiceValue::Integer(val) = choice_0.value {
            assert_eq!(val, 0, "Index 0 should produce shrink_towards value");
        } else {
            panic!("Expected integer value");
        }

        // Test index 1 should give first non-shrink_towards value (1)
        let choice_1 = engine.generate_at_index(1, ChoiceType::Integer, &constraints).unwrap();
        if let ChoiceValue::Integer(val) = choice_1.value {
            assert_eq!(val, 1, "Index 1 should produce value 1");
        } else {
            panic!("Expected integer value");
        }

        // Test index 2 should give negative direction (-1)
        let choice_2 = engine.generate_at_index(2, ChoiceType::Integer, &constraints).unwrap();
        if let ChoiceValue::Integer(val) = choice_2.value {
            assert_eq!(val, -1, "Index 2 should produce value -1");
        } else {
            panic!("Expected integer value");
        }

        println!("✓ Index-based integer generation test passed");
    }

    #[test]
    fn test_generate_at_index_floats() {
        let engine = create_test_engine();
        let constraints = Constraints::Float(FloatConstraints {
            min_value: -100.0,
            max_value: 100.0,
            allow_nan: false,
            smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
        });

        // Test that we can generate floats at different indices
        let choice_0 = engine.generate_at_index(0, ChoiceType::Float, &constraints).unwrap();
        let choice_1 = engine.generate_at_index(1, ChoiceType::Float, &constraints).unwrap();
        let choice_100 = engine.generate_at_index(100, ChoiceType::Float, &constraints).unwrap();

        // Verify they're all floats and within constraints
        match (&choice_0.value, &choice_1.value, &choice_100.value) {
            (ChoiceValue::Float(v0), ChoiceValue::Float(v1), ChoiceValue::Float(v100)) => {
                assert!(v0 >= &-100.0 && v0 <= &100.0, "Value 0 within constraints: {}", v0);
                assert!(v1 >= &-100.0 && v1 <= &100.0, "Value 1 within constraints: {}", v1);
                assert!(v100 >= &-100.0 && v100 <= &100.0, "Value 100 within constraints: {}", v100);
                
                // Values at different indices should generally be different
                assert_ne!(v0, v1, "Different indices should produce different values");
            }
            _ => panic!("Expected float values"),
        }

        println!("✓ Index-based float generation test passed");
    }

    #[test]
    fn test_generate_with_bias_booleans() {
        let engine = create_test_engine();
        let constraints = Constraints::Boolean(BooleanConstraints::default());

        // Test low bias (should favor False)
        let low_bias_choice = engine.generate_with_bias(0.2, ChoiceType::Boolean, &constraints).unwrap();
        if let ChoiceValue::Boolean(val) = low_bias_choice.value {
            assert!(!val, "Low bias should produce False");
        } else {
            panic!("Expected boolean value");
        }

        // Test high bias (should favor True)  
        let high_bias_choice = engine.generate_with_bias(0.8, ChoiceType::Boolean, &constraints).unwrap();
        if let ChoiceValue::Boolean(val) = high_bias_choice.value {
            assert!(val, "High bias should produce True");
        } else {
            panic!("Expected boolean value");
        }

        println!("✓ Biased boolean generation test passed");
    }

    #[test]
    fn test_generate_with_bias_integers() {
        let engine = create_test_engine();
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(-100),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        });

        // Test low bias (should be closer to shrink_towards)
        let low_bias_choice = engine.generate_with_bias(0.1, ChoiceType::Integer, &constraints).unwrap();
        if let ChoiceValue::Integer(val) = low_bias_choice.value {
            assert!(val.abs() <= 10, "Low bias should produce value close to shrink_towards: {}", val);
        } else {
            panic!("Expected integer value");
        }

        // Test high bias (should be farther from shrink_towards)
        let high_bias_choice = engine.generate_with_bias(0.9, ChoiceType::Integer, &constraints).unwrap();
        if let ChoiceValue::Integer(val) = high_bias_choice.value {
            assert!(val.abs() >= 80, "High bias should produce value far from shrink_towards: {}", val);
        } else {
            panic!("Expected integer value");
        }

        println!("✓ Biased integer generation test passed");
    }

    #[test]
    fn test_generate_with_bias_floats() {
        let engine = create_test_engine();
        let constraints = Constraints::Float(FloatConstraints {
            min_value: -1000.0,
            max_value: 1000.0,
            allow_nan: false,
            smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
        });

        // Test different bias values
        let low_bias_choice = engine.generate_with_bias(0.2, ChoiceType::Float, &constraints).unwrap();
        let high_bias_choice = engine.generate_with_bias(0.8, ChoiceType::Float, &constraints).unwrap();

        match (&low_bias_choice.value, &high_bias_choice.value) {
            (ChoiceValue::Float(low_val), ChoiceValue::Float(high_val)) => {
                assert!(low_val >= &-1000.0 && low_val <= &1000.0, "Low bias value within constraints");
                assert!(high_val >= &-1000.0 && high_val <= &1000.0, "High bias value within constraints");
                
                // Low bias should produce negative values, high bias positive
                assert!(low_val < &0.0, "Low bias should produce negative float: {}", low_val);
                assert!(high_val > &0.0, "High bias should produce positive float: {}", high_val);
            }
            _ => panic!("Expected float values"),
        }

        println!("✓ Biased float generation test passed");
    }

    #[test]
    fn test_generate_with_bias_strings() {
        let engine = create_test_engine();
        let constraints = Constraints::String(StringConstraints {
            min_size: 0,
            max_size: 20,
            intervals: IntervalSet::default(),
        });

        // Test bias affects string length
        let low_bias_choice = engine.generate_with_bias(0.1, ChoiceType::String, &constraints).unwrap();
        let high_bias_choice = engine.generate_with_bias(0.9, ChoiceType::String, &constraints).unwrap();

        match (&low_bias_choice.value, &high_bias_choice.value) {
            (ChoiceValue::String(low_str), ChoiceValue::String(high_str)) => {
                assert!(low_str.len() <= 2, "Low bias should produce short string: '{}'", low_str);
                assert!(high_str.len() >= 15, "High bias should produce long string: '{}'", high_str);
            }
            _ => panic!("Expected string values"),
        }

        println!("✓ Biased string generation test passed");
    }

    #[test]
    fn test_generate_custom_integer_templates() {
        let engine = create_test_engine();
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(-1000),
            max_value: Some(1000),
            weights: None,
            shrink_towards: Some(0),
        });

        // Test boundary_min template
        let min_choice = engine.generate_custom("boundary_min", ChoiceType::Integer, &constraints).unwrap();
        if let ChoiceValue::Integer(val) = min_choice.value {
            assert_eq!(val, -1000, "boundary_min should produce min_value");
        } else {
            panic!("Expected integer value");
        }

        // Test boundary_max template
        let max_choice = engine.generate_custom("boundary_max", ChoiceType::Integer, &constraints).unwrap();
        if let ChoiceValue::Integer(val) = max_choice.value {
            assert_eq!(val, 1000, "boundary_max should produce max_value");
        } else {
            panic!("Expected integer value");
        }

        // Test zero template
        let zero_choice = engine.generate_custom("zero", ChoiceType::Integer, &constraints).unwrap();
        if let ChoiceValue::Integer(val) = zero_choice.value {
            assert_eq!(val, 0, "zero template should produce 0");
        } else {
            panic!("Expected integer value");
        }

        // Test one template
        let one_choice = engine.generate_custom("one", ChoiceType::Integer, &constraints).unwrap();
        if let ChoiceValue::Integer(val) = one_choice.value {
            assert_eq!(val, 1, "one template should produce 1");
        } else {
            panic!("Expected integer value");
        }

        println!("✓ Custom integer template generation test passed");
    }

    #[test]
    fn test_generate_custom_float_templates() {
        let engine = create_test_engine();
        let constraints = Constraints::Float(FloatConstraints {
            min_value: f64::NEG_INFINITY,
            max_value: f64::INFINITY,
            allow_nan: true,
            smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
        });

        // Test special float values
        let zero_choice = engine.generate_custom("zero", ChoiceType::Float, &constraints).unwrap();
        if let ChoiceValue::Float(val) = zero_choice.value {
            assert_eq!(val, 0.0, "zero template should produce 0.0");
        } else {
            panic!("Expected float value");
        }

        let inf_choice = engine.generate_custom("infinity", ChoiceType::Float, &constraints).unwrap();
        if let ChoiceValue::Float(val) = inf_choice.value {
            assert!(val.is_infinite(), "infinity template should produce infinity");
        } else {
            panic!("Expected float value");
        }

        let nan_choice = engine.generate_custom("nan", ChoiceType::Float, &constraints).unwrap();
        if let ChoiceValue::Float(val) = nan_choice.value {
            assert!(val.is_nan(), "nan template should produce NaN");
        } else {
            panic!("Expected float value");
        }

        println!("✓ Custom float template generation test passed");
    }

    #[test]
    fn test_generate_custom_boolean_templates() {
        let engine = create_test_engine();
        let constraints = Constraints::Boolean(BooleanConstraints::default());

        let true_choice = engine.generate_custom("true", ChoiceType::Boolean, &constraints).unwrap();
        if let ChoiceValue::Boolean(val) = true_choice.value {
            assert!(val, "true template should produce true");
        } else {
            panic!("Expected boolean value");
        }

        let false_choice = engine.generate_custom("false", ChoiceType::Boolean, &constraints).unwrap();
        if let ChoiceValue::Boolean(val) = false_choice.value {
            assert!(!val, "false template should produce false");
        } else {
            panic!("Expected boolean value");
        }

        println!("✓ Custom boolean template generation test passed");
    }

    #[test]
    fn test_generate_custom_string_templates() {
        let engine = create_test_engine();
        let constraints = Constraints::String(StringConstraints {
            min_size: 0,
            max_size: 100,
            intervals: IntervalSet::default(),
        });

        let empty_choice = engine.generate_custom("empty", ChoiceType::String, &constraints).unwrap();
        if let ChoiceValue::String(val) = empty_choice.value {
            assert!(val.is_empty(), "empty template should produce empty string");
        } else {
            panic!("Expected string value");
        }

        let single_choice = engine.generate_custom("single_char", ChoiceType::String, &constraints).unwrap();
        if let ChoiceValue::String(val) = single_choice.value {
            assert_eq!(val.len(), 1, "single_char template should produce single character");
        } else {
            panic!("Expected string value");
        }

        let unicode_choice = engine.generate_custom("unicode", ChoiceType::String, &constraints).unwrap();
        if let ChoiceValue::String(val) = unicode_choice.value {
            assert!(!val.is_ascii(), "unicode template should produce non-ASCII string");
            assert!(val.contains('α'), "unicode template should contain Greek letters");
        } else {
            panic!("Expected string value");
        }

        println!("✓ Custom string template generation test passed");
    }

    #[test]
    fn test_generate_custom_bytes_templates() {
        let engine = create_test_engine();
        let constraints = Constraints::Bytes(BytesConstraints {
            min_size: 0,
            max_size: 100,
        });

        let empty_choice = engine.generate_custom("empty", ChoiceType::Bytes, &constraints).unwrap();
        if let ChoiceValue::Bytes(val) = empty_choice.value {
            assert!(val.is_empty(), "empty template should produce empty bytes");
        } else {
            panic!("Expected bytes value");
        }

        let single_choice = engine.generate_custom("single_byte", ChoiceType::Bytes, &constraints).unwrap();
        if let ChoiceValue::Bytes(val) = single_choice.value {
            assert_eq!(val.len(), 1, "single_byte template should produce single byte");
            assert_eq!(val[0], 0x42, "single_byte should produce 'B' (0x42)");
        } else {
            panic!("Expected bytes value");
        }

        let null_choice = engine.generate_custom("null_bytes", ChoiceType::Bytes, &constraints).unwrap();
        if let ChoiceValue::Bytes(val) = null_choice.value {
            assert_eq!(val.len(), 4, "null_bytes should produce 4 bytes");
            assert!(val.iter().all(|&b| b == 0x00), "null_bytes should contain only null bytes");
        } else {
            panic!("Expected bytes value");
        }

        println!("✓ Custom bytes template generation test passed");
    }

    #[test]
    fn test_generate_sequence_from_index() {
        let engine = create_test_engine();
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        });

        let sequence = engine.generate_sequence_from_index(0, 5, ChoiceType::Integer, &constraints).unwrap();
        
        assert_eq!(sequence.len(), 5, "Should generate exactly 5 choices");
        
        // Verify they're all integers and in shrinking order
        for (i, choice) in sequence.iter().enumerate() {
            if let ChoiceValue::Integer(val) = &choice.value {
                // First should be shrink_towards (0), then increasing distance
                if i == 0 {
                    assert_eq!(*val, 0, "First choice should be shrink_towards");
                }
                assert!(*val >= 0 && *val <= 100, "Value {} should be within constraints", val);
            } else {
                panic!("Expected integer value at index {}", i);
            }
        }

        println!("✓ Sequence generation from index test passed");
    }

    #[test]
    fn test_generate_biased_sequence() {
        let engine = create_test_engine();
        let constraints = Constraints::Boolean(BooleanConstraints::default());

        let sequence = engine.generate_biased_sequence(3, ChoiceType::Boolean, &constraints).unwrap();
        
        assert_eq!(sequence.len(), 3, "Should generate exactly 3 choices");
        
        // Check that bias is distributed: 0.0, 0.5, 1.0
        if let (ChoiceValue::Boolean(v0), ChoiceValue::Boolean(v1), ChoiceValue::Boolean(v2)) = 
            (&sequence[0].value, &sequence[1].value, &sequence[2].value) {
            
            assert!(!v0, "First choice (bias=0.0) should be false");
            assert!(!v1, "Second choice (bias=0.5) should be false");  
            assert!(*v2, "Third choice (bias=1.0) should be true");
        } else {
            panic!("Expected boolean values");
        }

        println!("✓ Biased sequence generation test passed");
    }

    #[test]
    fn test_generate_all_custom_templates() {
        let engine = create_test_engine();
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(-50),
            max_value: Some(50),
            weights: None,
            shrink_towards: Some(0),
        });

        let templates = engine.generate_all_custom_templates(ChoiceType::Integer, &constraints).unwrap();
        
        // Should have all integer templates: boundary_min, boundary_max, zero, one
        assert_eq!(templates.len(), 4, "Should generate all 4 integer templates");
        
        let template_names: Vec<&str> = templates.iter().map(|(name, _)| name.as_str()).collect();
        assert!(template_names.contains(&"boundary_min"), "Should include boundary_min");
        assert!(template_names.contains(&"boundary_max"), "Should include boundary_max");
        assert!(template_names.contains(&"zero"), "Should include zero");
        assert!(template_names.contains(&"one"), "Should include one");

        println!("✓ All custom templates generation test passed");
    }

    #[test]
    fn test_bias_validation() {
        let engine = create_test_engine();
        let constraints = Constraints::Boolean(BooleanConstraints::default());

        // Test invalid bias values
        let result_negative = engine.generate_with_bias(-0.1, ChoiceType::Boolean, &constraints);
        assert!(result_negative.is_err(), "Negative bias should fail");

        let result_too_high = engine.generate_with_bias(1.1, ChoiceType::Boolean, &constraints);
        assert!(result_too_high.is_err(), "Bias > 1.0 should fail");

        // Test valid edge cases
        let result_zero = engine.generate_with_bias(0.0, ChoiceType::Boolean, &constraints);
        assert!(result_zero.is_ok(), "Bias 0.0 should succeed");

        let result_one = engine.generate_with_bias(1.0, ChoiceType::Boolean, &constraints);
        assert!(result_one.is_ok(), "Bias 1.0 should succeed");

        println!("✓ Bias validation test passed");
    }

    #[test]
    fn test_unknown_custom_template_fallback() {
        let engine = create_test_engine();
        let constraints = Constraints::Integer(IntegerConstraints::default());

        // Unknown template should fall back to simplest choice
        let result = engine.generate_custom("unknown_template", ChoiceType::Integer, &constraints);
        assert!(result.is_ok(), "Unknown template should fall back gracefully");

        if let Ok(choice) = result {
            if let ChoiceValue::Integer(val) = choice.value {
                assert_eq!(val, 0, "Unknown template should fall back to simplest (shrink_towards)");
            } else {
                panic!("Expected integer value");
            }
        }

        println!("✓ Unknown custom template fallback test passed");
    }

    #[test]
    fn test_advanced_template_integration() {
        let engine = create_test_engine();
        
        // Test that all three advanced methods work together
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(-10),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        });

        // Generate using index
        let index_choice = engine.generate_at_index(3, ChoiceType::Integer, &int_constraints).unwrap();
        
        // Generate using bias
        let biased_choice = engine.generate_with_bias(0.7, ChoiceType::Integer, &int_constraints).unwrap();
        
        // Generate using custom template
        let custom_choice = engine.generate_custom("boundary_max", ChoiceType::Integer, &int_constraints).unwrap();

        // Verify all three methods produce valid choices
        match (&index_choice.value, &biased_choice.value, &custom_choice.value) {
            (ChoiceValue::Integer(idx_val), ChoiceValue::Integer(bias_val), ChoiceValue::Integer(custom_val)) => {
                assert!(*idx_val >= -10 && *idx_val <= 10, "Index choice within constraints");
                assert!(*bias_val >= -10 && *bias_val <= 10, "Biased choice within constraints");
                assert_eq!(*custom_val, 10, "Custom choice should be boundary_max");
                
                // All three should be different approaches
                println!("Index choice: {}, Biased choice: {}, Custom choice: {}", idx_val, bias_val, custom_val);
            }
            _ => panic!("Expected integer values"),
        }

        println!("✓ Advanced template integration test passed");
    }
}