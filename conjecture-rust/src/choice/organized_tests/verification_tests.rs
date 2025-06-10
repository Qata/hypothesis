//! Python parity verification tests
//! 
//! These tests implement the exact Python algorithms and verify our Rust
//! implementation produces identical results for critical test cases.

use crate::choice::*;

#[cfg(test)]
mod python_parity_verification {
    use super::*;

    /// Python's zigzag_index function implementation
    fn python_zigzag_index(value: i128, shrink_towards: i128) -> u128 {
        // Python: index = 2 * abs(shrink_towards - value)
        // Python: if value > shrink_towards: index -= 1
        let mut index = 2 * (shrink_towards - value).abs() as u128;
        if value > shrink_towards {
            index -= 1;
        }
        index
    }

    /// Python's zigzag_value function implementation
    fn python_zigzag_value(index: u128, shrink_towards: i128) -> i128 {
        // Python: n = (index + 1) // 2
        // Python: if (index % 2) == 0: n *= -1
        // Python: return shrink_towards + n
        let mut n = ((index + 1) / 2) as i128;
        if (index % 2) == 0 {
            n *= -1;
        }
        shrink_towards + n
    }

    /// Python's boolean indexing logic
    fn python_boolean_to_index(value: bool, p: f64) -> u128 {
        // Python: if not (2 ** (-64) < p < (1 - 2 ** (-64))): return 0
        // Python: return int(choice)
        if !(2_f64.powf(-64.0) < p && p < (1.0 - 2_f64.powf(-64.0))) {
            0
        } else {
            if value { 1 } else { 0 }
        }
    }

    #[test]
    fn test_python_zigzag_parity() {
        println!("VERIFICATION DEBUG: Testing zigzag function parity with Python");
        
        // Test with shrink_towards=0 (most common case)
        let test_cases = vec![
            // (value, expected_index)
            (0, 0), (1, 1), (-1, 2), (2, 3), (-2, 4), (3, 5), (-3, 6), (4, 7), (-4, 8)
        ];
        
        for (value, expected_python_index) in test_cases {
            let python_index = python_zigzag_index(value, 0);
            let rust_constraints = Constraints::Integer(IntegerConstraints::default());
            let rust_index = choice_to_index(&ChoiceValue::Integer(value), &rust_constraints);
            
            println!("VERIFICATION DEBUG: Value {} -> Python index {}, Rust index {}", 
                value, python_index, rust_index);
            
            assert_eq!(python_index, expected_python_index, 
                "Python zigzag function mismatch for value {}", value);
            assert_eq!(rust_index, python_index, 
                "Rust vs Python index mismatch for value {}", value);
            
            // Test reverse direction
            let python_value = python_zigzag_value(python_index, 0);
            let rust_value = choice_from_index(rust_index, "integer", &rust_constraints);
            
            assert_eq!(python_value, value, "Python zigzag_value mismatch");
            if let ChoiceValue::Integer(rust_val) = rust_value {
                assert_eq!(rust_val, value, "Rust choice_from_index mismatch");
            }
        }
        
        println!("VERIFICATION DEBUG: Zigzag parity test passed");
    }

    #[test]
    fn test_python_zigzag_custom_shrink_towards() {
        println!("VERIFICATION DEBUG: Testing zigzag with custom shrink_towards");
        
        // Test with shrink_towards=2
        let shrink_towards = 2;
        let test_values = vec![2, 3, 1, 4, 0, 5, -1, 6, -2];
        
        for (expected_index, value) in test_values.iter().enumerate() {
            let python_index = python_zigzag_index(*value, shrink_towards);
            
            let rust_constraints = Constraints::Integer(IntegerConstraints {
                min_value: None,
                max_value: None,
                weights: None,
                shrink_towards: Some(shrink_towards),
            });
            let rust_index = choice_to_index(&ChoiceValue::Integer(*value), &rust_constraints);
            
            println!("VERIFICATION DEBUG: Value {} -> Expected index {}, Python index {}, Rust index {}", 
                value, expected_index, python_index, rust_index);
            
            assert_eq!(python_index, expected_index as u128, 
                "Python zigzag index mismatch for value {} with shrink_towards={}", value, shrink_towards);
            assert_eq!(rust_index, python_index, 
                "Rust vs Python index mismatch for value {} with shrink_towards={}", value, shrink_towards);
        }
        
        println!("VERIFICATION DEBUG: Custom shrink_towards parity test passed");
    }

    #[test]
    fn test_python_boolean_parity() {
        println!("VERIFICATION DEBUG: Testing boolean indexing parity with Python");
        
        let test_cases = vec![
            // (p, expected_behavior)
            (0.0, "only_false"),
            (1.0, "only_true"), 
            (0.5, "both"),
            (2_f64.powf(-65.0), "only_false"),  // Very small p
            (1.0 - 2_f64.powf(-65.0), "only_true"),  // Very large p
        ];
        
        for (p, behavior) in test_cases {
            println!("VERIFICATION DEBUG: Testing boolean with p={}, behavior={}", p, behavior);
            
            let rust_constraints = Constraints::Boolean(BooleanConstraints { p });
            
            match behavior {
                "only_false" => {
                    // Only false should be valid
                    let python_false_index = python_boolean_to_index(false, p);
                    let rust_false_index = choice_to_index(&ChoiceValue::Boolean(false), &rust_constraints);
                    
                    assert_eq!(python_false_index, 0, "Python: false should have index 0 for p={}", p);
                    assert_eq!(rust_false_index, python_false_index, 
                        "Rust vs Python mismatch for false with p={}", p);
                    
                    // true should not be permitted in our system
                    // (Python would also return 0, but it's not meaningful)
                }
                "only_true" => {
                    // Only true should be valid
                    let python_true_index = python_boolean_to_index(true, p);
                    let rust_true_index = choice_to_index(&ChoiceValue::Boolean(true), &rust_constraints);
                    
                    assert_eq!(python_true_index, 0, "Python: true should have index 0 for p={}", p);
                    assert_eq!(rust_true_index, python_true_index, 
                        "Rust vs Python mismatch for true with p={}", p);
                }
                "both" => {
                    // Both true and false should be valid
                    let python_false_index = python_boolean_to_index(false, p);
                    let python_true_index = python_boolean_to_index(true, p);
                    let rust_false_index = choice_to_index(&ChoiceValue::Boolean(false), &rust_constraints);
                    let rust_true_index = choice_to_index(&ChoiceValue::Boolean(true), &rust_constraints);
                    
                    assert_eq!(python_false_index, 0, "Python: false should have index 0");
                    assert_eq!(python_true_index, 1, "Python: true should have index 1");
                    assert_eq!(rust_false_index, python_false_index, "Rust vs Python mismatch for false");
                    assert_eq!(rust_true_index, python_true_index, "Rust vs Python mismatch for true");
                }
                _ => panic!("Unknown behavior: {}", behavior),
            }
        }
        
        println!("VERIFICATION DEBUG: Boolean parity test passed");
    }

    #[test]
    fn test_python_bounded_integer_ordering() {
        println!("VERIFICATION DEBUG: Testing bounded integer ordering from Python test suite");
        
        // These are the exact test cases from Python's test_integer_choice_index
        let test_cases = vec![
            // (constraints, expected_ordering)
            (
                IntegerConstraints { min_value: Some(-3), max_value: Some(3), weights: None, shrink_towards: Some(0) },
                vec![0, 1, -1, 2, -2, 3, -3]
            ),
            (
                IntegerConstraints { min_value: Some(-3), max_value: Some(3), weights: None, shrink_towards: Some(1) },
                vec![1, 2, 0, 3, -1, -2, -3]
            ),
            (
                IntegerConstraints { min_value: Some(-3), max_value: Some(3), weights: None, shrink_towards: Some(-1) },
                vec![-1, 0, -2, 1, -3, 2, 3]
            ),
        ];
        
        for (constraints, expected_order) in test_cases {
            println!("VERIFICATION DEBUG: Testing constraints {:?}", constraints);
            let rust_constraints = Constraints::Integer(constraints.clone());
            
            for (expected_index, value) in expected_order.iter().enumerate() {
                let rust_index = choice_to_index(&ChoiceValue::Integer(*value), &rust_constraints);
                println!("VERIFICATION DEBUG: Value {} -> Expected index {}, Rust index {}", 
                    value, expected_index, rust_index);
                
                assert_eq!(rust_index, expected_index as u128, 
                    "Index mismatch for value {} with constraints {:?}", value, constraints);
                
                // Test reverse direction
                let recovered = choice_from_index(rust_index, "integer", &rust_constraints);
                if let ChoiceValue::Integer(recovered_val) = recovered {
                    assert_eq!(recovered_val, *value, 
                        "Roundtrip mismatch for value {}", value);
                }
            }
        }
        
        println!("VERIFICATION DEBUG: Bounded integer ordering test passed");
    }

    #[test]
    fn test_python_float_sign_bit_logic() {
        println!("VERIFICATION DEBUG: Testing float sign bit logic from Python");
        
        // Python's float indexing: sign = int(math.copysign(1.0, choice) < 0)
        // return (sign << 64) | float_to_lex(abs(choice))
        
        let test_cases = vec![
            (0.0, false),   // Positive zero
            (-0.0, true),   // Negative zero  
            (1.0, false),   // Positive
            (-1.0, true),   // Negative
            (f64::INFINITY, false),  // Positive infinity
            (f64::NEG_INFINITY, true), // Negative infinity
        ];
        
        for (value, expected_sign_bit) in test_cases {
            println!("VERIFICATION DEBUG: Testing float {} (sign bit should be {})", value, expected_sign_bit);
            
            let constraints = Constraints::Float(FloatConstraints::default());
            let index = choice_to_index(&ChoiceValue::Float(value), &constraints);
            
            // Extract sign bit (bit 64)
            let sign_bit = (index >> 64) != 0;
            
            assert_eq!(sign_bit, expected_sign_bit, 
                "Sign bit mismatch for float {}: expected {}, got {}", value, expected_sign_bit, sign_bit);
            
            // Test roundtrip
            let recovered = choice_from_index(index, "float", &constraints);
            if let ChoiceValue::Float(recovered_val) = recovered {
                // For regular values, should roundtrip exactly
                if value.is_finite() {
                    assert_eq!(recovered_val, value, "Float roundtrip mismatch for {}", value);
                } else if value.is_infinite() {
                    assert_eq!(recovered_val.is_infinite(), true, "Infinity not preserved");
                    assert_eq!(recovered_val.is_sign_positive(), value.is_sign_positive(), 
                        "Infinity sign not preserved for {}", value);
                }
                // NaN cases are tested elsewhere
            }
        }
        
        println!("VERIFICATION DEBUG: Float sign bit logic test passed");
    }

    #[test]
    fn test_edge_case_shrink_towards_clamping() {
        println!("VERIFICATION DEBUG: Testing shrink_towards clamping edge cases");
        
        // Python clamps shrink_towards to be within [min_value, max_value]
        let test_cases = vec![
            // (min, max, shrink_towards, expected_clamped, test_value)
            (Some(1), Some(5), 0, 1, 1),    // shrink_towards < min_value -> clamp to min
            (Some(1), Some(5), 10, 5, 5),   // shrink_towards > max_value -> clamp to max
            (Some(-10), None, -20, -10, -10), // shrink_towards < min_value, no max
            (None, Some(10), 20, 10, 10),   // shrink_towards > max_value, no min
        ];
        
        for (min_val, max_val, shrink_towards, expected_clamped, _test_value) in test_cases {
            println!("VERIFICATION DEBUG: Testing min={:?}, max={:?}, shrink_towards={}, expected_clamped={}", 
                min_val, max_val, shrink_towards, expected_clamped);
            
            let constraints = Constraints::Integer(IntegerConstraints {
                min_value: min_val,
                max_value: max_val,
                weights: None,
                shrink_towards: Some(shrink_towards),
            });
            
            // The clamped shrink_towards value should have index 0
            let clamped_index = choice_to_index(&ChoiceValue::Integer(expected_clamped), &constraints);
            assert_eq!(clamped_index, 0, 
                "Clamped shrink_towards value {} should have index 0", expected_clamped);
            
            // Index 0 should recover to the clamped value
            let recovered = choice_from_index(0, "integer", &constraints);
            if let ChoiceValue::Integer(recovered_val) = recovered {
                assert_eq!(recovered_val, expected_clamped, 
                    "Index 0 should recover to clamped shrink_towards value {}", expected_clamped);
            }
        }
        
        println!("VERIFICATION DEBUG: Shrink_towards clamping test passed");
    }
}