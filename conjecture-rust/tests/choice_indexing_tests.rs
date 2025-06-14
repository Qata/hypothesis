//! Ported Python tests for choice indexing functionality
//! 
//! These tests are direct ports of the Python tests from:
//! - tests/conjecture/test_choice.py
//! - tests/conjecture/test_test_data.py  
//! - tests/cover/test_replay_logic.py

use conjecture::choice::{
    ChoiceValue, Constraints, IntegerConstraints, BooleanConstraints, FloatConstraints,
    StringConstraints, BytesConstraints, IntervalSet, choice_equal
};
use conjecture::choice::indexing::{choice_to_index, choice_from_index, clamped_shrink_towards};

// Helper functions to create constraints (ported from Python test utilities)

fn integer_constr() -> IntegerConstraints {
    IntegerConstraints {
        min_value: None,
        max_value: None,
        weights: None,
        shrink_towards: Some(0),
    }
}

fn integer_constr_bounded(min_value: Option<i128>, max_value: Option<i128>) -> IntegerConstraints {
    IntegerConstraints {
        min_value,
        max_value,
        weights: None,
        shrink_towards: Some(0),
    }
}

fn integer_constr_with_shrink(min_value: Option<i128>, max_value: Option<i128>, shrink_towards: i128) -> IntegerConstraints {
    IntegerConstraints {
        min_value,
        max_value,
        weights: None,
        shrink_towards: Some(shrink_towards),
    }
}

fn boolean_constr(p: f64) -> BooleanConstraints {
    BooleanConstraints { p }
}

fn float_constr() -> FloatConstraints {
    FloatConstraints {
        min_value: f64::NEG_INFINITY,
        max_value: f64::INFINITY,
        allow_nan: true,
        smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
    }
}

fn string_constr_basic() -> StringConstraints {
    StringConstraints {
        min_size: 0,
        max_size: 10,
        intervals: IntervalSet {
            intervals: vec![(b'a' as u32, b'z' as u32)], // lowercase a-z
        },
    }
}

fn bytes_constr_basic() -> BytesConstraints {
    BytesConstraints {
        min_size: 0,
        max_size: 10,
    }
}

// === PORTED TESTS FROM test_choice.py ===

/// Port of test_choice_indices_are_positive()
/// Tests that all choice indices are non-negative (>= 0)
#[test]
fn test_choice_indices_are_positive() {
    // Test integer choices
    let int_constraints = Constraints::Integer(integer_constr());
    let test_values = vec![0, 1, -1, 2, -2, 100, -100];
    
    for val in test_values {
        let choice = ChoiceValue::Integer(val);
        let index = choice_to_index(&choice, &int_constraints);
        assert!(index < u128::MAX, "Integer {} should have valid index, got {}", val, index);
    }
    
    // Test boolean choices
    let bool_constraints = Constraints::Boolean(boolean_constr(0.5));
    for val in [true, false] {
        let choice = ChoiceValue::Boolean(val);
        let index = choice_to_index(&choice, &bool_constraints);
        assert!(index < u128::MAX, "Boolean {} should have valid index, got {}", val, index);
    }
    
    // Test float choices
    let float_constraints = Constraints::Float(float_constr());
    let float_values = vec![0.0, 1.0, -1.0, 3.14, -2.71];
    
    for val in float_values {
        let choice = ChoiceValue::Float(val);
        let index = choice_to_index(&choice, &float_constraints);
        assert!(index < u128::MAX, "Float {} should have valid index, got {}", val, index);
    }
    
    // Test string choices
    let string_constraints = Constraints::String(string_constr_basic());
    let string_values = vec!["", "a", "hello", "test"];
    
    for val in string_values {
        let choice = ChoiceValue::String(val.to_string());
        let index = choice_to_index(&choice, &string_constraints);
        assert!(index < u128::MAX, "String '{}' should have valid index, got {}", val, index);
    }
    
    // Test bytes choices
    let bytes_constraints = Constraints::Bytes(bytes_constr_basic());
    let bytes_values = vec![vec![], vec![0], vec![1, 2, 3], vec![255, 128, 64]];
    
    for val in bytes_values {
        let choice = ChoiceValue::Bytes(val.clone());
        let index = choice_to_index(&choice, &bytes_constraints);
        assert!(index < u128::MAX, "Bytes {:?} should have valid index, got {}", val, index);
    }
}

/// Port of test_shrink_towards_has_index_0()
/// Tests that the shrink target always has index 0
#[test]
fn test_shrink_towards_has_index_0() {
    // Test unbounded integers
    let constraints = integer_constr();
    let shrink_target = clamped_shrink_towards(&constraints);
    let choice = ChoiceValue::Integer(shrink_target);
    let index = choice_to_index(&choice, &Constraints::Integer(constraints.clone()));
    assert_eq!(index, 0, "Unbounded shrink target {} should have index 0", shrink_target);
    
    // Test bounded integers 
    let bounded_constraints = integer_constr_bounded(Some(-5), Some(5));
    let bounded_shrink_target = clamped_shrink_towards(&bounded_constraints);
    let bounded_choice = ChoiceValue::Integer(bounded_shrink_target);
    let bounded_index = choice_to_index(&bounded_choice, &Constraints::Integer(bounded_constraints.clone()));
    assert_eq!(bounded_index, 0, "Bounded shrink target {} should have index 0", bounded_shrink_target);
    
    // Test custom shrink_towards
    let custom_constraints = integer_constr_with_shrink(Some(-10), Some(10), 3);
    let custom_shrink_target = clamped_shrink_towards(&custom_constraints);
    let custom_choice = ChoiceValue::Integer(custom_shrink_target);
    let custom_index = choice_to_index(&custom_choice, &Constraints::Integer(custom_constraints.clone()));
    assert_eq!(custom_index, 0, "Custom shrink target {} should have index 0", custom_shrink_target);
    
    // Test with out-of-bounds shrink_towards (should be clamped)
    let clamped_constraints = integer_constr_with_shrink(Some(1), Some(5), -10); // shrink_towards outside bounds
    let clamped_shrink_target = clamped_shrink_towards(&clamped_constraints);
    assert_eq!(clamped_shrink_target, 1, "Out-of-bounds shrink_towards should be clamped to min_value");
    let clamped_choice = ChoiceValue::Integer(clamped_shrink_target);
    let clamped_index = choice_to_index(&clamped_choice, &Constraints::Integer(clamped_constraints.clone()));
    assert_eq!(clamped_index, 0, "Clamped shrink target {} should have index 0", clamped_shrink_target);
}

/// Port of test_choice_index_and_value_are_inverses()
/// Tests round-trip conversion: choice → index → choice
#[test]
fn test_choice_index_and_value_are_inverses() {
    // Test integer roundtrips
    let int_test_cases = vec![
        (integer_constr(), vec![0, 1, -1, 2, -2, 5, -5]),
        (integer_constr_bounded(Some(0), Some(10)), vec![0, 1, 2, 5, 10]),
        (integer_constr_with_shrink(Some(-3), Some(3), 1), vec![-3, -1, 0, 1, 2, 3]),
    ];
    
    for (constraints, test_values) in int_test_cases {
        for val in test_values {
            let choice = ChoiceValue::Integer(val);
            let constraints_enum = Constraints::Integer(constraints.clone());
            
            let index = choice_to_index(&choice, &constraints_enum);
            let recovered = choice_from_index(index, "integer", &constraints_enum);
            
            assert!(choice_equal(&choice, &recovered), 
                "Integer roundtrip failed: {} -> index {} -> {:?}", val, index, recovered);
        }
    }
    
    // Test boolean roundtrips
    let bool_test_cases = vec![
        (boolean_constr(0.0), vec![false]),     // Only false valid
        (boolean_constr(1.0), vec![true]),      // Only true valid  
        (boolean_constr(0.5), vec![true, false]), // Both valid
    ];
    
    for (constraints, valid_values) in bool_test_cases {
        for val in valid_values {
            let choice = ChoiceValue::Boolean(val);
            let constraints_enum = Constraints::Boolean(constraints.clone());
            
            let index = choice_to_index(&choice, &constraints_enum);
            let recovered = choice_from_index(index, "boolean", &constraints_enum);
            
            assert!(choice_equal(&choice, &recovered),
                "Boolean roundtrip failed: {} -> index {} -> {:?}", val, index, recovered);
        }
    }
    
    // Test float roundtrips (basic values to avoid precision issues)
    let float_constraints = Constraints::Float(float_constr());
    let float_values = vec![0.0, 1.0, -1.0, 2.0, -2.0];
    
    for val in float_values {
        let choice = ChoiceValue::Float(val);
        
        let index = choice_to_index(&choice, &float_constraints);
        let recovered = choice_from_index(index, "float", &float_constraints);
        
        assert!(choice_equal(&choice, &recovered),
            "Float roundtrip failed: {} -> index {} -> {:?}", val, index, recovered);
    }
    
    // Test string roundtrips
    let string_constraints = Constraints::String(string_constr_basic());
    let string_values = vec!["", "a", "ab", "abc"];
    
    for val in string_values {
        let choice = ChoiceValue::String(val.to_string());
        
        let index = choice_to_index(&choice, &string_constraints);
        let recovered = choice_from_index(index, "string", &string_constraints);
        
        assert!(choice_equal(&choice, &recovered),
            "String roundtrip failed: '{}' -> index {} -> {:?}", val, index, recovered);
    }
    
    // Test bytes roundtrips
    let bytes_constraints = Constraints::Bytes(bytes_constr_basic());
    let bytes_values = vec![vec![], vec![0], vec![1, 2], vec![255]];
    
    for val in bytes_values {
        let choice = ChoiceValue::Bytes(val.clone());
        
        let index = choice_to_index(&choice, &bytes_constraints);
        let recovered = choice_from_index(index, "bytes", &bytes_constraints);
        
        assert!(choice_equal(&choice, &recovered),
            "Bytes roundtrip failed: {:?} -> index {} -> {:?}", val, index, recovered);
    }
}

/// Port of test_choice_to_index_injective()
/// Tests that choice_to_index is injective (one-to-one mapping)
#[test]
fn test_choice_to_index_injective() {
    // Test integer injectivity for bounded range
    let constraints = Constraints::Integer(integer_constr_bounded(Some(-5), Some(5)));
    let mut choices = Vec::new();
    let mut indices = Vec::new();
    
    for val in -5..=5 {
        let choice = ChoiceValue::Integer(val);
        let index = choice_to_index(&choice, &constraints);
        choices.push(choice);
        indices.push(index);
    }
    
    // Check that different choices have different indices
    for i in 0..choices.len() {
        for j in i+1..choices.len() {
            if !choice_equal(&choices[i], &choices[j]) {
                assert_ne!(indices[i], indices[j], 
                    "Different choices should have different indices: {:?} and {:?} both have index {}", 
                    choices[i], choices[j], indices[i]);
            }
        }
    }
    
    // Test boolean injectivity
    let bool_constraints = Constraints::Boolean(boolean_constr(0.5));
    let true_index = choice_to_index(&ChoiceValue::Boolean(true), &bool_constraints);
    let false_index = choice_to_index(&ChoiceValue::Boolean(false), &bool_constraints);
    assert_ne!(true_index, false_index, "true and false should have different indices");
    
    // Test string injectivity for small alphabet
    let string_constraints = Constraints::String(StringConstraints {
        min_size: 0,
        max_size: 2,
        intervals: IntervalSet {
            intervals: vec![(b'a' as u32, b'b' as u32)], // just 'a' and 'b'
        },
    });
    
    let string_test_cases = vec!["", "a", "b", "aa", "ab", "ba", "bb"];
    let mut string_indices = Vec::new();
    
    for val in &string_test_cases {
        let choice = ChoiceValue::String(val.to_string());
        let index = choice_to_index(&choice, &string_constraints);
        string_indices.push(index);
    }
    
    // Check string index uniqueness
    for i in 0..string_test_cases.len() {
        for j in i+1..string_test_cases.len() {
            if string_test_cases[i] != string_test_cases[j] {
                assert_ne!(string_indices[i], string_indices[j],
                    "Different strings should have different indices: '{}' and '{}' both have index {}",
                    string_test_cases[i], string_test_cases[j], string_indices[i]);
            }
        }
    }
}

/// Port of test_integer_choice_index() with parametrized test cases
/// Tests specific integer choice ordering matching Python implementation
#[test]
fn test_integer_choice_index_ordering() {
    // Test cases from Python's parametrized test - exact ordering validation
    let test_cases = vec![
        // unbounded
        (integer_constr(), vec![0, 1, -1, 2, -2, 3, -3]),
        (integer_constr_with_shrink(None, None, 2), vec![2, 3, 1, 4, 0, 5, -1]),
        
        // semibounded (below) 
        (integer_constr_with_shrink(Some(3), None, 0), vec![3, 4, 5, 6, 7]),
        (integer_constr_with_shrink(Some(3), None, 5), vec![5, 6, 4, 7, 3, 8]),
        (integer_constr_with_shrink(Some(-3), None, 0), vec![0, 1, -1, 2, -2, 3, -3]),
        (integer_constr_with_shrink(Some(-3), None, -1), vec![-1, 0, -2, 1, -3, 2, 3]),
        
        // semibounded (above)
        (integer_constr_bounded(None, Some(3)), vec![0, 1, -1, 2, -2, 3, -3]),
        (integer_constr_with_shrink(None, Some(3), 1), vec![1, 2, 0, 3, -1, -2, -3]),
        (integer_constr_with_shrink(None, Some(-3), -5), vec![-5, -4, -6, -3, -7, -8]),
        
        // bounded
        (integer_constr_bounded(Some(-3), Some(3)), vec![0, 1, -1, 2, -2, 3, -3]),
        (integer_constr_with_shrink(Some(-3), Some(3), 1), vec![1, 2, 0, 3, -1, -2, -3]),
        (integer_constr_with_shrink(Some(-3), Some(3), -1), vec![-1, 0, -2, 1, -3, 2, 3]),
    ];
    
    for (test_idx, (constraints, expected_order)) in test_cases.into_iter().enumerate() {
        println!("Running integer choice index test case {}", test_idx);
        
        for (expected_index, choice_value) in expected_order.into_iter().enumerate() {
            let choice = ChoiceValue::Integer(choice_value);
            let constraints_enum = Constraints::Integer(constraints.clone());
            let actual_index = choice_to_index(&choice, &constraints_enum);
            
            assert_eq!(actual_index, expected_index as u128,
                "Test case {}: Choice {} should have index {}, got {}",
                test_idx, choice_value, expected_index, actual_index);
        }
    }
}

/// Port of boolean choice index specific test cases
#[test]
fn test_boolean_choice_index_explicit() {
    // Test p=1: only true is possible
    let true_only_constraints = Constraints::Boolean(boolean_constr(1.0));
    let true_index = choice_to_index(&ChoiceValue::Boolean(true), &true_only_constraints);
    assert_eq!(true_index, 0, "With p=1.0, true should have index 0");
    
    // Test p=0: only false is possible
    let false_only_constraints = Constraints::Boolean(boolean_constr(0.0));
    let false_index = choice_to_index(&ChoiceValue::Boolean(false), &false_only_constraints);
    assert_eq!(false_index, 0, "With p=0.0, false should have index 0");
    
    // Test p=0.5: both values possible, false=0, true=1
    let both_constraints = Constraints::Boolean(boolean_constr(0.5));
    let false_index_both = choice_to_index(&ChoiceValue::Boolean(false), &both_constraints);
    let true_index_both = choice_to_index(&ChoiceValue::Boolean(true), &both_constraints);
    assert_eq!(false_index_both, 0, "With p=0.5, false should have index 0");
    assert_eq!(true_index_both, 1, "With p=0.5, true should have index 1");
}

/// Port of float choice ordering tests
#[test]
fn test_float_choice_index_ordering() {
    let float_constraints = Constraints::Float(float_constr());
    
    // Test that positive numbers have smaller indices than negative numbers (Python's system)
    let positive_val = ChoiceValue::Float(1.0);
    let negative_val = ChoiceValue::Float(-1.0);
    
    let positive_index = choice_to_index(&positive_val, &float_constraints);
    let negative_index = choice_to_index(&negative_val, &float_constraints);
    
    assert!(positive_index < negative_index,
        "Positive numbers should have smaller indices than negative numbers: {} vs {}",
        positive_index, negative_index);
    
    // Test zero cases
    let zero_pos = ChoiceValue::Float(0.0);
    let zero_neg = ChoiceValue::Float(-0.0);
    
    let zero_pos_index = choice_to_index(&zero_pos, &float_constraints);
    let zero_neg_index = choice_to_index(&zero_neg, &float_constraints);
    
    // Both zeros should have reasonably small indices
    assert!(zero_pos_index < 1000, "Positive zero should have small index");
    assert!(zero_neg_index < u64::MAX as u128, "Negative zero should have valid index");
}

/// Port of string collection indexing tests
#[test]
fn test_string_choice_collection_ordering() {
    // Create simple alphabet constraint (just 'a', 'b', 'c')
    let alphabet_constraints = Constraints::String(StringConstraints {
        min_size: 0,
        max_size: 3,
        intervals: IntervalSet {
            intervals: vec![(b'a' as u32, b'c' as u32)], // 'a' to 'c'
        },
    });
    
    // Test that empty string comes first
    let empty_string = ChoiceValue::String(String::new());
    let empty_index = choice_to_index(&empty_string, &alphabet_constraints);
    
    let single_a = ChoiceValue::String("a".to_string());
    let single_a_index = choice_to_index(&single_a, &alphabet_constraints);
    
    assert!(empty_index < single_a_index, 
        "Empty string should have smaller index than single character");
    
    // Test that 'a' comes before 'b'
    let single_b = ChoiceValue::String("b".to_string());
    let single_b_index = choice_to_index(&single_b, &alphabet_constraints);
    
    assert!(single_a_index < single_b_index,
        "String 'a' should have smaller index than 'b'");
    
    // Test that single chars come before double chars
    let double_aa = ChoiceValue::String("aa".to_string());
    let double_aa_index = choice_to_index(&double_aa, &alphabet_constraints);
    
    assert!(single_a_index < double_aa_index,
        "Single character strings should come before double character strings");
    assert!(single_b_index < double_aa_index,
        "Single character strings should come before double character strings");
}

/// Port of bytes choice collection indexing tests  
#[test]
fn test_bytes_choice_collection_ordering() {
    let bytes_constraints = Constraints::Bytes(bytes_constr_basic());
    
    // Test that empty bytes comes first
    let empty_bytes = ChoiceValue::Bytes(Vec::new());
    let empty_index = choice_to_index(&empty_bytes, &bytes_constraints);
    
    let single_zero = ChoiceValue::Bytes(vec![0]);
    let single_zero_index = choice_to_index(&single_zero, &bytes_constraints);
    
    assert!(empty_index < single_zero_index,
        "Empty bytes should have smaller index than single byte");
    
    // Test that [0] comes before [1]
    let single_one = ChoiceValue::Bytes(vec![1]);
    let single_one_index = choice_to_index(&single_one, &bytes_constraints);
    
    assert!(single_zero_index < single_one_index,
        "Bytes [0] should have smaller index than [1]");
    
    // Test that single bytes come before double bytes
    let double_zero = ChoiceValue::Bytes(vec![0, 0]);
    let double_zero_index = choice_to_index(&double_zero, &bytes_constraints);
    
    assert!(single_zero_index < double_zero_index,
        "Single byte should come before double bytes");
    assert!(single_one_index < double_zero_index,
        "Single byte should come before double bytes");
}

// === EDGE CASE TESTS ===

/// Test handling of extreme integer values
#[test]
fn test_integer_choice_extreme_values() {
    let unbounded_constraints = Constraints::Integer(integer_constr());
    
    // Test very large positive value
    let large_positive = ChoiceValue::Integer(i128::MAX);
    let large_positive_index = choice_to_index(&large_positive, &unbounded_constraints);
    let recovered_large_positive = choice_from_index(large_positive_index, "integer", &unbounded_constraints);
    assert!(choice_equal(&large_positive, &recovered_large_positive),
        "Large positive integer should roundtrip correctly");
    
    // Test very large negative value
    let large_negative = ChoiceValue::Integer(i128::MIN);
    let large_negative_index = choice_to_index(&large_negative, &unbounded_constraints);
    let recovered_large_negative = choice_from_index(large_negative_index, "integer", &unbounded_constraints);
    assert!(choice_equal(&large_negative, &recovered_large_negative),
        "Large negative integer should roundtrip correctly");
}

/// Test handling of special float values
#[test]
fn test_float_choice_special_values() {
    let float_constraints = Constraints::Float(float_constr());
    
    // Test infinity
    let pos_inf = ChoiceValue::Float(f64::INFINITY);
    let pos_inf_index = choice_to_index(&pos_inf, &float_constraints);
    assert!(pos_inf_index < u128::MAX, "Positive infinity should have valid index");
    
    let neg_inf = ChoiceValue::Float(f64::NEG_INFINITY);
    let neg_inf_index = choice_to_index(&neg_inf, &float_constraints);
    assert!(neg_inf_index < u128::MAX, "Negative infinity should have valid index");
    
    // Test NaN
    let nan_val = ChoiceValue::Float(f64::NAN);
    let nan_index = choice_to_index(&nan_val, &float_constraints);
    assert!(nan_index < u128::MAX, "NaN should have valid index");
    
    // Test very small values
    let tiny_val = ChoiceValue::Float(f64::MIN_POSITIVE);
    let tiny_index = choice_to_index(&tiny_val, &float_constraints);
    let recovered_tiny = choice_from_index(tiny_index, "float", &float_constraints);
    assert!(choice_equal(&tiny_val, &recovered_tiny),
        "Tiny float should roundtrip correctly");
}

/// Test constraint boundary conditions
#[test]
fn test_choice_constraint_boundaries() {
    // Test integer at exact boundary
    let boundary_constraints = Constraints::Integer(integer_constr_bounded(Some(-10), Some(10)));
    
    let min_val = ChoiceValue::Integer(-10);
    let min_index = choice_to_index(&min_val, &boundary_constraints);
    let recovered_min = choice_from_index(min_index, "integer", &boundary_constraints);
    assert!(choice_equal(&min_val, &recovered_min), "Minimum boundary value should roundtrip");
    
    let max_val = ChoiceValue::Integer(10);
    let max_index = choice_to_index(&max_val, &boundary_constraints);
    let recovered_max = choice_from_index(max_index, "integer", &boundary_constraints);
    assert!(choice_equal(&max_val, &recovered_max), "Maximum boundary value should roundtrip");
}

/// Test string with unicode characters
#[test]
fn test_string_choice_unicode() {
    let unicode_constraints = Constraints::String(StringConstraints {
        min_size: 0,
        max_size: 5,
        intervals: IntervalSet {
            intervals: vec![
                (0x41, 0x5A), // A-Z
                (0x61, 0x7A), // a-z  
                (0x30, 0x39), // 0-9
            ],
        },
    });
    
    let test_strings = vec!["A", "a", "0", "Aa", "A0", "aZ"];
    
    for test_str in test_strings {
        let choice = ChoiceValue::String(test_str.to_string());
        let index = choice_to_index(&choice, &unicode_constraints);
        let recovered = choice_from_index(index, "string", &unicode_constraints);
        
        assert!(choice_equal(&choice, &recovered),
            "Unicode string '{}' should roundtrip correctly", test_str);
    }
}

// === PERFORMANCE / STRESS TESTS ===

/// Test large collection handling
#[test]
fn test_large_collection_handling() {
    // Test string with larger max size
    let large_string_constraints = Constraints::String(StringConstraints {
        min_size: 0,
        max_size: 100, // Larger than typical tests
        intervals: IntervalSet {
            intervals: vec![(b'a' as u32, b'z' as u32)],
        },
    });
    
    // Test progressively larger strings
    let mut long_string = String::new();
    for i in 0..10 {
        if i > 0 {
            long_string.push('a');
        }
        
        let choice = ChoiceValue::String(long_string.clone());
        let index = choice_to_index(&choice, &large_string_constraints);
        
        // Should not panic or produce invalid indices
        assert!(index < u128::MAX, "Large string should have valid index");
        
        // Don't test full roundtrip for very large strings due to performance
        if long_string.len() <= 5 {
            let recovered = choice_from_index(index, "string", &large_string_constraints);
            assert!(choice_equal(&choice, &recovered),
                "String of length {} should roundtrip correctly", long_string.len());
        }
    }
}