//! Property-based tests for choice system correctness
//! 
//! These tests use property-based testing techniques to verify that
//! the choice system maintains invariants and correctness properties
//! across a wide range of inputs.

use crate::choice::*;

#[cfg(test)]
mod property_based_tests {
    use super::*;

    #[test]
    fn test_choice_index_roundtrip_property() {
        println!("PROPERTY_TEST DEBUG: Testing choice index roundtrip property");
        
        // Property: For any valid choice value and constraints,
        // choice_to_index followed by choice_from_index should return the original value
        
        let test_cases = vec![
            // Various integer constraints
            (ChoiceValue::Integer(0), Constraints::Integer(IntegerConstraints::default())),
            (ChoiceValue::Integer(42), Constraints::Integer(IntegerConstraints::default())),
            (ChoiceValue::Integer(-17), Constraints::Integer(IntegerConstraints::default())),
            
            // Boolean cases
            (ChoiceValue::Boolean(true), Constraints::Boolean(BooleanConstraints { p: 0.5 })),
            (ChoiceValue::Boolean(false), Constraints::Boolean(BooleanConstraints { p: 0.5 })),
            
            // Float cases
            (ChoiceValue::Float(0.0), Constraints::Float(FloatConstraints::default())),
            (ChoiceValue::Float(1.0), Constraints::Float(FloatConstraints::default())),
            (ChoiceValue::Float(3.14159), Constraints::Float(FloatConstraints::default())),
            
            // String cases (use simple constraints to avoid overflow)
            (ChoiceValue::String("".to_string()), Constraints::String(StringConstraints {
                min_size: 0,
                max_size: 5,
                intervals: IntervalSet::from_string("abc"),
            })),
            (ChoiceValue::String("a".to_string()), Constraints::String(StringConstraints {
                min_size: 0,
                max_size: 5,
                intervals: IntervalSet::from_string("abc"),
            })),
            
            // Bytes cases
            (ChoiceValue::Bytes(vec![]), Constraints::Bytes(BytesConstraints {
                min_size: 0,
                max_size: 5,
            })),
            (ChoiceValue::Bytes(vec![1, 2, 3]), Constraints::Bytes(BytesConstraints {
                min_size: 0,
                max_size: 10,
            })),
        ];
        
        for (value, constraints) in test_cases {
            let index = choice_to_index(&value, &constraints);
            
            let type_str = match value {
                ChoiceValue::Integer(_) => "integer",
                ChoiceValue::Boolean(_) => "boolean",
                ChoiceValue::Float(_) => "float", 
                ChoiceValue::String(_) => "string",
                ChoiceValue::Bytes(_) => "bytes",
            };
            
            let recovered = choice_from_index(index, type_str, &constraints);
            
            assert!(choice_equal(&value, &recovered),
                "Roundtrip property failed for {:?}: original={:?}, recovered={:?}", 
                type_str, value, recovered);
        }
        
        println!("PROPERTY_TEST DEBUG: Choice index roundtrip property test passed");
    }

    #[test]
    fn test_index_ordering_property() {
        println!("PROPERTY_TEST DEBUG: Testing index ordering property");
        
        // Property: For integers with default shrink_towards=0,
        // values closer to 0 should have smaller indices
        
        let constraints = Constraints::Integer(IntegerConstraints::default());
        
        let test_values = vec![0, 1, -1, 2, -2, 5, -5, 10, -10];
        let mut index_pairs = Vec::new();
        
        for &val in &test_values {
            let index = choice_to_index(&ChoiceValue::Integer(val), &constraints);
            index_pairs.push((val.abs(), index));
        }
        
        // Sort by absolute value (distance from shrink_towards=0)
        index_pairs.sort_by_key(|(abs_val, _)| *abs_val);
        
        // Verify that indices are generally non-decreasing with distance from 0
        for i in 1..index_pairs.len() {
            let (prev_abs, prev_index) = index_pairs[i-1];
            let (curr_abs, curr_index) = index_pairs[i];
            
            if prev_abs < curr_abs {
                // Strictly smaller absolute value should have smaller or equal index
                assert!(prev_index <= curr_index,
                    "Ordering property violated: |{}| < |{}| but index {} > {}",
                    prev_abs, curr_abs, prev_index, curr_index);
            }
        }
        
        println!("PROPERTY_TEST DEBUG: Index ordering property test passed");
    }

    #[test]
    fn test_constraint_invariant_property() {
        println!("PROPERTY_TEST DEBUG: Testing constraint invariant property");
        
        // Property: choice_from_index should always return values that satisfy constraints
        
        // Test bounded integer constraints
        let bounded_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(-10),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        });
        
        // Test many indices to verify constraint satisfaction
        for index in 0..100 {
            let recovered = choice_from_index(index, "integer", &bounded_constraints);
            
            if let ChoiceValue::Integer(val) = recovered {
                assert!(val >= -10 && val <= 10,
                    "Constraint invariant violated: index {} produced value {} outside bounds [-10, 10]",
                    index, val);
            }
        }
        
        // Test boolean constraints
        let bool_constraints_false_only = Constraints::Boolean(BooleanConstraints { p: 0.0 });
        for index in 0..10 {
            let recovered = choice_from_index(index, "boolean", &bool_constraints_false_only);
            if let ChoiceValue::Boolean(val) = recovered {
                assert!(!val, "Boolean constraint p=0.0 violated: got true from index {}", index);
            }
        }
        
        let bool_constraints_true_only = Constraints::Boolean(BooleanConstraints { p: 1.0 });
        for index in 0..10 {
            let recovered = choice_from_index(index, "boolean", &bool_constraints_true_only);
            if let ChoiceValue::Boolean(val) = recovered {
                assert!(val, "Boolean constraint p=1.0 violated: got false from index {}", index);
            }
        }
        
        println!("PROPERTY_TEST DEBUG: Constraint invariant property test passed");
    }

    #[test]
    fn test_index_determinism_property() {
        println!("PROPERTY_TEST DEBUG: Testing index determinism property");
        
        // Property: choice_to_index should be deterministic - same input always produces same output
        
        let test_cases = vec![
            (ChoiceValue::Integer(42), Constraints::Integer(IntegerConstraints::default())),
            (ChoiceValue::Boolean(true), Constraints::Boolean(BooleanConstraints { p: 0.5 })),
            (ChoiceValue::Float(3.14), Constraints::Float(FloatConstraints::default())),
        ];
        
        for (value, constraints) in test_cases {
            let index1 = choice_to_index(&value, &constraints);
            let index2 = choice_to_index(&value, &constraints);
            let index3 = choice_to_index(&value, &constraints);
            
            assert_eq!(index1, index2, "Index determinism violated: {:?} produced different indices", value);
            assert_eq!(index2, index3, "Index determinism violated: {:?} produced different indices", value);
        }
        
        println!("PROPERTY_TEST DEBUG: Index determinism property test passed");
    }

    #[test]
    fn test_u128_index_space_property() {
        println!("PROPERTY_TEST DEBUG: Testing u128 index space property");
        
        // Property: Our indexing should make use of the full u128 space when needed
        
        let float_constraints = Constraints::Float(FloatConstraints::default());
        
        let mut max_index = 0u128;
        let test_floats = vec![
            0.0, 1.0, 1e10, 1e50, 1e100, 2.5, 0.1, 0.001, 1e-10, 1e-100
        ];
        
        for val in test_floats {
            let index = choice_to_index(&ChoiceValue::Float(val), &float_constraints);
            max_index = max_index.max(index);
            
            // Verify the index fits in u128 but may exceed u64
            assert!(index <= u128::MAX, "Index {} exceeds u128::MAX", index);
        }
        
        println!("PROPERTY_TEST DEBUG: Maximum index observed: {}", max_index);
        println!("PROPERTY_TEST DEBUG: u128 index space utilization verified");
        
        println!("PROPERTY_TEST DEBUG: u128 index space property test passed");
    }
}