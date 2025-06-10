//! Integration tests for choice system cross-module functionality
//! 
//! These tests verify that different components of the choice system
//! work correctly together, focusing on interfaces and data flow.

use crate::choice::*;

#[cfg(test)]
mod indexing_integration_tests {
    use super::*;

    fn integer_constr_helper(min_value: Option<i128>, max_value: Option<i128>) -> IntegerConstraints {
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

    #[test]
    fn test_choice_index_and_value_are_inverses() {
        println!("INTEGRATION_TEST DEBUG: Testing that choice_to_index and choice_from_index are inverses");
        
        // Test various integer values
        let test_cases = vec![
            (integer_constr_helper(None, None), vec![0, 1, -1, 2, -2, 5]),
            (integer_constr_helper(Some(0), Some(10)), vec![0, 1, 2, 5, 10]),
            (integer_constr_with_shrink(Some(-5), Some(5), 2), vec![-5, -2, 0, 2, 3, 5]),
        ];
        
        for (constraints, values) in test_cases {
            for val in values {
                let value = ChoiceValue::Integer(val);
                let constraints_enum = Constraints::Integer(constraints.clone());
                
                let index = choice_to_index(&value, &constraints_enum);
                let recovered = choice_from_index(index, "integer", &constraints_enum);
                
                println!("INTEGRATION_TEST DEBUG: {} -> index {} -> {:?}", val, index, recovered);
                assert!(choice_equal(&value, &recovered), 
                    "Failed for value {}: got {:?}, expected {:?}", val, recovered, value);
            }
        }
        
        // Test boolean values - only test valid combinations
        let bool_test_cases = vec![
            (BooleanConstraints { p: 0.0 }, vec![false]),  // Only false is valid
            (BooleanConstraints { p: 1.0 }, vec![true]),   // Only true is valid
            (BooleanConstraints { p: 0.5 }, vec![true, false]), // Both are valid
        ];
        
        for (constraints, valid_values) in bool_test_cases {
            for val in valid_values {
                let value = ChoiceValue::Boolean(val);
                let constraints_enum = Constraints::Boolean(constraints.clone());
                
                let index = choice_to_index(&value, &constraints_enum);
                let recovered = choice_from_index(index, "boolean", &constraints_enum);
                
                println!("INTEGRATION_TEST DEBUG: {} (p={}) -> index {} -> {:?}", val, constraints.p, index, recovered);
                assert!(choice_equal(&value, &recovered),
                    "Failed for boolean {}: got {:?}, expected {:?}", val, recovered, value);
            }
        }
        
        println!("INTEGRATION_TEST DEBUG: Choice index and value are inverses test passed");
    }

    #[test]
    fn test_multi_type_choice_indexing() {
        println!("INTEGRATION_TEST DEBUG: Testing multi-type choice indexing");
        
        // Test that different choice types can coexist and index correctly
        let test_cases = vec![
            (ChoiceValue::Integer(42), Constraints::Integer(IntegerConstraints::default())),
            (ChoiceValue::Boolean(true), Constraints::Boolean(BooleanConstraints::default())),
            (ChoiceValue::Float(3.14), Constraints::Float(FloatConstraints::default())),
            (ChoiceValue::String("a".to_string()), Constraints::String(StringConstraints {
                min_size: 0,
                max_size: 10,
                intervals: IntervalSet::from_string("abc"),
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
            
            println!("INTEGRATION_TEST DEBUG: {:?} -> index {} -> {:?}", value, index, recovered);
            assert!(choice_equal(&value, &recovered),
                "Multi-type indexing failed for {:?}", value);
        }
        
        println!("INTEGRATION_TEST DEBUG: Multi-type choice indexing test passed");
    }

    #[test]
    fn test_u128_index_space_usage() {
        println!("INTEGRATION_TEST DEBUG: Testing u128 index space usage");
        
        // Test that we can handle large indices across different types
        let float_constraints = Constraints::Float(FloatConstraints::default());
        
        // Large floats should produce large indices
        let large_float = 1e100;
        let index = choice_to_index(&ChoiceValue::Float(large_float), &float_constraints);
        
        println!("INTEGRATION_TEST DEBUG: Large float {} -> index {}", large_float, index);
        
        // Verify roundtrip works with large indices
        let recovered = choice_from_index(index, "float", &float_constraints);
        if let ChoiceValue::Float(recovered_val) = recovered {
            assert_eq!(recovered_val, large_float, 
                "Large index roundtrip failed: {} -> {} -> {}", 
                large_float, index, recovered_val);
        }
        
        println!("INTEGRATION_TEST DEBUG: u128 index space usage test passed");
    }

    #[test]
    fn test_constraint_validation_integration() {
        println!("INTEGRATION_TEST DEBUG: Testing constraint validation integration");
        
        // Test that constraints are properly validated during indexing
        let bounded_constraints = Constraints::Integer(integer_constr_helper(Some(0), Some(10)));
        
        // Value within bounds should work
        let valid_value = ChoiceValue::Integer(5);
        let index = choice_to_index(&valid_value, &bounded_constraints);
        let recovered = choice_from_index(index, "integer", &bounded_constraints);
        assert!(choice_equal(&valid_value, &recovered));
        
        // Test constraint preservation through roundtrip
        for i in 0..=10 {
            let value = ChoiceValue::Integer(i);
            let index = choice_to_index(&value, &bounded_constraints);
            let recovered = choice_from_index(index, "integer", &bounded_constraints);
            
            if let ChoiceValue::Integer(recovered_int) = recovered {
                assert!(recovered_int >= 0 && recovered_int <= 10,
                    "Recovered value {} out of constraint bounds [0, 10]", recovered_int);
            }
        }
        
        println!("INTEGRATION_TEST DEBUG: Constraint validation integration test passed");
    }
}