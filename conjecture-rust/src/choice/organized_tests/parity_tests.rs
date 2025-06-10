//! Python Hypothesis parity tests
//! 
//! These tests verify that our Rust implementation produces identical results
//! to Python Hypothesis for the same inputs, ensuring cross-language compatibility.

use crate::choice::*;

#[cfg(test)]
mod python_parity_verification {
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
    fn test_integer_choice_index_ordering() {
        println!("PARITY_TEST DEBUG: Testing integer choice index ordering matches Python");
        
        // Test cases from Python's test_integer_choice_index
        let test_cases = vec![
            // unbounded
            (integer_constr_helper(None, None), vec![0, 1, -1, 2, -2, 3, -3]),
            (integer_constr_with_shrink(None, None, 2), vec![2, 3, 1, 4, 0, 5, -1]),
            // bounded
            (integer_constr_helper(Some(-3), Some(3)), vec![0, 1, -1, 2, -2, 3, -3]),
            (integer_constr_with_shrink(Some(-3), Some(3), 1), vec![1, 2, 0, 3, -1, -2, -3]),
        ];
        
        for (constraints, expected_order) in test_cases {
            println!("PARITY_TEST DEBUG: Testing constraints {:?}", constraints);
            
            for (expected_index, value) in expected_order.iter().enumerate() {
                let choice_value = ChoiceValue::Integer(*value);
                let constraints_enum = Constraints::Integer(constraints.clone());
                let actual_index = choice_to_index(&choice_value, &constraints_enum);
                
                println!("PARITY_TEST DEBUG: Value {} should have index {}, got {}", 
                    value, expected_index, actual_index);
                assert_eq!(actual_index, expected_index as u128, 
                    "Value {} should have index {}, got {}", value, expected_index, actual_index);
            }
        }
        
        println!("PARITY_TEST DEBUG: Integer choice index ordering test passed");
    }

    #[test]
    fn test_shrink_towards_has_index_0() {
        println!("PARITY_TEST DEBUG: Testing that shrink_towards value has index 0");
        
        // Test unbounded
        let constraints = integer_constr_helper(None, None);
        let shrink_towards = clamped_shrink_towards(&constraints);
        let index = choice_to_index(&ChoiceValue::Integer(shrink_towards), &Constraints::Integer(constraints.clone()));
        assert_eq!(index, 0);
        
        let value = choice_from_index(0, "integer", &Constraints::Integer(constraints));
        assert!(choice_equal(&value, &ChoiceValue::Integer(shrink_towards)));
        println!("PARITY_TEST DEBUG: Unbounded shrink_towards test passed");
        
        // Test bounded
        let constraints = integer_constr_helper(Some(-5), Some(5));
        let shrink_towards = clamped_shrink_towards(&constraints);
        let index = choice_to_index(&ChoiceValue::Integer(shrink_towards), &Constraints::Integer(constraints.clone()));
        assert_eq!(index, 0);
        
        let value = choice_from_index(0, "integer", &Constraints::Integer(constraints));
        assert!(choice_equal(&value, &ChoiceValue::Integer(shrink_towards)));
        println!("PARITY_TEST DEBUG: Bounded shrink_towards test passed");
        
        // Test with custom shrink_towards
        let constraints = integer_constr_with_shrink(Some(-5), Some(5), 2);
        let shrink_towards = clamped_shrink_towards(&constraints);
        let index = choice_to_index(&ChoiceValue::Integer(shrink_towards), &Constraints::Integer(constraints.clone()));
        assert_eq!(index, 0);
        
        let value = choice_from_index(0, "integer", &Constraints::Integer(constraints));
        assert!(choice_equal(&value, &ChoiceValue::Integer(shrink_towards)));
        println!("PARITY_TEST DEBUG: Custom shrink_towards test passed");
        
        println!("PARITY_TEST DEBUG: Shrink_towards has index 0 test passed");
    }

    #[test]
    fn test_boolean_choice_index_explicit() {
        println!("PARITY_TEST DEBUG: Testing boolean choice index explicit cases");
        
        // p=1: only true is possible
        let constraints = Constraints::Boolean(BooleanConstraints { p: 1.0 });
        let index = choice_to_index(&ChoiceValue::Boolean(true), &constraints);
        assert_eq!(index, 0);
        
        // p=0: only false is possible
        let constraints = Constraints::Boolean(BooleanConstraints { p: 0.0 });
        let index = choice_to_index(&ChoiceValue::Boolean(false), &constraints);
        assert_eq!(index, 0);
        
        // p=0.5: false=0, true=1
        let constraints = Constraints::Boolean(BooleanConstraints { p: 0.5 });
        let index_false = choice_to_index(&ChoiceValue::Boolean(false), &constraints);
        let index_true = choice_to_index(&ChoiceValue::Boolean(true), &constraints);
        assert_eq!(index_false, 0);
        assert_eq!(index_true, 1);
        
        println!("PARITY_TEST DEBUG: Boolean choice index explicit test passed");
    }

    #[test]
    fn test_python_float_simple_integers() {
        println!("PARITY_TEST DEBUG: Testing Python float simple integer parity");
        
        // These should match Python's simple integer encoding exactly
        let simple_integers = vec![0.0, 1.0, 2.0, 3.0, 10.0, 100.0, 1000.0];
        let float_constraints = Constraints::Float(FloatConstraints::default());
        
        for val in simple_integers {
            let index = choice_to_index(&ChoiceValue::Float(val), &float_constraints);
            let recovered = choice_from_index(index, "float", &float_constraints);
            
            if let ChoiceValue::Float(recovered_val) = recovered {
                assert_eq!(recovered_val, val, 
                    "Python parity failed for simple integer {}: {} -> {} -> {}", 
                    val, val, index, recovered_val);
            }
        }
        
        println!("PARITY_TEST DEBUG: Python float simple integer parity test passed");
    }

    #[test]
    fn test_comprehensive_integer_choice_index_scenarios() {
        println!("PARITY_TEST DEBUG: Testing comprehensive integer choice index scenarios from Python");
        
        // Test cases ported from Python's test_integer_choice_index
        let test_cases = vec![
            // unbounded
            (integer_constr_helper(None, None), vec![0, 1, -1, 2, -2, 3, -3]),
            (integer_constr_with_shrink(None, None, 2), vec![2, 3, 1, 4, 0, 5, -1, 6, -2]),
            // semibounded (below)
            (integer_constr_with_shrink(Some(3), None, 0), vec![3, 4, 5, 6, 7]),
            (integer_constr_with_shrink(Some(3), None, 5), vec![5, 6, 4, 7, 3, 8, 9]),
            (integer_constr_with_shrink(Some(-3), None, 0), vec![0, 1, -1, 2, -2, 3, -3, 4, 5, 6]),
            (integer_constr_with_shrink(Some(-3), None, -1), vec![-1, 0, -2, 1, -3, 2, 3, 4]),
            // semibounded (above)
            (integer_constr_helper(None, Some(3)), vec![0, 1, -1, 2, -2, 3, -3, -4, -5, -6]),
            (integer_constr_with_shrink(None, Some(3), 1), vec![1, 2, 0, 3, -1, -2, -3, -4]),
            (integer_constr_helper(None, Some(-3)), vec![-3, -4, -5, -6, -7]),
            (integer_constr_with_shrink(None, Some(-3), -5), vec![-5, -4, -6, -3, -7, -8, -9]),
            // bounded
            (integer_constr_helper(Some(-3), Some(3)), vec![0, 1, -1, 2, -2, 3, -3]),
            (integer_constr_with_shrink(Some(-3), Some(3), 1), vec![1, 2, 0, 3, -1, -2, -3]),
            (integer_constr_with_shrink(Some(-3), Some(3), -1), vec![-1, 0, -2, 1, -3, 2, 3]),
        ];
        
        for (test_index, (constraints, expected_choices)) in test_cases.iter().enumerate() {
            println!("PARITY_TEST DEBUG: Running test case {}: constraints={:?}", test_index, constraints);
            println!("PARITY_TEST DEBUG: Expected order: {:?}", expected_choices);
            
            for (expected_index, choice) in expected_choices.iter().enumerate() {
                let choice_value = ChoiceValue::Integer(*choice);
                let constraints_enum = Constraints::Integer(constraints.clone());
                let actual_index = choice_to_index(&choice_value, &constraints_enum);
                
                println!("PARITY_TEST DEBUG: Choice {} should have index {}, got {}", choice, expected_index, actual_index);
                assert_eq!(actual_index, expected_index as u128, 
                    "Test case {}: Choice {} should have index {}, got {}", 
                    test_index, choice, expected_index, actual_index);
            }
        }
        
        println!("PARITY_TEST DEBUG: Comprehensive integer choice index test passed");
    }
}