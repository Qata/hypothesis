//! Tests ported from Python's test_choice.py for parity verification
//! 
//! This module contains direct ports of Python tests to ensure our Rust
//! implementation matches Python's behavior exactly.

use crate::choice::{
    ChoiceNode, ChoiceType, ChoiceValue, Constraints,
    IntegerConstraints, BooleanConstraints, FloatConstraints, 
    StringConstraints, BytesConstraints, IntervalSet,
    choice_permitted
};

/// Helper function to create integer constraints (port of integer_constr)
pub fn integer_constr(min_value: Option<i128>, max_value: Option<i128>) -> IntegerConstraints {
    println!("PYTHON_PARITY DEBUG: Creating integer constraints min={:?}, max={:?}", min_value, max_value);
    IntegerConstraints {
        min_value,
        max_value,
        weights: None,
        shrink_towards: Some(0),
    }
}

/// Helper function to create float constraints (port of float_constr)
pub fn float_constr(min_value: f64, max_value: f64) -> FloatConstraints {
    println!("PYTHON_PARITY DEBUG: Creating float constraints min={}, max={}", min_value, max_value);
    FloatConstraints {
        min_value,
        max_value,
        allow_nan: true,
        smallest_nonzero_magnitude: f64::MIN_POSITIVE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Port of test_choice_permitted from Python
    mod test_choice_permitted_python_parity {
        use super::*;

        #[test]
        fn test_integer_out_of_range() {
            println!("PYTHON_PARITY DEBUG: Testing integer out of range");
            
            // Python: (0, integer_constr(1, 2), False)
            let constraints = Constraints::Integer(integer_constr(Some(1), Some(2)));
            assert!(!choice_permitted(&ChoiceValue::Integer(0), &constraints));
            
            // Python: (2, integer_constr(0, 1), False) 
            let constraints = Constraints::Integer(integer_constr(Some(0), Some(1)));
            assert!(!choice_permitted(&ChoiceValue::Integer(2), &constraints));
            
            println!("PYTHON_PARITY DEBUG: Integer out of range test passed");
        }

        #[test]
        fn test_integer_in_range() {
            println!("PYTHON_PARITY DEBUG: Testing integer in range");
            
            // Python: (10, integer_constr(0, 20), True)
            let constraints = Constraints::Integer(integer_constr(Some(0), Some(20)));
            assert!(choice_permitted(&ChoiceValue::Integer(10), &constraints));
            
            println!("PYTHON_PARITY DEBUG: Integer in range test passed");
        }

        #[test] 
        fn test_float_nan_handling() {
            println!("PYTHON_PARITY DEBUG: Testing float NaN handling");
            
            // Python: (math.nan, float_constr(0.0, 0.0), True)
            let constraints = Constraints::Float(float_constr(0.0, 0.0));
            assert!(choice_permitted(&ChoiceValue::Float(f64::NAN), &constraints));
            
            // Python: (math.nan, float_constr(0.0, 0.0, allow_nan=False), False)
            let mut constraints = float_constr(0.0, 0.0);
            constraints.allow_nan = false;
            let constraints = Constraints::Float(constraints);
            assert!(!choice_permitted(&ChoiceValue::Float(f64::NAN), &constraints));
            
            println!("PYTHON_PARITY DEBUG: Float NaN handling test passed");
        }

        #[test]
        fn test_float_in_range() {
            println!("PYTHON_PARITY DEBUG: Testing float in range");
            
            // Python: (1.0, float_constr(1.0, 1.0), True)
            let constraints = Constraints::Float(float_constr(1.0, 1.0));
            assert!(choice_permitted(&ChoiceValue::Float(1.0), &constraints));
            
            println!("PYTHON_PARITY DEBUG: Float in range test passed");
        }

        #[test]
        fn test_string_size_constraints() {
            println!("PYTHON_PARITY DEBUG: Testing string size constraints");
            
            // Python: ("abcd", {"min_size": 10, "max_size": 20, "intervals": IntervalSet.from_string("abcd")}, False)
            let constraints = Constraints::String(StringConstraints {
                min_size: 10,
                max_size: 20,
                intervals: IntervalSet::from_string("abcd"),
            });
            assert!(!choice_permitted(&ChoiceValue::String("abcd".to_string()), &constraints));
            
            // Python: ("abcd", {"min_size": 1, "max_size": 3, "intervals": IntervalSet.from_string("abcd")}, False)
            let constraints = Constraints::String(StringConstraints {
                min_size: 1,
                max_size: 3,
                intervals: IntervalSet::from_string("abcd"),
            });
            assert!(!choice_permitted(&ChoiceValue::String("abcd".to_string()), &constraints));
            
            println!("PYTHON_PARITY DEBUG: String size constraints test passed");
        }

        #[test]
        fn test_string_character_constraints() {
            println!("PYTHON_PARITY DEBUG: Testing string character constraints");
            
            // Python: ("abcd", {"min_size": 1, "max_size": 10, "intervals": IntervalSet.from_string("e")}, False)
            let constraints = Constraints::String(StringConstraints {
                min_size: 1,
                max_size: 10,
                intervals: IntervalSet::from_string("e"),
            });
            assert!(!choice_permitted(&ChoiceValue::String("abcd".to_string()), &constraints));
            
            // Python: ("e", {"min_size": 1, "max_size": 10, "intervals": IntervalSet.from_string("e")}, True)
            assert!(choice_permitted(&ChoiceValue::String("e".to_string()), &constraints));
            
            println!("PYTHON_PARITY DEBUG: String character constraints test passed");
        }

        #[test]
        fn test_bytes_size_constraints() {
            println!("PYTHON_PARITY DEBUG: Testing bytes size constraints");
            
            // Python: (b"a", {"min_size": 2, "max_size": 2}, False)
            let constraints = Constraints::Bytes(BytesConstraints {
                min_size: 2,
                max_size: 2,
            });
            assert!(!choice_permitted(&ChoiceValue::Bytes(b"a".to_vec()), &constraints));
            
            // Python: (b"aa", {"min_size": 2, "max_size": 2}, True)
            assert!(choice_permitted(&ChoiceValue::Bytes(b"aa".to_vec()), &constraints));
            
            // Python: (b"aa", {"min_size": 0, "max_size": 3}, True)
            let constraints = Constraints::Bytes(BytesConstraints {
                min_size: 0,
                max_size: 3,
            });
            assert!(choice_permitted(&ChoiceValue::Bytes(b"aa".to_vec()), &constraints));
            
            println!("PYTHON_PARITY DEBUG: Bytes size constraints test passed");
        }

        #[test]
        fn test_boolean_probability_constraints() {
            println!("PYTHON_PARITY DEBUG: Testing boolean probability constraints");
            
            // Python: (True, {"p": 0}, False)
            let constraints = Constraints::Boolean(BooleanConstraints { p: 0.0 });
            assert!(!choice_permitted(&ChoiceValue::Boolean(true), &constraints));
            
            // Python: (False, {"p": 0}, True)
            assert!(choice_permitted(&ChoiceValue::Boolean(false), &constraints));
            
            // Python: (True, {"p": 1}, True)
            let constraints = Constraints::Boolean(BooleanConstraints { p: 1.0 });
            assert!(choice_permitted(&ChoiceValue::Boolean(true), &constraints));
            
            // Python: (False, {"p": 1}, False)
            assert!(!choice_permitted(&ChoiceValue::Boolean(false), &constraints));
            
            // Python: (True, {"p": 0.5}, True) and (False, {"p": 0.5}, True)
            let constraints = Constraints::Boolean(BooleanConstraints { p: 0.5 });
            assert!(choice_permitted(&ChoiceValue::Boolean(true), &constraints));
            assert!(choice_permitted(&ChoiceValue::Boolean(false), &constraints));
            
            println!("PYTHON_PARITY DEBUG: Boolean probability constraints test passed");
        }
    }

    // Port of test_forced_nodes_are_trivial from Python
    #[test]
    fn test_forced_nodes_are_trivial_python_parity() {
        println!("PYTHON_PARITY DEBUG: Testing forced nodes triviality (Python parity)");
        
        // Test various forced node types from Python examples
        let forced_nodes = vec![
            ChoiceNode::new(
                ChoiceType::Float,
                ChoiceValue::Float(0.0),
                Constraints::Float(float_constr(f64::NEG_INFINITY, f64::INFINITY)),
                true,
            ),
            ChoiceNode::new(
                ChoiceType::Boolean,
                ChoiceValue::Boolean(false),
                Constraints::Boolean(BooleanConstraints { p: 0.5 }),
                true,
            ),
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(50),
                Constraints::Integer(integer_constr(Some(50), Some(100))),
                true,
            ),
            ChoiceNode::new(
                ChoiceType::String,
                ChoiceValue::String("aaaa".to_string()),
                Constraints::String(StringConstraints {
                    min_size: 4,
                    max_size: 8192,
                    intervals: IntervalSet::from_string("bcda"),
                }),
                true,
            ),
            ChoiceNode::new(
                ChoiceType::Bytes,
                ChoiceValue::Bytes(vec![0; 8]),
                Constraints::Bytes(BytesConstraints {
                    min_size: 8,
                    max_size: 8,
                }),
                true,
            ),
        ];

        for node in forced_nodes {
            println!("PYTHON_PARITY DEBUG: Testing forced node: {:?}", node.choice_type);
            assert!(node.trivial(), "Forced node should be trivial: {:?}", node);
        }
        
        println!("PYTHON_PARITY DEBUG: All forced nodes triviality tests passed");
    }

    // Port of test_cannot_modify_forced_nodes from Python
    #[test]
    fn test_cannot_modify_forced_nodes_python_parity() {
        println!("PYTHON_PARITY DEBUG: Testing forced node modification prevention (Python parity)");
        
        let forced_node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(integer_constr(None, None)),
            true, // forced
        );

        // This should fail with assertion error (like Python's pytest.raises(AssertionError))
        let result = forced_node.copy_with_value(ChoiceValue::Integer(100));
        assert!(result.is_err(), "Should not be able to modify forced node");
        
        println!("PYTHON_PARITY DEBUG: Forced node modification prevention test passed");
    }

    // Port of test_choice_node_equality from Python  
    #[test]
    fn test_choice_node_equality_python_parity() {
        println!("PYTHON_PARITY DEBUG: Testing choice node equality (Python parity)");
        
        let node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(integer_constr(None, None)),
            false,
        );

        // Test self-equality
        assert_eq!(node, node);
        
        // Test inequality with different type (Python: assert node != 42)
        // We can't directly test node != 42 in Rust, but we can test the concept
        let different_node = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints::default()),
            false,
        );
        assert_ne!(node, different_node);
        
        println!("PYTHON_PARITY DEBUG: Choice node equality test passed");
    }

    // Port of test_choice_node_is_hashable from Python
    #[test]
    fn test_choice_node_is_hashable_python_parity() {
        println!("PYTHON_PARITY DEBUG: Testing choice node hashability (Python parity)");
        
        let node = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(3.14),
            Constraints::Float(float_constr(0.0, 10.0)),
            false,
        );

        // Test that we can hash the node (Python just calls hash(node))
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(node.clone(), "test_value");
        
        assert_eq!(map.get(&node), Some(&"test_value"));
        
        println!("PYTHON_PARITY DEBUG: Choice node hashability test passed");
    }
}