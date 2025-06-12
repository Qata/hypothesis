//! Unit tests for individual choice system components
//! 
//! These tests focus on testing individual functions and data structures
//! in isolation to ensure basic functionality works correctly.

use crate::choice::*;

#[cfg(test)]
mod choice_type_tests {
    use super::*;

    #[test]
    fn test_choice_type_display() {
        println!("UNIT_TEST DEBUG: Testing ChoiceType display formatting");
        assert_eq!(format!("{}", ChoiceType::Integer), "integer");
        assert_eq!(format!("{}", ChoiceType::Boolean), "boolean");
        assert_eq!(format!("{}", ChoiceType::Float), "float");
        assert_eq!(format!("{}", ChoiceType::String), "string");
        assert_eq!(format!("{}", ChoiceType::Bytes), "bytes");
        println!("UNIT_TEST DEBUG: All ChoiceType display tests passed");
    }

    #[test]
    fn test_choice_value_variants() {
        println!("UNIT_TEST DEBUG: Testing ChoiceValue variant creation");
        let _int_val = ChoiceValue::Integer(42);
        let _bool_val = ChoiceValue::Boolean(true);
        let _float_val = ChoiceValue::Float(3.14);
        let _string_val = ChoiceValue::String("hello".to_string());
        let _bytes_val = ChoiceValue::Bytes(vec![1, 2, 3]);
        println!("UNIT_TEST DEBUG: All ChoiceValue variants created successfully");
    }
}

#[cfg(test)]
mod constraint_tests {
    use super::*;

    #[test]
    fn test_integer_constraints_default() {
        println!("UNIT_TEST DEBUG: Testing IntegerConstraints default");
        let constraints = IntegerConstraints::default();
        println!("UNIT_TEST DEBUG: Default constraints: {:?}", constraints);
        
        assert_eq!(constraints.min_value, None);
        assert_eq!(constraints.max_value, None);
        assert_eq!(constraints.weights, None);
        assert_eq!(constraints.shrink_towards, Some(0));
        println!("UNIT_TEST DEBUG: IntegerConstraints default test passed");
    }

    #[test]
    fn test_boolean_constraints_default() {
        println!("UNIT_TEST DEBUG: Testing BooleanConstraints default");
        let constraints = BooleanConstraints::default();
        println!("UNIT_TEST DEBUG: Default constraints: {:?}", constraints);
        
        assert_eq!(constraints.p, 0.5);
        println!("UNIT_TEST DEBUG: BooleanConstraints default test passed");
    }

    #[test]
    fn test_float_constraints_default() {
        println!("UNIT_TEST DEBUG: Testing FloatConstraints default");
        let constraints = FloatConstraints::default();
        println!("UNIT_TEST DEBUG: Default constraints: {:?}", constraints);
        
        assert_eq!(constraints.min_value, f64::NEG_INFINITY);
        assert_eq!(constraints.max_value, f64::INFINITY);
        assert_eq!(constraints.allow_nan, true);
        assert_eq!(constraints.smallest_nonzero_magnitude, f64::MIN_POSITIVE);
        
        println!("UNIT_TEST DEBUG: FloatConstraints default test passed");
    }

    #[test]
    fn test_string_constraints_default() {
        println!("UNIT_TEST DEBUG: Testing StringConstraints default");
        let constraints = StringConstraints::default();
        println!("UNIT_TEST DEBUG: Default constraints: {:?}", constraints);
        
        assert_eq!(constraints.min_size, 0);
        assert_eq!(constraints.max_size, 8192);
        assert!(!constraints.intervals.is_empty());
        
        println!("UNIT_TEST DEBUG: StringConstraints default test passed");
    }

    #[test]
    fn test_bytes_constraints_default() {
        println!("UNIT_TEST DEBUG: Testing BytesConstraints default");
        let constraints = BytesConstraints::default();
        println!("UNIT_TEST DEBUG: Default constraints: {:?}", constraints);
        
        assert_eq!(constraints.min_size, 0);
        assert_eq!(constraints.max_size, 8192);
        
        println!("UNIT_TEST DEBUG: BytesConstraints default test passed");
    }

    #[test]
    fn test_interval_set_from_string() {
        println!("UNIT_TEST DEBUG: Testing IntervalSet from string");
        let intervals = IntervalSet::from_string("abc");
        println!("UNIT_TEST DEBUG: Created intervals: {:?}", intervals);
        
        // Should create intervals for 'a', 'b', 'c'
        assert_eq!(intervals.intervals.len(), 3);
        assert!(intervals.intervals.contains(&(97, 97))); // 'a'
        assert!(intervals.intervals.contains(&(98, 98))); // 'b'
        assert!(intervals.intervals.contains(&(99, 99))); // 'c'
        println!("UNIT_TEST DEBUG: IntervalSet from string test passed");
    }

    #[test]
    fn test_interval_set_empty() {
        println!("UNIT_TEST DEBUG: Testing empty IntervalSet");
        let intervals = IntervalSet::from_string("");
        println!("UNIT_TEST DEBUG: Empty intervals: {:?}", intervals);
        
        assert!(intervals.is_empty());
        println!("UNIT_TEST DEBUG: Empty IntervalSet test passed");
    }

    #[test]
    fn test_constraint_cloning() {
        println!("UNIT_TEST DEBUG: Testing constraint cloning");
        
        let int_constraints = IntegerConstraints {
            min_value: Some(-100),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(42),
        };
        let cloned = int_constraints.clone();
        assert_eq!(int_constraints, cloned);
        
        let float_constraints = FloatConstraints {
            min_value: 0.0,
            max_value: 1.0,
            allow_nan: false,
            smallest_nonzero_magnitude: Some(1e-10),
        };
        let cloned = float_constraints.clone();
        assert_eq!(float_constraints, cloned);
        
        println!("UNIT_TEST DEBUG: Constraint cloning test passed");
    }
}

#[cfg(test)]
mod node_tests {
    use super::*;

    fn integer_constr_helper(min_value: Option<i128>, max_value: Option<i128>) -> IntegerConstraints {
        IntegerConstraints {
            min_value,
            max_value,
            weights: None,
            shrink_towards: Some(0),
        }
    }

    #[test]
    fn test_choice_node_creation() {
        println!("UNIT_TEST DEBUG: Testing ChoiceNode creation");
        
        let constraints = integer_constr_helper(Some(0), Some(10));
        let value = ChoiceValue::Integer(5);
        let node = ChoiceNode::new(ChoiceType::Integer, value.clone(), Constraints::Integer(constraints), false);
        
        assert!(choice_equal(&node.value, &value));
        assert!(!node.was_forced);
        
        println!("UNIT_TEST DEBUG: ChoiceNode creation test passed");
    }

    #[test]
    fn test_choice_node_copy_with_value() {
        println!("UNIT_TEST DEBUG: Testing ChoiceNode copy_with_value");
        
        let constraints = integer_constr_helper(Some(0), Some(10));
        let original_value = ChoiceValue::Integer(5);
        let node = ChoiceNode::new(ChoiceType::Integer, original_value, Constraints::Integer(constraints), false);
        
        let new_value = ChoiceValue::Integer(7);
        let new_node = node.copy_with_value(new_value.clone()).unwrap();
        
        assert!(choice_equal(&new_node.value, &new_value));
        assert_eq!(new_node.was_forced, node.was_forced);
        
        println!("UNIT_TEST DEBUG: ChoiceNode copy_with_value test passed");
    }

    #[test]
    fn test_cannot_modify_forced_node() {
        println!("UNIT_TEST DEBUG: Testing that forced nodes cannot be modified");
        
        let constraints = integer_constr_helper(Some(0), Some(10));
        let forced_value = ChoiceValue::Integer(5);
        let node = ChoiceNode::new(ChoiceType::Integer, forced_value.clone(), Constraints::Integer(constraints), true);
        
        assert!(node.was_forced);
        assert!(choice_equal(&node.value, &forced_value));
        
        let new_value = ChoiceValue::Integer(7);
        let result = node.copy_with_value(new_value);
        
        // Should return an error for forced nodes
        assert!(result.is_err(), "Should not be able to modify forced nodes");
        
        println!("UNIT_TEST DEBUG: Cannot modify forced node test passed");
    }
}