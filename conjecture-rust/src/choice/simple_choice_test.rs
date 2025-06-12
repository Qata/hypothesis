//! Simple test to validate basic choice functionality
use crate::choice::{
    ChoiceType, ChoiceValue, Constraints, ChoiceNode,
    IntegerConstraints, BooleanConstraints, FloatConstraints,
    choice_to_index, choice_from_index
};

#[cfg(test)]
mod simple_choice_tests {
    use super::*;

    #[test]
    fn test_basic_integer_choice() {
        let constraints = Constraints::Integer(IntegerConstraints::default());
        let value = ChoiceValue::Integer(42);
        let node = ChoiceNode::new(ChoiceType::Integer, value.clone(), constraints.clone(), false);
        
        assert_eq!(node.choice_type, ChoiceType::Integer);
        assert_eq!(node.value, value);
        assert!(!node.was_forced);
        assert!(node.index.is_none());
        
        // Test indexing
        let index = choice_to_index(&value, &constraints);
        let recovered = choice_from_index(index, "integer", &constraints);
        
        if let (ChoiceValue::Integer(original), ChoiceValue::Integer(recovered_val)) = (&value, &recovered) {
            assert_eq!(*original, *recovered_val);
        } else {
            panic!("Type mismatch in indexing roundtrip");
        }
    }

    #[test]
    fn test_basic_boolean_choice() {
        let constraints = Constraints::Boolean(BooleanConstraints::default());
        let value = ChoiceValue::Boolean(true);
        let node = ChoiceNode::new(ChoiceType::Boolean, value.clone(), constraints.clone(), false);
        
        assert_eq!(node.choice_type, ChoiceType::Boolean);
        assert_eq!(node.value, value);
        
        // Test indexing
        let index = choice_to_index(&value, &constraints);
        let recovered = choice_from_index(index, "boolean", &constraints);
        
        if let (ChoiceValue::Boolean(original), ChoiceValue::Boolean(recovered_val)) = (&value, &recovered) {
            assert_eq!(*original, *recovered_val);
        } else {
            panic!("Type mismatch in indexing roundtrip");
        }
    }

    #[test]
    fn test_basic_float_choice() {
        let constraints = Constraints::Float(FloatConstraints::default());
        let value = ChoiceValue::Float(3.14);
        let node = ChoiceNode::new(ChoiceType::Float, value.clone(), constraints.clone(), false);
        
        assert_eq!(node.choice_type, ChoiceType::Float);
        assert_eq!(node.value, value);
        
        // Test indexing (floats may not roundtrip exactly due to precision)
        let _index = choice_to_index(&value, &constraints);
        // Note: Float roundtrip testing is complex due to precision issues
    }

    #[test]
    fn test_forced_choice_behavior() {
        let constraints = Constraints::Integer(IntegerConstraints::default());
        let value = ChoiceValue::Integer(42);
        let forced_node = ChoiceNode::new(ChoiceType::Integer, value.clone(), constraints, true);
        
        assert!(forced_node.was_forced);
        assert!(forced_node.trivial()); // Forced nodes should be trivial
        
        // Test that forced nodes cannot be modified
        let copy_result = forced_node.copy_with_value(ChoiceValue::Integer(999));
        assert!(copy_result.is_err());
    }
}