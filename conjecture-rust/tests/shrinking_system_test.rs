/// Test the shrinking system implementation
/// 
/// This file contains tests for the shrinking algorithms ported from Python Hypothesis.
/// Each test validates that the shrinking behavior matches Python's implementation.

use conjecture::{
    choice::{
        ChoiceNode, ChoiceValue, ChoiceType, Constraints,
        IntegerConstraints, BooleanConstraints
    },
    data::ConjectureData,
    shrinking::{Shrinker, IntegerShrinker, shrink_integer, shrink_conjecture_data, find_integer}
};

/// Test basic integer shrinking functionality
#[test]
fn test_integer_shrinking_basic() {
    // Test that integer shrinking moves toward smaller values
    let predicate = |x: i128| x > 10;
    let result = shrink_integer(100, predicate);
    
    // Should shrink to the smallest value that still satisfies the predicate
    assert!(result > 10, "Result {} should be greater than 10", result);
    assert!(result < 100, "Result {} should be less than original value 100", result);
    assert!(result <= 15, "Result {} should be quite small", result); // Should make significant progress
}

/// Test integer shrinking edge cases
#[test] 
fn test_integer_shrinking_edge_cases() {
    // Test shrinking to zero
    let predicate_zero = |x: i128| x >= 0;
    let result_zero = shrink_integer(1000, predicate_zero);
    assert_eq!(result_zero, 0, "Should shrink to zero when possible");
    
    // Test shrinking to exact boundary
    let predicate_boundary = |x: i128| x >= 50;
    let result_boundary = shrink_integer(100, predicate_boundary);
    assert!(result_boundary >= 50, "Should not go below boundary");
    assert!(result_boundary < 60, "Should get close to boundary");
}

/// Test ConjectureData shrinking with integer choices
#[test]
fn test_conjecture_data_shrinking() {
    let choices = vec![
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(50),
            Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(100),
                weights: None,
                shrink_towards: Some(0),
            }),
            false,
        ),
    ];
    
    let original = ConjectureData::from_choices(&choices, 1);
    
    let predicate = |data: &ConjectureData| -> bool {
        data.get_nodes().iter().any(|choice| {
            if let ChoiceValue::Integer(val) = &choice.value {
                *val > 10
            } else {
                false
            }
        })
    };
    
    let result = shrink_conjecture_data(original, predicate);
    
    if let ChoiceValue::Integer(value) = &result.get_nodes()[0].value {
        assert!(*value <= 50, "Should not increase from original");
        assert!(*value > 10, "Should still satisfy predicate");
        assert!(*value < 30, "Should make significant progress toward target");
    } else {
        panic!("Expected integer value");
    }
}

/// Test Shrinker direct API usage
#[test]
fn test_shrinker_direct_usage() {
    let choices = vec![
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(100),
            Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(200),
                weights: None,
                shrink_towards: Some(0),
            }),
            false,
        ),
    ];
    
    let original = ConjectureData::from_choices(&choices, 1);
    
    let predicate = Box::new(|data: &ConjectureData| -> bool {
        if let Some(node) = data.get_nodes().get(0) {
            if let ChoiceValue::Integer(val) = &node.value {
                *val >= 25
            } else {
                false
            }
        } else {
            false
        }
    });
    
    let mut shrinker = Shrinker::new(original, predicate);
    let result = shrinker.shrink();
    
    if let Some(node) = result.get_nodes().get(0) {
        if let ChoiceValue::Integer(val) = &node.value {
            assert!(*val >= 25, "Should still satisfy predicate");
            assert!(*val < 100, "Should shrink from original");
        } else {
            panic!("Expected integer value");
        }
    } else {
        panic!("Expected at least one choice");
    }
}

/// Test IntegerShrinker direct usage
#[test] 
fn test_integer_shrinker_direct() {
    let predicate = Box::new(|x: i128| x > 7 && x < 50);
    let mut shrinker = IntegerShrinker::new(42, predicate);
    let result = shrinker.shrink();
    
    assert!(result > 7, "Should satisfy lower bound");
    assert!(result < 50, "Should satisfy upper bound"); 
    assert!(result < 42, "Should shrink from original");
    assert!(result <= 15, "Should make good progress");
}

/// Test shrinking with boolean values 
#[test]
fn test_boolean_shrinking() {
    let choices = vec![
        ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints { p: 0.5 }),
            false,
        ),
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(50),
            Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(100),
                weights: None,
                shrink_towards: Some(0),
            }),
            false,
        ),
    ];
    
    let original = ConjectureData::from_choices(&choices, 1);
    
    // Predicate: at least one boolean must be true OR integer > 20
    let predicate = |data: &ConjectureData| -> bool {
        let nodes = data.get_nodes();
        let has_true_bool = nodes.iter().any(|n| {
            matches!(&n.value, ChoiceValue::Boolean(true))
        });
        let has_large_int = nodes.iter().any(|n| {
            if let ChoiceValue::Integer(val) = &n.value {
                *val > 20
            } else {
                false
            }
        });
        has_true_bool || has_large_int
    };
    
    let result = shrink_conjecture_data(original, predicate);
    let nodes = result.get_nodes();
    
    // Should prefer false boolean (smaller sort key) over large integer
    assert!(nodes.len() <= 2, "Should not increase sequence length");
    
    // Verify that the result still satisfies the predicate
    let has_true_bool = nodes.iter().any(|n| matches!(&n.value, ChoiceValue::Boolean(true)));
    let has_large_int = nodes.iter().any(|n| {
        if let ChoiceValue::Integer(val) = &n.value {
            *val > 20
        } else {
            false
        }
    });
    
    assert!(has_true_bool || has_large_int, "Result should still satisfy predicate");
}

/// Test find_integer function - core Python utility
#[test]
fn test_find_integer_functionality() {
    // Test basic quadratic function
    let result = find_integer(|n| n * n <= 100);
    assert_eq!(result, 10, "Square root of 100 should be 10");
    
    // Test boundary conditions
    let result = find_integer(|n| n < 50);
    assert_eq!(result, 49, "Largest integer less than 50 should be 49");
    
    // Test small values (linear scan phase)
    let result = find_integer(|n| n <= 2);
    assert_eq!(result, 2, "Should handle small values correctly");
    
    // Test edge case: predicate never true
    let result = find_integer(|_| false);
    assert_eq!(result, -1, "Should return -1 when predicate never holds");
    
    // Test larger values (exponential + binary search phase)
    let result = find_integer(|n| n < 1000);
    assert_eq!(result, 999, "Should handle larger values efficiently");
}