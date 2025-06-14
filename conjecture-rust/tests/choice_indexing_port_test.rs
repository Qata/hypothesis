//! Direct port of Python choice indexing tests from hypothesis-python/tests/conjecture/test_choice.py
//! 
//! This file ports existing Python test logic to Rust to ensure behavioral parity.
//! These tests validate choice sequence recording, indexing, and deterministic replay capability.

use conjecture_rust::choice::{
    ChoiceValue, Constraints, IntegerConstraints, FloatConstraints, BooleanConstraints,
    StringConstraints, BytesConstraints, IntervalSet, ChoiceType, ChoiceNode,
    choice_to_index, choice_from_index, choice_equal, choice_permitted, choices_key,
    compute_max_children, all_children
};
use conjecture_rust::data::{ConjectureData, Status, choices_size};
use std::collections::{HashMap, HashSet};

// Test helper functions to match Python test patterns
fn integer_constr(min_value: Option<i128>, max_value: Option<i128>) -> IntegerConstraints {
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

fn float_constr(min_value: f64, max_value: f64) -> FloatConstraints {
    FloatConstraints {
        min_value: Some(min_value),
        max_value: Some(max_value),
        allow_nan: true,
        smallest_nonzero_magnitude: None,
    }
}

fn float_constr_full() -> FloatConstraints {
    FloatConstraints {
        min_value: None,
        max_value: None,
        allow_nan: true,
        smallest_nonzero_magnitude: None,
    }
}

fn string_constr(intervals: IntervalSet, min_size: usize, max_size: usize) -> StringConstraints {
    StringConstraints {
        intervals,
        min_size,
        max_size,
    }
}

fn bytes_constr(min_size: usize, max_size: usize) -> BytesConstraints {
    BytesConstraints {
        min_size,
        max_size,
    }
}

fn boolean_constr(p: f64) -> BooleanConstraints {
    BooleanConstraints { p }
}

// Port of test_compute_max_children_is_positive
#[test]
fn test_compute_max_children_is_positive() {
    // Test various constraint types to ensure max_children is always positive
    let test_cases = vec![
        (ChoiceType::Integer, Constraints::Integer(integer_constr(Some(1), Some(2)))),
        (ChoiceType::Boolean, Constraints::Boolean(boolean_constr(0.5))),
        (ChoiceType::Float, Constraints::Float(float_constr(0.0, 1.0))),
        (ChoiceType::String, Constraints::String(string_constr(
            IntervalSet::from_string("abc"), 0, 10
        ))),
        (ChoiceType::Bytes, Constraints::Bytes(bytes_constr(0, 5))),
    ];

    for (choice_type, constraints) in test_cases {
        let max_children = compute_max_children(&choice_type, &constraints);
        assert!(max_children >= 0, "max_children should be >= 0 for {:?}", choice_type);
    }
}

// Port of specific test_compute_max_children cases
#[test]
fn test_compute_max_children_specific_cases() {
    // Integer with weights - 2 possibilities
    let mut weights = HashMap::new();
    weights.insert(1, 0.1);
    weights.insert(2, 0.1);
    let integer_weights = IntegerConstraints {
        min_value: Some(1),
        max_value: Some(2),
        weights: Some(weights),
        shrink_towards: Some(0),
    };
    assert_eq!(
        compute_max_children(&ChoiceType::Integer, &Constraints::Integer(integer_weights)),
        2
    );

    // Empty string interval - only possibility is empty string
    assert_eq!(
        compute_max_children(
            &ChoiceType::String,
            &Constraints::String(string_constr(IntervalSet::from_string(""), 0, 100))
        ),
        1
    );

    // Boolean with p=0.0 - only False possible
    assert_eq!(
        compute_max_children(&ChoiceType::Boolean, &Constraints::Boolean(boolean_constr(0.0))),
        1
    );

    // Boolean with p=1.0 - only True possible  
    assert_eq!(
        compute_max_children(&ChoiceType::Boolean, &Constraints::Boolean(boolean_constr(1.0))),
        1
    );

    // Boolean with p=0.5 - both True and False possible
    assert_eq!(
        compute_max_children(&ChoiceType::Boolean, &Constraints::Boolean(boolean_constr(0.5))),
        2
    );

    // Float with same min/max - only one value possible
    assert_eq!(
        compute_max_children(&ChoiceType::Float, &Constraints::Float(float_constr(0.0, 0.0))),
        1
    );
}

// Port of test_choice_indices_are_positive  
#[test]
fn test_choice_indices_are_positive() {
    let test_cases = vec![
        (ChoiceValue::Integer(42), Constraints::Integer(integer_constr(None, None))),
        (ChoiceValue::Boolean(true), Constraints::Boolean(boolean_constr(0.5))),
        (ChoiceValue::Boolean(false), Constraints::Boolean(boolean_constr(0.0))),
        (ChoiceValue::Boolean(true), Constraints::Boolean(boolean_constr(1.0))),
        (ChoiceValue::Float(1.5), Constraints::Float(float_constr_full())),
        (ChoiceValue::String("test".to_string()), Constraints::String(string_constr(
            IntervalSet::from_string("abcdefghijklmnopqrstuvwxyz"), 0, 100
        ))),
        (ChoiceValue::Bytes(vec![1, 2, 3]), Constraints::Bytes(bytes_constr(0, 10))),
    ];

    for (value, constraints) in test_cases {
        let index = choice_to_index(&value, &constraints);
        assert!(index >= 0, "Index should be >= 0 for value {:?}", value);
    }
}

// Port of test_shrink_towards_has_index_0
#[test]
fn test_shrink_towards_has_index_0() {
    let test_cases = vec![
        integer_constr(None, None), // unbounded, shrink_towards=0
        integer_constr_with_shrink(Some(-10), Some(10), 5), // shrink_towards=5
        integer_constr_with_shrink(Some(3), None, 5), // min bounded, shrink_towards=5
        integer_constr_with_shrink(None, Some(10), -2), // max bounded, shrink_towards=-2
    ];

    for constraints in test_cases {
        let shrink_towards = constraints.shrink_towards.unwrap_or(0);
        let value = ChoiceValue::Integer(shrink_towards);
        let index = choice_to_index(&value, &Constraints::Integer(constraints.clone()));
        assert_eq!(index, 0, "Shrink towards value should have index 0");
        
        // Test roundtrip
        let recovered = choice_from_index(0, &ChoiceType::Integer, &Constraints::Integer(constraints));
        match recovered {
            ChoiceValue::Integer(v) => assert_eq!(v, shrink_towards),
            _ => panic!("Expected integer value"),
        }
    }
}

// Port of test_choice_to_index_injective
#[test]
fn test_choice_to_index_injective() {
    // Test that choice_to_index is injective (no two different choices have same index)
    let constraints = Constraints::Integer(integer_constr(Some(-5), Some(5)));
    let max_children = compute_max_children(&ChoiceType::Integer, &constraints);
    let cap = std::cmp::min(max_children as usize, 1000);
    
    let mut indices = HashSet::new();
    let choices: Vec<_> = all_children(&ChoiceType::Integer, &constraints).take(cap).collect();
    
    for choice in choices {
        let index = choice_to_index(&choice, &constraints);
        assert!(!indices.contains(&index), "Duplicate index {} for choice {:?}", index, choice);
        indices.insert(index);
    }
}

// Port of test_choice_from_value_injective  
#[test]
fn test_choice_from_value_injective() {
    // Test that choice_from_index is injective (no two indices produce same choice)
    let constraints = Constraints::Integer(integer_constr(Some(-3), Some(3)));
    let max_children = compute_max_children(&ChoiceType::Integer, &constraints);
    let cap = std::cmp::min(max_children as usize, 100);
    
    let mut choices = HashSet::new();
    for index in 0..cap {
        let choice = choice_from_index(index as u128, &ChoiceType::Integer, &constraints);
        let choice_key = choices_key(&vec![choice.clone()]);
        assert!(!choices.contains(&choice_key), "Duplicate choice {:?} from index {}", choice, index);
        choices.insert(choice_key);
    }
}

// Port of test_choice_index_and_value_are_inverses
#[test]
fn test_choice_index_and_value_are_inverses() {
    let test_cases = vec![
        (ChoiceValue::Integer(42), Constraints::Integer(integer_constr(None, None))),
        (ChoiceValue::Boolean(true), Constraints::Boolean(boolean_constr(0.5))),
        (ChoiceValue::Boolean(false), Constraints::Boolean(boolean_constr(0.5))),
        (ChoiceValue::Float(1.5), Constraints::Float(float_constr(1.0, 2.0))),
        (ChoiceValue::String("test".to_string()), Constraints::String(string_constr(
            IntervalSet::from_string("test"), 0, 10
        ))),
        (ChoiceValue::Bytes(vec![1, 2, 3]), Constraints::Bytes(bytes_constr(0, 10))),
    ];

    for (value, constraints) in test_cases {
        let index = choice_to_index(&value, &constraints);
        let choice_type = match &value {
            ChoiceValue::Integer(_) => ChoiceType::Integer,
            ChoiceValue::Boolean(_) => ChoiceType::Boolean,
            ChoiceValue::Float(_) => ChoiceType::Float,
            ChoiceValue::String(_) => ChoiceType::String,
            ChoiceValue::Bytes(_) => ChoiceType::Bytes,
        };
        let recovered = choice_from_index(index, &choice_type, &constraints);
        assert!(choice_equal(&value, &recovered), 
                "Roundtrip failed: {:?} -> {} -> {:?}", value, index, recovered);
    }
}

// Port of specific test_choice_index_and_value_are_inverses_explicit cases
#[test]
fn test_choice_index_and_value_inverses_explicit() {
    // Boolean with p=1 - only True
    let choice = ChoiceValue::Boolean(true);
    let constraints = Constraints::Boolean(boolean_constr(1.0));
    let index = choice_to_index(&choice, &constraints);
    let recovered = choice_from_index(index, &ChoiceType::Boolean, &constraints);
    assert!(choice_equal(&choice, &recovered));

    // Boolean with p=0 - only False
    let choice = ChoiceValue::Boolean(false);
    let constraints = Constraints::Boolean(boolean_constr(0.0));
    let index = choice_to_index(&choice, &constraints);
    let recovered = choice_from_index(index, &ChoiceType::Boolean, &constraints);
    assert!(choice_equal(&choice, &recovered));

    // Integer range with shrink_towards
    let choices = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    let constraints = Constraints::Integer(integer_constr_with_shrink(Some(1), None, 4));
    for choice_val in choices {
        let choice = ChoiceValue::Integer(choice_val);
        let index = choice_to_index(&choice, &constraints);
        let recovered = choice_from_index(index, &ChoiceType::Integer, &constraints);
        assert!(choice_equal(&choice, &recovered));
    }
}

// Port of test_integer_choice_index ordering tests
#[test]
fn test_integer_choice_index_ordering() {
    // Test specific ordering patterns from Python tests
    
    // Unbounded with default shrink_towards=0
    let constraints = Constraints::Integer(integer_constr(None, None));
    let expected_order = vec![0i128, 1, -1, 2, -2, 3, -3];
    for (expected_index, expected_value) in expected_order.iter().enumerate() {
        let choice = ChoiceValue::Integer(*expected_value);
        let actual_index = choice_to_index(&choice, &constraints);
        assert_eq!(actual_index, expected_index as u128, 
                  "Wrong index for value {} in unbounded range", expected_value);
    }

    // Bounded range -3 to 3 with shrink_towards=0  
    let constraints = Constraints::Integer(integer_constr(Some(-3), Some(3)));
    let expected_order = vec![0i128, 1, -1, 2, -2, 3, -3];
    for (expected_index, expected_value) in expected_order.iter().enumerate() {
        let choice = ChoiceValue::Integer(*expected_value);
        let actual_index = choice_to_index(&choice, &constraints);
        assert_eq!(actual_index, expected_index as u128,
                  "Wrong index for value {} in bounded range -3..3", expected_value);
    }

    // Bounded range with custom shrink_towards=1
    let constraints = Constraints::Integer(integer_constr_with_shrink(Some(-3), Some(3), 1));
    let expected_order = vec![1i128, 2, 0, 3, -1, -2, -3];
    for (expected_index, expected_value) in expected_order.iter().enumerate() {
        let choice = ChoiceValue::Integer(*expected_value);
        let actual_index = choice_to_index(&choice, &constraints);
        assert_eq!(actual_index, expected_index as u128,
                  "Wrong index for value {} with shrink_towards=1", expected_value);
    }
}

// Port of test_all_children_are_permitted_values
#[test]
fn test_all_children_are_permitted_values() {
    let test_cases = vec![
        (ChoiceType::Integer, Constraints::Integer(integer_constr(Some(-10), Some(10)))),
        (ChoiceType::Boolean, Constraints::Boolean(boolean_constr(0.5))),
        (ChoiceType::String, Constraints::String(string_constr(
            IntervalSet::from_string("abc"), 1, 3
        ))),
        (ChoiceType::Bytes, Constraints::Bytes(bytes_constr(0, 2))),
    ];

    for (choice_type, constraints) in test_cases {
        let max_children = compute_max_children(&choice_type, &constraints);
        let cap = std::cmp::min(max_children as usize, 1000);
        
        let choices: Vec<_> = all_children(&choice_type, &constraints).take(cap).collect();
        for choice in choices {
            assert!(choice_permitted(&choice, &constraints), 
                   "Generated choice {:?} should be permitted by constraints", choice);
        }
    }
}

// Port of test_choice_permitted specific cases
#[test]
fn test_choice_permitted_specific_cases() {
    // Integer out of bounds
    assert!(!choice_permitted(
        &ChoiceValue::Integer(0),
        &Constraints::Integer(integer_constr(Some(1), Some(2)))
    ));
    assert!(!choice_permitted(
        &ChoiceValue::Integer(3),
        &Constraints::Integer(integer_constr(Some(1), Some(2)))
    ));
    assert!(choice_permitted(
        &ChoiceValue::Integer(1),
        &Constraints::Integer(integer_constr(Some(1), Some(2)))
    ));

    // Boolean with probability constraints
    assert!(!choice_permitted(
        &ChoiceValue::Boolean(true),
        &Constraints::Boolean(boolean_constr(0.0))
    ));
    assert!(choice_permitted(
        &ChoiceValue::Boolean(false),
        &Constraints::Boolean(boolean_constr(0.0))
    ));
    assert!(choice_permitted(
        &ChoiceValue::Boolean(true),
        &Constraints::Boolean(boolean_constr(1.0))
    ));
    assert!(!choice_permitted(
        &ChoiceValue::Boolean(false),
        &Constraints::Boolean(boolean_constr(1.0))
    ));

    // String size constraints
    assert!(!choice_permitted(
        &ChoiceValue::String("a".to_string()),
        &Constraints::String(string_constr(IntervalSet::from_string("a"), 2, 2))
    ));
    assert!(choice_permitted(
        &ChoiceValue::String("aa".to_string()),
        &Constraints::String(string_constr(IntervalSet::from_string("a"), 2, 2))
    ));

    // Bytes size constraints  
    assert!(!choice_permitted(
        &ChoiceValue::Bytes(vec![1]),
        &Constraints::Bytes(bytes_constr(2, 2))
    ));
    assert!(choice_permitted(
        &ChoiceValue::Bytes(vec![1, 2]),
        &Constraints::Bytes(bytes_constr(2, 2))
    ));
}

// Port of test_drawing_directly_matches_for_choices (ConjectureData behavior)
#[test]
fn test_drawing_directly_matches_for_choices() {
    // Test that ConjectureData.for_choices replays exact values
    let test_choices = vec![
        ChoiceValue::String("test".to_string()),
        ChoiceValue::Bytes(vec![1, 2, 3]),
        ChoiceValue::Float(1.5),
        ChoiceValue::Boolean(true),
        ChoiceValue::Integer(42),
    ];

    let mut data = ConjectureData::for_choices(test_choices.clone());
    
    // Draw string with matching constraints
    let drawn_string = data.draw_string(&IntervalSet::from_string("test"), 1, 10);
    assert_eq!(drawn_string, "test");
    
    // Draw bytes
    let drawn_bytes = data.draw_bytes(0, 10);
    assert_eq!(drawn_bytes, vec![1, 2, 3]);
    
    // Draw float with matching constraints  
    let drawn_float = data.draw_float(Some(0.0), Some(2.0), true, None);
    assert_eq!(drawn_float, 1.5);
    
    // Draw boolean
    let drawn_bool = data.draw_boolean(0.5);
    assert_eq!(drawn_bool, true);
    
    // Draw integer
    let drawn_int = data.draw_integer(None, None, None, None);
    assert_eq!(drawn_int, 42);
}

// Port of test_data_with_empty_choices_is_overrun
#[test]
fn test_data_with_empty_choices_is_overrun() {
    let mut data = ConjectureData::for_choices(Vec::new());
    
    // Attempting to draw from empty choices should cause overrun
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        data.draw_integer(None, None, None, None);
    }));
    
    assert!(result.is_err(), "Drawing from empty choices should panic");
    assert_eq!(data.status(), Status::Overrun);
}

// Port of test_choices_size_positive
#[test] 
fn test_choices_size_positive() {
    let test_cases = vec![
        vec![],
        vec![ChoiceValue::Integer(1)],
        vec![ChoiceValue::Boolean(true), ChoiceValue::Float(1.5)],
        vec![
            ChoiceValue::String("test".to_string()),
            ChoiceValue::Bytes(vec![1, 2, 3]),
            ChoiceValue::Integer(42),
        ],
    ];

    for choices in test_cases {
        let size = choices_size(&choices);
        assert!(size >= 0, "choices_size should be >= 0");
    }
}

// Port of test_choices_key_distinguishes_weird_cases
#[test]
fn test_choices_key_distinguishes_weird_cases() {
    // Test that choices_key can distinguish between similar but different choices
    let test_cases = vec![
        (vec![ChoiceValue::Boolean(true)], vec![ChoiceValue::Integer(1)]),
        (vec![ChoiceValue::Boolean(false)], vec![ChoiceValue::Integer(0)]),
        (vec![ChoiceValue::Boolean(false)], vec![ChoiceValue::Float(0.0)]),
        (vec![ChoiceValue::Float(0.0)], vec![ChoiceValue::Float(-0.0)]),
    ];

    for (choices1, choices2) in test_cases {
        let key1 = choices_key(&choices1);
        let key2 = choices_key(&choices2);
        assert_ne!(key1, key2, "Different choice types should have different keys: {:?} vs {:?}", 
                  choices1, choices2);
    }
}

// Port of ChoiceNode behavior tests
#[test]
fn test_choice_node_equality() {
    let node1 = ChoiceNode {
        choice_type: ChoiceType::Integer,
        value: ChoiceValue::Integer(42),
        constraints: Constraints::Integer(integer_constr(None, None)),
        was_forced: false,
    };

    let node2 = ChoiceNode {
        choice_type: ChoiceType::Integer,
        value: ChoiceValue::Integer(42),
        constraints: Constraints::Integer(integer_constr(None, None)),
        was_forced: false,
    };

    let node3 = ChoiceNode {
        choice_type: ChoiceType::Integer,
        value: ChoiceValue::Integer(43),
        constraints: Constraints::Integer(integer_constr(None, None)),
        was_forced: false,
    };

    assert_eq!(node1, node2, "Identical nodes should be equal");
    assert_ne!(node1, node3, "Different nodes should not be equal");
}

// Port of forced node behavior tests
#[test]
fn test_forced_nodes_are_trivial() {
    let forced_node = ChoiceNode {
        choice_type: ChoiceType::Integer,
        value: ChoiceValue::Integer(42),
        constraints: Constraints::Integer(integer_constr(None, None)),
        was_forced: true,
    };

    assert!(forced_node.trivial(), "Forced nodes should be trivial");
}

#[test]
fn test_data_with_same_forced_value_is_valid() {
    // Test that ConjectureData handles forced values correctly when they match
    let test_value = ChoiceValue::Integer(42);
    let mut data = ConjectureData::for_choices(vec![test_value.clone()]);
    
    // Draw the same forced value - should work fine
    let drawn = data.draw_integer(None, None, None, Some(42));
    assert_eq!(drawn, 42);
}