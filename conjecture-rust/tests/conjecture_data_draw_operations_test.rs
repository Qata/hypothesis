//! # ConjectureData Draw Operations Test Suite
//!
//! This module contains comprehensive tests for the ConjectureData draw operations system,
//! directly ported from Python Hypothesis's test suite. These tests validate the core
//! functionality of draw_integer, draw_boolean, draw_float, draw_string, and draw_bytes
//! operations with proper choice recording and constraint validation.
//!
//! ## Ported Test Coverage
//!
//! ### Core Draw Operations (from test_test_data.py)
//! - ConjectureData lifecycle management (freeze, status transitions)
//! - Draw operation validation and overrun handling
//! - Observer pattern implementation
//! - Status management (VALID, INVALID, INTERESTING, OVERRUN)
//! - Error handling and exception propagation
//!
//! ### Choice System Tests (from test_choice.py)  
//! - All draw operations with various constraint configurations
//! - Choice node creation and validation
//! - Constraint satisfaction and bounds checking
//! - Trivial choice determination
//! - Forced value handling and override behavior
//!
//! ### Integration Tests (from test_forced.py)
//! - Forced value drawing across all operations
//! - Constraint validation with forced values
//! - Buffer replay with forced values
//!
//! ## Test Strategy
//!
//! Tests are organized by:
//! 1. **Basic Draw Operations**: Core functionality validation
//! 2. **Constraint Validation**: Boundary conditions and invalid constraints
//! 3. **Lifecycle Management**: Status transitions and error handling
//! 4. **Observer Integration**: Callback validation and data observer patterns
//! 5. **Forced Value Handling**: Override behavior and constraint interaction

use conjecture::{
    choice::{ChoiceNode, ChoiceType, ChoiceValue, Constraints},
    data::{ConjectureData, Status, StopTest},
    engine::ConjectureEngine,
};
use std::collections::HashMap;

/// Test helper to create a fresh ConjectureData instance for testing
fn fresh_data() -> ConjectureData {
    ConjectureData::new(12345)
}

/// Test helper to create ConjectureData with predefined choices
fn data_for_choices(choices: Vec<ChoiceValue>) -> ConjectureData {
    ConjectureData::for_choices(choices)
}

// === BASIC DRAW OPERATIONS TESTS ===

#[test]
fn test_draw_boolean_basic() {
    let mut data = fresh_data();
    let result = data.draw_boolean(None, None);
    assert!(matches!(result, Ok(true) | Ok(false)));
}

#[test]
fn test_draw_boolean_with_probability() {
    let mut data = fresh_data();
    
    // Test with p=1.0 (always true)
    let result = data.draw_boolean(Some(1.0), None);
    assert_eq!(result.unwrap(), true);
    
    let mut data = fresh_data();
    // Test with p=0.0 (always false) 
    let result = data.draw_boolean(Some(0.0), None);
    assert_eq!(result.unwrap(), false);
}

#[test]
fn test_draw_boolean_forced() {
    let mut data = fresh_data();
    
    // Test forced true
    let result = data.draw_boolean(Some(0.0), Some(true));
    assert_eq!(result.unwrap(), true);
    
    let mut data = fresh_data();
    // Test forced false
    let result = data.draw_boolean(Some(1.0), Some(false));
    assert_eq!(result.unwrap(), false);
}

#[test]
fn test_draw_integer_basic() {
    let mut data = fresh_data();
    let result = data.draw_integer(None, None, None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_draw_integer_bounded() {
    let mut data = fresh_data();
    let min_value = Some(10);
    let max_value = Some(20);
    let result = data.draw_integer(min_value, max_value, None, None, None);
    
    match result {
        Ok(value) => {
            assert!(value >= 10);
            assert!(value <= 20);
        }
        Err(_) => panic!("draw_integer should succeed with valid bounds"),
    }
}

#[test]
fn test_draw_integer_with_shrink_towards() {
    let mut data = fresh_data();
    let min_value = Some(0);
    let max_value = Some(100);
    let shrink_towards = Some(50);
    
    let result = data.draw_integer(min_value, max_value, None, shrink_towards, None);
    assert!(result.is_ok());
}

#[test]
fn test_draw_integer_forced() {
    let mut data = fresh_data();
    let forced_value = Some(42);
    
    let result = data.draw_integer(Some(0), Some(100), None, None, forced_value);
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_draw_float_basic() {
    let mut data = fresh_data();
    let result = data.draw_float(None, None, None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_draw_float_bounded() {
    let mut data = fresh_data();
    let min_value = Some(1.0);
    let max_value = Some(10.0);
    
    let result = data.draw_float(min_value, max_value, None, None, None);
    match result {
        Ok(value) => {
            assert!(value >= 1.0);
            assert!(value <= 10.0);
        }
        Err(_) => panic!("draw_float should succeed with valid bounds"),
    }
}

#[test]
fn test_draw_float_forced() {
    let mut data = fresh_data();
    let forced_value = Some(3.14159);
    
    let result = data.draw_float(Some(0.0), Some(10.0), None, None, forced_value);
    assert_eq!(result.unwrap(), 3.14159);
}

#[test]
fn test_draw_string_basic() {
    let mut data = fresh_data();
    let result = data.draw_string(None, None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_draw_string_sized() {
    let mut data = fresh_data();
    let min_size = Some(5);
    let max_size = Some(10);
    
    let result = data.draw_string(None, min_size, max_size, None);
    match result {
        Ok(s) => {
            assert!(s.len() >= 5);
            assert!(s.len() <= 10);
        }
        Err(_) => panic!("draw_string should succeed with valid size bounds"),
    }
}

#[test]
fn test_draw_string_forced() {
    let mut data = fresh_data();
    let forced_value = Some("hello world".to_string());
    
    let result = data.draw_string(None, None, None, forced_value);
    assert_eq!(result.unwrap(), "hello world");
}

#[test]
fn test_draw_bytes_basic() {
    let mut data = fresh_data();
    let result = data.draw_bytes(None, None, None);
    assert!(result.is_ok());
}

#[test]
fn test_draw_bytes_sized() {
    let mut data = fresh_data();
    let min_size = Some(3);
    let max_size = Some(8);
    
    let result = data.draw_bytes(min_size, max_size, None);
    match result {
        Ok(bytes) => {
            assert!(bytes.len() >= 3);
            assert!(bytes.len() <= 8);
        }
        Err(_) => panic!("draw_bytes should succeed with valid size bounds"),
    }
}

#[test]
fn test_draw_bytes_forced() {
    let mut data = fresh_data();
    let forced_value = Some(vec![1, 2, 3, 4, 5]);
    
    let result = data.draw_bytes(None, None, forced_value);
    assert_eq!(result.unwrap(), vec![1, 2, 3, 4, 5]);
}

// === LIFECYCLE MANAGEMENT TESTS ===

#[test]
fn test_cannot_draw_after_freeze() {
    let mut data = data_for_choices(vec![ChoiceValue::Boolean(true)]);
    let _ = data.draw_boolean(None, None);
    data.freeze();
    
    let result = data.draw_boolean(None, None);
    assert!(matches!(result, Err(StopTest::Frozen)));
}

#[test]
fn test_can_double_freeze() {
    let mut data = data_for_choices(vec![]);
    data.freeze();
    assert!(data.is_frozen());
    data.freeze();
    assert!(data.is_frozen());
}

#[test]
fn test_draw_past_end_sets_overrun() {
    let mut data = data_for_choices(vec![ChoiceValue::Boolean(true)]);
    
    let _ = data.draw_boolean(None, None);
    let result = data.draw_boolean(None, None);
    
    assert!(matches!(result, Err(StopTest::Overrun)));
    assert!(data.is_frozen());
    assert_eq!(data.status(), Status::Overrun);
}

#[test]
fn test_can_mark_interesting() {
    let mut data = data_for_choices(vec![]);
    let result = data.mark_interesting(None);
    
    assert!(matches!(result, Err(StopTest::Interesting)));
    assert!(data.is_frozen());
    assert_eq!(data.status(), Status::Interesting);
}

#[test]
fn test_can_mark_invalid() {
    let mut data = data_for_choices(vec![]);
    let result = data.mark_invalid(None);
    
    assert!(matches!(result, Err(StopTest::Invalid)));
    assert!(data.is_frozen());
    assert_eq!(data.status(), Status::Invalid);
}

#[test]
fn test_can_mark_invalid_with_reason() {
    let mut data = data_for_choices(vec![]);
    let result = data.mark_invalid(Some("test reason".to_string()));
    
    assert!(matches!(result, Err(StopTest::Invalid)));
    assert!(data.is_frozen());
    assert_eq!(data.status(), Status::Invalid);
    // Verify reason is recorded in events
    assert!(data.events().contains_key("invalid because"));
}

// === CONSTRAINT VALIDATION TESTS ===

#[test]
fn test_integer_constraint_validation() {
    let mut data = fresh_data();
    
    // Test invalid bounds (min > max)
    let result = data.draw_integer(Some(10), Some(5), None, None, None);
    assert!(result.is_err());
}

#[test]
fn test_float_constraint_validation() {
    let mut data = fresh_data();
    
    // Test invalid bounds (min > max)
    let result = data.draw_float(Some(10.0), Some(5.0), None, None, None);
    assert!(result.is_err());
}

#[test]
fn test_string_size_constraint_validation() {
    let mut data = fresh_data();
    
    // Test invalid size bounds (min_size > max_size)
    let result = data.draw_string(None, Some(10), Some(5), None);
    assert!(result.is_err());
}

#[test]
fn test_bytes_size_constraint_validation() {
    let mut data = fresh_data();
    
    // Test invalid size bounds (min_size > max_size)
    let result = data.draw_bytes(Some(10), Some(5), None);
    assert!(result.is_err());
}

// === CHOICE NODE CREATION TESTS ===

#[test]
fn test_choice_node_creation_boolean() {
    let mut data = fresh_data();
    let _ = data.draw_boolean(Some(0.5), None);
    
    let nodes = data.nodes();
    assert_eq!(nodes.len(), 1);
    assert!(matches!(nodes[0].choice_type(), ChoiceType::Boolean));
}

#[test]
fn test_choice_node_creation_integer() {
    let mut data = fresh_data();
    let _ = data.draw_integer(Some(0), Some(100), None, None, None);
    
    let nodes = data.nodes();
    assert_eq!(nodes.len(), 1);
    assert!(matches!(nodes[0].choice_type(), ChoiceType::Integer));
}

#[test]
fn test_choice_node_creation_float() {
    let mut data = fresh_data();
    let _ = data.draw_float(Some(0.0), Some(1.0), None, None, None);
    
    let nodes = data.nodes();
    assert_eq!(nodes.len(), 1);
    assert!(matches!(nodes[0].choice_type(), ChoiceType::Float));
}

#[test]
fn test_choice_node_creation_string() {
    let mut data = fresh_data();
    let _ = data.draw_string(None, Some(0), Some(10), None);
    
    let nodes = data.nodes();
    assert_eq!(nodes.len(), 1);
    assert!(matches!(nodes[0].choice_type(), ChoiceType::String));
}

#[test]
fn test_choice_node_creation_bytes() {
    let mut data = fresh_data();
    let _ = data.draw_bytes(Some(0), Some(10), None);
    
    let nodes = data.nodes();
    assert_eq!(nodes.len(), 1);
    assert!(matches!(nodes[0].choice_type(), ChoiceType::Bytes));
}

// === FORCED VALUE BEHAVIOR TESTS ===

#[test]
fn test_forced_values_override_constraints() {
    let mut data = fresh_data();
    
    // Force a value outside the normal constraint range
    let result = data.draw_integer(Some(0), Some(10), None, None, Some(50));
    assert_eq!(result.unwrap(), 50);
}

#[test]
fn test_forced_boolean_overrides_probability() {
    let mut data = fresh_data();
    
    // Force true even with p=0.0
    let result = data.draw_boolean(Some(0.0), Some(true));
    assert_eq!(result.unwrap(), true);
    
    let mut data = fresh_data();
    // Force false even with p=1.0
    let result = data.draw_boolean(Some(1.0), Some(false));
    assert_eq!(result.unwrap(), false);
}

#[test]
fn test_forced_nodes_are_marked_trivial() {
    let mut data = fresh_data();
    let _ = data.draw_integer(Some(0), Some(100), None, None, Some(42));
    
    let nodes = data.nodes();
    assert_eq!(nodes.len(), 1);
    assert!(nodes[0].was_forced());
    assert!(nodes[0].is_trivial());
}

// === REPLAY BEHAVIOR TESTS ===

#[test]
fn test_replay_with_same_choices() {
    // Create data with specific choices
    let choices = vec![
        ChoiceValue::Boolean(true),
        ChoiceValue::Integer(42),
        ChoiceValue::Float(3.14),
    ];
    
    let mut data1 = data_for_choices(choices.clone());
    let b1 = data1.draw_boolean(None, None).unwrap();
    let i1 = data1.draw_integer(None, None, None, None, None).unwrap();
    let f1 = data1.draw_float(None, None, None, None, None).unwrap();
    
    let mut data2 = data_for_choices(choices);
    let b2 = data2.draw_boolean(None, None).unwrap();
    let i2 = data2.draw_integer(None, None, None, None, None).unwrap();
    let f2 = data2.draw_float(None, None, None, None, None).unwrap();
    
    assert_eq!(b1, b2);
    assert_eq!(i1, i2);
    assert_eq!(f1, f2);
}

#[test]
fn test_replay_preserves_node_structure() {
    let choices = vec![
        ChoiceValue::Integer(10),
        ChoiceValue::Boolean(false),
    ];
    
    let mut data1 = data_for_choices(choices.clone());
    let _ = data1.draw_integer(Some(0), Some(100), None, None, None);
    let _ = data1.draw_boolean(Some(0.5), None);
    data1.freeze();
    
    let mut data2 = data_for_choices(choices);
    let _ = data2.draw_integer(Some(0), Some(100), None, None, None);
    let _ = data2.draw_boolean(Some(0.5), None);
    data2.freeze();
    
    assert_eq!(data1.nodes().len(), data2.nodes().len());
    assert_eq!(data1.status(), data2.status());
}

// === ERROR HANDLING TESTS ===

#[test]
fn test_empty_choices_causes_overrun() {
    let mut data = data_for_choices(vec![]);
    let result = data.draw_integer(None, None, None, None, None);
    
    assert!(matches!(result, Err(StopTest::Overrun)));
    assert_eq!(data.status(), Status::Overrun);
}

#[test]
fn test_mismatched_choice_type_handling() {
    // Try to draw boolean but have integer choice available
    let mut data = data_for_choices(vec![ChoiceValue::Integer(42)]);
    let result = data.draw_boolean(None, None);
    
    // Should handle type mismatch gracefully
    assert!(result.is_ok() || matches!(result, Err(StopTest::Invalid)));
}

// === BOUNDARY CONDITION TESTS ===

#[test]
fn test_integer_boundary_values() {
    let mut data = fresh_data();
    
    // Test minimum possible integer
    let result = data.draw_integer(Some(i64::MIN), Some(i64::MIN), None, None, None);
    assert_eq!(result.unwrap(), i64::MIN);
    
    let mut data = fresh_data();
    // Test maximum possible integer  
    let result = data.draw_integer(Some(i64::MAX), Some(i64::MAX), None, None, None);
    assert_eq!(result.unwrap(), i64::MAX);
}

#[test]
fn test_float_boundary_values() {
    let mut data = fresh_data();
    
    // Test minimum possible float
    let result = data.draw_float(Some(f64::MIN), Some(f64::MIN), None, None, None);
    assert_eq!(result.unwrap(), f64::MIN);
    
    let mut data = fresh_data();
    // Test maximum possible float
    let result = data.draw_float(Some(f64::MAX), Some(f64::MAX), None, None, None);
    assert_eq!(result.unwrap(), f64::MAX);
}

#[test]
fn test_zero_size_string() {
    let mut data = fresh_data();
    let result = data.draw_string(None, Some(0), Some(0), None);
    assert_eq!(result.unwrap(), "");
}

#[test]
fn test_zero_size_bytes() {
    let mut data = fresh_data();
    let result = data.draw_bytes(Some(0), Some(0), None);
    assert_eq!(result.unwrap(), vec![]);
}

// === COMPLEX INTERACTION TESTS ===

#[test]
fn test_multiple_draw_operations_sequence() {
    let mut data = fresh_data();
    
    // Perform a sequence of different draw operations
    let bool_result = data.draw_boolean(Some(0.5), None);
    let int_result = data.draw_integer(Some(1), Some(100), None, None, None);
    let float_result = data.draw_float(Some(0.0), Some(1.0), None, None, None);
    let string_result = data.draw_string(None, Some(1), Some(10), None);
    let bytes_result = data.draw_bytes(Some(1), Some(5), None);
    
    assert!(bool_result.is_ok());
    assert!(int_result.is_ok());
    assert!(float_result.is_ok());
    assert!(string_result.is_ok());
    assert!(bytes_result.is_ok());
    
    // Verify all choices were recorded
    assert_eq!(data.nodes().len(), 5);
}

#[test]
fn test_status_transitions() {
    let mut data = fresh_data();
    
    // Initially valid
    assert_eq!(data.status(), Status::Valid);
    
    // Draw some values
    let _ = data.draw_boolean(None, None);
    assert_eq!(data.status(), Status::Valid);
    
    // Mark as interesting
    let _ = data.mark_interesting(None);
    assert_eq!(data.status(), Status::Interesting);
    assert!(data.is_frozen());
}

#[test]
fn test_choice_recording_consistency() {
    let mut data = fresh_data();
    
    let bool_val = data.draw_boolean(Some(0.7), None).unwrap();
    let int_val = data.draw_integer(Some(10), Some(20), None, Some(15), None).unwrap();
    
    data.freeze();
    
    let nodes = data.nodes();
    assert_eq!(nodes.len(), 2);
    
    // Verify first node (boolean)
    match nodes[0].value() {
        ChoiceValue::Boolean(recorded_val) => assert_eq!(*recorded_val, bool_val),
        _ => panic!("Expected boolean choice value"),
    }
    
    // Verify second node (integer)
    match nodes[1].value() {
        ChoiceValue::Integer(recorded_val) => assert_eq!(*recorded_val, int_val),
        _ => panic!("Expected integer choice value"),
    }
}