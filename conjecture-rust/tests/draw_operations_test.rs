//! Test the core draw operations with Python parity
//! 
//! This test verifies that our Rust implementation provides the same
//! functionality as the Python ConjectureData draw operations.

use conjecture::data::ConjectureData;
use conjecture::choice::IntervalSet;
use std::collections::HashMap;

#[test]
fn test_draw_integer_basic() {
    let mut data = ConjectureData::new(1000);
    
    // Test basic integer drawing
    let result = data.draw_integer(Some(0), Some(10), None, 0, None, true);
    assert!(result.is_ok());
    let value = result.unwrap();
    assert!(value >= 0 && value <= 10);
}

#[test]
fn test_draw_integer_with_weights() {
    let mut data = ConjectureData::new(1000);
    
    // Test weighted integer drawing
    let mut weights = HashMap::new();
    weights.insert(5, 0.5); // 50% chance for value 5
    
    let result = data.draw_integer(Some(0), Some(10), Some(weights), 0, None, true);
    assert!(result.is_ok());
    let value = result.unwrap();
    assert!(value >= 0 && value <= 10);
}

#[test]
fn test_draw_integer_forced() {
    let mut data = ConjectureData::new(1000);
    
    // Test forced integer value
    let result = data.draw_integer(Some(0), Some(10), None, 0, Some(7), true);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 7);
}

#[test]
fn test_draw_boolean_basic() {
    let mut data = ConjectureData::new(1000);
    
    // Test basic boolean drawing
    let result = data.draw_boolean(0.5, None, true);
    assert!(result.is_ok());
    let _value = result.unwrap(); // Either true or false, both valid
}

#[test]
fn test_draw_boolean_deterministic() {
    let mut data = ConjectureData::new(1000);
    
    // Test deterministic cases
    let result_false = data.draw_boolean(0.0, None, true);
    assert!(result_false.is_ok());
    assert_eq!(result_false.unwrap(), false);
    
    let result_true = data.draw_boolean(1.0, None, true);
    assert!(result_true.is_ok());
    assert_eq!(result_true.unwrap(), true);
}

#[test]
fn test_draw_float_basic() {
    let mut data = ConjectureData::new(1000);
    
    // Test basic float drawing
    let result = data.draw_float(0.0, 1.0, false, None, None, true);
    assert!(result.is_ok());
    let value = result.unwrap();
    assert!(value >= 0.0 && value <= 1.0);
    assert!(!value.is_nan()); // NaN not allowed
}

#[test]
fn test_draw_float_with_nan() {
    let mut data = ConjectureData::new(1000);
    
    // Test float drawing with NaN allowed
    let result = data.draw_float(f64::NEG_INFINITY, f64::INFINITY, true, None, None, true);
    assert!(result.is_ok());
    // Value could be anything including NaN, just check it doesn't error
}

#[test]
fn test_draw_string_basic() {
    let mut data = ConjectureData::new(1000);
    
    // Test basic string drawing
    let intervals = IntervalSet::from_string("abc");
    let result = data.draw_string(intervals, 0, 10, None, true);
    assert!(result.is_ok());
    let value = result.unwrap();
    assert!(value.len() <= 10);
    // Check all characters are from the alphabet
    for ch in value.chars() {
        assert!("abc".contains(ch));
    }
}

#[test]
fn test_draw_bytes_basic() {
    let mut data = ConjectureData::new(1000);
    
    // Test basic bytes drawing
    let result = data.draw_bytes(0, 10, None, true);
    assert!(result.is_ok());
    let value = result.unwrap();
    assert!(value.len() <= 10);
}

#[test]
fn test_constraint_validation() {
    let mut data = ConjectureData::new(1000);
    
    // Test invalid range should fail
    let result = data.draw_integer(Some(10), Some(5), None, 0, None, true);
    assert!(result.is_err());
    
    // Test invalid probability should fail
    let result = data.draw_boolean(1.5, None, true);
    assert!(result.is_err());
}

#[test]
fn test_choice_recording() {
    let mut data = ConjectureData::new(1000);
    
    // Initially no choices - check length field instead
    assert_eq!(data.length, 0);
    
    // Draw some values
    let _ = data.draw_integer(Some(0), Some(10), None, 0, None, true);
    assert!(data.length >= 1); // Length increases with draws
    
    let _ = data.draw_boolean(0.5, None, true);
    assert!(data.length >= 2);
    
    let _ = data.draw_float(0.0, 1.0, false, None, None, true);
    assert!(data.length >= 3);
}

#[test]
fn test_legacy_compatibility() {
    let mut data = ConjectureData::new(1000);
    
    // Test legacy methods still work
    let result = data.draw_integer_simple(0, 10);
    assert!(result.is_ok());
    
    let result = data.draw_float_simple();
    assert!(result.is_ok());
    
    let result = data.draw_string_simple("abc", 0, 5);
    assert!(result.is_ok());
    
    let result = data.draw_bytes_simple(5);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 5);
}