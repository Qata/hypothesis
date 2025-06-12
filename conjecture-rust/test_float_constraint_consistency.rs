//! Standalone test for Float Constraint Type System Consistency capability
//! 
//! This test verifies that the FloatConstraints.smallest_nonzero_magnitude field
//! is consistently typed as f64 (not Option<f64>) throughout the system and that
//! all constraint validation logic works correctly with this type system.

use conjecture::choice::constraints::{FloatConstraints, Constraints};
use conjecture::choice::values::{ChoiceValue, choice_permitted};

fn main() {
    println!("FLOAT_CONSTRAINT_TYPE_SYSTEM: Starting consistency verification tests");
    
    // Test 1: Type consistency - smallest_nonzero_magnitude is f64, not Option<f64>
    test_type_consistency();
    
    // Test 2: Constraint validation logic
    test_constraint_validation_logic();
    
    // Test 3: Edge cases and boundary conditions
    test_edge_cases();
    
    // Test 4: Python parity verification
    test_python_parity();
    
    println!("FLOAT_CONSTRAINT_TYPE_SYSTEM: All tests passed successfully");
}

fn test_type_consistency() {
    println!("FLOAT_CONSTRAINT_TYPE_SYSTEM: Testing core type consistency");
    
    // Test default construction
    let default_constraints = FloatConstraints::default();
    let _magnitude: f64 = default_constraints.smallest_nonzero_magnitude; // Should compile without Option unwrapping
    assert!(default_constraints.smallest_nonzero_magnitude > 0.0);
    
    // Test explicit construction
    let custom_constraints = FloatConstraints {
        min_value: 0.0,
        max_value: 100.0,
        allow_nan: false,
        smallest_nonzero_magnitude: 1e-6, // Direct f64 assignment, no Some() wrapper
    };
    assert_eq!(custom_constraints.smallest_nonzero_magnitude, 1e-6);
    
    // Test advanced constructor
    let advanced_constraints = FloatConstraints::with_smallest_nonzero_magnitude(
        Some(-10.0), Some(10.0), true, 1e-3
    ).expect("Valid constraints should construct successfully");
    assert_eq!(advanced_constraints.smallest_nonzero_magnitude, 1e-3);
    
    println!("FLOAT_CONSTRAINT_TYPE_SYSTEM: Type consistency test passed");
}

fn test_constraint_validation_logic() {
    println!("FLOAT_CONSTRAINT_TYPE_SYSTEM: Testing constraint validation logic");
    
    let constraints = FloatConstraints::with_smallest_nonzero_magnitude(
        Some(-10.0), Some(10.0), false, 1e-3
    ).unwrap();
    
    let rust_constraints = Constraints::Float(constraints.clone());
    
    // Test valid values
    assert!(choice_permitted(&ChoiceValue::Float(5.0), &rust_constraints));
    assert!(choice_permitted(&ChoiceValue::Float(-5.0), &rust_constraints));
    assert!(choice_permitted(&ChoiceValue::Float(0.0), &rust_constraints)); // Zero is always allowed
    assert!(choice_permitted(&ChoiceValue::Float(1e-3), &rust_constraints)); // Exactly at boundary
    assert!(choice_permitted(&ChoiceValue::Float(-1e-3), &rust_constraints)); // Exactly at boundary
    assert!(choice_permitted(&ChoiceValue::Float(1e-2), &rust_constraints)); // Above boundary
    
    // Test invalid values
    assert!(!choice_permitted(&ChoiceValue::Float(f64::NAN), &rust_constraints)); // NaN not allowed
    assert!(!choice_permitted(&ChoiceValue::Float(15.0), &rust_constraints)); // Above max
    assert!(!choice_permitted(&ChoiceValue::Float(-15.0), &rust_constraints)); // Below min
    assert!(!choice_permitted(&ChoiceValue::Float(1e-4), &rust_constraints)); // Too small positive magnitude
    assert!(!choice_permitted(&ChoiceValue::Float(-1e-4), &rust_constraints)); // Too small negative magnitude
    
    // Test with NaN allowed
    let nan_allowed_constraints = FloatConstraints::with_smallest_nonzero_magnitude(
        Some(-10.0), Some(10.0), true, 1e-3
    ).unwrap();
    let rust_nan_constraints = Constraints::Float(nan_allowed_constraints);
    assert!(choice_permitted(&ChoiceValue::Float(f64::NAN), &rust_nan_constraints));
    
    println!("FLOAT_CONSTRAINT_TYPE_SYSTEM: Constraint validation logic test passed");
}

fn test_edge_cases() {
    println!("FLOAT_CONSTRAINT_TYPE_SYSTEM: Testing edge cases");
    
    // Test with infinity bounds
    let inf_constraints = FloatConstraints::with_smallest_nonzero_magnitude(
        None, None, true, f64::MIN_POSITIVE
    ).unwrap();
    let rust_inf_constraints = Constraints::Float(inf_constraints);
    
    assert!(choice_permitted(&ChoiceValue::Float(f64::INFINITY), &rust_inf_constraints));
    assert!(choice_permitted(&ChoiceValue::Float(f64::NEG_INFINITY), &rust_inf_constraints));
    assert!(choice_permitted(&ChoiceValue::Float(f64::MAX), &rust_inf_constraints));
    assert!(choice_permitted(&ChoiceValue::Float(-f64::MAX), &rust_inf_constraints));
    
    // Test clamping functionality
    let clamp_constraints = FloatConstraints::with_smallest_nonzero_magnitude(
        Some(-5.0), Some(5.0), false, 1e-2
    ).unwrap();
    
    // Test clamping to range
    assert_eq!(clamp_constraints.clamp(10.0), 5.0); // Clamp to max
    assert_eq!(clamp_constraints.clamp(-10.0), -5.0); // Clamp to min
    
    // Test clamping small magnitudes
    assert_eq!(clamp_constraints.clamp(1e-3), 1e-2); // Small positive -> smallest positive
    assert_eq!(clamp_constraints.clamp(-1e-3), -1e-2); // Small negative -> smallest negative
    
    // Test values that don't need clamping
    assert_eq!(clamp_constraints.clamp(3.0), 3.0); // Valid value unchanged
    assert_eq!(clamp_constraints.clamp(0.0), 0.0); // Zero unchanged
    
    println!("FLOAT_CONSTRAINT_TYPE_SYSTEM: Edge cases test passed");
}

fn test_python_parity() {
    println!("FLOAT_CONSTRAINT_TYPE_SYSTEM: Testing Python parity");
    
    // Test default constructor matches Python behavior
    let default_constraints = FloatConstraints::default();
    assert_eq!(default_constraints.min_value, f64::NEG_INFINITY);
    assert_eq!(default_constraints.max_value, f64::INFINITY);
    assert_eq!(default_constraints.allow_nan, true);
    assert_eq!(default_constraints.smallest_nonzero_magnitude, f64::MIN_POSITIVE);
    
    // Test new constructor with sensible defaults
    let new_constraints = FloatConstraints::new(Some(-100.0), Some(100.0));
    assert_eq!(new_constraints.min_value, -100.0);
    assert_eq!(new_constraints.max_value, 100.0);
    assert_eq!(new_constraints.allow_nan, true);
    assert_eq!(new_constraints.smallest_nonzero_magnitude, f64::MIN_POSITIVE);
    
    // Test Python's constraint validation behavior
    let python_like = FloatConstraints::with_smallest_nonzero_magnitude(
        Some(-1000.0), Some(1000.0), true, 2.2250738585072014e-308 // Python's SMALLEST_SUBNORMAL
    ).unwrap();
    let rust_python_constraints = Constraints::Float(python_like.clone());
    
    // These should match Python's validation behavior
    assert!(choice_permitted(&ChoiceValue::Float(0.0), &rust_python_constraints));
    assert!(choice_permitted(&ChoiceValue::Float(1.0), &rust_python_constraints));
    assert!(choice_permitted(&ChoiceValue::Float(-1.0), &rust_python_constraints));
    
    // Test validation method directly
    assert!(python_like.validate(0.0));
    assert!(python_like.validate(1.0));
    assert!(python_like.validate(-1.0));
    assert!(python_like.validate(f64::INFINITY));
    assert!(python_like.validate(f64::NEG_INFINITY));
    assert!(python_like.validate(f64::NAN));
    
    println!("FLOAT_CONSTRAINT_TYPE_SYSTEM: Python parity test passed");
}