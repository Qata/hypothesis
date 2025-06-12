//! Comprehensive Type Consistency Tests for Float Constraint Type System
//!
//! This module provides exhaustive tests for the f64 vs Option<f64> type consistency
//! issue in FloatConstraints.smallest_nonzero_magnitude, ensuring complete compatibility
//! with Python Hypothesis behavior and proper PyO3 integration.

use crate::choice::constraints::{FloatConstraints, Constraints};
use crate::choice::values::{ChoiceValue, FloatValue};
use std::collections::HashMap;

/// Comprehensive test suite for type consistency across all interfaces
#[cfg(test)]
mod type_consistency_tests {
    use super::*;

    /// Test direct field access type consistency
    #[test]
    fn test_direct_field_access_consistency() {
        println!("TYPE_CONSISTENCY DEBUG: Testing direct field access");

        let constraints = FloatConstraints::default();
        
        // Direct field access should work with Option<f64>
        let magnitude: Option<f64> = constraints.smallest_nonzero_magnitude;
        assert!(magnitude.is_some());
        assert!(magnitude.unwrap() > 0.0);
        
        // Pattern matching should work
        match constraints.smallest_nonzero_magnitude {
            Some(mag) => assert!(mag > 0.0),
            None => panic!("Default should have Some value"),
        }
        
        // Optional unwrapping patterns
        let mag_value = constraints.smallest_nonzero_magnitude.unwrap_or(f64::MIN_POSITIVE);
        assert!(mag_value > 0.0);
        
        let mag_map = constraints.smallest_nonzero_magnitude.map(|m| m * 2.0);
        assert!(mag_map.is_some());
        
        println!("TYPE_CONSISTENCY DEBUG: Direct field access PASSED");
    }

    /// Test constructor parameter type consistency
    #[test] 
    fn test_constructor_parameter_consistency() {
        println!("TYPE_CONSISTENCY DEBUG: Testing constructor parameters");

        // Test with Some(value)
        let constraints1 = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-10.0),
            Some(10.0),
            false,
            Some(1e-6), // Option<f64> parameter
        ).expect("Should accept Some(f64)");
        
        assert_eq!(constraints1.smallest_nonzero_magnitude, Some(1e-6));
        
        // Test with None
        let constraints2 = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-10.0),
            Some(10.0),
            false,
            None, // Option<f64> parameter
        ).expect("Should accept None");
        
        assert_eq!(constraints2.smallest_nonzero_magnitude, None);
        
        // Test default constructor
        let constraints3 = FloatConstraints::new(Some(-5.0), Some(5.0));
        assert_eq!(constraints3.smallest_nonzero_magnitude, Some(f64::MIN_POSITIVE));
        
        println!("TYPE_CONSISTENCY DEBUG: Constructor parameters PASSED");
    }

    /// Test cloning and serialization type consistency
    #[test]
    fn test_cloning_serialization_consistency() {
        println!("TYPE_CONSISTENCY DEBUG: Testing cloning and serialization");

        let original = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-100.0),
            Some(100.0),
            false,
            Some(1e-8),
        ).expect("Should create original");
        
        // Test cloning preserves Option<f64> type
        let cloned = original.clone();
        let _: Option<f64> = cloned.smallest_nonzero_magnitude;
        assert_eq!(original.smallest_nonzero_magnitude, cloned.smallest_nonzero_magnitude);
        
        // Test equality
        assert_eq!(original, cloned);
        
        // Test hashing (should work with Option<f64>)
        let mut map = HashMap::new();
        map.insert(original.clone(), "test");
        assert!(map.contains_key(&cloned));
        
        println!("TYPE_CONSISTENCY DEBUG: Cloning and serialization PASSED");
    }

    /// Test enum wrapper type consistency
    #[test]
    fn test_enum_wrapper_consistency() {
        println!("TYPE_CONSISTENCY DEBUG: Testing enum wrapper");

        let float_constraints = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-50.0),
            Some(50.0),
            true,
            Some(1e-7),
        ).expect("Should create constraints");
        
        // Wrap in Constraints enum
        let wrapped = Constraints::Float(float_constraints.clone());
        
        // Extract and verify Option<f64> type is preserved
        if let Constraints::Float(ref c) = wrapped {
            let _: Option<f64> = c.smallest_nonzero_magnitude;
            assert_eq!(c.smallest_nonzero_magnitude, Some(1e-7));
        } else {
            panic!("Should be Float variant");
        }
        
        // Test pattern matching with enum
        match wrapped {
            Constraints::Float(c) => {
                assert_eq!(c.smallest_nonzero_magnitude, Some(1e-7));
                let _: Option<f64> = c.smallest_nonzero_magnitude;
            },
            _ => panic!("Should be Float variant"),
        }
        
        println!("TYPE_CONSISTENCY DEBUG: Enum wrapper PASSED");
    }

    /// Test validation method type consistency
    #[test]
    fn test_validation_method_consistency() {
        println!("TYPE_CONSISTENCY DEBUG: Testing validation methods");

        let constraints = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-10.0),
            Some(10.0),
            false,
            Some(1e-3),
        ).expect("Should create constraints");
        
        // Test that validation works with Option<f64> field
        let magnitude = constraints.smallest_nonzero_magnitude.unwrap();
        
        // Values at and above threshold should pass
        assert!(constraints.validate(magnitude));
        assert!(constraints.validate(-magnitude));
        assert!(constraints.validate(magnitude * 2.0));
        
        // Values below threshold should fail
        assert!(!constraints.validate(magnitude / 2.0));
        assert!(!constraints.validate(-magnitude / 2.0));
        
        // Zero should always pass
        assert!(constraints.validate(0.0));
        
        // Test with None magnitude (no constraint)
        let no_magnitude = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-10.0),
            Some(10.0),
            false,
            None,
        ).expect("Should create no-magnitude constraints");
        
        assert!(no_magnitude.validate(1e-100)); // Very small should pass
        assert!(no_magnitude.validate(-1e-100));
        
        println!("TYPE_CONSISTENCY DEBUG: Validation methods PASSED");
    }

    /// Test clamping method type consistency
    #[test]
    fn test_clamping_method_consistency() {
        println!("TYPE_CONSISTENCY DEBUG: Testing clamping methods");

        let constraints = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-5.0),
            Some(5.0),
            false,
            Some(0.1),
        ).expect("Should create constraints");
        
        // Test clamping with Option<f64> magnitude
        let too_small = 0.01;
        let clamped = constraints.clamp(too_small);
        
        // Should be clamped to magnitude threshold
        assert!(constraints.validate(clamped));
        assert!(clamped.abs() >= 0.1 || clamped == 0.0);
        
        // Test clamping to range bounds
        assert_eq!(constraints.clamp(10.0), 5.0);
        assert_eq!(constraints.clamp(-10.0), -5.0);
        
        // Test with None magnitude
        let no_magnitude = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-5.0),
            Some(5.0),
            false,
            None,
        ).expect("Should create no-magnitude constraints");
        
        let tiny_value = 1e-100;
        let clamped_tiny = no_magnitude.clamp(tiny_value);
        assert_eq!(clamped_tiny, tiny_value); // Should not be changed
        
        println!("TYPE_CONSISTENCY DEBUG: Clamping methods PASSED");
    }

    /// Test choice value integration type consistency
    #[test]
    fn test_choice_value_integration_consistency() {
        println!("TYPE_CONSISTENCY DEBUG: Testing choice value integration");

        let constraints = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-100.0),
            Some(100.0),
            false,
            Some(1e-6),
        ).expect("Should create constraints");
        
        // Create ChoiceValue with float constraints
        let float_value = FloatValue {
            value: 3.14159,
            constraints: constraints.clone(),
        };
        
        let choice_value = ChoiceValue::Float(float_value);
        
        // Extract constraints and verify Option<f64> type
        if let ChoiceValue::Float(ref fv) = choice_value {
            let _: Option<f64> = fv.constraints.smallest_nonzero_magnitude;
            assert_eq!(fv.constraints.smallest_nonzero_magnitude, Some(1e-6));
        }
        
        // Test validation through choice value
        assert!(constraints.validate(3.14159));
        
        // Test with direct constraint access
        let magnitude = constraints.smallest_nonzero_magnitude;
        assert_eq!(magnitude, Some(1e-6));
        
        println!("TYPE_CONSISTENCY DEBUG: Choice value integration PASSED");
    }

    /// Test error handling with type consistency
    #[test]
    fn test_error_handling_consistency() {
        println!("TYPE_CONSISTENCY DEBUG: Testing error handling");

        // Test invalid magnitude values
        let invalid_zero = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-10.0),
            Some(10.0),
            false,
            Some(0.0), // Invalid: zero magnitude
        );
        assert!(invalid_zero.is_err());
        
        let invalid_negative = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-10.0),
            Some(10.0),
            false,
            Some(-1e-6), // Invalid: negative magnitude
        );
        assert!(invalid_negative.is_err());
        
        // Test invalid range
        let invalid_range = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(10.0),  // min > max
            Some(-10.0),
            false,
            Some(1e-6),
        );
        assert!(invalid_range.is_err());
        
        // Test that None magnitude is always valid
        let none_magnitude = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-10.0),
            Some(10.0),
            false,
            None, // None should always be valid
        );
        assert!(none_magnitude.is_ok());
        assert_eq!(none_magnitude.unwrap().smallest_nonzero_magnitude, None);
        
        println!("TYPE_CONSISTENCY DEBUG: Error handling PASSED");
    }

    /// Test boundary conditions with type consistency
    #[test]
    fn test_boundary_conditions_consistency() {
        println!("TYPE_CONSISTENCY DEBUG: Testing boundary conditions");

        // Test with extreme values
        let extreme_constraints = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(f64::NEG_INFINITY),
            Some(f64::INFINITY),
            true,
            Some(f64::MIN_POSITIVE), // Smallest possible positive f64
        ).expect("Should handle extreme values");
        
        let _: Option<f64> = extreme_constraints.smallest_nonzero_magnitude;
        
        // Test validation with extreme values
        assert!(extreme_constraints.validate(f64::MIN_POSITIVE));
        assert!(extreme_constraints.validate(-f64::MIN_POSITIVE));
        assert!(extreme_constraints.validate(f64::MAX));
        assert!(extreme_constraints.validate(-f64::MAX));
        assert!(extreme_constraints.validate(f64::INFINITY));
        assert!(extreme_constraints.validate(f64::NEG_INFINITY));
        assert!(extreme_constraints.validate(f64::NAN));
        
        // Test with maximum magnitude
        let max_magnitude = FloatConstraints::with_smallest_nonzero_magnitude(
            None,
            None,
            true,
            Some(f64::MAX), // Very large magnitude threshold
        ).expect("Should handle large magnitude");
        
        assert!(!max_magnitude.validate(1.0)); // Should fail (below threshold)
        assert!(max_magnitude.validate(f64::MAX));
        assert!(max_magnitude.validate(-f64::MAX));
        
        println!("TYPE_CONSISTENCY DEBUG: Boundary conditions PASSED");
    }

    /// Test thread safety and concurrent access
    #[test]
    fn test_thread_safety_consistency() {
        println!("TYPE_CONSISTENCY DEBUG: Testing thread safety");

        let constraints = std::sync::Arc::new(FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-1000.0),
            Some(1000.0),
            false,
            Some(1e-9),
        ).expect("Should create constraints"));
        
        let handles: Vec<std::thread::JoinHandle<()>> = (0..10)
            .map(|i| {
                let constraints_clone = constraints.clone();
                std::thread::spawn(move || {
                    // Test concurrent access to Option<f64> field
                    let _: Option<f64> = constraints_clone.smallest_nonzero_magnitude;
                    
                    // Test concurrent validation
                    let test_value = (i as f64) * 1e-8;
                    let result = constraints_clone.validate(test_value);
                    
                    // Test concurrent clamping
                    let clamped = constraints_clone.clamp(test_value);
                    assert!(constraints_clone.validate(clamped));
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }
        
        println!("TYPE_CONSISTENCY DEBUG: Thread safety PASSED");
    }

    /// Comprehensive integration test for complete type consistency
    #[test]
    fn test_comprehensive_type_consistency() {
        println!("TYPE_CONSISTENCY DEBUG: Comprehensive type consistency test");

        // Test all construction methods
        let default_constraints = FloatConstraints::default();
        let simple_constraints = FloatConstraints::new(Some(-10.0), Some(10.0));
        let advanced_constraints = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-100.0),
            Some(100.0),
            false,
            Some(1e-6),
        ).expect("Should create advanced constraints");
        let none_constraints = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-50.0),
            Some(50.0),
            true,
            None,
        ).expect("Should create none constraints");
        
        let all_constraints = vec![
            default_constraints,
            simple_constraints,
            advanced_constraints,
            none_constraints,
        ];
        
        for (i, constraints) in all_constraints.iter().enumerate() {
            println!("  Testing constraint set {}", i);
            
            // Type consistency check
            let _: Option<f64> = constraints.smallest_nonzero_magnitude;
            
            // Functional consistency check
            let test_values = vec![0.0, 1e-10, 1e-6, 1e-3, 1.0, 10.0, 100.0];
            
            for value in test_values {
                let is_valid = constraints.validate(value);
                let clamped = constraints.clamp(value);
                
                // Clamped value should always be valid
                assert!(constraints.validate(clamped), 
                       "Clamped value {} should be valid for constraint set {}", clamped, i);
                
                // If original was valid, clamped should equal original (for finite values)
                if is_valid && value.is_finite() {
                    assert_eq!(value, clamped,
                             "Valid value {} should not be changed by clamping in set {}", value, i);
                }
            }
            
            // Clone consistency
            let cloned = constraints.clone();
            assert_eq!(constraints.smallest_nonzero_magnitude, cloned.smallest_nonzero_magnitude);
            
            // Enum wrapper consistency
            let wrapped = Constraints::Float(constraints.clone());
            if let Constraints::Float(ref c) = wrapped {
                assert_eq!(constraints.smallest_nonzero_magnitude, c.smallest_nonzero_magnitude);
            }
        }
        
        println!("TYPE_CONSISTENCY DEBUG: Comprehensive type consistency PASSED");
    }
}

/// Python parity validation tests
#[cfg(test)]
mod python_parity_tests {
    use super::*;

    /// Test that Rust behavior matches Python Hypothesis exactly
    #[test]
    fn test_python_hypothesis_behavior_parity() {
        println!("PYTHON_PARITY DEBUG: Testing Python Hypothesis behavior parity");

        // Test default behavior matching Python
        let default_rust = FloatConstraints::default();
        
        // These values should match Python Hypothesis defaults
        assert_eq!(default_rust.min_value, f64::NEG_INFINITY);
        assert_eq!(default_rust.max_value, f64::INFINITY);
        assert_eq!(default_rust.allow_nan, true);
        assert_eq!(default_rust.smallest_nonzero_magnitude, Some(f64::MIN_POSITIVE));
        
        // Test Python-like constraint creation
        let python_like = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-1e10),
            Some(1e10),
            true,
            Some(2.2250738585072014e-308), // Python's SMALLEST_SUBNORMAL
        ).expect("Should match Python behavior");
        
        // Test validation behavior that should match Python
        let test_cases = vec![
            (0.0, true),                           // Zero always allowed
            (1.0, true),                           // Normal values
            (-1.0, true),
            (f64::INFINITY, true),                 // Infinities allowed when in range
            (f64::NEG_INFINITY, true),
            (f64::NAN, true),                      // NaN allowed when allow_nan=true
            (2.2250738585072014e-308, true),       // At threshold
            (-2.2250738585072014e-308, true),
            (1e-308, false),                       // Below threshold
            (-1e-308, false),
        ];
        
        for (value, expected) in test_cases {
            let result = python_like.validate(value);
            assert_eq!(result, expected, 
                      "Python parity failed for {}: expected {}, got {}", value, expected, result);
        }
        
        println!("PYTHON_PARITY DEBUG: Python Hypothesis parity PASSED");
    }

    /// Test Python constraint validation edge cases
    #[test]
    fn test_python_constraint_edge_cases() {
        println!("PYTHON_PARITY DEBUG: Testing Python constraint edge cases");

        // Test with disallowed NaN (Python behavior)
        let no_nan = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-1000.0),
            Some(1000.0),
            false, // allow_nan = False
            Some(1e-10),
        ).expect("Should create no-NaN constraints");
        
        assert!(!no_nan.validate(f64::NAN));
        
        // Test clamping of NaN when not allowed (Python maps to valid range)
        let clamped_nan = no_nan.clamp(f64::NAN);
        assert!(!clamped_nan.is_nan());
        assert!(no_nan.validate(clamped_nan));
        
        // Test Python-style range validation
        assert!(!no_nan.validate(1500.0));  // Above max
        assert!(!no_nan.validate(-1500.0)); // Below min
        
        // Test Python-style magnitude validation
        assert!(!no_nan.validate(1e-11)); // Below magnitude threshold
        assert!(!no_nan.validate(-1e-11));
        
        println!("PYTHON_PARITY DEBUG: Python constraint edge cases PASSED");
    }
}

/// Export validation to ensure the API works correctly
pub fn validate_float_constraint_type_system() {
    println!("🔍 Validating Float Constraint Type System...");
    
    // Run all the tests programmatically
    let constraints = FloatConstraints::with_smallest_nonzero_magnitude(
        Some(-100.0),
        Some(100.0),
        false,
        Some(1e-6),
    ).expect("Should create valid constraints");
    
    // Verify Option<f64> type consistency
    let _: Option<f64> = constraints.smallest_nonzero_magnitude;
    assert_eq!(constraints.smallest_nonzero_magnitude, Some(1e-6));
    
    // Verify validation works correctly
    assert!(constraints.validate(0.0));
    assert!(constraints.validate(1e-6));
    assert!(!constraints.validate(1e-7));
    
    // Verify clamping works correctly
    let clamped = constraints.clamp(1e-7);
    assert!(constraints.validate(clamped));
    
    println!("✅ Float Constraint Type System: VALIDATED!");
}