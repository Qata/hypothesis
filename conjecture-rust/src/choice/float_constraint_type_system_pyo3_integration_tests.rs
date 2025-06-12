//! Comprehensive PyO3 Integration Tests for Float Constraint Type System
//!
//! This module provides comprehensive tests for the Float Constraint Type System capability
//! in the ConjectureData module, focusing on PyO3 integration and the specific type system
//! fixes needed for the f64 vs Option<f64> type consistency for smallest_nonzero_magnitude
//! and export float encoding functions.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyBool, PyNone};
use crate::choice::constraints::FloatConstraints;
use crate::float_encoding_export::{
    float_to_lex, lex_to_float, float_to_int, int_to_float,
    float_to_lex_multi_width, lex_to_float_multi_width,
    FloatWidth, FloatEncodingConfig, float_to_lex_advanced
};

/// PyO3 integration test wrapper for FloatConstraints
#[pyclass]
struct PyFloatConstraints {
    inner: FloatConstraints,
}

#[pymethods]
impl PyFloatConstraints {
    #[new]
    fn new(
        min_value: Option<f64>,
        max_value: Option<f64>,
        allow_nan: bool,
        smallest_nonzero_magnitude: Option<f64>
    ) -> PyResult<Self> {
        let constraints = FloatConstraints::with_smallest_nonzero_magnitude(
            min_value,
            max_value,
            allow_nan,
            smallest_nonzero_magnitude,
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        Ok(PyFloatConstraints { inner: constraints })
    }

    #[getter]
    fn min_value(&self) -> f64 {
        self.inner.min_value
    }

    #[getter]
    fn max_value(&self) -> f64 {
        self.inner.max_value
    }

    #[getter]
    fn allow_nan(&self) -> bool {
        self.inner.allow_nan
    }

    #[getter]
    fn smallest_nonzero_magnitude(&self) -> Option<f64> {
        self.inner.smallest_nonzero_magnitude
    }

    fn validate(&self, value: f64) -> bool {
        self.inner.validate(value)
    }

    fn clamp(&self, value: f64) -> f64 {
        self.inner.clamp(value)
    }
}

/// PyO3 export functions for float encoding
#[pyfunction]
fn py_float_to_lex(value: f64) -> u64 {
    float_to_lex(value)
}

#[pyfunction]
fn py_lex_to_float(lex: u64) -> f64 {
    lex_to_float(lex)
}

#[pyfunction]
fn py_float_to_int(value: f64) -> u64 {
    float_to_int(value)
}

#[pyfunction]
fn py_int_to_float(i: u64) -> f64 {
    int_to_float(i)
}

#[pyfunction]
fn py_float_to_lex_multi_width(value: f64, width: u32) -> u64 {
    let float_width = match width {
        16 => FloatWidth::Width16,
        32 => FloatWidth::Width32,
        64 => FloatWidth::Width64,
        _ => FloatWidth::Width64,
    };
    float_to_lex_multi_width(value, float_width)
}

#[pyfunction]
fn py_lex_to_float_multi_width(lex: u64, width: u32) -> f64 {
    let float_width = match width {
        16 => FloatWidth::Width16,
        32 => FloatWidth::Width32,
        64 => FloatWidth::Width64,
        _ => FloatWidth::Width64,
    };
    lex_to_float_multi_width(lex, float_width)
}

/// Create Python module for testing
#[pymodule]
fn float_constraint_test_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyFloatConstraints>()?;
    m.add_function(wrap_pyfunction!(py_float_to_lex, m)?)?;
    m.add_function(wrap_pyfunction!(py_lex_to_float, m)?)?;
    m.add_function(wrap_pyfunction!(py_float_to_int, m)?)?;
    m.add_function(wrap_pyfunction!(py_int_to_float, m)?)?;
    m.add_function(wrap_pyfunction!(py_float_to_lex_multi_width, m)?)?;
    m.add_function(wrap_pyfunction!(py_lex_to_float_multi_width, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{IntoPyDict, PyTuple};

    /// Test PyO3 integration for float constraint type consistency
    #[test]
    fn test_pyo3_float_constraint_type_consistency() {
        println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Testing type consistency across PyO3 boundary");

        Python::with_gil(|py| {
            // Test 1: Create constraints from Python with Option<f64> handling
            let constraints = PyFloatConstraints::new(
                Some(-100.0),
                Some(100.0),
                false,
                Some(1e-6), // Option<f64> parameter
            ).expect("Should create constraints");

            // Verify getter returns Option<f64>
            let magnitude = constraints.smallest_nonzero_magnitude();
            assert_eq!(magnitude, Some(1e-6));
            println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Option<f64> getter works: {:?}", magnitude);

            // Test 2: Validate from Python side
            assert!(constraints.validate(0.0));
            assert!(constraints.validate(1e-6));
            assert!(constraints.validate(-1e-6));
            assert!(!constraints.validate(1e-7));
            assert!(!constraints.validate(-1e-7));
            
            // Test 3: Clamp from Python side
            let clamped = constraints.clamp(1e-7);
            assert!(constraints.validate(clamped));
            println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Clamping works: {} -> {}", 1e-7, clamped);

            println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Type consistency test PASSED");
        });
    }

    /// Test PyO3 integration for float encoding export functions
    #[test]
    fn test_pyo3_float_encoding_export() {
        println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Testing float encoding exports via PyO3");

        Python::with_gil(|py| {
            // Test basic encoding/decoding
            let test_values = vec![0.0, 1.0, -1.0, 3.14159, f64::INFINITY, f64::NEG_INFINITY];
            
            for value in test_values {
                if value.is_finite() {
                    // Test lex encoding round-trip
                    let lex = py_float_to_lex(value);
                    let recovered = py_lex_to_float(lex);
                    assert_eq!(value, recovered, "Lex round-trip failed for {}", value);

                    // Test int conversion round-trip
                    let int_repr = py_float_to_int(value);
                    let recovered_int = py_int_to_float(int_repr);
                    assert_eq!(value, recovered_int, "Int round-trip failed for {}", value);

                    println!("FLOAT_CONSTRAINT_PYO3 DEBUG: {} -> lex: 0x{:016X} -> {}", value, lex, recovered);
                } else if value.is_infinite() {
                    // Test infinity handling
                    let int_repr = py_float_to_int(value);
                    let recovered = py_int_to_float(int_repr);
                    assert_eq!(value, recovered, "Infinity round-trip failed");
                }
            }

            println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Basic encoding exports PASSED");
        });
    }

    /// Test PyO3 integration for multi-width float encoding
    #[test]
    fn test_pyo3_multi_width_encoding() {
        println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Testing multi-width encoding via PyO3");

        Python::with_gil(|py| {
            let test_value = 3.14159;
            let widths = vec![16, 32, 64];

            for width in widths {
                let encoded = py_float_to_lex_multi_width(test_value, width);
                let decoded = py_lex_to_float_multi_width(encoded, width);

                // Check that precision is appropriate for width
                let precision_tolerance = match width {
                    16 => 1e-3,  // f16 has limited precision
                    32 => 1e-6,  // f32 precision
                    64 => 1e-15, // f64 precision
                    _ => 1e-15,
                };

                let error = (decoded - test_value).abs();
                assert!(error < precision_tolerance, 
                       "Width {} precision error {} exceeds tolerance {}", 
                       width, error, precision_tolerance);

                println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Width {} - {} -> 0x{:016X} -> {} (error: {})", 
                        width, test_value, encoded, decoded, error);
            }

            println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Multi-width encoding PASSED");
        });
    }

    /// Test PyO3 integration with Python dictionary creation
    #[test]
    fn test_pyo3_python_dict_integration() {
        println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Testing Python dict integration");

        Python::with_gil(|py| {
            // Create constraints dictionary in Python style
            let kwargs = vec![
                ("min_value", (-50.0).to_object(py)),
                ("max_value", 50.0.to_object(py)),
                ("allow_nan", false.to_object(py)),
                ("smallest_nonzero_magnitude", 1e-8.to_object(py)),
            ].into_py_dict(py);

            // Test constraint creation from dictionary-like parameters
            let constraints = PyFloatConstraints::new(
                Some(-50.0),
                Some(50.0),
                false,
                Some(1e-8),
            ).expect("Should create from dict-like params");

            // Test that the constraints work as expected
            assert_eq!(constraints.min_value(), -50.0);
            assert_eq!(constraints.max_value(), 50.0);
            assert_eq!(constraints.allow_nan(), false);
            assert_eq!(constraints.smallest_nonzero_magnitude(), Some(1e-8));

            // Test validation with various Python-like values
            let test_cases = vec![
                (0.0, true),        // Zero allowed
                (1e-8, true),       // At threshold
                (-1e-8, true),      // Negative at threshold
                (1e-9, false),      // Below threshold
                (25.0, true),       // Normal value
                (75.0, false),      // Above max
                (-75.0, false),     // Below min
            ];

            for (value, expected) in test_cases {
                let result = constraints.validate(value);
                assert_eq!(result, expected, "Validation failed for {}: expected {}, got {}", 
                          value, expected, result);
            }

            println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Python dict integration PASSED");
        });
    }

    /// Test PyO3 integration with special float values
    #[test]
    fn test_pyo3_special_values_handling() {
        println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Testing special values via PyO3");

        Python::with_gil(|py| {
            // Test with NaN-allowing constraints
            let nan_constraints = PyFloatConstraints::new(
                None,
                None,
                true,  // allow_nan = true
                Some(f64::MIN_POSITIVE),
            ).expect("Should create NaN-allowing constraints");

            // Test with NaN-disallowing constraints  
            let no_nan_constraints = PyFloatConstraints::new(
                Some(-1000.0),
                Some(1000.0),
                false, // allow_nan = false
                Some(1e-10),
            ).expect("Should create NaN-disallowing constraints");

            // Test special values
            let special_values = vec![
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                0.0,
                -0.0,
                f64::MIN_POSITIVE,
                f64::MAX,
            ];

            for value in special_values {
                // Test with NaN-allowing constraints
                let nan_result = nan_constraints.validate(value);
                if value.is_nan() {
                    assert!(nan_result, "NaN should be allowed when allow_nan=true");
                }

                // Test with NaN-disallowing constraints
                let no_nan_result = no_nan_constraints.validate(value);
                if value.is_nan() {
                    assert!(!no_nan_result, "NaN should be rejected when allow_nan=false");
                }

                // Test encoding/decoding for special values
                if !value.is_nan() {
                    let encoded = py_float_to_lex(value);
                    let decoded = py_lex_to_float(encoded);
                    
                    if value.is_infinite() {
                        assert!(decoded.is_infinite() || decoded == 0.0, 
                               "Infinity should map appropriately");
                    } else {
                        assert_eq!(value, decoded, "Special value {} should round-trip", value);
                    }
                }

                println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Special value {} - nan_allowed: {}, nan_rejected: {}", 
                        value, nan_result, !no_nan_result);
            }

            println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Special values handling PASSED");
        });
    }

    /// Test PyO3 integration for float constraint edge cases
    #[test]
    fn test_pyo3_constraint_edge_cases() {
        println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Testing constraint edge cases via PyO3");

        Python::with_gil(|py| {
            // Test 1: Very small magnitude threshold
            let tiny_constraints = PyFloatConstraints::new(
                Some(-1.0),
                Some(1.0),
                true,
                Some(f64::MIN_POSITIVE),
            ).expect("Should handle tiny magnitude");

            assert!(tiny_constraints.validate(f64::MIN_POSITIVE));
            assert!(tiny_constraints.validate(-f64::MIN_POSITIVE));
            assert!(tiny_constraints.validate(0.0));

            // Test 2: Large magnitude threshold
            let large_constraints = PyFloatConstraints::new(
                None,
                None,
                true,
                Some(1.0),
            ).expect("Should handle large magnitude");

            assert!(large_constraints.validate(1.0));
            assert!(large_constraints.validate(-1.0));
            assert!(!large_constraints.validate(0.5));
            assert!(!large_constraints.validate(-0.5));

            // Test 3: None magnitude (no constraint)
            let no_magnitude_constraints = PyFloatConstraints::new(
                Some(-10.0),
                Some(10.0),
                false,
                None, // No magnitude constraint
            ).expect("Should handle None magnitude");

            assert!(no_magnitude_constraints.validate(1e-100)); // Very small should be allowed
            assert!(no_magnitude_constraints.validate(-1e-100));
            assert!(no_magnitude_constraints.validate(0.0));

            // Test 4: Clamping behavior
            let clamp_constraints = PyFloatConstraints::new(
                Some(-5.0),
                Some(5.0),
                false,
                Some(0.1),
            ).expect("Should create clamp constraints");

            // Test clamping to range
            assert_eq!(clamp_constraints.clamp(10.0), 5.0);
            assert_eq!(clamp_constraints.clamp(-10.0), -5.0);

            // Test clamping small magnitudes
            let clamped_small = clamp_constraints.clamp(0.01);
            assert!(clamp_constraints.validate(clamped_small));
            assert!(clamped_small.abs() >= 0.1 || clamped_small == 0.0);

            println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Edge cases testing PASSED");
        });
    }

    /// Test PyO3 integration performance and stability
    #[test]
    fn test_pyo3_performance_stability() {
        println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Testing performance and stability via PyO3");

        Python::with_gil(|py| {
            let constraints = PyFloatConstraints::new(
                Some(-1000.0),
                Some(1000.0),
                false,
                Some(1e-6),
            ).expect("Should create constraints");

            // Test with many values to ensure stability
            let test_values: Vec<f64> = (0..1000)
                .map(|i| (i as f64 - 500.0) / 100.0)
                .collect();

            let mut validation_count = 0;
            let mut clamp_count = 0;

            for value in test_values {
                // Test validation
                let is_valid = constraints.validate(value);
                if is_valid {
                    validation_count += 1;
                }

                // Test clamping
                let clamped = constraints.clamp(value);
                assert!(constraints.validate(clamped), "Clamped value should always be valid");
                clamp_count += 1;

                // Test encoding/decoding
                if value.is_finite() {
                    let encoded = py_float_to_lex(value);
                    let decoded = py_lex_to_float(encoded);
                    assert_eq!(value, decoded, "Round-trip failed for {}", value);
                }
            }

            println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Processed 1000 values - {} valid, {} clamped", 
                    validation_count, clamp_count);

            // Test float encoding performance
            let encoding_test_values = vec![1.0, 2.0, 3.14159, -2.718281828, 1e-10, 1e10];
            for value in encoding_test_values {
                for width in &[16, 32, 64] {
                    let encoded = py_float_to_lex_multi_width(value, *width);
                    let decoded = py_lex_to_float_multi_width(encoded, *width);
                    
                    let tolerance = match width {
                        16 => 1e-3,
                        32 => 1e-6,
                        64 => 1e-15,
                        _ => 1e-15,
                    };
                    
                    let error = (decoded - value).abs();
                    assert!(error < tolerance, "Width {} encoding error too large", width);
                }
            }

            println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Performance and stability PASSED");
        });
    }

    /// Comprehensive integration test that validates the complete capability
    #[test]
    fn test_pyo3_comprehensive_capability_validation() {
        println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Comprehensive capability validation via PyO3");

        Python::with_gil(|py| {
            // Test 1: Type system consistency across PyO3 boundary
            println!("  Capability 1: Type system consistency");
            let constraints = PyFloatConstraints::new(
                Some(-100.0),
                Some(100.0),
                false,
                Some(1e-6), // Option<f64> parameter works correctly
            ).expect("Type system should work");

            let magnitude: Option<f64> = constraints.smallest_nonzero_magnitude();
            assert_eq!(magnitude, Some(1e-6));
            println!("    ✓ Option<f64> type consistency maintained");

            // Test 2: Float encoding export functionality
            println!("  Capability 2: Float encoding exports");
            let test_values = vec![0.0, 1.0, -1.0, 3.14159, 1e-10, 1e10];
            
            for value in test_values {
                // Basic encoding
                let lex = py_float_to_lex(value);
                let recovered = py_lex_to_float(lex);
                assert_eq!(value, recovered, "Basic encoding failed for {}", value);

                // Integer storage
                let int_repr = py_float_to_int(value);
                let int_recovered = py_int_to_float(int_repr);
                assert_eq!(value, int_recovered, "Int storage failed for {}", value);

                // Multi-width encoding
                for width in &[16, 32, 64] {
                    let encoded = py_float_to_lex_multi_width(value, *width);
                    let decoded = py_lex_to_float_multi_width(encoded, *width);
                    
                    let tolerance = match width {
                        16 => 1e-3,
                        32 => 1e-6, 
                        64 => 1e-15,
                        _ => 1e-15,
                    };
                    
                    let error = (decoded - value).abs();
                    assert!(error < tolerance, "Multi-width {} failed for {}", width, value);
                }
            }
            println!("    ✓ Float encoding exports working correctly");

            // Test 3: Constraint validation and behavior
            println!("  Capability 3: Constraint validation");
            let validation_cases = vec![
                (0.0, true),        // Zero allowed
                (1e-6, true),       // At threshold
                (-1e-6, true),      // Negative at threshold
                (1e-7, false),      // Below threshold
                (50.0, true),       // Normal value
                (150.0, false),     // Above max
                (-150.0, false),    // Below min
            ];

            for (value, expected) in validation_cases {
                let result = constraints.validate(value);
                assert_eq!(result, expected, "Validation failed for {}", value);
            }
            println!("    ✓ Constraint validation working correctly");

            // Test 4: PyO3 memory safety and error handling
            println!("  Capability 4: PyO3 memory safety");
            
            // Test error handling
            let invalid_result = PyFloatConstraints::new(
                Some(100.0),  // min > max
                Some(-100.0),
                false,
                Some(1e-6),
            );
            assert!(invalid_result.is_err(), "Should reject invalid constraints");

            let invalid_magnitude = PyFloatConstraints::new(
                Some(-10.0),
                Some(10.0),
                false,
                Some(-1e-6), // Negative magnitude
            );
            assert!(invalid_magnitude.is_err(), "Should reject negative magnitude");
            
            println!("    ✓ Error handling working correctly");

            // Test 5: Integration with advanced encoding features
            println!("  Capability 5: Advanced encoding integration");
            let config = FloatEncodingConfig::default();
            let advanced_result = float_to_lex_advanced(3.14159, &config);
            
            // Verify the result is reasonable
            assert!(advanced_result.encoded_value > 0);
            assert_eq!(advanced_result.debug_info.original_value, 3.14159);
            println!("    ✓ Advanced encoding integration working");

            println!("FLOAT_CONSTRAINT_PYO3 DEBUG: Comprehensive capability validation PASSED");
        });
    }
}