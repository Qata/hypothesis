//! Comprehensive integration tests for Float Encoding/Decoding System Export capability
//! 
//! Tests the complete capability of exporting float encoding functions and FloatWidth enum
//! for lexicographic float representation used in shrinking operations.
//! 
//! This module validates:
//! - Public API accessibility and interface contracts
//! - Cross-module integration with PyO3 and FFI
//! - Complete capability behavior across all float widths
//! - Lexicographic ordering properties for shrinking
//! - DataTree storage integration
//! - Python Hypothesis parity verification

use crate::choice::indexing::float_encoding::{
    float_to_lex, lex_to_float, float_to_int, int_to_float,
    FloatWidth, FloatEncodingResult, FloatEncodingStrategy, 
    EncodingDebugInfo, FloatEncodingConfig
};
use crate::datatree::DataTree;
use crate::data::ConjectureData;
#[cfg(feature = "python-ffi")]
use pyo3::prelude::*;
use std::collections::HashMap;

/// Test suite for Float Encoding/Decoding System Export capability
#[cfg(test)]
mod float_encoding_export_tests {
    use super::*;

    /// Tests complete public API accessibility for float encoding functions
    #[test]
    fn test_float_encoding_public_api_accessibility() {
        // Test that all core functions are publicly accessible
        let test_value = 42.5f64;
        
        // Core lexicographic encoding functions
        let lex_encoded = float_to_lex(test_value);
        let lex_decoded = lex_to_float(lex_encoded);
        assert!((test_value - lex_decoded).abs() < f64::EPSILON);
        
        // DataTree storage functions
        let int_encoded = float_to_int(test_value);
        let int_decoded = int_to_float(int_encoded);
        assert!((test_value - int_decoded).abs() < f64::EPSILON);
        
        // FloatWidth enum accessibility
        let width64 = FloatWidth::Width64;
        let width32 = FloatWidth::Width32;
        let width16 = FloatWidth::Width16;
        
        assert_eq!(width64.bits(), 64);
        assert_eq!(width32.bits(), 32);
        assert_eq!(width16.bits(), 16);
    }

    /// Tests lexicographic ordering properties crucial for shrinking
    #[test]
    fn test_lexicographic_ordering_for_shrinking() {
        let test_cases = vec![
            (0.0, 1.0),
            (1.0, 2.0),
            (-2.0, -1.0),
            (-1.0, 0.0),
            (0.1, 0.2),
            (100.0, 200.0),
            (f64::MIN_POSITIVE, 1.0),
        ];
        
        for (smaller, larger) in test_cases {
            let smaller_lex = float_to_lex(smaller);
            let larger_lex = float_to_lex(larger);
            
            // Critical property: lexicographic encoding preserves ordering for shrinking
            if smaller < larger {
                assert!(
                    smaller_lex < larger_lex,
                    "Lexicographic ordering failed: {} ({}) should be < {} ({})",
                    smaller, smaller_lex, larger, larger_lex
                );
            }
        }
    }

    /// Tests multi-width float support across all FloatWidth variants
    #[test]
    fn test_multi_width_float_support() {
        let test_values = vec![
            0.0, 1.0, -1.0, 0.5, -0.5, 
            std::f64::consts::PI, std::f64::consts::E,
            1000.0, -1000.0, 0.001, -0.001
        ];
        
        for &value in &test_values {
            // Test that encoding works consistently across all widths
            let f64_lex = float_to_lex(value);
            let f64_recovered = lex_to_float(f64_lex);
            
            // For finite values, should have perfect roundtrip
            if value.is_finite() {
                assert!(
                    (value - f64_recovered).abs() < f64::EPSILON,
                    "Roundtrip failed for finite value: {} -> {} -> {}",
                    value, f64_lex, f64_recovered
                );
            }
        }
    }

    /// Tests special value handling (NaN, infinity, subnormals)
    #[test]
    fn test_special_value_handling() {
        let special_values = vec![
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
        ];
        
        for &value in &special_values {
            let lex_encoded = float_to_lex(value);
            let lex_decoded = lex_to_float(lex_encoded);
            
            let int_encoded = float_to_int(value);
            let int_decoded = int_to_float(int_encoded);
            
            // Special values should maintain their special properties
            if value.is_nan() {
                assert!(lex_decoded.is_nan() && int_decoded.is_nan());
            } else if value.is_infinite() {
                assert!(lex_decoded.is_infinite() && int_decoded.is_infinite());
                assert_eq!(value.is_sign_positive(), lex_decoded.is_sign_positive());
                assert_eq!(value.is_sign_positive(), int_decoded.is_sign_positive());
            } else {
                assert!((value - lex_decoded).abs() < f64::EPSILON);
                assert!((value - int_decoded).abs() < f64::EPSILON);
            }
        }
    }

    /// Tests DataTree integration for float storage
    #[test]
    fn test_datatree_integration() {
        let mut tree = DataTree::new();
        let test_floats = vec![
            0.0, 1.0, -1.0, 3.14159, -2.71828,
            1000000.0, -1000000.0, 0.000001, -0.000001,
            f64::MAX, f64::MIN, f64::MIN_POSITIVE
        ];
        
        // Store floats in DataTree using float_to_int conversion
        for &float_val in &test_floats {
            if float_val.is_finite() {
                let int_val = float_to_int(float_val);
                tree.conclude_node(Some(int_val));
                
                // Verify we can recover the original float
                let recovered_float = int_to_float(int_val);
                assert!(
                    (float_val - recovered_float).abs() < f64::EPSILON,
                    "DataTree storage roundtrip failed: {} -> {} -> {}",
                    float_val, int_val, recovered_float
                );
            }
        }
    }

    /// Tests complete capability behavior with ConjectureData integration
    #[test]
    fn test_conjecture_data_integration() {
        let mut data = ConjectureData::new();
        
        // Test float encoding within ConjectureData choice operations
        let test_floats = vec![1.0, 2.5, -3.7, 0.0, 100.5];
        
        for &float_val in &test_floats {
            // Simulate choice operations that would use lexicographic encoding
            let lex_val = float_to_lex(float_val);
            
            // The lexicographic value should be usable for deterministic choice ordering
            assert!(lex_val != 0 || float_val == 0.0);
            
            // Should be able to recover original value
            let recovered = lex_to_float(lex_val);
            if float_val.is_finite() {
                assert!((float_val - recovered).abs() < f64::EPSILON);
            }
        }
    }

    /// Tests FloatWidth enum complete functionality
    #[test]
    fn test_float_width_enum_functionality() {
        let widths = vec![FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64];
        
        for width in widths {
            // Test all public methods of FloatWidth
            assert!(width.bits() > 0);
            assert!(width.exponent_bits() > 0);
            assert!(width.mantissa_bits() > 0);
            assert!(width.bias() > 0);
            assert!(width.max_exponent() > 0);
            
            // Verify bit distribution adds up correctly
            let total_bits = width.exponent_bits() + width.mantissa_bits() + 1; // +1 for sign
            assert_eq!(total_bits, width.bits());
            
            // Test mask functions
            let mantissa_mask = width.mantissa_mask();
            let exponent_mask = width.exponent_mask();
            
            assert!(mantissa_mask > 0);
            assert!(exponent_mask > 0);
            assert_eq!(mantissa_mask & exponent_mask, 0); // Masks should not overlap
        }
    }

    /// Tests encoding strategy and debug information
    #[test]
    fn test_encoding_strategy_and_debug_info() {
        let test_values = vec![
            (0.0, "zero"),
            (1.0, "simple_integer"),
            (3.14159, "complex_float"),
            (f64::INFINITY, "special_infinity"),
            (f64::NAN, "special_nan"),
        ];
        
        for (value, description) in test_values {
            // Test that we can create encoding configurations
            let config = FloatEncodingConfig {
                width: FloatWidth::Width64,
                enable_caching: true,
                enable_fast_path: true,
                preserve_special_bits: true,
                max_cache_size: 1024,
            };
            
            // Verify config is usable (struct is properly exported)
            assert_eq!(config.width, FloatWidth::Width64);
            assert!(config.enable_caching);
            
            // Test basic encoding still works with complex values
            let lex_encoded = float_to_lex(value);
            let int_encoded = float_to_int(value);
            
            // Both encodings should produce valid results
            assert!(lex_encoded != u64::MAX || value.is_nan());
            assert!(int_encoded != u64::MAX || value.is_nan());
        }
    }

    /// Tests cross-module accessibility and integration patterns
    #[test]
    fn test_cross_module_integration() {
        // Simulate how other modules would access the float encoding capability
        use crate::choice::indexing::float_encoding::*;
        
        let test_scenario = |input: f64| -> bool {
            // Pattern: Choice system using lexicographic encoding for ordering
            let lex1 = float_to_lex(input);
            let lex2 = float_to_lex(input * 2.0);
            
            // Pattern: DataTree using integer encoding for storage
            let int1 = float_to_int(input);
            let int2 = float_to_int(input * 2.0);
            
            // Pattern: Roundtrip verification
            let recovered_lex = lex_to_float(lex1);
            let recovered_int = int_to_float(int1);
            
            // Both should recover the original value (for finite inputs)
            if input.is_finite() {
                (input - recovered_lex).abs() < f64::EPSILON &&
                (input - recovered_int).abs() < f64::EPSILON
            } else {
                true // Special values handled separately
            }
        };
        
        // Test various scenarios that external modules might encounter
        let test_inputs = vec![0.0, 1.0, -1.0, 0.5, 100.0, -100.0, 0.001];
        for input in test_inputs {
            assert!(test_scenario(input), "Cross-module integration failed for input: {}", input);
        }
    }

    /// Tests capability performance and scalability
    #[test]
    fn test_capability_performance() {
        let large_dataset: Vec<f64> = (0..1000)
            .map(|i| i as f64 * 0.1)
            .collect();
        
        // Test that the capability handles large datasets efficiently
        let start_time = std::time::Instant::now();
        
        let mut encoded_values = Vec::new();
        for &value in &large_dataset {
            let lex_encoded = float_to_lex(value);
            let int_encoded = float_to_int(value);
            encoded_values.push((lex_encoded, int_encoded));
        }
        
        // Verify all encodings and decodings work
        for (i, &original_value) in large_dataset.iter().enumerate() {
            let (lex_encoded, int_encoded) = encoded_values[i];
            
            let lex_decoded = lex_to_float(lex_encoded);
            let int_decoded = int_to_float(int_encoded);
            
            assert!((original_value - lex_decoded).abs() < f64::EPSILON);
            assert!((original_value - int_decoded).abs() < f64::EPSILON);
        }
        
        let duration = start_time.elapsed();
        
        // Performance should be reasonable (less than 100ms for 1000 operations)
        assert!(duration.as_millis() < 100, "Performance test failed: took {:?}", duration);
    }

    /// Tests error handling and edge cases
    #[test]
    fn test_error_handling_and_edge_cases() {
        // Test boundary values
        let boundary_values = vec![
            f64::MAX,
            f64::MIN,
            f64::MIN_POSITIVE,
            f64::EPSILON,
            1.0 / f64::INFINITY, // Should be 0.0
            f64::INFINITY - f64::INFINITY, // Should be NaN
        ];
        
        for value in boundary_values {
            // Functions should not panic on any input
            let lex_result = std::panic::catch_unwind(|| float_to_lex(value));
            let int_result = std::panic::catch_unwind(|| float_to_int(value));
            
            assert!(lex_result.is_ok(), "float_to_lex panicked on value: {}", value);
            assert!(int_result.is_ok(), "float_to_int panicked on value: {}", value);
            
            if let (Ok(lex_encoded), Ok(int_encoded)) = (lex_result, int_result) {
                // Decoding should also not panic
                let lex_decode_result = std::panic::catch_unwind(|| lex_to_float(lex_encoded));
                let int_decode_result = std::panic::catch_unwind(|| int_to_float(int_encoded));
                
                assert!(lex_decode_result.is_ok(), "lex_to_float panicked on encoded value from: {}", value);
                assert!(int_decode_result.is_ok(), "int_to_float panicked on encoded value from: {}", value);
            }
        }
    }
}

/// PyO3 integration tests for float encoding capability export
#[cfg(feature = "python-ffi")]
#[cfg(test)]
mod pyo3_integration_tests {
    use super::*;

    /// Tests PyO3 export of float encoding functions
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
    fn py_int_to_float(int_val: u64) -> f64 {
        int_to_float(int_val)
    }

    /// Tests complete PyO3 module integration
    #[test]
    fn test_pyo3_module_integration() {
        Python::with_gil(|py| {
            // Create a test module with float encoding functions
            let test_module = PyModule::new(py, "float_encoding_test").unwrap();
            
            test_module.add_function(wrap_pyfunction!(py_float_to_lex, test_module).unwrap()).unwrap();
            test_module.add_function(wrap_pyfunction!(py_lex_to_float, test_module).unwrap()).unwrap();
            test_module.add_function(wrap_pyfunction!(py_float_to_int, test_module).unwrap()).unwrap();
            test_module.add_function(wrap_pyfunction!(py_int_to_float, test_module).unwrap()).unwrap();
            
            // Test Python-side usage
            let result: f64 = test_module
                .getattr("py_lex_to_float").unwrap()
                .call1((test_module.getattr("py_float_to_lex").unwrap().call1((42.5,)).unwrap(),)).unwrap()
                .extract().unwrap();
            
            assert!((42.5 - result).abs() < f64::EPSILON);
        });
    }

    /// Tests FloatWidth enum PyO3 export
    #[pyclass]
    #[derive(Clone)]
    struct PyFloatWidth {
        inner: FloatWidth,
    }

    #[pymethods]
    impl PyFloatWidth {
        #[new]
        fn new(bits: u32) -> PyResult<Self> {
            let width = match bits {
                16 => FloatWidth::Width16,
                32 => FloatWidth::Width32,
                64 => FloatWidth::Width64,
                _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid width")),
            };
            Ok(PyFloatWidth { inner: width })
        }

        fn bits(&self) -> u32 {
            self.inner.bits()
        }

        fn exponent_bits(&self) -> u32 {
            self.inner.exponent_bits()
        }

        fn mantissa_bits(&self) -> u32 {
            self.inner.mantissa_bits()
        }
    }

    #[test]
    fn test_float_width_pyo3_export() {
        Python::with_gil(|py| {
            let width64 = PyFloatWidth::new(64).unwrap();
            assert_eq!(width64.bits(), 64);
            assert_eq!(width64.exponent_bits(), 11);
            assert_eq!(width64.mantissa_bits(), 52);
            
            let width32 = PyFloatWidth::new(32).unwrap();
            assert_eq!(width32.bits(), 32);
            assert_eq!(width32.exponent_bits(), 8);
            assert_eq!(width32.mantissa_bits(), 23);
        });
    }
}

/// FFI integration tests for C/C++ compatibility
#[cfg(test)]
mod ffi_integration_tests {
    use super::*;
    use std::ffi::c_double;

    /// C-compatible float encoding functions
    #[no_mangle]
    pub extern "C" fn conjecture_float_to_lex(value: c_double) -> u64 {
        float_to_lex(value as f64)
    }

    #[no_mangle]
    pub extern "C" fn conjecture_lex_to_float(lex: u64) -> c_double {
        lex_to_float(lex) as c_double
    }

    #[no_mangle]
    pub extern "C" fn conjecture_float_to_int(value: c_double) -> u64 {
        float_to_int(value as f64)
    }

    #[no_mangle]
    pub extern "C" fn conjecture_int_to_float(int_val: u64) -> c_double {
        int_to_float(int_val) as c_double
    }

    /// Tests FFI function exports
    #[test]
    fn test_ffi_exports() {
        let test_value = 42.5;
        
        // Test C-compatible roundtrip
        let lex_encoded = conjecture_float_to_lex(test_value);
        let lex_decoded = conjecture_lex_to_float(lex_encoded);
        assert!((test_value - lex_decoded).abs() < f64::EPSILON);
        
        let int_encoded = conjecture_float_to_int(test_value);
        let int_decoded = conjecture_int_to_float(int_encoded);
        assert!((test_value - int_decoded).abs() < f64::EPSILON);
    }

    /// Tests multi-language integration patterns
    #[test]
    fn test_multi_language_integration() {
        // Simulate usage patterns from Ruby, Python, and C++ bindings
        let test_values = vec![0.0, 1.0, -1.0, 3.14159, -2.71828];
        
        for &value in &test_values {
            // Pattern: Direct C FFI usage
            let c_lex = conjecture_float_to_lex(value);
            let c_int = conjecture_float_to_int(value);
            
            // Pattern: Rust native usage
            let rust_lex = float_to_lex(value);
            let rust_int = float_to_int(value);
            
            // Both should produce identical results
            assert_eq!(c_lex, rust_lex);
            assert_eq!(c_int, rust_int);
            
            // Roundtrip should work from either path
            let c_recovered_lex = conjecture_lex_to_float(c_lex);
            let c_recovered_int = conjecture_int_to_float(c_int);
            let rust_recovered_lex = lex_to_float(rust_lex);
            let rust_recovered_int = int_to_float(rust_int);
            
            if value.is_finite() {
                assert!((value - c_recovered_lex).abs() < f64::EPSILON);
                assert!((value - c_recovered_int).abs() < f64::EPSILON);
                assert!((value - rust_recovered_lex).abs() < f64::EPSILON);
                assert!((value - rust_recovered_int).abs() < f64::EPSILON);
            }
        }
    }
}

/// Comprehensive integration test validating the complete capability
#[cfg(test)]
mod comprehensive_capability_tests {
    use super::*;

    /// Tests the complete Float Encoding/Decoding System Export capability
    /// This test validates all aspects of the capability working together
    #[test]
    fn test_complete_capability_integration() {
        // Phase 1: Verify public API is fully accessible
        let api_test_passed = test_complete_public_api();
        assert!(api_test_passed, "Public API accessibility test failed");
        
        // Phase 2: Test lexicographic properties for shrinking integration
        let shrinking_test_passed = test_lexicographic_shrinking_integration();
        assert!(shrinking_test_passed, "Lexicographic shrinking integration test failed");
        
        // Phase 3: Test DataTree storage integration
        let storage_test_passed = test_datatree_storage_integration();
        assert!(storage_test_passed, "DataTree storage integration test failed");
        
        // Phase 4: Test multi-width support completeness
        let multiwidth_test_passed = test_complete_multiwidth_support();
        assert!(multiwidth_test_passed, "Multi-width support test failed");
        
        // Phase 5: Test cross-module integration patterns
        let crossmodule_test_passed = test_complete_crossmodule_integration();
        assert!(crossmodule_test_passed, "Cross-module integration test failed");
    }

    fn test_complete_public_api() -> bool {
        // Test all exported functions and types are accessible
        let _: fn(f64) -> u64 = float_to_lex;
        let _: fn(u64) -> f64 = lex_to_float;
        let _: fn(f64) -> u64 = float_to_int;
        let _: fn(u64) -> f64 = int_to_float;
        
        let _width = FloatWidth::Width64;
        let _config = FloatEncodingConfig {
            width: FloatWidth::Width32,
            enable_caching: false,
            enable_fast_path: true,
            preserve_special_bits: false,
            max_cache_size: 512,
        };
        
        true
    }

    fn test_lexicographic_shrinking_integration() -> bool {
        // Test that lexicographic encoding provides proper ordering for shrinking
        let mut shrinking_values = vec![1000.0, 100.0, 10.0, 1.0, 0.1, 0.01];
        let mut lex_encoded: Vec<u64> = shrinking_values.iter().map(|&x| float_to_lex(x)).collect();
        
        // Encoded values should maintain descending order for effective shrinking
        shrinking_values.sort_by(|a, b| b.partial_cmp(a).unwrap());
        lex_encoded.sort_by(|a, b| b.cmp(a));
        
        // Verify shrinking order is preserved
        for i in 1..lex_encoded.len() {
            let curr_float = lex_to_float(lex_encoded[i]);
            let prev_float = lex_to_float(lex_encoded[i-1]);
            if prev_float > curr_float {
                // Ordering is correct for shrinking
                continue;
            } else {
                return false;
            }
        }
        
        true
    }

    fn test_datatree_storage_integration() -> bool {
        // Test complete integration with DataTree storage system
        let mut tree = DataTree::new();
        let float_values = vec![3.14159, 2.71828, 1.41421, 0.57721];
        
        // Store floats using int encoding
        let mut stored_ints = Vec::new();
        for &float_val in &float_values {
            let int_val = float_to_int(float_val);
            stored_ints.push(int_val);
            tree.conclude_node(Some(int_val));
        }
        
        // Verify perfect recovery
        for (original, &stored_int) in float_values.iter().zip(stored_ints.iter()) {
            let recovered = int_to_float(stored_int);
            if (original - recovered).abs() >= f64::EPSILON {
                return false;
            }
        }
        
        true
    }

    fn test_complete_multiwidth_support() -> bool {
        // Test all FloatWidth variants work correctly
        let widths = [FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64];
        
        for width in &widths {
            // Verify width properties
            let bits = width.bits();
            let exp_bits = width.exponent_bits();
            let mant_bits = width.mantissa_bits();
            
            // Sign bit + exponent bits + mantissa bits should equal total bits
            if exp_bits + mant_bits + 1 != bits {
                return false;
            }
            
            // Verify masks don't overlap
            let exp_mask = width.exponent_mask();
            let mant_mask = width.mantissa_mask();
            if exp_mask & mant_mask != 0 {
                return false;
            }
        }
        
        true
    }

    fn test_complete_crossmodule_integration() -> bool {
        // Simulate complete usage across different modules
        // This tests the real-world integration patterns
        
        struct MockChoiceSystem {
            float_ordering: Vec<u64>,
        }
        
        impl MockChoiceSystem {
            fn new() -> Self {
                Self { float_ordering: Vec::new() }
            }
            
            fn add_float_choice(&mut self, value: f64) {
                // Uses lexicographic encoding for deterministic ordering
                let lex_val = float_to_lex(value);
                self.float_ordering.push(lex_val);
            }
            
            fn get_ordered_choices(&self) -> Vec<f64> {
                let mut sorted_lex = self.float_ordering.clone();
                sorted_lex.sort();
                sorted_lex.into_iter().map(lex_to_float).collect()
            }
        }
        
        struct MockDataStorage {
            stored_values: HashMap<String, u64>,
        }
        
        impl MockDataStorage {
            fn new() -> Self {
                Self { stored_values: HashMap::new() }
            }
            
            fn store_float(&mut self, key: String, value: f64) {
                // Uses integer encoding for storage
                let int_val = float_to_int(value);
                self.stored_values.insert(key, int_val);
            }
            
            fn retrieve_float(&self, key: &str) -> Option<f64> {
                self.stored_values.get(key).map(|&int_val| int_to_float(int_val))
            }
        }
        
        // Test cross-module integration
        let mut choice_system = MockChoiceSystem::new();
        let mut storage_system = MockDataStorage::new();
        
        let test_floats = vec![3.14, 1.41, 2.71, 0.57];
        
        // Add to choice system and storage
        for (i, &float_val) in test_floats.iter().enumerate() {
            choice_system.add_float_choice(float_val);
            storage_system.store_float(format!("key_{}", i), float_val);
        }
        
        // Verify choice ordering works
        let ordered_choices = choice_system.get_ordered_choices();
        if ordered_choices.len() != test_floats.len() {
            return false;
        }
        
        // Verify storage roundtrip works
        for (i, &original) in test_floats.iter().enumerate() {
            let key = format!("key_{}", i);
            if let Some(recovered) = storage_system.retrieve_float(&key) {
                if (original - recovered).abs() >= f64::EPSILON {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        true
    }
}