//! Comprehensive FFI Tests for Float Encoding Export Functions
//!
//! This module provides exhaustive tests for the float encoding export functions
//! via C FFI, PyO3, and WebAssembly interfaces, ensuring complete functionality
//! and compatibility with external consumers.

use crate::float_encoding_export::{
    // Rust API
    float_to_lex, lex_to_float, float_to_int, int_to_float,
    float_to_lex_multi_width, lex_to_float_multi_width,
    build_exponent_tables, build_exponent_tables_for_width_export,
    FloatWidth, FloatEncodingConfig, float_to_lex_advanced,
    
    // C FFI API
    conjecture_float_to_lex, conjecture_lex_to_float,
    conjecture_float_to_int, conjecture_int_to_float,
    conjecture_float_width_bits, conjecture_float_width_mantissa_bits,
    conjecture_float_width_exponent_bits,
};

/// Test data structure for comprehensive validation
#[derive(Debug, Clone)]
struct FloatTestCase {
    name: &'static str,
    value: f64,
    expect_exact_roundtrip: bool,
    precision_tolerance: f64,
}

impl FloatTestCase {
    fn new(name: &'static str, value: f64) -> Self {
        Self {
            name,
            value,
            expect_exact_roundtrip: value.is_finite(),
            precision_tolerance: if value.is_finite() { 0.0 } else { f64::EPSILON },
        }
    }
    
    fn with_tolerance(name: &'static str, value: f64, tolerance: f64) -> Self {
        Self {
            name,
            value,
            expect_exact_roundtrip: false,
            precision_tolerance: tolerance,
        }
    }
}

/// Comprehensive test cases covering all float scenarios
fn get_comprehensive_test_cases() -> Vec<FloatTestCase> {
    vec![
        // Simple values
        FloatTestCase::new("zero", 0.0),
        FloatTestCase::new("negative_zero", -0.0),
        FloatTestCase::new("one", 1.0),
        FloatTestCase::new("negative_one", -1.0),
        FloatTestCase::new("two", 2.0),
        FloatTestCase::new("ten", 10.0),
        FloatTestCase::new("hundred", 100.0),
        
        // Mathematical constants
        FloatTestCase::new("pi", std::f64::consts::PI),
        FloatTestCase::new("e", std::f64::consts::E),
        FloatTestCase::new("sqrt_2", std::f64::consts::SQRT_2),
        FloatTestCase::new("ln_2", std::f64::consts::LN_2),
        
        // Fractional values
        FloatTestCase::new("half", 0.5),
        FloatTestCase::new("quarter", 0.25),
        FloatTestCase::new("eighth", 0.125),
        FloatTestCase::new("third", 1.0/3.0),
        
        // Small values
        FloatTestCase::new("small_positive", 1e-10),
        FloatTestCase::new("small_negative", -1e-10),
        FloatTestCase::new("tiny_positive", 1e-100),
        FloatTestCase::new("tiny_negative", -1e-100),
        FloatTestCase::new("min_positive", f64::MIN_POSITIVE),
        
        // Large values
        FloatTestCase::new("large_positive", 1e10),
        FloatTestCase::new("large_negative", -1e10),
        FloatTestCase::new("huge_positive", 1e100),
        FloatTestCase::new("huge_negative", -1e100),
        FloatTestCase::new("max_value", f64::MAX),
        FloatTestCase::new("min_value", f64::MIN),
        
        // Special values
        FloatTestCase::new("positive_infinity", f64::INFINITY),
        FloatTestCase::new("negative_infinity", f64::NEG_INFINITY),
        FloatTestCase::new("nan", f64::NAN),
        
        // Precision edge cases
        FloatTestCase::with_tolerance("near_zero_positive", 1e-308, 1e-15),
        FloatTestCase::with_tolerance("near_zero_negative", -1e-308, 1e-15),
        FloatTestCase::with_tolerance("near_max_positive", f64::MAX - 1e100, 1e100),
        FloatTestCase::with_tolerance("near_min_negative", f64::MIN + 1e100, 1e100),
    ]
}

#[cfg(test)]
mod rust_api_tests {
    use super::*;

    /// Test basic Rust API float encoding/decoding
    #[test]
    fn test_rust_api_basic_encoding() {
        println!("FFI_TESTS DEBUG: Testing Rust API basic encoding");

        for case in get_comprehensive_test_cases() {
            println!("  Testing case: {}", case.name);
            
            // Test lex encoding round-trip
            let lex_encoded = float_to_lex(case.value);
            let lex_decoded = lex_to_float(lex_encoded);
            
            // Test int conversion round-trip
            let int_encoded = float_to_int(case.value);
            let int_decoded = int_to_float(int_encoded);
            
            if case.value.is_nan() {
                assert!(lex_decoded.is_nan(), "Lex: NaN should round-trip to NaN for {}", case.name);
                assert!(int_decoded.is_nan(), "Int: NaN should round-trip to NaN for {}", case.name);
            } else if case.expect_exact_roundtrip {
                assert_eq!(case.value, lex_decoded, "Lex: Exact round-trip failed for {}", case.name);
                assert_eq!(case.value, int_decoded, "Int: Exact round-trip failed for {}", case.name);
            } else {
                let lex_error = (lex_decoded - case.value).abs();
                let int_error = (int_decoded - case.value).abs();
                assert!(lex_error <= case.precision_tolerance, 
                       "Lex: Precision error {} > {} for {}", lex_error, case.precision_tolerance, case.name);
                assert!(int_error <= case.precision_tolerance,
                       "Int: Precision error {} > {} for {}", int_error, case.precision_tolerance, case.name);
            }
            
            println!("    {} -> lex: 0x{:016X} -> {}, int: 0x{:016X} -> {}", 
                    case.value, lex_encoded, lex_decoded, int_encoded, int_decoded);
        }
        
        println!("FFI_TESTS DEBUG: Rust API basic encoding PASSED");
    }

    /// Test Rust API multi-width encoding
    #[test]
    fn test_rust_api_multi_width_encoding() {
        println!("FFI_TESTS DEBUG: Testing Rust API multi-width encoding");

        let test_values = vec![0.0, 1.0, -1.0, 3.14159, 1e-6, 1e6];
        let widths = vec![
            (FloatWidth::Width16, 16, 1e-3),
            (FloatWidth::Width32, 32, 1e-6),
            (FloatWidth::Width64, 64, 1e-15),
        ];

        for value in test_values {
            for (width, width_bits, tolerance) in &widths {
                let encoded = float_to_lex_multi_width(value, *width);
                let decoded = lex_to_float_multi_width(encoded, *width);
                
                if value.is_finite() {
                    let error = (decoded - value).abs();
                    assert!(error < *tolerance, 
                           "Width {} precision error {} > {} for value {}", 
                           width_bits, error, tolerance, value);
                }
                
                println!("    Value {} (width {}) -> 0x{:016X} -> {} (error: {})", 
                        value, width_bits, encoded, decoded, (decoded - value).abs());
            }
        }
        
        println!("FFI_TESTS DEBUG: Rust API multi-width encoding PASSED");
    }

    /// Test Rust API advanced encoding features
    #[test]
    fn test_rust_api_advanced_encoding() {
        println!("FFI_TESTS DEBUG: Testing Rust API advanced encoding");

        let config = FloatEncodingConfig::default();
        let test_values = vec![1.0, 1.5, 42.0, 3.14159, -2.718281828];

        for value in test_values {
            let result = float_to_lex_advanced(value, &config);
            
            // Verify the result structure
            assert_eq!(result.debug_info.original_value, value);
            assert!(result.encoded_value > 0);
            
            // Verify that the basic encoding matches
            let basic_encoding = float_to_lex(value);
            assert_eq!(result.encoded_value, basic_encoding);
            
            println!("    Advanced encoding {} -> strategy: {:?}, value: 0x{:016X}", 
                    value, result.strategy, result.encoded_value);
        }
        
        println!("FFI_TESTS DEBUG: Rust API advanced encoding PASSED");
    }

    /// Test Rust API exponent table generation
    #[test]
    fn test_rust_api_exponent_tables() {
        println!("FFI_TESTS DEBUG: Testing Rust API exponent tables");

        // Test default f64 tables
        let (encoding, decoding) = build_exponent_tables();
        assert_eq!(encoding.len(), 2048, "f64 should have 2048 exponent entries");
        assert_eq!(decoding.len(), 2048, "f64 should have 2048 decoding entries");
        
        // Verify round-trip through tables
        for i in 0..encoding.len() {
            let encoded = encoding[i];
            let decoded = decoding[encoded as usize];
            assert_eq!(i as u32, decoded, "Exponent table round-trip failed for index {}", i);
        }

        // Test width-specific tables
        let widths = vec![FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64];
        for width in widths {
            let (enc, dec) = build_exponent_tables_for_width_export(width);
            let expected_size = (width.max_exponent() + 1) as usize;
            
            assert_eq!(enc.len(), expected_size, "{:?} should have {} entries", width, expected_size);
            assert_eq!(dec.len(), expected_size, "{:?} should have {} entries", width, expected_size);
            
            // Verify round-trip
            for i in 0..enc.len() {
                let encoded = enc[i];
                if (encoded as usize) < dec.len() {
                    let decoded = dec[encoded as usize];
                    assert_eq!(i as u32, decoded, "{:?} exponent table round-trip failed", width);
                }
            }
            
            println!("    {:?} tables: {} entries, round-trip verified", width, enc.len());
        }
        
        println!("FFI_TESTS DEBUG: Rust API exponent tables PASSED");
    }
}

#[cfg(test)]
mod c_ffi_tests {
    use super::*;

    /// Test C FFI basic functions
    #[test]
    fn test_c_ffi_basic_functions() {
        println!("FFI_TESTS DEBUG: Testing C FFI basic functions");

        for case in get_comprehensive_test_cases() {
            println!("  Testing C FFI case: {}", case.name);
            
            // Test C FFI lex encoding
            let c_lex_encoded = conjecture_float_to_lex(case.value);
            let c_lex_decoded = conjecture_lex_to_float(c_lex_encoded);
            
            // Test C FFI int conversion
            let c_int_encoded = conjecture_float_to_int(case.value);
            let c_int_decoded = conjecture_int_to_float(c_int_encoded);
            
            // Compare with Rust API results
            let rust_lex_encoded = float_to_lex(case.value);
            let rust_int_encoded = float_to_int(case.value);
            
            assert_eq!(c_lex_encoded, rust_lex_encoded, "C FFI lex encoding should match Rust API for {}", case.name);
            assert_eq!(c_int_encoded, rust_int_encoded, "C FFI int encoding should match Rust API for {}", case.name);
            
            if case.value.is_nan() {
                assert!(c_lex_decoded.is_nan(), "C FFI lex: NaN should round-trip for {}", case.name);
                assert!(c_int_decoded.is_nan(), "C FFI int: NaN should round-trip for {}", case.name);
            } else if case.expect_exact_roundtrip {
                assert_eq!(case.value, c_lex_decoded, "C FFI lex: Exact round-trip failed for {}", case.name);
                assert_eq!(case.value, c_int_decoded, "C FFI int: Exact round-trip failed for {}", case.name);
            }
            
            println!("    C FFI {} -> lex: 0x{:016X} -> {}, int: 0x{:016X} -> {}", 
                    case.value, c_lex_encoded, c_lex_decoded, c_int_encoded, c_int_decoded);
        }
        
        println!("FFI_TESTS DEBUG: C FFI basic functions PASSED");
    }

    /// Test C FFI width information functions
    #[test]
    fn test_c_ffi_width_info() {
        println!("FFI_TESTS DEBUG: Testing C FFI width information");

        let width_tests = vec![
            (16, 16, 10, 5),   // f16: 16 total, 10 mantissa, 5 exponent
            (32, 32, 23, 8),   // f32: 32 total, 23 mantissa, 8 exponent
            (64, 64, 52, 11),  // f64: 64 total, 52 mantissa, 11 exponent
        ];

        for (width_input, expected_bits, expected_mantissa, expected_exponent) in width_tests {
            let bits = conjecture_float_width_bits(width_input);
            let mantissa_bits = conjecture_float_width_mantissa_bits(width_input);
            let exponent_bits = conjecture_float_width_exponent_bits(width_input);
            
            assert_eq!(bits, expected_bits, "Width {} should have {} total bits", width_input, expected_bits);
            assert_eq!(mantissa_bits, expected_mantissa, "Width {} should have {} mantissa bits", width_input, expected_mantissa);
            assert_eq!(exponent_bits, expected_exponent, "Width {} should have {} exponent bits", width_input, expected_exponent);
            
            println!("    Width {}: {} bits, {} mantissa, {} exponent", 
                    width_input, bits, mantissa_bits, exponent_bits);
        }
        
        // Test invalid width (should default to f64)
        let invalid_bits = conjecture_float_width_bits(128);
        assert_eq!(invalid_bits, 64, "Invalid width should default to f64");
        
        println!("FFI_TESTS DEBUG: C FFI width information PASSED");
    }

    /// Test C FFI compatibility with different calling conventions
    #[test]
    fn test_c_ffi_calling_conventions() {
        println!("FFI_TESTS DEBUG: Testing C FFI calling conventions");

        // Test that C FFI functions can be called safely from different contexts
        let test_value = 3.14159;
        
        // Direct function calls
        let result1 = conjecture_float_to_lex(test_value);
        let result2 = conjecture_float_to_lex(test_value);
        assert_eq!(result1, result2, "Repeated calls should give same result");
        
        // Function pointer calls (simulating C usage)
        let float_to_lex_ptr: fn(f64) -> u64 = conjecture_float_to_lex;
        let lex_to_float_ptr: fn(u64) -> f64 = conjecture_lex_to_float;
        
        let encoded = float_to_lex_ptr(test_value);
        let decoded = lex_to_float_ptr(encoded);
        assert_eq!(test_value, decoded, "Function pointer calls should work");
        
        // Test with various data types that C might pass
        let c_compatible_values = vec![
            0.0f64, 1.0f64, -1.0f64,
            f64::from(0.0f32), f64::from(1.0f32), f64::from(-1.0f32),
        ];
        
        for value in c_compatible_values {
            let encoded = conjecture_float_to_lex(value);
            let decoded = conjecture_lex_to_float(encoded);
            assert_eq!(value, decoded, "C-compatible value {} should round-trip", value);
        }
        
        println!("FFI_TESTS DEBUG: C FFI calling conventions PASSED");
    }
}

#[cfg(feature = "python-ffi")]
#[cfg(test)]
mod pyo3_ffi_tests {
    use super::*;
    use pyo3::prelude::*;
    use crate::float_encoding_export::{
        py_float_to_lex, py_lex_to_float, py_float_to_int, py_int_to_float,
        py_float_to_lex_multi_width, py_lex_to_float_multi_width,
    };

    /// Test PyO3 FFI functions
    #[test]
    fn test_pyo3_ffi_functions() {
        println!("FFI_TESTS DEBUG: Testing PyO3 FFI functions");

        Python::with_gil(|py| {
            for case in get_comprehensive_test_cases() {
                println!("  Testing PyO3 case: {}", case.name);
                
                // Test PyO3 encoding functions
                let py_lex_encoded = py_float_to_lex(case.value);
                let py_lex_decoded = py_lex_to_float(py_lex_encoded);
                let py_int_encoded = py_float_to_int(case.value);
                let py_int_decoded = py_int_to_float(py_int_encoded);
                
                // Compare with Rust API
                let rust_lex_encoded = float_to_lex(case.value);
                let rust_int_encoded = float_to_int(case.value);
                
                assert_eq!(py_lex_encoded, rust_lex_encoded, "PyO3 lex should match Rust for {}", case.name);
                assert_eq!(py_int_encoded, rust_int_encoded, "PyO3 int should match Rust for {}", case.name);
                
                if case.value.is_nan() {
                    assert!(py_lex_decoded.is_nan(), "PyO3 lex: NaN should round-trip for {}", case.name);
                    assert!(py_int_decoded.is_nan(), "PyO3 int: NaN should round-trip for {}", case.name);
                } else if case.expect_exact_roundtrip {
                    assert_eq!(case.value, py_lex_decoded, "PyO3 lex: Exact round-trip failed for {}", case.name);
                    assert_eq!(case.value, py_int_decoded, "PyO3 int: Exact round-trip failed for {}", case.name);
                }
            }
        });
        
        println!("FFI_TESTS DEBUG: PyO3 FFI functions PASSED");
    }

    /// Test PyO3 multi-width functions
    #[test]
    fn test_pyo3_multi_width_functions() {
        println!("FFI_TESTS DEBUG: Testing PyO3 multi-width functions");

        Python::with_gil(|py| {
            let test_values = vec![0.0, 1.0, -1.0, 3.14159];
            let widths = vec![16, 32, 64];
            
            for value in test_values {
                for width in &widths {
                    let py_encoded = py_float_to_lex_multi_width(value, *width);
                    let py_decoded = py_lex_to_float_multi_width(py_encoded, *width);
                    
                    // Compare with Rust API
                    let rust_width = match width {
                        16 => FloatWidth::Width16,
                        32 => FloatWidth::Width32,
                        64 => FloatWidth::Width64,
                        _ => FloatWidth::Width64,
                    };
                    let rust_encoded = float_to_lex_multi_width(value, rust_width);
                    
                    assert_eq!(py_encoded, rust_encoded, "PyO3 multi-width should match Rust for {} width {}", value, width);
                    
                    if value.is_finite() {
                        let tolerance = match width {
                            16 => 1e-3,
                            32 => 1e-6,
                            64 => 1e-15,
                            _ => 1e-15,
                        };
                        let error = (py_decoded - value).abs();
                        assert!(error < tolerance, "PyO3 width {} precision error too large for {}", width, value);
                    }
                }
            }
        });
        
        println!("FFI_TESTS DEBUG: PyO3 multi-width functions PASSED");
    }
}

#[cfg(target_arch = "wasm32")]
#[cfg(test)]
mod wasm_ffi_tests {
    use super::*;
    use crate::float_encoding_export::{
        wasm_float_to_lex, wasm_lex_to_float, wasm_float_to_int, wasm_int_to_float,
    };

    /// Test WebAssembly FFI functions
    #[test]
    fn test_wasm_ffi_functions() {
        println!("FFI_TESTS DEBUG: Testing WASM FFI functions");

        for case in get_comprehensive_test_cases() {
            println!("  Testing WASM case: {}", case.name);
            
            // Test WASM encoding functions
            let wasm_lex_encoded = wasm_float_to_lex(case.value);
            let wasm_lex_decoded = wasm_lex_to_float(wasm_lex_encoded);
            let wasm_int_encoded = wasm_float_to_int(case.value);
            let wasm_int_decoded = wasm_int_to_float(wasm_int_encoded);
            
            // Compare with Rust API
            let rust_lex_encoded = float_to_lex(case.value);
            let rust_int_encoded = float_to_int(case.value);
            
            assert_eq!(wasm_lex_encoded, rust_lex_encoded, "WASM lex should match Rust for {}", case.name);
            assert_eq!(wasm_int_encoded, rust_int_encoded, "WASM int should match Rust for {}", case.name);
            
            if case.value.is_nan() {
                assert!(wasm_lex_decoded.is_nan(), "WASM lex: NaN should round-trip for {}", case.name);
                assert!(wasm_int_decoded.is_nan(), "WASM int: NaN should round-trip for {}", case.name);
            } else if case.expect_exact_roundtrip {
                assert_eq!(case.value, wasm_lex_decoded, "WASM lex: Exact round-trip failed for {}", case.name);
                assert_eq!(case.value, wasm_int_decoded, "WASM int: Exact round-trip failed for {}", case.name);
            }
        }
        
        println!("FFI_TESTS DEBUG: WASM FFI functions PASSED");
    }
}

#[cfg(test)]
mod cross_interface_compatibility_tests {
    use super::*;

    /// Test that all FFI interfaces produce identical results
    #[test]
    fn test_cross_interface_compatibility() {
        println!("FFI_TESTS DEBUG: Testing cross-interface compatibility");

        let test_values = vec![0.0, 1.0, -1.0, 3.14159, 1e-6, 1e6, f64::INFINITY];
        
        for value in test_values {
            if !value.is_finite() && !value.is_infinite() {
                continue; // Skip NaN for this test
            }
            
            println!("  Testing cross-interface compatibility for: {}", value);
            
            // Get results from all interfaces
            let rust_lex = float_to_lex(value);
            let rust_int = float_to_int(value);
            
            let c_lex = conjecture_float_to_lex(value);
            let c_int = conjecture_float_to_int(value);
            
            #[cfg(feature = "python-ffi")]
            {
                let py_lex = py_float_to_lex(value);
                let py_int = py_float_to_int(value);
                assert_eq!(rust_lex, py_lex, "Rust and PyO3 lex should match for {}", value);
                assert_eq!(rust_int, py_int, "Rust and PyO3 int should match for {}", value);
            }
            
            #[cfg(target_arch = "wasm32")]
            {
                let wasm_lex = wasm_float_to_lex(value);
                let wasm_int = wasm_float_to_int(value);
                assert_eq!(rust_lex, wasm_lex, "Rust and WASM lex should match for {}", value);
                assert_eq!(rust_int, wasm_int, "Rust and WASM int should match for {}", value);
            }
            
            // All interfaces should produce identical results
            assert_eq!(rust_lex, c_lex, "Rust and C FFI lex should match for {}", value);
            assert_eq!(rust_int, c_int, "Rust and C FFI int should match for {}", value);
            
            // Test decoding compatibility
            let rust_lex_decoded = lex_to_float(rust_lex);
            let c_lex_decoded = conjecture_lex_to_float(c_lex);
            
            if value.is_finite() {
                assert_eq!(value, rust_lex_decoded, "Rust lex round-trip should work for {}", value);
                assert_eq!(value, c_lex_decoded, "C FFI lex round-trip should work for {}", value);
            }
            
            println!("    All interfaces produce identical results: lex=0x{:016X}, int=0x{:016X}", rust_lex, rust_int);
        }
        
        println!("FFI_TESTS DEBUG: Cross-interface compatibility PASSED");
    }
}

#[cfg(test)]
mod performance_stress_tests {
    use super::*;

    /// Test FFI performance and stability under load
    #[test]
    fn test_ffi_performance_stress() {
        println!("FFI_TESTS DEBUG: Testing FFI performance and stability");

        let iterations = 10000;
        let test_values: Vec<f64> = (0..100)
            .map(|i| (i as f64 - 50.0) / 10.0)
            .collect();
        
        // Stress test Rust API
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            for &value in &test_values {
                let encoded = float_to_lex(value);
                let decoded = lex_to_float(encoded);
                if value.is_finite() {
                    assert_eq!(value, decoded, "Stress test failed for {}", value);
                }
            }
        }
        let rust_duration = start.elapsed();
        
        // Stress test C FFI
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            for &value in &test_values {
                let encoded = conjecture_float_to_lex(value);
                let decoded = conjecture_lex_to_float(encoded);
                if value.is_finite() {
                    assert_eq!(value, decoded, "C FFI stress test failed for {}", value);
                }
            }
        }
        let c_ffi_duration = start.elapsed();
        
        println!("    Rust API: {} iterations in {:?}", iterations * test_values.len(), rust_duration);
        println!("    C FFI: {} iterations in {:?}", iterations * test_values.len(), c_ffi_duration);
        
        // C FFI should not be significantly slower than Rust API
        let slowdown_ratio = c_ffi_duration.as_secs_f64() / rust_duration.as_secs_f64();
        assert!(slowdown_ratio < 2.0, "C FFI should not be more than 2x slower than Rust API");
        
        println!("FFI_TESTS DEBUG: Performance stress test PASSED (C FFI slowdown: {:.2}x)", slowdown_ratio);
    }

    /// Test memory safety under concurrent access
    #[test]
    fn test_ffi_concurrent_safety() {
        println!("FFI_TESTS DEBUG: Testing FFI concurrent safety");

        let test_values = vec![1.0, 2.0, 3.14159, -2.718281828, 1e-6, 1e6];
        let handles: Vec<std::thread::JoinHandle<()>> = (0..10)
            .map(|thread_id| {
                let values = test_values.clone();
                std::thread::spawn(move || {
                    for (i, &value) in values.iter().enumerate() {
                        // Test Rust API
                        let rust_encoded = float_to_lex(value);
                        let rust_decoded = lex_to_float(rust_encoded);
                        
                        // Test C FFI
                        let c_encoded = conjecture_float_to_lex(value);
                        let c_decoded = conjecture_lex_to_float(c_encoded);
                        
                        assert_eq!(rust_encoded, c_encoded, 
                                  "Thread {} iteration {} mismatch for {}", thread_id, i, value);
                        
                        if value.is_finite() {
                            assert_eq!(value, rust_decoded, 
                                      "Thread {} Rust round-trip failed for {}", thread_id, value);
                            assert_eq!(value, c_decoded,
                                      "Thread {} C FFI round-trip failed for {}", thread_id, value);
                        }
                    }
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }
        
        println!("FFI_TESTS DEBUG: Concurrent safety test PASSED");
    }
}

/// Comprehensive validation function for the entire FFI capability
pub fn validate_float_encoding_export_ffi() {
    println!("🔍 Validating Float Encoding Export FFI Capability...");
    
    // Test basic functionality
    let test_value = 3.14159;
    
    // Rust API
    let rust_lex = float_to_lex(test_value);
    let rust_decoded = lex_to_float(rust_lex);
    assert_eq!(test_value, rust_decoded, "Rust API should work");
    
    // C FFI
    let c_lex = conjecture_float_to_lex(test_value);
    let c_decoded = conjecture_lex_to_float(c_lex);
    assert_eq!(test_value, c_decoded, "C FFI should work");
    assert_eq!(rust_lex, c_lex, "Rust and C should match");
    
    // Width information
    assert_eq!(conjecture_float_width_bits(64), 64);
    assert_eq!(conjecture_float_width_mantissa_bits(64), 52);
    assert_eq!(conjecture_float_width_exponent_bits(64), 11);
    
    // Multi-width encoding
    for width in &[16, 32, 64] {
        let encoded = float_to_lex_multi_width(test_value, match width {
            16 => FloatWidth::Width16,
            32 => FloatWidth::Width32,
            64 => FloatWidth::Width64,
            _ => FloatWidth::Width64,
        });
        assert!(encoded > 0, "Multi-width encoding should work for width {}", width);
    }
    
    println!("✅ Float Encoding Export FFI Capability: VALIDATED!");
}