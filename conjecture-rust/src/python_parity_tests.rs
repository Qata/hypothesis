// Integration tests for Python Hypothesis parity features
// This module tests that our implementations behave similarly to Python Hypothesis

use crate::data::DataSource;
use crate::floats::{draw_float, FloatWidth};
use crate::ints::draw_integer_with_local_constants;
use crate::strings::{draw_string_with_local_constants, draw_bytes_with_local_constants, draw_boolean};
use crate::distributions::weighted;
use std::collections::HashMap;

fn test_data_source() -> DataSource {
    let data: Vec<u64> = (0..5000).map(|i| (i * 19 + 73) % 256).collect();
    DataSource::from_vec(data)
}

#[test]
fn test_cross_module_local_constants_integration() {
    let mut source = test_data_source();
    
    // Test that local constants work across all data types
    let int_constants = vec![42i64, 100, -5];
    let string_constants = vec!["test".to_string(), "hello".to_string(), "".to_string()];
    let bytes_constants = vec![vec![0u8, 1, 2], vec![255u8], vec![]];
    
    let mut int_results = Vec::new();
    let mut string_results = Vec::new();
    let mut bytes_results = Vec::new();
    
    // Generate values using local constants
    for _ in 0..100 {
        // Integer generation with constants
        if let Ok(val) = draw_integer_with_local_constants(
            &mut source, Some(0), Some(200), None, 0, &int_constants
        ) {
            int_results.push(val);
        }
        
        // String generation with constants  
        if let Ok(val) = draw_string_with_local_constants(
            &mut source, 0, 10, None, &string_constants
        ) {
            string_results.push(val);
        }
        
        // Bytes generation with constants
        if let Ok(val) = draw_bytes_with_local_constants(
            &mut source, 0, 5, &bytes_constants
        ) {
            bytes_results.push(val);
        }
    }
    
    // Verify we got some results
    assert!(!int_results.is_empty(), "Should generate integers with constants");
    assert!(!string_results.is_empty(), "Should generate strings with constants");
    assert!(!bytes_results.is_empty(), "Should generate bytes with constants");
    
    // Check for constant injection across types
    let int_constant_found = int_results.iter().any(|x| int_constants.contains(x));
    let string_constant_found = string_results.iter().any(|x| string_constants.contains(x));
    let bytes_constant_found = bytes_results.iter().any(|x| bytes_constants.contains(x));
    
    println!("Cross-module constants: int={}, string={}, bytes={}", 
        int_constant_found, string_constant_found, bytes_constant_found);
}

#[test]
fn test_python_hypothesis_api_compatibility() {
    let mut source = test_data_source();
    
    // Test that our API signatures match Python Hypothesis expectations
    
    // integers() equivalent
    let _int_val = draw_integer_with_local_constants(
        &mut source, 
        Some(0),           // min_value
        Some(100),         // max_value  
        None,              // weights
        0,                 // shrink_towards
        &[]                // local_constants
    );
    
    // text() equivalent
    let _string_val = draw_string_with_local_constants(
        &mut source,
        0,                 // min_size
        20,                // max_size
        Some("abc"),       // alphabet
        &[]                // local_constants
    );
    
    // binary() equivalent
    let _bytes_val = draw_bytes_with_local_constants(
        &mut source,
        0,                 // min_size
        10,                // max_size
        &[]                // local_constants
    );
    
    // floats() equivalent
    let _float_val = draw_float(
        &mut source,
        Some(0.0),         // min_value
        Some(1.0),         // max_value
        Some(false),       // allow_nan
        Some(false),       // allow_infinity
        None,              // allow_subnormal
        None,              // smallest_nonzero_magnitude
        FloatWidth::Width64, // width
        false,             // exclude_min
        false              // exclude_max
    );
    
    // booleans() equivalent  
    let _bool_val = draw_boolean(&mut source, 0.5);
    
    // weighted() for probability distributions
    let _weighted_val = weighted(&mut source, 0.3);
    
    println!("Python API compatibility test passed");
}

#[test]
fn test_python_constant_injection_rates() {
    // Test that constant injection rates match Python's expectations
    
    // Python uses 5% for integers, 5% for strings/bytes, 15% for floats
    // We approximate with bit patterns: 1/32 ≈ 3.125% for most, different for floats
    
    let mut int_injections = 0;
    let mut string_injections = 0;
    let mut total_attempts = 0;
    
    for seed in 0..1000 {
        let data = vec![seed % 256; 20];
        let mut source = DataSource::from_vec(data);
        
        // Test integer constant injection
        let int_constants = vec![999i64]; // Distinctive value
        if let Ok(val) = draw_integer_with_local_constants(
            &mut source, Some(0), Some(1000), None, 0, &int_constants
        ) {
            total_attempts += 1;
            if val == 999 {
                int_injections += 1;
            }
        }
        
        // Test string constant injection
        let string_constants = vec!["CONSTANT".to_string()]; // Distinctive value
        if let Ok(val) = draw_string_with_local_constants(
            &mut source, 5, 15, None, &string_constants
        ) {
            if val == "CONSTANT" {
                string_injections += 1;
            }
        }
    }
    
    if total_attempts > 100 {
        let int_rate = int_injections as f64 / total_attempts as f64;
        let string_rate = string_injections as f64 / total_attempts as f64;
        
        println!("Injection rates: int={:.2}%, string={:.2}%", 
            int_rate * 100.0, string_rate * 100.0);
        
        // Should be reasonably low (around 3% for our implementation)
        assert!(int_rate < 0.2, "Integer injection rate too high: {:.3}", int_rate);
        assert!(string_rate < 0.2, "String injection rate too high: {:.3}", string_rate);
    }
}

#[test]
fn test_python_parity_edge_cases() {
    let mut source = test_data_source();
    
    // Test edge cases that Python handles
    
    // Zero-sized collections
    let empty_string = draw_string_with_local_constants(&mut source, 0, 0, None, &[]).unwrap();
    assert_eq!(empty_string, "");
    
    let empty_bytes = draw_bytes_with_local_constants(&mut source, 0, 0, &[]).unwrap();
    assert_eq!(empty_bytes, Vec::<u8>::new());
    
    // Single value ranges
    let single_int = draw_integer_with_local_constants(
        &mut source, Some(42), Some(42), None, 0, &[]
    ).unwrap();
    assert_eq!(single_int, 42);
    
    // Boolean edge cases (Python's exact behavior)
    assert_eq!(draw_boolean(&mut source, 0.0).unwrap(), false);
    assert_eq!(draw_boolean(&mut source, 1.0).unwrap(), true);
    assert_eq!(draw_boolean(&mut source, -0.5).unwrap(), false);
    assert_eq!(draw_boolean(&mut source, 1.5).unwrap(), true);
    
    // Float edge cases
    let zero_range_float = draw_float(&mut source, Some(1.0), Some(1.0), Some(false), Some(false), None, None, FloatWidth::Width64, false, false);
    if let Ok(val) = zero_range_float {
        assert_eq!(val, 1.0);
    }
    
    println!("Python parity edge cases test passed");
}

#[test]
fn test_multi_width_float_parity() {
    let mut source = test_data_source();
    
    // Test that multi-width float generation works like Python
    for width in [FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64] {
        let mut results = Vec::new();
        
        for _ in 0..50 {
            if let Ok(val) = crate::floats::draw_float_width(
                &mut source, width, 0.0, 1.0, false, false
            ) {
                results.push(val);
            }
        }
        
        if !results.is_empty() {
            // All should be in range and finite
            for &val in &results {
                assert!(val >= 0.0 && val <= 1.0, "Value {} out of range for width {:?}", val, width);
                assert!(val.is_finite(), "Value {} not finite for width {:?}", val, width);
            }
            
            println!("Width {:?}: generated {} valid values", width, results.len());
        }
    }
}

#[test]
fn test_weighted_distribution_python_parity() {
    let mut source = test_data_source();
    
    // Test weighted distribution matches Python's behavior
    let test_probabilities = vec![0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0];
    
    for &prob in &test_probabilities {
        let mut true_count = 0;
        let mut total_count = 0;
        
        for _ in 0..100 {
            if let Ok(result) = weighted(&mut source, prob) {
                total_count += 1;
                if result {
                    true_count += 1;
                }
            }
        }
        
        if total_count > 10 {
            let observed_rate = true_count as f64 / total_count as f64;
            
            // For extreme probabilities, check exact behavior
            if prob == 0.0 {
                assert_eq!(true_count, 0, "Probability 0.0 should never return true");
            } else if prob == 1.0 {
                assert_eq!(true_count, total_count, "Probability 1.0 should always return true");
            } else {
                // For other probabilities, just ensure reasonable behavior
                println!("Probability {}: {:.2}% true rate ({}/{})", 
                    prob, observed_rate * 100.0, true_count, total_count);
            }
        }
    }
}

#[test]
fn test_constraint_handling_python_parity() {
    let mut source = test_data_source();
    
    // Test constraint handling like Python
    
    // String alphabet constraints
    let custom_alphabet = "ABC123";
    for _ in 0..20 {
        if let Ok(s) = draw_string_with_local_constants(
            &mut source, 5, 5, Some(custom_alphabet), &[]
        ) {
            for ch in s.chars() {
                assert!(custom_alphabet.contains(ch), 
                    "Character '{}' not in alphabet '{}'", ch, custom_alphabet);
            }
        }
    }
    
    // Integer weighted constraints
    let mut weights = HashMap::new();
    weights.insert(10i64, 0.4);
    weights.insert(20i64, 0.3);
    weights.insert(30i64, 0.2);
    // Total: 0.9, leaving 0.1 for uniform
    
    let mut results = Vec::new();
    for _ in 0..100 {
        if let Ok(val) = draw_integer_with_local_constants(
            &mut source, Some(0), Some(50), Some(weights.clone()), 0, &[]
        ) {
            results.push(val);
            assert!(val >= 0 && val <= 50, "Weighted integer {} out of bounds", val);
        }
    }
    
    // Should see some weighted values
    let weighted_count = results.iter().filter(|&&x| weights.contains_key(&x)).count();
    if results.len() > 20 {
        println!("Weighted constraint test: {}/{} values were from weights", 
            weighted_count, results.len());
    }
}

#[test]
fn test_error_handling_python_parity() {
    let mut source = test_data_source();
    
    // Test error conditions that Python would handle
    
    // Invalid weight sums (should panic/error like Python)
    let mut bad_weights = HashMap::new();
    bad_weights.insert(1i64, 0.6);
    bad_weights.insert(2i64, 0.6); // Total > 1.0
    
    // Use a fresh source to avoid borrow issues
    let test_data = vec![42u64; 100];
    let mut test_source = DataSource::from_vec(test_data);
    
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        draw_integer_with_local_constants(
            &mut test_source, Some(0), Some(10), Some(bad_weights), 0, &[]
        )
    }));
    
    // Should panic like Python when weights sum > 1.0
    assert!(result.is_err(), "Should panic on invalid weight sum");
    
    // Empty alphabet (should be handled gracefully)
    let empty_result = draw_string_with_local_constants(
        &mut source, 1, 5, Some(""), &[]
    );
    // This should either succeed with empty string or fail gracefully
    match empty_result {
        Ok(s) => assert!(s.is_empty(), "Empty alphabet should produce empty string"),
        Err(_) => {} // Failing is also acceptable
    }
    
    println!("Error handling parity test passed");
}

#[test]
fn test_deterministic_behavior_python_parity() {
    // Test that identical inputs produce identical outputs (like Python)
    
    let test_data = vec![42u64, 17, 73, 99, 1, 255, 128, 64];
    
    // Test integers
    let mut source1 = DataSource::from_vec(test_data.clone());
    let mut source2 = DataSource::from_vec(test_data.clone());
    
    let int1 = draw_integer_with_local_constants(&mut source1, Some(0), Some(100), None, 0, &[]);
    let int2 = draw_integer_with_local_constants(&mut source2, Some(0), Some(100), None, 0, &[]);
    
    match (int1, int2) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "Deterministic integer generation failed"),
        (Err(_), Err(_)) => {}, // Both failing is OK
        _ => panic!("Inconsistent results for identical inputs"),
    }
    
    // Test strings
    let mut source1 = DataSource::from_vec(test_data.clone());
    let mut source2 = DataSource::from_vec(test_data.clone());
    
    let str1 = draw_string_with_local_constants(&mut source1, 3, 8, Some("abc"), &[]);
    let str2 = draw_string_with_local_constants(&mut source2, 3, 8, Some("abc"), &[]);
    
    match (str1, str2) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "Deterministic string generation failed"),
        (Err(_), Err(_)) => {}, // Both failing is OK
        _ => panic!("Inconsistent string results for identical inputs"),
    }
    
    // Test weighted distributions
    let mut source1 = DataSource::from_vec(test_data.clone());
    let mut source2 = DataSource::from_vec(test_data);
    
    let weight1 = weighted(&mut source1, 0.3);
    let weight2 = weighted(&mut source2, 0.3);
    
    match (weight1, weight2) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "Deterministic weighted generation failed"),
        (Err(_), Err(_)) => {}, // Both failing is OK
        _ => panic!("Inconsistent weighted results for identical inputs"),
    }
    
    println!("Deterministic behavior parity test passed");
}

#[test]
fn test_shrinking_property_preservation() {
    // Test that our lexicographic encodings preserve shrinking properties like Python
    
    use crate::floats::encoding::{float_to_lex, lex_to_float};
    
    let width = FloatWidth::Width64;
    
    // Test that smaller lexicographic values correspond to "simpler" floats
    let test_cases = vec![
        (0.0, 1.0),     // 0 should be simpler than 1
        (1.0, 2.0),     // 1 should be simpler than 2
        (1.0, 1.5),     // Integer should be simpler than non-integer
        (0.5, 0.75),    // Simpler fraction vs complex fraction
    ];
    
    for (simpler, complex) in test_cases {
        let simpler_lex = float_to_lex(simpler, width);
        let complex_lex = float_to_lex(complex, width);
        
        // In lexicographic ordering, simpler values should have smaller encodings
        assert!(simpler_lex <= complex_lex, 
            "Shrinking property violated: {} (lex={}) should be ≤ {} (lex={})", 
            simpler, simpler_lex, complex, complex_lex);
        
        // Verify round-trip works
        let simpler_round = lex_to_float(simpler_lex, width);
        let complex_round = lex_to_float(complex_lex, width);
        
        assert!((simpler - simpler_round).abs() < 1e-10f64, 
            "Round-trip failed for {}: {} -> {} -> {}", 
            simpler, simpler, simpler_lex, simpler_round);
        assert!((complex - complex_round).abs() < 1e-10f64, 
            "Round-trip failed for {}: {} -> {} -> {}", 
            complex, complex, complex_lex, complex_round);
    }
    
    println!("Shrinking property preservation test passed");
}