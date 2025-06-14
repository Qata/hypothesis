// Integration test to verify our sophisticated float generation works

#[cfg(test)]
mod float_integration_tests {
    use super::*;
    use crate::floats::{draw_float_for_provider, FloatWidth, float_to_lex, lex_to_float};
    use crate::choice::FloatConstraints;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_advanced_float_generation() {
        println!("Testing sophisticated float generation integration...");
        
        let mut rng = ChaCha8Rng::from_seed([42u8; 32]);
        let constraints = FloatConstraints {
            min_value: 0.0,
            max_value: 1.0,
            allow_nan: false,
            smallest_nonzero_magnitude: None,
        };
        
        // Generate several floats and verify they're in bounds
        for i in 0..5 {
            match draw_float_for_provider(&mut rng, &constraints) {
                Ok(value) => {
                    println!("  Generated sophisticated float {}: {}", i, value);
                    assert!(value >= 0.0 && value <= 1.0, "Float {} out of bounds: {}", i, value);
                    assert!(!value.is_nan(), "Float {} is NaN when NaN disabled", i);
                }
                Err(e) => panic!("Advanced float generation failed: {}", e),
            }
        }
    }

    #[test]
    fn test_provider_integration() {
        println!("Testing provider-based float generation...");
        
        let mut data = crate::data::ConjectureData::new(123);
        
        // Generate several floats through the provider system
        for i in 0..5 {
            match data.draw_float(0.0, 10.0, false, None, None, true) {
                Ok(value) => {
                    println!("  Provider generated float {}: {}", i, value);
                    assert!(value >= 0.0 && value <= 10.0, "Provider float {} out of bounds: {}", i, value);
                    assert!(!value.is_nan(), "Provider float {} is NaN when NaN disabled", i);
                }
                Err(e) => panic!("Provider float generation failed: {:?}", e),
            }
        }
    }

    #[test]
    fn test_lexicographic_properties() {
        println!("Testing lexicographic encoding roundtrip...");
        
        // Note: float_to_lex/lex_to_float are designed for magnitudes only
        // Negative numbers require sign handling at a higher level (e.g., draw_float)
        let test_values = [0.0, 1.0, 2.0, 0.5, std::f64::consts::PI];
        
        for &val in &test_values {
            let lex = float_to_lex(val, FloatWidth::Width64);
            let recovered = lex_to_float(lex, FloatWidth::Width64);
            println!("  {} -> lex: {} -> recovered: {}", val, lex, recovered);
            
            // Allow for floating point precision differences
            let diff = (val - recovered).abs();
            assert!(diff < 1e-10 || (val.is_nan() && recovered.is_nan()), 
                   "Roundtrip failed: {} != {} (diff: {})", val, recovered, diff);
        }
        
        // Test that negative values become positive (magnitude encoding)
        let neg_val = -1.0;
        let lex = float_to_lex(neg_val, FloatWidth::Width64);
        let recovered = lex_to_float(lex, FloatWidth::Width64);
        println!("  {} -> lex: {} -> recovered: {} (magnitude encoding)", neg_val, lex, recovered);
        assert_eq!(recovered, 1.0, "Negative values should encode as their absolute value");
    }

    #[test]
    fn test_constant_injection_probability() {
        println!("Testing constant injection works...");
        
        let mut rng = ChaCha8Rng::from_seed([1u8; 32]);
        let constraints = FloatConstraints {
            min_value: f64::NEG_INFINITY,
            max_value: f64::INFINITY,
            allow_nan: true,
            smallest_nonzero_magnitude: None,
        };
        
        let mut special_values_found = 0;
        let total_samples = 100;
        
        for _ in 0..total_samples {
            if let Ok(value) = draw_float_for_provider(&mut rng, &constraints) {
                // Check if this looks like a special constant (exact values from our constant pool)
                if value == 0.0 || value == 1.0 || value == -1.0 || value == 0.5 || 
                   value.is_infinite() || value.is_nan() {
                    special_values_found += 1;
                }
            }
        }
        
        // We expect constant injection to produce some special values
        // Should be roughly 15% injection rate, but allow for variance
        println!("  Found {} special values out of {} samples ({}%)", 
                special_values_found, total_samples, 
                (special_values_found as f64 / total_samples as f64) * 100.0);
        
        assert!(special_values_found > 0, "Expected some special constant values to be injected");
    }
}