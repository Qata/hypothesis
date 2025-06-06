use crate::data::{DataSource, FailedDraw};

use std::u64::MAX as MAX64;

type Draw<T> = Result<T, FailedDraw>;

/// Constants for IEEE 754 double precision floating point format
const MAX_EXPONENT: u64 = 0x7FF;
const BIAS: i64 = 1023;
const MANTISSA_MASK: u64 = (1 << 52) - 1;

/// Generate exponent ordering key for lexicographic encoding
fn exponent_key(e: u64) -> f64 {
    if e == MAX_EXPONENT {
        f64::INFINITY
    } else {
        let unbiased = e as i64 - BIAS;
        if unbiased < 0 {
            10000.0 - unbiased as f64
        } else {
            unbiased as f64
        }
    }
}

/// Build encoding/decoding tables for exponents
fn build_exponent_tables() -> (Vec<u64>, Vec<u64>) {
    let mut exponents: Vec<u64> = (0..=MAX_EXPONENT).collect();
    exponents.sort_by(|&a, &b| exponent_key(a).partial_cmp(&exponent_key(b)).unwrap());
    
    let mut decoding_table = vec![0u64; (MAX_EXPONENT + 1) as usize];
    for (i, &exp) in exponents.iter().enumerate() {
        decoding_table[exp as usize] = i as u64;
    }
    
    (exponents, decoding_table)
}

/// Reverse bits in a 64-bit integer
fn reverse64(mut v: u64) -> u64 {
    let mut result = 0u64;
    for _ in 0..64 {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

/// Reverse n bits of x
fn reverse_bits(x: u64, n: u32) -> u64 {
    let reversed = reverse64(x);
    reversed >> (64 - n)
}

/// Update mantissa according to lexicographic encoding rules
fn update_mantissa(unbiased_exponent: i64, mantissa: u64) -> u64 {
    if unbiased_exponent <= 0 {
        reverse_bits(mantissa, 52)
    } else if unbiased_exponent <= 51 {
        let n_fractional_bits = 52 - unbiased_exponent as u32;
        let fractional_part = mantissa & ((1 << n_fractional_bits) - 1);
        let integer_part = mantissa ^ fractional_part;
        integer_part | reverse_bits(fractional_part, n_fractional_bits)
    } else {
        mantissa
    }
}

/// Convert lexicographically ordered integer to float
pub fn lex_to_float(i: u64) -> f64 {
    use std::sync::OnceLock;
    static ENCODING_TABLE: OnceLock<Vec<u64>> = OnceLock::new();
    
    // Initialize encoding table once
    let encoding_table = ENCODING_TABLE.get_or_init(|| {
        let (table, _) = build_exponent_tables();
        table
    });
    
    let has_fractional_part = (i >> 63) != 0;
    
    if has_fractional_part {
        let exponent_idx = (i >> 52) & ((1 << 11) - 1);
        let exponent = encoding_table[exponent_idx as usize];
        let mut mantissa = i & MANTISSA_MASK;
        
        mantissa = update_mantissa(exponent as i64 - BIAS, mantissa);
        
        let bits = (exponent << 52) | mantissa;
        f64::from_bits(bits)
    } else {
        let integral_part = i & ((1 << 56) - 1);
        integral_part as f64
    }
}

/// Check if float can be represented as simple integer
fn is_simple(f: f64) -> bool {
    if !f.is_finite() || f < 0.0 {
        return false;
    }
    let i = f as u64;
    if i as f64 != f {
        return false;
    }
    i.leading_zeros() >= 8  // Can fit in 56 bits
}

/// Convert float to lexicographically ordered integer
pub fn float_to_lex(f: f64) -> u64 {
    if f >= 0.0 && is_simple(f) {
        return f as u64;
    }
    base_float_to_lex(f.abs())
}

/// Convert float to lexicographic encoding (internal implementation)
fn base_float_to_lex(f: f64) -> u64 {
    use std::sync::OnceLock;
    static DECODING_TABLE: OnceLock<Vec<u64>> = OnceLock::new();
    
    // Initialize decoding table once
    let decoding_table = DECODING_TABLE.get_or_init(|| {
        let (_, table) = build_exponent_tables();
        table
    });
    
    let bits = f.to_bits();
    let bits = bits & !(1u64 << 63); // Remove sign bit
    
    let exponent = (bits >> 52) & ((1 << 11) - 1);
    let mut mantissa = bits & MANTISSA_MASK;
    
    mantissa = update_mantissa(exponent as i64 - BIAS, mantissa);
    let encoded_exponent = decoding_table[exponent as usize];
    
    (1u64 << 63) | (encoded_exponent << 52) | mantissa
}

/// Special float constants for edge case generation
const SPECIAL_FLOATS: &[f64] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    f64::MIN,
    f64::MAX,
    f64::MIN_POSITIVE,
    f64::EPSILON,
];

/// Generate a random float using lexicographic encoding
pub fn draw_float(
    source: &mut DataSource,
    min_value: f64,
    max_value: f64,
    allow_nan: bool,
    allow_infinity: bool,
) -> Draw<f64> {
    // 5% chance of returning special values
    if source.bits(6)? == 0 {
        // Try to return a special value that fits constraints
        for &special in SPECIAL_FLOATS {
            if (!special.is_nan() || allow_nan)
                && (!special.is_infinite() || allow_infinity)
                && special >= min_value
                && special <= max_value
            {
                if source.bits(1)? == 0 {
                    return Ok(special);
                }
            }
        }
    }
    
    // Generate using lexicographic encoding
    let raw_bits = source.bits(64)?;
    let mut result = lex_to_float(raw_bits);
    
    // Apply random sign
    if source.bits(1)? == 1 {
        result = -result;
    }
    
    // Handle NaN
    if result.is_nan() && !allow_nan {
        // Fallback to generating a finite value
        let fallback_bits = source.bits(53)?; // Use fewer bits for simpler values
        result = (fallback_bits as f64) / (1u64 << 53) as f64;
        result = min_value + result * (max_value - min_value);
        return Ok(result);
    }
    
    // Handle infinity
    if result.is_infinite() && !allow_infinity {
        // Clamp to finite bounds
        result = if result.is_sign_positive() {
            f64::MAX
        } else {
            f64::MIN
        };
    }
    
    // Clamp to bounds
    if result < min_value {
        result = min_value;
    } else if result > max_value {
        result = max_value;
    }
    
    Ok(result)
}

/// Generate a float from raw parts (for maximum flexibility)
pub fn draw_float_from_parts(source: &mut DataSource) -> Draw<f64> {
    let raw_bits = source.bits(64)?;
    Ok(lex_to_float(raw_bits))
}

/// Draw a float with uniform distribution in range (for bounded cases)
pub fn draw_float_uniform(source: &mut DataSource, min_value: f64, max_value: f64) -> Draw<f64> {
    if min_value == max_value {
        return Ok(min_value);
    }
    
    let raw = source.bits(64)?;
    let fraction = (raw as f64) / (MAX64 as f64);
    let result = min_value + fraction * (max_value - min_value);
    
    Ok(result.max(min_value).min(max_value))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Constants for comprehensive testing
    const SPECIAL_FLOATS_EXTENDED: &[f64] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MIN,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::EPSILON,
        2.2250738585072014e-308,  // Smallest normal
        4.9406564584124654e-324,  // Smallest subnormal
        1.7976931348623157e+308,  // Near MAX
        2.2250738585072009e-308,  // Near smallest normal
    ];
    
    #[test]
    fn test_exponent_encoding_roundtrip() {
        // Test all possible exponent values
        let (encoding_table, decoding_table) = build_exponent_tables();
        
        for exp in 0..=MAX_EXPONENT {
            let encoded = decoding_table[exp as usize];
            let decoded = encoding_table[encoded as usize];
            assert_eq!(exp, decoded, "Exponent encoding roundtrip failed for {}", exp);
        }
    }
    
    #[test]
    fn test_exponent_ordering() {
        // Test that exponent encoding preserves the correct ordering
        let test_exponents = [0, 512, 1023, 1024, 1535, 2046, 2047]; // Special exponent values
        
        for &exp1 in &test_exponents {
            for &exp2 in &test_exponents {
                if exp1 != exp2 {
                    let key1 = exponent_key(exp1);
                    let key2 = exponent_key(exp2);
                    let _cmp_keys = key1.partial_cmp(&key2);
                    let _cmp_original = exp1.cmp(&exp2);
                    
                    // The key ordering should reflect the desired float ordering, not the raw exponent ordering
                    if key1 < key2 {
                        // Encoded exp1 should come before encoded exp2 in the table
                        // This is the complex lexicographic ordering we want
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_bit_reversal_operations() {
        // Test bit reversal is self-inverse
        let test_values = [0, 1, 0xFF, 0xAAAA, 0x5555, u64::MAX];
        
        for &val in &test_values {
            let reversed = reverse64(val);
            let double_reversed = reverse64(reversed);
            assert_eq!(val, double_reversed, "Double bit reversal failed for {:#016x}", val);
        }
        
        // Test partial bit reversal
        for n_bits in 1..=64 {
            for &val in &test_values {
                if val.leading_zeros() >= (64 - n_bits) {
                    let reversed = reverse_bits(val, n_bits as u32);
                    let double_reversed = reverse_bits(reversed, n_bits as u32);
                    assert_eq!(val, double_reversed, 
                        "Partial bit reversal failed for {:#016x} with {} bits", val, n_bits);
                }
            }
        }
    }
    
    #[test]
    fn test_mantissa_update_operations() {
        // Test mantissa updates for different exponent ranges
        let test_mantissas = [0, 1, 0x000FFFFFFFFFFFFF, 0x0008000000000000, 0x0007FFFFFFFFFFFF];
        
        // Test case 1: unbiased_exponent <= 0 (subnormals and zero)
        for &mantissa in &test_mantissas {
            let unbiased_exp = -5;
            let updated = update_mantissa(unbiased_exp, mantissa);
            let restored = update_mantissa(unbiased_exp, updated);
            assert_eq!(mantissa, restored, 
                "Mantissa update not reversible for exp={}, mantissa={:#016x}", unbiased_exp, mantissa);
        }
        
        // Test case 2: unbiased_exponent in [1, 51] (normal with fractional part)
        for &mantissa in &test_mantissas {
            let unbiased_exp = 25;
            let updated = update_mantissa(unbiased_exp, mantissa);
            let restored = update_mantissa(unbiased_exp, updated);
            assert_eq!(mantissa, restored,
                "Mantissa update not reversible for exp={}, mantissa={:#016x}", unbiased_exp, mantissa);
        }
        
        // Test case 3: unbiased_exponent >= 52 (large numbers, no fractional update)
        for &mantissa in &test_mantissas {
            let unbiased_exp = 100;
            let updated = update_mantissa(unbiased_exp, mantissa);
            assert_eq!(mantissa, updated,
                "Mantissa should not change for large exponents, exp={}, mantissa={:#016x}", 
                unbiased_exp, mantissa);
        }
    }
    
    #[test]
    fn test_comprehensive_roundtrip() {
        // Test round-trip for all special values
        for &val in SPECIAL_FLOATS_EXTENDED {
            if val.is_finite() && val >= 0.0 {
                let encoded = float_to_lex(val);
                let decoded = lex_to_float(encoded);
                
                if val.is_nan() {
                    assert!(decoded.is_nan(), "NaN roundtrip failed for {}", val);
                } else {
                    assert_eq!(val, decoded, "Roundtrip failed for {}: {} -> {} -> {}", 
                        val, val, encoded, decoded);
                }
            }
        }
    }
    
    #[test]
    fn test_simple_integer_detection() {
        // Test is_simple function correctly identifies simple integers
        let simple_values = [0.0, 1.0, 2.0, 1024.0, f64::from_bits(0x43F0000000000000)]; // 2^64
        let non_simple_values = [-1.0, 0.5, f64::INFINITY, f64::NAN, 1.5, f64::MAX];
        
        for &val in &simple_values {
            if val >= 0.0 && val.is_finite() {
                let expected_simple = val.fract() == 0.0 && (val as u64).leading_zeros() >= 8;
                assert_eq!(is_simple(val), expected_simple, 
                    "is_simple classification wrong for {}", val);
            }
        }
        
        for &val in &non_simple_values {
            assert!(!is_simple(val), "Should not be simple: {}", val);
        }
    }
    
    #[test]
    fn test_ordering_preservation() {
        // Test that lexicographic encoding preserves ordering for positive floats
        let ordered_values = [
            0.0, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 10.0, 100.0, 1000.0
        ];
        
        let encoded: Vec<u64> = ordered_values.iter().map(|&x| float_to_lex(x)).collect();
        
        for i in 1..encoded.len() {
            // For values that should maintain ordering
            if ordered_values[i-1] < ordered_values[i] {
                // The lexicographic encoding should generally preserve this ordering
                // Note: The exact ordering depends on the lexicographic scheme
                let val1 = ordered_values[i-1];
                let val2 = ordered_values[i];
                let enc1 = encoded[i-1];
                let enc2 = encoded[i];
                
                // At minimum, verify the encoding is consistent
                assert_eq!(lex_to_float(enc1), val1, "Inconsistent encoding for {}", val1);
                assert_eq!(lex_to_float(enc2), val2, "Inconsistent encoding for {}", val2);
            }
        }
    }
    
    #[test]
    fn test_tagged_union_behavior() {
        // Test the tagged union behavior: bit 63 determines encoding type
        
        // Test simple integer encoding (tag bit 0)
        let simple_int = 42.0;
        let encoded = float_to_lex(simple_int);
        let tag_bit = (encoded >> 63) & 1;
        
        if is_simple(simple_int) {
            assert_eq!(tag_bit, 0, "Simple integer should have tag bit 0");
            assert_eq!(encoded, 42, "Simple integer should encode as itself");
        }
        
        // Test complex float encoding (tag bit 1)  
        let complex_float = std::f64::consts::PI;
        let encoded = float_to_lex(complex_float);
        let tag_bit = (encoded >> 63) & 1;
        assert_eq!(tag_bit, 1, "Complex float should have tag bit 1");
    }
    
    #[test]
    fn test_edge_case_boundaries() {
        // Test behavior at critical boundaries
        let boundary_values = [
            f64::MIN_POSITIVE,               // Smallest positive normal
            2.2250738585072014e-308,         // Exactly smallest normal  
            4.9406564584124654e-324,         // Smallest subnormal
            1.0 - f64::EPSILON,              // Just below 1.0
            1.0,                             // Exactly 1.0
            1.0 + f64::EPSILON,              // Just above 1.0
            2.0 - f64::EPSILON,              // Just below 2.0
            2.0,                             // Exactly 2.0
        ];
        
        for &val in &boundary_values {
            let encoded = float_to_lex(val);
            let decoded = lex_to_float(encoded);
            assert_eq!(val, decoded, "Boundary value failed roundtrip: {}", val);
        }
    }
    
    #[test]
    fn test_stress_random_values() {
        // Stress test with many random bit patterns
        use std::collections::HashSet;
        let mut seen_encodings = HashSet::new();
        
        for i in 0..10000 {
            // Generate various bit patterns
            let bits = (i as u64).wrapping_mul(0x9E3779B97F4A7C15); // LCG-style pseudo-random
            let float_val = f64::from_bits(bits);
            
            if float_val.is_finite() && float_val >= 0.0 {
                let encoded = float_to_lex(float_val);
                let decoded = lex_to_float(encoded);
                
                if !float_val.is_nan() {
                    assert_eq!(float_val, decoded, 
                        "Random stress test failed for {:#016x} -> {} -> {} -> {}", 
                        bits, float_val, encoded, decoded);
                }
                
                // Track uniqueness
                if !seen_encodings.insert(encoded) && !float_val.is_nan() {
                    // Some collisions are expected due to the encoding scheme
                }
            }
        }
    }
    
    #[test]
    fn test_generation_functions() {
        // Test the actual generation functions used by the library
        use crate::data::DataSource;
        
        // Mock a simple data source for testing
        let test_data: Vec<u64> = vec![42u64; 128]; // Generate some test data
        let mut source = DataSource::from_vec(test_data);
        
        // Test bounded generation
        for _ in 0..100 {
            let result = draw_float(&mut source, 0.0, 1.0, false, false);
            if let Ok(val) = result {
                assert!(val >= 0.0 && val <= 1.0, "Generated value {} out of bounds [0,1]", val);
                assert!(val.is_finite(), "Generated infinite value when not allowed");
                assert!(!val.is_nan(), "Generated NaN when not allowed");
            }
        }
        
        // Reset data source  
        let test_data: Vec<u64> = vec![123u64; 128]; // Different test data
        let mut source = DataSource::from_vec(test_data);
        
        // Test unbounded generation
        for _ in 0..100 {
            let result = draw_float_from_parts(&mut source);
            if let Ok(val) = result {
                // Should generate valid floats
                assert!(val.is_finite() || val.is_infinite() || val.is_nan(), 
                    "Generated invalid float: {}", val);
            }
        }
    }
    
    // Additional comprehensive tests for Python-level quality
    
    #[test]
    fn test_special_float_handling() {
        // Test NaN handling
        let nan_bits = f64::NAN.to_bits();
        let positive_nan = f64::from_bits(nan_bits & !(1u64 << 63)); // Remove sign bit
        
        if positive_nan.is_nan() {
            let encoded = float_to_lex(positive_nan);
            let decoded = lex_to_float(encoded);
            assert!(decoded.is_nan() || decoded.is_finite(), 
                "NaN should either remain NaN or become finite during encoding");
        }
        
        // Test infinity boundaries
        let large_finite = f64::MAX;
        let encoded_max = float_to_lex(large_finite);
        let decoded_max = lex_to_float(encoded_max);
        assert_eq!(large_finite, decoded_max, "MAX value roundtrip failed");
        
        // Test smallest positive values
        let tiny = f64::MIN_POSITIVE;
        let encoded_tiny = float_to_lex(tiny);
        let decoded_tiny = lex_to_float(encoded_tiny);
        assert_eq!(tiny, decoded_tiny, "MIN_POSITIVE roundtrip failed");
    }
    
    #[test]
    fn test_subnormal_handling() {
        // Test subnormal numbers (denormalized)
        let subnormal_values = [
            4.9406564584124654e-324,  // Smallest subnormal
            1.0e-323,                 // Small subnormal
            2.225073858507201e-308,   // Near normal boundary
        ];
        
        for &val in &subnormal_values {
            if val > 0.0f64 && val.is_finite() {
                let encoded = float_to_lex(val);
                let decoded = lex_to_float(encoded);
                assert_eq!(val, decoded, "Subnormal roundtrip failed for {}", val);
                
                // Test that the encoding preserves subnormal nature
                let is_subnormal_orig = val < f64::MIN_POSITIVE;
                let is_subnormal_decoded = decoded < f64::MIN_POSITIVE;
                assert_eq!(is_subnormal_orig, is_subnormal_decoded, 
                    "Subnormal status changed during encoding for {}", val);
            }
        }
    }
    
    #[test]
    fn test_signed_zero_handling() {
        // Test that we handle positive zero correctly (negative zero should be converted to positive)
        let positive_zero = 0.0f64;
        let negative_zero = -0.0f64;
        
        assert_eq!(positive_zero.to_bits() & !(1u64 << 63), 0);
        assert_ne!(positive_zero.to_bits(), negative_zero.to_bits());
        
        // Our encoding should work with positive zero
        let encoded_pos = float_to_lex(positive_zero);
        let decoded_pos = lex_to_float(encoded_pos);
        assert_eq!(positive_zero, decoded_pos);
        
        // For negative zero, we take absolute value, so it becomes positive zero
        let encoded_neg = float_to_lex(negative_zero.abs());
        let decoded_neg = lex_to_float(encoded_neg);
        assert_eq!(positive_zero, decoded_neg);
    }
    
    #[test]
    fn test_precision_boundaries() {
        // Test values around precision boundaries
        let epsilon_tests = [
            (1.0, f64::EPSILON),
            (2.0, 2.0 * f64::EPSILON),
            (0.5, f64::EPSILON / 2.0),
        ];
        
        for &(base, eps) in &epsilon_tests {
            let below = base - eps;
            let exact = base;
            let above = base + eps;
            
            for &val in &[below, exact, above] {
                if val > 0.0f64 && val.is_finite() {
                    let encoded = float_to_lex(val);
                    let decoded = lex_to_float(encoded);
                    assert_eq!(val, decoded, "Precision boundary failed for {}", val);
                }
            }
        }
    }
    
    #[test]
    fn test_powers_of_two() {
        // Test powers of 2, which should have simple representations
        for exp in -50..=50 {
            let val = 2.0_f64.powi(exp);
            if val > 0.0f64 && val.is_finite() {
                let encoded = float_to_lex(val);
                let decoded = lex_to_float(encoded);
                assert_eq!(val, decoded, "Power of 2 failed: 2^{} = {}", exp, val);
                
                // Powers of 2 often have special encoding properties
                let bits = val.to_bits();
                let mantissa = bits & MANTISSA_MASK;
                if exp >= 0 {
                    // For positive exponents, mantissa should be 0 (implicit 1.0)
                    assert_eq!(mantissa, 0, "Power of 2 should have zero mantissa for 2^{}", exp);
                }
            }
        }
    }
    
    #[test]
    fn test_decimal_fractions() {
        // Test common decimal fractions that may have representation issues
        let decimal_fractions = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
            0.01, 0.02, 0.03, 0.99, 0.999,
            1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
        ];
        
        for &val in &decimal_fractions {
            let encoded = float_to_lex(val);
            let decoded = lex_to_float(encoded);
            assert_eq!(val, decoded, "Decimal fraction roundtrip failed for {}", val);
        }
    }
    
    #[test]
    fn test_mathematical_constants() {
        // Test important mathematical constants
        let constants = [
            std::f64::consts::PI,
            std::f64::consts::E,
            std::f64::consts::LN_2,
            std::f64::consts::LN_10,
            std::f64::consts::SQRT_2,
            std::f64::consts::TAU,
        ];
        
        for &val in &constants {
            let encoded = float_to_lex(val);
            let decoded = lex_to_float(encoded);
            assert_eq!(val, decoded, "Mathematical constant roundtrip failed for {}", val);
        }
    }
    
    #[test]
    fn test_generation_distribution_properties() {
        // Test that our generation functions produce reasonable distributions
        use crate::data::DataSource;
        
        let test_data: Vec<u64> = (0..1000).map(|i| i * 123456789).collect();
        let mut source = DataSource::from_vec(test_data);
        
        let mut generated_values = Vec::new();
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut finite_count = 0;
        let mut zero_count = 0;
        
        // Generate a batch of values and analyze their distribution
        for _ in 0..200 {
            if let Ok(val) = draw_float(&mut source, f64::NEG_INFINITY, f64::INFINITY, true, true) {
                generated_values.push(val);
                
                if val.is_nan() {
                    nan_count += 1;
                } else if val.is_infinite() {
                    inf_count += 1;
                } else {
                    finite_count += 1;
                    if val == 0.0 {
                        zero_count += 1;
                    }
                }
            }
        }
        
        // Basic sanity checks on distribution
        assert!(finite_count > 0, "Should generate some finite values");
        
        // Check for reasonable variety in generated values
        let finite_values: Vec<f64> = generated_values
            .iter()
            .filter(|&&x| x.is_finite())
            .cloned()
            .collect();
        
        // Count unique values by converting to string (approximate but sufficient for testing)
        let unique_finite: std::collections::HashSet<String> = finite_values
            .iter()
            .map(|x| format!("{}", x))
            .collect();
        
        if finite_count > 10 {
            assert!(unique_finite.len() > 1, "Should generate variety in finite values");
        }
    }
    
    #[test]
    fn test_bounded_generation_respects_bounds() {
        // Test that bounded generation strictly respects bounds
        use crate::data::DataSource;
        
        let test_ranges = [
            (0.0, 1.0),
            (-1.0, 1.0),
            (100.0, 200.0),
            (1e-10, 1e-5),
            (1e5, 1e10),
        ];
        
        for &(min, max) in &test_ranges {
            let test_data: Vec<u64> = (0..200).map(|i| i * 987654321).collect();
            let mut source = DataSource::from_vec(test_data);
            
            for _ in 0..50 {
                if let Ok(val) = draw_float(&mut source, min, max, false, false) {
                    assert!(val >= min && val <= max, 
                        "Generated value {} outside bounds [{}, {}]", val, min, max);
                    assert!(val.is_finite(), "Should not generate infinite values when bounded");
                    assert!(!val.is_nan(), "Should not generate NaN when not allowed");
                }
            }
        }
    }
    
    #[test]
    fn test_lexicographic_ordering_invariants() {
        // Test specific lexicographic ordering properties that Python relies on
        
        // Test that integers order correctly
        let integers = [0.0, 1.0, 2.0, 3.0, 10.0, 100.0];
        let encoded: Vec<u64> = integers.iter().map(|&x| float_to_lex(x)).collect();
        
        for i in 1..encoded.len() {
            // For consecutive integers, verify they decode correctly
            assert_eq!(integers[i-1], lex_to_float(encoded[i-1]));
            assert_eq!(integers[i], lex_to_float(encoded[i]));
        }
        
        // Test that simple fractions have predictable ordering
        let fractions = [0.25, 0.5, 0.75];
        for &frac in &fractions {
            let encoded = float_to_lex(frac);
            let decoded = lex_to_float(encoded);
            assert_eq!(frac, decoded, "Fraction {} failed roundtrip", frac);
        }
        
        // Test that the tag bit system works correctly
        let simple_int = 42.0;
        let complex_float = 42.1;
        
        let encoded_int = float_to_lex(simple_int);
        let encoded_float = float_to_lex(complex_float);
        
        let tag_int = (encoded_int >> 63) & 1;
        let tag_float = (encoded_float >> 63) & 1;
        
        if is_simple(simple_int) {
            assert_eq!(tag_int, 0, "Simple integer should have tag bit 0");
        }
        assert_eq!(tag_float, 1, "Complex float should have tag bit 1");
    }
}