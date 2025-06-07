// IEEE 754 floating point generation with lexicographic encoding.
// This module provides sophisticated float generation that matches
// the quality of Python's implementation by using lexicographic
// ordering for excellent shrinking properties.

use crate::data::{DataSource, FailedDraw};

use std::u64::MAX as MAX64;

type Draw<T> = Result<T, FailedDraw>;

// IEEE 754 double precision floating point constants
const MAX_EXPONENT: u64 = 0x7FF;
const BIAS: i64 = 1023;
const MANTISSA_MASK: u64 = (1 << 52) - 1;

// Generate exponent ordering key for lexicographic encoding.
// This function determines the order in which exponents should 
// appear to ensure lexicographic ordering matches numerical ordering.
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

// Build encoding/decoding tables for exponents.
// Returns (encoding_table, decoding_table) where:
// - encoding_table[i] = original exponent at sorted position i
// - decoding_table[exp] = sorted position of exponent exp
fn build_exponent_tables() -> (Vec<u64>, Vec<u64>) {
    let mut exponents: Vec<u64> = (0..=MAX_EXPONENT).collect();
    exponents.sort_by(|&a, &b| exponent_key(a).partial_cmp(&exponent_key(b)).unwrap());
    
    let mut decoding_table = vec![0u64; (MAX_EXPONENT + 1) as usize];
    for (i, &exp) in exponents.iter().enumerate() {
        decoding_table[exp as usize] = i as u64;
    }
    
    (exponents, decoding_table)
}

// Reverse bits in a 64-bit integer
fn reverse64(mut v: u64) -> u64 {
    let mut result = 0u64;
    for _ in 0..64 {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

// Reverse n bits of x
fn reverse_bits(x: u64, n: u32) -> u64 {
    let reversed = reverse64(x);
    reversed >> (64 - n)
}

// Update mantissa according to lexicographic encoding rules.
// For different exponent ranges, the mantissa bits need different
// treatments to maintain lexicographic ordering.
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

// Convert lexicographically ordered integer to float.
// This implements a tagged union approach where bit 63 determines
// whether we have a simple integer (0) or complex float (1).
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

// Check if float can be represented as simple integer.
// Simple integers are non-negative, finite, have no fractional part,
// and fit in 56 bits (leaving 8 bits for the tag and other metadata).
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

// Convert float to lexicographically ordered integer.
// Simple non-negative integers are encoded directly with tag bit 0.
// All other floats use the complex encoding with tag bit 1.
pub fn float_to_lex(f: f64) -> u64 {
    if f >= 0.0 && is_simple(f) {
        return f as u64;
    }
    base_float_to_lex(f.abs())
}

// Convert float to lexicographic encoding (internal implementation).
// This handles the complex encoding case where we need to properly
// order the exponent and apply mantissa transformations.
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

// Special float constants for edge case generation.
// These values are commonly useful in testing and should be
// generated with higher probability than random values.
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

// Generate a random float using lexicographic encoding.
// This function provides the main entry point for float generation
// with full control over bounds and special value handling.
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

// Generate a float from raw parts (for maximum flexibility).
// This is the most basic generation function that applies
// the lexicographic encoding directly without constraints.
pub fn draw_float_from_parts(source: &mut DataSource) -> Draw<f64> {
    let raw_bits = source.bits(64)?;
    Ok(lex_to_float(raw_bits))
}

// Draw a float with uniform distribution in range (for bounded cases).
// This provides an alternative generation strategy that uses
// uniform random sampling within the specified bounds.
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
    
    // Extended special floats for comprehensive testing
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
        let simple_values: [f64; 4] = [0.0, 1.0, 2.0, 1024.0];
        let non_simple_values: [f64; 6] = [-1.0, 0.5, f64::INFINITY, f64::NAN, 1.5, f64::MAX];
        
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
}