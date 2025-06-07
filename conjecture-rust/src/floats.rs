// IEEE 754 floating point generation with lexicographic encoding.
// This module provides comprehensive float generation with multi-width support
// (16, 32, and 64-bit) and lexicographic ordering for excellent shrinking properties.

use crate::data::{DataSource, FailedDraw};
use half::f16;

type Draw<T> = Result<T, FailedDraw>;

// Float width enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatWidth {
    Width16,
    Width32, 
    Width64,
}

impl FloatWidth {
    pub fn bits(self) -> u32 {
        match self {
            FloatWidth::Width16 => 16,
            FloatWidth::Width32 => 32,
            FloatWidth::Width64 => 64,
        }
    }
    
    pub fn exponent_bits(self) -> u32 {
        match self {
            FloatWidth::Width16 => 5,
            FloatWidth::Width32 => 8,
            FloatWidth::Width64 => 11,
        }
    }
    
    pub fn mantissa_bits(self) -> u32 {
        match self {
            FloatWidth::Width16 => 10,
            FloatWidth::Width32 => 23,
            FloatWidth::Width64 => 52,
        }
    }
    
    pub fn bias(self) -> i32 {
        match self {
            FloatWidth::Width16 => 15,
            FloatWidth::Width32 => 127,
            FloatWidth::Width64 => 1023,
        }
    }
    
    pub fn max_exponent(self) -> u32 {
        (1 << self.exponent_bits()) - 1
    }
    
    pub fn mantissa_mask(self) -> u64 {
        (1u64 << self.mantissa_bits()) - 1
    }
}

// Generate exponent ordering key for lexicographic encoding.
// This function determines the order in which exponents should 
// appear to ensure lexicographic ordering matches numerical ordering.
fn exponent_key(e: u32, width: FloatWidth) -> f64 {
    let max_exp = width.max_exponent();
    if e == max_exp {
        f64::INFINITY
    } else {
        let unbiased = e as i32 - width.bias();
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
fn build_exponent_tables(width: FloatWidth) -> (Vec<u32>, Vec<u32>) {
    let max_exp = width.max_exponent();
    let mut exponents: Vec<u32> = (0..=max_exp).collect();
    exponents.sort_by(|&a, &b| exponent_key(a, width).partial_cmp(&exponent_key(b, width)).unwrap());
    
    let mut decoding_table = vec![0u32; (max_exp + 1) as usize];
    for (i, &exp) in exponents.iter().enumerate() {
        decoding_table[exp as usize] = i as u32;
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
fn update_mantissa(unbiased_exponent: i32, mantissa: u64, width: FloatWidth) -> u64 {
    let mantissa_bits = width.mantissa_bits();
    
    if unbiased_exponent <= 0 {
        reverse_bits(mantissa, mantissa_bits)
    } else if unbiased_exponent <= mantissa_bits as i32 - 1 {
        let n_fractional_bits = mantissa_bits - unbiased_exponent as u32;
        let fractional_part = mantissa & ((1 << n_fractional_bits) - 1);
        let integer_part = mantissa ^ fractional_part;
        integer_part | reverse_bits(fractional_part, n_fractional_bits)
    } else {
        mantissa
    }
}

// Convert lexicographically ordered integer to float of specified width.
// This implements a tagged union approach where the high bit determines
// whether we have a simple integer (0) or complex float (1).
pub fn lex_to_float_width(i: u64, width: FloatWidth) -> f64 {
    use std::sync::OnceLock;
    static ENCODING_TABLES_16: OnceLock<Vec<u32>> = OnceLock::new();
    static ENCODING_TABLES_32: OnceLock<Vec<u32>> = OnceLock::new();
    static ENCODING_TABLES_64: OnceLock<Vec<u32>> = OnceLock::new();
    
    // Get the appropriate encoding table
    let encoding_table = match width {
        FloatWidth::Width16 => ENCODING_TABLES_16.get_or_init(|| {
            let (table, _) = build_exponent_tables(width);
            table
        }),
        FloatWidth::Width32 => ENCODING_TABLES_32.get_or_init(|| {
            let (table, _) = build_exponent_tables(width);
            table
        }),
        FloatWidth::Width64 => ENCODING_TABLES_64.get_or_init(|| {
            let (table, _) = build_exponent_tables(width);
            table
        }),
    };
    
    let total_bits = width.bits();
    let exp_bits = width.exponent_bits();
    let mantissa_bits = width.mantissa_bits();
    let mantissa_mask = width.mantissa_mask();
    
    // Use appropriate bit for the tag based on width
    let tag_bit = total_bits - 1;
    let has_fractional_part = (i >> tag_bit) != 0;
    
    if has_fractional_part {
        let exponent_idx = (i >> mantissa_bits) & ((1 << exp_bits) - 1);
        let exponent = encoding_table[exponent_idx as usize];
        let mut mantissa = i & mantissa_mask;
        
        mantissa = update_mantissa(exponent as i32 - width.bias(), mantissa, width);
        
        // Construct the float bits
        let bits = ((exponent as u64) << mantissa_bits) | mantissa;
        
        // Convert to appropriate float type then to f64
        match width {
            FloatWidth::Width16 => {
                let f16_val = f16::from_bits(bits as u16);
                f16_val.to_f64()
            },
            FloatWidth::Width32 => {
                let f32_val = f32::from_bits(bits as u32);
                f32_val as f64
            },
            FloatWidth::Width64 => {
                f64::from_bits(bits)
            },
        }
    } else {
        // Simple integer encoding
        let max_simple_bits = total_bits - 8; // Leave room for tag and metadata
        let integral_part = i & ((1 << max_simple_bits) - 1);
        integral_part as f64
    }
}

// Check if float can be represented as simple integer for given width.
// Simple integers are non-negative, finite, have no fractional part,
// and fit in the available bits for the width.
fn is_simple_width(f: f64, width: FloatWidth) -> bool {
    if !f.is_finite() || f < 0.0 {
        return false;
    }
    let i = f as u64;
    if i as f64 != f {
        return false;
    }
    let max_simple_bits = width.bits() - 8; // Leave room for tag and metadata
    i.leading_zeros() >= (64 - max_simple_bits)
}

// Convert float to lexicographically ordered integer for specified width.
// Simple non-negative integers are encoded directly with tag bit 0.
// All other floats use the complex encoding with tag bit 1.
pub fn float_to_lex_width(f: f64, width: FloatWidth) -> u64 {
    if f >= 0.0 && is_simple_width(f, width) {
        return f as u64;
    }
    base_float_to_lex_width(f.abs(), width)
}

// Convert float to lexicographic encoding (internal implementation).
// This handles the complex encoding case where we need to properly
// order the exponent and apply mantissa transformations.
fn base_float_to_lex_width(f: f64, width: FloatWidth) -> u64 {
    use std::sync::OnceLock;
    static DECODING_TABLES_16: OnceLock<Vec<u32>> = OnceLock::new();
    static DECODING_TABLES_32: OnceLock<Vec<u32>> = OnceLock::new();
    static DECODING_TABLES_64: OnceLock<Vec<u32>> = OnceLock::new();
    
    // Get the appropriate decoding table
    let decoding_table = match width {
        FloatWidth::Width16 => DECODING_TABLES_16.get_or_init(|| {
            let (_, table) = build_exponent_tables(width);
            table
        }),
        FloatWidth::Width32 => DECODING_TABLES_32.get_or_init(|| {
            let (_, table) = build_exponent_tables(width);
            table
        }),
        FloatWidth::Width64 => DECODING_TABLES_64.get_or_init(|| {
            let (_, table) = build_exponent_tables(width);
            table
        }),
    };
    
    let mantissa_bits = width.mantissa_bits();
    let mantissa_mask = width.mantissa_mask();
    let exp_bits = width.exponent_bits();
    let tag_bit = width.bits() - 1;
    
    // Convert f64 to appropriate width first
    let bits = match width {
        FloatWidth::Width16 => {
            let f16_val = f16::from_f64(f);
            f16_val.to_bits() as u64
        },
        FloatWidth::Width32 => {
            let f32_val = f as f32;
            f32_val.to_bits() as u64
        },
        FloatWidth::Width64 => {
            f.to_bits()
        },
    };
    
    // Remove sign bit
    let bits = bits & !(1u64 << (width.bits() - 1));
    
    let exponent = (bits >> mantissa_bits) & ((1 << exp_bits) - 1);
    let mut mantissa = bits & mantissa_mask;
    
    mantissa = update_mantissa(exponent as i32 - width.bias(), mantissa, width);
    let encoded_exponent = decoding_table[exponent as usize] as u64;
    
    (1u64 << tag_bit) | (encoded_exponent << mantissa_bits) | mantissa
}

// Backward compatibility functions for existing f64-only API
pub fn lex_to_float(i: u64) -> f64 {
    lex_to_float_width(i, FloatWidth::Width64)
}

pub fn float_to_lex(f: f64) -> u64 {
    float_to_lex_width(f, FloatWidth::Width64)
}

// Special float constants for edge case generation (width-aware)
fn special_floats_for_width(width: FloatWidth) -> &'static [f64] {
    match width {
        FloatWidth::Width16 => &[
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            // f16 specific constants
            65504.0,  // f16::MAX
            6.103515625e-5,  // f16::MIN_POSITIVE
        ],
        FloatWidth::Width32 => &[
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            // f32 specific constants  
            3.4028235e38,  // f32::MAX
            1.1754944e-38, // f32::MIN_POSITIVE
            1.1920929e-7,  // f32::EPSILON
        ],
        FloatWidth::Width64 => &[
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
        ],
    }
}

// Generate a random float using lexicographic encoding with width support.
// This function provides the main entry point for width-aware float generation
// with full control over bounds and special value handling.
pub fn draw_float_width(
    source: &mut DataSource,
    width: FloatWidth,
    min_value: f64,
    max_value: f64,
    allow_nan: bool,
    allow_infinity: bool,
) -> Draw<f64> {
    let special_floats = special_floats_for_width(width);
    
    // 5% chance of returning special values
    if source.bits(6)? == 0 {
        // Try to return a special value that fits constraints
        for &special in special_floats {
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
    let raw_bits = source.bits(width.bits() as u64)?;
    let mut result = lex_to_float_width(raw_bits, width);
    
    // Apply random sign
    if source.bits(1)? == 1 {
        result = -result;
    }
    
    // Handle NaN
    if result.is_nan() && !allow_nan {
        // Fallback to generating a finite value
        let fallback_bits = source.bits(width.mantissa_bits() as u64 + 1)?;
        result = (fallback_bits as f64) / (1u64 << (width.mantissa_bits() + 1)) as f64;
        result = min_value + result * (max_value - min_value);
        return Ok(result);
    }
    
    // Handle infinity
    if result.is_infinite() && !allow_infinity {
        // Clamp to finite bounds for the width
        result = if result.is_sign_positive() {
            match width {
                FloatWidth::Width16 => 65504.0,    // f16::MAX
                FloatWidth::Width32 => 3.4028235e38, // f32::MAX  
                FloatWidth::Width64 => f64::MAX,
            }
        } else {
            match width {
                FloatWidth::Width16 => -65504.0,   // f16::MIN
                FloatWidth::Width32 => -3.4028235e38, // f32::MIN
                FloatWidth::Width64 => f64::MIN,
            }
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

// Backward compatibility function for f64 generation
pub fn draw_float(
    source: &mut DataSource,
    min_value: f64,
    max_value: f64,
    allow_nan: bool,
    allow_infinity: bool,
) -> Draw<f64> {
    draw_float_width(source, FloatWidth::Width64, min_value, max_value, allow_nan, allow_infinity)
}

// Bit-level reinterpretation utilities for float/integer conversion.
// These functions provide direct access to the underlying bit representations
// of floats without any encoding transformations.

// Convert float to raw integer bits for specified width.
// This provides direct access to the IEEE 754 representation.
pub fn float_to_int(value: f64, width: FloatWidth) -> u64 {
    match width {
        FloatWidth::Width16 => {
            let f16_val = f16::from_f64(value);
            f16_val.to_bits() as u64
        },
        FloatWidth::Width32 => {
            let f32_val = value as f32;
            f32_val.to_bits() as u64
        },
        FloatWidth::Width64 => {
            value.to_bits()
        },
    }
}

// Convert raw integer bits to float for specified width.
// This interprets the bits directly as IEEE 754 representation.
pub fn int_to_float(value: u64, width: FloatWidth) -> f64 {
    match width {
        FloatWidth::Width16 => {
            let f16_val = f16::from_bits(value as u16);
            f16_val.to_f64()
        },
        FloatWidth::Width32 => {
            let f32_val = f32::from_bits(value as u32);
            f32_val as f64
        },
        FloatWidth::Width64 => {
            f64::from_bits(value)
        },
    }
}

// Generic reinterpretation wrapper that handles conversion between
// different float formats. This is the main utility for bit-level operations.
pub fn reinterpret_bits(value: f64, from_width: FloatWidth, to_width: FloatWidth) -> f64 {
    if from_width == to_width {
        return value;
    }
    
    // Convert to bits in source format, then interpret in target format
    let bits = float_to_int(value, from_width);
    
    // For conversions between different widths, we need to handle bit layout differences
    match (from_width, to_width) {
        (FloatWidth::Width16, FloatWidth::Width32) => {
            // Expand f16 to f32: sign(1) + exp(5->8) + mantissa(10->23)
            let f16_bits = bits as u16;
            let sign = (f16_bits >> 15) as u32;
            let exp = ((f16_bits >> 10) & 0x1F) as u32;
            let mantissa = (f16_bits & 0x3FF) as u32;
            
            let f32_bits = if exp == 0x1F {
                // Special values (inf/nan)
                (sign << 31) | (0xFF << 23) | (mantissa << 13)
            } else if exp == 0 {
                // Zero or subnormal
                if mantissa == 0 {
                    sign << 31  // Zero
                } else {
                    // Convert subnormal to normal
                    let leading_zeros = mantissa.leading_zeros() - 22;
                    let normalized_mantissa = (mantissa << (leading_zeros + 1)) & 0x7FFFFF;
                    let normalized_exp = 127 - 15 - leading_zeros;
                    (sign << 31) | (normalized_exp << 23) | normalized_mantissa
                }
            } else {
                // Normal number
                let f32_exp = exp + 127 - 15;
                (sign << 31) | (f32_exp << 23) | (mantissa << 13)
            };
            
            f32::from_bits(f32_bits) as f64
        },
        (FloatWidth::Width16, FloatWidth::Width64) => {
            // Expand f16 to f64: similar to f16->f32 but with f64 layout
            let f16_val = f16::from_bits(bits as u16);
            f16_val.to_f64()
        },
        (FloatWidth::Width32, FloatWidth::Width16) => {
            // Compress f32 to f16: may lose precision
            let f32_val = f32::from_bits(bits as u32);
            let f16_val = f16::from_f32(f32_val);
            f16_val.to_f64()
        },
        (FloatWidth::Width32, FloatWidth::Width64) => {
            // Expand f32 to f64
            let f32_val = f32::from_bits(bits as u32);
            f32_val as f64
        },
        (FloatWidth::Width64, FloatWidth::Width16) => {
            // Compress f64 to f16: may lose precision
            let f16_val = f16::from_f64(value);
            f16_val.to_f64()
        },
        (FloatWidth::Width64, FloatWidth::Width32) => {
            // Compress f64 to f32: may lose precision
            (value as f32) as f64
        },
        // Same width cases (should not reach here due to early return)
        _ => value,
    }
}

// Generate a float from raw parts with width support.
pub fn draw_float_from_parts_width(source: &mut DataSource, width: FloatWidth) -> Draw<f64> {
    let raw_bits = source.bits(width.bits() as u64)?;
    Ok(lex_to_float_width(raw_bits, width))
}

// Backward compatibility function
pub fn draw_float_from_parts(source: &mut DataSource) -> Draw<f64> {
    draw_float_from_parts_width(source, FloatWidth::Width64)
}

// Draw a float with uniform distribution in range with width support.
pub fn draw_float_uniform_width(
    source: &mut DataSource, 
    width: FloatWidth,
    min_value: f64, 
    max_value: f64
) -> Draw<f64> {
    if min_value == max_value {
        return Ok(min_value);
    }
    
    let raw = source.bits(width.bits() as u64)?;
    let max_val = (1u64 << width.bits()) - 1;
    let fraction = (raw as f64) / (max_val as f64);
    let result = min_value + fraction * (max_value - min_value);
    
    Ok(result.max(min_value).min(max_value))
}

// Backward compatibility function
pub fn draw_float_uniform(source: &mut DataSource, min_value: f64, max_value: f64) -> Draw<f64> {
    draw_float_uniform_width(source, FloatWidth::Width64, min_value, max_value)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_width_constants() {
        assert_eq!(FloatWidth::Width16.bits(), 16);
        assert_eq!(FloatWidth::Width32.bits(), 32);
        assert_eq!(FloatWidth::Width64.bits(), 64);
        
        assert_eq!(FloatWidth::Width16.exponent_bits(), 5);
        assert_eq!(FloatWidth::Width32.exponent_bits(), 8);
        assert_eq!(FloatWidth::Width64.exponent_bits(), 11);
        
        assert_eq!(FloatWidth::Width16.mantissa_bits(), 10);
        assert_eq!(FloatWidth::Width32.mantissa_bits(), 23);
        assert_eq!(FloatWidth::Width64.mantissa_bits(), 52);
    }
    
    #[test]
    fn test_multi_width_exponent_encoding() {
        for width in [FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64] {
            let (encoding_table, decoding_table) = build_exponent_tables(width);
            let max_exp = width.max_exponent();
            
            for exp in 0..=max_exp {
                let encoded = decoding_table[exp as usize];
                let decoded = encoding_table[encoded as usize];
                assert_eq!(exp, decoded, "Exponent encoding roundtrip failed for width {:?}, exp {}", width, exp);
            }
        }
    }
    
    #[test]
    fn test_multi_width_roundtrip() {
        let test_values: [f64; 5] = [0.0, 1.0, 2.0, 0.5, 42.0];
        
        for width in [FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64] {
            for &val in &test_values {
                if val >= 0.0 && val.is_finite() {
                    let encoded = float_to_lex_width(val, width);
                    let decoded = lex_to_float_width(encoded, width);
                    
                    // Allow for precision loss in smaller widths
                    let tolerance = match width {
                        FloatWidth::Width16 => 1e-3,
                        FloatWidth::Width32 => 1e-6,
                        FloatWidth::Width64 => 0.0,
                    };
                    
                    assert!((val - decoded).abs() <= tolerance, 
                        "Roundtrip failed for width {:?}, val {}: {} -> {} -> {}", 
                        width, val, val, encoded, decoded);
                }
            }
        }
    }
    
    #[test]
    fn test_width_aware_generation() {
        use crate::data::DataSource;
        
        for width in [FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64] {
            let test_data: Vec<u64> = vec![42u64; 128];
            let mut source = DataSource::from_vec(test_data);
            
            // Test bounded generation
            for _ in 0..50 {
                let result = draw_float_width(&mut source, width, 0.0, 1.0, false, false);
                if let Ok(val) = result {
                    assert!(val >= 0.0 && val <= 1.0, "Generated value {} out of bounds [0,1] for width {:?}", val, width);
                    assert!(val.is_finite(), "Generated infinite value when not allowed for width {:?}", width);
                    assert!(!val.is_nan(), "Generated NaN when not allowed for width {:?}", width);
                }
            }
        }
    }
    
    #[test]
    fn test_simple_integer_detection_multi_width() {
        for width in [FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64] {
            // Test simple values that should work for all widths
            let simple_values: [f64; 4] = [0.0, 1.0, 2.0, 10.0];
            let non_simple_values: [f64; 5] = [-1.0, 0.5, f64::INFINITY, f64::NAN, 1.5];
            
            for &val in &simple_values {
                if val >= 0.0 && val.is_finite() {
                    let is_simple = is_simple_width(val, width);
                    // Simple integers should be detected correctly for reasonable values
                    if val <= 100.0 { // Small integers should be simple for all widths
                        assert!(is_simple || val.fract() != 0.0, 
                            "Small integer {} should be simple for width {:?}", val, width);
                    }
                }
            }
            
            for &val in &non_simple_values {
                assert!(!is_simple_width(val, width), 
                    "Value {} should not be simple for width {:?}", val, width);
            }
        }
    }
    
    #[test] 
    fn test_backward_compatibility() {
        // Ensure old f64-only functions still work
        let test_values: [f64; 5] = [0.0, 1.0, 2.0, 42.0, std::f64::consts::PI];
        
        for &val in &test_values {
            if val >= 0.0 && val.is_finite() {
                let encoded_old = float_to_lex(val);
                let encoded_new = float_to_lex_width(val, FloatWidth::Width64);
                assert_eq!(encoded_old, encoded_new, "Backward compatibility broken for {}", val);
                
                let decoded_old = lex_to_float(encoded_old);
                let decoded_new = lex_to_float_width(encoded_old, FloatWidth::Width64);
                assert_eq!(decoded_old, decoded_new, "Backward compatibility broken for decoding {}", val);
            }
        }
    }
    
    #[test]
    fn test_float_to_int_conversion() {
        let test_values: [f64; 6] = [0.0, 1.0, -1.0, 42.5, f64::INFINITY, f64::NAN];
        
        for width in [FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64] {
            for &val in &test_values {
                let bits = float_to_int(val, width);
                let reconstructed = int_to_float(bits, width);
                
                // Handle special cases
                if val.is_nan() {
                    assert!(reconstructed.is_nan(), "NaN not preserved through bit conversion for width {:?}", width);
                } else if val.is_infinite() {
                    assert_eq!(val.is_sign_positive(), reconstructed.is_sign_positive(), 
                        "Infinity sign not preserved for width {:?}", width);
                    assert!(reconstructed.is_infinite(), "Infinity not preserved for width {:?}", width);
                } else {
                    // Allow for precision loss in smaller widths
                    let tolerance = match width {
                        FloatWidth::Width16 => 1e-3,
                        FloatWidth::Width32 => 1e-6,
                        FloatWidth::Width64 => 0.0,
                    };
                    
                    assert!((val - reconstructed).abs() <= tolerance, 
                        "Float/int roundtrip failed for width {:?}, val {}: {} -> {} -> {}", 
                        width, val, val, bits, reconstructed);
                }
            }
        }
    }
    
    #[test]
    fn test_bit_reinterpretation_same_width() {
        let test_values: [f64; 4] = [0.0, 1.0, 42.5, std::f64::consts::PI];
        
        for width in [FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64] {
            for &val in &test_values {
                let result = reinterpret_bits(val, width, width);
                
                // Same width should preserve exact value (with precision limits)
                let tolerance = match width {
                    FloatWidth::Width16 => 1e-3,
                    FloatWidth::Width32 => 1e-6,
                    FloatWidth::Width64 => 0.0,
                };
                
                assert!((val - result).abs() <= tolerance, 
                    "Same-width reinterpretation changed value for width {:?}: {} -> {}", 
                    width, val, result);
            }
        }
    }
    
    #[test]
    fn test_bit_reinterpretation_cross_width() {
        let test_values: [f64; 4] = [0.0, 1.0, 2.0, 42.0];
        
        // Test f16 <-> f32 conversions
        for &val in &test_values {
            if val.is_finite() && val.abs() <= 65504.0 { // Within f16 range
                let f16_to_f32 = reinterpret_bits(val, FloatWidth::Width16, FloatWidth::Width32);
                let f32_to_f16 = reinterpret_bits(f16_to_f32, FloatWidth::Width32, FloatWidth::Width16);
                
                // Should roundtrip with some precision loss
                assert!((val - f32_to_f16).abs() <= 1e-3, 
                    "f16<->f32 roundtrip failed: {} -> {} -> {}", val, f16_to_f32, f32_to_f16);
            }
        }
        
        // Test f32 <-> f64 conversions
        for &val in &test_values {
            if val.is_finite() {
                let f32_to_f64 = reinterpret_bits(val, FloatWidth::Width32, FloatWidth::Width64);
                let f64_to_f32 = reinterpret_bits(f32_to_f64, FloatWidth::Width64, FloatWidth::Width32);
                
                // Should roundtrip with f32 precision
                assert!((val - f64_to_f32).abs() <= 1e-6, 
                    "f32<->f64 roundtrip failed: {} -> {} -> {}", val, f32_to_f64, f64_to_f32);
            }
        }
    }
    
    #[test]
    fn test_bit_reinterpretation_special_values() {
        // Test that special values are handled correctly across widths
        let special_values = [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY];
        
        for &val in &special_values {
            for from_width in [FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64] {
                for to_width in [FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64] {
                    let result = reinterpret_bits(val, from_width, to_width);
                    
                    if val == 0.0 {
                        assert_eq!(result, 0.0, "Zero not preserved in reinterpretation");
                    } else if val == -0.0 {
                        assert!(result == -0.0 || result == 0.0, "Negative zero handling failed");
                    } else if val.is_infinite() {
                        assert!(result.is_infinite(), "Infinity not preserved for {:?} -> {:?}", from_width, to_width);
                        assert_eq!(val.is_sign_positive(), result.is_sign_positive(), 
                            "Infinity sign not preserved for {:?} -> {:?}", from_width, to_width);
                    }
                }
            }
        }
    }
    
    #[test]
    fn test_raw_bit_access() {
        // Test that float_to_int and int_to_float provide raw bit access
        
        // Test known bit patterns
        let zero_bits = 0u64;
        let one_bits = 0x3FF0000000000000u64; // 1.0 in f64
        
        // f64 tests
        assert_eq!(int_to_float(zero_bits, FloatWidth::Width64), 0.0);
        assert_eq!(int_to_float(one_bits, FloatWidth::Width64), 1.0);
        assert_eq!(float_to_int(0.0, FloatWidth::Width64), zero_bits);
        assert_eq!(float_to_int(1.0, FloatWidth::Width64), one_bits);
        
        // f32 tests
        let f32_one_bits = 0x3F800000u64; // 1.0 in f32
        assert_eq!(int_to_float(f32_one_bits, FloatWidth::Width32), 1.0);
        assert_eq!(float_to_int(1.0, FloatWidth::Width32), f32_one_bits);
        
        // f16 tests  
        let f16_one_bits = 0x3C00u64; // 1.0 in f16
        assert_eq!(int_to_float(f16_one_bits, FloatWidth::Width16), 1.0);
        assert_eq!(float_to_int(1.0, FloatWidth::Width16), f16_one_bits);
    }
}