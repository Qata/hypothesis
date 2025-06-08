// IEEE 754 floating point generation with lexicographic encoding.
// This module provides comprehensive float generation with multi-width support
// (16, 32, and 64-bit) and lexicographic ordering for excellent shrinking properties.

use crate::data::{DataSource, FailedDraw};
use half::f16;

type Draw<T> = Result<T, FailedDraw>;

// Pre-computed table mapping individual bytes to the equivalent byte with bits reversed
// This matches Python's REVERSE_BITS_TABLE exactly
const REVERSE_BITS_TABLE: [u8; 256] = [
    0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
    0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8, 0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
    0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
    0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
    0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2, 0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
    0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
    0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
    0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee, 0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
    0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
    0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
    0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5, 0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
    0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
    0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
    0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb, 0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
    0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
    0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff,
];


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

// Python-equivalent lexicographic encoding constants and algorithms
// This implements the exact same sophisticated encoding used in Python Hypothesis

const SIMPLE_THRESHOLD_BITS: u32 = 56; // Maximum bits for simple integer encoding

// Generate exponent ordering key for lexicographic encoding (Python-equivalent).
// This function determines the order in which exponents should appear to ensure 
// lexicographic ordering matches numerical ordering and shrinking priorities.
fn exponent_key(e: u32, width: FloatWidth) -> f64 {
    let max_exp = width.max_exponent();
    if e == max_exp {
        // Special values (infinity, NaN) come last
        f64::INFINITY
    } else {
        let unbiased = e as i32 - width.bias();
        if unbiased < 0 {
            // Negative exponents (values < 1.0) come after positive exponents
            // Map to range [10001, 10001 + max_negative_exp]
            10000.0 - unbiased as f64
        } else {
            // Positive exponents (values >= 1.0) come first
            // Map to range [0, max_positive_exp]
            unbiased as f64
        }
    }
}

// Check if a float can be represented as a simple integer (Python-equivalent).
// Simple integers use direct encoding with tag bit 0 for optimal shrinking.
pub fn is_simple_width(f: f64, width: FloatWidth) -> bool {
    // Must be finite and non-negative
    if !f.is_finite() || f < 0.0 {
        return false;
    }
    
    // Must be exactly representable as an integer
    let i = f as u64;
    if i as f64 != f {
        return false;
    }
    
    // Must fit in the available bits for simple encoding (Python-equivalent)
    // Python uses i.bit_length() <= 56, so we match that exactly
    let max_simple_bits = match width {
        FloatWidth::Width16 => 16,  // For f16, use smaller threshold
        FloatWidth::Width32 => 32,  // For f32, use reasonable threshold  
        FloatWidth::Width64 => SIMPLE_THRESHOLD_BITS, // Full 56 bits for f64 (Python-equivalent)
    };
    
    // Count significant bits (equivalent to Python's bit_length())
    let bit_length = if i == 0 { 0 } else { 64 - i.leading_zeros() };
    bit_length <= max_simple_bits
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

// Reverse bits in a 64-bit integer using the lookup table (matches Python's reverse64)
fn reverse64(v: u64) -> u64 {
    (REVERSE_BITS_TABLE[((v >> 0) & 0xFF) as usize] as u64) << 56
        | (REVERSE_BITS_TABLE[((v >> 8) & 0xFF) as usize] as u64) << 48
        | (REVERSE_BITS_TABLE[((v >> 16) & 0xFF) as usize] as u64) << 40
        | (REVERSE_BITS_TABLE[((v >> 24) & 0xFF) as usize] as u64) << 32
        | (REVERSE_BITS_TABLE[((v >> 32) & 0xFF) as usize] as u64) << 24
        | (REVERSE_BITS_TABLE[((v >> 40) & 0xFF) as usize] as u64) << 16
        | (REVERSE_BITS_TABLE[((v >> 48) & 0xFF) as usize] as u64) << 8
        | (REVERSE_BITS_TABLE[((v >> 56) & 0xFF) as usize] as u64) << 0
}

// Reverse n bits of x
fn reverse_bits(x: u64, n: u32) -> u64 {
    if n == 0 {
        return 0;
    }
    if n >= 64 {
        return reverse64(x);
    }
    let reversed = reverse64(x);
    reversed >> (64 - n)
}

// Update mantissa according to Python's lexicographic encoding rules.
// This implements the exact same mantissa bit reversal algorithm used in Python Hypothesis.
// The reversal ensures that lexicographically smaller encodings represent "simpler" values.
fn update_mantissa(unbiased_exponent: i32, mantissa: u64, width: FloatWidth) -> u64 {
    let mantissa_bits = width.mantissa_bits();
    
    if unbiased_exponent <= 0 {
        // For values < 2.0 (unbiased exponent <= 0):
        // Reverse all mantissa bits to ensure proper ordering
        reverse_bits(mantissa, mantissa_bits)
    } else if unbiased_exponent <= mantissa_bits as i32 {
        // For values 2.0 to 2^mantissa_bits (e.g., 2.0 to 2^52 for f64):
        // Reverse only the fractional part bits, leave integer part unchanged
        let n_fractional_bits = mantissa_bits - unbiased_exponent as u32;
        let fractional_mask = (1u64 << n_fractional_bits) - 1;
        let fractional_part = mantissa & fractional_mask;
        let integer_part = mantissa & !fractional_mask;
        
        // Reconstruct with reversed fractional bits
        integer_part | reverse_bits(fractional_part, n_fractional_bits)
    } else {
        // For unbiased_exponent > mantissa_bits (very large integers):
        // No fractional part exists, leave mantissa unchanged
        mantissa
    }
}

// Convert lexicographically ordered integer to float of specified width (Python-equivalent).
// This implements the exact two-branch tagged union approach used in Python Hypothesis:
// - Tag bit 0: Simple integer encoding (direct representation)
// - Tag bit 1: Complex float encoding (IEEE 754 with transformations)
pub fn lex_to_float_width(i: u64, width: FloatWidth) -> f64 {
    use std::sync::OnceLock;
    static ENCODING_TABLES_16: OnceLock<Vec<u32>> = OnceLock::new();
    static ENCODING_TABLES_32: OnceLock<Vec<u32>> = OnceLock::new();
    static ENCODING_TABLES_64: OnceLock<Vec<u32>> = OnceLock::new();
    
    let total_bits = width.bits();
    let tag_bit_pos = total_bits - 1;
    let has_fractional_part = (i >> tag_bit_pos) != 0;
    
    if has_fractional_part {
        // Complex branch (tag = 1): IEEE 754 with transformations
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
        
        let exp_bits = width.exponent_bits();
        let mantissa_bits = width.mantissa_bits();
        let mantissa_mask = width.mantissa_mask();
        
        // Extract reordered exponent and transformed mantissa
        let exponent_idx = (i >> mantissa_bits) & ((1 << exp_bits) - 1);
        let exponent = encoding_table[exponent_idx as usize];
        let mut mantissa = i & mantissa_mask;
        
        // Reverse the mantissa transformation (decode operation)
        mantissa = update_mantissa(exponent as i32 - width.bias(), mantissa, width);
        
        // Construct IEEE 754 bit representation
        let ieee_bits = ((exponent as u64) << mantissa_bits) | mantissa;
        
        // Convert to appropriate float type then to f64
        match width {
            FloatWidth::Width16 => {
                let f16_val = f16::from_bits(ieee_bits as u16);
                f16_val.to_f64()
            },
            FloatWidth::Width32 => {
                let f32_val = f32::from_bits(ieee_bits as u32);
                f32_val as f64
            },
            FloatWidth::Width64 => {
                f64::from_bits(ieee_bits)
            },
        }
    } else {
        // Simple branch (tag = 0): Direct integer representation
        // Extract the integer value from the available bits
        let max_simple_bits = match width {
            FloatWidth::Width16 => 8,
            FloatWidth::Width32 => 24,
            FloatWidth::Width64 => SIMPLE_THRESHOLD_BITS,
        };
        
        let integral_part = i & ((1u64 << max_simple_bits) - 1);
        integral_part as f64
    }
}

// Note: is_simple_width is defined above with the Python-equivalent implementation

// Convert float to lexicographically ordered integer for specified width (Python-equivalent).
// This implements the exact two-branch encoding used in Python Hypothesis:
// - Simple integers use direct encoding with tag bit 0
// - All other floats use complex IEEE 754 encoding with tag bit 1
pub fn float_to_lex_width(f: f64, width: FloatWidth) -> u64 {
    // Handle negative numbers by taking absolute value (sign encoded separately)
    let abs_f = f.abs();
    
    if abs_f >= 0.0 && is_simple_width(abs_f, width) {
        // Simple branch (tag = 0): Direct integer encoding
        abs_f as u64
    } else {
        // Complex branch (tag = 1): IEEE 754 with transformations
        base_float_to_lex_width(abs_f, width)
    }
}

// Convert float to lexicographic encoding (internal implementation for complex branch).
// This handles the complex encoding case with IEEE 754 transformations (Python-equivalent).
fn base_float_to_lex_width(f: f64, width: FloatWidth) -> u64 {
    use std::sync::OnceLock;
    static DECODING_TABLES_16: OnceLock<Vec<u32>> = OnceLock::new();
    static DECODING_TABLES_32: OnceLock<Vec<u32>> = OnceLock::new();
    static DECODING_TABLES_64: OnceLock<Vec<u32>> = OnceLock::new();
    
    // Get the appropriate decoding table for exponent reordering
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
    let tag_bit_pos = width.bits() - 1;
    
    // Convert f64 to appropriate width IEEE 754 representation
    let ieee_bits = match width {
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
    
    // Remove sign bit (we only handle positive values in this branch)
    let unsigned_bits = ieee_bits & !(1u64 << (width.bits() - 1));
    
    // Extract IEEE 754 components
    let exponent = (unsigned_bits >> mantissa_bits) & ((1 << exp_bits) - 1);
    let mut mantissa = unsigned_bits & mantissa_mask;
    
    // Apply mantissa transformation (encoding operation)
    mantissa = update_mantissa(exponent as i32 - width.bias(), mantissa, width);
    
    // Reorder exponent for lexicographic properties
    let reordered_exponent = decoding_table[exponent as usize] as u64;
    
    // Construct complex encoding with tag bit 1
    (1u64 << tag_bit_pos) | (reordered_exponent << mantissa_bits) | mantissa
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

// Successor/predecessor and subnormal handling utilities.
// These functions provide precise control over float ordering and
// handle edge cases around subnormal numbers.

// Get the next representable float after the given value for specified width.
// Handles special cases including subnormals, infinities, and NaN.
pub fn next_float_width(value: f64, width: FloatWidth) -> f64 {
    if value.is_nan() {
        return value;
    }
    
    if value == f64::INFINITY {
        return value;
    }
    
    // Convert to appropriate width and get raw bits
    let bits = float_to_int(value, width);
    let max_bits = if width.bits() == 64 { u64::MAX } else { (1u64 << width.bits()) - 1 };
    
    // Handle negative infinity -> most negative finite
    if value == f64::NEG_INFINITY {
        let sign_bit = if width.bits() == 64 { 1u64 << 63 } else { 1u64 << (width.bits() - 1) };
        let max_exp = ((1u64 << width.exponent_bits()) - 2) << width.mantissa_bits(); // Not infinity
        let max_mantissa = (1u64 << width.mantissa_bits()) - 1;
        return int_to_float(sign_bit | max_exp | max_mantissa, width);
    }
    
    // For positive numbers, increment bits
    if value >= 0.0 {
        if bits == max_bits - 1 { // Just before positive infinity
            return f64::INFINITY;
        }
        return int_to_float(bits + 1, width);
    }
    
    // For negative numbers, decrement bits (moving toward zero)
    let sign_bit = if width.bits() == 64 { 1u64 << 63 } else { 1u64 << (width.bits() - 1) };
    if bits == sign_bit { // Negative zero -> smallest negative subnormal
        return int_to_float(sign_bit | 1, width);
    }
    
    int_to_float(bits - 1, width)
}

// Get the previous representable float before the given value for specified width.
// Handles special cases including subnormals, infinities, and NaN.
pub fn prev_float_width(value: f64, width: FloatWidth) -> f64 {
    if value.is_nan() {
        return value;
    }
    
    if value == f64::NEG_INFINITY {
        return value;
    }
    
    // Convert to appropriate width and get raw bits
    let bits = float_to_int(value, width);
    
    // Handle positive infinity -> most positive finite
    if value == f64::INFINITY {
        let max_exp = ((1u64 << width.exponent_bits()) - 2) << width.mantissa_bits();
        let max_mantissa = (1u64 << width.mantissa_bits()) - 1;
        return int_to_float(max_exp | max_mantissa, width);
    }
    
    // For positive numbers (including positive zero), decrement bits (moving toward zero/negative)
    if value >= 0.0 {
        let sign_bit = if width.bits() == 64 { 1u64 << 63 } else { 1u64 << (width.bits() - 1) };
        if bits == 0 { // Positive zero -> smallest negative subnormal
            return int_to_float(sign_bit | 1, width);
        }
        return int_to_float(bits - 1, width);
    }
    
    // For negative numbers, increment bits (moving away from zero)
    let sign_bit = if width.bits() == 64 { 1u64 << 63 } else { 1u64 << (width.bits() - 1) };
    let max_bits = if width.bits() == 64 { u64::MAX } else { (1u64 << width.bits()) - 1 };
    
    if bits == sign_bit { // Negative zero -> smallest positive subnormal  
        return int_to_float(1, width);
    }
    
    if bits == max_bits - 1 { // Just before negative infinity
        return f64::NEG_INFINITY;
    }
    
    int_to_float(bits + 1, width)
}

// Check if a float is subnormal (denormalized) for the given width.
// Subnormal numbers have exponent bits all zero but non-zero mantissa.
pub fn is_subnormal_width(value: f64, width: FloatWidth) -> bool {
    if !value.is_finite() || value == 0.0 {
        return false;
    }
    
    let bits = float_to_int(value, width);
    let mantissa_bits = width.mantissa_bits();
    let exp_bits = width.exponent_bits();
    
    // Remove sign bit
    let sign_mask = if width.bits() == 64 { 1u64 << 63 } else { 1u64 << (width.bits() - 1) };
    let unsigned_bits = bits & !sign_mask;
    
    // Extract exponent
    let exponent = (unsigned_bits >> mantissa_bits) & ((1u64 << exp_bits) - 1);
    
    // Extract mantissa
    let mantissa = unsigned_bits & ((1u64 << mantissa_bits) - 1);
    
    // Subnormal: exponent is zero, mantissa is non-zero
    exponent == 0 && mantissa != 0
}

// Get the smallest positive subnormal number for the given width.
pub fn min_positive_subnormal_width(width: FloatWidth) -> f64 {
    // Smallest subnormal has only the least significant mantissa bit set
    int_to_float(1, width)
}

// Get the largest subnormal number for the given width.
pub fn max_subnormal_width(width: FloatWidth) -> f64 {
    // Largest subnormal has all mantissa bits set, exponent zero
    let mantissa_mask = (1u64 << width.mantissa_bits()) - 1;
    int_to_float(mantissa_mask, width)
}

// Get the smallest positive normal number for the given width.
pub fn min_positive_normal_width(width: FloatWidth) -> f64 {
    // Smallest normal has exponent = 1, mantissa = 0
    let min_normal_exp = 1u64 << width.mantissa_bits();
    int_to_float(min_normal_exp, width)
}

// Backward compatibility functions for f64
pub fn next_float(value: f64) -> f64 {
    next_float_width(value, FloatWidth::Width64)
}

pub fn prev_float(value: f64) -> f64 {
    prev_float_width(value, FloatWidth::Width64)
}

pub fn is_subnormal(value: f64) -> bool {
    is_subnormal_width(value, FloatWidth::Width64)
}

// Float counting and cardinality utilities.
// These functions provide precise counting of representable floats
// within specified ranges for different widths.

// Count the number of representable floats between min and max (inclusive).
// Returns None if the range is invalid or contains infinite values.
pub fn count_floats_in_range_width(min: f64, max: f64, width: FloatWidth) -> Option<u64> {
    if min > max || min.is_infinite() || max.is_infinite() || min.is_nan() || max.is_nan() {
        return None;
    }
    
    // Convert to width precision
    let min_bits = float_to_int(min, width);
    let max_bits = float_to_int(max, width);
    
    // Handle the case where both values map to the same representable float
    if min_bits == max_bits {
        return Some(1);
    }
    
    // For finite ranges, count by bit difference
    // This works because IEEE 754 ordering matches bit ordering for positive numbers
    if min >= 0.0 && max >= 0.0 {
        // Both positive
        Some(max_bits - min_bits + 1)
    } else if min < 0.0 && max < 0.0 {
        // Both negative - bit ordering is reversed for negative numbers
        Some(min_bits - max_bits + 1)
    } else {
        // Range spans zero - count negative part + zero + positive part
        let zero_bits = 0u64;
        let neg_zero_bits = if width.bits() == 64 { 1u64 << 63 } else { 1u64 << (width.bits() - 1) };
        
        // Count from min to negative zero
        let negative_count = if min < 0.0 {
            let min_neg_bits = float_to_int(min, width);
            min_neg_bits - neg_zero_bits + 1
        } else {
            0
        };
        
        // Count from positive zero to max
        let positive_count = if max > 0.0 {
            let max_pos_bits = float_to_int(max, width);
            max_pos_bits - zero_bits + 1
        } else {
            0
        };
        
        // Add 1 for positive zero (negative zero is counted in negative_count if min < 0)
        let zero_count = if min <= 0.0 && max >= 0.0 { 1 } else { 0 };
        
        Some(negative_count + zero_count + positive_count)
    }
}

// Count all finite representable floats for a given width.
// This excludes infinities and NaN values.
pub fn count_finite_floats_width(width: FloatWidth) -> u64 {
    // Count normal floats + subnormal floats + two zeros
    count_normal_floats_width(width) + count_subnormal_floats_width(width) + 2
}

// Count all normal (non-subnormal, non-zero) floats for a given width.
pub fn count_normal_floats_width(width: FloatWidth) -> u64 {
    let exp_bits = width.exponent_bits();
    let mantissa_bits = width.mantissa_bits();
    
    // Normal numbers have exponent from 1 to (2^exp_bits - 2)
    // (0 is subnormal/zero, 2^exp_bits - 1 is infinity/NaN)
    let normal_exponents = (1u64 << exp_bits) - 2;
    
    // Each normal exponent can have any mantissa value
    let mantissa_combinations = 1u64 << mantissa_bits;
    
    // Count for both positive and negative
    2 * normal_exponents * mantissa_combinations
}

// Count all subnormal floats for a given width.
pub fn count_subnormal_floats_width(width: FloatWidth) -> u64 {
    let mantissa_bits = width.mantissa_bits();
    
    // Subnormal numbers have exponent = 0 and mantissa != 0
    // Each non-zero mantissa pattern gives a subnormal number
    let subnormal_patterns = (1u64 << mantissa_bits) - 1; // -1 to exclude mantissa = 0 (which is zero)
    
    // Count for both positive and negative
    2 * subnormal_patterns
}

// Get the n-th representable float in the range [min, max] for given width.
// Returns None if n is out of bounds or the range is invalid.
pub fn nth_float_in_range_width(min: f64, max: f64, n: u64, width: FloatWidth) -> Option<f64> {
    let count = count_floats_in_range_width(min, max, width)?;
    
    if n >= count {
        return None;
    }
    
    // Convert to width precision
    let min_bits = float_to_int(min, width);
    let _max_bits = float_to_int(max, width);
    
    if min >= 0.0 && max >= 0.0 {
        // Both positive - simple bit arithmetic
        Some(int_to_float(min_bits + n, width))
    } else if min < 0.0 && max < 0.0 {
        // Both negative - reverse bit arithmetic
        Some(int_to_float(min_bits - n, width))
    } else {
        // Range spans zero - more complex logic needed
        let zero_bits = 0u64;
        let neg_zero_bits = if width.bits() == 64 { 1u64 << 63 } else { 1u64 << (width.bits() - 1) };
        
        // Count from min to negative zero
        let negative_count = if min < 0.0 {
            let min_neg_bits = float_to_int(min, width);
            min_neg_bits - neg_zero_bits + 1
        } else {
            0
        };
        
        if n < negative_count {
            // n-th element is in negative range
            let min_neg_bits = float_to_int(min, width);
            Some(int_to_float(min_neg_bits - n, width))
        } else if n == negative_count && min <= 0.0 && max >= 0.0 {
            // n-th element is zero
            Some(0.0)
        } else {
            // n-th element is in positive range
            let positive_offset = n - negative_count - (if min <= 0.0 && max >= 0.0 { 1 } else { 0 });
            Some(int_to_float(zero_bits + positive_offset, width))
        }
    }
}

// Find the index of a specific float within the range [min, max] for given width.
// Returns None if the value is not in the range or the range is invalid.
pub fn index_of_float_in_range_width(value: f64, min: f64, max: f64, width: FloatWidth) -> Option<u64> {
    if value < min || value > max || min > max {
        return None;
    }
    
    if min.is_infinite() || max.is_infinite() || value.is_infinite() {
        return None;
    }
    
    if value.is_nan() || min.is_nan() || max.is_nan() {
        return None;
    }
    
    // Convert to width precision
    let value_bits = float_to_int(value, width);
    let min_bits = float_to_int(min, width);
    let _max_bits = float_to_int(max, width);
    
    if min >= 0.0 && max >= 0.0 {
        // Both positive
        Some(value_bits - min_bits)
    } else if min < 0.0 && max < 0.0 {
        // Both negative
        Some(min_bits - value_bits)
    } else {
        // Range spans zero
        let _zero_bits = 0u64;
        let neg_zero_bits = if width.bits() == 64 { 1u64 << 63 } else { 1u64 << (width.bits() - 1) };
        
        if value < 0.0 {
            // Value is in negative range
            let min_neg_bits = float_to_int(min, width);
            Some(min_neg_bits - value_bits)
        } else if value == 0.0 {
            // Value is zero
            let negative_count = if min < 0.0 {
                let min_neg_bits = float_to_int(min, width);
                min_neg_bits - neg_zero_bits + 1
            } else {
                0
            };
            Some(negative_count)
        } else {
            // Value is positive
            let negative_count = if min < 0.0 {
                let min_neg_bits = float_to_int(min, width);
                min_neg_bits - neg_zero_bits + 1
            } else {
                0
            };
            let zero_count = if min <= 0.0 { 1 } else { 0 };
            Some(negative_count + zero_count + value_bits)
        }
    }
}

// Backward compatibility functions for f64
pub fn count_floats_in_range(min: f64, max: f64) -> Option<u64> {
    count_floats_in_range_width(min, max, FloatWidth::Width64)
}

pub fn nth_float_in_range(min: f64, max: f64, n: u64) -> Option<f64> {
    nth_float_in_range_width(min, max, n, FloatWidth::Width64)
}

pub fn index_of_float_in_range(value: f64, min: f64, max: f64) -> Option<u64> {
    index_of_float_in_range_width(value, min, max, FloatWidth::Width64)
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
mod tests;
