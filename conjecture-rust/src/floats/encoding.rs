// Core lexicographic encoding functions for floats
// This module contains the heart of the float generation strategy

use super::constants::{FloatWidth, REVERSE_BITS_TABLE, SIMPLE_THRESHOLD_BITS};
use half::f16;
use std::sync::OnceLock;

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
pub fn lex_to_float(i: u64, width: FloatWidth) -> f64 {
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

// Convert float to lexicographically ordered integer for specified width (Python-equivalent).
// This implements the exact two-branch encoding used in Python Hypothesis:
// - Simple integers use direct encoding with tag bit 0
// - All other floats use complex IEEE 754 encoding with tag bit 1
pub fn float_to_lex(f: f64, width: FloatWidth) -> u64 {
    // Handle negative numbers by taking absolute value (sign encoded separately)
    let abs_f = f.abs();
    
    if abs_f >= 0.0 && is_simple_width(abs_f, width) {
        // Simple branch (tag = 0): Direct integer encoding
        abs_f as u64
    } else {
        // Complex branch (tag = 1): IEEE 754 with transformations
        base_float_to_lex(abs_f, width)
    }
}

// Convert float to lexicographic encoding (internal implementation for complex branch).
// This handles the complex encoding case with IEEE 754 transformations (Python-equivalent).
fn base_float_to_lex(f: f64, width: FloatWidth) -> u64 {
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


#[cfg(test)]
mod tests {
    use super::*;
    
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
    fn test_reverse64_matches_python() {
        // Test our reverse64 function against known values
        assert_eq!(reverse64(0), 0);
        assert_eq!(reverse64(1), 0x8000000000000000);  // 1 -> MSB set
        assert_eq!(reverse64(0x8000000000000000), 1);  // MSB -> 1
        
        // Test double reverse is identity
        let test_values = [0u64, 1, 0xFF, 0xFFFF, u64::MAX];
        for val in test_values {
            assert_eq!(reverse64(reverse64(val)), val, "Double reverse failed for 0x{:016x}", val);
        }
    }
    
    #[test]
    fn test_update_mantissa_matches_python() {
        // Test our update_mantissa function matches Python's behavior
        let width = FloatWidth::Width64;
        let mantissa = 0x123456789abcd;
        
        // Test case 1: unbiased_exponent <= 0 (reverse all bits)
        let updated = update_mantissa(-1, mantissa, width);
        let expected = reverse_bits(mantissa, 52);
        assert_eq!(updated, expected, "Failed for unbiased_exponent <= 0");
        
        // Test case 2: unbiased_exponent in [1, 51] (reverse fractional part only)
        let unbiased_exp = 10;
        let updated = update_mantissa(unbiased_exp, mantissa, width);
        let n_fractional_bits = 52 - unbiased_exp as u32;
        let fractional_mask = (1u64 << n_fractional_bits) - 1;
        let fractional_part = mantissa & fractional_mask;
        let integer_part = mantissa & !fractional_mask;
        let expected = integer_part | reverse_bits(fractional_part, n_fractional_bits);
        assert_eq!(updated, expected, "Failed for unbiased_exponent in [1, 51]");
        
        // Test case 3: unbiased_exponent > 51 (no change)
        let updated = update_mantissa(100, mantissa, width);
        assert_eq!(updated, mantissa, "Failed for unbiased_exponent > 51");
    }
    
    #[test]
    fn test_reverse_bits_table_reverses_bits() {
        // Matches Python's test_reverse_bits_table_reverses_bits exactly
        for i in 0..=255u8 {
            let reversed = REVERSE_BITS_TABLE[i as usize];
            
            // Check that each bit is in the correct position
            for bit_pos in 0..8 {
                let original_bit = (i >> bit_pos) & 1;
                let reversed_bit = (reversed >> (7 - bit_pos)) & 1;
                assert_eq!(original_bit, reversed_bit, 
                    "Bit reversal failed for byte {} at position {}", i, bit_pos);
            }
        }
    }
    
    #[test]
    fn test_reverse_bits_table_has_right_elements() {
        // Matches Python's test_reverse_bits_table_has_right_elements exactly
        assert_eq!(REVERSE_BITS_TABLE.len(), 256, "Table should have 256 elements");
        assert_eq!(REVERSE_BITS_TABLE[0], 0);
        assert_eq!(REVERSE_BITS_TABLE[1], 128);  // 0b00000001 -> 0b10000000
        assert_eq!(REVERSE_BITS_TABLE[128], 1);  // 0b10000000 -> 0b00000001
        assert_eq!(REVERSE_BITS_TABLE[255], 255); // 0b11111111 -> 0b11111111
    }
    
    #[test]
    fn test_double_reverse() {
        // Test that reversing bits twice returns original value
        let test_values = [0u64, 1, 0x123456789abcdef0, u64::MAX];
        for i in test_values {
            let j = reverse64(i);
            let double_reversed = reverse64(j);
            assert_eq!(double_reversed, i, 
                "Double reverse64 should be identity for {:#x}", i);
        }
    }
    
    #[test]
    fn test_exponent_ordering_matches_python() {
        // Test that our exponent ordering exactly matches Python's
        let width = FloatWidth::Width64;
        let (encoding_table, decoding_table) = build_exponent_tables(width);
        
        // Verify some key positions match Python's expected order
        // Position 0 should be exponent 1023 (bias point, unbiased = 0)
        assert_eq!(encoding_table[0], 1023, "Position 0 should be bias point");
        
        // Position 2047 should be max exponent (infinity/NaN)
        assert_eq!(encoding_table[2047], 2047, "Position 2047 should be max exponent");
        
        // Position 2046 should be exponent 0 (most negative unbiased exponent)
        assert_eq!(encoding_table[2046], 0, "Position 2046 should be exponent 0");
        
        // Verify decode is inverse of encode
        for exp in 0..=2047u32 {
            let encoded_pos = decoding_table[exp as usize];
            let decoded_exp = encoding_table[encoded_pos as usize];
            assert_eq!(exp, decoded_exp, "Encoding/decoding should be inverse");
        }
    }

    #[test]
    fn test_reverse_bits_table_matches_python() {
        // Verify our REVERSE_BITS_TABLE exactly matches Python's
        let expected_first_16 = [0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 
                                0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0];
        let expected_last_16 = [0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef,
                               0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff];
        
        for (i, &expected) in expected_first_16.iter().enumerate() {
            assert_eq!(REVERSE_BITS_TABLE[i], expected, 
                "REVERSE_BITS_TABLE[{}] should be 0x{:02x}, got 0x{:02x}", 
                i, expected, REVERSE_BITS_TABLE[i]);
        }
        
        for (i, &expected) in expected_last_16.iter().enumerate() {
            let idx = 256 - 16 + i;
            assert_eq!(REVERSE_BITS_TABLE[idx], expected,
                "REVERSE_BITS_TABLE[{}] should be 0x{:02x}, got 0x{:02x}",
                idx, expected, REVERSE_BITS_TABLE[idx]);
        }
    }

    #[test]
    fn test_bit_level_compatibility_with_python() {
        // Test our implementation produces exactly the same bit patterns as Python
        let width = FloatWidth::Width64;
        
        // Test update_mantissa with Python-verified results
        let update_test_cases = [
            (-1, 0x123456789abcd, 0xb3d591e6a2c48),
            (10, 0xfffffffffffff, 0xfffffffffffff),
            (100, 0x123456789abcd, 0x123456789abcd),
            (0, 0x789abcdef0123, 0xc480f7b3d591e),
            (52, 0x0abcdef123456, 0x0abcdef123456),
        ];
        
        for &(unbiased_exp, mantissa, expected) in &update_test_cases {
            let result = update_mantissa(unbiased_exp, mantissa, width);
            assert_eq!(result, expected,
                "update_mantissa({}, 0x{:013x}) = 0x{:013x}, expected 0x{:013x}",
                unbiased_exp, mantissa, result, expected);
        }
        
        // Test reverse64 with Python-verified results
        let reverse_test_cases = [
            (0x0000000000000000, 0x0000000000000000),
            (0x0000000000000001, 0x8000000000000000),
            (0x00000000000000ff, 0xff00000000000000),
            (0x8000000000000000, 0x0000000000000001),
            (0x123456789abcdef0, 0x0f7b3d591e6a2c48),
        ];
        
        for &(input, expected) in &reverse_test_cases {
            let result = reverse64(input);
            assert_eq!(result, expected,
                "reverse64(0x{:016x}) = 0x{:016x}, expected 0x{:016x}",
                input, result, expected);
        }
    }

    #[test]
    fn test_exponent_ordering_table_integrity() {
        // Test that exponent ordering tables are correctly built and used
        // This would catch issues if table building was simplified incorrectly
        
        // Test with simple known values rather than constructing floats from bits
        let test_cases = vec![
            (1.0, FloatWidth::Width64),   // Normal case
            (2.0, FloatWidth::Width64),   // Power of 2
            (0.5, FloatWidth::Width64),   // Fractional
            (1.0, FloatWidth::Width32),   // f32 case
            (2.5, FloatWidth::Width32),   // f32 fractional
        ];
        
        for (val, width) in test_cases {
            let encoded = float_to_lex(val, width);
            let decoded = lex_to_float(encoded, width);
            
            // Allow precision differences based on width
            let tolerance = match width {
                FloatWidth::Width16 => val.abs() * 1e-3 + 1e-6,
                FloatWidth::Width32 => val.abs() * 1e-6 + 1e-12,
                FloatWidth::Width64 => 0.0,
            };
            
            assert!((val - decoded).abs() <= tolerance,
                    "Exponent encoding failed for {} at width {:?}: {} != {} (diff: {})", 
                    val, width, val, decoded, (val - decoded).abs());
        }
        
        // Test that the encoding/decoding tables are inverses for f64
        let width = FloatWidth::Width64;
        let (encoding_table, decoding_table) = build_exponent_tables(width);
        
        for (i, &exp) in encoding_table.iter().enumerate() {
            if exp < decoding_table.len() as u32 {
                assert_eq!(decoding_table[exp as usize], i as u32, 
                          "Exponent table inverse failed at position {}", i);
            }
        }
    }

    #[test]
    fn test_bit_reversal_lookup_table_usage() {
        // Test that the REVERSE_BITS_TABLE is used correctly in all contexts
        // This would catch issues if table usage was simplified incorrectly
        
        // Test direct reverse64 function
        for i in 0..=255u64 {
            let reversed = reverse64(i);
            let double_reversed = reverse64(reversed);
            assert_eq!(i, double_reversed, "Double reversal failed for {}", i);
        }
        
        // Test that table is used correctly in mantissa updates
        let test_mantissa = 0x123456789ABCDEFu64;
        let reversed = reverse64(test_mantissa);
        
        // Manually reverse using the table to verify it matches
        let mut manual_reverse = 0u64;
        for byte_pos in 0..8 {
            let byte_val = (test_mantissa >> (byte_pos * 8)) & 0xFF;
            let reversed_byte = REVERSE_BITS_TABLE[byte_val as usize] as u64;
            manual_reverse |= reversed_byte << ((7 - byte_pos) * 8);
        }
        
        assert_eq!(reversed, manual_reverse, "Table usage inconsistency");
    }
}