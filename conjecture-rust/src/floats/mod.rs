// IEEE 754 floating point generation with lexicographic encoding.
// This module provides comprehensive float generation with multi-width support
// (16, 32, and 64-bit) and lexicographic ordering for excellent shrinking properties.

mod constants;
mod encoding;

pub use constants::{FloatWidth, REVERSE_BITS_TABLE, SIMPLE_THRESHOLD_BITS};
pub use encoding::{
    is_simple_width, lex_to_float_width, float_to_lex_width, 
    lex_to_float, float_to_lex
};

// Note: Functions defined in this module are automatically exported with pub

use crate::data::{DataSource, FailedDraw};
use half::f16;

type Draw<T> = Result<T, FailedDraw>;

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

// Auto-detect whether subnormals should be allowed based on bounds.
// This matches Python Hypothesis behavior: if the bounds require subnormals
// to represent any values in the range, then subnormals are automatically enabled.
fn should_allow_subnormals_auto(min_value: f64, max_value: f64, width: FloatWidth) -> bool {
    let smallest_normal = width.smallest_normal();
    
    // If range includes values smaller than smallest normal, we need subnormals
    if min_value > 0.0 && min_value < smallest_normal {
        return true;
    }
    if max_value < 0.0 && max_value > -smallest_normal {
        return true;
    }
    // If range spans zero and includes small values, we might need subnormals
    if min_value <= 0.0 && max_value >= 0.0 {
        if (min_value < 0.0 && min_value > -smallest_normal) || 
           (max_value > 0.0 && max_value < smallest_normal) {
            return true;
        }
    }
    
    false
}

// Validate that the bounds are compatible with subnormal settings.
// Returns an error if subnormals are disabled but required for the bounds.
fn validate_bounds_subnormal_compatibility(
    min_value: f64, 
    max_value: f64, 
    width: FloatWidth, 
    allow_subnormal: bool
) -> Result<(), &'static str> {
    if allow_subnormal {
        return Ok(()); // Always valid if subnormals are allowed
    }
    
    let smallest_normal = width.smallest_normal();
    
    // Check if bounds require subnormals but they're disabled
    if min_value > 0.0 && min_value < smallest_normal {
        return Err("min_value requires subnormal numbers, but allow_subnormal=false");
    }
    if max_value < 0.0 && max_value > -smallest_normal {
        return Err("max_value requires subnormal numbers, but allow_subnormal=false");
    }
    if min_value < 0.0 && min_value > -smallest_normal && max_value >= 0.0 {
        return Err("min_value requires subnormal numbers, but allow_subnormal=false");
    }
    if max_value > 0.0 && max_value < smallest_normal && min_value <= 0.0 {
        return Err("max_value requires subnormal numbers, but allow_subnormal=false");
    }
    
    Ok(())
}

// Get the next normal float after a value, skipping subnormals.
// This finds the smallest normal number greater than the given value.
pub fn next_up_normal_width(value: f64, width: FloatWidth) -> f64 {
    if value.is_nan() {
        return value;
    }
    
    let smallest_normal = width.smallest_normal();
    
    // If we're below the smallest positive normal, jump to it
    if value < smallest_normal {
        return smallest_normal;
    }
    
    // If we're negative and above the negative smallest normal, jump to positive smallest normal
    if value >= -smallest_normal && value < 0.0 {
        return smallest_normal;
    }
    
    // Otherwise find the next representable normal number
    let mut candidate = next_float_width(value, width);
    while candidate.is_finite() && candidate > value && is_subnormal_width(candidate, width) {
        candidate = next_float_width(candidate, width);
    }
    
    candidate
}

// Get the previous normal float before a value, skipping subnormals.
// This finds the largest normal number less than the given value.
pub fn next_down_normal_width(value: f64, width: FloatWidth) -> f64 {
    if value.is_nan() {
        return value;
    }
    
    let smallest_normal = width.smallest_normal();
    
    // If we're above the negative smallest normal, jump to it
    if value > -smallest_normal {
        return -smallest_normal;
    }
    
    // If we're positive and below the positive smallest normal, jump to negative smallest normal
    if value <= smallest_normal && value > 0.0 {
        return -smallest_normal;
    }
    
    // Otherwise find the previous representable normal number
    let mut candidate = prev_float_width(value, width);
    while candidate.is_finite() && candidate < value && is_subnormal_width(candidate, width) {
        candidate = prev_float_width(candidate, width);
    }
    
    candidate
}

// Generate a random float using lexicographic encoding with width support.
// This function provides the main entry point for width-aware float generation
// with full control over bounds, special values, and subnormal handling.
pub fn draw_float_width(
    source: &mut DataSource,
    width: FloatWidth,
    min_value: f64,
    max_value: f64,
    allow_nan: bool,
    allow_infinity: bool,
) -> Draw<f64> {
    draw_float_width_with_subnormals(source, width, min_value, max_value, allow_nan, allow_infinity, None)
}

// Generate a random float with explicit subnormal control.
// When allow_subnormal is None, auto-detection logic is used (matching Python).
// When allow_subnormal is Some(bool), it provides explicit control.
pub fn draw_float_width_with_subnormals(
    source: &mut DataSource,
    width: FloatWidth,
    min_value: f64,
    max_value: f64,
    allow_nan: bool,
    allow_infinity: bool,
    allow_subnormal: Option<bool>,
) -> Draw<f64> {
    // Determine subnormal policy
    let subnormal_allowed = match allow_subnormal {
        Some(explicit) => {
            // Validate explicit setting against bounds
            if let Err(_msg) = validate_bounds_subnormal_compatibility(min_value, max_value, width, explicit) {
                return Err(FailedDraw);
            }
            explicit
        },
        None => {
            // Auto-detect based on bounds (Python Hypothesis behavior)
            should_allow_subnormals_auto(min_value, max_value, width)
        }
    };
    
    let special_floats = special_floats_for_width(width);
    
    // 5% chance of returning special values
    if source.bits(6)? == 0 {
        // Try to return a special value that fits constraints
        for &special in special_floats {
            if (!special.is_nan() || allow_nan)
                && (!special.is_infinite() || allow_infinity)
                && (!is_subnormal_width(special, width) || subnormal_allowed)
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
    
    // Handle subnormal exclusion
    if is_subnormal_width(result, width) && !subnormal_allowed {
        // Replace subnormal with nearest normal number
        if result > 0.0 {
            result = width.smallest_normal();
        } else {
            result = -width.smallest_normal();
        }
    }
    
    // Clamp to bounds
    if result < min_value {
        result = min_value;
    } else if result > max_value {
        result = max_value;
    }
    
    // Final subnormal check after clamping
    if is_subnormal_width(result, width) && !subnormal_allowed {
        // If clamping produced a subnormal, find nearest normal in bounds
        if result > 0.0 {
            let next_normal = next_up_normal_width(result, width);
            if next_normal <= max_value {
                result = next_normal;
            } else {
                result = next_down_normal_width(max_value, width);
            }
        } else {
            let next_normal = next_down_normal_width(result, width);
            if next_normal >= min_value {
                result = next_normal;
            } else {
                result = next_up_normal_width(min_value, width);
            }
        }
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

// Backward compatibility function for f64 generation with subnormal control
pub fn draw_float_with_subnormals(
    source: &mut DataSource,
    min_value: f64,
    max_value: f64,
    allow_nan: bool,
    allow_infinity: bool,
    allow_subnormal: Option<bool>,
) -> Draw<f64> {
    draw_float_width_with_subnormals(source, FloatWidth::Width64, min_value, max_value, allow_nan, allow_infinity, allow_subnormal)
}

// Backward compatibility functions for normal-only helpers
pub fn next_up_normal(value: f64) -> f64 {
    next_up_normal_width(value, FloatWidth::Width64)
}

pub fn next_down_normal(value: f64) -> f64 {
    next_down_normal_width(value, FloatWidth::Width64)
}

// Flush-to-Zero (FTZ) detection utilities.
// These functions help detect broken subnormal support in the runtime environment.

// Check if the current environment supports subnormal numbers properly.
// Returns false if subnormals are flushed to zero (FTZ mode enabled).
pub fn environment_supports_subnormals() -> bool {
    // Test with the smallest positive subnormal for f64
    let tiny_subnormal = min_positive_subnormal_width(FloatWidth::Width64);
    
    // If FTZ is enabled, arithmetic with subnormals will produce zero
    let result = tiny_subnormal * 0.5;
    
    // In proper IEEE 754, this should produce an even smaller subnormal
    // In FTZ mode, this will be flushed to zero
    result != 0.0 && result < tiny_subnormal
}

// Detect if subnormals are flushed to zero for a specific width.
pub fn detect_ftz_for_width(width: FloatWidth) -> bool {
    let tiny_subnormal = min_positive_subnormal_width(width);
    
    // Perform operation that should produce a smaller subnormal
    let result = tiny_subnormal * 0.5;
    
    // If result is zero, FTZ is likely enabled
    result == 0.0
}

// Get a warning message if subnormals are not properly supported.
pub fn subnormal_support_warning() -> Option<&'static str> {
    if !environment_supports_subnormals() {
        Some("Warning: Subnormal numbers appear to be flushed to zero in this environment. \
              This may affect the accuracy of float generation when allow_subnormal=true.")
    } else {
        None
    }
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
