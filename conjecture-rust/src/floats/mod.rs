// IEEE 754 floating point generation with lexicographic encoding.
// This module provides comprehensive float generation with multi-width support
// (16, 32, and 64-bit) and lexicographic ordering for excellent shrinking properties.

mod constants;
mod encoding;

pub use constants::{FloatWidth, REVERSE_BITS_TABLE, SIMPLE_THRESHOLD_BITS};
pub use encoding::{
    is_simple_width, lex_to_float, float_to_lex
};

// Note: Functions defined in this module are automatically exported with pub

use crate::data::{DataSource, FailedDraw};
use half::f16;
use std::collections::HashMap;

type Draw<T> = Result<T, FailedDraw>;

// Python Hypothesis global float constants for enhanced test diversity (15% injection probability)
// Based on _constant_floats from Python implementation
static GLOBAL_FLOAT_CONSTANTS: &[f64] = &[
    // Mathematical constants and common values
    0.5, 1.1, 1.5, 1.9,
    0.3333333333333333, // 1.0/3
    10e6, 10e-6,
    
    // Width-specific boundary values
    1.175494351e-38,    // f32 smallest normal (2^-126)
    5.960464477539063e-8, // f16 smallest normal (2^-24) 
    6.103515625e-5,     // f16 smallest positive normal
    1.1754943508222875e-38, // f32 MIN_POSITIVE
    2.2250738585072014e-308, // f64 smallest normal
    
    // Important mathematical boundaries  
    1.7976931348623157e308, // f64 MAX
    3.4028235e38,       // f32 MAX
    65504.0,            // f16 MAX
    9007199254740992.0, // 2^53 - largest precise integer in f64
    
    // Epsilon values for different widths
    1.192092896e-07,    // f32 EPSILON (2^-23)
    2.220446049250313e-16, // f64 EPSILON (2^-52)
    0.0009765625,       // f16 EPSILON (2^-10)
    
    // Boundary arithmetic values
    0.999999,           // 1 - 10e-6
    2.00001,            // 2 + 10e-6
    
    // Width-specific minimum subnormals  
    5.877471754111438e-39,  // 2^-149 (f32 smallest subnormal)
    6.103515625e-05,        // 2^-14 (f16 related)
    5.960464477539063e-8,   // 2^-24 (f32 related)
    1.390671161567e-309,    // 2^-126 adjusted for f64
    
    // Subnormal variants (Python generates these dynamically)
    1.1125369292536007e-308, // float_info.min / 2
    2.225073858507201e-309,  // float_info.min / 10  
    2.225073858507201e-312,  // float_info.min / 1000
    2.2250738585072014e-313, // float_info.min / 100_000
    
    // Zero (will be duplicated as +0.0/-0.0 during generation)
    0.0,
];

// Signaling NaN support (Python's SIGNALING_NAN = 0x7FF8_0000_0000_0001)
const SIGNALING_NAN_BITS: u64 = 0x7FF8_0000_0000_0001;

fn signaling_nan() -> f64 {
    f64::from_bits(SIGNALING_NAN_BITS)
}

fn negative_signaling_nan() -> f64 {
    f64::from_bits(SIGNALING_NAN_BITS | (1u64 << 63))
}

// Constant injection system with Python Hypothesis parity
struct ConstantPool {
    // Global constants: always available mathematical values
    global_constants: Vec<f64>,
    // Local constants: extracted from user code via AST parsing (e.g., from Swift FFI)
    local_constants: Vec<f64>,
    // Cache for constraint-filtered constants
    constraint_cache: HashMap<String, Vec<f64>>,
}

impl ConstantPool {
    fn with_local_constants(local_constants: &[f64]) -> Self {
        let mut global_constants = Vec::new();
        
        // Add all positive constants from GLOBAL_FLOAT_CONSTANTS
        global_constants.extend_from_slice(GLOBAL_FLOAT_CONSTANTS);
        
        // Add negative versions (Python does this)
        for &constant in GLOBAL_FLOAT_CONSTANTS {
            if constant != 0.0 { // Don't duplicate zero
                global_constants.push(-constant);
            }
        }
        
        // Add special NaN varieties
        global_constants.push(f64::NAN);
        global_constants.push(-f64::NAN);
        global_constants.push(signaling_nan());
        global_constants.push(negative_signaling_nan());
        
        // Add infinities
        global_constants.push(f64::INFINITY);
        global_constants.push(f64::NEG_INFINITY);
        
        // Add signed zeros
        global_constants.push(0.0);
        global_constants.push(-0.0);
        
        Self {
            global_constants,
            local_constants: local_constants.to_vec(),
            constraint_cache: HashMap::new(),
        }
    }
    
    fn get_valid_constants(&mut self, 
                          min_value: Option<f64>, 
                          max_value: Option<f64>,
                          allow_nan: bool,
                          allow_infinity: bool,
                          allow_subnormal: bool,
                          smallest_nonzero_magnitude: f64,
                          width: FloatWidth) -> &[f64] {
        // Create cache key based on constraints and local constants
        let local_constants_hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            self.local_constants.len().hash(&mut hasher);
            for &constant in &self.local_constants {
                constant.to_bits().hash(&mut hasher);
            }
            hasher.finish()
        };
        let cache_key = format!("{}:{:?}:{:?}:{}:{}:{}:{}:{:x}", 
            width.bits(), min_value, max_value, allow_nan, allow_infinity, 
            allow_subnormal, smallest_nonzero_magnitude, local_constants_hash);
        
        if !self.constraint_cache.contains_key(&cache_key) {
            let mut valid_constants = Vec::new();
            
            // Add valid global constants
            for &constant in &self.global_constants {
                if is_constant_valid(constant, min_value, max_value, allow_nan, 
                                   allow_infinity, allow_subnormal, 
                                   smallest_nonzero_magnitude, width) {
                    valid_constants.push(constant);
                }
            }
            
            // Add valid local constants (extracted from user code via AST parsing)
            for &constant in &self.local_constants {
                if is_constant_valid(constant, min_value, max_value, allow_nan, 
                                   allow_infinity, allow_subnormal, 
                                   smallest_nonzero_magnitude, width) {
                    valid_constants.push(constant);
                }
            }
            
            self.constraint_cache.insert(cache_key.clone(), valid_constants);
        }
        
        self.constraint_cache.get(&cache_key).unwrap()
    }
}

// Validate if a constant meets all constraints (Python's choice_permitted equivalent)
fn is_constant_valid(value: f64,
                    min_value: Option<f64>,
                    max_value: Option<f64>, 
                    allow_nan: bool,
                    allow_infinity: bool,
                    allow_subnormal: bool,
                    smallest_nonzero_magnitude: f64,
                    width: FloatWidth) -> bool {
    // Check NaN constraint
    if value.is_nan() && !allow_nan {
        return false;
    }
    
    // Check infinity constraint
    if value.is_infinite() && !allow_infinity {
        return false;
    }
    
    // Check subnormal constraint
    if is_subnormal_width(value, width) && !allow_subnormal {
        return false;
    }
    
    // Check smallest_nonzero_magnitude constraint
    if value != 0.0 && value.abs() < smallest_nonzero_magnitude {
        return false;
    }
    
    // Check bounds
    if let Some(min) = min_value {
        if value < min {
            return false;
        }
    }
    
    if let Some(max) = max_value {
        if value > max {
            return false;
        }
    }
    
    true
}

// Sophisticated float clamper with mantissa-based resampling (Python's make_float_clamper)
// This implements the advanced clamping logic from Python Hypothesis
fn make_float_clamper(
    min_value: f64,
    max_value: f64,
    allow_nan: bool,
    smallest_nonzero_magnitude: f64,
    width: FloatWidth,
) -> impl Fn(f64) -> f64 {
    let range_size = (max_value - min_value).min(f64::MAX);
    let mantissa_mask = width.mantissa_mask();
    
    move |f: f64| -> f64 {
        // If value already meets all constraints, return as-is
        if is_constant_valid(f, Some(min_value), Some(max_value), allow_nan,
                           true, true, smallest_nonzero_magnitude, width) {
            return f;
        }
        
        // Outside bounds; pick a new value using mantissa bits for resampling
        // This matches Python's sophisticated resampling approach
        let mant = float_to_int(f.abs(), width) & mantissa_mask;
        let mant_fraction = (mant as f64) / (mantissa_mask as f64);
        let mut result = min_value + range_size * mant_fraction;
        
        // Handle smallest_nonzero_magnitude constraint (Python's exact logic)
        if result != 0.0 && result.abs() < smallest_nonzero_magnitude {
            result = smallest_nonzero_magnitude;
            
            // Python's sign logic: if smallest_nonzero_magnitude > max_value,
            // then -smallest_nonzero_magnitude must be valid, so use negative
            if smallest_nonzero_magnitude > max_value {
                result = -result;
            }
        }
        
        // Re-enforce bounds (Python does this to protect against FP arithmetic errors)
        clamp_float(min_value, result, max_value)
    }
}

// Utility function for clamping with proper float ordering
fn clamp_float(min_val: f64, value: f64, max_val: f64) -> f64 {
    if value < min_val {
        min_val
    } else if value > max_val {
        max_val
    } else {
        value
    }
}

// Generate multiple NaN varieties with different bit patterns (Python parity)
fn generate_nan_varieties(width: FloatWidth) -> Vec<f64> {
    let mut nans = vec![f64::NAN, -f64::NAN];
    
    // Add some different NaN bit patterns (simplified version of Python's approach)
    let base_nan_bits = float_to_int(f64::NAN, width);
    let mantissa_mask = width.mantissa_mask();
    
    // Generate a few different mantissa patterns for NaN
    for i in [1u64, mantissa_mask / 2, mantissa_mask - 1] {
        if i != 0 && i < mantissa_mask {
            let sign_bit = if width.bits() == 64 { 1u64 << 63 } else { 1u64 << (width.bits() - 1) };
            
            // Positive NaN with different mantissa
            let pos_nan_bits = (base_nan_bits & !mantissa_mask) | i;
            nans.push(int_to_float(pos_nan_bits, width));
            
            // Negative NaN with different mantissa  
            let neg_nan_bits = pos_nan_bits | sign_bit;
            nans.push(int_to_float(neg_nan_bits, width));
        }
    }
    
    nans
}

fn special_floats_for_width(width: FloatWidth) -> Vec<f64> {
    let mut specials = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    
    // Add multiple NaN varieties
    specials.extend(generate_nan_varieties(width));
    
    // Add width-specific constants
    match width {
        FloatWidth::Width16 => {
            specials.extend([
                65504.0,  // f16::MAX
                -65504.0, // f16::MIN
                6.103515625e-5,  // f16::MIN_POSITIVE
                width.smallest_normal(),
                max_subnormal_width(width),
                -max_subnormal_width(width),
            ]);
        },
        FloatWidth::Width32 => {
            specials.extend([
                3.4028235e38,  // f32::MAX
                -3.4028235e38, // f32::MIN
                1.1754944e-38, // f32::MIN_POSITIVE
                1.1920929e-7,  // f32::EPSILON
                width.smallest_normal(),
                max_subnormal_width(width),
                -max_subnormal_width(width),
            ]);
        },
        FloatWidth::Width64 => {
            specials.extend([
                f64::MIN,
                f64::MAX,
                f64::MIN_POSITIVE,
                f64::EPSILON,
                width.smallest_normal(),
                max_subnormal_width(width),
                -max_subnormal_width(width),
            ]);
        },
    }
    
    specials
}

// Python's "weird floats" - boundary and special values with 5% probability
// This matches Python's implementation exactly (lines 788-810 in providers.py)
fn generate_weird_floats(
    min_value: Option<f64>,
    max_value: Option<f64>, 
    allow_nan: bool,
    allow_infinity: bool,
    allow_subnormal: bool,
    smallest_nonzero_magnitude: f64,
    width: FloatWidth
) -> Vec<f64> {
    let mut weird_floats = Vec::new();
    
    // Base special values (Python's weird_floats list)
    let candidates = [
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        signaling_nan(),
        negative_signaling_nan(),
    ];
    
    // Add base special values if they meet constraints
    for &candidate in &candidates {
        if is_constant_valid(candidate, min_value, max_value, allow_nan,
                           allow_infinity, allow_subnormal, 
                           smallest_nonzero_magnitude, width) {
            weird_floats.push(candidate);
        }
    }
    
    // Add dynamic boundary values based on actual bounds
    if let Some(min) = min_value {
        let boundary_candidates = [
            min,
            next_float_width(min, width),
            min + 1.0,
        ];
        
        for &candidate in &boundary_candidates {
            if is_constant_valid(candidate, min_value, max_value, allow_nan,
                               allow_infinity, allow_subnormal,
                               smallest_nonzero_magnitude, width) {
                weird_floats.push(candidate);
            }
        }
    }
    
    if let Some(max) = max_value {
        let boundary_candidates = [
            max,
            prev_float_width(max, width),
            max - 1.0,
        ];
        
        for &candidate in &boundary_candidates {
            if is_constant_valid(candidate, min_value, max_value, allow_nan,
                               allow_infinity, allow_subnormal,
                               smallest_nonzero_magnitude, width) {
                weird_floats.push(candidate);
            }
        }
    }
    
    weird_floats
}


// Adjust bounds for open intervals by finding the next/previous representable float.
// This implements the exclude_min/exclude_max functionality from Python Hypothesis.
fn adjust_bounds_for_exclusions(
    min_value: Option<f64>, 
    max_value: Option<f64>, 
    exclude_min: bool, 
    exclude_max: bool, 
    width: FloatWidth
) -> Result<(Option<f64>, Option<f64>), &'static str> {
    let mut adjusted_min = min_value;
    let mut adjusted_max = max_value;
    
    // Validate that we can't exclude None bounds
    if exclude_min && min_value.is_none() {
        return Err("Cannot exclude minimum when min_value is None");
    }
    if exclude_max && max_value.is_none() {
        return Err("Cannot exclude maximum when max_value is None");
    }
    
    // Adjust minimum bound if excluded
    if exclude_min {
        if let Some(min) = min_value {
            // Special handling for zeros - excluding either zero excludes both
            if min == 0.0 || min == -0.0 {
                adjusted_min = Some(next_float_width(0.0, width)); // Smallest positive subnormal
            } else {
                adjusted_min = Some(next_float_width(min, width));
            }
        }
    }
    
    // Adjust maximum bound if excluded  
    if exclude_max {
        if let Some(max) = max_value {
            // Special handling for zeros - excluding either zero excludes both
            if max == 0.0 || max == -0.0 {
                adjusted_max = Some(prev_float_width(0.0, width)); // Largest negative subnormal
            } else {
                adjusted_max = Some(prev_float_width(max, width));
            }
        }
    }
    
    // Validate that bounds are still valid after adjustment
    if let (Some(min), Some(max)) = (adjusted_min, adjusted_max) {
        if min > max {
            return Err("Excluding endpoints resulted in empty interval");
        }
    }
    
    Ok((adjusted_min, adjusted_max))
}

// Auto-detect whether NaN should be allowed based on bounds.
// Python Hypothesis logic: allow NaN only when both min_value and max_value are None.
fn should_allow_nan_auto(min_value: Option<f64>, max_value: Option<f64>) -> bool {
    min_value.is_none() && max_value.is_none()
}

// Auto-detect whether infinity should be allowed based on bounds.
// Python Hypothesis logic: allow infinity unless both bounds are finite and would exclude it.
fn should_allow_infinity_auto(min_value: Option<f64>, max_value: Option<f64>) -> bool {
    match (min_value, max_value) {
        (None, None) => true, // No bounds, allow infinity
        (Some(_min), None) => true, // Only lower bound, infinity could be above it
        (None, Some(_max)) => true, // Only upper bound, infinity could be below it
        (Some(min), Some(max)) => {
            // Both bounds present - don't allow infinity if both bounds are finite
            // If either bound is infinite, then infinity is in range
            !min.is_finite() || !max.is_finite()
        }
    }
}

// Auto-detect whether subnormals should be allowed based on bounds.
// This matches Python Hypothesis behavior: if the bounds require subnormals
// to represent any values in the range, then subnormals are automatically enabled.
fn should_allow_subnormals_auto(min_value: Option<f64>, max_value: Option<f64>, width: FloatWidth) -> bool {
    let smallest_normal = width.smallest_normal();
    
    match (min_value, max_value) {
        (None, None) => false, // No bounds, default to no subnormals
        (Some(min), None) => {
            // Only lower bound - need subnormals if bound is in subnormal range
            (min > 0.0 && min < smallest_normal) || (min < 0.0 && min > -smallest_normal)
        },
        (None, Some(max)) => {
            // Only upper bound - need subnormals if bound is in subnormal range  
            (max > 0.0 && max < smallest_normal) || (max < 0.0 && max > -smallest_normal)
        },
        (Some(min), Some(max)) => {
            // Both bounds present - check if range includes subnormal values
            if min > 0.0 && min < smallest_normal {
                return true;
            }
            if max < 0.0 && max > -smallest_normal {
                return true;
            }
            // If range spans zero and includes small values, we might need subnormals
            if min <= 0.0 && max >= 0.0 {
                if (min < 0.0 && min > -smallest_normal) || 
                   (max > 0.0 && max < smallest_normal) {
                    return true;
                }
            }
            false
        }
    }
}

// Validate that the bounds are compatible with subnormal settings.
// Returns an error if subnormals are disabled but required for the bounds.
fn validate_bounds_subnormal_compatibility(
    min_value: Option<f64>, 
    max_value: Option<f64>, 
    width: FloatWidth, 
    allow_subnormal: bool
) -> Result<(), &'static str> {
    if allow_subnormal {
        return Ok(()); // Always valid if subnormals are allowed
    }
    
    // If subnormals would be auto-detected as needed, but user explicitly disabled them
    if should_allow_subnormals_auto(min_value, max_value, width) {
        return Err("Bounds require subnormal numbers, but allow_subnormal=false");
    }
    
    Ok(())
}

// Check if a value can be exactly represented at the given width (Python parity)
fn validate_exact_representability(value: f64, width: FloatWidth) -> Result<f64, String> {
    let converted = int_to_float(float_to_int(value, width), width);
    if converted != value && !(value.is_nan() && converted.is_nan()) {
        return Err(format!(
            "Value {:.17} cannot be exactly represented as a float of width {} - use {:.17} instead",
            value, width.bits(), converted
        ));
    }
    Ok(converted)
}

// Enhanced environment validation (Python parity)
fn validate_float_environment() -> Result<(), &'static str> {
    // Check signed zero support (Python checks math.copysign(1.0, -0.0) == 1.0)
    let neg_zero: f64 = -0.0;
    if neg_zero.signum() >= 0.0 {
        return Err(
            "Your system can't represent -0.0, which is required by IEEE-754. \
             This is probably because it was compiled with -ffast-math."
        );
    }
    
    Ok(())
}

// Comprehensive parameter validation with helpful error messages.
fn validate_float_generation_params(
    min_value: Option<f64>,
    max_value: Option<f64>, 
    allow_nan: Option<bool>,
    allow_infinity: Option<bool>,
    allow_subnormal: Option<bool>,
    smallest_nonzero_magnitude: Option<f64>,
    exclude_min: bool,
    exclude_max: bool,
    width: FloatWidth,
) -> Result<(Option<f64>, Option<f64>), String> {
    // Validate environment first
    if let Err(msg) = validate_float_environment() {
        return Err(msg.to_string());
    }
    
    // Validate width
    if !matches!(width, FloatWidth::Width16 | FloatWidth::Width32 | FloatWidth::Width64) {
        return Err(format!("Invalid width: only 16, 32, and 64 are supported"));
    }
    
    // Validate and convert bounds to exact representations
    let validated_min = if let Some(min) = min_value {
        Some(validate_exact_representability(min, width)?)
    } else {
        None
    };
    
    let validated_max = if let Some(max) = max_value {
        Some(validate_exact_representability(max, width)?)
    } else {
        None
    };
    
    // Validate basic bound relationship
    if let (Some(min), Some(max)) = (validated_min, validated_max) {
        if min > max {
            return Err("min_value cannot be greater than max_value".to_string());
        }
    }
    
    // Validate exclude parameters
    if exclude_min && validated_min.is_none() {
        return Err("Cannot exclude minimum when min_value is None - use allow_infinity=false for finite floats".to_string());
    }
    if exclude_max && validated_max.is_none() {
        return Err("Cannot exclude maximum when max_value is None - use allow_infinity=false for finite floats".to_string());
    }
    
    // Validate allow_nan with bounds
    if let Some(true) = allow_nan {
        if validated_min.is_some() || validated_max.is_some() {
            return Err("Cannot allow NaN when min_value or max_value is specified".to_string());
        }
    }
    
    // Validate allow_infinity with finite bounds
    if let Some(true) = allow_infinity {
        if let (Some(min), Some(max)) = (validated_min, validated_max) {
            if min.is_finite() && max.is_finite() {
                return Err("Cannot allow infinity when both bounds are finite".to_string());
            }
        }
    }
    
    // Validate smallest_nonzero_magnitude
    if let Some(magnitude) = smallest_nonzero_magnitude {
        if magnitude <= 0.0 || !magnitude.is_finite() {
            return Err("smallest_nonzero_magnitude must be a positive finite number".to_string());
        }
    }
    
    // Validate subnormal compatibility
    if let Some(explicit_subnormal) = allow_subnormal {
        if let Err(msg) = validate_bounds_subnormal_compatibility(validated_min, validated_max, width, explicit_subnormal) {
            return Err(msg.to_string());
        }
    }
    
    Ok((validated_min, validated_max))
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
            if let Err(_msg) = validate_bounds_subnormal_compatibility(Some(min_value), Some(max_value), width, explicit) {
                return Err(FailedDraw);
            }
            explicit
        },
        None => {
            // Auto-detect based on bounds (Python Hypothesis behavior)
            should_allow_subnormals_auto(Some(min_value), Some(max_value), width)
        }
    };
    
    let special_floats = special_floats_for_width(width);
    
    // 5% chance of returning special values
    if source.bits(6)? == 0 {
        // Try to return a special value that fits constraints
        for special in special_floats {
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
    let mut result = lex_to_float(raw_bits, width);
    
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



// Enhanced float generation with full Python Hypothesis API compatibility.
// This function supports all the features of Python's floats() strategy:
// - Optional bounds with None support
// - Open intervals with exclude_min/exclude_max
// - Intelligent defaults for special values
// - smallest_nonzero_magnitude parameter for fine-grained control
// - Comprehensive validation with helpful error messages
/// Draw a float with support for local constants from AST parsing (e.g., Swift FFI)
pub fn draw_float_with_local_constants(
    source: &mut DataSource,
    min_value: Option<f64>,
    max_value: Option<f64>,
    allow_nan: Option<bool>,
    allow_infinity: Option<bool>,
    allow_subnormal: Option<bool>,
    smallest_nonzero_magnitude: Option<f64>,
    width: FloatWidth,
    exclude_min: bool,
    exclude_max: bool,
    local_constants: &[f64],
) -> Draw<f64> {
    // Store local constants in a thread-local or pass them through the call chain
    // For now, we'll implement this by modifying the constant pool creation directly
    _draw_float_impl(
        source, min_value, max_value, allow_nan, allow_infinity, 
        allow_subnormal, smallest_nonzero_magnitude, width, exclude_min, exclude_max, local_constants
    )
}

pub fn draw_float(
    source: &mut DataSource,
    min_value: Option<f64>,
    max_value: Option<f64>,
    allow_nan: Option<bool>,
    allow_infinity: Option<bool>,
    allow_subnormal: Option<bool>,
    smallest_nonzero_magnitude: Option<f64>,
    width: FloatWidth,
    exclude_min: bool,
    exclude_max: bool,
) -> Draw<f64> {
    draw_float_with_local_constants(
        source, min_value, max_value, allow_nan, allow_infinity, 
        allow_subnormal, smallest_nonzero_magnitude, width, exclude_min, exclude_max, &[]
    )
}

fn _draw_float_impl(
    source: &mut DataSource,
    min_value: Option<f64>,
    max_value: Option<f64>,
    allow_nan: Option<bool>,
    allow_infinity: Option<bool>,
    allow_subnormal: Option<bool>,
    smallest_nonzero_magnitude: Option<f64>,
    width: FloatWidth,
    exclude_min: bool,
    exclude_max: bool,
    local_consts: &[f64],
) -> Draw<f64> {
    // Validate all parameters and get exact representations
    let (validated_min, validated_max) = match validate_float_generation_params(
        min_value, max_value, allow_nan, allow_infinity, allow_subnormal, 
        smallest_nonzero_magnitude, exclude_min, exclude_max, width
    ) {
        Ok(bounds) => bounds,
        Err(_msg) => return Err(FailedDraw),
    };
    
    // Adjust bounds for open intervals
    let (adjusted_min, adjusted_max) = match adjust_bounds_for_exclusions(
        validated_min, validated_max, exclude_min, exclude_max, width
    ) {
        Ok(bounds) => bounds,
        Err(_msg) => return Err(FailedDraw),
    };
    
    // Determine intelligent defaults for special values
    let nan_allowed = allow_nan.unwrap_or_else(|| should_allow_nan_auto(adjusted_min, adjusted_max));
    let infinity_allowed = allow_infinity.unwrap_or_else(|| should_allow_infinity_auto(adjusted_min, adjusted_max));
    let subnormal_allowed = allow_subnormal.unwrap_or_else(|| should_allow_subnormals_auto(adjusted_min, adjusted_max, width));
    
    // Use provided smallest_nonzero_magnitude or infer from subnormal settings
    let effective_smallest_nonzero = if let Some(magnitude) = smallest_nonzero_magnitude {
        magnitude
    } else if subnormal_allowed {
        min_positive_subnormal_width(width)
    } else {
        width.smallest_normal()
    };
    
    // **NEW: Constant Injection System (15% probability like Python)**
    // This is the most impactful missing feature from Python Hypothesis
    if source.bits(7)? < 19 { // 19/128 ≈ 15% probability (Python uses p=0.15)
        let mut constant_pool = ConstantPool::with_local_constants(local_consts);
        let valid_constants = constant_pool.get_valid_constants(
            adjusted_min, adjusted_max, nan_allowed, infinity_allowed,
            subnormal_allowed, effective_smallest_nonzero, width
        );
        
        if !valid_constants.is_empty() {
            let index = source.bits(16)? as usize % valid_constants.len();
            return Ok(valid_constants[index]);
        }
    }
    
    // Generate sophisticated special values (Python parity)
    let mut special_candidates = special_floats_for_width(width);
    special_candidates.extend(generate_weird_floats(adjusted_min, adjusted_max, true, true, true, 0.0, width));
    
    // Filter special values that meet constraints
    let valid_specials: Vec<f64> = special_candidates
        .into_iter()
        .filter(|&val| {
            (!val.is_nan() || nan_allowed)
                && (!val.is_infinite() || infinity_allowed)
                && (!is_subnormal_width(val, width) || subnormal_allowed)
                && (val == 0.0 || val.abs() >= effective_smallest_nonzero)
                && (adjusted_min.map_or(true, |min| val >= min))
                && (adjusted_max.map_or(true, |max| val <= max))
        })
        .collect();
    
    // 5% chance of special values, then 5% chance of boundary values
    if !valid_specials.is_empty() && source.bits(5)? == 0 {
        // Use entropy to select from valid special values
        let index = source.bits(16)? as usize % valid_specials.len();
        return Ok(valid_specials[index]);
    }
    
    // Convert to concrete bounds for the core generation logic
    let concrete_min = adjusted_min.unwrap_or(f64::NEG_INFINITY);
    let concrete_max = adjusted_max.unwrap_or(f64::INFINITY);
    
    // Call the core implementation with resolved parameters
    draw_float_width_with_subnormals_impl(
        source, width, concrete_min, concrete_max, 
        nan_allowed, infinity_allowed, Some(subnormal_allowed), Some(effective_smallest_nonzero)
    )
}

// Internal implementation that works with concrete bounds.
// This is the core generation logic separated from parameter processing.
fn draw_float_width_with_subnormals_impl(
    source: &mut DataSource,
    width: FloatWidth,
    min_value: f64,
    max_value: f64,
    allow_nan: bool,
    allow_infinity: bool,
    allow_subnormal: Option<bool>,
    smallest_nonzero_magnitude: Option<f64>,
) -> Draw<f64> {
    // Determine subnormal policy
    let subnormal_allowed = match allow_subnormal {
        Some(explicit) => explicit,
        None => {
            // Auto-detect based on bounds (Python Hypothesis behavior)
            should_allow_subnormals_auto(Some(min_value), Some(max_value), width)
        }
    };
    
    // Use provided smallest_nonzero_magnitude or infer from subnormal settings
    let effective_smallest_nonzero = if let Some(magnitude) = smallest_nonzero_magnitude {
        magnitude
    } else if subnormal_allowed {
        min_positive_subnormal_width(width)
    } else {
        width.smallest_normal()
    };
    
    
    // **NEW: "Weird Floats" Generation (5% probability like Python)**
    // This provides boundary-focused generation beyond constants
    if source.bits(5)? == 0 { // 1/32 ≈ 3.125%, close to Python's 5%
        let weird_floats = generate_weird_floats(
            Some(min_value), Some(max_value), allow_nan, allow_infinity,
            subnormal_allowed, effective_smallest_nonzero, width
        );
        
        if !weird_floats.is_empty() {
            let index = source.bits(8)? as usize % weird_floats.len();
            return Ok(weird_floats[index]);
        }
    }
    
    // Fallback to basic special values (legacy behavior, reduced probability)
    let special_floats = special_floats_for_width(width);
    if source.bits(8)? == 0 { // Reduced from 6 bits to 8 bits (lower probability)
        // Try to return a special value that fits constraints
        for special in special_floats {
            if (!special.is_nan() || allow_nan)
                && (!special.is_infinite() || allow_infinity)
                && (!is_subnormal_width(special, width) || subnormal_allowed)
                && (special == 0.0 || special.abs() >= effective_smallest_nonzero)
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
    let mut result = lex_to_float(raw_bits, width);
    
    // Apply random sign
    if source.bits(1)? == 1 {
        result = -result;
    }
    
    // **NEW: Use sophisticated clamper for better constraint handling**
    // This replaces the basic clamping with Python's mantissa-based approach
    let clamper = make_float_clamper(
        min_value, max_value, allow_nan, effective_smallest_nonzero, width
    );
    
    // Handle NaN - if not allowed, use clamper to generate alternative
    if result.is_nan() && !allow_nan {
        // Use clamper with a fallback finite value
        let fallback_bits = source.bits(width.mantissa_bits() as u64 + 1)?;
        let fallback = (fallback_bits as f64) / (1u64 << (width.mantissa_bits() + 1)) as f64;
        result = clamper(fallback);
        return Ok(result);
    }
    
    // Handle infinity - if not allowed, use clamper  
    if result.is_infinite() && !allow_infinity {
        // Generate a large finite value and let clamper handle it
        let large_finite = if result.is_sign_positive() {
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
        result = clamper(large_finite);
        return Ok(result);
    }
    
    // Handle subnormal exclusion - let clamper handle this sophisticated logic
    if is_subnormal_width(result, width) && !subnormal_allowed {
        // The clamper will handle smallest_nonzero_magnitude constraints properly
        result = clamper(result);
        return Ok(result);
    }
    
    // Apply the sophisticated clamper to handle all constraints
    result = clamper(result);
    
    Ok(result)
}

// Convenience function matching Python Hypothesis floats() signature exactly.
// This provides the most user-friendly API with intelligent defaults.
pub fn floats(
    min_value: Option<f64>,
    max_value: Option<f64>,
    allow_nan: Option<bool>,
    allow_infinity: Option<bool>,
    allow_subnormal: Option<bool>,
    smallest_nonzero_magnitude: Option<f64>,
    width: FloatWidth,
    exclude_min: bool,
    exclude_max: bool,
) -> impl Fn(&mut DataSource) -> Draw<f64> {
    move |source: &mut DataSource| {
        draw_float(
            source, min_value, max_value, allow_nan, allow_infinity, 
            allow_subnormal, smallest_nonzero_magnitude, width, exclude_min, exclude_max
        )
    }
}

/// Generate floats with local constants from AST parsing (e.g., Swift FFI)
pub fn floats_with_local_constants(
    min_value: Option<f64>,
    max_value: Option<f64>,
    allow_nan: Option<bool>,
    allow_infinity: Option<bool>,
    allow_subnormal: Option<bool>,
    smallest_nonzero_magnitude: Option<f64>,
    width: FloatWidth,
    exclude_min: bool,
    exclude_max: bool,
    local_constants: Vec<f64>,
) -> impl Fn(&mut DataSource) -> Draw<f64> {
    move |source: &mut DataSource| {
        draw_float_with_local_constants(
            source, min_value, max_value, allow_nan, allow_infinity, 
            allow_subnormal, smallest_nonzero_magnitude, width, exclude_min, exclude_max, &local_constants
        )
    }
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
pub fn draw_float_from_parts(source: &mut DataSource, width: FloatWidth) -> Draw<f64> {
    let raw_bits = source.bits(width.bits() as u64)?;
    Ok(lex_to_float(raw_bits, width))
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


// Draw a float with uniform distribution in range with width support.
pub fn draw_float_uniform(
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



#[cfg(test)]
mod tests;
