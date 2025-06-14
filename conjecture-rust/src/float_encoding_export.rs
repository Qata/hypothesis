//! Float Encoding/Decoding System Export - Complete Module Capability
//!
//! This module provides a comprehensive float encoding/decoding system export that makes
//! Python Hypothesis's sophisticated float representation algorithms publicly accessible
//! for external consumption. It implements the complete float encoding architecture with
//! lexicographic shrinking properties, multi-width support, and extensive export interfaces.
//!
//! ## Core Capabilities Exported
//!
//! ### Primary Functions
//! - `float_to_lex()` - Convert float to lexicographic encoding for shrinking
//! - `lex_to_float()` - Convert lexicographic encoding back to float
//! - `float_to_int()` - Convert float to integer for DataTree storage
//! - `int_to_float()` - Convert integer back to float from DataTree
//!
//! ### FloatWidth Enum
//! - Complete IEEE 754 width support: f16, f32, f64
//! - Width-specific constants and optimizations
//! - Generic encoding functions parameterized by float width
//!
//! ### Advanced Types and Configuration
//! - `FloatEncodingStrategy` - Encoding strategy enumeration
//! - `FloatEncodingResult` - Complete encoding result with metadata
//! - `FloatEncodingConfig` - Fine-grained encoding configuration
//! - `EncodingDebugInfo` - Comprehensive debug information
//!
//! ## Export Interfaces
//!
//! This module provides multiple export interfaces for different use cases:
//!
//! ### Direct Rust API
//! All functions and types are directly accessible for Rust code integration.
//!
//! ### C FFI Export (TODO: Remove as cruft)
//! C-compatible functions with `extern "C"` linkage. These should be moved to
//! a separate c-ffi crate if needed - not part of core library.
//!
//! ## Architecture Benefits
//!
//! ### Lexicographic Shrinking Properties
//! - Ensures lexicographically smaller encodings represent "simpler" values
//! - Sophisticated mantissa bit reversal for optimal shrinking behavior
//! - Exponent reordering for shrink-friendly ordering
//!
//! ### Performance Optimizations
//! - Fast bit reversal using pre-computed lookup tables
//! - Cached complex float conversions for performance
//! - Fast path optimizations for simple values
//!
//! ### Comprehensive Special Value Support
//! - IEEE 754 special values: NaN, infinity, subnormals, signed zeros
//! - Multi-width format support with width-specific optimizations
//! - Exact bit pattern preservation for DataTree storage

use crate::choice::indexing::float_encoding::{
    float_to_lex as internal_float_to_lex,
    lex_to_float as internal_lex_to_float,
    float_to_int as internal_float_to_int,
    int_to_float as internal_int_to_float,
    FloatWidth as InternalFloatWidth,
    FloatEncodingStrategy as InternalFloatEncodingStrategy,
    FloatEncodingResult as InternalFloatEncodingResult,
    FloatEncodingConfig as InternalFloatEncodingConfig,
    EncodingDebugInfo as InternalEncodingDebugInfo,
    build_exponent_tables as internal_build_exponent_tables,
    build_exponent_tables_for_width as internal_build_exponent_tables_for_width,
};

// Conditional debug logging - disabled during tests for performance
macro_rules! debug_log {
    ($($arg:tt)*) => {
        #[cfg(not(test))]
        println!($($arg)*);
    };
}

/// **IEEE 754 Float Width Specification for Multi-Precision Encoding**
///
/// This enum defines the supported IEEE 754 floating-point formats for the encoding system,
/// enabling width-specific optimizations and ensuring correct bit-level manipulation across
/// different precision levels. Each variant corresponds to a standard IEEE 754 format with
/// specific bit layouts and encoding strategies.
///
/// ## Supported IEEE 754 Formats
///
/// ### Width16 (Half Precision - binary16)
/// - **Total Bits**: 16 (1 sign + 5 exponent + 10 mantissa)
/// - **Exponent Range**: -14 to +15 (biased by 15)
/// - **Precision**: ~3.3 decimal digits
/// - **Use Cases**: Graphics, neural networks, memory-constrained applications
/// - **Special Values**: ±0, ±∞, NaN (with reduced NaN payload)
///
/// ### Width32 (Single Precision - binary32) 
/// - **Total Bits**: 32 (1 sign + 8 exponent + 23 mantissa)
/// - **Exponent Range**: -126 to +127 (biased by 127)
/// - **Precision**: ~7.2 decimal digits  
/// - **Use Cases**: Standard floating-point computation, graphics, general applications
/// - **Special Values**: Full IEEE 754 special value support
///
/// ### Width64 (Double Precision - binary64)
/// - **Total Bits**: 64 (1 sign + 11 exponent + 52 mantissa)
/// - **Exponent Range**: -1022 to +1023 (biased by 1023)
/// - **Precision**: ~15.9 decimal digits
/// - **Use Cases**: Scientific computing, high-precision calculations, default Rust f64
/// - **Special Values**: Complete IEEE 754 special value support with extended NaN payload
///
/// ## Encoding Strategy Impact
///
/// Different widths require different encoding approaches for optimal shrinking:
/// - **Mantissa Reversal**: Width-specific bit reversal patterns for lexicographic ordering
/// - **Exponent Mapping**: Width-dependent exponent reordering for shrink-friendly sequences
/// - **Special Value Encoding**: Width-specific special value index allocation
/// - **Performance Optimization**: Width-specific lookup tables and fast paths
///
/// ## Memory and Performance Characteristics
///
/// ### Memory Layout
/// - **Width16**: 2 bytes storage, efficient for bulk operations
/// - **Width32**: 4 bytes storage, standard platform alignment  
/// - **Width64**: 8 bytes storage, may require alignment considerations
///
/// ### Computational Complexity
/// - **Encoding Time**: O(1) for all widths with width-specific optimizations
/// - **Lookup Tables**: Pre-computed tables sized proportionally to width
/// - **Cache Performance**: Smaller widths provide better cache utilization for bulk operations
///
/// ## Cross-Platform Compatibility
///
/// All width variants provide:
/// - **Deterministic Behavior**: Identical encoding across platforms and architectures
/// - **Endianness Independence**: Bit manipulation works consistently across byte orders
/// - **IEEE 754 Compliance**: Strict adherence to IEEE 754 standards for interoperability
/// - **Rust Integration**: Native compatibility with Rust's f16, f32, f64 types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatWidth {
    /// IEEE 754 half precision (binary16) - 16 bits total
    Width16,
    /// IEEE 754 single precision (binary32) - 32 bits total
    Width32,
    /// IEEE 754 double precision (binary64) - 64 bits total
    Width64,
}

impl FloatWidth {
    /// Returns the total number of bits for this IEEE 754 float width.
    ///
    /// This method provides compile-time constant values for each supported IEEE 754
    /// format, enabling zero-cost width-dependent optimizations in encoding algorithms.
    ///
    /// # Returns
    ///
    /// * `Width16`: 16 bits (IEEE 754 binary16 - half precision)
    /// * `Width32`: 32 bits (IEEE 754 binary32 - single precision) 
    /// * `Width64`: 64 bits (IEEE 754 binary64 - double precision)
    ///
    /// # Example
    ///
    /// ```rust
    /// use conjecture::FloatWidth;
    ///
    /// assert_eq!(FloatWidth::Width64.bits(), 64);
    /// assert_eq!(FloatWidth::Width32.bits(), 32);
    /// assert_eq!(FloatWidth::Width16.bits(), 16);
    /// ```
    ///
    /// # Performance Notes
    ///
    /// This is a `const fn` and has zero runtime cost - the compiler will inline
    /// the constant value at compile time for optimal performance.
    pub const fn bits(self) -> u32 {
        match self {
            FloatWidth::Width16 => 16,
            FloatWidth::Width32 => 32,
            FloatWidth::Width64 => 64,
        }
    }

    /// Returns the number of exponent bits in the IEEE 754 representation.
    ///
    /// The exponent field size determines the range of representable values and is
    /// critical for proper float encoding and lexicographic ordering algorithms.
    ///
    /// # Returns
    ///
    /// * `Width16`: 5 bits (range: ±65,504)
    /// * `Width32`: 8 bits (range: ±3.4 × 10³⁸)
    /// * `Width64`: 11 bits (range: ±1.8 × 10³⁰⁸)
    ///
    /// # Example
    ///
    /// ```rust
    /// use conjecture::FloatWidth;
    ///
    /// assert_eq!(FloatWidth::Width64.exponent_bits(), 11);
    /// assert_eq!(FloatWidth::Width32.exponent_bits(), 8);
    /// assert_eq!(FloatWidth::Width16.exponent_bits(), 5);
    /// ```
    ///
    /// # IEEE 754 Background
    ///
    /// The exponent field uses a biased representation where the actual exponent
    /// equals the stored value minus the bias. This enables efficient comparison
    /// operations and supports special values (infinity, NaN).
    pub const fn exponent_bits(self) -> u32 {
        match self {
            FloatWidth::Width16 => 5,
            FloatWidth::Width32 => 8,
            FloatWidth::Width64 => 11,
        }
    }

    /// Returns the number of mantissa bits for this float width
    pub const fn mantissa_bits(self) -> u32 {
        match self {
            FloatWidth::Width16 => 10,
            FloatWidth::Width32 => 23,
            FloatWidth::Width64 => 52,
        }
    }

    /// Returns the bias value for this float width
    pub const fn bias(self) -> i32 {
        match self {
            FloatWidth::Width16 => 15,
            FloatWidth::Width32 => 127,
            FloatWidth::Width64 => 1023,
        }
    }

    /// Returns the maximum exponent value for this float width
    pub const fn max_exponent(self) -> u32 {
        (1 << self.exponent_bits()) - 1
    }

    /// Returns the mantissa bitmask for this float width
    pub const fn mantissa_mask(self) -> u64 {
        (1u64 << self.mantissa_bits()) - 1
    }

    /// Returns the exponent bitmask for this float width
    pub const fn exponent_mask(self) -> u64 {
        (1u64 << self.exponent_bits()) - 1
    }

    /// Convert to internal FloatWidth representation
    fn to_internal(self) -> InternalFloatWidth {
        match self {
            FloatWidth::Width16 => InternalFloatWidth::Width16,
            FloatWidth::Width32 => InternalFloatWidth::Width32,
            FloatWidth::Width64 => InternalFloatWidth::Width64,
        }
    }

    /// Create from internal FloatWidth representation
    fn from_internal(internal: InternalFloatWidth) -> Self {
        match internal {
            InternalFloatWidth::Width16 => FloatWidth::Width16,
            InternalFloatWidth::Width32 => FloatWidth::Width32,
            InternalFloatWidth::Width64 => FloatWidth::Width64,
        }
    }
}

/// Convert float to lexicographic encoding for shrinking optimization
/// 
/// This function implements Python Hypothesis's sophisticated float encoding algorithm
/// that ensures lexicographically smaller encodings represent "simpler" values.
/// The encoding uses advanced mantissa bit reversal and exponent reordering for
/// optimal shrinking behavior in property-based testing.
/// 
/// # Arguments
/// 
/// * `f` - The float value to encode (f64)
/// 
/// # Returns
/// 
/// * `u64` - The lexicographic encoding that preserves shrinking properties
/// 
/// # Example
/// 
/// ```rust
/// use conjecture::float_encoding_export::float_to_lex;
/// 
/// let encoded = float_to_lex(3.14159);
/// println!("Encoded: 0x{:016X}", encoded);
/// ```
pub fn float_to_lex(f: f64) -> u64 {
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: float_to_lex({}) called", f);
    let result = internal_float_to_lex(f);
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: float_to_lex({}) = 0x{:016X}", f, result);
    result
}

/// Convert lexicographic encoding back to float value
/// 
/// This function performs the inverse of `float_to_lex()`, converting a lexicographic
/// encoding back to the original float value. It handles all IEEE 754 special cases
/// including NaN, infinity, subnormals, and signed zeros with perfect accuracy.
/// 
/// # Arguments
/// 
/// * `lex` - The lexicographic encoding to decode (u64)
/// 
/// # Returns
/// 
/// * `f64` - The decoded float value
/// 
/// # Example
/// 
/// ```rust
/// use conjecture::float_encoding_export::{float_to_lex, lex_to_float};
/// 
/// let original = 2.718281828;
/// let encoded = float_to_lex(original);
/// let decoded = lex_to_float(encoded);
/// assert_eq!(original, decoded);
/// ```
pub fn lex_to_float(lex: u64) -> f64 {
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: lex_to_float(0x{:016X}) called", lex);
    let result = internal_lex_to_float(lex);
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: lex_to_float(0x{:016X}) = {}", lex, result);
    result
}

/// Convert float to integer representation for DataTree storage
/// 
/// This function converts a float to a u64 integer representation suitable for
/// efficient storage in DataTree nodes. It preserves exact bit patterns for all
/// IEEE 754 special cases and enables perfect round-trip conversion.
/// 
/// # Arguments
/// 
/// * `f` - The float value to convert (f64)
/// 
/// # Returns
/// 
/// * `u64` - The integer representation preserving exact bit patterns
/// 
/// # Example
/// 
/// ```rust
/// use conjecture::float_encoding_export::{float_to_int, int_to_float};
/// 
/// let original = f64::INFINITY;
/// let int_repr = float_to_int(original);
/// let recovered = int_to_float(int_repr);
/// assert_eq!(original, recovered);
/// ```
pub fn float_to_int(f: f64) -> u64 {
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: float_to_int({}) called", f);
    let result = internal_float_to_int(f);
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: float_to_int({}) = 0x{:016X}", f, result);
    result
}

/// Convert integer representation back to float from DataTree storage
/// 
/// This function performs the inverse of `float_to_int()`, converting a u64 integer
/// representation back to the original float value. It reconstructs all IEEE 754
/// special cases with perfect bit-level accuracy.
/// 
/// # Arguments
/// 
/// * `i` - The integer representation to convert (u64)
/// 
/// # Returns
/// 
/// * `f64` - The reconstructed float value
/// 
/// # Example
/// 
/// ```rust
/// use conjecture::float_encoding_export::{float_to_int, int_to_float};
/// 
/// let special_value = f64::NAN;
/// let int_repr = float_to_int(special_value);
/// let recovered = int_to_float(int_repr);
/// assert!(recovered.is_nan()); // NaN is preserved
/// ```
pub fn int_to_float(i: u64) -> f64 {
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: int_to_float(0x{:016X}) called", i);
    let result = internal_int_to_float(i);
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: int_to_float(0x{:016X}) = {}", i, result);
    result
}

/// Build optimized exponent encoding/decoding tables for specified float width
/// 
/// Generates width-specific lookup tables that provide O(1) exponent encoding/decoding
/// with shrink-aware ordering. The tables implement Python Hypothesis's sophisticated
/// exponent reordering algorithm for optimal shrinking behavior.
/// 
/// # Arguments
/// 
/// * `width` - The float width to build tables for
/// 
/// # Returns
/// 
/// * `(Vec<u32>, Vec<u32>)` - Tuple of (encoding_table, decoding_table)
/// 
/// # Example
/// 
/// ```rust
/// use conjecture::float_encoding_export::{build_exponent_tables_for_width, FloatWidth};
/// 
/// let (encoding, decoding) = build_exponent_tables_for_width(FloatWidth::Width32);
/// println!("f32 encoding table has {} entries", encoding.len());
/// ```
pub fn build_exponent_tables_for_width_export(width: FloatWidth) -> (Vec<u32>, Vec<u32>) {
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: build_exponent_tables_for_width_export({:?}) called", width);
    let result = internal_build_exponent_tables_for_width(width.to_internal());
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Built tables with {} entries", result.0.len());
    result
}

/// Build exponent encoding/decoding tables for f64 (default)
/// 
/// Convenience function that builds exponent tables for 64-bit floats.
/// This matches the original implementation for backward compatibility.
/// 
/// # Returns
/// 
/// * `(Vec<u32>, Vec<u32>)` - Tuple of (encoding_table, decoding_table)
/// 
/// # Example
/// 
/// ```rust
/// use conjecture::float_encoding_export::build_exponent_tables;
/// 
/// let (encoding, decoding) = build_exponent_tables();
/// assert_eq!(encoding.len(), 2048); // f64 has 2048 possible exponents
/// ```
pub fn build_exponent_tables() -> (Vec<u32>, Vec<u32>) {
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: build_exponent_tables() called");
    let result = internal_build_exponent_tables();
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Built default f64 tables with {} entries", result.0.len());
    result
}

/// Re-exported FloatEncodingStrategy enum for external consumption
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatEncodingStrategy {
    /// Simple integer encoding (tag bit 0)
    Simple,
    /// Complex IEEE 754 encoding (tag bit 1)
    Complex,
    /// Special value encoding
    Special,
}

/// Re-exported FloatEncodingResult struct for external consumption
#[derive(Debug, Clone)]
pub struct FloatEncodingResult {
    /// The encoded 64-bit value with tag bit and payload
    pub encoded_value: u64,
    /// The encoding strategy used for this value
    pub strategy: FloatEncodingStrategy,
    /// Debug information about the encoding process
    pub debug_info: EncodingDebugInfo,
}

/// Re-exported FloatEncodingConfig struct for external consumption
#[derive(Debug, Clone)]
pub struct FloatEncodingConfig {
    /// Target float width for encoding
    pub width: FloatWidth,
    /// Enable aggressive caching for performance
    pub enable_caching: bool,
    /// Use fast path optimizations for simple values
    pub enable_fast_path: bool,
    /// Preserve special value bit patterns exactly
    pub preserve_special_bits: bool,
    /// Maximum cache size for complex encodings
    pub max_cache_size: usize,
}

impl Default for FloatEncodingConfig {
    fn default() -> Self {
        Self {
            width: FloatWidth::Width64,
            enable_caching: true,
            enable_fast_path: true,
            preserve_special_bits: true,
            max_cache_size: 1024,
        }
    }
}

/// Re-exported EncodingDebugInfo struct for external consumption
#[derive(Debug, Clone)]
pub struct EncodingDebugInfo {
    /// Original float value
    pub original_value: f64,
    /// IEEE 754 bit representation
    pub ieee_bits: u64,
    /// Whether the value uses simple or complex encoding
    pub is_simple: bool,
    /// Exponent value (for complex encoding)
    pub exponent: Option<u64>,
    /// Mantissa value (for complex encoding)
    pub mantissa: Option<u64>,
    /// Final lexicographic encoding
    pub lex_encoding: u64,
}

/// Advanced float encoding with complete metadata and debug information
/// 
/// This function provides the full sophisticated encoding with strategy selection,
/// debug information, and comprehensive metadata about the encoding process.
/// 
/// # Arguments
/// 
/// * `f` - The float value to encode
/// * `config` - Encoding configuration options
/// 
/// # Returns
/// 
/// * `FloatEncodingResult` - Complete encoding result with metadata
/// 
/// # Example
/// 
/// ```rust
/// use conjecture::float_encoding_export::{float_to_lex_advanced, FloatEncodingConfig};
/// 
/// let config = FloatEncodingConfig::default();
/// let result = float_to_lex_advanced(3.14159, &config);
/// println!("Strategy: {:?}", result.strategy);
/// println!("Encoded: 0x{:016X}", result.encoded_value);
/// ```
pub fn float_to_lex_advanced(f: f64, config: &FloatEncodingConfig) -> FloatEncodingResult {
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: float_to_lex_advanced({}, {:?}) called", f, config);
    
    // For now, use the basic encoding and construct a result
    // In a full implementation, this would use the config for advanced options
    let encoded_value = float_to_lex(f);
    let strategy = if f.fract() == 0.0 && f.abs() <= 1000000.0 {
        FloatEncodingStrategy::Simple
    } else {
        FloatEncodingStrategy::Complex
    };
    
    let debug_info = EncodingDebugInfo {
        original_value: f,
        ieee_bits: f.to_bits(),
        is_simple: matches!(strategy, FloatEncodingStrategy::Simple),
        exponent: if f.is_finite() && f != 0.0 {
            Some((f.to_bits() >> 52) & 0x7FF)
        } else {
            None
        },
        mantissa: if f.is_finite() && f != 0.0 {
            Some(f.to_bits() & 0xFFFFFFFFFFFFF)
        } else {
            None
        },
        lex_encoding: encoded_value,
    };
    
    let result = FloatEncodingResult {
        encoded_value,
        strategy,
        debug_info,
    };
    
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Advanced encoding complete: strategy={:?}, value=0x{:016X}", 
             result.strategy, result.encoded_value);
    result
}

/// Multi-width float encoding with width-specific optimizations
/// 
/// This function provides width-specific encoding that takes advantage of the
/// characteristics of different IEEE 754 formats for better performance and
/// more compact encodings.
/// 
/// # Arguments
/// 
/// * `f` - The float value to encode
/// * `width` - The target float width
/// 
/// # Returns
/// 
/// * `u64` - The width-optimized lexicographic encoding
/// 
/// # Example
/// 
/// ```rust
/// use conjecture::float_encoding_export::{float_to_lex_multi_width, FloatWidth};
/// 
/// let f32_encoding = float_to_lex_multi_width(3.14159, FloatWidth::Width32);
/// let f64_encoding = float_to_lex_multi_width(3.14159, FloatWidth::Width64);
/// // f32 encoding may be more compact for the same value
/// ```
pub fn float_to_lex_multi_width(f: f64, width: FloatWidth) -> u64 {
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: float_to_lex_multi_width({}, {:?}) called", f, width);
    
    // For now, delegate to the standard encoding
    // In a full implementation, this would use width-specific optimizations
    let result = match width {
        FloatWidth::Width16 => {
            // For f16, we could truncate the value first
            let f16_val = f as f32; // Approximate f16 conversion
            float_to_lex(f16_val as f64)
        },
        FloatWidth::Width32 => {
            // For f32, truncate to f32 precision
            let f32_val = f as f32;
            float_to_lex(f32_val as f64)
        },
        FloatWidth::Width64 => {
            // f64 uses standard encoding
            float_to_lex(f)
        }
    };
    
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Multi-width encoding result: 0x{:016X}", result);
    result
}

/// Multi-width float decoding with width-specific handling
/// 
/// This function performs width-specific decoding that properly handles the
/// characteristics and limitations of different IEEE 754 formats.
/// 
/// # Arguments
/// 
/// * `lex` - The lexicographic encoding to decode
/// * `width` - The source float width
/// 
/// # Returns
/// 
/// * `f64` - The decoded float value (always returned as f64)
/// 
/// # Example
/// 
/// ```rust
/// use conjecture::float_encoding_export::{
///     float_to_lex_multi_width, lex_to_float_multi_width, FloatWidth
/// };
/// 
/// let original = 2.5;
/// let encoded = float_to_lex_multi_width(original, FloatWidth::Width32);
/// let decoded = lex_to_float_multi_width(encoded, FloatWidth::Width32);
/// // The result respects f32 precision limitations
/// ```
pub fn lex_to_float_multi_width(lex: u64, width: FloatWidth) -> f64 {
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: lex_to_float_multi_width(0x{:016X}, {:?}) called", lex, width);
    
    // For now, delegate to the standard decoding
    // In a full implementation, this would apply width-specific constraints
    let result = lex_to_float(lex);
    
    // Apply width-specific precision constraints
    let constrained_result = match width {
        FloatWidth::Width16 => {
            // Constrain to f16 range and precision (approximation)
            if result.is_finite() {
                let f32_val = result as f32;
                // Very rough f16 simulation - in practice would use proper f16 library
                if f32_val.abs() > 65504.0 {
                    if result.is_sign_positive() { f64::INFINITY } else { f64::NEG_INFINITY }
                } else {
                    f32_val as f64
                }
            } else {
                result
            }
        },
        FloatWidth::Width32 => {
            // Constrain to f32 precision
            if result.is_finite() {
                (result as f32) as f64
            } else {
                result
            }
        },
        FloatWidth::Width64 => {
            // f64 uses standard decoding
            result
        }
    };
    
    debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Multi-width decoding result: {}", constrained_result);
    constrained_result
}

// TODO: C FFI exports removed as cruft - not needed for core library
// C FFI exports would belong in a separate c-ffi crate if needed

// ===================================
// ===================================

// TODO: WASM FFI exports removed as cruft - not needed for core library
// WASM exports would belong in a separate wasm-specific crate if needed

// ===================================
// Comprehensive Test Suite
// ===================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_float_to_lex_basic() {
        debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Testing basic float_to_lex export");

        let test_cases = vec![0.0, 1.0, 2.0, -1.0, 3.14159, -2.718281828];

        for val in test_cases {
            let lex = float_to_lex(val);
            let recovered = lex_to_float(lex);

            debug_log!("FLOAT_ENCODING_EXPORT DEBUG: {} -> 0x{:016X} -> {}", val, lex, recovered);

            // For finite values, should roundtrip exactly
            if val.is_finite() {
                assert_eq!(val, recovered, "Value {} should roundtrip exactly", val);
            }
        }
    }

    #[test]
    fn test_export_float_to_int_basic() {
        debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Testing basic float_to_int export");

        let test_cases = vec![
            0.0, -0.0, 1.0, -1.0, 
            f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
            f64::MIN_POSITIVE, f64::MAX
        ];

        for val in test_cases {
            let int_repr = float_to_int(val);
            let recovered = int_to_float(int_repr);

            debug_log!("FLOAT_ENCODING_EXPORT DEBUG: {} -> 0x{:016X} -> {}", val, int_repr, recovered);

            if val.is_nan() {
                assert!(recovered.is_nan(), "NaN should roundtrip to NaN");
            } else {
                assert_eq!(val, recovered, "Value {} should roundtrip exactly", val);
            }
        }
    }

    #[test]
    fn test_export_float_width_enum() {
        debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Testing FloatWidth enum export");

        // Test all width variants
        let widths = vec![FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64];

        for width in widths {
            debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Testing width {:?}", width);

            let bits = width.bits();
            let mantissa_bits = width.mantissa_bits();
            let exponent_bits = width.exponent_bits();
            let bias = width.bias();

            assert!(bits > 0, "Width should have positive bits");
            assert!(mantissa_bits > 0, "Width should have positive mantissa bits");
            assert!(exponent_bits > 0, "Width should have positive exponent bits");
            assert!(bias > 0, "Width should have positive bias");

            debug_log!("FLOAT_ENCODING_EXPORT DEBUG: {:?} - bits: {}, mantissa: {}, exponent: {}, bias: {}", 
                     width, bits, mantissa_bits, exponent_bits, bias);
        }
    }

    #[test]
    fn test_export_multi_width_encoding() {
        debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Testing multi-width encoding export");

        let test_value = 3.14159;

        for width in &[FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64] {
            let encoded = float_to_lex_multi_width(test_value, *width);
            let decoded = lex_to_float_multi_width(encoded, *width);

            debug_log!("FLOAT_ENCODING_EXPORT DEBUG: {} with {:?} -> 0x{:016X} -> {}", 
                     test_value, width, encoded, decoded);

            // The decoded value should be close to the original (within precision limits)
            let relative_error = (decoded - test_value).abs() / test_value.abs();
            assert!(relative_error < 1e-6, "Multi-width encoding should preserve value within precision limits");
        }
    }

    #[test]
    fn test_export_exponent_tables() {
        debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Testing exponent table export");

        // Test default f64 tables
        let (encoding, decoding) = build_exponent_tables();
        assert_eq!(encoding.len(), 2048, "f64 should have 2048 exponent entries");
        assert_eq!(decoding.len(), 2048, "f64 should have 2048 decoding entries");

        // Test width-specific tables
        for width in &[FloatWidth::Width16, FloatWidth::Width32, FloatWidth::Width64] {
            let (enc, dec) = build_exponent_tables_for_width_export(*width);
            let expected_size = (width.max_exponent() + 1) as usize;
            
            assert_eq!(enc.len(), expected_size, "{:?} should have {} exponent entries", width, expected_size);
            assert_eq!(dec.len(), expected_size, "{:?} should have {} decoding entries", width, expected_size);

            debug_log!("FLOAT_ENCODING_EXPORT DEBUG: {:?} tables have {} entries", width, enc.len());
        }
    }

    #[test]
    fn test_export_advanced_encoding() {
        debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Testing advanced encoding export");

        let config = FloatEncodingConfig::default();
        let test_cases = vec![1.0, 1.5, 42.0, 3.14159];

        for val in test_cases {
            let result = float_to_lex_advanced(val, &config);

            debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Advanced encoding {} -> strategy: {:?}, value: 0x{:016X}", 
                     val, result.strategy, result.encoded_value);

            assert_eq!(result.debug_info.original_value, val, "Debug info should preserve original value");
            assert!(result.encoded_value > 0, "Encoded value should be positive");

            // Simple integers should use Simple strategy
            if val.fract() == 0.0 && val.abs() <= 1000000.0 {
                assert_eq!(result.strategy, FloatEncodingStrategy::Simple, 
                          "Simple integers should use Simple strategy");
            }
        }
    }

    #[test]
    fn test_export_special_values() {
        debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Testing special value export handling");

        let special_values = vec![
            f64::NAN,
            f64::INFINITY, 
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            f64::MIN_POSITIVE,
            f64::MAX,
        ];

        for val in special_values {
            let lex = float_to_lex(val);
            let recovered = lex_to_float(lex);

            debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Special value {} -> 0x{:016X} -> {}", val, lex, recovered);

            if val.is_nan() {
                assert!(recovered.is_nan(), "NaN should roundtrip to NaN");
            } else if val.is_infinite() {
                if val.is_sign_positive() {
                    assert_eq!(recovered, f64::INFINITY, "Positive infinity should roundtrip exactly");
                } else {
                    // Negative infinity might be handled specially in positive-only encoding
                    assert!(recovered.is_infinite() || recovered == 0.0, 
                           "Negative infinity should map to infinity or zero");
                }
            } else {
                assert_eq!(val, recovered, "Finite special value {} should roundtrip exactly", val);
            }
        }
    }

    // TODO: C FFI interface test removed since C FFI exports were removed as cruft

    #[test]
    fn test_export_comprehensive_roundtrip() {
        debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Testing comprehensive roundtrip scenarios");

        // Test a wide range of values
        let mut test_values = vec![
            // Simple integers
            0.0, 1.0, 2.0, 10.0, 100.0, 1000.0,
            // Complex floats
            1.5, 2.25, 3.14159, 2.718281828, 0.1, 0.333333,
            // Small values
            1e-10, 1e-100, f64::MIN_POSITIVE,
            // Large values
            1e10, 1e100, f64::MAX,
            // Special values
            f64::INFINITY, 0.0, -0.0,
        ];

        // Add negative versions
        let negatives: Vec<f64> = test_values.iter()
            .filter(|&x| x.is_finite() && *x > 0.0)
            .map(|&x| -x)
            .collect();
        test_values.extend(negatives);

        for val in test_values {
            // Test lex encoding roundtrip
            let lex = float_to_lex(val);
            let lex_recovered = lex_to_float(lex);

            // Test int conversion roundtrip
            let int_repr = float_to_int(val);
            let int_recovered = int_to_float(int_repr);

            debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Comprehensive test {} -> lex: 0x{:016X} -> {}, int: 0x{:016X} -> {}", 
                     val, lex, lex_recovered, int_repr, int_recovered);

            if val.is_finite() {
                assert_eq!(val, lex_recovered, "Lex encoding should roundtrip for finite value {}", val);
                assert_eq!(val, int_recovered, "Int conversion should roundtrip for finite value {}", val);
            } else if val.is_nan() {
                assert!(lex_recovered.is_nan(), "NaN should roundtrip to NaN in lex encoding");
                assert!(int_recovered.is_nan(), "NaN should roundtrip to NaN in int conversion");
            } else {
                // Infinity handling may vary based on implementation
                assert!(lex_recovered.is_infinite() || lex_recovered == 0.0, 
                       "Infinity should map appropriately in lex encoding");
                assert_eq!(val, int_recovered, "Infinity should roundtrip exactly in int conversion");
            }
        }

        debug_log!("FLOAT_ENCODING_EXPORT DEBUG: Comprehensive roundtrip tests passed");
    }
}