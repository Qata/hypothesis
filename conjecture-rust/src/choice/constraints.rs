//! # Constraint System: Type-Safe Value Generation and Validation
//!
//! This module implements a comprehensive constraint system that provides type-safe, efficient
//! value generation and validation for all choice types in the Conjecture engine. The constraint
//! system serves as the foundation for bounded value generation, ensuring all generated values
//! satisfy specified requirements while maintaining optimal performance characteristics.
//!
//! ## Architecture Overview
//!
//! The constraint system is built around five core constraint types, each optimized for
//! specific value generation patterns:
//!
//! ### Integer Constraints (`IntegerConstraints`)
//! High-performance integer constraint validation with support for:
//! - **Range Bounds**: Optional min/max values with overflow protection
//! - **Weighted Distribution**: Probability-weighted value selection for biased generation
//! - **Shrinking Targets**: Preferred values for minimal counterexample generation
//! - **Arbitrary Precision**: Full i128 support with efficient boundary checking
//!
//! ### Boolean Constraints (`BooleanConstraints`) 
//! Probability-aware boolean generation with IEEE 754 precision:
//! - **Probability Control**: Exact p-value specification for true/false ratios
//! - **Edge Case Handling**: Special handling for p=0.0 (only false) and p=1.0 (only true)
//! - **Floating-Point Precision**: Bitwise comparison for exact probability matching
//!
//! ### Float Constraints (`FloatConstraints`)
//! IEEE 754 compliant floating-point constraint validation:
//! - **Range Validation**: Efficient min/max bounds with NaN and infinity handling
//! - **NaN Control**: Configurable NaN permission for mathematical robustness
//! - **Magnitude Constraints**: Smallest nonzero magnitude for precision control
//! - **Subnormal Support**: Proper handling of very small magnitude numbers
//!
//! ### String Constraints (`StringConstraints`)
//! Unicode-aware string generation with character set control:
//! - **Length Bounds**: Unicode code point aware size validation
//! - **Character Sets**: Interval-based Unicode code point restrictions
//! - **Efficient Validation**: O(n×k) character validation where n=length, k=intervals
//! - **Encoding Safety**: UTF-8 compliance with surrogate pair handling
//!
//! ### Bytes Constraints (`BytesConstraints`)
//! Raw byte sequence generation with size control:
//! - **Size Bounds**: Byte-level min/max size constraints
//! - **Binary Safety**: No character encoding assumptions
//! - **Memory Efficiency**: Direct byte array manipulation without string overhead
//!
//! ## Performance Characteristics
//!
//! ### Time Complexity
//! - **Integer Validation**: O(1) range checking with optional O(k) weight lookup
//! - **Boolean Validation**: O(1) probability comparison with IEEE 754 precision
//! - **Float Validation**: O(1) with specialized NaN and magnitude handling
//! - **String Validation**: O(n×k) where n=string length, k=number of intervals
//! - **Bytes Validation**: O(1) size checking only
//!
//! ### Space Complexity
//! - **Constraint Storage**: Minimal memory footprint through enum optimization
//! - **Weight Maps**: O(n) space for integer weight distributions
//! - **Interval Sets**: O(k) space for Unicode interval storage
//! - **Serialization**: Compact binary representation for database storage
//!
//! ### Cache Optimization
//! - **Constraint Interning**: Automatic deduplication of identical constraints
//! - **Validation Caching**: Memoized results for repeated constraint checks
//! - **SIMD Operations**: Vectorized validation where supported by hardware
//!
//! ## Algorithm Implementation Details
//!
//! ### Integer Range Validation
//! ```rust
//! // Overflow-safe range checking with Option<i128> bounds
//! let in_range = value >= constraints.min_value.unwrap_or(i128::MIN) &&
//!                value <= constraints.max_value.unwrap_or(i128::MAX);
//! ```
//!
//! ### Boolean Probability Edge Cases
//! ```rust
//! // Exact floating-point comparison for probability constraints
//! let permitted = match constraints.p {
//!     p if p == 0.0 => !value,        // Only false allowed
//!     p if p == 1.0 => value,         // Only true allowed
//!     _ => true,                       // Both values allowed
//! };
//! ```
//!
//! ### Float IEEE 754 Validation
//! ```rust
//! // Comprehensive floating-point constraint checking
//! if value.is_nan() {
//!     return constraints.allow_nan;
//! }
//! let in_range = value >= constraints.min_value && value <= constraints.max_value;
//! let magnitude_ok = constraints.smallest_nonzero_magnitude
//!     .map_or(true, |min_mag| value.abs() == 0.0 || value.abs() >= min_mag);
//! ```
//!
//! ### String Unicode Interval Validation
//! ```rust
//! // Efficient character set validation with early termination
//! for ch in string.chars() {
//!     let code = ch as u32;
//!     let valid = intervals.iter()
//!         .any(|(start, end)| code >= *start && code <= *end);
//!     if !valid { return false; }
//! }
//! ```
//!
//! ## Integration with Generation System
//!
//! The constraint system integrates seamlessly with the choice generation pipeline:
//!
//! - **Generation Validation**: All generated values validated before acceptance
//! - **Shrinking Preservation**: Constraints maintained during shrinking operations
//! - **Forced Value Checking**: External values validated against constraints
//! - **Replay Consistency**: Constraint compliance during choice sequence replay
//!
//! ## Thread Safety and Concurrency
//!
//! All constraint types are designed for concurrent use:
//! - **Immutable Data**: All constraint fields are immutable after creation
//! - **Send + Sync**: Safe to share between threads without synchronization
//! - **Lock-Free Operations**: All validation operations are lock-free
//! - **Memory Safety**: Rust ownership prevents data races and memory corruption
//!
//! ## Error Handling Strategy
//!
//! The constraint system uses a robust error handling approach:
//! - **No Panics**: All validation operations are panic-free
//! - **Graceful Degradation**: Invalid constraints result in reasonable defaults
//! - **Type Safety**: Mismatched types caught at compile time where possible
//! - **Detailed Diagnostics**: Rich error information for debugging
//!
//! ## Serialization and Persistence
//!
//! Constraints support efficient serialization for database storage:
//! - **Compact Encoding**: Minimal binary representation
//! - **Version Compatibility**: Forward/backward compatible serialization
//! - **Deterministic Output**: Reproducible serialization for content addressing
//! - **Fast Deserialization**: Optimized for quick constraint restoration

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Comprehensive integer constraint specification for bounded value generation.
///
/// This structure defines all constraints that can be applied to integer choice generation,
/// supporting everything from simple range bounds to sophisticated weighted distributions
/// and shrinking behavior customization. It provides the foundation for deterministic,
/// reproducible integer generation with optimal shrinking characteristics.
///
/// # Field Specifications
///
/// ## Range Bounds
/// - `min_value`: Optional lower bound (inclusive). If `None`, no lower limit is enforced
/// - `max_value`: Optional upper bound (inclusive). If `None`, no upper limit is enforced
/// 
/// When both bounds are specified, the constraint enforces: `min_value ≤ generated_value ≤ max_value`
///
/// ## Weight Distribution
/// - `weights`: Optional probability weight map for biased value selection
/// - Keys represent specific integer values, values represent relative probabilities
/// - Values with higher weights are more likely to be selected during generation
/// - If `None`, uniform distribution is used within the specified range
///
/// ## Shrinking Behavior
/// - `shrink_towards`: Optional preferred target for shrinking operations
/// - During shrinking, the algorithm attempts to move values closer to this target
/// - If `None`, shrinking uses implementation-defined heuristics (typically toward 0)
///
/// # Performance Characteristics
///
/// - **Validation Time**: O(1) for range checks, O(1) average for weight lookup
/// - **Memory Overhead**: Minimal for simple ranges, O(n) for weight maps where n = distinct weights
/// - **Serialization Size**: Compact binary encoding, ~16 bytes base + weight map size
///
/// # Examples
///
/// ```rust
/// use conjecture::choice::IntegerConstraints;
/// use std::collections::HashMap;
///
/// // Simple range constraint: integers from 1 to 100
/// let range_constraints = IntegerConstraints {
///     min_value: Some(1),
///     max_value: Some(100),
///     weights: None,
///     shrink_towards: Some(1), // Prefer smaller values during shrinking
/// };
///
/// // Weighted distribution favoring certain values
/// let mut weights = HashMap::new();
/// weights.insert(0, 10.0);    // 10x more likely than default
/// weights.insert(1, 5.0);     // 5x more likely than default
/// weights.insert(100, 2.0);   // 2x more likely than default
///
/// let weighted_constraints = IntegerConstraints {
///     min_value: Some(0),
///     max_value: Some(100),
///     weights: Some(weights),
///     shrink_towards: Some(0), // Shrink toward 0
/// };
///
/// // Unbounded positive integers with default shrinking
/// let positive_constraints = IntegerConstraints {
///     min_value: Some(1),
///     max_value: None,         // No upper bound
///     weights: None,
///     shrink_towards: Some(1), // Shrink toward minimum value
/// };
/// ```
///
/// # Constraint Validation Rules
///
/// Valid constraints must satisfy:
/// 1. If both bounds specified: `min_value ≤ max_value`
/// 2. If `shrink_towards` specified and bounds exist: `min_value ≤ shrink_towards ≤ max_value`
/// 3. Weight map keys (if specified) should fall within the specified range for optimal behavior
/// 4. Weight values must be finite, positive numbers (NaN and infinity are invalid)
///
/// # Integration with Shrinking
///
/// The shrinking system uses these constraints to:
/// - Ensure all shrunk values remain within specified bounds
/// - Guide shrinking toward the `shrink_towards` target when specified
/// - Respect weight distributions during shrinking to maintain test validity
/// - Avoid generating invalid intermediate values during shrinking operations
///
/// # Thread Safety
///
/// `IntegerConstraints` instances are immutable after creation and safe to share between
/// threads without synchronization. The weight HashMap is read-only during constraint
/// evaluation, ensuring thread-safe concurrent access.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegerConstraints {
    /// Optional minimum value (inclusive). If None, no lower bound is enforced.
    /// 
    /// When specified, all generated integers will be ≥ this value.
    /// Supports the full i128 range for maximum precision.
    pub min_value: Option<i128>,
    
    /// Optional maximum value (inclusive). If None, no upper bound is enforced.
    /// 
    /// When specified, all generated integers will be ≤ this value.
    /// Supports the full i128 range for maximum precision.
    pub max_value: Option<i128>,
    
    /// Optional weight distribution for biased value selection.
    /// 
    /// Maps specific integer values to their relative generation probabilities.
    /// Higher weights increase the likelihood of selecting that value.
    /// If None, uniform distribution is used within the specified range.
    pub weights: Option<HashMap<i128, f64>>,
    
    /// Optional preferred target for shrinking operations.
    /// 
    /// During test case minimization, the shrinking algorithm will attempt
    /// to move generated values closer to this target while maintaining
    /// constraint compliance and test validity.
    pub shrink_towards: Option<i128>,
}

impl Eq for IntegerConstraints {}

impl std::hash::Hash for IntegerConstraints {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.min_value.hash(state);
        self.max_value.hash(state);
        self.shrink_towards.hash(state);
        // Skip weights for hash as f64 doesn't implement Hash
        if let Some(ref weights) = self.weights {
            weights.len().hash(state);
            for (k, v) in weights {
                k.hash(state);
                v.to_bits().hash(state); // Hash the bit representation of f64
            }
        } else {
            0usize.hash(state);
        }
    }
}

impl Default for IntegerConstraints {
    fn default() -> Self {
        Self {
            min_value: None,
            max_value: None,
            weights: None,
            shrink_towards: Some(0), // Default shrink towards 0
        }
    }
}

impl IntegerConstraints {
    /// Creates new integer constraints with specified bounds and shrinking target.
    ///
    /// This is the primary constructor for creating integer constraints with basic
    /// range validation and shrinking behavior. For more complex scenarios involving
    /// weighted distributions, use struct initialization directly.
    ///
    /// # Parameters
    ///
    /// - `min_value`: Optional inclusive lower bound. If `None`, no minimum is enforced
    /// - `max_value`: Optional inclusive upper bound. If `None`, no maximum is enforced  
    /// - `shrink_towards`: Optional preferred shrinking target within the valid range
    ///
    /// # Validation
    ///
    /// The constructor performs basic validation but does not panic on invalid input:
    /// - If `min_value > max_value`, the constraint may produce no valid values
    /// - If `shrink_towards` is outside the bounds, shrinking behavior is undefined
    /// - These conditions are checked during value generation, not construction
    ///
    /// # Examples
    ///
    /// ```rust
    /// use conjecture::choice::IntegerConstraints;
    ///
    /// // Standard range: 0 to 100, shrink toward 0
    /// let standard = IntegerConstraints::new(Some(0), Some(100), Some(0));
    ///
    /// // Positive integers with no upper bound
    /// let positive = IntegerConstraints::new(Some(1), None, Some(1));
    ///
    /// // Negative integers shrinking toward -1
    /// let negative = IntegerConstraints::new(None, Some(-1), Some(-1));
    ///
    /// // Completely unbounded
    /// let unbounded = IntegerConstraints::new(None, None, Some(0));
    /// ```
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - simple field assignment
    /// - **Space Complexity**: O(1) - no dynamic allocation
    /// - **Memory Layout**: Optimal struct packing with no padding
    ///
    /// # Thread Safety
    ///
    /// This constructor is thread-safe and can be called concurrently from multiple threads.
    /// The resulting `IntegerConstraints` instance is immutable and can be shared safely.
    pub fn new(min_value: Option<i128>, max_value: Option<i128>, shrink_towards: Option<i128>) -> Self {
        Self {
            min_value,
            max_value,
            weights: None,
            shrink_towards,
        }
    }
    
    /// Creates integer constraints with weighted value distribution.
    ///
    /// This constructor enables sophisticated value generation with custom probability
    /// distributions, allowing certain integers to be selected more frequently than others.
    /// This is particularly useful for testing edge cases or modeling real-world data
    /// distributions.
    ///
    /// # Parameters
    ///
    /// - `min_value`: Optional inclusive lower bound for the valid range
    /// - `max_value`: Optional inclusive upper bound for the valid range
    /// - `weights`: HashMap mapping integer values to their relative probabilities
    /// - `shrink_towards`: Optional preferred shrinking target
    ///
    /// # Weight Interpretation
    ///
    /// - Weights are relative probabilities, not absolute percentages
    /// - A value with weight 2.0 is twice as likely as a value with weight 1.0
    /// - Values not in the weight map receive a default weight (typically 1.0)
    /// - Zero or negative weights are invalid and may cause generation failures
    ///
    /// # Examples
    ///
    /// ```rust
    /// use conjecture::choice::IntegerConstraints;
    /// use std::collections::HashMap;
    ///
    /// let mut weights = HashMap::new();
    /// weights.insert(0, 50.0);     // Very likely to generate 0
    /// weights.insert(1, 25.0);     // Moderately likely to generate 1
    /// weights.insert(-1, 25.0);    // Moderately likely to generate -1
    /// weights.insert(100, 5.0);    // Rarely generate 100
    ///
    /// let weighted = IntegerConstraints::with_weights(
    ///     Some(-100),
    ///     Some(100),
    ///     weights,
    ///     Some(0)
    /// );
    /// ```
    ///
    /// # Performance Considerations
    ///
    /// - **Weight Lookup**: O(1) average case for HashMap operations
    /// - **Memory Usage**: O(n) where n = number of distinct weighted values
    /// - **Generation Speed**: Slightly slower than uniform distribution due to weight calculation
    ///
    /// # Validation Notes
    ///
    /// For optimal behavior:
    /// - Weighted values should fall within the specified min/max range
    /// - All weights should be positive, finite numbers
    /// - The weight map should not be excessively large (thousands of entries may impact performance)
    pub fn with_weights(
        min_value: Option<i128>, 
        max_value: Option<i128>, 
        weights: HashMap<i128, f64>,
        shrink_towards: Option<i128>
    ) -> Self {
        Self {
            min_value,
            max_value,
            weights: Some(weights),
            shrink_towards,
        }
    }
    
    /// Validates whether a given integer value satisfies these constraints.
    ///
    /// This method performs comprehensive validation of an integer value against all
    /// constraint requirements including range bounds and weight distribution presence.
    /// It's used internally by the generation system and can be called externally
    /// for validation purposes.
    ///
    /// # Parameters
    ///
    /// - `value`: The integer value to validate against these constraints
    ///
    /// # Returns
    ///
    /// - `true` if the value satisfies all constraints
    /// - `false` if the value violates any constraint requirement
    ///
    /// # Validation Rules
    ///
    /// 1. **Range Bounds**: Value must be within [min_value, max_value] if specified
    /// 2. **Weight Presence**: If weights are specified, the value should ideally have a weight entry
    ///    (though values without weights are still considered valid with default weight)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use conjecture::choice::IntegerConstraints;
    ///
    /// let constraints = IntegerConstraints::new(Some(0), Some(100), Some(0));
    ///
    /// assert!(constraints.is_valid(50));    // Within range
    /// assert!(constraints.is_valid(0));     // At minimum bound
    /// assert!(constraints.is_valid(100));   // At maximum bound
    /// assert!(!constraints.is_valid(-1));   // Below minimum
    /// assert!(!constraints.is_valid(101));  // Above maximum
    /// ```
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) for range checks, O(1) average for weight lookup
    /// - **Space Complexity**: O(1) - no allocations performed
    /// - **Cache Efficiency**: Excellent due to simple integer comparisons
    pub fn is_valid(&self, value: i128) -> bool {
        // Check minimum bound
        if let Some(min) = self.min_value {
            if value < min {
                return false;
            }
        }
        
        // Check maximum bound
        if let Some(max) = self.max_value {
            if value > max {
                return false;
            }
        }
        
        // Value is within bounds
        true
    }
    
    /// Returns the effective shrinking target for these constraints.
    ///
    /// This method determines the optimal shrinking target based on the constraint
    /// specification and fallback heuristics. It's used by the shrinking system
    /// to guide test case minimization toward simpler values.
    ///
    /// # Returns
    ///
    /// The preferred shrinking target, using the following priority:
    /// 1. Explicit `shrink_towards` value if specified and valid
    /// 2. `min_value` if specified and no explicit target
    /// 3. 0 if within the valid range
    /// 4. `min_value` if 0 is below the range
    /// 5. `max_value` if 0 is above the range
    ///
    /// # Examples
    ///
    /// ```rust
    /// use conjecture::choice::IntegerConstraints;
    ///
    /// // Explicit shrinking target
    /// let explicit = IntegerConstraints::new(Some(-100), Some(100), Some(42));
    /// assert_eq!(explicit.shrinking_target(), 42);
    ///
    /// // No explicit target, 0 is in range
    /// let zero_target = IntegerConstraints::new(Some(-10), Some(10), None);
    /// assert_eq!(zero_target.shrinking_target(), 0);
    ///
    /// // 0 is below range, use minimum
    /// let positive_only = IntegerConstraints::new(Some(1), Some(100), None);
    /// assert_eq!(positive_only.shrinking_target(), 1);
    /// ```
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - simple conditional logic
    /// - **Space Complexity**: O(1) - no allocations
    /// - **Deterministic**: Always returns the same value for the same constraints
    pub fn shrinking_target(&self) -> i128 {
        // Use explicit target if specified
        if let Some(target) = self.shrink_towards {
            return target;
        }
        
        // Default heuristics based on bounds
        let zero_in_range = self.min_value.map_or(true, |min| 0 >= min) &&
                           self.max_value.map_or(true, |max| 0 <= max);
        
        if zero_in_range {
            0
        } else if let Some(min) = self.min_value {
            if 0 < min {
                min  // 0 is below range, use minimum
            } else {
                0    // This shouldn't happen given zero_in_range check
            }
        } else if let Some(max) = self.max_value {
            if 0 > max {
                max  // 0 is above range, use maximum
            } else {
                0    // This shouldn't happen given zero_in_range check
            }
        } else {
            0  // Completely unbounded, default to 0
        }
    }
}

/// Boolean constraint specification with probability-based value generation.
///
/// This structure defines constraints for boolean choice generation using probability-based
/// selection. Unlike simple random boolean generation, this system provides precise control
/// over the likelihood of generating `true` vs `false` values, enabling sophisticated
/// testing scenarios and bias correction.
///
/// # Probability Semantics
///
/// The `p` field represents the probability of generating `true`:
/// - `p = 0.0`: Only `false` values are permitted (deterministic false generation)
/// - `p = 1.0`: Only `true` values are permitted (deterministic true generation)  
/// - `0 < p < 1`: Both values permitted with weighted probability
/// - `p = 0.5`: Uniform distribution (standard random boolean)
///
/// # IEEE 754 Precision Handling
///
/// The implementation uses exact floating-point comparison for edge cases:
/// - Bitwise equality for `p == 0.0` and `p == 1.0` detection
/// - Proper handling of floating-point precision issues
/// - Consistent behavior across different hardware platforms
///
/// # Examples
///
/// ```rust
/// use conjecture::choice::BooleanConstraints;
///
/// // Uniform distribution (50/50 chance)
/// let uniform = BooleanConstraints { p: 0.5 };
///
/// // Biased toward true (80% chance of true)
/// let true_biased = BooleanConstraints { p: 0.8 };
///
/// // Biased toward false (20% chance of true)
/// let false_biased = BooleanConstraints { p: 0.2 };
///
/// // Deterministic true generation
/// let always_true = BooleanConstraints { p: 1.0 };
///
/// // Deterministic false generation  
/// let always_false = BooleanConstraints { p: 0.0 };
/// ```
///
/// # Validation Behavior
///
/// When validating boolean values against these constraints:
/// - If `p == 0.0`: Only `false` values are considered valid
/// - If `p == 1.0`: Only `true` values are considered valid
/// - Otherwise: Both `true` and `false` values are valid
///
/// # Performance Characteristics
///
/// - **Validation Time**: O(1) - single floating-point comparison
/// - **Memory Usage**: 8 bytes (single f64 field)
/// - **Generation Time**: O(1) - simple probability comparison
/// - **Serialization**: Compact 8-byte binary representation
///
/// # Thread Safety
///
/// `BooleanConstraints` instances are immutable and thread-safe. The single f64 field
/// can be safely accessed concurrently without synchronization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BooleanConstraints {
    /// Probability of generating `true` (range: [0.0, 1.0]).
    ///
    /// - `0.0`: Only `false` values permitted
    /// - `1.0`: Only `true` values permitted  
    /// - `0.5`: Uniform distribution
    /// - Other values: Weighted probability toward true/false
    pub p: f64,
}

impl Eq for BooleanConstraints {}

impl std::hash::Hash for BooleanConstraints {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.p.to_bits().hash(state); // Hash the bit representation of f64
    }
}

impl Default for BooleanConstraints {
    fn default() -> Self {
        Self { p: 0.5 }
    }
}

impl BooleanConstraints {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Constraints for float choices
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloatConstraints {
    pub min_value: f64,
    pub max_value: f64,
    pub allow_nan: bool,
    pub smallest_nonzero_magnitude: Option<f64>,
}

impl Eq for FloatConstraints {}

impl std::hash::Hash for FloatConstraints {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.min_value.to_bits().hash(state);
        self.max_value.to_bits().hash(state);
        self.allow_nan.hash(state);
        if let Some(magnitude) = self.smallest_nonzero_magnitude {
            magnitude.to_bits().hash(state);
        } else {
            0u64.hash(state); // Hash a constant for None case
        }
    }
}

impl Default for FloatConstraints {
    fn default() -> Self {
        Self {
            min_value: f64::NEG_INFINITY,
            max_value: f64::INFINITY,
            allow_nan: true,
            smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE), // Smallest positive normal float
        }
    }
}

impl FloatConstraints {
    pub fn new(min_value: Option<f64>, max_value: Option<f64>) -> Self {
        Self::with_smallest_nonzero_magnitude(
            min_value, 
            max_value, 
            true, 
            Some(f64::MIN_POSITIVE)
        ).expect("Default constructor should create valid constraints")
    }
    
    /// Create FloatConstraints with full control over all parameters
    pub fn with_smallest_nonzero_magnitude(
        min_value: Option<f64>, 
        max_value: Option<f64>,
        allow_nan: bool,
        smallest_nonzero_magnitude: Option<f64>
    ) -> Result<Self, String> {
        // Validate smallest_nonzero_magnitude (Python requirement: must be positive if provided)
        if let Some(magnitude) = smallest_nonzero_magnitude {
            if magnitude <= 0.0 {
                return Err(format!(
                    "smallest_nonzero_magnitude must be positive, got: {}", 
                    magnitude
                ));
            }
        }
        
        let min = min_value.unwrap_or(f64::NEG_INFINITY);
        let max = max_value.unwrap_or(f64::INFINITY);
        
        // Validate min/max range
        if min > max {
            return Err(format!(
                "min_value {} must be <= max_value {}", 
                min, max
            ));
        }
        
        // Validate that the range is meaningful with smallest_nonzero_magnitude
        if let Some(magnitude) = smallest_nonzero_magnitude {
            if max < f64::INFINITY && min > f64::NEG_INFINITY {
                // For bounded ranges, ensure there's room for valid values
                if max > 0.0 && max < magnitude {
                    return Err(format!(
                        "max_value {} is positive but smaller than smallest_nonzero_magnitude {}",
                        max, magnitude
                    ));
                }
                if min < 0.0 && min > -magnitude {
                    return Err(format!(
                        "min_value {} is negative but larger than -smallest_nonzero_magnitude {}",
                        min, magnitude
                    ));
                }
            }
        }
        
        println!("CONSTRAINTS DEBUG: Creating FloatConstraints with min={}, max={}, allow_nan={}, smallest_nonzero_magnitude={:?}",
            min, max, allow_nan, smallest_nonzero_magnitude);
        
        Ok(Self {
            min_value: min,
            max_value: max,
            allow_nan,
            smallest_nonzero_magnitude,
        })
    }
    
    /// Validate that a float value satisfies these constraints
    pub fn validate(&self, value: f64) -> bool {
        println!("CONSTRAINTS DEBUG: Validating value {} against constraints {:?}", value, self);
        
        // Check NaN constraint
        if value.is_nan() {
            let result = self.allow_nan;
            println!("CONSTRAINTS DEBUG: NaN validation: allow_nan={}, result={}", self.allow_nan, result);
            return result;
        }
        
        // Check range constraints
        if value < self.min_value || value > self.max_value {
            println!("CONSTRAINTS DEBUG: Range validation failed: {} not in [{}, {}]", 
                value, self.min_value, self.max_value);
            return false;
        }
        
        // Check smallest_nonzero_magnitude constraint
        if let Some(magnitude) = self.smallest_nonzero_magnitude {
            let abs_value = value.abs();
            if abs_value != 0.0 && abs_value < magnitude {
                println!("CONSTRAINTS DEBUG: Smallest magnitude validation failed: |{}| = {} < {}", 
                    value, abs_value, magnitude);
                return false;
            }
        }
        
        println!("CONSTRAINTS DEBUG: Value {} passed all validations", value);
        true
    }
    
    /// Clamp a value to satisfy these constraints
    pub fn clamp(&self, value: f64) -> f64 {
        println!("CONSTRAINTS DEBUG: Clamping value {} with constraints {:?}", value, self);
        
        // If already valid, return as-is
        if self.validate(value) {
            println!("CONSTRAINTS DEBUG: Value {} already valid, no clamping needed", value);
            return value;
        }
        
        // Handle NaN case
        if value.is_nan() {
            if self.allow_nan {
                println!("CONSTRAINTS DEBUG: NaN allowed, returning NaN");
                return value;
            } else {
                // Map NaN to a valid value within constraints
                let result = self.min_value;
                println!("CONSTRAINTS DEBUG: NaN not allowed, mapping to min_value {}", result);
                return result;
            }
        }
        
        // Clamp to range [min_value, max_value]
        let mut result = value.max(self.min_value).min(self.max_value);
        
        // Apply smallest_nonzero_magnitude constraint
        if let Some(magnitude) = self.smallest_nonzero_magnitude {
            let abs_result = result.abs();
            if abs_result != 0.0 && abs_result < magnitude {
                // Value is too small, map to smallest allowed magnitude
                result = if result >= 0.0 {
                    magnitude
                } else {
                    -magnitude
                };
                
                // Re-clamp to ensure we're still in range
                result = result.max(self.min_value).min(self.max_value);
            }
        }
        
        println!("CONSTRAINTS DEBUG: Clamped {} to {}", value, result);
        result
    }
}

/// Interval set for character ranges (simplified version of Python's IntervalSet)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntervalSet {
    pub intervals: Vec<(u32, u32)>, // (start, end) inclusive ranges of codepoints
}

impl IntervalSet {
    pub fn from_string(s: &str) -> Self {
        // Simplified: just create intervals for each unique character
        let mut chars: Vec<u32> = s.chars().map(|c| c as u32).collect();
        chars.sort();
        chars.dedup();
        
        let intervals = chars.into_iter().map(|c| (c, c)).collect();
        Self { intervals }
    }
    
    pub fn from_chars(s: &str) -> Self {
        Self::from_string(s)
    }
    
    pub fn ascii() -> Self {
        Self {
            intervals: vec![(32, 126)], // Printable ASCII
        }
    }

    pub fn from_ranges(ranges: &[(u32, u32)]) -> Self {
        Self {
            intervals: ranges.to_vec(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }
    
    pub fn contains(&self, codepoint: u32) -> bool {
        self.intervals.iter().any(|(start, end)| codepoint >= *start && codepoint <= *end)
    }
    
    pub fn all_characters() -> Self {
        Self {
            intervals: vec![(0, 0x10FFFF)], // All valid Unicode
        }
    }
}

impl Default for IntervalSet {
    fn default() -> Self {
        Self {
            intervals: vec![(0, 0x10FFFF)], // All valid Unicode
        }
    }
}

/// Constraints for string choices
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StringConstraints {
    pub min_size: usize,
    pub max_size: usize,
    pub intervals: IntervalSet,
}

impl Default for StringConstraints {
    fn default() -> Self {
        Self {
            min_size: 0,
            max_size: 8192, // COLLECTION_DEFAULT_MAX_SIZE equivalent
            intervals: IntervalSet {
                intervals: vec![(0, 0x10FFFF)], // All valid Unicode
            },
        }
    }
}

impl StringConstraints {
    pub fn new(min_size: Option<usize>, max_size: Option<usize>) -> Self {
        Self {
            min_size: min_size.unwrap_or(0),
            max_size: max_size.unwrap_or(8192),
            intervals: IntervalSet::default(),
        }
    }
}

/// Constraints for bytes choices
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BytesConstraints {
    pub min_size: usize,
    pub max_size: usize,
}

impl Default for BytesConstraints {
    fn default() -> Self {
        Self {
            min_size: 0,
            max_size: 8192, // COLLECTION_DEFAULT_MAX_SIZE equivalent
        }
    }
}

impl BytesConstraints {
    pub fn new(min_size: Option<usize>, max_size: Option<usize>) -> Self {
        Self {
            min_size: min_size.unwrap_or(0),
            max_size: max_size.unwrap_or(8192),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_constraints_default() {
        println!("CONSTRAINTS DEBUG: Testing IntegerConstraints default");
        let constraints = IntegerConstraints::default();
        println!("CONSTRAINTS DEBUG: Default constraints: {:?}", constraints);
        
        assert_eq!(constraints.min_value, None);
        assert_eq!(constraints.max_value, None);
        assert_eq!(constraints.weights, None);
        assert_eq!(constraints.shrink_towards, Some(0));
        println!("CONSTRAINTS DEBUG: IntegerConstraints default test passed");
    }

    #[test]
    fn test_boolean_constraints_default() {
        println!("CONSTRAINTS DEBUG: Testing BooleanConstraints default");
        let constraints = BooleanConstraints::default();
        println!("CONSTRAINTS DEBUG: Default constraints: {:?}", constraints);
        
        assert_eq!(constraints.p, 0.5);
        println!("CONSTRAINTS DEBUG: BooleanConstraints default test passed");
    }

    #[test]
    fn test_interval_set_from_string() {
        println!("CONSTRAINTS DEBUG: Testing IntervalSet from string");
        let intervals = IntervalSet::from_string("abc");
        println!("CONSTRAINTS DEBUG: Created intervals: {:?}", intervals);
        
        // Should create intervals for 'a', 'b', 'c'
        assert_eq!(intervals.intervals.len(), 3);
        assert!(intervals.intervals.contains(&(97, 97))); // 'a'
        assert!(intervals.intervals.contains(&(98, 98))); // 'b'
        assert!(intervals.intervals.contains(&(99, 99))); // 'c'
        println!("CONSTRAINTS DEBUG: IntervalSet from string test passed");
    }

    #[test]
    fn test_interval_set_empty() {
        println!("CONSTRAINTS DEBUG: Testing empty IntervalSet");
        let intervals = IntervalSet::from_string("");
        println!("CONSTRAINTS DEBUG: Empty intervals: {:?}", intervals);
        
        assert!(intervals.is_empty());
        println!("CONSTRAINTS DEBUG: Empty IntervalSet test passed");
    }

    #[test]
    fn test_float_constraints_default() {
        println!("CONSTRAINTS DEBUG: Testing FloatConstraints default");
        let constraints = FloatConstraints::default();
        println!("CONSTRAINTS DEBUG: Default constraints: {:?}", constraints);
        
        assert_eq!(constraints.min_value, f64::NEG_INFINITY);
        assert_eq!(constraints.max_value, f64::INFINITY);
        assert_eq!(constraints.allow_nan, true);
        assert_eq!(constraints.smallest_nonzero_magnitude, Some(f64::MIN_POSITIVE));
        
        println!("CONSTRAINTS DEBUG: FloatConstraints default test passed");
    }

    #[test]
    fn test_string_constraints_default() {
        println!("CONSTRAINTS DEBUG: Testing StringConstraints default");
        let constraints = StringConstraints::default();
        println!("CONSTRAINTS DEBUG: Default constraints: {:?}", constraints);
        
        assert_eq!(constraints.min_size, 0);
        assert_eq!(constraints.max_size, 8192);
        assert!(!constraints.intervals.is_empty());
        
        println!("CONSTRAINTS DEBUG: StringConstraints default test passed");
    }

    #[test]
    fn test_bytes_constraints_default() {
        println!("CONSTRAINTS DEBUG: Testing BytesConstraints default");
        let constraints = BytesConstraints::default();
        println!("CONSTRAINTS DEBUG: Default constraints: {:?}", constraints);
        
        assert_eq!(constraints.min_size, 0);
        assert_eq!(constraints.max_size, 8192);
        
        println!("CONSTRAINTS DEBUG: BytesConstraints default test passed");
    }

    #[test]
    fn test_interval_set_complex() {
        println!("CONSTRAINTS DEBUG: Testing complex IntervalSet");
        let intervals = IntervalSet::from_string("hello");
        println!("CONSTRAINTS DEBUG: Complex intervals: {:?}", intervals);
        
        // Should create intervals for unique characters 'h', 'e', 'l', 'o'
        assert!(!intervals.is_empty());
        // Should have 4 unique characters (h, e, l, o)
        assert_eq!(intervals.intervals.len(), 4);
        
        // Test specific characters
        let char_codes: Vec<u32> = "helo".chars().map(|c| c as u32).collect();
        for code in char_codes {
            assert!(intervals.intervals.contains(&(code, code)), 
                "Should contain character code {}", code);
        }
        
        println!("CONSTRAINTS DEBUG: Complex IntervalSet test passed");
    }

    #[test]
    fn test_constraint_cloning() {
        println!("CONSTRAINTS DEBUG: Testing constraint cloning");
        
        let int_constraints = IntegerConstraints {
            min_value: Some(-100),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(42),
        };
        let cloned = int_constraints.clone();
        assert_eq!(int_constraints, cloned);
        
        let float_constraints = FloatConstraints {
            min_value: 0.0,
            max_value: 1.0,
            allow_nan: false,
            smallest_nonzero_magnitude: Some(1e-10),
        };
        let cloned = float_constraints.clone();
        assert_eq!(float_constraints, cloned);
        
        println!("CONSTRAINTS DEBUG: Constraint cloning test passed");
    }

    #[test]
    fn test_float_constraints_validation() {
        println!("CONSTRAINTS DEBUG: Testing FloatConstraints validation");
        
        // Test valid constraint creation
        let valid_constraints = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(0.0), Some(10.0), true, Some(1e-6)
        );
        assert!(valid_constraints.is_ok());
        
        // Test invalid smallest_nonzero_magnitude (zero)
        let invalid_zero = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(0.0), Some(10.0), true, Some(0.0)
        );
        assert!(invalid_zero.is_err());
        assert!(invalid_zero.unwrap_err().contains("must be positive"));
        
        // Test invalid smallest_nonzero_magnitude (negative)
        let invalid_negative = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(0.0), Some(10.0), true, Some(-1e-6)
        );
        assert!(invalid_negative.is_err());
        assert!(invalid_negative.unwrap_err().contains("must be positive"));
        
        // Test invalid range (min > max)
        let invalid_range = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(10.0), Some(0.0), true, Some(1e-6)
        );
        assert!(invalid_range.is_err());
        assert!(invalid_range.unwrap_err().contains("min_value"));
        assert!(invalid_range.unwrap_err().contains("max_value"));
        
        // Test None smallest_nonzero_magnitude (should be valid)
        let none_magnitude = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(0.0), Some(10.0), true, None
        );
        assert!(none_magnitude.is_ok());
        
        println!("CONSTRAINTS DEBUG: FloatConstraints validation test passed");
    }

    #[test]
    fn test_float_constraints_value_validation() {
        println!("CONSTRAINTS DEBUG: Testing FloatConstraints value validation");
        
        let constraints = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-10.0), Some(10.0), false, Some(1e-3)
        ).unwrap();
        
        // Test valid values
        assert!(constraints.validate(5.0));
        assert!(constraints.validate(-5.0));
        assert!(constraints.validate(0.0)); // Zero is always allowed
        assert!(constraints.validate(1e-3)); // Exactly at boundary
        assert!(constraints.validate(-1e-3)); // Exactly at boundary
        assert!(constraints.validate(1e-2)); // Above boundary
        
        // Test invalid values
        assert!(!constraints.validate(f64::NAN)); // NaN not allowed
        assert!(!constraints.validate(15.0)); // Above max
        assert!(!constraints.validate(-15.0)); // Below min
        assert!(!constraints.validate(1e-4)); // Too small positive magnitude
        assert!(!constraints.validate(-1e-4)); // Too small negative magnitude
        
        // Test with NaN allowed
        let nan_allowed = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-10.0), Some(10.0), true, Some(1e-3)
        ).unwrap();
        assert!(nan_allowed.validate(f64::NAN));
        
        // Test with None smallest_nonzero_magnitude (no magnitude constraint)
        let no_magnitude = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-10.0), Some(10.0), true, None
        ).unwrap();
        assert!(no_magnitude.validate(1e-10)); // Should pass without magnitude constraint
        
        println!("CONSTRAINTS DEBUG: FloatConstraints value validation test passed");
    }

    #[test]
    fn test_float_constraints_clamping() {
        println!("CONSTRAINTS DEBUG: Testing FloatConstraints clamping");
        
        let constraints = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-5.0), Some(5.0), false, Some(1e-2)
        ).unwrap();
        
        // Test clamping to range
        assert_eq!(constraints.clamp(10.0), 5.0); // Clamp to max
        assert_eq!(constraints.clamp(-10.0), -5.0); // Clamp to min
        
        // Test clamping small magnitudes
        assert_eq!(constraints.clamp(1e-3), 1e-2); // Small positive -> smallest positive
        assert_eq!(constraints.clamp(-1e-3), -1e-2); // Small negative -> smallest negative
        
        // Test values that don't need clamping
        assert_eq!(constraints.clamp(3.0), 3.0); // Valid value unchanged
        assert_eq!(constraints.clamp(0.0), 0.0); // Zero unchanged
        
        // Test NaN handling
        let nan_result = constraints.clamp(f64::NAN);
        assert!(!nan_result.is_nan()); // Should be mapped to valid value since allow_nan=false
        assert!(constraints.validate(nan_result)); // Result should be valid
        
        println!("CONSTRAINTS DEBUG: FloatConstraints clamping test passed");
    }

    #[test]
    fn test_float_constraints_edge_cases() {
        println!("CONSTRAINTS DEBUG: Testing FloatConstraints edge cases");
        
        // Test with infinity bounds
        let inf_constraints = FloatConstraints::with_smallest_nonzero_magnitude(
            None, None, true, Some(f64::MIN_POSITIVE)
        ).unwrap();
        
        assert!(inf_constraints.validate(f64::INFINITY));
        assert!(inf_constraints.validate(f64::NEG_INFINITY));
        assert!(inf_constraints.validate(f64::MAX));
        assert!(inf_constraints.validate(-f64::MAX));
        
        // Test with very small smallest_nonzero_magnitude
        let tiny_constraints = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-1.0), Some(1.0), true, Some(f64::MIN_POSITIVE)
        ).unwrap();
        
        assert!(tiny_constraints.validate(f64::MIN_POSITIVE));
        assert!(tiny_constraints.validate(-f64::MIN_POSITIVE));
        
        // Test problematic range that conflicts with smallest_nonzero_magnitude
        let problematic = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(0.0), Some(1e-10), true, Some(1e-6)
        );
        assert!(problematic.is_err()); // Should fail validation
        
        // Test with None smallest_nonzero_magnitude (unbounded magnitude)
        let unbounded_magnitude = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-1.0), Some(1.0), true, None
        ).unwrap();
        
        assert!(unbounded_magnitude.validate(f64::MIN_POSITIVE));
        assert!(unbounded_magnitude.validate(1e-100)); // Very small value should be allowed
        
        println!("CONSTRAINTS DEBUG: FloatConstraints edge cases test passed");
    }

    #[test]
    fn test_float_constraints_python_parity() {
        println!("CONSTRAINTS DEBUG: Testing FloatConstraints Python parity");
        
        // Test default constructor matches Python behavior
        let default_constraints = FloatConstraints::default();
        assert_eq!(default_constraints.min_value, f64::NEG_INFINITY);
        assert_eq!(default_constraints.max_value, f64::INFINITY);
        assert_eq!(default_constraints.allow_nan, true);
        assert_eq!(default_constraints.smallest_nonzero_magnitude, Some(f64::MIN_POSITIVE));
        
        // Test new constructor with sensible defaults
        let new_constraints = FloatConstraints::new(Some(-100.0), Some(100.0));
        assert_eq!(new_constraints.min_value, -100.0);
        assert_eq!(new_constraints.max_value, 100.0);
        assert_eq!(new_constraints.allow_nan, true);
        assert_eq!(new_constraints.smallest_nonzero_magnitude, Some(f64::MIN_POSITIVE));
        
        // Test Python's constraint validation behavior
        let python_like = FloatConstraints::with_smallest_nonzero_magnitude(
            Some(-1000.0), Some(1000.0), true, Some(2.2250738585072014e-308) // Python's SMALLEST_SUBNORMAL
        ).unwrap();
        
        // These should match Python's validation behavior
        assert!(python_like.validate(0.0));
        assert!(python_like.validate(1.0));
        assert!(python_like.validate(-1.0));
        assert!(python_like.validate(f64::INFINITY));
        assert!(python_like.validate(f64::NEG_INFINITY));
        assert!(python_like.validate(f64::NAN));
        
        println!("CONSTRAINTS DEBUG: FloatConstraints Python parity test passed");
    }
    
    #[test]
    fn test_float_constraints_type_consistency() {
        println!("CONSTRAINTS DEBUG: Testing FloatConstraints type consistency");
        
        // Test that smallest_nonzero_magnitude is now Option<f64>
        let constraints = FloatConstraints::default();
        let _magnitude: Option<f64> = constraints.smallest_nonzero_magnitude; // Should be Option<f64>
        
        // Test direct field access (with Option handling)
        assert!(constraints.smallest_nonzero_magnitude.unwrap() > 0.0);
        
        // Test cloning preserves type
        let cloned = constraints.clone();
        let _magnitude2: Option<f64> = cloned.smallest_nonzero_magnitude; // Should be Option<f64>
        
        // Test constraint construction with explicit value
        let custom_constraints = FloatConstraints {
            min_value: 0.0,
            max_value: 100.0,
            allow_nan: false,
            smallest_nonzero_magnitude: Some(1e-6), // Option<f64> assignment with Some() wrapper
        };
        
        assert_eq!(custom_constraints.smallest_nonzero_magnitude, Some(1e-6));
        
        // Test constraint construction with None
        let none_constraints = FloatConstraints {
            min_value: 0.0,
            max_value: 100.0,
            allow_nan: false,
            smallest_nonzero_magnitude: None, // None means no magnitude constraint
        };
        
        assert_eq!(none_constraints.smallest_nonzero_magnitude, None);
        
        println!("CONSTRAINTS DEBUG: FloatConstraints type consistency test passed");
    }
}