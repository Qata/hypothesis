//! Constraint definitions for different choice types

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Constraints for integer choices
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegerConstraints {
    pub min_value: Option<i128>,
    pub max_value: Option<i128>,
    pub weights: Option<HashMap<i128, f64>>,
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
    pub fn new(min_value: Option<i128>, max_value: Option<i128>, shrink_towards: Option<i128>) -> Self {
        Self {
            min_value,
            max_value,
            weights: None,
            shrink_towards,
        }
    }
}

/// Constraints for boolean choices  
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BooleanConstraints {
    pub p: f64, // Probability of True
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