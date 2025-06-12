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
    pub smallest_nonzero_magnitude: f64,
}

impl Eq for FloatConstraints {}

impl std::hash::Hash for FloatConstraints {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.min_value.to_bits().hash(state);
        self.max_value.to_bits().hash(state);
        self.allow_nan.hash(state);
        self.smallest_nonzero_magnitude.to_bits().hash(state);
    }
}

impl Default for FloatConstraints {
    fn default() -> Self {
        Self {
            min_value: f64::NEG_INFINITY,
            max_value: f64::INFINITY,
            allow_nan: true,
            smallest_nonzero_magnitude: f64::MIN_POSITIVE, // Smallest positive normal float
        }
    }
}

impl FloatConstraints {
    pub fn new(min_value: Option<f64>, max_value: Option<f64>) -> Self {
        Self {
            min_value: min_value.unwrap_or(f64::NEG_INFINITY),
            max_value: max_value.unwrap_or(f64::INFINITY),
            allow_nan: true,
            smallest_nonzero_magnitude: f64::MIN_POSITIVE,
        }
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

    pub fn from_ranges(ranges: &[(u32, u32)]) -> Self {
        Self {
            intervals: ranges.to_vec(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
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
        assert_eq!(constraints.smallest_nonzero_magnitude, f64::MIN_POSITIVE);
        
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
            smallest_nonzero_magnitude: 1e-10,
        };
        let cloned = float_constraints.clone();
        assert_eq!(float_constraints, cloned);
        
        println!("CONSTRAINTS DEBUG: Constraint cloning test passed");
    }
}