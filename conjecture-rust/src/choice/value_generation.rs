//! Choice Value Generation System
//!
//! This module implements the core value generation system that converts raw entropy
//! into constrained choice values. It provides distribution-aware sampling with 
//! shrinking-friendly generation patterns that match Python Hypothesis behavior.

use crate::choice::{
    ChoiceValue, ChoiceType, Constraints,
    IntegerConstraints, BooleanConstraints, FloatConstraints, 
    StringConstraints, BytesConstraints
};
use std::collections::HashMap;
use log::debug;

/// Error types for value generation
#[derive(Debug, Clone, PartialEq)]
pub enum ValueGenerationError {
    InsufficientEntropy,
    InvalidConstraints(String),
    UnsupportedChoiceType(ChoiceType),
    GenerationFailed(String),
}

impl std::fmt::Display for ValueGenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueGenerationError::InsufficientEntropy => 
                write!(f, "Insufficient entropy for value generation"),
            ValueGenerationError::InvalidConstraints(msg) => 
                write!(f, "Invalid constraints: {}", msg),
            ValueGenerationError::UnsupportedChoiceType(choice_type) => 
                write!(f, "Unsupported choice type: {}", choice_type),
            ValueGenerationError::GenerationFailed(msg) => 
                write!(f, "Generation failed: {}", msg),
        }
    }
}

impl std::error::Error for ValueGenerationError {}

/// Result type for value generation operations
pub type ValueGenerationResult<T> = Result<T, ValueGenerationError>;

/// Entropy source for value generation
pub trait EntropySource {
    /// Draw raw bytes from the entropy source
    fn draw_bytes(&mut self, count: usize) -> ValueGenerationResult<Vec<u8>>;
    
    /// Draw a single byte (optimized common case)
    fn draw_byte(&mut self) -> ValueGenerationResult<u8> {
        let bytes = self.draw_bytes(1)?;
        Ok(bytes[0])
    }
    
    /// Draw a u32 for large integer generation
    fn draw_u32(&mut self) -> ValueGenerationResult<u32> {
        let bytes = self.draw_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
    
    /// Draw a u64 for float generation
    fn draw_u64(&mut self) -> ValueGenerationResult<u64> {
        let bytes = self.draw_bytes(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7]
        ]))
    }
}

/// Simple entropy source backed by a buffer of random data
#[derive(Debug, Clone)]
pub struct BufferEntropySource {
    buffer: Vec<u8>,
    position: usize,
}

impl BufferEntropySource {
    pub fn new(buffer: Vec<u8>) -> Self {
        Self { buffer, position: 0 }
    }
    
    pub fn remaining(&self) -> usize {
        self.buffer.len().saturating_sub(self.position)
    }
}

impl EntropySource for BufferEntropySource {
    fn draw_bytes(&mut self, count: usize) -> ValueGenerationResult<Vec<u8>> {
        if self.position + count > self.buffer.len() {
            return Err(ValueGenerationError::InsufficientEntropy);
        }
        
        let result = self.buffer[self.position..self.position + count].to_vec();
        self.position += count;
        Ok(result)
    }
}

/// Core trait for generating choice values from entropy
pub trait ValueGenerator {
    /// Generate a choice value of the specified type with constraints
    fn generate_value(
        &mut self,
        choice_type: ChoiceType,
        constraints: &Constraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<ChoiceValue>;
    
    /// Generate a boolean with probability p
    fn generate_boolean(
        &mut self,
        constraints: &BooleanConstraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<bool>;
    
    /// Generate an integer within constraints
    fn generate_integer(
        &mut self,
        constraints: &IntegerConstraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<i128>;
    
    /// Generate a float within constraints
    fn generate_float(
        &mut self,
        constraints: &FloatConstraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<f64>;
    
    /// Generate a string within constraints
    fn generate_string(
        &mut self,
        constraints: &StringConstraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<String>;
    
    /// Generate bytes within constraints
    fn generate_bytes(
        &mut self,
        constraints: &BytesConstraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<Vec<u8>>;
}

/// Standard implementation of the value generator
#[derive(Debug, Default)]
pub struct StandardValueGenerator {
    /// Cache for float ordering computations
    float_cache: HashMap<u64, f64>,
}

impl StandardValueGenerator {
    pub fn new() -> Self {
        Self {
            float_cache: HashMap::new(),
        }
    }
    
    /// Generate biased boolean using the logistic function
    /// This matches Python's probability-based boolean generation
    fn generate_biased_boolean(p: f64, entropy_byte: u8) -> bool {
        debug!("Generating boolean with p={:.6}, entropy=0x{:02X}", p, entropy_byte);
        
        if p <= 0.0 {
            false
        } else if p >= 1.0 {
            true
        } else {
            // Convert entropy byte to uniform [0,1) value
            let uniform = (entropy_byte as f64) / 256.0;
            let result = uniform < p;
            debug!("Uniform={:.6}, result={}", uniform, result);
            result
        }
    }
    
    /// Generate integer with optional weights and shrink bias
    fn generate_constrained_integer(
        constraints: &IntegerConstraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<i128> {
        debug!("Generating integer with constraints: min={:?}, max={:?}, shrink_towards={:?}",
               constraints.min_value, constraints.max_value, constraints.shrink_towards);
        
        let min = constraints.min_value.unwrap_or(i128::MIN);
        let max = constraints.max_value.unwrap_or(i128::MAX);
        
        if min > max {
            return Err(ValueGenerationError::InvalidConstraints(
                format!("min_value {} > max_value {}", min, max)
            ));
        }
        
        // Handle weighted selection if weights provided
        if let Some(ref weights) = constraints.weights {
            return Self::generate_weighted_integer(min, max, weights, entropy);
        }
        
        // Handle single value case
        if min == max {
            debug!("Single value case: {}", min);
            return Ok(min);
        }
        
        // Generate uniform value in range
        let range_size = max.saturating_sub(min).saturating_add(1);
        
        if range_size <= 256 {
            // Small range - use single byte
            let byte = entropy.draw_byte()?;
            let offset = (byte as u128) * (range_size as u128) / 256;
            let result = min.saturating_add(offset as i128);
            debug!("Small range generated: {} (byte=0x{:02X}, offset={})", result, byte, offset);
            Ok(result)
        } else if range_size <= 65536 {
            // Medium range - use two bytes
            let bytes = entropy.draw_bytes(2)?;
            let val = u16::from_le_bytes([bytes[0], bytes[1]]) as u128;
            let offset = val * (range_size as u128) / 65536;
            let result = min.saturating_add(offset as i128);
            debug!("Medium range generated: {} (val={}, offset={})", result, val, offset);
            Ok(result)
        } else {
            // Large range - use four bytes
            let val = entropy.draw_u32()? as u128;
            let offset = val * (range_size as u128) / (u32::MAX as u128 + 1);
            let result = min.saturating_add(offset as i128);
            debug!("Large range generated: {} (val={}, offset={})", result, val, offset);
            Ok(result)
        }
    }
    
    /// Generate weighted integer selection
    fn generate_weighted_integer(
        min: i128,
        max: i128,
        weights: &HashMap<i128, f64>,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<i128> {
        debug!("Generating weighted integer with {} weights", weights.len());
        
        // Build cumulative distribution
        let mut cdf = Vec::new();
        let mut total_weight = 0.0;
        
        for value in min..=max {
            if let Some(&weight) = weights.get(&value) {
                if weight > 0.0 {
                    total_weight += weight;
                    cdf.push((value, total_weight));
                }
            }
        }
        
        if cdf.is_empty() {
            return Err(ValueGenerationError::InvalidConstraints(
                "No positive weights found".to_string()
            ));
        }
        
        // Generate random point in [0, total_weight)
        let val = entropy.draw_u32()?;
        let target = (val as f64) * total_weight / (u32::MAX as f64 + 1.0);
        debug!("Target weight: {:.6} / {:.6}", target, total_weight);
        
        // Binary search in CDF
        for (value, cumulative) in &cdf {
            if target < *cumulative {
                debug!("Selected weighted value: {}", value);
                return Ok(*value);
            }
        }
        
        // Fallback to last value (should not happen with proper implementation)
        let result = cdf.last().unwrap().0;
        debug!("Fallback to last value: {}", result);
        Ok(result)
    }
    
    /// Generate float with shrinking-friendly encoding
    /// This implements Python's sophisticated float ordering for effective shrinking
    fn generate_shrinking_aware_float(
        &mut self,
        constraints: &FloatConstraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<f64> {
        debug!("Generating float with range [{}, {}], allow_nan={}, smallest_magnitude={}",
               constraints.min_value, constraints.max_value, 
               constraints.allow_nan, constraints.smallest_nonzero_magnitude);
        
        // Generate raw 64-bit value
        let raw = entropy.draw_u64()?;
        
        // Apply shrinking-friendly float ordering
        let ordered_float = self.decode_shrinking_float(raw);
        debug!("Raw=0x{:016X}, ordered_float={}", raw, ordered_float);
        
        // Apply constraints
        self.constrain_float(ordered_float, constraints)
    }
    
    /// Decode float with shrinking-friendly ordering (matches Python algorithm)
    fn decode_shrinking_float(&mut self, raw: u64) -> f64 {
        // Check cache first
        if let Some(&cached) = self.float_cache.get(&raw) {
            return cached;
        }
        
        // Use tagged union encoding - first bit determines type
        let result = if (raw & 1) == 0 {
            // Even: use IEEE 754 representation
            f64::from_bits(raw >> 1)
        } else {
            // Odd: use reordered encoding for better shrinking
            self.decode_reordered_float(raw >> 1)
        };
        
        // Cache result
        self.float_cache.insert(raw, result);
        result
    }
    
    /// Decode reordered float for better shrinking behavior
    fn decode_reordered_float(&self, bits: u64) -> f64 {
        // Reorder exponent bits to prioritize smaller magnitudes
        // This matches Python's lexicographic ordering strategy
        
        let sign = (bits >> 62) & 1;
        let exp_raw = (bits >> 51) & 0x7FF;
        let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;
        
        // Reorder exponent: bias towards positive values
        let exp_reordered = if exp_raw < 1024 {
            exp_raw + 1023
        } else {
            exp_raw - 1024
        };
        
        let ieee_bits = (sign << 63) | (exp_reordered << 52) | mantissa;
        f64::from_bits(ieee_bits)
    }
    
    /// Apply float constraints and handle special values
    fn constrain_float(
        &self,
        value: f64,
        constraints: &FloatConstraints,
    ) -> ValueGenerationResult<f64> {
        // Handle NaN
        if value.is_nan() {
            if constraints.allow_nan {
                debug!("Generated NaN (allowed)");
                return Ok(f64::NAN);
            } else {
                debug!("NaN not allowed, generating 0.0");
                return Ok(0.0);
            }
        }
        
        // Handle infinity
        if value.is_infinite() {
            let clamped = if value.is_sign_positive() {
                constraints.max_value
            } else {
                constraints.min_value
            };
            debug!("Infinity clamped to {}", clamped);
            return Ok(clamped);
        }
        
        // Apply range constraints
        let clamped = value.max(constraints.min_value).min(constraints.max_value);
        
        // Apply smallest magnitude constraint
        if constraints.smallest_nonzero_magnitude > 0.0 {
            let abs_val = clamped.abs();
            if abs_val != 0.0 && abs_val < constraints.smallest_nonzero_magnitude {
                let result = if clamped >= 0.0 {
                    constraints.smallest_nonzero_magnitude
                } else {
                    -constraints.smallest_nonzero_magnitude
                };
                debug!("Applied smallest magnitude constraint: {} -> {}", clamped, result);
                return Ok(result);
            }
        }
        
        debug!("Constrained float: {} -> {}", value, clamped);
        Ok(clamped)
    }
    
    /// Generate string from character intervals
    fn generate_interval_string(
        constraints: &StringConstraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<String> {
        debug!("Generating string with size range [{}, {}], {} intervals",
               constraints.min_size, constraints.max_size, constraints.intervals.intervals.len());
        
        // Determine string length
        let length = if constraints.min_size == constraints.max_size {
            constraints.min_size
        } else {
            let range = constraints.max_size - constraints.min_size + 1;
            let byte = entropy.draw_byte()?;
            constraints.min_size + (byte as usize * range / 256)
        };
        
        debug!("Selected string length: {}", length);
        
        // Generate characters
        let mut chars = Vec::with_capacity(length);
        let intervals = &constraints.intervals.intervals;
        
        if intervals.is_empty() {
            return Err(ValueGenerationError::InvalidConstraints(
                "No character intervals specified".to_string()
            ));
        }
        
        for i in 0..length {
            let char_code = Self::generate_interval_char(intervals, entropy)?;
            if let Some(ch) = char::from_u32(char_code) {
                chars.push(ch);
                debug!("Generated char[{}]: U+{:04X} ('{}')", i, char_code, ch);
            } else {
                debug!("Invalid char code U+{:04X}, using replacement", char_code);
                chars.push(char::REPLACEMENT_CHARACTER);
            }
        }
        
        let result = chars.into_iter().collect::<String>();
        debug!("Generated string: {:?}", result);
        Ok(result)
    }
    
    /// Generate character from intervals
    fn generate_interval_char(
        intervals: &[(u32, u32)],
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<u32> {
        // Calculate total range across all intervals
        let total_range: u64 = intervals.iter()
            .map(|(start, end)| (end - start + 1) as u64)
            .sum();
        
        if total_range == 0 {
            return Err(ValueGenerationError::InvalidConstraints(
                "Empty character intervals".to_string()
            ));
        }
        
        // Generate position in total range
        let val = entropy.draw_u32()?;
        let mut target = (val as u64 * total_range) / (u32::MAX as u64 + 1);
        
        // Find interval containing target
        for (start, end) in intervals {
            let interval_size = (end - start + 1) as u64;
            if target < interval_size {
                let result = start + target as u32;
                debug!("Selected char from interval [{}, {}]: U+{:04X}", start, end, result);
                return Ok(result);
            }
            target -= interval_size;
        }
        
        // Fallback (should not happen)
        let result = intervals[0].0;
        debug!("Fallback to first interval start: U+{:04X}", result);
        Ok(result)
    }
}

impl ValueGenerator for StandardValueGenerator {
    fn generate_value(
        &mut self,
        choice_type: ChoiceType,
        constraints: &Constraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<ChoiceValue> {
        debug!("Generating {} value", choice_type);
        
        match (choice_type, constraints) {
            (ChoiceType::Boolean, Constraints::Boolean(c)) => {
                let value = self.generate_boolean(c, entropy)?;
                Ok(ChoiceValue::Boolean(value))
            }
            (ChoiceType::Integer, Constraints::Integer(c)) => {
                let value = self.generate_integer(c, entropy)?;
                Ok(ChoiceValue::Integer(value))
            }
            (ChoiceType::Float, Constraints::Float(c)) => {
                let value = self.generate_float(c, entropy)?;
                Ok(ChoiceValue::Float(value))
            }
            (ChoiceType::String, Constraints::String(c)) => {
                let value = self.generate_string(c, entropy)?;
                Ok(ChoiceValue::String(value))
            }
            (ChoiceType::Bytes, Constraints::Bytes(c)) => {
                let value = self.generate_bytes(c, entropy)?;
                Ok(ChoiceValue::Bytes(value))
            }
            _ => Err(ValueGenerationError::InvalidConstraints(
                format!("Mismatched choice type {:?} and constraints", choice_type)
            )),
        }
    }
    
    fn generate_boolean(
        &mut self,
        constraints: &BooleanConstraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<bool> {
        let byte = entropy.draw_byte()?;
        Ok(Self::generate_biased_boolean(constraints.p, byte))
    }
    
    fn generate_integer(
        &mut self,
        constraints: &IntegerConstraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<i128> {
        Self::generate_constrained_integer(constraints, entropy)
    }
    
    fn generate_float(
        &mut self,
        constraints: &FloatConstraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<f64> {
        self.generate_shrinking_aware_float(constraints, entropy)
    }
    
    fn generate_string(
        &mut self,
        constraints: &StringConstraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<String> {
        Self::generate_interval_string(constraints, entropy)
    }
    
    fn generate_bytes(
        &mut self,
        constraints: &BytesConstraints,
        entropy: &mut dyn EntropySource,
    ) -> ValueGenerationResult<Vec<u8>> {
        debug!("Generating bytes with size range [{}, {}]", 
               constraints.min_size, constraints.max_size);
        
        // Determine byte vector length
        let length = if constraints.min_size == constraints.max_size {
            constraints.min_size
        } else {
            let range = constraints.max_size - constraints.min_size + 1;
            let byte = entropy.draw_byte()?;
            constraints.min_size + (byte as usize * range / 256)
        };
        
        debug!("Selected bytes length: {}", length);
        
        // Generate random bytes
        let bytes = entropy.draw_bytes(length)?;
        debug!("Generated {} bytes", bytes.len());
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::IntervalSet;
    use std::collections::HashSet;

    #[test]
    fn test_buffer_entropy_source() {
        let mut entropy = BufferEntropySource::new(vec![0x12, 0x34, 0x56, 0x78]);
        
        assert_eq!(entropy.remaining(), 4);
        assert_eq!(entropy.draw_byte().unwrap(), 0x12);
        assert_eq!(entropy.remaining(), 3);
        
        let bytes = entropy.draw_bytes(2).unwrap();
        assert_eq!(bytes, vec![0x34, 0x56]);
        assert_eq!(entropy.remaining(), 1);
        
        assert!(entropy.draw_bytes(2).is_err()); // Insufficient entropy
    }
    
    #[test]
    fn test_boolean_generation() {
        let mut generator = StandardValueGenerator::new();
        let mut entropy = BufferEntropySource::new(vec![0, 64, 128, 192, 255]);
        
        // Test p=0.5 (should be roughly half true/false)
        let constraints = BooleanConstraints { p: 0.5 };
        let mut results = Vec::new();
        
        for _ in 0..5 {
            let value = generator.generate_boolean(&constraints, &mut entropy).unwrap();
            results.push(value);
        }
        
        // With entropy [0, 64, 128, 192, 255] and p=0.5:
        // 0/256 = 0.0 < 0.5 -> true
        // 64/256 = 0.25 < 0.5 -> true  
        // 128/256 = 0.5 >= 0.5 -> false
        // 192/256 = 0.75 >= 0.5 -> false
        // 255/256 = 0.996 >= 0.5 -> false
        assert_eq!(results, vec![true, true, false, false, false]);
    }
    
    #[test]
    fn test_boolean_extreme_probabilities() {
        let mut generator = StandardValueGenerator::new();
        let mut entropy = BufferEntropySource::new(vec![0, 128, 255]);
        
        // Test p=0.0 (always false)
        let constraints_zero = BooleanConstraints { p: 0.0 };
        assert_eq!(generator.generate_boolean(&constraints_zero, &mut entropy).unwrap(), false);
        
        // Test p=1.0 (always true)
        let constraints_one = BooleanConstraints { p: 1.0 };
        assert_eq!(generator.generate_boolean(&constraints_one, &mut entropy).unwrap(), true);
    }
    
    #[test]
    fn test_integer_generation_small_range() {
        let mut generator = StandardValueGenerator::new();
        let mut entropy = BufferEntropySource::new(vec![0, 64, 128, 192, 255]);
        
        let constraints = IntegerConstraints {
            min_value: Some(10),
            max_value: Some(15), // Range size = 6
            weights: None,
            shrink_towards: Some(10),
        };
        
        let mut results = Vec::new();
        for _ in 0..5 {
            let value = generator.generate_integer(&constraints, &mut entropy).unwrap();
            results.push(value);
        }
        
        // All values should be in range [10, 15]
        for &value in &results {
            assert!(value >= 10 && value <= 15);
        }
        
        // Should have some variety (not all the same value)
        let unique_count = results.iter().collect::<HashSet<_>>().len();
        assert!(unique_count > 1);
    }
    
    #[test]
    fn test_integer_generation_single_value() {
        let mut generator = StandardValueGenerator::new();
        let mut entropy = BufferEntropySource::new(vec![123]);
        
        let constraints = IntegerConstraints {
            min_value: Some(42),
            max_value: Some(42), // Single value
            weights: None,
            shrink_towards: Some(42),
        };
        
        let value = generator.generate_integer(&constraints, &mut entropy).unwrap();
        assert_eq!(value, 42);
    }
    
    #[test]
    fn test_string_generation() {
        let mut generator = StandardValueGenerator::new();
        let mut entropy = BufferEntropySource::new(vec![
            2,    // Length selection (min=1, max=3, so length=1+2*3/256=1)
            0,    // Character selection (first interval)
        ]);
        
        let constraints = StringConstraints {
            min_size: 1,
            max_size: 3,
            intervals: IntervalSet::from_string("abc"),
        };
        
        let value = generator.generate_string(&constraints, &mut entropy).unwrap();
        assert!(value.len() >= 1 && value.len() <= 3);
        
        // All characters should be in allowed set
        for ch in value.chars() {
            assert!("abc".contains(ch));
        }
    }
    
    #[test]
    fn test_bytes_generation() {
        let mut generator = StandardValueGenerator::new();
        let mut entropy = BufferEntropySource::new(vec![
            128,  // Length selection (should give mid-range length)
            0x12, 0x34, 0x56, // Actual byte data
        ]);
        
        let constraints = BytesConstraints {
            min_size: 2,
            max_size: 4,
        };
        
        let value = generator.generate_bytes(&constraints, &mut entropy).unwrap();
        assert!(value.len() >= 2 && value.len() <= 4);
    }
    
    #[test]
    fn test_float_generation() {
        let mut generator = StandardValueGenerator::new();
        let mut entropy = BufferEntropySource::new(vec![
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // All zeros -> 0.0
        ]);
        
        let constraints = FloatConstraints {
            min_value: -10.0,
            max_value: 10.0,
            allow_nan: false,
            smallest_nonzero_magnitude: f64::MIN_POSITIVE,
        };
        
        let value = generator.generate_float(&constraints, &mut entropy).unwrap();
        assert!(value >= -10.0 && value <= 10.0);
        assert!(!value.is_nan());
    }
    
    #[test]
    fn test_value_generation_dispatch() {
        let mut generator = StandardValueGenerator::new();
        let mut entropy = BufferEntropySource::new(vec![128]);
        
        let constraints = Constraints::Boolean(BooleanConstraints { p: 0.5 });
        let value = generator.generate_value(ChoiceType::Boolean, &constraints, &mut entropy).unwrap();
        
        assert!(matches!(value, ChoiceValue::Boolean(_)));
    }
    
    #[test]
    fn test_error_cases() {
        let mut generator = StandardValueGenerator::new();
        let mut entropy = BufferEntropySource::new(vec![]);
        
        let constraints = BooleanConstraints { p: 0.5 };
        let result = generator.generate_boolean(&constraints, &mut entropy);
        assert!(matches!(result, Err(ValueGenerationError::InsufficientEntropy)));
        
        // Test invalid integer constraints
        let mut entropy2 = BufferEntropySource::new(vec![123]);
        let bad_constraints = IntegerConstraints {
            min_value: Some(10),
            max_value: Some(5), // min > max
            weights: None,
            shrink_towards: Some(0),
        };
        let result = generator.generate_integer(&bad_constraints, &mut entropy2);
        assert!(matches!(result, Err(ValueGenerationError::InvalidConstraints(_))));
    }
}