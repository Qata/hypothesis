//! Float Constraint Type System - Complete Module Capability
//!
//! This module provides a comprehensive float constraint type system that integrates
//! FloatConstraints with the sophisticated float encoding system for value generation,
//! validation, and shrinking. It serves as the complete capability for handling
//! floating-point values in the ConjectureData system.
//!
//! ## Core Capabilities
//!
//! ### 1. Enhanced Float Constraint Management
//! - Comprehensive `FloatConstraints` validation and enforcement
//! - Type-safe `smallest_nonzero_magnitude: Option<f64>` handling
//! - Python parity constraint semantics with Rust idioms
//!
//! ### 2. Float Encoding Integration
//! - Lexicographic encoding for optimal shrinking properties
//! - Integration with DataTree storage via `float_to_int()` conversion
//! - Multi-width IEEE 754 support (f16, f32, f64)
//!
//! ### 3. Constant-Aware Generation
//! - Predefined pools of "weird floats" (special values, boundary cases)
//! - Probability-based injection of edge cases during generation
//! - Sophisticated float generation strategies matching Python Hypothesis behavior
//!
//! ### 4. Provider Integration
//! - Enhanced provider system with constraint-aware float generation
//! - Encoding-aware generation for better shrinking properties
//! - Support for forced values and replay functionality

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::choice::constraints::FloatConstraints;
use crate::float_encoding_export::{
    float_to_lex, lex_to_float, float_to_int, int_to_float,
    FloatWidth, FloatEncodingStrategy, FloatEncodingResult, FloatEncodingConfig,
};
// Custom primitive provider trait for float constraint type system
pub trait FloatPrimitiveProvider {
    fn generate_u64(&mut self) -> u64;
    fn generate_f64(&mut self) -> f64;
    fn generate_usize(&mut self) -> usize;
    fn generate_bool(&mut self) -> bool;
    fn generate_float(&mut self, constraints: &FloatConstraints) -> f64;
}

// Debug logging macro
macro_rules! debug_log {
    ($($arg:tt)*) => {
        #[cfg(not(test))]
        println!("FLOAT_TYPE_SYSTEM DEBUG: {}", format!($($arg)*));
    };
}

/// Enhanced float generation strategy with encoding awareness
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FloatGenerationStrategy {
    /// Generate using uniform random sampling
    Uniform,
    /// Generate using lexicographic sampling for better shrinking
    Lexicographic,
    /// Generate with bias towards special values
    ConstantBiased {
        /// Probability of selecting from constant pool (0.0 to 1.0)
        constant_probability: f64,
    },
    /// Generate with sophisticated constraint-aware sampling
    ConstraintAware,
}

impl Default for FloatGenerationStrategy {
    fn default() -> Self {
        FloatGenerationStrategy::ConstantBiased {
            constant_probability: 0.15, // 15% probability matching Python Hypothesis
        }
    }
}

/// Comprehensive float constraint type system with encoding integration
#[derive(Debug, Clone)]
pub struct FloatConstraintTypeSystem {
    /// The underlying float constraints
    pub constraints: FloatConstraints,
    /// Float generation strategy
    pub generation_strategy: FloatGenerationStrategy,
    /// Float width for encoding (defaults to f64)
    pub width: FloatWidth,
    /// Cached constant pool for efficient generation
    constant_pool: Vec<f64>,
    /// Encoding configuration for advanced use cases
    encoding_config: FloatEncodingConfig,
}

impl FloatConstraintTypeSystem {
    /// Create a new float constraint type system with default settings
    pub fn new(constraints: FloatConstraints) -> Self {
        debug_log!("Creating FloatConstraintTypeSystem with constraints: {:?}", constraints);
        
        let mut system = Self {
            constraints: constraints.clone(),
            generation_strategy: FloatGenerationStrategy::default(),
            width: FloatWidth::Width64,
            constant_pool: Vec::new(),
            encoding_config: FloatEncodingConfig::default(),
        };
        
        system.build_constant_pool();
        debug_log!("Created FloatConstraintTypeSystem with {} constants", system.constant_pool.len());
        system
    }
    
    /// Create a new system with custom generation strategy
    pub fn with_strategy(constraints: FloatConstraints, strategy: FloatGenerationStrategy) -> Self {
        debug_log!("Creating FloatConstraintTypeSystem with strategy: {:?}", strategy);
        
        let mut system = Self {
            constraints: constraints.clone(),
            generation_strategy: strategy,
            width: FloatWidth::Width64,
            constant_pool: Vec::new(),
            encoding_config: FloatEncodingConfig::default(),
        };
        
        system.build_constant_pool();
        system
    }
    
    /// Create a new system with custom width
    pub fn with_width(constraints: FloatConstraints, width: FloatWidth) -> Self {
        debug_log!("Creating FloatConstraintTypeSystem with width: {:?}", width);
        
        let mut system = Self {
            constraints: constraints.clone(),
            generation_strategy: FloatGenerationStrategy::default(),
            width,
            constant_pool: Vec::new(),
            encoding_config: FloatEncodingConfig::default(),
        };
        
        system.build_constant_pool();
        system
    }
    
    /// Build the constant pool of "weird floats" for edge case testing
    fn build_constant_pool(&mut self) {
        debug_log!("Building constant pool for constraints: {:?}", self.constraints);
        
        let mut constants = Vec::new();
        
        // Always include zero and negative zero
        constants.push(0.0);
        constants.push(-0.0);
        
        // Add infinity values if allowed by constraints
        if self.constraints.min_value <= f64::NEG_INFINITY {
            constants.push(f64::NEG_INFINITY);
        }
        if self.constraints.max_value >= f64::INFINITY {
            constants.push(f64::INFINITY);
        }
        
        // Add NaN if allowed
        if self.constraints.allow_nan {
            constants.push(f64::NAN);
        }
        
        // Add boundary values
        if self.constraints.min_value.is_finite() {
            constants.push(self.constraints.min_value);
            // Add value just above min if possible
            let next_up = next_float_up(self.constraints.min_value);
            if next_up <= self.constraints.max_value {
                constants.push(next_up);
            }
        }
        
        if self.constraints.max_value.is_finite() {
            constants.push(self.constraints.max_value);
            // Add value just below max if possible
            let next_down = next_float_down(self.constraints.max_value);
            if next_down >= self.constraints.min_value {
                constants.push(next_down);
            }
        }
        
        // Add smallest_nonzero_magnitude and its negative if specified
        if let Some(magnitude) = self.constraints.smallest_nonzero_magnitude {
            if magnitude >= self.constraints.min_value && magnitude <= self.constraints.max_value {
                constants.push(magnitude);
            }
            if -magnitude >= self.constraints.min_value && -magnitude <= self.constraints.max_value {
                constants.push(-magnitude);
            }
        }
        
        // Add some standard IEEE 754 edge cases that fit constraints
        let edge_cases = [
            f64::MIN_POSITIVE,        // Smallest positive normal
            -f64::MIN_POSITIVE,       // Largest negative normal
            2.2250738585072014e-308,  // Smallest positive subnormal
            -2.2250738585072014e-308, // Largest negative subnormal
            1.0,
            -1.0,
            2.0,
            -2.0,
            f64::MAX,
            -f64::MAX,
        ];
        
        for &value in &edge_cases {
            if self.constraints.validate(value) {
                constants.push(value);
            }
        }
        
        // Remove duplicates and sort for consistent behavior
        constants.sort_by(|a, b| {
            // Handle NaN comparison properly
            if a.is_nan() && b.is_nan() {
                std::cmp::Ordering::Equal
            } else if a.is_nan() {
                std::cmp::Ordering::Greater
            } else if b.is_nan() {
                std::cmp::Ordering::Less
            } else {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        constants.dedup_by(|a, b| {
            if a.is_nan() && b.is_nan() {
                true
            } else {
                a.to_bits() == b.to_bits() // Use bit comparison for proper -0.0 vs 0.0 handling
            }
        });
        
        self.constant_pool = constants;
        debug_log!("Built constant pool with {} values: {:?}", self.constant_pool.len(), self.constant_pool);
    }
    
    /// Generate a float value using the configured strategy
    pub fn generate_float<P: FloatPrimitiveProvider>(&self, provider: &mut P) -> f64 {
        debug_log!("Generating float with strategy: {:?}", self.generation_strategy);
        
        match &self.generation_strategy {
            FloatGenerationStrategy::Uniform => {
                self.generate_uniform(provider)
            }
            FloatGenerationStrategy::Lexicographic => {
                self.generate_lexicographic(provider)
            }
            FloatGenerationStrategy::ConstantBiased { constant_probability } => {
                self.generate_constant_biased(provider, *constant_probability)
            }
            FloatGenerationStrategy::ConstraintAware => {
                self.generate_constraint_aware(provider)
            }
        }
    }
    
    /// Generate float using uniform sampling within constraints
    fn generate_uniform<P: FloatPrimitiveProvider>(&self, provider: &mut P) -> f64 {
        debug_log!("Generating uniform float");
        
        // Use provider's built-in float generation with basic constraint enforcement
        let mut attempts = 0;
        loop {
            let value = provider.generate_float(&self.constraints);
            if self.constraints.validate(value) {
                debug_log!("Generated uniform float: {}", value);
                return value;
            }
            
            attempts += 1;
            if attempts > 1000 {
                // Fallback to clamping if we can't find a valid value
                let clamped = self.constraints.clamp(value);
                debug_log!("Uniform generation failed, clamped to: {}", clamped);
                return clamped;
            }
        }
    }
    
    /// Generate float using lexicographic sampling for better shrinking properties
    fn generate_lexicographic<P: FloatPrimitiveProvider>(&self, provider: &mut P) -> f64 {
        debug_log!("Generating lexicographic float");
        
        // Generate a random 64-bit value and convert to float using lexicographic mapping
        let random_bits = provider.generate_u64();
        let raw_float = lex_to_float(random_bits);
        
        // Apply constraints
        let constrained = self.constraints.clamp(raw_float);
        debug_log!("Generated lexicographic float: {} (raw: {}, bits: {:#018X})", 
                  constrained, raw_float, random_bits);
        constrained
    }
    
    /// Generate float with bias towards constant pool values
    fn generate_constant_biased<P: FloatPrimitiveProvider>(&self, provider: &mut P, constant_probability: f64) -> f64 {
        debug_log!("Generating constant-biased float with p={}", constant_probability);
        
        // Use constant pool with specified probability
        if !self.constant_pool.is_empty() && provider.generate_f64() < constant_probability {
            let index = provider.generate_usize() % self.constant_pool.len();
            let constant = self.constant_pool[index];
            debug_log!("Selected constant from pool: {}", constant);
            return constant;
        }
        
        // Otherwise generate normally
        self.generate_lexicographic(provider)
    }
    
    /// Generate float using sophisticated constraint-aware sampling
    fn generate_constraint_aware<P: FloatPrimitiveProvider>(&self, provider: &mut P) -> f64 {
        debug_log!("Generating constraint-aware float");
        
        // First try constant pool with high probability for edge cases
        if !self.constant_pool.is_empty() && provider.generate_f64() < 0.3 {
            let index = provider.generate_usize() % self.constant_pool.len();
            let constant = self.constant_pool[index];
            debug_log!("Selected constraint-aware constant: {}", constant);
            return constant;
        }
        
        // If we have magnitude constraints, be more careful about generation
        if let Some(magnitude) = self.constraints.smallest_nonzero_magnitude {
            // 50% chance of generating larger magnitude values
            if provider.generate_f64() < 0.5 {
                let range_min = magnitude.max(self.constraints.min_value.max(-f64::MAX));
                let range_max = self.constraints.max_value.min(f64::MAX);
                
                if range_max > range_min {
                    let uniform_value = provider.generate_f64();
                    let scaled = range_min + uniform_value * (range_max - range_min);
                    let sign = if provider.generate_bool() { 1.0 } else { -1.0 };
                    let result = scaled * sign;
                    
                    if self.constraints.validate(result) {
                        debug_log!("Generated constraint-aware magnitude float: {}", result);
                        return result;
                    }
                }
            }
        }
        
        // Fallback to lexicographic generation
        self.generate_lexicographic(provider)
    }
    
    /// Convert float to integer representation for DataTree storage
    pub fn float_to_storage_int(&self, value: f64) -> u64 {
        let encoded = float_to_int(value);
        debug_log!("Converted float {} to storage int: {:#018X}", value, encoded);
        encoded
    }
    
    /// Convert integer representation back to float from DataTree storage
    pub fn storage_int_to_float(&self, encoded: u64) -> f64 {
        let value = int_to_float(encoded);
        debug_log!("Converted storage int {:#018X} to float: {}", encoded, value);
        value
    }
    
    /// Convert float to lexicographic representation for shrinking
    pub fn float_to_shrink_order(&self, value: f64) -> u64 {
        let lex_encoded = float_to_lex(value);
        debug_log!("Converted float {} to shrink order: {:#018X}", value, lex_encoded);
        lex_encoded
    }
    
    /// Convert lexicographic representation back to float
    pub fn shrink_order_to_float(&self, lex_encoded: u64) -> f64 {
        let value = lex_to_float(lex_encoded);
        debug_log!("Converted shrink order {:#018X} to float: {}", lex_encoded, value);
        value
    }
    
    /// Validate that a float satisfies all constraints
    pub fn validate_float(&self, value: f64) -> bool {
        let result = self.constraints.validate(value);
        debug_log!("Validated float {}: {}", value, result);
        result
    }
    
    /// Apply constraints to ensure float is valid
    pub fn constrain_float(&self, value: f64) -> f64 {
        let result = self.constraints.clamp(value);
        debug_log!("Constrained float {} to {}", value, result);
        result
    }
    
    /// Get the constant pool for external inspection
    pub fn get_constant_pool(&self) -> &[f64] {
        &self.constant_pool
    }
    
    /// Update constraints and rebuild constant pool
    pub fn update_constraints(&mut self, new_constraints: FloatConstraints) {
        debug_log!("Updating constraints from {:?} to {:?}", self.constraints, new_constraints);
        self.constraints = new_constraints;
        self.build_constant_pool();
    }
}

/// Utility function to get the next representable float value above the given value
fn next_float_up(value: f64) -> f64 {
    if value.is_nan() {
        return value;
    }
    if value == f64::INFINITY {
        return value;
    }
    if value == -0.0 {
        return f64::MIN_POSITIVE;
    }
    
    let bits = value.to_bits();
    if value >= 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// Utility function to get the next representable float value below the given value
fn next_float_down(value: f64) -> f64 {
    if value.is_nan() {
        return value;
    }
    if value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::MIN_POSITIVE;
    }
    
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

impl Default for FloatConstraintTypeSystem {
    fn default() -> Self {
        Self::new(FloatConstraints::default())
    }
}

// Enhanced provider trait extension for constraint-aware float generation
pub trait FloatConstraintAwareProvider: FloatPrimitiveProvider {
    /// Generate a float using the constraint type system
    fn generate_constrained_float(&mut self, system: &FloatConstraintTypeSystem) -> f64
    where
        Self: Sized,
    {
        system.generate_float(self)
    }
    
    /// Generate a float optimized for shrinking properties
    fn generate_shrinkable_float(&mut self, system: &FloatConstraintTypeSystem) -> f64
    where
        Self: Sized,
    {
        let lex_system = FloatConstraintTypeSystem::with_strategy(
            system.constraints.clone(),
            FloatGenerationStrategy::Lexicographic,
        );
        lex_system.generate_float(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::constraints::FloatConstraints;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    
    // Test provider for testing purposes
    struct TestProvider {
        rng: ChaCha8Rng,
    }
    
    impl TestProvider {
        fn new() -> Self {
            Self {
                rng: ChaCha8Rng::from_entropy(),
            }
        }
    }
    
    impl FloatPrimitiveProvider for TestProvider {
        fn generate_u64(&mut self) -> u64 {
            self.rng.gen()
        }
        
        fn generate_f64(&mut self) -> f64 {
            self.rng.gen()
        }
        
        fn generate_usize(&mut self) -> usize {
            self.rng.gen()
        }
        
        fn generate_bool(&mut self) -> bool {
            self.rng.gen()
        }
        
        fn generate_float(&mut self, constraints: &FloatConstraints) -> f64 {
            let value = self.rng.gen::<f64>();
            if constraints.validate(value) {
                value
            } else {
                constraints.clamp(value)
            }
        }
    }
    
    impl FloatConstraintAwareProvider for TestProvider {}
    
    #[test]
    fn test_float_constraint_type_system_creation() {
        let constraints = FloatConstraints::default();
        let system = FloatConstraintTypeSystem::new(constraints);
        
        assert!(!system.get_constant_pool().is_empty());
        assert_eq!(system.width, FloatWidth::Width64);
    }
    
    #[test]
    fn test_constant_pool_building() {
        let constraints = FloatConstraints {
            min_value: -100.0,
            max_value: 100.0,
            allow_nan: true,
            smallest_nonzero_magnitude: Some(1e-6),
        };
        
        let system = FloatConstraintTypeSystem::new(constraints);
        let pool = system.get_constant_pool();
        
        // Should contain zeros
        assert!(pool.contains(&0.0));
        assert!(pool.contains(&-0.0));
        
        // Should contain NaN
        assert!(pool.iter().any(|&x| x.is_nan()));
        
        // Should contain boundaries
        assert!(pool.contains(&-100.0));
        assert!(pool.contains(&100.0));
        
        // Should contain magnitude constraints
        assert!(pool.contains(&1e-6));
        assert!(pool.contains(&-1e-6));
    }
    
    #[test]
    fn test_float_generation_strategies() {
        let constraints = FloatConstraints::default();
        let mut provider = TestProvider::new();
        
        // Test uniform generation
        let uniform_system = FloatConstraintTypeSystem::with_strategy(
            constraints.clone(),
            FloatGenerationStrategy::Uniform,
        );
        let uniform_value = uniform_system.generate_float(&mut provider);
        assert!(constraints.validate(uniform_value));
        
        // Test lexicographic generation
        let lex_system = FloatConstraintTypeSystem::with_strategy(
            constraints.clone(),
            FloatGenerationStrategy::Lexicographic,
        );
        let lex_value = lex_system.generate_float(&mut provider);
        assert!(constraints.validate(lex_value));
        
        // Test constant-biased generation
        let biased_system = FloatConstraintTypeSystem::with_strategy(
            constraints.clone(),
            FloatGenerationStrategy::ConstantBiased { constant_probability: 1.0 },
        );
        let biased_value = biased_system.generate_float(&mut provider);
        assert!(constraints.validate(biased_value));
        assert!(biased_system.get_constant_pool().contains(&biased_value) || biased_value.is_nan());
    }
    
    #[test]
    fn test_encoding_integration() {
        let constraints = FloatConstraints::default();
        let system = FloatConstraintTypeSystem::new(constraints);
        
        let original_value = 42.0;
        
        // Test storage encoding round-trip
        let storage_int = system.float_to_storage_int(original_value);
        let recovered_value = system.storage_int_to_float(storage_int);
        assert_eq!(original_value, recovered_value);
        
        // Test shrink order encoding round-trip
        let shrink_order = system.float_to_shrink_order(original_value);
        let recovered_shrink = system.shrink_order_to_float(shrink_order);
        assert_eq!(original_value, recovered_shrink);
    }
    
    #[test]
    fn test_constraint_validation_and_clamping() {
        let constraints = FloatConstraints {
            min_value: -10.0,
            max_value: 10.0,
            allow_nan: false,
            smallest_nonzero_magnitude: Some(1e-3),
        };
        
        let system = FloatConstraintTypeSystem::new(constraints);
        
        // Valid values should pass validation
        assert!(system.validate_float(5.0));
        assert!(system.validate_float(-5.0));
        assert!(system.validate_float(0.0));
        assert!(system.validate_float(1e-3));
        
        // Invalid values should fail validation
        assert!(!system.validate_float(15.0)); // Above max
        assert!(!system.validate_float(-15.0)); // Below min
        assert!(!system.validate_float(f64::NAN)); // NaN not allowed
        assert!(!system.validate_float(1e-4)); // Below magnitude threshold
        
        // Test clamping
        assert_eq!(system.constrain_float(15.0), 10.0); // Clamp to max
        assert_eq!(system.constrain_float(-15.0), -10.0); // Clamp to min
        assert_eq!(system.constrain_float(1e-4), 1e-3); // Clamp to magnitude
    }
    
    #[test]
    fn test_enhanced_provider_integration() {
        let constraints = FloatConstraints {
            min_value: 0.0,
            max_value: 100.0,
            allow_nan: false,
            smallest_nonzero_magnitude: Some(1e-6),
        };
        
        let system = FloatConstraintTypeSystem::new(constraints);
        let mut provider = TestProvider::new();
        
        // Test constraint-aware generation
        for _ in 0..100 {
            let value = provider.generate_constrained_float(&system);
            assert!(system.validate_float(value), "Generated invalid float: {}", value);
        }
        
        // Test shrinkable generation
        for _ in 0..100 {
            let value = provider.generate_shrinkable_float(&system);
            assert!(system.validate_float(value), "Generated invalid shrinkable float: {}", value);
        }
    }
    
    #[test]
    fn test_smallest_nonzero_magnitude_option_type() {
        // Test that smallest_nonzero_magnitude is properly typed as Option<f64>
        let constraints = FloatConstraints {
            min_value: -1.0,
            max_value: 1.0,
            allow_nan: false,
            smallest_nonzero_magnitude: None, // Should be Option<f64>
        };
        
        let system = FloatConstraintTypeSystem::new(constraints);
        
        // With None magnitude, very small values should be allowed
        assert!(system.validate_float(1e-100));
        assert!(system.validate_float(-1e-100));
        
        // Test with Some value
        let constrained = FloatConstraints {
            min_value: -1.0,
            max_value: 1.0,
            allow_nan: false,
            smallest_nonzero_magnitude: Some(1e-6), // Option<f64> with Some wrapper
        };
        
        let constrained_system = FloatConstraintTypeSystem::new(constrained);
        
        // With Some(1e-6), values smaller than 1e-6 should be invalid
        assert!(!constrained_system.validate_float(1e-7));
        assert!(constrained_system.validate_float(1e-5));
    }
}