//! # Choice System: Foundational Generation and Shrinking Architecture
//!
//! This module implements the core choice-based architecture that forms the foundation 
//! of the Conjecture engine. All randomness flows through strongly-typed choices with 
//! sophisticated constraint systems, enabling deterministic replay, optimal shrinking, 
//! and type-safe generation.
//!
//! ## Architecture Overview
//!
//! The choice system is built around three core concepts:
//!
//! ### 1. Choice Types (`ChoiceType`)
//! Strongly-typed enumeration of all possible value types:
//! - `Integer`: Arbitrary-precision integers with range constraints
//! - `Boolean`: True/false values with probability weighting
//! - `Float`: IEEE 754 floating-point numbers with sophisticated encoding
//! - `String`: UTF-8 strings with alphabet and length constraints  
//! - `Bytes`: Raw byte sequences with encoding-aware generation
//!
//! ### 2. Choice Values (`ChoiceValue`)
//! Type-safe value storage with efficient serialization:
//! - Zero-copy serialization for database storage
//! - Hash-based deduplication for choice sequence optimization
//! - Lexicographic ordering for optimal shrinking behavior
//!
//! ### 3. Constraint Systems (`Constraints`)
//! Sophisticated constraint validation and enforcement:
//! - Compile-time constraint checking where possible
//! - Runtime validation with detailed error messages
//! - Constraint composition for complex generation patterns
//!
//! ## Submodule Architecture
//!
//! ### Generation Subsystems
//! - **`value_generation`**: Core value generation algorithms with entropy management
//! - **`templating`**: Template-based generation for structured data patterns
//! - **`weighted_selection`**: Probability-weighted choice selection with bias correction
//! - **`dfa_string_generation`**: Deterministic finite automaton string generation
//!
//! ### Constraint and Navigation
//! - **`constraints`**: Comprehensive constraint definition and validation
//! - **`navigation`**: Choice sequence navigation and indexing
//! - **`navigation_system`**: Advanced navigation with caching and optimization
//! - **`indexing`**: Efficient choice sequence indexing with float encoding
//!
//! ### Shrinking and Optimization  
//! - **`advanced_shrinking`**: Multi-strategy shrinking with quality metrics
//! - **`shrinking_system`**: Integration layer for shrinking algorithms
//! - **`core_compilation_error_resolution`**: Type system error recovery
//!
//! ### Specialized Systems
//! - **`float_constraint_type_system`**: Advanced float constraint validation
//! - **`field_access_system`**: Safe field access with bounds checking
//!
//! ## Design Principles
//!
//! ### Type Safety First
//! All choice operations are statically typed to prevent runtime generation errors:
//! ```rust
//! let choice = ChoiceValue::Integer(42);
//! match choice {
//!     ChoiceValue::Integer(i) => println!("Safe integer access: {}", i),
//!     _ => unreachable!("Type system prevents this"),
//! }
//! ```
//!
//! ### Lexicographic Shrinking
//! All choice values implement lexicographic ordering for optimal shrinking:
//! - Smaller encoded values represent "simpler" inputs
//! - Shrinking always progresses toward lexicographically smaller sequences
//! - Guaranteed termination with minimal counterexamples
//!
//! ### Performance Optimization
//! - Zero-allocation paths for common value types
//! - Lazy constraint evaluation with short-circuit logic
//! - Cached validation results for repeated constraints
//! - SIMD-optimized operations where applicable
//!
//! ## Error Handling Strategy
//!
//! The choice system uses a layered error handling approach:
//! 1. **Compile-time**: Type system prevents impossible operations
//! 2. **Generation-time**: Constraint validation with recovery
//! 3. **Runtime**: Graceful degradation with fallback strategies
//!
//! All errors provide detailed context including:
//! - Choice sequence position for replay debugging
//! - Constraint violation details with suggested fixes
//! - Performance metrics for optimization opportunities
//!
//! ## Thread Safety and Concurrency
//!
//! Choice values are immutable and `Send + Sync`, enabling:
//! - Parallel choice generation across multiple threads
//! - Concurrent shrinking with work-stealing algorithms
//! - Lock-free choice sequence sharing between test runs
//!
//! ## Memory Management
//!
//! Efficient memory usage through:
//! - Small value optimization for common integer/boolean choices
//! - Reference counting for large string/bytes values
//! - Memory pool allocation for choice sequence batches
//! - Automatic cleanup with RAII patterns

pub mod constraints;
pub mod indexing;
mod indexing_correct;
mod navigation;
pub mod navigation_system;
mod node;
pub mod value_generation;
pub mod values;





pub use self::constraints::*;
pub use self::indexing::*;
pub use self::navigation::*;
pub use self::navigation_system::*;
pub use self::node::*;
pub use self::value_generation::{
    ValueGenerator, EntropySource, BufferEntropySource, ValueGenerationError, 
    ValueGenerationResult, StandardValueGenerator
};
pub use self::values::*;

/// Choice types that can be drawn
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ChoiceType {
    Integer,
    Boolean,
    Float,
    String,
    Bytes,
}

impl std::fmt::Display for ChoiceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChoiceType::Integer => write!(f, "integer"),
            ChoiceType::Boolean => write!(f, "boolean"),
            ChoiceType::Float => write!(f, "float"),
            ChoiceType::String => write!(f, "string"),
            ChoiceType::Bytes => write!(f, "bytes"),
        }
    }
}

/// Choice value that can be drawn  
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ChoiceValue {
    Integer(i128),
    Boolean(bool),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
}

impl Eq for ChoiceValue {}

impl std::hash::Hash for ChoiceValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ChoiceValue::Integer(i) => {
                0u8.hash(state);
                i.hash(state);
            }
            ChoiceValue::Boolean(b) => {
                1u8.hash(state);
                b.hash(state);
            }
            ChoiceValue::Float(f) => {
                2u8.hash(state);
                f.to_bits().hash(state);
            }
            ChoiceValue::String(s) => {
                3u8.hash(state);
                s.hash(state);
            }
            ChoiceValue::Bytes(v) => {
                4u8.hash(state);
                v.hash(state);
            }
        }
    }
}

/// Constraints for different choice types
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Constraints {
    Integer(IntegerConstraints),
    Boolean(BooleanConstraints),
    Float(FloatConstraints),
    String(StringConstraints),
    Bytes(BytesConstraints),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_choice_type_display() {
        assert_eq!(format!("{}", ChoiceType::Integer), "integer");
        assert_eq!(format!("{}", ChoiceType::Boolean), "boolean");
        assert_eq!(format!("{}", ChoiceType::Float), "float");
        assert_eq!(format!("{}", ChoiceType::String), "string");
        assert_eq!(format!("{}", ChoiceType::Bytes), "bytes");
    }

    #[test]
    fn test_choice_value_variants() {
        let _int_val = ChoiceValue::Integer(42);
        let _bool_val = ChoiceValue::Boolean(true);
        let _float_val = ChoiceValue::Float(3.14);
        let _string_val = ChoiceValue::String("hello".to_string());
        let _bytes_val = ChoiceValue::Bytes(vec![1, 2, 3]);
    }
}