//! Choice system for Conjecture engine
//! 
//! This module implements Python's choice-based architecture where all randomness
//! flows through strongly-typed choices with associated constraints.

pub mod advanced_shrinking;
pub mod constraints;
pub mod dfa_string_generation;
pub mod field_access_system;
pub mod float_constraint_type_system;
pub mod indexing;
mod indexing_correct;
mod navigation;
mod node;
pub mod shrinking_system;
pub mod shrinking_demo;
pub mod templating;
pub mod value_generation;
pub mod weighted_selection;
pub mod values;

#[cfg(test)]
mod python_parity_tests;

#[cfg(test)]
mod choice_debug;

#[cfg(test)]
mod organized_tests;

#[cfg(test)]
mod navigation_integration_tests;

#[cfg(test)]
mod navigation_comprehensive_tests;

#[cfg(test)]
mod navigation_system_comprehensive_tests;

#[cfg(test)]
mod navigation_capability_ffi_tests;

#[cfg(test)]
mod choice_templating_forcing_tests;

#[cfg(test)]
mod templating_comprehensive_capability_tests;

#[cfg(test)]
mod value_generation_tests;

#[cfg(test)]
mod weighted_selection_capability_ffi_tests;

#[cfg(test)]
mod weighted_selection_comprehensive_capability_tests;

#[cfg(test)]
mod weighted_selection_capability_integration_tests;

#[cfg(test)]
mod weighted_selection_complete_capability_integration_tests;

#[cfg(test)]
mod advanced_template_generation_tests;

#[cfg(test)]
mod advanced_shrinking_tests;

#[cfg(test)]
mod advanced_shrinking_comprehensive_tests;

#[cfg(test)]
mod field_access_system_tests;

#[cfg(test)]
mod dfa_basic_test;

#[cfg(test)]
mod choice_type_system_integration_capability_tests;

#[cfg(test)]
mod shrinking_system_float_constraint_integration_comprehensive_capability_tests;

#[cfg(test)]
mod simple_choice_test;

pub use self::advanced_shrinking::{AdvancedShrinkingEngine as NewAdvancedShrinkingEngine, ShrinkResult as NewShrinkResult, ShrinkingMetrics as NewShrinkingMetrics, ChoicePattern, StringPatternType, ShrinkingContext, shrink_duplicated_blocks, shrink_floats_to_integers, shrink_strings_to_more_structured, lexicographic_weight, minimize_individual_choice_at, constraint_repair_shrinking, calculate_sequence_quality};
pub use self::constraints::*;
pub use self::dfa_string_generation::{
    DFAError, DFAState, LearnedDFA, LStarLearner, MembershipOracle, 
    RegexOracle, CustomOracle, DFAStringGenerator, PatternRecognitionEngine,
    AlphabetOptimizer, AdvancedDFALearner, DFAStatistics, GenerationStatistics
};
pub use self::field_access_system::*;
pub use self::float_constraint_type_system::{
    FloatConstraintTypeSystem, FloatGenerationStrategy, FloatConstraintAwareProvider
};
pub use self::indexing::*;
pub use self::navigation::*;
pub use self::node::*;
pub use self::shrinking_system::*;
pub use self::shrinking_demo::*;
pub use self::templating::*;
pub use self::value_generation::{
    ValueGenerator, EntropySource, BufferEntropySource, ValueGenerationError, 
    ValueGenerationResult, StandardValueGenerator
};
pub use self::weighted_selection::*;
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