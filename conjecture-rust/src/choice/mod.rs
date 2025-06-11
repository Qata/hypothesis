//! Choice system for Conjecture engine
//! 
//! This module implements Python's choice-based architecture where all randomness
//! flows through strongly-typed choices with associated constraints.

pub mod advanced_shrinking;
mod constraints;
pub mod indexing;
mod indexing_correct;
mod navigation;
mod node;
pub mod shrinking_system;
pub mod shrinking_demo;
pub mod templating;
mod values;

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

// #[cfg(test)]
// mod advanced_shrinking_tests;

pub use self::advanced_shrinking::*;
pub use self::constraints::*;
pub use self::indexing::*;
pub use self::navigation::*;
pub use self::node::*;
pub use self::shrinking_system::*;
pub use self::shrinking_demo::*;
pub use self::templating::*;
pub use self::values::*;

/// Choice types that can be drawn
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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