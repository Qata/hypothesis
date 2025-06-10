//! Choice system for Conjecture engine
//! 
//! This module implements Python's choice-based architecture where all randomness
//! flows through strongly-typed choices with associated constraints.

mod constraints;
mod node;
mod values;

#[cfg(test)]
mod python_parity_tests;

pub use constraints::*;
pub use node::*;
pub use values::*;

/// Choice types that can be drawn
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

/// Constraints for different choice types
#[derive(Debug, Clone, PartialEq)]
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
        println!("CHOICE DEBUG: Testing ChoiceType display formatting");
        assert_eq!(format!("{}", ChoiceType::Integer), "integer");
        assert_eq!(format!("{}", ChoiceType::Boolean), "boolean");
        assert_eq!(format!("{}", ChoiceType::Float), "float");
        assert_eq!(format!("{}", ChoiceType::String), "string");
        assert_eq!(format!("{}", ChoiceType::Bytes), "bytes");
        println!("CHOICE DEBUG: All ChoiceType display tests passed");
    }

    #[test]
    fn test_choice_value_variants() {
        println!("CHOICE DEBUG: Testing ChoiceValue variant creation");
        let _int_val = ChoiceValue::Integer(42);
        let _bool_val = ChoiceValue::Boolean(true);
        let _float_val = ChoiceValue::Float(3.14);
        let _string_val = ChoiceValue::String("hello".to_string());
        let _bytes_val = ChoiceValue::Bytes(vec![1, 2, 3]);
        println!("CHOICE DEBUG: All ChoiceValue variants created successfully");
    }
}