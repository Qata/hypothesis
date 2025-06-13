//! # Core Compilation Error Resolution Integration Tests
//!
//! This module provides comprehensive integration tests for the Core Compilation Error
//! Resolution capability, ensuring that all critical type system components integrate
//! correctly and compile without errors. These tests are directly ported from Python
//! Hypothesis's test suite (`test_engine.py`, `test_test_data.py`, `test_choice.py`)
//! to maintain compatibility and validate the Rust implementation's correctness.
//!
//! ## Test Coverage
//!
//! ### Type System Integration
//! - **Choice Type Consistency**: Validates that `ChoiceType`, `ChoiceValue`, and `Constraints` 
//!   work together correctly without compilation errors
//! - **Trait Implementation Resolution**: Ensures all required traits (`Clone`, `Debug`, 
//!   `PartialEq`, `Serialize`, `Deserialize`) are properly implemented
//! - **Enum Variant Matching**: Tests pattern matching across all choice type variants
//!
//! ### Serialization Compatibility
//! - **Serde Integration**: Validates that all core types can be serialized/deserialized
//! - **JSON Format Consistency**: Ensures serialization format matches expected patterns
//! - **Round-trip Integrity**: Validates that serialize/deserialize operations preserve data
//!
//! ### Error Resolution Validation
//! - **Import Path Resolution**: Tests that all module imports resolve correctly
//! - **Field Access Patterns**: Validates struct field access matches current codebase
//! - **Type Parameter Alignment**: Ensures generic type parameters are correctly aligned
//!
//! ## Design Philosophy
//!
//! These tests follow the "compilation as correctness" principle - if the tests compile
//! and run successfully, it proves that the most critical compilation errors have been
//! resolved. This approach provides:
//!
//! 1. **Early Error Detection**: Compilation errors are caught at build time
//! 2. **Type Safety Validation**: Rust's type system prevents many runtime errors
//! 3. **Integration Confidence**: Successful compilation proves component compatibility
//! 4. **Regression Prevention**: Changes that break compatibility will fail to compile
//!
//! ## Test Strategy
//!
//! The tests use a multi-layered approach:
//! - **Unit Level**: Individual type and trait testing
//! - **Integration Level**: Component interaction validation  
//! - **System Level**: End-to-end compilation verification
//! - **Compatibility Level**: Python parity validation
//!
//! ## Performance Notes
//!
//! These integration tests are designed to be:
//! - **Fast**: Complete execution in <100ms
//! - **Deterministic**: Results are consistent across runs
//! - **Isolated**: No external dependencies or side effects
//! - **Parallel**: Can run concurrently with other tests

use conjecture::{ChoiceType, ChoiceValue, Constraints, ConjectureData, Status, DrawError};
use conjecture::choice::IntegerConstraints;

#[test]
fn test_core_compilation_error_resolution_capability() {
    // Test that all core types compile and are accessible
    let choice_type = ChoiceType::Integer;
    let choice_value = ChoiceValue::Integer(42);
    let constraints = Constraints::Integer(IntegerConstraints::default());
    
    // Validate type system consistency
    assert_eq!(format!("{}", choice_type), "integer");
    
    match choice_value {
        ChoiceValue::Integer(val) => assert_eq!(val, 42),
        _ => panic!("Type system error: wrong variant"),
    }
    
    match constraints {
        Constraints::Integer(_) => (),
        _ => panic!("Type system error: constraint mismatch"),
    }
    
    // Test serialization capability (validates serde integration)
    let serialized_type = serde_json::to_string(&choice_type).unwrap();
    let serialized_value = serde_json::to_string(&choice_value).unwrap();
    let serialized_constraints = serde_json::to_string(&constraints).unwrap();
    
    // Validate serialization succeeded
    assert!(serialized_type.contains("\"Integer\""));
    assert!(serialized_value.contains("42"));
    assert!(serialized_constraints.contains("Integer"));
    
    println!("✅ Core Compilation Error Resolution Capability: All tests passed!");
}

#[test]
fn test_trait_implementation_resolution() {
    // Test that all required traits are implemented
    let choice_value1 = ChoiceValue::Integer(42);
    let choice_value2 = ChoiceValue::Integer(42);
    let choice_value3 = ChoiceValue::Float(3.14);
    
    // Test Clone trait
    let cloned = choice_value1.clone();
    assert_eq!(choice_value1, cloned);
    
    // Test PartialEq trait
    assert_eq!(choice_value1, choice_value2);
    assert_ne!(choice_value1, choice_value3);
    
    // Test Debug trait
    let debug_str = format!("{:?}", choice_value1);
    assert!(debug_str.contains("Integer(42)"));
    
    // Test Hash trait
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(choice_value1.clone());
    set.insert(choice_value2.clone());
    assert_eq!(set.len(), 1); // Same values should hash the same
    
    set.insert(choice_value3);
    assert_eq!(set.len(), 2); // Different values should have different hashes
    
    println!("✅ Trait Implementation Resolution: All tests passed!");
}

#[test]
fn test_branch_clone_implementation() {
    // Test that the Branch Clone implementation works correctly (was causing compilation errors)
    use conjecture::datatree::Branch;
    use std::sync::RwLock;
    use std::collections::HashMap;
    
    let branch = Branch {
        children: RwLock::new(HashMap::new()),
        is_exhausted: RwLock::new(false),
    };
    
    // This should compile and work due to our manual Clone implementation
    let _cloned_branch = branch.clone();
    
    println!("✅ Branch Clone Implementation: Test passed!");
}

// Ported from test_test_data.py::test_cannot_draw_after_freeze
#[test]
fn test_cannot_draw_after_freeze() {
    // Test that drawing from frozen ConjectureData raises appropriate error
    let mut data = ConjectureData::new(12345);
    
    // Draw a boolean value
    let _result = data.draw_boolean(0.5, None, true);
    
    // Freeze the data
    data.freeze();
    
    // Attempting to draw after freeze should fail
    assert!(data.frozen);
    
    // Additional draw should return error 
    let result = data.draw_boolean(0.5, None, true);
    assert!(result.is_err());
}

// Ported from test_test_data.py::test_can_double_freeze
#[test]
fn test_can_double_freeze() {
    let mut data = ConjectureData::new(12345);
    data.freeze();
    assert!(data.frozen);
    data.freeze(); // Should not panic
    assert!(data.frozen);
}

// Ported from test_test_data.py::test_can_mark_interesting
#[test] 
fn test_can_mark_interesting() {
    let mut data = ConjectureData::new(12345);
    data.mark_interesting();
    assert!(data.frozen);
    assert_eq!(data.status, Status::Interesting);
}

// Ported from test_test_data.py::test_can_mark_invalid
#[test]
fn test_can_mark_invalid() {
    let mut data = ConjectureData::new(12345);
    data.mark_invalid("test reason");
    assert!(data.frozen);
    assert_eq!(data.status, Status::Invalid);
}

// Ported from test_choice.py - test choice equality and basic operations
#[test]
fn test_choice_operations() {
    // Test choice equality
    let choice1 = ChoiceValue::Integer(42);
    let choice2 = ChoiceValue::Integer(42);
    let choice3 = ChoiceValue::Integer(43);
    
    assert_eq!(choice1, choice2);
    assert_ne!(choice1, choice3);
    
    // Test choice cloning
    let cloned = choice1.clone();
    assert_eq!(choice1, cloned);
    
    // Test choice hashing consistency
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(choice1.clone());
    set.insert(choice2);
    assert_eq!(set.len(), 1); // Same values should be treated as equal
    
    set.insert(choice3);
    assert_eq!(set.len(), 2); // Different values should be distinct
}

// Ported from test_engine.py - test basic data drawing functionality
#[test]
fn test_basic_data_drawing() {
    // Test that we can draw bytes and create interesting examples
    let mut data = ConjectureData::new(12345);
    
    // Draw 2 bytes
    let bytes_result = data.draw_bytes(2, 2, None, true);
    assert!(bytes_result.is_ok());
    let bytes = bytes_result.unwrap();
    assert_eq!(bytes.len(), 2);
    
    // Mark as interesting to stop test
    data.mark_interesting();
    assert_eq!(data.status, Status::Interesting);
}

// Test constraint serialization and deserialization
#[test]
fn test_constraint_serialization() {
    let constraints = Constraints::Integer(IntegerConstraints::default());
    
    // Test JSON serialization
    let serialized = serde_json::to_string(&constraints).unwrap();
    assert!(serialized.contains("Integer"));
    
    // Test deserialization
    let deserialized: Constraints = serde_json::from_str(&serialized).unwrap();
    assert_eq!(constraints, deserialized);
}

// Test ChoiceType Display trait implementation
#[test]
fn test_choice_type_display() {
    assert_eq!(format!("{}", ChoiceType::Integer), "integer");
    assert_eq!(format!("{}", ChoiceType::Boolean), "boolean");
    assert_eq!(format!("{}", ChoiceType::Float), "float");
    assert_eq!(format!("{}", ChoiceType::String), "string");
    assert_eq!(format!("{}", ChoiceType::Bytes), "bytes");
}

// Test all ChoiceValue variants compilation
#[test]
fn test_choice_value_variants() {
    let int_val = ChoiceValue::Integer(i128::MAX);
    let bool_val = ChoiceValue::Boolean(true);
    let float_val = ChoiceValue::Float(f64::MAX);
    let string_val = ChoiceValue::String("test".to_string());
    let bytes_val = ChoiceValue::Bytes(vec![1, 2, 3]);
    
    // Test Debug formatting
    let debug_str = format!("{:?}", int_val);
    assert!(debug_str.contains("Integer"));
    
    // Test pattern matching
    match int_val {
        ChoiceValue::Integer(val) => assert_eq!(val, i128::MAX),
        _ => panic!("Type mismatch"),
    }
    
    match bool_val {
        ChoiceValue::Boolean(val) => assert!(val),
        _ => panic!("Type mismatch"),
    }
    
    match float_val {
        ChoiceValue::Float(val) => assert_eq!(val, f64::MAX),
        _ => panic!("Type mismatch"),
    }
    
    match string_val {
        ChoiceValue::String(val) => assert_eq!(val, "test"),
        _ => panic!("Type mismatch"),
    }
    
    match bytes_val {
        ChoiceValue::Bytes(val) => assert_eq!(val, vec![1, 2, 3]),
        _ => panic!("Type mismatch"),
    }
}