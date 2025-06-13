//! Integration test for Core Compilation Error Resolution Capability

use conjecture_rust::choice::*;

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
    use conjecture_rust::datatree::Branch;
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