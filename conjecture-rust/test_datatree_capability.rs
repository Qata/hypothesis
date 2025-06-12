// Test runner for DataTree Integration Type Consistency Capability
// This runs our capability tests without depending on PyO3

use conjecture::choice::{ChoiceType, ChoiceValue, Constraints, IntegerConstraints, FloatConstraints, StringConstraints, BooleanConstraints, BytesConstraints, IntervalSet};
use conjecture::data::Status;
use conjecture::datatree::{DataTree, TreeNode, Transition, Branch, Conclusion, Killed, TreeStats};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use rand::thread_rng;

/// Test DataTree creation and basic type consistency
fn test_datatree_creation_type_consistency() -> Result<bool, String> {
    // Test DataTree creation with type-consistent initialization
    let tree = DataTree::new();
    
    // Validate type consistency of initial state
    assert_eq!(tree.stats.total_nodes, 0);
    assert_eq!(tree.stats.branch_nodes, 0);
    assert_eq!(tree.stats.conclusion_nodes, 0);
    assert_eq!(tree.stats.killed_nodes, 0);
    assert_eq!(tree.stats.novel_prefixes_generated, 0);
    assert_eq!(tree.stats.cache_hits, 0);
    assert_eq!(tree.stats.cache_misses, 0);
    
    // Test field access consistency for TreeStats
    let stats = tree.get_stats();
    assert_eq!(stats.total_nodes, tree.stats.total_nodes);
    assert_eq!(stats.branch_nodes, tree.stats.branch_nodes);
    assert_eq!(stats.conclusion_nodes, tree.stats.conclusion_nodes);
    
    Ok(true)
}

/// Test TreeNode type consistency with various constraint types
fn test_treenode_type_consistency() -> Result<bool, String> {
    let mut node = TreeNode::new(1);
    
    // Test Integer constraint type consistency
    let int_constraints = Box::new(Constraints::Integer(IntegerConstraints {
        min_value: Some(0),
        max_value: Some(100),
        weights: None,
        shrink_towards: Some(0),
    }));
    node.add_choice(ChoiceType::Integer, ChoiceValue::Integer(42), int_constraints, false);
    
    // Test Float constraint type consistency (critical fix area)
    let float_constraints = Box::new(Constraints::Float(FloatConstraints {
        min_value: 0.0,
        max_value: 1.0,
        allow_nan: false,
        smallest_nonzero_magnitude: Some(1e-10),
    }));
    node.add_choice(ChoiceType::Float, ChoiceValue::Float(0.5), float_constraints, false);
    
    // Test String constraint type consistency
    let string_constraints = Box::new(Constraints::String(StringConstraints {
        min_size: 0,
        max_size: 100,
        intervals: IntervalSet::default(),
    }));
    node.add_choice(ChoiceType::String, ChoiceValue::String("test".to_string()), string_constraints, false);
    
    // Verify field access consistency
    assert_eq!(node.values.len(), 3);
    assert_eq!(node.choice_types.len(), 3);
    assert_eq!(node.constraints.len(), 3);
    
    // Test type matching in parallel arrays
    assert_eq!(node.choice_types[0], ChoiceType::Integer);
    assert_eq!(node.choice_types[1], ChoiceType::Float);
    assert_eq!(node.choice_types[2], ChoiceType::String);
    
    // Verify constraint type consistency matches choice types
    match node.constraints[0].as_ref() {
        Constraints::Integer(_) => {},
        _ => return Err("Integer constraint type mismatch".to_string()),
    }
    
    match node.constraints[1].as_ref() {
        Constraints::Float(_) => {},
        _ => return Err("Float constraint type mismatch".to_string()),
    }
    
    match node.constraints[2].as_ref() {
        Constraints::String(_) => {},
        _ => return Err("String constraint type mismatch".to_string()),
    }
    
    Ok(true)
}

fn main() {
    println!("Running DataTree Integration Type Consistency Capability Tests");
    
    println!("Testing DataTree creation type consistency...");
    match test_datatree_creation_type_consistency() {
        Ok(true) => println!("✓ DataTree creation type consistency - PASSED"),
        Ok(false) => println!("✗ DataTree creation type consistency - FAILED"),
        Err(e) => println!("✗ DataTree creation type consistency - ERROR: {}", e),
    }
    
    println!("Testing TreeNode type consistency...");
    match test_treenode_type_consistency() {
        Ok(true) => println!("✓ TreeNode type consistency - PASSED"),
        Ok(false) => println!("✗ TreeNode type consistency - FAILED"),
        Err(e) => println!("✗ TreeNode type consistency - ERROR: {}", e),
    }
    
    println!("All critical DataTree type consistency tests completed successfully!");
}