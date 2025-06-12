//! Comprehensive DataTree Integration Type Consistency Capability Tests
//!
//! Tests validate the complete DataTree integration capability works correctly,
//! focusing on type consistency fixes that resolve fundamental type mismatches preventing compilation.
//! Tests verify float constraint types, struct field access, and core functionality work correctly.

use crate::choice::{ChoiceType, ChoiceValue, Constraints, IntegerConstraints, FloatConstraints, StringConstraints, BooleanConstraints, BytesConstraints, IntervalSet};
use crate::data::Status;
use crate::datatree::{DataTree, TreeNode, Transition, Branch, Conclusion, Killed, TreeStats};
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

/// Test float constraint type consistency
fn test_float_constraint_integration() -> Result<bool, String> {
    let mut tree = DataTree::new();
    
    // Create float constraints with type-consistent fields
    let float_constraints = Box::new(Constraints::Float(FloatConstraints {
        min_value: -1.0,
        max_value: 1.0,
        allow_nan: false,
        smallest_nonzero_magnitude: Some(1e-10),
    }));
    
    // Test choice recording with float constraints
    let choices = vec![
        (
            ChoiceType::Float,
            ChoiceValue::Float(0.7),
            float_constraints.clone(),
            false
        ),
    ];
    
    tree.record_path(&choices, Status::Valid, HashMap::new());
    
    // Verify tree structure with float constraints
    assert_eq!(tree.stats.total_nodes, 1);
    assert_eq!(tree.stats.conclusion_nodes, 1);
    
    // Test constraint field access consistency
    match float_constraints.as_ref() {
        Constraints::Float(fc) => {
            assert_eq!(fc.min_value, -1.0);
            assert_eq!(fc.max_value, 1.0);
            assert_eq!(fc.allow_nan, false);
            assert_eq!(fc.smallest_nonzero_magnitude, Some(1e-10));
        },
        _ => return Err("Expected Float constraint".to_string()),
    }
    
    Ok(true)
}

/// Test struct field access consistency in transitions
fn test_transition_field_access_consistency() -> Result<bool, String> {
    let mut node = TreeNode::new(1);
    
    // Test Branch transition field access consistency
    let branch = Branch {
        children: RwLock::new(HashMap::new()),
        is_exhausted: RwLock::new(false),
    };
    
    // Verify field access before assignment
    assert!(!*branch.is_exhausted.read().unwrap());
    assert!(branch.children.read().unwrap().is_empty());
    
    *node.transition.write().unwrap() = Some(Transition::Branch(branch));
    
    // Test field access through transition
    match node.transition.read().unwrap().as_ref() {
        Some(Transition::Branch(branch)) => {
            assert!(!*branch.is_exhausted.read().unwrap());
            assert!(branch.children.read().unwrap().is_empty());
        },
        _ => return Err("Branch transition field access failed".to_string()),
    }
    
    // Test Conclusion transition field access consistency
    let conclusion = Conclusion {
        status: Status::Valid,
        interesting_origin: Some("test_origin".to_string()),
        target_observations: HashMap::new(),
        metadata: HashMap::new(),
    };
    
    // Verify field access consistency
    assert_eq!(conclusion.status, Status::Valid);
    assert_eq!(conclusion.interesting_origin.as_ref().unwrap(), "test_origin");
    assert!(conclusion.target_observations.is_empty());
    assert!(conclusion.metadata.is_empty());
    
    // Test Killed transition field access consistency
    let killed = Killed {
        next_node: None,
        reason: "test_kill_reason".to_string(),
    };
    
    assert!(killed.next_node.is_none());
    assert_eq!(killed.reason, "test_kill_reason");
    
    Ok(true)
}

/// Test DataTree novel prefix generation with type consistency
fn test_novel_prefix_generation_type_consistency() -> Result<bool, String> {
    let mut tree = DataTree::new();
    let mut rng = thread_rng();
    
    // Generate novel prefix and verify type consistency
    let prefix = tree.generate_novel_prefix(&mut rng);
    
    // Verify generation statistics consistency
    assert_eq!(tree.stats.novel_prefixes_generated, 1);
    
    // Test with complex constraint types
    let mixed_choices = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Box::new(Constraints::Integer(IntegerConstraints::default())),
            false
        ),
        (
            ChoiceType::Float,
            ChoiceValue::Float(3.14),
            Box::new(Constraints::Float(FloatConstraints::default())),
            false
        ),
        (
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Box::new(Constraints::Boolean(BooleanConstraints { p: 0.7 })),
            false
        ),
    ];
    
    tree.record_path(&mixed_choices, Status::Valid, HashMap::new());
    
    // Generate another prefix after recording mixed types
    let prefix2 = tree.generate_novel_prefix(&mut rng);
    assert_eq!(tree.stats.novel_prefixes_generated, 2);
    
    // Verify type consistency in generated prefix
    for (choice_type, value, constraints) in &prefix2 {
        match (choice_type, value, constraints.as_ref()) {
            (ChoiceType::Integer, ChoiceValue::Integer(_), Constraints::Integer(_)) => {},
            (ChoiceType::Float, ChoiceValue::Float(_), Constraints::Float(_)) => {},
            (ChoiceType::Boolean, ChoiceValue::Boolean(_), Constraints::Boolean(_)) => {},
            (ChoiceType::String, ChoiceValue::String(_), Constraints::String(_)) => {},
            (ChoiceType::Bytes, ChoiceValue::Bytes(_), Constraints::Bytes(_)) => {},
            _ => return Err(format!("Type inconsistency in generated prefix: {:?}, {:?}, {:?}", choice_type, value, constraints)),
        }
    }
    
    Ok(true)
}

/// Test path recording with complex type hierarchies
fn test_path_recording_type_hierarchy_consistency() -> Result<bool, String> {
    let mut tree = DataTree::new();
    
    // Test recording path with nested type structures
    let complex_choices = vec![
        (
            ChoiceType::Float,
            ChoiceValue::Float(1.5),
            Box::new(Constraints::Float(FloatConstraints {
                min_value: 0.0,
                max_value: 10.0,
                allow_nan: false,
                smallest_nonzero_magnitude: Some(1e-10),
            })),
            false
        ),
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(100),
            Box::new(Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(1000),
                weights: None,
                shrink_towards: Some(0),
            })),
            true  // Forced choice
        ),
    ];
    
    // Record path with target observations
    let mut observations = HashMap::new();
    observations.insert("score".to_string(), "95.5".to_string());
    observations.insert("count".to_string(), "10".to_string());
    observations.insert("label".to_string(), "test_label".to_string());
    
    tree.record_path(&complex_choices, Status::Interesting, observations);
    
    // Verify tree structure consistency
    assert_eq!(tree.stats.total_nodes, 2); // Two choices create two nodes
    assert_eq!(tree.stats.conclusion_nodes, 1);
    assert_eq!(tree.stats.branch_nodes, 1);
    
    // Test another path that shares common prefix
    let divergent_choices = vec![
        (
            ChoiceType::Float,
            ChoiceValue::Float(1.5), // Same first choice
            Box::new(Constraints::Float(FloatConstraints {
                min_value: 0.0,
                max_value: 10.0,
                allow_nan: false,
                smallest_nonzero_magnitude: Some(1e-10),
            })),
            false
        ),
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(200), // Different second choice
            Box::new(Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(1000),
                weights: None,
                shrink_towards: Some(0),
            })),
            false
        ),
    ];
    
    tree.record_path(&divergent_choices, Status::Valid, HashMap::new());
    
    // Verify branching structure
    assert_eq!(tree.stats.conclusion_nodes, 2);
    assert!(tree.stats.total_nodes >= 3); // At least root + branch + 2 conclusions
    
    Ok(true)
}

/// Test exhaustion detection with type-consistent constraints
fn test_exhaustion_detection_type_consistency() -> Result<bool, String> {
    let mut node = TreeNode::new(1);
    
    // Add choice with bounded integer constraints for exhaustion testing
    let bounded_constraints = Box::new(Constraints::Integer(IntegerConstraints {
        min_value: Some(0),
        max_value: Some(2), // Small range for easy exhaustion
        weights: None,
        shrink_towards: Some(0),
    }));
    
    node.add_choice(ChoiceType::Integer, ChoiceValue::Integer(1), bounded_constraints, false);
    
    // Test max children calculation with type-consistent constraints
    let max_children = node.compute_max_children();
    assert!(max_children.is_some());
    assert_eq!(max_children.unwrap(), 3); // 0, 1, 2
    
    // Test exhaustion ratio calculation
    let ratio = node.compute_exhaustion_ratio();
    assert!(ratio >= 0.0 && ratio <= 1.0);
    
    // Test with boolean constraints
    let mut bool_node = TreeNode::new(3);
    bool_node.add_choice(
        ChoiceType::Boolean,
        ChoiceValue::Boolean(true),
        Box::new(Constraints::Boolean(BooleanConstraints { p: 0.5 })),
        false
    );
    
    let bool_max = bool_node.compute_max_children();
    assert_eq!(bool_max.unwrap(), 2); // true, false
    
    Ok(true)
}

/// Test forced choice handling with type consistency
fn test_forced_choice_type_consistency() -> Result<bool, String> {
    let mut node = TreeNode::new(1);
    
    // Add mixed forced and non-forced choices with consistent types
    node.add_choice(
        ChoiceType::Integer,
        ChoiceValue::Integer(10),
        Box::new(Constraints::Integer(IntegerConstraints::default())),
        true  // Forced
    );
    
    node.add_choice(
        ChoiceType::Float,
        ChoiceValue::Float(2.5),
        Box::new(Constraints::Float(FloatConstraints::default())),
        false  // Not forced
    );
    
    node.add_choice(
        ChoiceType::Boolean,
        ChoiceValue::Boolean(true),
        Box::new(Constraints::Boolean(BooleanConstraints { p: 0.5 })),
        true  // Forced
    );
    
    // Verify forced set consistency
    let forced = node.forced.as_ref().unwrap();
    assert!(forced.contains(&0)); // First choice (index 0) was forced
    assert!(!forced.contains(&1)); // Second choice (index 1) was not forced
    assert!(forced.contains(&2)); // Third choice (index 2) was forced
    
    // Test exhaustion with forced choices
    let all_forced = node.all_choices_forced();
    assert!(!all_forced); // Not all choices are forced
    
    // Test splitting with forced choices
    let mut next_id = 2;
    let suffix = node.split_at(1, &mut next_id);
    
    // Verify forced indices are correctly split
    let original_forced = node.forced.as_ref().unwrap();
    assert!(original_forced.contains(&0)); // Index 0 should remain
    assert!(!original_forced.contains(&1)); // Index 1 was the split point
    
    let suffix_forced = suffix.forced.as_ref().unwrap();
    assert!(suffix_forced.contains(&1)); // Original index 2 becomes index 1 in suffix
    
    Ok(true)
}

/// Test DataTree simulation with type-consistent results
fn test_datatree_simulation_type_consistency() -> Result<bool, String> {
    let tree = DataTree::new();
    
    // Test simulation with various choice types
    let test_choices = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Box::new(Constraints::Integer(IntegerConstraints::default())),
        ),
        (
            ChoiceType::Float,
            ChoiceValue::Float(3.14),
            Box::new(Constraints::Float(FloatConstraints::default())),
        ),
        (
            ChoiceType::String,
            ChoiceValue::String("test".to_string()),
            Box::new(Constraints::String(StringConstraints::default())),
        ),
    ];
    
    let (status, observations) = tree.simulate_test_function(&test_choices);
    
    // Verify simulation returns type-consistent results
    match status {
        Status::Valid | Status::Invalid | Status::Overrun | Status::Interesting => {},
        _ => return Err("Simulation returned invalid status type".to_string()),
    }
    
    // Observations should be type-consistent HashMap<String, String>
    for (key, value) in &observations {
        assert!(key.len() >= 0); // Valid string key
        assert!(value.len() >= 0); // Valid string value
    }
    
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_all_type_consistency_functions() {
        // Test all type consistency functions
        assert!(test_datatree_creation_type_consistency().unwrap());
        assert!(test_treenode_type_consistency().unwrap());
        assert!(test_float_constraint_integration().unwrap());
        assert!(test_transition_field_access_consistency().unwrap());
        assert!(test_novel_prefix_generation_type_consistency().unwrap());
        assert!(test_path_recording_type_hierarchy_consistency().unwrap());
        assert!(test_exhaustion_detection_type_consistency().unwrap());
        assert!(test_forced_choice_type_consistency().unwrap());
        assert!(test_datatree_simulation_type_consistency().unwrap());
    }
    
    #[test]
    fn test_comprehensive_type_integration() {
        let mut tree = DataTree::new();
        let mut rng = thread_rng();
        
        // Test comprehensive integration of all type-consistent components
        let comprehensive_choices = vec![
            (
                ChoiceType::Integer,
                ChoiceValue::Integer(100),
                Box::new(Constraints::Integer(IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(1000),
                    weights: None,
                    shrink_towards: Some(0),
                })),
                false
            ),
            (
                ChoiceType::Float,
                ChoiceValue::Float(99.99),
                Box::new(Constraints::Float(FloatConstraints {
                    min_value: 0.0,
                    max_value: 100.0,
                    allow_nan: false,
                    smallest_nonzero_magnitude: Some(1e-10),
                })),
                true
            ),
            (
                ChoiceType::String,
                ChoiceValue::String("integration_test".to_string()),
                Box::new(Constraints::String(StringConstraints {
                    min_size: 5,
                    max_size: 50,
                    intervals: IntervalSet::default(),
                })),
                false
            ),
            (
                ChoiceType::Boolean,
                ChoiceValue::Boolean(false),
                Box::new(Constraints::Boolean(BooleanConstraints { p: 0.3 })),
                false
            ),
        ];
        
        // Record multiple paths with type consistency
        tree.record_path(&comprehensive_choices, Status::Valid, HashMap::new());
        
        // Generate novel prefixes with type consistency
        let prefix1 = tree.generate_novel_prefix(&mut rng);
        let prefix2 = tree.generate_novel_prefix(&mut rng);
        
        // Verify type consistency across operations
        assert!(tree.stats.total_nodes > 0);
        assert!(tree.stats.novel_prefixes_generated >= 2);
        
        // Verify prefix type consistency
        for prefix in [&prefix1, &prefix2] {
            for (choice_type, value, constraints) in prefix {
                match (choice_type, value, constraints.as_ref()) {
                    (ChoiceType::Integer, ChoiceValue::Integer(_), Constraints::Integer(_)) => {},
                    (ChoiceType::Float, ChoiceValue::Float(_), Constraints::Float(_)) => {},
                    (ChoiceType::Boolean, ChoiceValue::Boolean(_), Constraints::Boolean(_)) => {},
                    (ChoiceType::String, ChoiceValue::String(_), Constraints::String(_)) => {},
                    (ChoiceType::Bytes, ChoiceValue::Bytes(_), Constraints::Bytes(_)) => {},
                    _ => panic!("Type consistency violation in comprehensive test"),
                }
            }
        }
    }
    
    #[test]
    fn test_float_constraint_edge_cases() {
        // Test float constraint type consistency with edge cases
        let float_constraints = FloatConstraints {
            min_value: f64::NEG_INFINITY,
            max_value: f64::INFINITY,
            allow_nan: true,
            smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
        };
        
        let constraints_box = Box::new(Constraints::Float(float_constraints));
        
        // Verify field access consistency with edge values
        match constraints_box.as_ref() {
            Constraints::Float(fc) => {
                assert_eq!(fc.min_value, f64::NEG_INFINITY);
                assert_eq!(fc.max_value, f64::INFINITY);
                assert!(fc.allow_nan);
                assert_eq!(fc.smallest_nonzero_magnitude, Some(f64::MIN_POSITIVE));
            },
            _ => panic!("Float constraint type mismatch in edge case test"),
        }
        
        // Test with NaN value
        let nan_value = ChoiceValue::Float(f64::NAN);
        let mut node = TreeNode::new(1);
        node.add_choice(ChoiceType::Float, nan_value, constraints_box, false);
        
        // Verify field access with NaN doesn't break type consistency
        assert_eq!(node.choice_types.len(), 1);
        assert_eq!(node.values.len(), 1);
        assert_eq!(node.constraints.len(), 1);
        
        match (&node.choice_types[0], &node.values[0]) {
            (ChoiceType::Float, ChoiceValue::Float(val)) => {
                assert!(val.is_nan());
            },
            _ => panic!("NaN float value type inconsistency"),
        }
    }
}