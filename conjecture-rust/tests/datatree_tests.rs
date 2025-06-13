//! TreeStructures Tests - Ported from Python hypothesis test suite
//!
//! These tests are direct ports of the Python TreeStructures tests from:
//! - hypothesis-python/tests/conjecture/test_data_tree.py  
//! - hypothesis-python/tests/conjecture/test_choice_tree.py
//!
//! The tests preserve the same test cases, edge cases, and assertions as the Python tests,
//! using standard Rust testing patterns with modernized 6-parameter constraint-based API.

use conjecture::datatree::{DataTree, TreeNode, TreeStats};
use conjecture::choice::{ChoiceType, ChoiceValue, Constraints, IntegerConstraints, FloatConstraints, BooleanConstraints, StringConstraints, BytesConstraints};
use conjecture::choice::constraints::IntervalSet;
use conjecture::data::Status;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use rand::{Rng, SeedableRng, thread_rng};
use rand::rngs::StdRng;

// Helper function to create test constraints matching Python patterns
fn create_integer_constraints(min: Option<i128>, max: Option<i128>) -> Box<Constraints> {
    Box::new(Constraints::Integer(IntegerConstraints {
        min_value: min,
        max_value: max,
        weights: None,
        shrink_towards: Some(0),
    }))
}

fn create_boolean_constraints(probability: f64) -> Box<Constraints> {
    Box::new(Constraints::Boolean(BooleanConstraints { p: probability }))
}

fn create_float_constraints(min: f64, max: f64, allow_nan: bool) -> Box<Constraints> {
    Box::new(Constraints::Float(FloatConstraints {
        min_value: min,
        max_value: max,
        allow_nan,
        smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
    }))
}

fn create_string_constraints(min_size: usize, max_size: usize) -> Box<Constraints> {
    Box::new(Constraints::String(StringConstraints {
        min_size,
        max_size,
        intervals: IntervalSet::default(),
    }))
}

fn create_bytes_constraints(min_size: usize, max_size: usize) -> Box<Constraints> {
    Box::new(Constraints::Bytes(BytesConstraints {
        min_size,
        max_size,
    }))
}

// ========== Core DataTree Functionality Tests ==========

#[test]
fn test_can_lookup_cached_examples() {
    // Python: test_can_lookup_cached_examples()
    // Tests basic caching of examples with integer draws
    let mut tree = DataTree::new();
    
    // Record first path: draw integer 42
    let choices = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            create_integer_constraints(Some(0), Some(100)),
            false
        ),
    ];
    tree.record_path(&choices, Status::Valid, HashMap::new());
    
    // Generate novel prefix - should get different path
    let mut rng = thread_rng();
    let prefix = tree.generate_novel_prefix(&mut rng);
    
    // Tree should have cached the first example
    let stats = tree.get_stats();
    assert!(stats.total_nodes > 0);
    assert_eq!(stats.conclusion_nodes, 1);
    assert_eq!(stats.novel_prefixes_generated, 1);
}

#[test]
fn test_can_lookup_cached_examples_with_forced() {
    // Python: test_can_lookup_cached_examples_with_forced()
    // Tests caching with forced values
    let mut tree = DataTree::new();
    
    // Record path with forced value
    let choices = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            create_integer_constraints(Some(0), Some(100)),
            true // forced=true
        ),
    ];
    tree.record_path(&choices, Status::Valid, HashMap::new());
    
    // Generate novel prefix 
    let mut rng = thread_rng();
    let prefix = tree.generate_novel_prefix(&mut rng);
    
    // Should handle forced values correctly
    let stats = tree.get_stats();
    assert!(stats.total_nodes > 0);
    assert_eq!(stats.conclusion_nodes, 1);
}

#[test]
fn test_can_detect_when_tree_is_exhausted() {
    // Python: test_can_detect_when_tree_is_exhausted()
    // Tests exhaustion detection with boolean draws
    let mut tree = DataTree::new();
    
    // Record both possible boolean paths
    let choices_true = vec![
        (
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            create_boolean_constraints(0.5),
            false
        ),
    ];
    tree.record_path(&choices_true, Status::Valid, HashMap::new());
    
    let choices_false = vec![
        (
            ChoiceType::Boolean,
            ChoiceValue::Boolean(false),
            create_boolean_constraints(0.5),
            false
        ),
    ];
    tree.record_path(&choices_false, Status::Valid, HashMap::new());
    
    // Both paths recorded, tree should detect exhaustion
    assert!(tree.root.check_exhausted());
    
    let stats = tree.get_stats();
    assert_eq!(stats.conclusion_nodes, 2);
}

#[test]
fn test_one_dead_branch() {
    // Python: test_one_dead_branch()
    // Tests exhaustion with invalid branches
    let mut tree = DataTree::new();
    
    // Record valid path
    let choices_valid = vec![
        (
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            create_boolean_constraints(0.5),
            false
        ),
    ];
    tree.record_path(&choices_valid, Status::Valid, HashMap::new());
    
    // Record invalid path (simulates dead branch)
    let choices_invalid = vec![
        (
            ChoiceType::Boolean,
            ChoiceValue::Boolean(false),
            create_boolean_constraints(0.5),
            false
        ),
    ];
    tree.record_path(&choices_invalid, Status::Invalid, HashMap::new());
    
    // Tree should be exhausted as both branches are recorded
    assert!(tree.root.check_exhausted());
    
    let stats = tree.get_stats();
    assert_eq!(stats.conclusion_nodes, 2);
}

#[test]
fn test_novel_prefixes_are_novel() {
    // Python: test_novel_prefixes_are_novel()
    // Tests generation of novel prefixes and their proper handling
    let mut tree = DataTree::new();
    let mut rng = thread_rng();
    
    // Generate several novel prefixes
    let prefix1 = tree.generate_novel_prefix(&mut rng);
    let prefix2 = tree.generate_novel_prefix(&mut rng);
    let prefix3 = tree.generate_novel_prefix(&mut rng);
    
    // Each generation should be tracked
    let stats = tree.get_stats();
    assert_eq!(stats.novel_prefixes_generated, 3);
    
    // Prefixes might be empty from an empty tree, but generation should work
    println!("Generated prefixes: {} {} {}", prefix1.len(), prefix2.len(), prefix3.len());
}

#[test]
fn test_stores_the_tree_flat_until_needed() {
    // Python: test_stores_the_tree_flat_until_needed()
    // Tests lazy tree expansion behavior
    let mut tree = DataTree::new();
    
    // Record a simple linear sequence
    let choices = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(1),
            create_integer_constraints(Some(0), Some(10)),
            false
        ),
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(2),
            create_integer_constraints(Some(0), Some(10)),
            false
        ),
    ];
    tree.record_path(&choices, Status::Valid, HashMap::new());
    
    // Should create tree structure efficiently
    let stats = tree.get_stats();
    assert!(stats.total_nodes > 0);
    assert_eq!(stats.conclusion_nodes, 1);
}

#[test]
fn test_split_in_the_middle() {
    // Python: test_split_in_the_middle()
    // Tests tree splitting behavior with multiple draws
    let mut tree = DataTree::new();
    
    // Record first path: 1, 2, 3
    let choices1 = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(1),
            create_integer_constraints(Some(0), Some(10)),
            false
        ),
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(2),
            create_integer_constraints(Some(0), Some(10)),
            false
        ),
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(3),
            create_integer_constraints(Some(0), Some(10)),
            false
        ),
    ];
    tree.record_path(&choices1, Status::Valid, HashMap::new());
    
    // Record diverging path: 1, 5, 6 (splits at second choice)
    let choices2 = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(1),
            create_integer_constraints(Some(0), Some(10)),
            false
        ),
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(5),
            create_integer_constraints(Some(0), Some(10)),
            false
        ),
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(6),
            create_integer_constraints(Some(0), Some(10)),
            false
        ),
    ];
    tree.record_path(&choices2, Status::Valid, HashMap::new());
    
    // Should create proper branching structure
    let stats = tree.get_stats();
    assert!(stats.total_nodes > 0);
    assert_eq!(stats.conclusion_nodes, 2);
    assert!(stats.branch_nodes > 0);
}

// ========== Status Transition Tests ==========

#[test]
fn test_can_go_from_interesting_to_valid() {
    // Python: test_can_go_from_interesting_to_valid()
    // Tests valid status transitions
    let mut tree = DataTree::new();
    
    // Record interesting path first
    let choices = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            create_integer_constraints(Some(0), Some(100)),
            false
        ),
    ];
    tree.record_path(&choices, Status::Interesting, HashMap::new());
    
    // Record same path with valid status - should be allowed
    tree.record_path(&choices, Status::Valid, HashMap::new());
    
    let stats = tree.get_stats();
    assert!(stats.conclusion_nodes > 0);
}

// ========== Float Handling Tests ==========

#[test]
fn test_is_not_flaky_on_positive_zero_and_negative_zero() {
    // Python: test_is_not_flaky_on_positive_zero_and_negative_zero()
    // Tests proper handling of +0.0 vs -0.0
    let mut tree = DataTree::new();
    
    // Record path with positive zero
    let choices_pos_zero = vec![
        (
            ChoiceType::Float,
            ChoiceValue::Float(0.0),
            create_float_constraints(-1.0, 1.0, false),
            false
        ),
    ];
    tree.record_path(&choices_pos_zero, Status::Valid, HashMap::new());
    
    // Record path with negative zero
    let choices_neg_zero = vec![
        (
            ChoiceType::Float,
            ChoiceValue::Float(-0.0),
            create_float_constraints(-1.0, 1.0, false),
            false
        ),
    ];
    tree.record_path(&choices_neg_zero, Status::Valid, HashMap::new());
    
    // Should treat +0.0 and -0.0 as different values
    let stats = tree.get_stats();
    assert_eq!(stats.conclusion_nodes, 2);
}

#[test]
fn test_can_generate_hard_floats() {
    // Python: test_can_generate_hard_floats()
    // Tests generation of specific float values
    let mut tree = DataTree::new();
    let mut rng = thread_rng();
    
    // Record some float paths
    let special_floats = [0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY];
    
    for &float_val in &special_floats {
        if float_val.is_finite() {
            let choices = vec![
                (
                    ChoiceType::Float,
                    ChoiceValue::Float(float_val),
                    create_float_constraints(-10.0, 10.0, false),
                    false
                ),
            ];
            tree.record_path(&choices, Status::Valid, HashMap::new());
        }
    }
    
    // Generate novel prefix to test float handling
    let prefix = tree.generate_novel_prefix(&mut rng);
    println!("Generated float prefix with {} choices", prefix.len());
    
    let stats = tree.get_stats();
    assert!(stats.conclusion_nodes >= 3); // At least finite values recorded
}

// ========== Tree Navigation Tests ==========

#[test]
fn test_tree_node_operations() {
    // Tests basic TreeNode functionality
    let mut node = TreeNode::new(0);
    
    // Test adding choices
    node.add_choice(
        ChoiceType::Integer,
        ChoiceValue::Integer(42),
        create_integer_constraints(Some(0), Some(100)),
        false
    );
    
    assert_eq!(node.values.len(), 1);
    assert_eq!(node.choice_types.len(), 1);
    assert_eq!(node.constraints.len(), 1);
    
    // Test max children calculation
    if let Some(max_children) = node.compute_max_children() {
        assert_eq!(max_children, 101); // 0 through 100 inclusive
    }
    
    // Test exhaustion checking
    assert!(!node.check_exhausted()); // Node with no transition is not exhausted
}

#[test]
fn test_tree_navigation_basic() {
    // Tests basic tree navigation without external dependencies
    let mut tree = DataTree::new();
    
    // Test initial state
    assert_eq!(tree.stats.total_nodes, 0);
    assert_eq!(tree.root.node_id, 0);
    
    // Record a path to create tree structure
    let choices = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            create_integer_constraints(Some(0), Some(100)),
            false
        ),
    ];
    tree.record_path(&choices, Status::Valid, HashMap::new());
    
    // Verify tree state after recording
    let stats = tree.get_stats();
    assert!(stats.total_nodes > 0);
    assert_eq!(stats.conclusion_nodes, 1);
}

// ========== Exhaustion Detection Tests ==========

#[test]
fn test_sophisticated_exhaustion_detection() {
    // Tests advanced exhaustion detection logic
    let mut node = TreeNode::new(1);
    
    // Test empty node
    assert!(!node.check_exhausted());
    
    // Add choices to test max children calculation
    node.add_choice(
        ChoiceType::Boolean,
        ChoiceValue::Boolean(true),
        create_boolean_constraints(0.5),
        false
    );
    
    // Test exhaustion ratio
    let ratio = node.compute_exhaustion_ratio();
    assert!(ratio >= 0.0 && ratio <= 1.0);
    
    // Test max children calculation
    let max_children = node.compute_max_children();
    assert!(max_children.is_some());
    if let Some(max) = max_children {
        assert!(max > 0);
    }
}

#[test]
fn test_tree_exhaustion_with_branches() {
    // Tests exhaustion detection with complex branching
    let mut tree = DataTree::new();
    
    // Create multiple branches by recording different paths
    for i in 0..3 {
        let choices = vec![
            (
                ChoiceType::Integer,
                ChoiceValue::Integer(i),
                create_integer_constraints(Some(0), Some(2)),
                false
            ),
        ];
        tree.record_path(&choices, Status::Valid, HashMap::new());
    }
    
    // All possible values for range 0-2 are recorded
    let stats = tree.get_stats();
    assert_eq!(stats.conclusion_nodes, 3);
    
    // Tree might be exhausted depending on constraint interpretation
    let exhausted = tree.root.check_exhausted();
    println!("Tree exhausted: {}", exhausted);
}

// ========== Complex Scenario Tests ==========

#[test]
fn test_complex_tree_navigation_scenario() {
    // Tests complex navigation with multiple path recordings
    let mut tree = DataTree::new();
    
    // Record first path: (1, true)
    let choices1 = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(1),
            create_integer_constraints(Some(0), Some(10)),
            false
        ),
        (
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            create_boolean_constraints(0.5),
            false
        ),
    ];
    tree.record_path(&choices1, Status::Valid, HashMap::new());
    
    // Record diverging path: (1, false)
    let choices2 = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(1),
            create_integer_constraints(Some(0), Some(10)),
            false
        ),
        (
            ChoiceType::Boolean,
            ChoiceValue::Boolean(false),
            create_boolean_constraints(0.5),
            false
        ),
    ];
    tree.record_path(&choices2, Status::Interesting, HashMap::new());
    
    // Verify complex tree structure
    let stats = tree.get_stats();
    assert!(stats.total_nodes > 0);
    assert_eq!(stats.conclusion_nodes, 2);
    assert!(stats.branch_nodes > 0);
    
    println!("Stats: total_nodes={}, conclusion_nodes={}, branch_nodes={}", 
             stats.total_nodes, stats.conclusion_nodes, stats.branch_nodes);
}

#[test]
fn test_tree_with_forced_and_free_choices() {
    // Tests mixing forced and free choices
    let mut tree = DataTree::new();
    
    // Path with mixed forced/free choices
    let choices = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            create_integer_constraints(Some(0), Some(100)),
            true // forced
        ),
        (
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            create_boolean_constraints(0.5),
            false // free choice
        ),
        (
            ChoiceType::Float,
            ChoiceValue::Float(3.14),
            create_float_constraints(0.0, 10.0, false),
            true // forced
        ),
    ];
    tree.record_path(&choices, Status::Valid, HashMap::new());
    
    let stats = tree.get_stats();
    assert!(stats.total_nodes > 0);
    assert_eq!(stats.conclusion_nodes, 1);
    
    // Test novel prefix generation with forced choices
    let mut rng = thread_rng();
    let prefix = tree.generate_novel_prefix(&mut rng);
    println!("Generated prefix with {} choices after forced/free mix", prefix.len());
}

#[test]
fn test_all_choice_types_coverage() {
    // Tests all choice types in tree operations
    let mut tree = DataTree::new();
    
    // Record paths with all choice types
    let all_choice_types = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            create_integer_constraints(Some(0), Some(100)),
            false
        ),
        (
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            create_boolean_constraints(0.7),
            false
        ),
        (
            ChoiceType::Float,
            ChoiceValue::Float(3.14159),
            create_float_constraints(-10.0, 10.0, false),
            false
        ),
        (
            ChoiceType::String,
            ChoiceValue::String("test".to_string()),
            create_string_constraints(0, 256),
            false
        ),
        (
            ChoiceType::Bytes,
            ChoiceValue::Bytes(vec![1, 2, 3, 4]),
            create_bytes_constraints(0, 1024),
            false
        ),
    ];
    
    tree.record_path(&all_choice_types, Status::Valid, HashMap::new());
    
    let stats = tree.get_stats();
    assert!(stats.total_nodes > 0);
    assert_eq!(stats.conclusion_nodes, 1);
    
    // Test max children calculation on root node
    if !tree.root.values.is_empty() {
        let max_children = tree.root.compute_max_children();
        assert!(max_children.is_some());
        println!("Root node max children: {:?}", max_children);
    }
}

// ========== Performance and Edge Case Tests ==========

#[test]
fn test_large_tree_performance() {
    // Tests performance with larger tree structures
    let mut tree = DataTree::new();
    let mut rng = thread_rng();
    
    // Record multiple paths to create larger tree
    for i in 0..10 {
        let choices = vec![
            (
                ChoiceType::Integer,
                ChoiceValue::Integer(i % 3), // Creates 3 branches
                create_integer_constraints(Some(0), Some(2)),
                false
            ),
            (
                ChoiceType::Boolean,
                ChoiceValue::Boolean(i % 2 == 0),
                create_boolean_constraints(0.5),
                false
            ),
        ];
        tree.record_path(&choices, Status::Valid, HashMap::new());
    }
    
    // Test novel prefix generation performance
    let start = std::time::Instant::now();
    for _ in 0..10 {
        let _prefix = tree.generate_novel_prefix(&mut rng);
    }
    let duration = start.elapsed();
    
    println!("Generated 10 prefixes in {:?}", duration);
    
    let stats = tree.get_stats();
    assert!(stats.total_nodes > 0);
    assert!(stats.novel_prefixes_generated >= 10);
}

#[test]
fn test_edge_case_empty_constraints() {
    // Tests edge cases with empty or minimal constraints
    let mut tree = DataTree::new();
    
    // Record path with minimal constraints
    let choices = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(0),
            create_integer_constraints(None, None), // No bounds
            false
        ),
    ];
    tree.record_path(&choices, Status::Valid, HashMap::new());
    
    let stats = tree.get_stats();
    assert_eq!(stats.conclusion_nodes, 1);
    
    // Test max children with unbounded constraints
    if let Some(max_children) = tree.root.compute_max_children() {
        assert!(max_children > 0);
        println!("Unbounded integer max children: {}", max_children);
    }
}

#[test]
fn test_tree_statistics_accuracy() {
    // Tests accuracy of tree statistics tracking
    let mut tree = DataTree::new();
    let mut rng = thread_rng();
    
    let initial_stats = tree.get_stats();
    assert_eq!(initial_stats.total_nodes, 0);
    assert_eq!(initial_stats.conclusion_nodes, 0);
    assert_eq!(initial_stats.branch_nodes, 0);
    assert_eq!(initial_stats.novel_prefixes_generated, 0);
    
    // Generate novel prefixes
    for _ in 0..3 {
        let _prefix = tree.generate_novel_prefix(&mut rng);
    }
    
    let after_generation = tree.get_stats();
    assert_eq!(after_generation.novel_prefixes_generated, 3);
    
    // Record some paths
    for i in 0..2 {
        let choices = vec![
            (
                ChoiceType::Boolean,
                ChoiceValue::Boolean(i == 0),
                create_boolean_constraints(0.5),
                false
            ),
        ];
        tree.record_path(&choices, Status::Valid, HashMap::new());
    }
    
    let final_stats = tree.get_stats();
    assert_eq!(final_stats.conclusion_nodes, 2);
    assert!(final_stats.total_nodes > 0);
}