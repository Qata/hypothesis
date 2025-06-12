#!/usr/bin/env rust-script
//! Standalone test to verify DataTree node navigation system capability
//! This test validates the complete functionality without requiring PyO3 dependencies

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;

// Import the crate modules
use conjecture_rust::datatree::{DataTree, TreeNode, Branch, Conclusion, Transition, TreeStats};
use conjecture_rust::choice::{ChoiceType, ChoiceValue, Constraints, IntegerConstraints, FloatConstraints, BooleanConstraints};
use conjecture_rust::data::Status;

fn main() {
    println!("DATATREE NAVIGATION VERIFICATION: Starting comprehensive capability tests...");
    
    // Test 1: Basic node navigation capability
    test_basic_node_navigation();
    
    // Test 2: Tree recording and structure building
    test_tree_recording_capability();
    
    // Test 3: Node splitting and radix tree compression
    test_node_splitting_capability();
    
    // Test 4: Navigation through complex tree structures
    test_complex_navigation_capability();
    
    // Test 5: Weighted selection and exploration
    test_weighted_selection_capability();
    
    println!("DATATREE NAVIGATION VERIFICATION: All tests passed! ✅");
}

fn test_basic_node_navigation() {
    println!("TEST 1: Basic node navigation capability");
    
    let mut tree = DataTree::new();
    let mut rng = StdRng::seed_from_u64(12345);
    
    // Test initial state
    let initial_stats = tree.get_stats();
    assert_eq!(initial_stats.total_nodes, 0);
    assert_eq!(initial_stats.novel_prefixes_generated, 0);
    
    // Test novel prefix generation from empty tree
    let prefix1 = tree.generate_novel_prefix(&mut rng);
    let stats_after_generation = tree.get_stats();
    assert_eq!(stats_after_generation.novel_prefixes_generated, 1);
    
    println!("  ✓ Basic navigation works");
}

fn test_tree_recording_capability() {
    println!("TEST 2: Tree recording and structure building");
    
    let mut tree = DataTree::new();
    
    // Record a simple path
    let simple_path = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Box::new(Constraints::Integer(IntegerConstraints::default())),
            false
        ),
    ];
    
    tree.record_path(&simple_path, Status::Valid, HashMap::new());
    let stats = tree.get_stats();
    assert!(stats.total_nodes > 0);
    assert_eq!(stats.conclusion_nodes, 1);
    
    // Record a branching path
    let branching_path1 = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(100),
            Box::new(Constraints::Integer(IntegerConstraints::default())),
            false
        ),
        (
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Box::new(Constraints::Boolean(BooleanConstraints { p: 0.5 })),
            false
        ),
    ];
    
    tree.record_path(&branching_path1, Status::Valid, HashMap::new());
    
    let branching_path2 = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(100),
            Box::new(Constraints::Integer(IntegerConstraints::default())),
            false
        ),
        (
            ChoiceType::Boolean,
            ChoiceValue::Boolean(false),
            Box::new(Constraints::Boolean(BooleanConstraints { p: 0.5 })),
            false
        ),
    ];
    
    tree.record_path(&branching_path2, Status::Valid, HashMap::new());
    
    let final_stats = tree.get_stats();
    assert!(final_stats.branch_nodes > 0);
    assert_eq!(final_stats.conclusion_nodes, 3);
    
    println!("  ✓ Tree recording works");
}

fn test_node_splitting_capability() {
    println!("TEST 3: Node splitting and radix tree compression");
    
    let mut node = TreeNode::new(1);
    let mut next_id = 2;
    
    // Add multiple choices to create a compressible sequence
    node.add_choice(
        ChoiceType::Integer,
        ChoiceValue::Integer(10),
        DataTree::create_integer_constraints(Some(0), Some(100)),
        false
    );
    node.add_choice(
        ChoiceType::Integer,
        ChoiceValue::Integer(20),
        DataTree::create_integer_constraints(Some(0), Some(100)),
        false
    );
    node.add_choice(
        ChoiceType::Integer,
        ChoiceValue::Integer(30),
        DataTree::create_integer_constraints(Some(0), Some(100)),
        false
    );
    
    assert_eq!(node.values.len(), 3);
    
    // Split at index 1
    let suffix = node.split_at(1, &mut next_id);
    
    // Verify split worked correctly
    assert_eq!(node.values.len(), 1);
    assert_eq!(suffix.values.len(), 2);
    
    if let ChoiceValue::Integer(val) = &node.values[0] {
        assert_eq!(*val, 10);
    } else {
        panic!("Expected integer value");
    }
    
    if let ChoiceValue::Integer(val) = &suffix.values[0] {
        assert_eq!(*val, 20);
    } else {
        panic!("Expected integer value");
    }
    
    println!("  ✓ Node splitting works");
}

fn test_complex_navigation_capability() {
    println!("TEST 4: Complex navigation through tree structures");
    
    let mut tree = DataTree::new();
    let mut rng = StdRng::seed_from_u64(54321);
    
    // Create a deep path with multiple choice points
    let deep_path = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(1),
            DataTree::create_integer_constraints(Some(0), Some(10)),
            false
        ),
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(2),
            DataTree::create_integer_constraints(Some(0), Some(10)),
            false
        ),
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(3),
            DataTree::create_integer_constraints(Some(0), Some(10)),
            false
        ),
    ];
    
    tree.record_path(&deep_path, Status::Valid, HashMap::new());
    
    // Create alternative branches
    let alt_path = vec![
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(1),
            DataTree::create_integer_constraints(Some(0), Some(10)),
            false
        ),
        (
            ChoiceType::Integer,
            ChoiceValue::Integer(99), // Different choice
            DataTree::create_integer_constraints(Some(0), Some(10)),
            false
        ),
    ];
    
    tree.record_path(&alt_path, Status::Invalid, HashMap::new());
    
    let stats = tree.get_stats();
    assert!(stats.total_nodes > 3);
    assert_eq!(stats.conclusion_nodes, 2);
    assert!(stats.branch_nodes > 0);
    
    // Test navigation through deep tree
    for _ in 0..5 {
        let _prefix = tree.generate_novel_prefix(&mut rng);
    }
    
    assert!(tree.get_stats().novel_prefixes_generated >= 5);
    
    println!("  ✓ Complex navigation works");
}

fn test_weighted_selection_capability() {
    println!("TEST 5: Weighted selection and exploration");
    
    let tree = DataTree::new();
    let mut rng = StdRng::seed_from_u64(33333);
    
    // Create nodes with different exploration characteristics
    let unexplored_node = Arc::new(TreeNode::new(1));
    let explored_node = Arc::new(TreeNode::new(2));
    
    // Add branch to explored node
    let branch = Branch {
        children: RwLock::new({
            let mut children = HashMap::new();
            children.insert(ChoiceValue::Integer(1), Arc::new(TreeNode::new(3)));
            children.insert(ChoiceValue::Integer(2), Arc::new(TreeNode::new(4)));
            children
        }),
        is_exhausted: RwLock::new(false),
    };
    *explored_node.transition.write().unwrap() = Some(Transition::Branch(branch));
    
    // Test weight calculation
    let unexplored_weight = tree.calculate_exploration_weight(&unexplored_node, 5);
    let explored_weight = tree.calculate_exploration_weight(&explored_node, 5);
    
    // Unexplored node should have higher weight
    assert!(unexplored_weight > explored_weight);
    
    println!("  ✓ Weighted selection works");
}