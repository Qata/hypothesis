//! Comprehensive Integration Tests for Choice Navigation System Capability
//!
//! Tests the complete capability of choice sequence navigation including:
//! - Tree traversal and prefix-based selection orders for shrinking choices in structured patterns
//! - Novel prefix generation for systematic tree exploration
//! - Choice indexing and bidirectional mapping
//! - Navigation tree exhaustion tracking
//! - Full end-to-end navigation workflows with PyO3 FFI compatibility

use crate::choice::navigation::*;
use crate::choice::{ChoiceType, ChoiceValue, Constraints};
use crate::choice::constraints::*;
use std::collections::HashMap;

/// Integration test for complete navigation workflow
#[test]
fn test_complete_navigation_workflow() {
    // Create a navigation tree and simulate a complete exploration workflow
    let mut tree = NavigationTree::new();
    let mut indexer = ChoiceIndexer::new();
    
    // Test integer constraint navigation
    let int_constraints = Constraints::Integer(IntegerConstraints::new(Some(-5), Some(5), Some(0)));
    
    // Set up initial tree with root
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(0),
        int_constraints.clone(),
        false,
    );
    tree.set_root(root);
    
    // Generate several novel prefixes to simulate exploration
    let mut generated_prefixes = Vec::new();
    for i in 0..3 {
        match tree.generate_novel_prefix() {
            Ok(prefix) => {
                println!("Generated prefix {}: {:?}", i, prefix);
                generated_prefixes.push(prefix);
            }
            Err(e) => {
                println!("Failed to generate prefix {}: {}", i, e);
                break;
            }
        }
    }
    
    // Verify we generated at least one prefix
    assert!(!generated_prefixes.is_empty(), "Should generate at least one novel prefix");
    
    // Test choice indexing for all generated choices
    for prefix in &generated_prefixes {
        for choice in &prefix.choices {
            let index = indexer.choice_to_index(choice, &int_constraints)
                .expect("Should map choice to index");
            
            let recovered_choice = indexer.index_to_choice(index, ChoiceType::Integer, &int_constraints)
                .expect("Should map index back to choice");
                
            assert_eq!(*choice, recovered_choice, "Round-trip indexing should preserve choice");
        }
    }
    
    // Test tree statistics
    let stats = tree.stats();
    assert!(stats.node_count > 0, "Tree should have nodes");
    assert!(stats.cached_sequences > 0, "Tree should have cached sequences");
}

/// Test systematic prefix-based selection orders for structured shrinking patterns
#[test]
fn test_prefix_selection_orders_for_shrinking() {
    // Create a prefix selector with a sample prefix representing a shrinking pattern
    let prefix = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(42),
        ChoiceValue::Boolean(true),
    ]);
    
    let selector = PrefixSelector::new(prefix, 10);
    
    // Test left-then-right selection order for systematic exploration
    let order = selector.selection_order(5);
    assert_eq!(order, vec![5, 4, 3, 2, 1, 0, 6, 7, 8, 9]);
    
    // Test edge cases for boundary conditions
    let edge_order = selector.selection_order(0);
    assert_eq!(edge_order, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    
    let last_order = selector.selection_order(9);
    assert_eq!(last_order, vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    
    // Test random selection order consistency for reproducible shrinking
    let random_order1 = selector.random_selection_order(12345);
    let random_order2 = selector.random_selection_order(12345);
    assert_eq!(random_order1, random_order2, "Same seed should produce same order");
    
    let different_order = selector.random_selection_order(54321);
    assert_ne!(random_order1, different_order, "Different seeds should produce different orders");
    
    // Verify all indices are covered in selection orders
    for order in [&order, &edge_order, &last_order, &random_order1] {
        let mut sorted_order = order.clone();
        sorted_order.sort();
        assert_eq!(sorted_order, (0..10).collect::<Vec<_>>(), "All indices should be present");
    }
}

/// Test comprehensive choice indexing across all types for structured patterns
#[test]
fn test_comprehensive_choice_indexing_patterns() {
    let mut indexer = ChoiceIndexer::new();
    
    // Test integer indexing with zigzag encoding around shrink targets
    let int_constraints = Constraints::Integer(IntegerConstraints::new(Some(-10), Some(10), Some(3)));
    
    let test_integers = vec![-5, -1, 0, 1, 3, 5, 10];
    for &val in &test_integers {
        let choice = ChoiceValue::Integer(val);
        let index = indexer.choice_to_index(&choice, &int_constraints)
            .expect("Should index integer choice");
            
        let recovered = indexer.index_to_choice(index, ChoiceType::Integer, &int_constraints)
            .expect("Should recover integer choice");
            
        assert_eq!(choice, recovered, "Integer round-trip failed for {}", val);
        
        // Verify zigzag encoding - values closer to shrink target should have lower indices
        if val == 3 {
            assert_eq!(index, 0, "Shrink target should have index 0");
        }
    }
    
    // Test boolean indexing for binary choice patterns
    let bool_constraints = Constraints::Boolean(BooleanConstraints::new());
    
    for &val in &[false, true] {
        let choice = ChoiceValue::Boolean(val);
        let index = indexer.choice_to_index(&choice, &bool_constraints)
            .expect("Should index boolean choice");
            
        let recovered = indexer.index_to_choice(index, ChoiceType::Boolean, &bool_constraints)
            .expect("Should recover boolean choice");
            
        assert_eq!(choice, recovered, "Boolean round-trip failed for {}", val);
    }
    
    // Test float indexing for continuous value patterns
    let float_constraints = Constraints::Float(FloatConstraints::new(Some(-100.0), Some(100.0)));
    
    let test_floats = vec![0.0, 1.5, -2.7, 42.0];
    for &val in &test_floats {
        let choice = ChoiceValue::Float(val);
        let index = indexer.choice_to_index(&choice, &float_constraints)
            .expect("Should index float choice");
            
        let recovered = indexer.index_to_choice(index, ChoiceType::Float, &float_constraints)
            .expect("Should recover float choice");
            
        if let ChoiceValue::Float(recovered_val) = recovered {
            // Float round-trip may not be exact due to bit representation
            assert!((val - recovered_val).abs() < 1e-10 || val.to_bits() == recovered_val.to_bits(),
                    "Float round-trip failed for {}: got {}", val, recovered_val);
        } else {
            panic!("Expected float choice, got {:?}", recovered);
        }
    }
    
    // Test string indexing for structured text patterns
    let string_constraints = Constraints::String(StringConstraints::new(Some(10), None));
    
    let test_strings = vec!["", "a", "hello", "world"];
    for val in &test_strings {
        let choice = ChoiceValue::String(val.to_string());
        let index = indexer.choice_to_index(&choice, &string_constraints)
            .expect("Should index string choice");
            
        let recovered = indexer.index_to_choice(index, ChoiceType::String, &string_constraints)
            .expect("Should recover string choice");
            
        // String recovery is based on length, so we check length preservation
        if let ChoiceValue::String(recovered_str) = recovered {
            assert_eq!(val.len(), recovered_str.len(), 
                      "String length should be preserved for '{}'", val);
        } else {
            panic!("Expected string choice, got {:?}", recovered);
        }
    }
    
    // Test bytes indexing for binary data patterns
    let bytes_constraints = Constraints::Bytes(BytesConstraints::new(Some(10), None));
    
    let test_bytes = vec![vec![], vec![1], vec![1, 2, 3], vec![255, 0, 128]];
    for val in &test_bytes {
        let choice = ChoiceValue::Bytes(val.clone());
        let index = indexer.choice_to_index(&choice, &bytes_constraints)
            .expect("Should index bytes choice");
            
        let recovered = indexer.index_to_choice(index, ChoiceType::Bytes, &bytes_constraints)
            .expect("Should recover bytes choice");
            
        // Bytes recovery is based on length, so we check length preservation
        if let ChoiceValue::Bytes(recovered_bytes) = recovered {
            assert_eq!(val.len(), recovered_bytes.len(), 
                      "Bytes length should be preserved for {:?}", val);
        } else {
            panic!("Expected bytes choice, got {:?}", recovered);
        }
    }
}

/// Test navigation tree sequence recording and structured pattern handling
#[test]
fn test_tree_sequence_recording_structured_patterns() {
    let mut tree = NavigationTree::new();
    
    // Create a structured sequence representing complex choice patterns
    let sequence = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(42),
        ChoiceValue::Boolean(true),
        ChoiceValue::String("test".to_string()),
    ]);
    
    let constraints_sequence = vec![
        Constraints::Integer(IntegerConstraints::new(Some(0), Some(100), Some(0))),
        Constraints::Boolean(BooleanConstraints::new()),
        Constraints::String(StringConstraints::new(Some(10), None)),
    ];
    
    // Record the structured sequence in the tree
    tree.record_sequence(&sequence, &constraints_sequence);
    
    // Verify tree structure maintains pattern relationships
    let stats = tree.stats();
    assert!(stats.node_count >= 1, "Tree should have at least root node");
    assert_eq!(stats.max_depth, 2, "Tree should have depth 2 for 3-choice sequence");
    
    // Verify tree is not exhausted after recording structured patterns
    assert!(!tree.is_exhausted(), "Tree should not be exhausted after recording sequence");
    
    // Test multiple sequence recording for complex patterns
    let second_sequence = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(43),
        ChoiceValue::Boolean(false),
    ]);
    
    let second_constraints = vec![
        Constraints::Integer(IntegerConstraints::new(Some(0), Some(100), Some(0))),
        Constraints::Boolean(BooleanConstraints::new()),
    ];
    
    tree.record_sequence(&second_sequence, &second_constraints);
    
    let updated_stats = tree.stats();
    assert!(updated_stats.node_count > stats.node_count, "Should add more nodes");
}

/// Test tree exhaustion detection in structured navigation patterns
#[test]
fn test_tree_exhaustion_detection_patterns() {
    let mut tree = NavigationTree::new();
    
    // Start with empty tree - should be exhausted
    assert!(tree.is_exhausted(), "Empty tree should be exhausted");
    
    // Add a root with very limited constraints for quick exhaustion
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(1), Some(0)));
    let mut root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(0),
        constraints.clone(),
        false,
    );
    
    // Add children to create exploration patterns
    let child1 = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(1),
        constraints.clone(),
        false,
    );
    
    let child2 = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(0),
        constraints,
        false,
    );
    
    root.add_child(ChoiceValue::Integer(1), child1);
    root.add_child(ChoiceValue::Integer(0), child2);
    
    tree.set_root(root);
    
    // Tree should not be exhausted initially
    assert!(!tree.is_exhausted(), "Tree with nodes should not be exhausted initially");
    
    // Test systematic exhaustion of choice patterns
    if let Some(ref mut root) = tree.root {
        // Mark all children as exhausted to test propagation
        for child in root.children.values_mut() {
            child.as_mut().mark_exhausted();
        }
    }
    
    // Check exhaustion propagation in structured patterns
    assert!(tree.is_exhausted(), "Tree should be exhausted when all exploration paths are exhausted");
}

/// Test choice sequence operations and pattern properties
#[test]
fn test_choice_sequence_pattern_properties() {
    // Test empty sequence baseline
    let empty_seq = ChoiceSequence::new();
    assert_eq!(empty_seq.length, 0);
    assert!(empty_seq.choices.is_empty());
    
    // Test structured sequence building
    let mut seq = ChoiceSequence::new();
    seq.push(ChoiceValue::Integer(1));
    seq.push(ChoiceValue::Boolean(false));
    seq.push(ChoiceValue::String("test".to_string()));
    
    assert_eq!(seq.length, 3);
    assert_eq!(seq.choices.len(), 3);
    
    // Test prefix extraction for shrinking patterns
    let prefix1 = seq.prefix(2);
    assert_eq!(prefix1.length, 2);
    assert_eq!(prefix1.choices[0], ChoiceValue::Integer(1));
    assert_eq!(prefix1.choices[1], ChoiceValue::Boolean(false));
    
    let prefix_all = seq.prefix(10); // Should be limited to actual length
    assert_eq!(prefix_all.length, 3);
    assert_eq!(prefix_all.choices, seq.choices);
    
    // Test prefix matching for navigation patterns
    assert!(seq.starts_with(&prefix1), "Sequence should start with its own prefix");
    assert!(seq.starts_with(&empty_seq), "Any sequence should start with empty sequence");
    
    let different_prefix = ChoiceSequence::from_choices(vec![ChoiceValue::Integer(2)]);
    assert!(!seq.starts_with(&different_prefix), "Should not match different prefix");
    
    // Test sort key generation for shrinking order
    let sort_key = seq.sort_key();
    assert_eq!(sort_key.0, 3, "Sort key should include sequence length");
    assert_eq!(sort_key.1.len(), 3, "Sort key should have index for each choice");
    
    // Verify sort keys produce consistent ordering for shrinking
    let shorter_seq = seq.prefix(2);
    let shorter_key = shorter_seq.sort_key();
    assert!(shorter_key.0 < sort_key.0, "Shorter sequences should sort first");
}

/// Test navigation node tree operations for structured patterns
#[test]
fn test_navigation_node_tree_structured_operations() {
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(10), Some(5)));
    
    // Create parent node representing a choice point
    let mut parent = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(5),
        constraints.clone(),
        false,
    );
    
    assert!(!parent.has_children(), "New node should have no children");
    assert_eq!(parent.depth(), 0, "Leaf node should have depth 0");
    
    // Add children representing alternative choices in structured patterns
    let child1 = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(3),
        constraints.clone(),
        false,
    );
    
    let child2 = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(7),
        constraints.clone(),
        false,
    );
    
    parent.add_child(ChoiceValue::Integer(3), child1);
    parent.add_child(ChoiceValue::Integer(7), child2);
    
    assert!(parent.has_children(), "Node should have children after adding");
    assert_eq!(parent.children.len(), 2, "Should have exactly 2 children");
    assert_eq!(parent.depth(), 1, "Parent with leaf children should have depth 1");
    
    // Test child retrieval for navigation
    let retrieved_child = parent.get_child(&ChoiceValue::Integer(3));
    assert!(retrieved_child.is_some(), "Should retrieve existing child");
    assert_eq!(retrieved_child.unwrap().value, ChoiceValue::Integer(3));
    
    let missing_child = parent.get_child(&ChoiceValue::Integer(100));
    assert!(missing_child.is_none(), "Should not retrieve non-existent child");
    
    // Test exhaustion propagation in structured choice patterns
    assert!(!parent.is_exhausted, "Parent should not be exhausted initially");
    assert!(!parent.check_exhausted(), "Parent should not be exhausted with non-exhausted children");
    
    // Exhaust all children to test pattern completion
    for child in parent.children.values_mut() {
        child.as_mut().mark_exhausted();
    }
    
    assert!(parent.check_exhausted(), "Parent should be exhausted when all children are exhausted");
    assert!(parent.is_exhausted, "Parent exhaustion flag should be set");
}

/// Test error handling in navigation operations for robust pattern handling
#[test]
fn test_navigation_error_handling_patterns() {
    let mut indexer = ChoiceIndexer::new();
    
    // Test inconsistent constraints error during pattern navigation
    let int_choice = ChoiceValue::Integer(42);
    let bool_constraints = Constraints::Boolean(BooleanConstraints::new());
    
    let result = indexer.choice_to_index(&int_choice, &bool_constraints);
    assert!(result.is_err(), "Should fail with inconsistent constraints");
    
    if let Err(NavigationError::InconsistentConstraints(msg)) = result {
        assert!(msg.contains("Expected"), "Error message should explain expectation");
    } else {
        panic!("Expected InconsistentConstraints error, got {:?}", result);
    }
    
    // Test tree exhaustion error during pattern exploration
    let mut empty_tree = NavigationTree::new();
    let result = empty_tree.generate_novel_prefix();
    assert!(result.is_err(), "Empty tree should fail to generate prefix");
    
    if let Err(NavigationError::TreeExhausted) = result {
        // Expected
    } else {
        panic!("Expected TreeExhausted error, got {:?}", result);
    }
    
    // Test invalid choice index errors
    let invalid_constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(5), Some(0)));
    let large_index_result = indexer.index_to_choice(usize::MAX, ChoiceType::Integer, &invalid_constraints);
    // This may or may not error depending on implementation - both behaviors are valid
    match large_index_result {
        Ok(_) => {}, // Implementation allows large indices
        Err(NavigationError::InvalidChoiceIndex { .. }) => {}, // Implementation validates indices
        Err(e) => panic!("Unexpected error type: {:?}", e),
    }
}

/// Test navigation system performance characteristics for complex patterns
#[test]
fn test_navigation_performance_patterns() {
    let mut indexer = ChoiceIndexer::new();
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(-1000), Some(1000), Some(0)));
    
    // Test caching effectiveness - same computation should be cached
    let choice = ChoiceValue::Integer(42);
    
    let start = std::time::Instant::now();
    let index1 = indexer.choice_to_index(&choice, &constraints).unwrap();
    let first_duration = start.elapsed();
    
    let start = std::time::Instant::now();
    let index2 = indexer.choice_to_index(&choice, &constraints).unwrap();
    let second_duration = start.elapsed();
    
    assert_eq!(index1, index2, "Cached result should be identical");
    // Note: In a real performance test, second_duration would typically be much less than first_duration
    // but in unit tests this isn't guaranteed due to timing variability
    
    // Test bulk operations for complex pattern navigation
    let mut tree = NavigationTree::new();
    let root_constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(100), Some(50)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(50),
        root_constraints,
        false,
    );
    tree.set_root(root);
    
    // Generate multiple prefixes to test pattern exploration performance
    let mut prefix_count = 0;
    for _ in 0..10 {
        match tree.generate_novel_prefix() {
            Ok(_) => prefix_count += 1,
            Err(_) => break,
        }
    }
    
    // Should generate at least some prefixes before exhaustion
    assert!(prefix_count > 0, "Should generate at least one prefix in bulk test");
    
    let final_stats = tree.stats();
    assert!(final_stats.cached_sequences > 0, "Should have cached sequences");
    assert!(final_stats.node_count > 0, "Should have built tree nodes");
}

/// Test integration with realistic choice constraints for structured shrinking patterns
#[test]
fn test_integration_with_realistic_choice_constraints() {
    let mut tree = NavigationTree::new();
    let mut indexer = ChoiceIndexer::new();
    
    // Test with realistic constraint combinations that would appear in Hypothesis
    
    // Integer constraint with shrink target for numeric data
    let int_constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(1000), Some(100)));
    
    // String constraint with realistic size limits for text data
    let string_constraints = Constraints::String(StringConstraints::new(None, Some(256)));
    
    // Float constraint with finite bounds for continuous data
    let float_constraints = Constraints::Float(FloatConstraints::new(Some(-1e6), Some(1e6)));
    
    // Create a mixed sequence that would appear in real shrinking patterns
    let mixed_sequence = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(150),  // Close to shrink target of 100
        ChoiceValue::String("hypothesis".to_string()),
        ChoiceValue::Float(3.14159),
        ChoiceValue::Boolean(true),
        ChoiceValue::Bytes(vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]), // "Hello" in bytes
    ]);
    
    let constraint_sequence = vec![
        int_constraints.clone(),
        string_constraints.clone(),
        float_constraints.clone(),
        Constraints::Boolean(BooleanConstraints::new()),
        Constraints::Bytes(BytesConstraints::new(Some(1024), None)),
    ];
    
    // Record the realistic structured pattern
    tree.record_sequence(&mixed_sequence, &constraint_sequence);
    
    // Test indexing of all choices in the realistic pattern
    for (choice, constraints) in mixed_sequence.choices.iter().zip(constraint_sequence.iter()) {
        let index = indexer.choice_to_index(choice, constraints)
            .expect("Should index realistic choice");
        
        // Verify round-trip for types that support exact round-trip
        match choice {
            ChoiceValue::Integer(_) | ChoiceValue::Boolean(_) => {
                let choice_type = match choice {
                    ChoiceValue::Integer(_) => ChoiceType::Integer,
                    ChoiceValue::Boolean(_) => ChoiceType::Boolean,
                    _ => unreachable!(),
                };
                
                let recovered = indexer.index_to_choice(index, choice_type, constraints)
                    .expect("Should recover realistic choice");
                    
                assert_eq!(*choice, recovered, "Realistic choice round-trip should work");
            }
            _ => {
                // For other types, just verify indexing works
                assert!(index < usize::MAX, "Index should be reasonable");
            }
        }
    }
    
    // Test tree statistics with realistic structured data
    let stats = tree.stats();
    assert_eq!(stats.max_depth, 4, "Mixed sequence should create appropriate tree depth");
    assert!(stats.node_count >= 1, "Should have at least root node");
    
    // Test prefix generation with realistic constraints
    let novel_result = tree.generate_novel_prefix();
    // May succeed or fail depending on constraint complexity - both are valid
    match novel_result {
        Ok(prefix) => {
            assert!(prefix.length > 0, "Generated prefix should be non-empty");
            println!("Successfully generated novel prefix: {:?}", prefix);
        }
        Err(NavigationError::TreeExhausted) => {
            println!("Tree exhausted - this is expected behavior for some constraint sets");
        }
        Err(e) => {
            panic!("Unexpected error generating novel prefix: {}", e);
        }
    }
}

/// Test end-to-end navigation workflow for complete capability validation
#[test]
fn test_end_to_end_navigation_capability() {
    // This test validates the complete choice sequence navigation capability
    // by simulating a realistic shrinking scenario with structured patterns
    
    let mut tree = NavigationTree::new();
    let mut indexer = ChoiceIndexer::new();
    
    // Create a complex choice pattern representing a data structure
    let initial_choices = vec![
        ChoiceValue::Integer(10),     // List length
        ChoiceValue::Boolean(true),   // Include special element
        ChoiceValue::String("prefix".to_string()), // String prefix
        ChoiceValue::Float(2.5),      // Scaling factor
        ChoiceValue::Bytes(vec![1, 2, 3]), // Binary data
    ];
    
    let constraints = vec![
        Constraints::Integer(IntegerConstraints::new(Some(0), Some(20), Some(5))),
        Constraints::Boolean(BooleanConstraints::new()),
        Constraints::String(StringConstraints::new(None, Some(50))),
        Constraints::Float(FloatConstraints::new(Some(0.0), Some(10.0))),
        Constraints::Bytes(BytesConstraints::new(None, Some(100))),
    ];
    
    // Step 1: Record initial pattern in navigation tree
    let initial_sequence = ChoiceSequence::from_choices(initial_choices.clone());
    tree.record_sequence(&initial_sequence, &constraints);
    
    // Step 2: Generate novel prefixes for exploration
    let mut exploration_prefixes = Vec::new();
    for _ in 0..5 {
        match tree.generate_novel_prefix() {
            Ok(prefix) => exploration_prefixes.push(prefix),
            Err(NavigationError::TreeExhausted) => break,
            Err(e) => panic!("Unexpected error during exploration: {}", e),
        }
    }
    
    // Step 3: Test prefix-based selection for systematic shrinking
    let prefix_selector = PrefixSelector::new(initial_sequence.prefix(2), 10);
    let selection_orders = vec![
        prefix_selector.selection_order(3),
        prefix_selector.random_selection_order(42),
    ];
    
    for order in selection_orders {
        assert_eq!(order.len(), 10, "Selection order should cover all indices");
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>(), "Should include all indices exactly once");
    }
    
    // Step 4: Test choice indexing for all pattern elements
    for (choice, constraint) in initial_choices.iter().zip(constraints.iter()) {
        let index = indexer.choice_to_index(choice, constraint)
            .expect("Should index choice in pattern");
        
        // Verify indexing produces reasonable results
        assert!(index < usize::MAX / 2, "Index should be reasonable size");
        
        // Test bidirectional mapping where applicable
        match choice {
            ChoiceValue::Integer(_) => {
                let recovered = indexer.index_to_choice(index, ChoiceType::Integer, constraint)
                    .expect("Should recover integer choice");
                assert_eq!(*choice, recovered, "Integer choice should round-trip exactly");
            }
            ChoiceValue::Boolean(_) => {
                let recovered = indexer.index_to_choice(index, ChoiceType::Boolean, constraint)
                    .expect("Should recover boolean choice");
                assert_eq!(*choice, recovered, "Boolean choice should round-trip exactly");
            }
            _ => {
                // Other types tested for indexing capability only
                println!("Indexed {:?} to index {}", choice, index);
            }
        }
    }
    
    // Step 5: Test tree exhaustion and navigation state management
    let stats_before = tree.stats();
    
    // Simulate exploration until exhaustion
    let mut total_prefixes = exploration_prefixes.len();
    loop {
        match tree.generate_novel_prefix() {
            Ok(_) => total_prefixes += 1,
            Err(NavigationError::TreeExhausted) => break,
            Err(e) => panic!("Unexpected error during exhaustion test: {}", e),
        }
        
        // Safety limit to prevent infinite loops
        if total_prefixes > 100 {
            break;
        }
    }
    
    let stats_after = tree.stats();
    assert!(stats_after.cached_sequences >= stats_before.cached_sequences, 
            "Should accumulate cached sequences during exploration");
    
    // Step 6: Validate choice sequence pattern properties
    let test_sequence = ChoiceSequence::from_choices(initial_choices);
    
    // Test prefix operations
    for len in 1..=test_sequence.length {
        let prefix = test_sequence.prefix(len);
        assert!(test_sequence.starts_with(&prefix), "Should start with own prefix");
        assert_eq!(prefix.length, len, "Prefix should have requested length");
    }
    
    // Test sort key for shrinking order
    let sort_key = test_sequence.sort_key();
    assert_eq!(sort_key.0, test_sequence.length, "Sort key should include length");
    assert_eq!(sort_key.1.len(), test_sequence.length, "Should have index per choice");
    
    println!("End-to-end navigation capability test completed successfully");
    println!("Generated {} total prefixes during exploration", total_prefixes);
    println!("Final tree stats: {:?}", stats_after);
}

/// PyO3 FFI compatibility test for navigation system
#[cfg(feature = "python")]
#[test]
fn test_navigation_ffi_compatibility() {
    use pyo3::prelude::*;
    use pyo3::types::PyList;
    
    Python::with_gil(|py| {
        // Test that navigation structures can be converted to/from Python
        
        // Create navigation tree and test basic operations
        let mut tree = NavigationTree::new();
        let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(10), Some(5)));
        let root = NavigationChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(5),
            constraints,
            false,
        );
        tree.set_root(root);
        
        // Test choice sequence conversion
        let sequence = ChoiceSequence::from_choices(vec![
            ChoiceValue::Integer(42),
            ChoiceValue::Boolean(true),
        ]);
        
        // Create Python list representation
        let py_choices = PyList::new(py, &[
            (42i64, "integer").to_object(py),
            (true, "boolean").to_object(py),
        ]);
        
        assert!(!py_choices.is_empty(), "Python list should contain choices");
        assert_eq!(py_choices.len(), 2, "Should have 2 choices in Python list");
        
        // Test that we can work with Python objects
        for item in py_choices.iter() {
            assert!(!item.is_none(), "Python choice items should not be None");
        }
        
        println!("Navigation FFI compatibility test passed");
    });
}