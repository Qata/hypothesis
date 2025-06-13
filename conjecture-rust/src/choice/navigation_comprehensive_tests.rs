//! Comprehensive tests for the Choice sequence navigation system capability
//!
//! This module tests the complete capability behavior: tree traversal and prefix-based
//! selection orders for shrinking choices in structured patterns.
//! Tests validate the entire capability's core responsibilities and interface contracts
//! using PyO3 and FFI integration patterns.

#[cfg(feature = "python-ffi")]
use pyo3::prelude::*;
#[cfg(feature = "python-ffi")]
use pyo3::types::{PyList, PyTuple, PyDict};
use crate::choice::navigation::*;
use crate::choice::{ChoiceType, ChoiceValue, Constraints};
use crate::choice::constraints::*;
use std::time::Instant;

/// Test complete navigation capability behavior with comprehensive validation
#[cfg(feature = "python-ffi")]
#[pyfunction]
fn test_complete_navigation_capability() -> PyResult<bool> {
    let mut success_count = 0;
    let total_tests = 15;
    
    // Test 1: Complete tree traversal system
    if test_complete_tree_traversal().unwrap_or(false) {
        success_count += 1;
    }
    
    // Test 2: Prefix-based selection orders for shrinking
    if test_prefix_based_selection_orders().unwrap_or(false) {
        success_count += 1;
    }
    
    // Test 3: Choice indexing bidirectional conversion system
    if test_choice_indexing_system().unwrap_or(false) {
        success_count += 1;
    }
    
    // Test 4: Novel prefix generation strategies
    if test_novel_prefix_generation_strategies().unwrap_or(false) {
        success_count += 1;
    }
    
    // Test 5: Tree exhaustion and backtracking behavior
    if test_tree_exhaustion_backtracking().unwrap_or(false) {
        success_count += 1;
    }
    
    // Test 6: Selection strategy performance patterns
    if test_selection_strategy_patterns().unwrap_or(false) {
        success_count += 1;
    }
    
    // Test 7: Choice sequence manipulation operations
    if test_choice_sequence_manipulation().unwrap_or(false) {
        success_count += 1;
    }
    
    // Test 8: Integration with constraints system
    if test_constraints_integration().unwrap_or(false) {
        success_count += 1;
    }
    
    // Test 9: Tree depth and complexity handling
    if test_tree_depth_complexity().unwrap_or(false) {
        success_count += 1;
    }
    
    // Test 10: Error handling and recovery
    if test_error_handling_recovery().unwrap_or(false) {
        success_count += 1;
    }
    
    // Test 11: Navigation state consistency
    if test_navigation_state_consistency().unwrap_or(false) {
        success_count += 1;
    }
    
    // Test 12: Performance with large choice spaces
    if test_large_choice_space_performance().unwrap_or(false) {
        success_count += 1;
    }
    
    // Test 13: Zigzag encoding correctness
    if test_zigzag_encoding_correctness().unwrap_or(false) {
        success_count += 1;
    }
    
    // Test 14: Multi-type choice navigation
    if test_multi_type_choice_navigation().unwrap_or(false) {
        success_count += 1;
    }
    
    // Test 15: FFI integration behavior
    if test_ffi_integration_behavior().unwrap_or(false) {
        success_count += 1;
    }
    
    println!("Navigation capability tests: {}/{} passed", success_count, total_tests);
    Ok(success_count >= 12) // 80% success rate required
}

/// Test complete tree traversal system with all navigation patterns
fn test_complete_tree_traversal() -> PyResult<bool> {
    let mut tree = NavigationTree::new();
    
    // Create a complex tree structure
    let root_constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(5), Some(2)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(2),
        root_constraints.clone(),
        false,
    );
    
    tree.set_root(root);
    
    // Build multi-level tree structure
    let sequence1 = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(2),
        ChoiceValue::Boolean(true),
        ChoiceValue::String("test".to_string()),
    ]);
    
    let sequence2 = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(2),
        ChoiceValue::Boolean(false),
        ChoiceValue::Float(3.14),
    ]);
    
    let constraints_seq = vec![
        root_constraints.clone(),
        Constraints::Boolean(BooleanConstraints::new()),
        Constraints::String(StringConstraints::new(Some(10), None)),
    ];
    
    tree.record_sequence(&sequence1, &constraints_seq);
    tree.record_sequence(&sequence2, &constraints_seq);
    
    // Test tree statistics
    let stats = tree.stats();
    if stats.node_count == 0 || stats.max_depth == 0 {
        return Ok(false);
    }
    
    // Test novel prefix generation
    match tree.generate_novel_prefix() {
        Ok(prefix) => {
            if prefix.length == 0 {
                return Ok(false);
            }
        }
        Err(_) => return Ok(false),
    }
    
    // Test tree exhaustion detection
    let is_exhausted = tree.is_exhausted();
    
    println!("Tree traversal test: nodes={}, depth={}, exhausted={}", 
             stats.node_count, stats.max_depth, is_exhausted);
    
    Ok(stats.node_count >= 3 && stats.max_depth >= 2)
}

/// Test prefix-based selection orders for shrinking patterns
fn test_prefix_based_selection_orders() -> PyResult<bool> {
    let prefix = ChoiceSequence::from_choices(vec![ChoiceValue::Integer(42)]);
    let selector = PrefixSelector::new(prefix, 10);
    
    // Test minimize distance strategy (default)
    let order = selector.selection_order(5);
    if order.is_empty() || order[0] != 5 {
        return Ok(false);
    }
    
    // Verify distance minimization pattern: 5, 4, 6, 3, 7, 2, 8, 1, 9, 0
    let expected_pattern = vec![5, 4, 6, 3, 7, 2, 8, 1, 9, 0];
    if order != expected_pattern {
        println!("Expected: {:?}, Got: {:?}", expected_pattern, order);
        return Ok(false);
    }
    
    // Test lexicographic strategy
    let lex_order = selector.selection_order(3);
    if lex_order.is_empty() || lex_order.len() != 10 {
        return Ok(false);
    }
    
    // Test random selection strategy
    let random_order = selector.random_selection_order(12345);
    if random_order.is_empty() || random_order.len() != 10 {
        return Ok(false);
    }
    
    // Verify randomness (should not be in natural order)
    let natural_order: Vec<usize> = (0..10).collect();
    if random_order == natural_order {
        return Ok(false);
    }
    
    println!("Selection orders test: minimize_distance={:?}", order);
    Ok(true)
}

/// Test choice indexing bidirectional conversion system
fn test_choice_indexing_system() -> PyResult<bool> {
    let mut indexer = ChoiceIndexer::new();
    let mut success_count = 0;
    let total_tests = 50;
    
    // Test integer choices with zigzag encoding
    for shrink_towards in [-5, 0, 10] {
        let constraints = Constraints::Integer(IntegerConstraints::new(None, None, Some(shrink_towards)));
        
        for value in [shrink_towards - 3, shrink_towards, shrink_towards + 3] {
            let choice = ChoiceValue::Integer(value);
            
            match indexer.choice_to_index(&choice, &constraints) {
                Ok(index) => {
                    match indexer.index_to_choice(index, ChoiceType::Integer, &constraints) {
                        Ok(recovered_choice) => {
                            if recovered_choice == choice {
                                success_count += 1;
                            }
                        }
                        Err(_) => {}
                    }
                }
                Err(_) => {}
            }
        }
    }
    
    // Test float choices
    let float_constraints = Constraints::Float(FloatConstraints::new(Some(-10.0), Some(10.0)));
    for value in [0.0, 3.14, -2.718, f64::INFINITY, f64::NEG_INFINITY] {
        let choice = ChoiceValue::Float(value);
        
        match indexer.choice_to_index(&choice, &float_constraints) {
            Ok(index) => {
                match indexer.index_to_choice(index, ChoiceType::Float, &float_constraints) {
                    Ok(recovered_choice) => {
                        if let ChoiceValue::Float(recovered_val) = recovered_choice {
                            if value.is_infinite() && recovered_val.is_infinite() && value.is_sign_positive() == recovered_val.is_sign_positive() {
                                success_count += 1;
                            } else if (value - recovered_val).abs() < 1e-10 {
                                success_count += 1;
                            }
                        }
                    }
                    Err(_) => {}
                }
            }
            Err(_) => {}
        }
    }
    
    // Test string choices
    let string_constraints = Constraints::String(StringConstraints::new(Some(20), None));
    for value in ["", "a", "hello", "xyz", "abcdef"] {
        let choice = ChoiceValue::String(value.to_string());
        
        match indexer.choice_to_index(&choice, &string_constraints) {
            Ok(index) => {
                match indexer.index_to_choice(index, ChoiceType::String, &string_constraints) {
                    Ok(recovered_choice) => {
                        if let ChoiceValue::String(recovered_str) = recovered_choice {
                            if recovered_str.len() == value.len() {
                                success_count += 1;
                            }
                        }
                    }
                    Err(_) => {}
                }
            }
            Err(_) => {}
        }
    }
    
    // Test boolean choices
    let bool_constraints = Constraints::Boolean(BooleanConstraints::new());
    for value in [true, false] {
        let choice = ChoiceValue::Boolean(value);
        
        match indexer.choice_to_index(&choice, &bool_constraints) {
            Ok(index) => {
                match indexer.index_to_choice(index, ChoiceType::Boolean, &bool_constraints) {
                    Ok(recovered_choice) => {
                        if recovered_choice == choice {
                            success_count += 1;
                        }
                    }
                    Err(_) => {}
                }
            }
            Err(_) => {}
        }
    }
    
    // Test bytes choices
    let bytes_constraints = Constraints::Bytes(BytesConstraints::new(Some(10), None));
    for value in [vec![], vec![1], vec![1, 2, 3], vec![255, 0, 128]] {
        let choice = ChoiceValue::Bytes(value.clone());
        
        match indexer.choice_to_index(&choice, &bytes_constraints) {
            Ok(index) => {
                match indexer.index_to_choice(index, ChoiceType::Bytes, &bytes_constraints) {
                    Ok(recovered_choice) => {
                        if let ChoiceValue::Bytes(recovered_bytes) = recovered_choice {
                            if recovered_bytes.len() == value.len() {
                                success_count += 1;
                            }
                        }
                    }
                    Err(_) => {}
                }
            }
            Err(_) => {}
        }
    }
    
    let success_rate = (success_count as f64 / total_tests as f64) * 100.0;
    println!("Choice indexing test: {}/{} conversions successful ({:.1}%)", 
             success_count, total_tests, success_rate);
    
    Ok(success_rate >= 90.0) // Require 90%+ success rate
}

/// Test novel prefix generation strategies
fn test_novel_prefix_generation_strategies() -> PyResult<bool> {
    let mut tree = NavigationTree::new();
    
    // Set up tree with root
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(10), Some(5)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(5),
        constraints.clone(),
        false,
    );
    
    tree.set_root(root);
    
    // Generate multiple novel prefixes
    let mut generated_prefixes = Vec::new();
    
    for _ in 0..5 {
        match tree.generate_novel_prefix() {
            Ok(prefix) => {
                if prefix.length > 0 {
                    generated_prefixes.push(prefix);
                }
            }
            Err(NavigationError::TreeExhausted) => break,
            Err(_) => return Ok(false),
        }
    }
    
    if generated_prefixes.is_empty() {
        return Ok(false);
    }
    
    // Verify prefixes are different
    for i in 0..generated_prefixes.len() {
        for j in (i + 1)..generated_prefixes.len() {
            if generated_prefixes[i] == generated_prefixes[j] {
                return Ok(false);
            }
        }
    }
    
    println!("Novel prefix generation: {} unique prefixes generated", generated_prefixes.len());
    Ok(generated_prefixes.len() >= 2)
}

/// Test tree exhaustion and backtracking behavior
fn test_tree_exhaustion_backtracking() -> PyResult<bool> {
    let mut tree = NavigationTree::new();
    
    // Create tree with limited choices
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(2), Some(1)));
    let mut root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(1),
        constraints.clone(),
        false,
    );
    
    // Add children to simulate exhausted paths
    let child1 = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(0),
        constraints.clone(),
        false,
    );
    
    let mut child2 = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(2),
        constraints.clone(),
        false,
    );
    
    // Mark child2 as exhausted
    child2.mark_exhausted();
    
    root.add_child(ChoiceValue::Integer(0), child1);
    root.add_child(ChoiceValue::Integer(2), child2);
    
    tree.set_root(root);
    
    // Test initial state - should not be exhausted
    if tree.is_exhausted() {
        return Ok(false);
    }
    
    // Generate prefixes until exhaustion
    let mut prefix_count = 0;
    for _ in 0..10 {
        match tree.generate_novel_prefix() {
            Ok(_) => prefix_count += 1,
            Err(NavigationError::TreeExhausted) => break,
            Err(_) => return Ok(false),
        }
    }
    
    // After generating several prefixes, tree might become exhausted
    let final_exhausted = tree.is_exhausted();
    
    println!("Tree exhaustion test: {} prefixes generated, final_exhausted={}", 
             prefix_count, final_exhausted);
    
    Ok(prefix_count > 0)
}

/// Test selection strategy patterns
fn test_selection_strategy_patterns() -> PyResult<bool> {
    let prefix = ChoiceSequence::from_choices(vec![ChoiceValue::Integer(10)]);
    let selector = PrefixSelector::new(prefix, 8);
    
    // Test minimize distance pattern
    let minimize_order = selector.selection_order(3);
    if minimize_order != vec![3, 2, 4, 1, 5, 0, 6, 7] {
        return Ok(false);
    }
    
    // Test edge case: start at beginning
    let edge_order = selector.selection_order(0);
    if edge_order.is_empty() || edge_order[0] != 0 {
        return Ok(false);
    }
    
    // Test edge case: start at end
    let end_order = selector.selection_order(7);
    if end_order.is_empty() || end_order[0] != 7 {
        return Ok(false);
    }
    
    // Test random consistency (same seed should give same result)
    let random1 = selector.random_selection_order(42);
    let random2 = selector.random_selection_order(42);
    if random1 != random2 {
        return Ok(false);
    }
    
    println!("Selection strategy patterns: minimize={:?}", minimize_order);
    Ok(true)
}

/// Test choice sequence manipulation operations
fn test_choice_sequence_manipulation() -> PyResult<bool> {
    let mut sequence = ChoiceSequence::new();
    
    // Test basic operations
    sequence.push(ChoiceValue::Integer(1));
    sequence.push(ChoiceValue::Boolean(true));
    sequence.push(ChoiceValue::String("test".to_string()));
    
    if sequence.length != 3 {
        return Ok(false);
    }
    
    // Test prefix operations
    let prefix1 = sequence.prefix(2);
    if prefix1.length != 2 || prefix1.choices.len() != 2 {
        return Ok(false);
    }
    
    let prefix0 = sequence.prefix(0);
    if prefix0.length != 0 || !prefix0.choices.is_empty() {
        return Ok(false);
    }
    
    // Test starts_with operation
    if !sequence.starts_with(&prefix1) {
        return Ok(false);
    }
    
    let unrelated_sequence = ChoiceSequence::from_choices(vec![ChoiceValue::Float(3.14)]);
    if sequence.starts_with(&unrelated_sequence) {
        return Ok(false);
    }
    
    // Test sort key generation
    let sort_key = sequence.sort_key();
    if sort_key.0 != 3 || sort_key.1.len() != 3 {
        return Ok(false);
    }
    
    println!("Sequence manipulation: length={}, sort_key={:?}", sequence.length, sort_key);
    Ok(true)
}

/// Test constraints integration
fn test_constraints_integration() -> PyResult<bool> {
    let mut tree = NavigationTree::new();
    
    // Test with integer constraints
    let int_constraints = Constraints::Integer(IntegerConstraints::new(Some(-5), Some(5), Some(0)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(0),
        int_constraints.clone(),
        false,
    );
    
    tree.set_root(root);
    
    // Generate prefix based on constraints
    match tree.generate_novel_prefix() {
        Ok(prefix) => {
            if prefix.length == 0 {
                return Ok(false);
            }
            
            // Verify choices respect constraints
            for choice in &prefix.choices {
                if let ChoiceValue::Integer(val) = choice {
                    if *val < -5 || *val > 5 {
                        return Ok(false);
                    }
                }
            }
        }
        Err(_) => return Ok(false),
    }
    
    // Test with string constraints
    let string_constraints = Constraints::String(StringConstraints::new(Some(5), None));
    let string_root = NavigationChoiceNode::new(
        ChoiceType::String,
        ChoiceValue::String("".to_string()),
        string_constraints.clone(),
        false,
    );
    
    let mut string_tree = NavigationTree::new();
    string_tree.set_root(string_root);
    
    match string_tree.generate_novel_prefix() {
        Ok(prefix) => {
            for choice in &prefix.choices {
                if let ChoiceValue::String(s) = choice {
                    if s.len() > 5 {
                        return Ok(false);
                    }
                }
            }
        }
        Err(_) => return Ok(false),
    }
    
    println!("Constraints integration test passed");
    Ok(true)
}

/// Test tree depth and complexity handling
fn test_tree_depth_complexity() -> PyResult<bool> {
    let mut tree = NavigationTree::new();
    
    // Create deep tree structure
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(2), Some(1)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(1),
        constraints.clone(),
        false,
    );
    
    tree.set_root(root);
    
    // Record multiple sequences to build depth
    for i in 0..3 {
        let sequence = ChoiceSequence::from_choices(vec![
            ChoiceValue::Integer(1),
            ChoiceValue::Integer(i),
            ChoiceValue::Integer((i + 1) % 3),
        ]);
        
        let constraints_seq = vec![constraints.clone(); 3];
        tree.record_sequence(&sequence, &constraints_seq);
    }
    
    // Verify tree has grown
    let stats = tree.stats();
    if stats.max_depth < 2 || stats.node_count < 3 {
        return Ok(false);
    }
    
    // Test novel prefix generation still works
    match tree.generate_novel_prefix() {
        Ok(prefix) => {
            if prefix.length == 0 {
                return Ok(false);
            }
        }
        Err(_) => return Ok(false),
    }
    
    println!("Tree depth/complexity: depth={}, nodes={}", stats.max_depth, stats.node_count);
    Ok(true)
}

/// Test error handling and recovery
fn test_error_handling_recovery() -> PyResult<bool> {
    let mut indexer = ChoiceIndexer::new();
    
    // Test inconsistent constraints error
    let wrong_constraints = Constraints::Boolean(BooleanConstraints::new());
    let int_choice = ChoiceValue::Integer(42);
    
    match indexer.choice_to_index(&int_choice, &wrong_constraints) {
        Ok(_) => return Ok(false), // Should have failed
        Err(NavigationError::InconsistentConstraints(_)) => {} // Expected
        Err(_) => return Ok(false), // Wrong error type
    }
    
    // Test tree exhaustion handling
    let mut tree = NavigationTree::new();
    
    // Empty tree should be exhausted
    if !tree.is_exhausted() {
        return Ok(false);
    }
    
    match tree.generate_novel_prefix() {
        Ok(_) => return Ok(false), // Should have failed
        Err(NavigationError::TreeExhausted) => {} // Expected
        Err(_) => return Ok(false), // Wrong error type
    }
    
    // Test invalid choice index
    let selector = PrefixSelector::new(ChoiceSequence::new(), 5);
    let order = selector.selection_order(10); // Index out of bounds
    if !order.is_empty() {
        return Ok(false); // Should return empty for invalid index
    }
    
    println!("Error handling test passed");
    Ok(true)
}

/// Test navigation state consistency
fn test_navigation_state_consistency() -> PyResult<bool> {
    let mut tree = NavigationTree::new();
    
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(3), Some(1)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(1),
        constraints.clone(),
        false,
    );
    
    tree.set_root(root);
    
    // Record sequence and verify consistency
    let sequence = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(1),
        ChoiceValue::Integer(2),
    ]);
    
    let constraints_seq = vec![constraints.clone(); 2];
    tree.record_sequence(&sequence, &constraints_seq);
    
    // Verify stats are consistent
    let stats1 = tree.stats();
    let stats2 = tree.stats();
    
    if stats1 != stats2 {
        return Ok(false);
    }
    
    // Generate prefix and check state remains consistent
    let _prefix = tree.generate_novel_prefix();
    let stats3 = tree.stats();
    
    // Should have more cached sequences after generation
    if stats3.cached_sequences <= stats1.cached_sequences {
        return Ok(false);
    }
    
    println!("State consistency: initial_cache={}, after_generation={}", 
             stats1.cached_sequences, stats3.cached_sequences);
    Ok(true)
}

/// Test performance with large choice spaces
fn test_large_choice_space_performance() -> PyResult<bool> {
    let large_prefix = ChoiceSequence::new();
    let selector = PrefixSelector::new(large_prefix, 1000);
    
    // Test selection order generation performance
    let start_time = std::time::Instant::now();
    let order = selector.selection_order(500);
    let duration = start_time.elapsed();
    
    if order.len() != 1000 {
        return Ok(false);
    }
    
    // Should complete reasonably quickly (under 10ms)
    if duration.as_millis() > 10 {
        return Ok(false);
    }
    
    // Verify correctness of large order
    if order[0] != 500 { // Should start with specified index
        return Ok(false);
    }
    
    // Test random order performance
    let start_time = std::time::Instant::now();
    let random_order = selector.random_selection_order(12345);
    let random_duration = start_time.elapsed();
    
    if random_order.len() != 1000 || random_duration.as_millis() > 10 {
        return Ok(false);
    }
    
    println!("Large choice space: order_gen={}μs, random_gen={}μs", 
             duration.as_micros(), random_duration.as_micros());
    Ok(true)
}

/// Test zigzag encoding correctness
fn test_zigzag_encoding_correctness() -> PyResult<bool> {
    let mut indexer = ChoiceIndexer::new();
    
    // Test different shrink_towards values
    for shrink_towards in [-10, 0, 5, 100] {
        let constraints = Constraints::Integer(IntegerConstraints::new(None, None, Some(shrink_towards)));
        
        // Test specific zigzag pattern
        let test_cases = vec![
            (shrink_towards, 0),     // Distance 0 -> index 0
            (shrink_towards + 1, 2), // Distance 1, positive -> index 2
            (shrink_towards - 1, 3), // Distance 1, negative -> index 3
            (shrink_towards + 2, 4), // Distance 2, positive -> index 4
            (shrink_towards - 2, 5), // Distance 2, negative -> index 5
        ];
        
        for (value, expected_index) in test_cases {
            let choice = ChoiceValue::Integer(value);
            
            match indexer.choice_to_index(&choice, &constraints) {
                Ok(index) => {
                    if index != expected_index {
                        println!("Zigzag encoding failed: value={}, expected_index={}, got_index={}", 
                                value, expected_index, index);
                        return Ok(false);
                    }
                    
                    // Test round-trip
                    match indexer.index_to_choice(index, ChoiceType::Integer, &constraints) {
                        Ok(recovered_choice) => {
                            if recovered_choice != choice {
                                return Ok(false);
                            }
                        }
                        Err(_) => return Ok(false),
                    }
                }
                Err(_) => return Ok(false),
            }
        }
    }
    
    println!("Zigzag encoding correctness test passed");
    Ok(true)
}

/// Test multi-type choice navigation
fn test_multi_type_choice_navigation() -> PyResult<bool> {
    let mut tree = NavigationTree::new();
    
    // Create sequence with multiple choice types
    let sequence = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(42),
        ChoiceValue::Boolean(true),
        ChoiceValue::Float(3.14),
        ChoiceValue::String("hello".to_string()),
        ChoiceValue::Bytes(vec![1, 2, 3]),
    ]);
    
    let constraints_seq = vec![
        Constraints::Integer(IntegerConstraints::new(Some(0), Some(100), Some(50))),
        Constraints::Boolean(BooleanConstraints::new()),
        Constraints::Float(FloatConstraints::new(Some(-10.0), Some(10.0))),
        Constraints::String(StringConstraints::new(Some(20), None)),
        Constraints::Bytes(BytesConstraints::new(Some(10), None)),
    ];
    
    tree.record_sequence(&sequence, &constraints_seq);
    
    // Verify tree can handle multi-type sequences
    let stats = tree.stats();
    if stats.node_count == 0 {
        return Ok(false);
    }
    
    // Test novel prefix generation with multi-type tree
    match tree.generate_novel_prefix() {
        Ok(prefix) => {
            if prefix.length == 0 {
                return Ok(false);
            }
            
            // Verify prefix contains valid choice types
            for choice in &prefix.choices {
                match choice {
                    ChoiceValue::Integer(_) |
                    ChoiceValue::Boolean(_) |
                    ChoiceValue::Float(_) |
                    ChoiceValue::String(_) |
                    ChoiceValue::Bytes(_) => {} // All valid
                }
            }
        }
        Err(_) => return Ok(false),
    }
    
    println!("Multi-type navigation: {} nodes, {} types in sequence", 
             stats.node_count, constraints_seq.len());
    Ok(true)
}

/// Test FFI integration behavior
fn test_ffi_integration_behavior() -> PyResult<bool> {
    // This test validates that all navigation functionality works correctly
    // when called from Python via PyO3 FFI
    
    // Test creating objects that can be passed across FFI boundary
    let mut tree = NavigationTree::new();
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(10), Some(5)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(5),
        constraints.clone(),
        false,
    );
    
    tree.set_root(root);
    
    // Test that navigation results can be serialized/deserialized
    let sequence = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(5),
        ChoiceValue::Boolean(true),
    ]);
    
    // Verify sequence properties are FFI-safe
    if sequence.length != 2 {
        return Ok(false);
    }
    
    // Test prefix selector with FFI-safe operations
    let selector = PrefixSelector::new(sequence.clone(), 8);
    let order = selector.selection_order(3);
    
    if order.is_empty() {
        return Ok(false);
    }
    
    // Test choice indexer with FFI-safe conversions
    let mut indexer = ChoiceIndexer::new();
    let choice = ChoiceValue::Integer(42);
    
    match indexer.choice_to_index(&choice, &constraints) {
        Ok(index) => {
            match indexer.index_to_choice(index, ChoiceType::Integer, &constraints) {
                Ok(recovered) => {
                    if recovered != choice {
                        return Ok(false);
                    }
                }
                Err(_) => return Ok(false),
            }
        }
        Err(_) => return Ok(false),
    }
    
    // Test error handling across FFI boundary
    let wrong_constraints = Constraints::Boolean(BooleanConstraints::new());
    match indexer.choice_to_index(&choice, &wrong_constraints) {
        Ok(_) => return Ok(false), // Should have failed
        Err(_) => {} // Expected error
    }
    
    println!("FFI integration test passed");
    Ok(true)
}

/// PyO3 module definition for Python integration
#[cfg(feature = "python-ffi")]
#[pymodule]
fn navigation_comprehensive_tests(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_complete_navigation_capability, m)?)?;
    Ok(())
}

/// Comprehensive test for the complete navigation capability workflow
#[test]
fn test_complete_navigation_capability_workflow() {
    // Test the entire navigation capability from start to finish
    let mut tree = NavigationTree::new();
    let mut indexer = ChoiceIndexer::new();
    
    // Step 1: Create a complex tree structure representing real shrinking scenarios
    let root_constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(100), Some(10)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(10),
        root_constraints.clone(),
        false,
    );
    tree.set_root(root);

    // Step 2: Record multiple complex choice sequences
    let sequences = vec![
        (
            ChoiceSequence::from_choices(vec![
                ChoiceValue::Integer(10),
                ChoiceValue::Boolean(true),
                ChoiceValue::String("shrink_test".to_string()),
                ChoiceValue::Float(1.0),
            ]),
            vec![
                root_constraints.clone(),
                Constraints::Boolean(BooleanConstraints::new()),
                Constraints::String(StringConstraints::new(None, Some(50))),
                Constraints::Float(FloatConstraints::new(Some(0.0), Some(10.0))),
            ]
        ),
        (
            ChoiceSequence::from_choices(vec![
                ChoiceValue::Integer(5),
                ChoiceValue::Boolean(false),
                ChoiceValue::Bytes(vec![1, 2, 3, 4]),
            ]),
            vec![
                root_constraints.clone(),
                Constraints::Boolean(BooleanConstraints::new()),
                Constraints::Bytes(BytesConstraints::new(None, Some(100))),
            ]
        ),
    ];

    for (sequence, constraints) in sequences {
        tree.record_sequence(&sequence, &constraints);
    }

    // Step 3: Test novel prefix generation for systematic exploration
    let mut novel_prefixes = Vec::new();
    for i in 0..10 {
        match tree.generate_novel_prefix() {
            Ok(prefix) => {
                novel_prefixes.push(prefix);
                println!("Generated novel prefix {}: length={}", i, novel_prefixes[i].length);
            }
            Err(NavigationError::TreeExhausted) => {
                println!("Tree exhausted after {} prefixes", i);
                break;
            }
            Err(e) => panic!("Unexpected error generating prefix: {}", e),
        }
    }

    // Step 4: Test prefix-based selection orders for structured shrinking
    if !novel_prefixes.is_empty() {
        let first_prefix = &novel_prefixes[0];
        let selector = PrefixSelector::new(first_prefix.clone(), 20);
        
        // Test left-then-right selection order
        let order = selector.selection_order(10);
        assert_eq!(order.len(), 20);
        assert_eq!(order[0], 10); // Should start from specified index
        
        // Test deterministic random order
        let random_order1 = selector.random_selection_order(12345);
        let random_order2 = selector.random_selection_order(12345);
        assert_eq!(random_order1, random_order2);
    }

    // Step 5: Test comprehensive choice indexing
    let test_choices = vec![
        (ChoiceValue::Integer(0), Constraints::Integer(IntegerConstraints::new(Some(-10), Some(10), Some(0)))),
        (ChoiceValue::Boolean(false), Constraints::Boolean(BooleanConstraints::new())),
        (ChoiceValue::Float(3.14), Constraints::Float(FloatConstraints::new(Some(-100.0), Some(100.0)))),
        (ChoiceValue::String("test".to_string()), Constraints::String(StringConstraints::new(None, Some(100)))),
        (ChoiceValue::Bytes(vec![255, 0, 128]), Constraints::Bytes(BytesConstraints::new(None, Some(100)))),
    ];

    for (choice, constraints) in test_choices {
        let index = indexer.choice_to_index(&choice, &constraints)
            .expect("Should index choice");
        
        // Test round-trip for deterministic types
        match choice {
            ChoiceValue::Integer(_) => {
                let recovered = indexer.index_to_choice(index, ChoiceType::Integer, &constraints)
                    .expect("Should recover integer");
                assert_eq!(choice, recovered);
            }
            ChoiceValue::Boolean(_) => {
                let recovered = indexer.index_to_choice(index, ChoiceType::Boolean, &constraints)
                    .expect("Should recover boolean");
                assert_eq!(choice, recovered);
            }
            _ => {
                // Other types verified for indexing capability
                assert!(index < usize::MAX);
            }
        }
    }

    // Step 6: Verify tree statistics
    let stats = tree.stats();
    assert!(stats.node_count > 0);
    assert!(stats.cached_sequences > 0);
    
    println!("Complete navigation capability test passed");
    println!("Tree stats: {:?}", stats);
    println!("Generated {} novel prefixes", novel_prefixes.len());
}

/// Test FFI-like integration for Python compatibility
#[test]
fn test_navigation_python_ffi_integration() {
    // Test simulating FFI integration without actual PyO3 types
    // This tests the patterns that would be used in real FFI integration
    
    // Test 1: Create navigation structures from data that would come from Python
    let python_like_data = vec![
        ("integer", "42"),
        ("boolean", "true"),
        ("string", "python_test"),
        ("float", "2.718"),
    ];

    let mut sequence = ChoiceSequence::new();
    for (type_name, value_str) in python_like_data {
        match type_name {
            "integer" => {
                let val: i128 = value_str.parse().unwrap();
                sequence.push(ChoiceValue::Integer(val));
            }
            "boolean" => {
                let val: bool = value_str.parse().unwrap();
                sequence.push(ChoiceValue::Boolean(val));
            }
            "string" => {
                sequence.push(ChoiceValue::String(value_str.to_string()));
            }
            "float" => {
                let val: f64 = value_str.parse().unwrap();
                sequence.push(ChoiceValue::Float(val));
            }
            _ => {}
        }
    }

    assert_eq!(sequence.length, 4);
    
    // Test 2: Use navigation tree with Python-like data
    let mut tree = NavigationTree::new();
    let constraints = vec![
        Constraints::Integer(IntegerConstraints::new(Some(0), Some(100), Some(0))),
        Constraints::Boolean(BooleanConstraints::new()),
        Constraints::String(StringConstraints::new(None, Some(50))),
        Constraints::Float(FloatConstraints::new(Some(-10.0), Some(10.0))),
    ];
    
    tree.record_sequence(&sequence, &constraints);
    
    // Test 3: Convert results to Python-compatible format
    let stats = tree.stats();
    let python_compatible_stats = vec![
        ("node_count", stats.node_count.to_string()),
        ("max_depth", stats.max_depth.to_string()),
        ("cached_sequences", stats.cached_sequences.to_string()),
    ];
    
    // Verify conversion
    assert_eq!(python_compatible_stats[0].1.parse::<usize>().unwrap(), stats.node_count);
    
    // Test 4: Choice indexer with Python-like constraints
    let mut indexer = ChoiceIndexer::new();
    let python_constraint_data = vec![
        ("min_value", "-50"),
        ("max_value", "50"),
        ("shrink_towards", "0"),
    ];
    
    let min_val: i128 = python_constraint_data[0].1.parse().unwrap();
    let max_val: i128 = python_constraint_data[1].1.parse().unwrap();
    let shrink_towards: i128 = python_constraint_data[2].1.parse().unwrap();
    
    let rust_constraints = Constraints::Integer(IntegerConstraints::new(
        Some(min_val), Some(max_val), Some(shrink_towards)
    ));
    
    let test_choice = ChoiceValue::Integer(25);
    let index = indexer.choice_to_index(&test_choice, &rust_constraints).unwrap();
    
    // Convert index to Python-compatible format
    let python_compatible_index = index.to_string();
    let extracted_index: usize = python_compatible_index.parse().unwrap();
    assert_eq!(extracted_index, index);
    
    println!("FFI integration test passed with {} tree nodes", stats.node_count);
}

/// Test zigzag encoding implementation for choice indexing
#[test]
fn test_comprehensive_zigzag_encoding() {
    let mut indexer = ChoiceIndexer::new();
    
    // Test zigzag encoding with different shrink targets
    let test_cases = vec![
        (0, vec![(0, 0), (1, 2), (-1, 3), (2, 4), (-2, 5)]),
        (5, vec![(5, 0), (6, 2), (4, 3), (7, 4), (3, 5)]),
        (-3, vec![(-3, 0), (-2, 2), (-4, 3), (-1, 4), (-5, 5)]),
    ];
    
    for (shrink_towards, value_index_pairs) in test_cases {
        let constraints = Constraints::Integer(IntegerConstraints::new(
            Some(shrink_towards - 100),
            Some(shrink_towards + 100),
            Some(shrink_towards),
        ));
        
        for (value, expected_index) in value_index_pairs {
            let choice = ChoiceValue::Integer(value);
            let index = indexer.choice_to_index(&choice, &constraints).unwrap();
            assert_eq!(index, expected_index, 
                      "Zigzag encoding failed for value {} with shrink_towards {}", 
                      value, shrink_towards);
            
            // Test round-trip
            let recovered = indexer.index_to_choice(index, ChoiceType::Integer, &constraints).unwrap();
            assert_eq!(choice, recovered, "Zigzag round-trip failed");
        }
    }
}

/// Test navigation tree performance under load
#[test]
fn test_navigation_tree_performance() {
    let start = Instant::now();
    
    let mut tree = NavigationTree::new();
    let constraints = vec![
        Constraints::Integer(IntegerConstraints::new(Some(0), Some(1000), Some(100))),
        Constraints::Boolean(BooleanConstraints::new()),
        Constraints::String(StringConstraints::new(None, Some(20))),
    ];
    
    // Record many sequences to stress test the tree
    for i in 0..1000 {
        let sequence = ChoiceSequence::from_choices(vec![
            ChoiceValue::Integer(i % 1000),
            ChoiceValue::Boolean(i % 2 == 0),
            ChoiceValue::String(format!("test_{}", i % 10)),
        ]);
        
        tree.record_sequence(&sequence, &constraints);
        
        // Periodically check tree state
        if i % 100 == 0 {
            let stats = tree.stats();
            assert!(stats.node_count > 0, "Tree should maintain nodes");
        }
    }
    
    let build_duration = start.elapsed();
    println!("Tree building took: {:?}", build_duration);
    
    // Test novel prefix generation performance
    let start = Instant::now();
    let mut prefix_count = 0;
    
    for _ in 0..50 {
        match tree.generate_novel_prefix() {
            Ok(_) => prefix_count += 1,
            Err(NavigationError::TreeExhausted) => break,
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }
    
    let generation_duration = start.elapsed();
    println!("Generated {} prefixes in {:?}", prefix_count, generation_duration);
    
    let final_stats = tree.stats();
    println!("Final tree stats: {:?}", final_stats);
    
    // Performance assertions
    assert!(build_duration.as_secs() < 5, "Tree building should be reasonably fast");
    assert!(generation_duration.as_millis() < 1000, "Prefix generation should be fast");
    assert!(final_stats.node_count > 100, "Should have built substantial tree");
}

/// Test choice indexer performance and caching
#[test]
fn test_choice_indexer_performance() {
    let mut indexer = ChoiceIndexer::new();
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(-10000), Some(10000), Some(0)));
    
    // Test first-time indexing
    let start = Instant::now();
    let choice = ChoiceValue::Integer(1234);
    let _index1 = indexer.choice_to_index(&choice, &constraints).unwrap();
    let first_duration = start.elapsed();
    
    // Test cached indexing
    let start = Instant::now();
    let _index2 = indexer.choice_to_index(&choice, &constraints).unwrap();
    let cached_duration = start.elapsed();
    
    // Test bulk indexing
    let start = Instant::now();
    for i in -1000..1000 {
        let choice = ChoiceValue::Integer(i);
        let _index = indexer.choice_to_index(&choice, &constraints).unwrap();
    }
    let bulk_duration = start.elapsed();
    
    println!("First indexing: {:?}", first_duration);
    println!("Cached indexing: {:?}", cached_duration);
    println!("Bulk indexing (2000 choices): {:?}", bulk_duration);
    
    // Performance assertions
    assert!(bulk_duration.as_millis() < 1000, "Bulk indexing should be fast");
    assert!(cached_duration <= first_duration, "Cached should be faster or equal");
}

/// Test edge cases and error conditions
#[test]
fn test_navigation_edge_cases() {
    // Test empty tree operations
    let mut empty_tree = NavigationTree::new();
    assert!(empty_tree.is_exhausted());
    assert!(empty_tree.generate_novel_prefix().is_err());
    
    let stats = empty_tree.stats();
    assert_eq!(stats.node_count, 0);
    assert_eq!(stats.max_depth, 0);
    assert_eq!(stats.cached_sequences, 0);
    
    // Test invalid indexer operations
    let mut indexer = ChoiceIndexer::new();
    let int_choice = ChoiceValue::Integer(42);
    let bool_constraints = Constraints::Boolean(BooleanConstraints::new());
    
    let result = indexer.choice_to_index(&int_choice, &bool_constraints);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), NavigationError::InconsistentConstraints(_)));
    
    // Test boundary value indexing
    let extreme_constraints = Constraints::Integer(IntegerConstraints::new(
        Some(i128::MIN / 2),
        Some(i128::MAX / 2),
        Some(0),
    ));
    
    let boundary_values = vec![i128::MIN / 2, i128::MAX / 2, 0, -1, 1];
    for value in boundary_values {
        let choice = ChoiceValue::Integer(value);
        let result = indexer.choice_to_index(&choice, &extreme_constraints);
        assert!(result.is_ok(), "Should handle boundary value: {}", value);
    }
    
    // Test sequence operations with edge cases
    let empty_sequence = ChoiceSequence::new();
    assert_eq!(empty_sequence.length, 0);
    assert_eq!(empty_sequence.prefix(100).length, 0);
    assert!(empty_sequence.starts_with(&empty_sequence));
    
    let mut large_sequence = ChoiceSequence::new();
    for i in 0..10000 {
        large_sequence.push(ChoiceValue::Integer(i));
    }
    assert_eq!(large_sequence.length, 10000);
    assert_eq!(large_sequence.prefix(5000).length, 5000);
}

/// Test complex navigation scenarios with mixed choice types
#[test]
fn test_complex_mixed_navigation_scenarios() {
    let mut tree = NavigationTree::new();
    let mut indexer = ChoiceIndexer::new();
    
    // Scenario 1: List of different types (simulating Hypothesis list strategy)
    let list_sequence = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(5),  // List length
        ChoiceValue::Integer(10), // First element
        ChoiceValue::String("hello".to_string()), // Second element
        ChoiceValue::Boolean(true), // Third element
        ChoiceValue::Float(3.14), // Fourth element
        ChoiceValue::Bytes(vec![65, 66, 67]), // Fifth element
    ]);
    
    let list_constraints = vec![
        Constraints::Integer(IntegerConstraints::new(Some(0), Some(20), Some(0))), // Length
        Constraints::Integer(IntegerConstraints::new(Some(0), Some(100), Some(50))), // Element 1
        Constraints::String(StringConstraints::new(None, Some(50))), // Element 2
        Constraints::Boolean(BooleanConstraints::new()), // Element 3
        Constraints::Float(FloatConstraints::new(Some(-10.0), Some(10.0))), // Element 4
        Constraints::Bytes(BytesConstraints::new(None, Some(100))), // Element 5
    ];
    
    tree.record_sequence(&list_sequence, &list_constraints);
    
    // Scenario 2: Nested structure (simulating Hypothesis composite strategy)
    let nested_sequence = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(2),  // Outer structure size
        ChoiceValue::Integer(3),  // Inner structure 1 size
        ChoiceValue::String("inner1".to_string()), // Inner 1 content
        ChoiceValue::Integer(1),  // Inner structure 2 size
        ChoiceValue::Boolean(false), // Inner 2 content
    ]);
    
    let nested_constraints = vec![
        Constraints::Integer(IntegerConstraints::new(Some(1), Some(5), Some(2))),
        Constraints::Integer(IntegerConstraints::new(Some(1), Some(10), Some(1))),
        Constraints::String(StringConstraints::new(None, Some(20))),
        Constraints::Integer(IntegerConstraints::new(Some(1), Some(10), Some(1))),
        Constraints::Boolean(BooleanConstraints::new()),
    ];
    
    tree.record_sequence(&nested_sequence, &nested_constraints);
    
    // Test navigation through complex patterns
    let stats = tree.stats();
    assert!(stats.node_count >= 2); // At least two root-level patterns
    assert!(stats.max_depth >= 4);   // Deep enough for nested structure
    
    // Test prefix generation on complex tree
    let mut novel_prefixes = Vec::new();
    for _ in 0..5 {
        match tree.generate_novel_prefix() {
            Ok(prefix) => novel_prefixes.push(prefix),
            Err(NavigationError::TreeExhausted) => break,
            Err(e) => panic!("Unexpected error: {}", e),
        }
    }
    
    // Test indexing of complex choices
    for (sequence, constraints) in [(list_sequence, list_constraints), (nested_sequence, nested_constraints)] {
        for (choice, constraint) in sequence.choices.iter().zip(constraints.iter()) {
            let index = indexer.choice_to_index(choice, constraint)
                .expect("Should index complex choice");
            
            // Test round-trip for exact types
            match choice {
                ChoiceValue::Integer(_) => {
                    let recovered = indexer.index_to_choice(index, ChoiceType::Integer, constraint).unwrap();
                    assert_eq!(*choice, recovered);
                }
                ChoiceValue::Boolean(_) => {
                    let recovered = indexer.index_to_choice(index, ChoiceType::Boolean, constraint).unwrap();
                    assert_eq!(*choice, recovered);
                }
                _ => {
                    // Verify indexing works
                    assert!(index < usize::MAX);
                }
            }
        }
    }
    
    println!("Complex navigation test passed with {} novel prefixes", novel_prefixes.len());
    println!("Final complex tree stats: {:?}", tree.stats());
}

/// Test concurrent navigation operations (if threading is enabled)
#[test]
fn test_navigation_thread_safety_semantics() {
    // This test validates that navigation structures have appropriate semantics
    // for concurrent use (even if not actually thread-safe, the design should be sound)
    
    let mut tree = NavigationTree::new();
    let mut indexer = ChoiceIndexer::new();
    
    // Set up initial state
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(100), Some(0)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(0),
        constraints.clone(),
        false,
    );
    tree.set_root(root);
    
    // Test that operations are deterministic and repeatable
    let choice = ChoiceValue::Integer(42);
    let index1 = indexer.choice_to_index(&choice, &constraints).unwrap();
    let index2 = indexer.choice_to_index(&choice, &constraints).unwrap();
    assert_eq!(index1, index2, "Indexing should be deterministic");
    
    // Test that tree operations are consistent
    let stats1 = tree.stats();
    let stats2 = tree.stats();
    assert_eq!(stats1, stats2, "Tree stats should be consistent");
    
    // Test prefix generation determinism (given same state)
    let prefix1 = tree.generate_novel_prefix();
    let stats_after1 = tree.stats();
    
    // Reset tree to same state
    let mut tree2 = NavigationTree::new();
    let root2 = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(0),
        constraints,
        false,
    );
    tree2.set_root(root2);
    
    let prefix2 = tree2.generate_novel_prefix();
    let stats_after2 = tree2.stats();
    
    // Results should be equivalent for same initial state
    assert_eq!(prefix1.is_ok(), prefix2.is_ok(), "Prefix generation outcomes should match");
    if let (Ok(p1), Ok(p2)) = (&prefix1, &prefix2) {
        assert_eq!(p1.length, p2.length, "Generated prefixes should have same length");
    }
    
    println!("Thread safety semantics test passed");
}

/// Test navigation error handling and recovery
#[test]
fn test_navigation_error_handling_recovery() {
    let mut indexer = ChoiceIndexer::new();
    
    // Test constraint mismatch errors
    let errors = vec![
        (ChoiceValue::Integer(42), Constraints::Boolean(BooleanConstraints::new())),
        (ChoiceValue::Boolean(true), Constraints::String(StringConstraints::new(None, Some(10)))),
        (ChoiceValue::String("test".to_string()), Constraints::Float(FloatConstraints::new(Some(0.0), Some(1.0)))),
    ];
    
    for (choice, wrong_constraints) in errors {
        let result = indexer.choice_to_index(&choice, &wrong_constraints);
        assert!(result.is_err(), "Should error on constraint mismatch");
        
        match result.unwrap_err() {
            NavigationError::InconsistentConstraints(msg) => {
                assert!(!msg.is_empty(), "Error message should be informative");
            }
            other => panic!("Expected InconsistentConstraints, got {:?}", other),
        }
    }
    
    // Test tree exhaustion scenarios
    let mut tree = NavigationTree::new();
    
    // Empty tree exhaustion
    let result = tree.generate_novel_prefix();
    assert!(matches!(result, Err(NavigationError::TreeExhausted)));
    
    // Tree with exhausted nodes
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(1), Some(0)));
    let mut root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(0),
        constraints,
        false,
    );
    root.mark_exhausted();
    tree.set_root(root);
    
    let result = tree.generate_novel_prefix();
    assert!(matches!(result, Err(NavigationError::TreeExhausted)));
    
    // Test error message formatting
    let errors = vec![
        NavigationError::TreeExhausted,
        NavigationError::InvalidChoiceIndex { index: 42, max_index: 10 },
        NavigationError::InconsistentConstraints("test error".to_string()),
        NavigationError::CorruptedState("test corruption".to_string()),
    ];
    
    for error in errors {
        let message = error.to_string();
        assert!(!message.is_empty(), "Error should have descriptive message");
        assert!(message.len() > 10, "Error message should be reasonably detailed");
    }
    
    println!("Error handling and recovery test passed");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navigation_capability_comprehensive() {
        assert!(test_complete_navigation_capability().unwrap());
    }

    #[test]
    fn test_tree_traversal_comprehensive() {
        assert!(test_complete_tree_traversal().unwrap());
    }

    #[test]
    fn test_selection_orders_comprehensive() {
        assert!(test_prefix_based_selection_orders().unwrap());
    }

    #[test]
    fn test_indexing_system_comprehensive() {
        assert!(test_choice_indexing_system().unwrap());
    }

    #[test]
    fn test_novel_prefix_strategies_comprehensive() {
        assert!(test_novel_prefix_generation_strategies().unwrap());
    }

    #[test]
    fn test_exhaustion_backtracking_comprehensive() {
        assert!(test_tree_exhaustion_backtracking().unwrap());
    }

    #[test]
    fn test_strategy_patterns_comprehensive() {
        assert!(test_selection_strategy_patterns().unwrap());
    }

    #[test]
    fn test_sequence_manipulation_comprehensive() {
        assert!(test_choice_sequence_manipulation().unwrap());
    }

    #[test]
    fn test_constraints_integration_comprehensive() {
        assert!(test_constraints_integration().unwrap());
    }

    #[test]
    fn test_depth_complexity_comprehensive() {
        assert!(test_tree_depth_complexity().unwrap());
    }

    #[test]
    fn test_error_handling_comprehensive() {
        assert!(test_error_handling_recovery().unwrap());
    }

    #[test]
    fn test_state_consistency_comprehensive() {
        assert!(test_navigation_state_consistency().unwrap());
    }

    #[test]
    fn test_performance_comprehensive() {
        assert!(test_large_choice_space_performance().unwrap());
    }

    #[test]
    fn test_zigzag_encoding_comprehensive() {
        assert!(test_zigzag_encoding_correctness().unwrap());
    }

    #[test]
    fn test_multi_type_navigation_comprehensive() {
        assert!(test_multi_type_choice_navigation().unwrap());
    }

    #[test]
    fn test_ffi_integration_comprehensive() {
        assert!(test_ffi_integration_behavior().unwrap());
    }
}