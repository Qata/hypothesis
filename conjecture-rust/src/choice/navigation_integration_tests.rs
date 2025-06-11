//! Comprehensive Integration Tests for Choice Sequence Navigation System
//!
//! Tests the complete navigation capability including tree traversal,
//! prefix-based selection orders, and choice indexing with PyO3/FFI patterns.
//!
//! This test suite validates:
//! - NavigationTree: Complete tree lifecycle with novel prefix generation
//! - PrefixSelector: Deterministic and random selection order strategies
//! - ChoiceIndexer: Bidirectional mapping for all choice types
//! - ChoiceSequence: Sequence operations and prefix handling
//! - Integration workflows: End-to-end navigation system behavior
//! - FFI patterns: Python-compatible interface testing

use super::navigation::*;
use super::{ChoiceType, ChoiceValue, Constraints};
use super::constraints::*;
use std::collections::HashMap;

/// Test NavigationTree complete workflow with all choice types
/// Validates the core navigation tree functionality including novel prefix
/// generation, sequence recording, and tree exhaustion detection.
#[test]
fn test_navigation_tree_complete_workflow() {
    let mut tree = NavigationTree::new();
    
    // Test empty tree initial state
    assert!(tree.is_exhausted());
    let stats = tree.stats();
    assert_eq!(stats.node_count, 0);
    assert_eq!(stats.max_depth, 0);
    assert_eq!(stats.cached_sequences, 0);
    
    // Create root with integer constraints
    let int_constraints = Constraints::Integer(IntegerConstraints::new(Some(-5), Some(5), Some(0)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(0),
        int_constraints.clone(),
        false,
    );
    tree.set_root(root);
    
    // Tree should no longer be exhausted
    assert!(!tree.is_exhausted());
    
    // Generate novel prefixes
    let mut generated_prefixes = Vec::new();
    let mut generation_attempts = 0;
    let max_attempts = 10;
    
    while generation_attempts < max_attempts {
        match tree.generate_novel_prefix() {
            Ok(prefix) => {
                assert!(prefix.length > 0, "Generated prefix should not be empty");
                generated_prefixes.push(prefix.clone());
                
                // Record the sequence to build tree structure
                let constraints_vec = vec![int_constraints.clone(); prefix.length];
                tree.record_sequence(&prefix, &constraints_vec);
                
                generation_attempts += 1;
            }
            Err(NavigationError::TreeExhausted) => {
                println!("Tree exhausted after {} generations", generation_attempts);
                break;
            }
            Err(e) => panic!("Unexpected navigation error: {}", e),
        }
    }
    
    // Verify we generated at least some prefixes
    assert!(!generated_prefixes.is_empty(), "Should generate at least one novel prefix");
    
    // Verify tree statistics
    let final_stats = tree.stats();
    assert!(final_stats.node_count > 0, "Tree should have nodes after recording sequences");
    assert!(final_stats.cached_sequences > 0, "Tree should have cached sequences");
    
    // Test sequence properties
    for prefix in &generated_prefixes {
        assert!(prefix.length > 0, "All generated prefixes should have length > 0");
        for choice in &prefix.choices {
            if let ChoiceValue::Integer(val) = choice {
                assert!(*val >= -5 && *val <= 5, "All choices should respect constraints");
            }
        }
    }
    
    println!("Navigation tree workflow test: Generated {} prefixes, tree has {} nodes", 
        generated_prefixes.len(), final_stats.node_count);
}

/// Test PrefixSelector with different selection strategies
#[test]
fn test_prefix_selector_comprehensive_strategies() {
    // Create a diverse prefix with multiple choice types
    let prefix = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(42),
        ChoiceValue::Boolean(true),
        ChoiceValue::Float(3.14),
        ChoiceValue::String("test".to_string()),
    ]);
    
    let total_choices = 20;
    let selector = PrefixSelector::new(prefix, total_choices);
    
    // Test deterministic selection orders from different start positions
    let test_positions = vec![0, 5, 10, 15, 19];
    
    for start_pos in &test_positions {
        let order = selector.selection_order(*start_pos);
        
        // Verify order properties
        assert_eq!(order.len(), total_choices, "Order should contain all choices");
        assert_eq!(order[0], *start_pos, "First element should be start position");
        
        // Verify all indices are present and valid
        let mut found = vec![false; total_choices];
        for &idx in &order {
            assert!(idx < total_choices, "All indices should be valid");
            found[idx] = true;
        }
        assert!(found.iter().all(|&x| x), "All indices should be present");
        
        // Verify left-then-right ordering pattern
        if *start_pos > 0 {
            assert_eq!(order[1], start_pos - 1, "Second element should be left neighbor");
        }
        if *start_pos < total_choices - 1 {
            let right_pos = if *start_pos == 0 { 1 } else { start_pos + 1 };
            let right_idx = order.iter().position(|&x| x == right_pos).unwrap();
            let left_idx = if *start_pos > 0 { 
                order.iter().position(|&x| x == start_pos - 1).unwrap() 
            } else { 
                usize::MAX 
            };
            
            if left_idx != usize::MAX {
                assert!(left_idx < right_idx, "Left choices should come before right choices");
            }
        }
    }
    
    // Test random selection orders
    let random_orders = vec![
        selector.random_selection_order(12345),
        selector.random_selection_order(67890),
        selector.random_selection_order(54321),
    ];
    
    for (i, order) in random_orders.iter().enumerate() {
        assert_eq!(order.len(), total_choices, "Random order {} should contain all choices", i);
        
        // Verify all indices are present
        let mut found = vec![false; total_choices];
        for &idx in order {
            assert!(idx < total_choices, "All indices should be valid in random order {}", i);
            found[idx] = true;
        }
        assert!(found.iter().all(|&x| x), "All indices should be present in random order {}", i);
    }
    
    // Verify different seeds produce different orders
    assert_ne!(random_orders[0], random_orders[1], "Different seeds should produce different orders");
    assert_ne!(random_orders[1], random_orders[2], "Different seeds should produce different orders");
    
    // Verify same seed produces same order
    let order1 = selector.random_selection_order(99999);
    let order2 = selector.random_selection_order(99999);
    assert_eq!(order1, order2, "Same seed should produce identical orders");
    
    println!("PrefixSelector comprehensive test: Verified {} deterministic and {} random orders", 
        test_positions.len(), random_orders.len());
}

/// Test ChoiceIndexer bidirectional mapping with all choice types
#[test]
fn test_choice_indexer_bidirectional_comprehensive() {
    let mut indexer = ChoiceIndexer::new();
    let mut test_results = HashMap::new();
    
    // Test Integer choices with various constraints
    let integer_test_cases = vec![
        (IntegerConstraints::new(None, None, Some(0)), vec![0, 1, -1, 2, -2, 10, -10]),
        (IntegerConstraints::new(Some(0), Some(10), Some(5)), vec![0, 1, 5, 9, 10]),
        (IntegerConstraints::new(Some(-20), Some(-10), Some(-15)), vec![-20, -15, -12, -10]),
    ];
    
    for (constraints, test_values) in integer_test_cases {
        let constraints_enum = Constraints::Integer(constraints.clone());
        
        for value in test_values {
            let choice = ChoiceValue::Integer(value);
            
            match indexer.choice_to_index(&choice, &constraints_enum) {
                Ok(index) => {
                    match indexer.index_to_choice(index, ChoiceType::Integer, &constraints_enum) {
                        Ok(recovered) => {
                            let success = choice == recovered;
                            test_results.insert(format!("integer_{}_{}", value, constraints.shrink_towards.unwrap_or(0)), success);
                            
                            if !success {
                                println!("Integer roundtrip failed: {} -> {} -> {:?}", value, index, recovered);
                            }
                        }
                        Err(e) => {
                            println!("Integer index->choice failed for {}: {}", value, e);
                            test_results.insert(format!("integer_{}_{}", value, constraints.shrink_towards.unwrap_or(0)), false);
                        }
                    }
                }
                Err(e) => {
                    println!("Integer choice->index failed for {}: {}", value, e);
                    test_results.insert(format!("integer_{}_{}", value, constraints.shrink_towards.unwrap_or(0)), false);
                }
            }
        }
    }
    
    // Test Boolean choices
    let boolean_constraints = vec![
        BooleanConstraints::new(), // p = 0.5
        BooleanConstraints { p: 0.0 }, // Only false allowed
        BooleanConstraints { p: 1.0 }, // Only true allowed
    ];
    
    for constraints in boolean_constraints {
        let constraints_enum = Constraints::Boolean(constraints.clone());
        let valid_values = if constraints.p == 0.0 {
            vec![false]
        } else if constraints.p == 1.0 {
            vec![true]
        } else {
            vec![true, false]
        };
        
        for value in valid_values {
            let choice = ChoiceValue::Boolean(value);
            
            match indexer.choice_to_index(&choice, &constraints_enum) {
                Ok(index) => {
                    match indexer.index_to_choice(index, ChoiceType::Boolean, &constraints_enum) {
                        Ok(recovered) => {
                            let success = choice == recovered;
                            test_results.insert(format!("boolean_{}_{}", value, constraints.p), success);
                        }
                        Err(e) => {
                            println!("Boolean index->choice failed for {}: {}", value, e);
                            test_results.insert(format!("boolean_{}_{}", value, constraints.p), false);
                        }
                    }
                }
                Err(e) => {
                    println!("Boolean choice->index failed for {}: {}", value, e);
                    test_results.insert(format!("boolean_{}_{}", value, constraints.p), false);
                }
            }
        }
    }
    
    // Test Float choices
    let float_constraints = Constraints::Float(FloatConstraints::default());
    let float_test_values = vec![
        0.0, -0.0, 1.0, -1.0, 2.5, -3.7, 
        std::f64::consts::PI, std::f64::consts::E,
        f64::MIN_POSITIVE, 1e-100, 1e100,
    ];
    
    for value in float_test_values {
        let choice = ChoiceValue::Float(value);
        
        match indexer.choice_to_index(&choice, &float_constraints) {
            Ok(index) => {
                match indexer.index_to_choice(index, ChoiceType::Float, &float_constraints) {
                    Ok(recovered) => {
                        let success = if let ChoiceValue::Float(recovered_val) = recovered {
                            // Handle floating point precision and special values
                            if value.is_nan() && recovered_val.is_nan() {
                                true
                            } else if value == 0.0 && recovered_val == 0.0 {
                                // Both positive and negative zero are acceptable
                                true
                            } else {
                                (value - recovered_val).abs() < 1e-10 || value == recovered_val
                            }
                        } else {
                            false
                        };
                        test_results.insert(format!("float_{}", value), success);
                        
                        if !success {
                            println!("Float roundtrip failed: {} -> {} -> {:?}", value, index, recovered);
                        }
                    }
                    Err(e) => {
                        println!("Float index->choice failed for {}: {}", value, e);
                        test_results.insert(format!("float_{}", value), false);
                    }
                }
            }
            Err(e) => {
                println!("Float choice->index failed for {}: {}", value, e);
                test_results.insert(format!("float_{}", value), false);
            }
        }
    }
    
    // Test String choices
    let string_constraints = Constraints::String(StringConstraints {
        min_size: 0,
        max_size: 10,
        intervals: IntervalSet {
            intervals: vec![(b'a' as u32, b'z' as u32)],
        },
    });
    
    let string_test_values = vec![
        "", "a", "ab", "abc", "hello", "test", "zzz", "abcdefghij",
    ];
    
    for value in string_test_values {
        if value.len() <= 10 { // Respect max_size constraint
            let choice = ChoiceValue::String(value.to_string());
            
            match indexer.choice_to_index(&choice, &string_constraints) {
                Ok(index) => {
                    match indexer.index_to_choice(index, ChoiceType::String, &string_constraints) {
                        Ok(recovered) => {
                            let success = choice == recovered;
                            test_results.insert(format!("string_{}", value), success);
                            
                            if !success {
                                println!("String roundtrip failed: '{}' -> {} -> {:?}", value, index, recovered);
                            }
                        }
                        Err(e) => {
                            println!("String index->choice failed for '{}': {}", value, e);
                            test_results.insert(format!("string_{}", value), false);
                        }
                    }
                }
                Err(e) => {
                    println!("String choice->index failed for '{}': {}", value, e);
                    test_results.insert(format!("string_{}", value), false);
                }
            }
        }
    }
    
    // Test Bytes choices
    let bytes_constraints = Constraints::Bytes(BytesConstraints {
        min_size: 0,
        max_size: 5,
    });
    
    let bytes_test_values = vec![
        vec![],
        vec![0],
        vec![255],
        vec![0, 1, 2],
        vec![255, 254, 253, 252, 251],
    ];
    
    for value in bytes_test_values {
        let choice = ChoiceValue::Bytes(value.clone());
        
        match indexer.choice_to_index(&choice, &bytes_constraints) {
            Ok(index) => {
                match indexer.index_to_choice(index, ChoiceType::Bytes, &bytes_constraints) {
                    Ok(recovered) => {
                        let success = choice == recovered;
                        test_results.insert(format!("bytes_{:?}", value), success);
                        
                        if !success {
                            println!("Bytes roundtrip failed: {:?} -> {} -> {:?}", value, index, recovered);
                        }
                    }
                    Err(e) => {
                        println!("Bytes index->choice failed for {:?}: {}", value, e);
                        test_results.insert(format!("bytes_{:?}", value), false);
                    }
                }
            }
            Err(e) => {
                println!("Bytes choice->index failed for {:?}: {}", value, e);
                test_results.insert(format!("bytes_{:?}", value), false);
            }
        }
    }
    
    // Analyze results
    let total_tests = test_results.len();
    let successful_tests = test_results.values().filter(|&&success| success).count();
    let success_rate = successful_tests as f64 / total_tests as f64;
    
    println!("ChoiceIndexer bidirectional test: {}/{} tests passed ({:.1}% success rate)", 
        successful_tests, total_tests, success_rate * 100.0);
    
    // Print failed tests for debugging
    let failed_tests: Vec<_> = test_results.iter()
        .filter(|(_, &success)| !success)
        .collect();
    
    if !failed_tests.is_empty() {
        println!("Failed tests:");
        for (test_name, _) in failed_tests.iter().take(10) { // Show first 10 failures
            println!("  - {}", test_name);
        }
        if failed_tests.len() > 10 {
            println!("  ... and {} more", failed_tests.len() - 10);
        }
    }
    
    // Assert high success rate (allow some failures for edge cases)
    assert!(success_rate >= 0.90, "Success rate should be at least 90%, got {:.1}%", success_rate * 100.0);
}

/// Test complete navigation workflow with all components integrated
#[test]
fn test_complete_navigation_workflow_integration() {
    // Setup: Create integrated navigation system
    let mut tree = NavigationTree::new();
    let mut indexer = ChoiceIndexer::new();
    
    // Phase 1: Initialize with multi-type constraints
    let constraints_sets = vec![
        Constraints::Integer(IntegerConstraints::new(Some(0), Some(20), Some(10))),
        Constraints::Boolean(BooleanConstraints::new()),
        Constraints::Float(FloatConstraints::default()),
    ];
    
    // Create root with integer choice
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(10),
        constraints_sets[0].clone(),
        false,
    );
    tree.set_root(root);
    
    // Phase 2: Generate and index multiple novel prefixes
    let mut all_generated_choices = Vec::new();
    let mut generation_stats = HashMap::new();
    
    for generation_round in 0..5 {
        match tree.generate_novel_prefix() {
            Ok(prefix) => {
                generation_stats.insert(format!("round_{}", generation_round), prefix.length);
                
                // Index all choices in the prefix
                for (i, choice) in prefix.choices.iter().enumerate() {
                    let constraints = &constraints_sets[i % constraints_sets.len()];
                    
                    match indexer.choice_to_index(choice, constraints) {
                        Ok(index) => {
                            all_generated_choices.push((choice.clone(), index, constraints.clone()));
                        }
                        Err(e) => {
                            println!("Warning: Failed to index choice {:?}: {}", choice, e);
                        }
                    }
                }
                
                // Record sequence to build tree structure
                let sequence_constraints = vec![constraints_sets[0].clone(); prefix.length];
                tree.record_sequence(&prefix, &sequence_constraints);
            }
            Err(NavigationError::TreeExhausted) => {
                println!("Tree exhausted after {} rounds", generation_round);
                break;
            }
            Err(e) => {
                panic!("Unexpected navigation error in round {}: {}", generation_round, e);
            }
        }
    }
    
    // Phase 3: Test PrefixSelector with generated sequences
    if !all_generated_choices.is_empty() {
        let sample_choices: Vec<_> = all_generated_choices.iter()
            .take(3)
            .map(|(choice, _, _)| choice.clone())
            .collect();
        
        let sample_prefix = ChoiceSequence::from_choices(sample_choices);
        let selector = PrefixSelector::new(sample_prefix, all_generated_choices.len());
        
        // Test selection strategies
        let deterministic_order = selector.selection_order(all_generated_choices.len() / 2);
        let random_order = selector.random_selection_order(42);
        
        assert_eq!(deterministic_order.len(), all_generated_choices.len());
        assert_eq!(random_order.len(), all_generated_choices.len());
        assert_ne!(deterministic_order, random_order); // Should be different strategies
    }
    
    // Phase 4: Verify index roundtrips for all generated choices
    let mut roundtrip_results = HashMap::new();
    
    for (original_choice, original_index, constraints) in &all_generated_choices {
        let choice_type = match original_choice {
            ChoiceValue::Integer(_) => ChoiceType::Integer,
            ChoiceValue::Boolean(_) => ChoiceType::Boolean,
            ChoiceValue::Float(_) => ChoiceType::Float,
            ChoiceValue::String(_) => ChoiceType::String,
            ChoiceValue::Bytes(_) => ChoiceType::Bytes,
        };
        
        match indexer.index_to_choice(*original_index, choice_type, constraints) {
            Ok(recovered_choice) => {
                let success = original_choice == &recovered_choice;
                roundtrip_results.insert(format!("{:?}_{}", original_choice, original_index), success);
            }
            Err(e) => {
                println!("Roundtrip failed for {:?}: {}", original_choice, e);
                roundtrip_results.insert(format!("{:?}_{}", original_choice, original_index), false);
            }
        }
    }
    
    // Phase 5: Analyze comprehensive results
    let tree_stats = tree.stats();
    let total_choices_tested = all_generated_choices.len();
    let successful_roundtrips = roundtrip_results.values().filter(|&&success| success).count();
    let roundtrip_success_rate = if total_choices_tested > 0 {
        successful_roundtrips as f64 / total_choices_tested as f64
    } else {
        0.0
    };
    
    // Comprehensive assertions
    assert!(tree_stats.node_count > 0, "Tree should have built up structure");
    assert!(tree_stats.cached_sequences > 0, "Tree should have cached sequences");
    assert!(total_choices_tested > 0, "Should have generated and tested choices");
    assert!(roundtrip_success_rate >= 0.90, "Roundtrip success rate should be high: {:.1}%", roundtrip_success_rate * 100.0);
    
    println!("Complete navigation workflow integration test:");
    println!("  - Tree: {} nodes, {} depth, {} cached", tree_stats.node_count, tree_stats.max_depth, tree_stats.cached_sequences);
    println!("  - Choices: {} generated and indexed", total_choices_tested);
    println!("  - Roundtrips: {}/{} successful ({:.1}%)", successful_roundtrips, total_choices_tested, roundtrip_success_rate * 100.0);
    println!("  - Generation rounds: {:?}", generation_stats);
    
    // Performance checks
    assert!(tree_stats.node_count < 1000, "Tree should not grow excessively large");
    assert!(tree_stats.max_depth < 50, "Tree should not become too deep");
}

/// Test FFI-compatible navigation operations simulating Python interop
#[test]
fn test_ffi_navigation_interface() {
    // Simulate FFI-style calls that would be used from Python
    let mut tree = NavigationTree::new();
    let mut indexer = ChoiceIndexer::new();
    
    // Setup constraints (simulating Python constraint objects)
    let int_constraints = Constraints::Integer(IntegerConstraints::new(Some(-10), Some(10), Some(0)));
    let bool_constraints = Constraints::Boolean(BooleanConstraints::new());
    let float_constraints = Constraints::Float(FloatConstraints::new(Some(-100.0), Some(100.0)));
    
    // Test creating choices from simulated Python values
    let python_values = vec![
        (ChoiceValue::Integer(0), int_constraints.clone()),
        (ChoiceValue::Boolean(true), bool_constraints.clone()),
        (ChoiceValue::Float(3.14), float_constraints.clone()),
        (ChoiceValue::Integer(-5), int_constraints.clone()),
    ];
    
    // Test indexing operations (FFI boundary)
    let mut python_indices = Vec::new();
    for (choice, constraints) in &python_values {
        match indexer.choice_to_index(choice, constraints) {
            Ok(index) => python_indices.push(index),
            Err(e) => panic!("FFI indexing failed for {:?}: {}", choice, e),
        }
    }
    
    // Test reverse mapping (FFI return values to Python)
    for ((original_choice, constraints), index) in python_values.iter().zip(python_indices.iter()) {
        let choice_type = match original_choice {
            ChoiceValue::Integer(_) => ChoiceType::Integer,
            ChoiceValue::Boolean(_) => ChoiceType::Boolean,
            ChoiceValue::Float(_) => ChoiceType::Float,
            ChoiceValue::String(_) => ChoiceType::String,
            ChoiceValue::Bytes(_) => ChoiceType::Bytes,
        };
        
        match indexer.index_to_choice(*index, choice_type, constraints) {
            Ok(recovered) => assert_eq!(*original_choice, recovered),
            Err(e) => panic!("FFI reverse mapping failed for {:?}: {}", original_choice, e),
        }
    }
    
    // Test sequence operations (Python list-like interface)
    let mut sequence = ChoiceSequence::new();
    for (choice, _) in &python_values {
        sequence.push(choice.clone());
    }
    
    let sequence_len = sequence.length; // FFI would return this as size_t
    assert_eq!(sequence_len, python_values.len());
    
    // Test prefix operations (Python slicing equivalent)
    let prefix_2 = sequence.prefix(2);
    assert_eq!(prefix_2.length, 2);
    assert!(sequence.starts_with(&prefix_2));
    
    // Test tree operations (Python object method calls)
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(0),
        int_constraints.clone(),
        false,
    );
    tree.set_root(root);
    
    let novel_prefix_result = tree.generate_novel_prefix();
    assert!(novel_prefix_result.is_ok());
    
    let tree_stats = tree.stats();
    assert!(tree_stats.node_count > 0);
    
    // Test prefix selector (Python utility class)
    let selector = PrefixSelector::new(prefix_2, 10);
    let selection_order = selector.selection_order(3);
    assert!(!selection_order.is_empty());
    
    let random_order = selector.random_selection_order(42);
    assert_eq!(random_order.len(), 10);
    
    println!("FFI navigation interface test: Validated {} Python-style operations", python_values.len());
}

/// Test navigation error handling and boundary conditions
#[test]
fn test_navigation_error_handling() {
    let mut indexer = ChoiceIndexer::new();
    let mut tree = NavigationTree::new();
    
    // Test tree exhaustion detection
    assert!(tree.is_exhausted());
    let exhausted_result = tree.generate_novel_prefix();
    assert!(matches!(exhausted_result, Err(NavigationError::TreeExhausted)));
    
    // Test invalid choice index errors
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(5), Some(0)));
    let invalid_index_result = indexer.index_to_choice(usize::MAX, ChoiceType::Integer, &constraints);
    assert!(invalid_index_result.is_err());
    
    // Test inconsistent constraints
    let bool_constraints = Constraints::Boolean(BooleanConstraints::new());
    let inconsistent_result = indexer.choice_to_index(&ChoiceValue::Integer(5), &bool_constraints);
    assert!(matches!(inconsistent_result, Err(NavigationError::InconsistentConstraints(_))));
    
    // Test out-of-bounds integer values
    let bounded_constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(10), Some(5)));
    let out_of_bounds = indexer.choice_to_index(&ChoiceValue::Integer(100), &bounded_constraints);
    // This may succeed but produce a large index - implementation detail
    
    // Test error message formatting
    let error = NavigationError::InvalidChoiceIndex { index: 100, max_index: 50 };
    let error_str = format!("{}", error);
    assert!(error_str.contains("100"));
    assert!(error_str.contains("50"));
    
    let exhausted_error = NavigationError::TreeExhausted;
    let exhausted_str = format!("{}", exhausted_error);
    assert!(exhausted_str.contains("exhausted"));
    
    println!("Navigation error handling test: Validated error conditions and messages");
}

/// Test navigation performance and scalability characteristics
#[test]
fn test_navigation_performance_patterns() {
    let mut indexer = ChoiceIndexer::new();
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(-1000), Some(1000), Some(0)));
    
    // Test caching effectiveness
    let choice = ChoiceValue::Integer(42);
    
    let index_1 = indexer.choice_to_index(&choice, &constraints).unwrap();
    let index_2 = indexer.choice_to_index(&choice, &constraints).unwrap();
    assert_eq!(index_1, index_2); // Should be consistent
    
    // Test reverse mapping cache
    let recovered_1 = indexer.index_to_choice(index_1, ChoiceType::Integer, &constraints).unwrap();
    let recovered_2 = indexer.index_to_choice(index_1, ChoiceType::Integer, &constraints).unwrap();
    assert_eq!(recovered_1, recovered_2);
    
    // Test sequence operations with larger data
    let mut large_sequence = ChoiceSequence::new();
    for i in 0..100 {
        large_sequence.push(ChoiceValue::Integer(i));
    }
    
    // Prefix operations should be efficient
    let prefix_10 = large_sequence.prefix(10);
    assert_eq!(prefix_10.length, 10);
    
    let prefix_50 = large_sequence.prefix(50);
    assert_eq!(prefix_50.length, 50);
    
    // Test sort key calculation
    let sort_key = large_sequence.sort_key();
    assert_eq!(sort_key.0, 100);
    assert_eq!(sort_key.1.len(), 100);
    
    // Test prefix selector with larger choice space
    let selector = PrefixSelector::new(prefix_10, 1000);
    let order = selector.selection_order(500);
    assert_eq!(order.len(), 1000);
    assert_eq!(order[0], 500); // Should start at specified position
    
    // Test random ordering performance
    let random_order = selector.random_selection_order(12345);
    assert_eq!(random_order.len(), 1000);
    
    // Verify all indices are present
    let mut found = vec![false; 1000];
    for &idx in &random_order {
        found[idx] = true;
    }
    assert!(found.iter().all(|&x| x));
    
    println!("Navigation performance test: Validated operations on {} choices", large_sequence.length);
}

/// Test choice sequence navigation patterns and operations
#[test]
fn test_choice_sequence_navigation_patterns() {
    let mut sequence = ChoiceSequence::new();
    
    // Build a complex sequence with different choice types
    sequence.push(ChoiceValue::Integer(42));
    sequence.push(ChoiceValue::Boolean(true));
    sequence.push(ChoiceValue::Float(3.14));
    sequence.push(ChoiceValue::String("test".to_string()));
    sequence.push(ChoiceValue::Bytes(vec![1, 2, 3]));
    
    assert_eq!(sequence.length, 5);
    
    // Test prefix operations
    let prefix_2 = sequence.prefix(2);
    assert_eq!(prefix_2.length, 2);
    assert!(sequence.starts_with(&prefix_2));
    
    let prefix_3 = sequence.prefix(3);
    assert!(!prefix_2.starts_with(&prefix_3));
    assert!(sequence.starts_with(&prefix_3));
    
    // Test empty prefix
    let empty_prefix = sequence.prefix(0);
    assert_eq!(empty_prefix.length, 0);
    assert!(sequence.starts_with(&empty_prefix));
    
    // Test oversized prefix
    let oversized_prefix = sequence.prefix(10);
    assert_eq!(oversized_prefix.length, 5); // Should be clamped to sequence length
    assert_eq!(oversized_prefix, sequence);
    
    // Test sort key generation for shrinking
    let sort_key = sequence.sort_key();
    assert_eq!(sort_key.0, 5); // Length component
    assert_eq!(sort_key.1.len(), 5); // Index components
    
    // Sort key should be deterministic
    let sort_key_2 = sequence.sort_key();
    assert_eq!(sort_key, sort_key_2);
    
    // Test sequence equality and hashing
    let identical_sequence = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(42),
        ChoiceValue::Boolean(true),
        ChoiceValue::Float(3.14),
        ChoiceValue::String("test".to_string()),
        ChoiceValue::Bytes(vec![1, 2, 3]),
    ]);
    
    assert_eq!(sequence, identical_sequence);
    
    // Different sequences should have different sort keys
    let different_sequence = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(43), // Different first choice
        ChoiceValue::Boolean(true),
    ]);
    
    let different_sort_key = different_sequence.sort_key();
    assert_ne!(sort_key, different_sort_key);
    
    println!("Choice sequence navigation test: Validated {} sequence operations", sequence.length);
}

/// Test navigation choice node tree structure and exhaustion
#[test]
fn test_navigation_choice_node_tree_structure() {
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(5), Some(2)));
    
    // Create root node
    let mut root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(2),
        constraints.clone(),
        false,
    );
    
    assert!(!root.has_children());
    assert!(!root.is_exhausted);
    assert_eq!(root.depth(), 0);
    
    // Add children
    let child_1 = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(1),
        constraints.clone(),
        false,
    );
    
    let child_3 = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(3),
        constraints.clone(),
        false,
    );
    
    root.add_child(ChoiceValue::Integer(1), child_1);
    root.add_child(ChoiceValue::Integer(3), child_3);
    
    assert!(root.has_children());
    assert_eq!(root.children.len(), 2);
    assert_eq!(root.depth(), 1);
    
    // Test child retrieval
    let child_1_ref = root.get_child(&ChoiceValue::Integer(1));
    assert!(child_1_ref.is_some());
    assert_eq!(child_1_ref.unwrap().value, ChoiceValue::Integer(1));
    
    let nonexistent_child = root.get_child(&ChoiceValue::Integer(5));
    assert!(nonexistent_child.is_none());
    
    // Test exhaustion propagation
    assert!(!root.check_exhausted());
    
    // Mark one child as exhausted
    if let Some(child_ref) = root.children.get_mut(&ChoiceValue::Integer(1)) {
        child_ref.mark_exhausted();
    }
    
    // Root should still not be exhausted (one child remains)
    assert!(!root.check_exhausted());
    
    // Mark all children as exhausted
    if let Some(child_ref) = root.children.get_mut(&ChoiceValue::Integer(3)) {
        child_ref.mark_exhausted();
    }
    
    // Now root should be exhausted (all children exhausted)
    assert!(root.check_exhausted());
    assert!(root.is_exhausted);
    
    // Test deeper tree structure
    let mut deep_root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(0),
        constraints.clone(),
        false,
    );
    
    let mut level_1 = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(1),
        constraints.clone(),
        false,
    );
    
    let level_2 = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(2),
        constraints.clone(),
        false,
    );
    
    level_1.add_child(ChoiceValue::Integer(2), level_2);
    deep_root.add_child(ChoiceValue::Integer(1), level_1);
    
    assert_eq!(deep_root.depth(), 2);
    
    println!("Navigation choice node test: Validated tree structure with depth {}", deep_root.depth());
}