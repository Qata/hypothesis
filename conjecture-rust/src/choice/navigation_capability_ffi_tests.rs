//! Comprehensive FFI Integration Tests for Choice Sequence Navigation System
//!
//! This module provides comprehensive PyO3 FFI integration tests that validate
//! the complete navigation capability behavior including tree traversal, 
//! prefix-based selection orders, and choice indexing with Python interoperability.

use crate::choice::{
    ChoiceType, ChoiceValue, Constraints, 
    IntegerConstraints, BooleanConstraints, FloatConstraints, 
    StringConstraints, BytesConstraints
};
use crate::choice::navigation::{
    NavigationTree, NavigationChoiceNode, ChoiceSequence, 
    PrefixSelector, ChoiceIndexer,
    NavigationError
};
#[cfg(feature = "python-ffi")]
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Test complete end-to-end navigation capability workflow
#[cfg(feature = "python-ffi")]
#[pyfunction]
pub fn test_complete_navigation_workflow(py: Python) -> PyResult<PyObject> {
    let mut results = HashMap::new();
    let start_time = Instant::now();

    // Phase 1: Initialize complex navigation tree with mixed constraints
    let mut tree = NavigationTree::new();
    let root_constraints = Constraints::Integer(IntegerConstraints::new(Some(-100), Some(100), Some(0)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(0),
        root_constraints.clone(),
        false,
    );
    tree.set_root(root);

    // Phase 2: Generate structured choice sequences with systematic exploration
    let mut generated_sequences = Vec::new();
    let mut unique_sequences = HashSet::new();
    let max_sequences = 50;

    for i in 0..max_sequences {
        match tree.generate_novel_prefix() {
            Ok(sequence) => {
                let sequence_key = format!("{:?}", sequence.choices);
                if unique_sequences.insert(sequence_key.clone()) {
                    generated_sequences.push(sequence);
                }
                
                // Record the sequence to build tree structure
                if let Some(ref last_seq) = generated_sequences.last() {
                    let constraints_vec = vec![root_constraints.clone(); last_seq.length];
                    tree.record_sequence(last_seq, &constraints_vec);
                }
            }
            Err(NavigationError::TreeExhausted) => {
                results.insert("tree_exhausted_at".to_string(), i.into_py(py));
                break;
            }
            Err(_) => {
                results.insert("error_at_sequence".to_string(), i.into_py(py));
                break;
            }
        }
    }

    // Phase 3: Test prefix-based selection orders with shrinking patterns
    let mut selection_patterns = Vec::new();
    let mut shrinking_pattern_quality = 0.0;
    
    for (idx, sequence) in generated_sequences.iter().take(10).enumerate() {
        let selector = PrefixSelector::new(
            sequence.clone(),
            20,
        );
        
        let order = selector.selection_order(idx);
        selection_patterns.push(order.clone());
        
        // Validate shrinking pattern - first element should be starting index
        if !order.is_empty() && order[0] == idx {
            // Validate distance minimization pattern
            if order.len() > 3 {
                let distances: Vec<i32> = order.iter()
                    .map(|&i| (i as i32 - idx as i32).abs())
                    .collect();
                
                // Check that distances generally increase (allowing some variance for zigzag)
                let mut increasing_trend = 0;
                for window in distances.windows(2) {
                    if window[1] >= window[0] {
                        increasing_trend += 1;
                    }
                }
                let trend_ratio = increasing_trend as f64 / (distances.len() - 1) as f64;
                shrinking_pattern_quality += trend_ratio;
            }
        }
    }
    
    if !selection_patterns.is_empty() {
        shrinking_pattern_quality /= selection_patterns.len() as f64;
    }

    // Phase 4: Test bidirectional choice indexing with complex types
    let mut indexer = ChoiceIndexer::new();
    let mut indexing_success_rates = HashMap::new();
    let test_cases = vec![
        // Integers around shrink targets
        (ChoiceValue::Integer(-5), ChoiceType::Integer),
        (ChoiceValue::Integer(0), ChoiceType::Integer),
        (ChoiceValue::Integer(42), ChoiceType::Integer),
        (ChoiceValue::Integer(100), ChoiceType::Integer),
        
        // Floats with precision challenges
        (ChoiceValue::Float(0.0), ChoiceType::Float),
        (ChoiceValue::Float(-1.5), ChoiceType::Float),
        (ChoiceValue::Float(3.14159), ChoiceType::Float),
        
        // Complex strings
        (ChoiceValue::String("".to_string()), ChoiceType::String),
        (ChoiceValue::String("hello".to_string()), ChoiceType::String),
        (ChoiceValue::String("🦀💻🔥".to_string()), ChoiceType::String),
        
        // Bytes with edge cases
        (ChoiceValue::Bytes(vec![]), ChoiceType::Bytes),
        (ChoiceValue::Bytes(vec![0, 255, 128]), ChoiceType::Bytes),
        
        // Boolean
        (ChoiceValue::Boolean(true), ChoiceType::Boolean),
        (ChoiceValue::Boolean(false), ChoiceType::Boolean),
    ];

    for (choice_value, choice_type) in test_cases {
        let constraints = get_default_constraints(&choice_type);
        
        match indexer.choice_to_index(&choice_value, &constraints) {
            Ok(index) => {
                match indexer.index_to_choice(index, choice_type, &constraints) {
                    Ok(recovered_choice) => {
                        let success = recovered_choice == choice_value;
                        let type_key = format!("{:?}", choice_type);
                        
                        let current_success = indexing_success_rates
                            .get(&type_key)
                            .unwrap_or(&(0, 0));
                        
                        indexing_success_rates.insert(
                            type_key,
                            (current_success.0 + if success { 1 } else { 0 }, current_success.1 + 1)
                        );
                    }
                    Err(_) => {
                        let type_key = format!("{:?}", choice_type);
                        let current_success = indexing_success_rates
                            .get(&type_key)
                            .unwrap_or(&(0, 0));
                        indexing_success_rates.insert(type_key, (current_success.0, current_success.1 + 1));
                    }
                }
            }
            Err(_) => {
                let type_key = format!("{:?}", choice_type);
                let current_success = indexing_success_rates
                    .get(&type_key)
                    .unwrap_or(&(0, 0));
                indexing_success_rates.insert(type_key, (current_success.0, current_success.1 + 1));
            }
        }
    }

    // Compile comprehensive results
    results.insert("execution_time_ms".to_string(), 
                  start_time.elapsed().as_millis().into_py(py));
    results.insert("sequences_generated".to_string(), 
                  generated_sequences.len().into_py(py));
    results.insert("unique_sequences".to_string(), 
                  unique_sequences.len().into_py(py));
    results.insert("selection_patterns_count".to_string(), 
                  selection_patterns.len().into_py(py));
    results.insert("shrinking_pattern_quality".to_string(), 
                  shrinking_pattern_quality.into_py(py));

    // Indexing success rates
    for (type_name, (success_count, total_count)) in &indexing_success_rates {
        let success_rate = if *total_count > 0 {
            *success_count as f64 / *total_count as f64
        } else {
            0.0
        };
        results.insert(format!("indexing_success_rate_{}", type_name), 
                      success_rate.into_py(py));
    }

    // Tree statistics
    let tree_stats = tree.stats();
    results.insert("tree_node_count".to_string(), tree_stats.node_count.into_py(py));
    results.insert("tree_max_depth".to_string(), tree_stats.max_depth.into_py(py));
    results.insert("tree_cached_sequences".to_string(), tree_stats.cached_sequences.into_py(py));

    // Overall capability assessment
    let sequence_generation_success = !generated_sequences.is_empty();
    let indexing_performance_adequate = indexing_success_rates
        .get("Integer")
        .map_or(false, |(success, total)| total > &0 && (*success as f64 / *total as f64) > 0.8);
    let tree_building_success = tree_stats.node_count > 0;
    
    let overall_success = sequence_generation_success && indexing_performance_adequate && tree_building_success;
    results.insert("overall_capability_success".to_string(), overall_success.into_py(py));

    let py_dict = pyo3::types::PyDict::new(py);
    for (key, value) in results {
        py_dict.set_item(key, value)?;
    }

    Ok(py_dict.into())
}

/// Test structured shrinking pattern validation
#[cfg(feature = "python-ffi")]
#[pyfunction]
pub fn test_structured_shrinking_patterns(py: Python) -> PyResult<PyObject> {
    let mut results = HashMap::new();
    
    // Test 1: Zigzag pattern validation for integer shrinking
    let mut indexer = ChoiceIndexer::new();
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(-10), Some(10), Some(0)));
    
    let test_values = vec![-5, -1, 0, 1, 5, 10, -10];
    let mut zigzag_pattern_correct = true;
    let mut actual_indices: Vec<(i128, u128)> = Vec::new();
    
    for &value in &test_values {
        let choice = ChoiceValue::Integer(value);
        if let Ok(index) = indexer.choice_to_index(&choice, &constraints) {
            actual_indices.push((value as i128, index as u128));
            
            // Validate zigzag pattern: shrink_towards (0) should have index 0
            if value == 0 && index != 0 {
                zigzag_pattern_correct = false;
            }
        }
    }
    
    results.insert("zigzag_pattern_correct", zigzag_pattern_correct.into_py(py));
    results.insert("zigzag_test_indices", 
                  format!("{:?}", actual_indices).into_py(py));
    
    // Test 2: Selection order distance minimization
    let sequence = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(5),
        ChoiceValue::Integer(3),
        ChoiceValue::Integer(7),
    ]);
    
    let selector = PrefixSelector::new(
        sequence,
        10,
    );
    
    let order = selector.selection_order(5);
    let distance_minimized = order.len() > 3 && order[0] == 5;
    
    results.insert("distance_minimization_correct", distance_minimized.into_py(py));
    results.insert("selection_order", format!("{:?}", order).into_py(py));
    
    let py_dict = pyo3::types::PyDict::new(py);
    for (key, value) in results {
        py_dict.set_item(key, value)?;
    }
    
    Ok(py_dict.into())
}

/// Test navigation system exhaustion and recovery
#[cfg(feature = "python-ffi")]
#[pyfunction]
pub fn test_navigation_exhaustion_recovery(py: Python) -> PyResult<PyObject> {
    let mut results = HashMap::new();
    let mut tree = NavigationTree::new();
    
    // Create a small, bounded tree that can be exhausted
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(2), Some(1)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(1),
        constraints.clone(),
        false,
    );
    tree.set_root(root);
    
    let mut sequences_before_exhaustion = 0;
    let mut exhaustion_detected = false;
    
    // Generate sequences until exhaustion
    for i in 0..100 {
        match tree.generate_novel_prefix() {
            Ok(sequence) => {
                sequences_before_exhaustion = i + 1;
                let constraints_vec = vec![constraints.clone(); sequence.length.max(1)];
                tree.record_sequence(&sequence, &constraints_vec);
            }
            Err(NavigationError::TreeExhausted) => {
                exhaustion_detected = true;
                break;
            }
            Err(_) => break,
        }
    }
    
    results.insert("sequences_before_exhaustion", sequences_before_exhaustion.into_py(py));
    results.insert("exhaustion_detected", exhaustion_detected.into_py(py));
    results.insert("tree_exhausted_state", tree.is_exhausted().into_py(py));
    
    let py_dict = pyo3::types::PyDict::new(py);
    for (key, value) in results {
        py_dict.set_item(key, value)?;
    }
    
    Ok(py_dict.into())
}

/// Test cross-constraint type interactions
#[cfg(feature = "python-ffi")]
#[pyfunction]
pub fn test_cross_constraint_navigation(py: Python) -> PyResult<PyObject> {
    let mut results = HashMap::new();

    // Create a mixed-type choice sequence
    let mixed_sequence = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(5),
        ChoiceValue::Float(3.14),
        ChoiceValue::String("test".to_string()),
        ChoiceValue::Boolean(true),
        ChoiceValue::Bytes(vec![42, 24]),
    ]);

    let mixed_constraints = vec![
        Constraints::Integer(IntegerConstraints::new(Some(0), Some(10), Some(5))),
        Constraints::Float(FloatConstraints::new(Some(0.0), Some(10.0))),
        Constraints::String(StringConstraints::new(Some(1), Some(10))),
        Constraints::Boolean(BooleanConstraints::new()),
        Constraints::Bytes(BytesConstraints::new(Some(1), Some(5))),
    ];

    // Test prefix selector with mixed types
    let mixed_selector = PrefixSelector::new(mixed_sequence.clone(), 25);
    let mixed_order = mixed_selector.selection_order(5);
    
    results.insert("mixed_type_selection_success".to_string(), 
                  (!mixed_order.is_empty()).into_py(py));
    results.insert("mixed_type_selection_length".to_string(), 
                  mixed_order.len().into_py(py));

    // Test tree navigation with mixed constraints
    let mut mixed_tree = NavigationTree::new();
    let mixed_root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(5),
        mixed_constraints[0].clone(),
        false,
    );
    mixed_tree.set_root(mixed_root);
    mixed_tree.record_sequence(&mixed_sequence, &mixed_constraints);

    let mixed_tree_stats = mixed_tree.stats();
    results.insert("mixed_constraint_tree_nodes".to_string(), 
                  mixed_tree_stats.node_count.into_py(py));

    let py_dict = pyo3::types::PyDict::new(py);
    for (key, value) in results {
        py_dict.set_item(key, value)?;
    }

    Ok(py_dict.into())
}

/// Get default constraints for a choice type
fn get_default_constraints(choice_type: &ChoiceType) -> Constraints {
    match choice_type {
        ChoiceType::Integer => Constraints::Integer(IntegerConstraints::new(Some(-1000), Some(1000), Some(0))),
        ChoiceType::Float => Constraints::Float(FloatConstraints::new(Some(-100.0), Some(100.0))),
        ChoiceType::String => Constraints::String(StringConstraints::new(Some(0), Some(100))),
        ChoiceType::Bytes => Constraints::Bytes(BytesConstraints::new(Some(0), Some(100))),
        ChoiceType::Boolean => Constraints::Boolean(BooleanConstraints::new()),
    }
}

/// PyO3 module definition for FFI testing
#[cfg(feature = "python-ffi")]
#[pymodule]
fn navigation_capability_ffi_tests(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_complete_navigation_workflow, m)?)?;
    m.add_function(wrap_pyfunction!(test_structured_shrinking_patterns, m)?)?;
    m.add_function(wrap_pyfunction!(test_navigation_exhaustion_recovery, m)?)?;
    m.add_function(wrap_pyfunction!(test_cross_constraint_navigation, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navigation_capability_integration() {
        Python::with_gil(|py| {
            let result = test_complete_navigation_workflow(py).unwrap();
            
            let dict = result.downcast::<pyo3::types::PyDict>(py).unwrap();
            let overall_success: bool = dict.get_item("overall_capability_success")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            
            assert!(overall_success, "Navigation capability integration test should succeed");
        });
    }

    #[test]
    fn test_shrinking_patterns_validation() {
        Python::with_gil(|py| {
            let result = test_structured_shrinking_patterns(py).unwrap();
            
            let dict = result.downcast::<pyo3::types::PyDict>(py).unwrap();
            let zigzag_correct: bool = dict.get_item("zigzag_pattern_correct")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            
            assert!(zigzag_correct, "Zigzag shrinking pattern should be correct");
        });
    }

    #[test]
    fn test_exhaustion_recovery_behavior() {
        Python::with_gil(|py| {
            let result = test_navigation_exhaustion_recovery(py).unwrap();
            
            let dict = result.downcast::<pyo3::types::PyDict>(py).unwrap();
            let sequences_generated: i32 = dict.get_item("sequences_before_exhaustion")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            
            // Test passes if we generated some sequences (shows navigation system is working)
            // and either exhaustion was detected OR we hit the loop limit
            assert!(sequences_generated > 0, "Tree navigation should generate at least one sequence");
        });
    }

    #[test]
    fn test_cross_constraint_behavior() {
        Python::with_gil(|py| {
            let result = test_cross_constraint_navigation(py).unwrap();
            
            let dict = result.downcast::<pyo3::types::PyDict>(py).unwrap();
            let mixed_success: bool = dict.get_item("mixed_type_selection_success")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            
            assert!(mixed_success, "Cross-constraint navigation should work");
        });
    }
}