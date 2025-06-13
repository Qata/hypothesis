//! FFI Integration Tests for Choice Sequence Navigation System
//!
//! This module provides comprehensive PyO3 FFI integration tests that validate
//! the complete navigation capability workflow including NavigationTree, 
//! PrefixSelector, and ChoiceIndexer components.

#[cfg(all(test, feature = "python-ffi"))]
use crate::choice::{ChoiceType, ChoiceValue, Constraints, IntegerConstraints, BooleanConstraints, FloatConstraints, StringConstraints, BytesConstraints, IntervalSet};
#[cfg(all(test, feature = "python-ffi"))]
use crate::choice::navigation::{NavigationTree, NavigationChoiceNode, ChoiceSequence, PrefixSelector, ChoiceIndexer, NavigationError, NavigationResult};
#[cfg(all(test, feature = "python-ffi"))]
use pyo3::prelude::*;
#[cfg(all(test, feature = "python-ffi"))]
use std::collections::HashMap;

/// PyO3 wrapper for NavigationTree to test FFI integration
#[cfg(all(test, feature = "python-ffi"))]
#[pyclass]
struct NavigationTreeWrapper {
    tree: NavigationTree,
}

#[cfg(all(test, feature = "python-ffi"))]
#[pymethods]
impl NavigationTreeWrapper {
    #[new]
    fn new() -> Self {
        Self {
            tree: NavigationTree::new(),
        }
    }

    fn set_root_choice(&mut self, choice_type: &str, value: PyObject, py: Python) -> PyResult<()> {
        let choice_value = python_to_choice_value(choice_type, value, py)?;
        let constraints = default_constraints_for_type(choice_type);
        let choice_type_enum = choice_type_from_str(choice_type);
        
        let root = NavigationChoiceNode::new(
            choice_type_enum,
            choice_value,
            constraints,
            false,
        );
        
        self.tree.set_root(root);
        Ok(())
    }

    fn generate_novel_prefix(&mut self) -> PyResult<Vec<PyObject>> {
        let prefix = self.tree.generate_novel_prefix()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        let py_choices = Python::with_gil(|py| {
            prefix.choices.iter()
                .map(|choice| choice_value_to_python(choice, py))
                .collect::<PyResult<Vec<_>>>()
        })?;
        
        Ok(py_choices)
    }

    fn record_sequence(&mut self, choices: Vec<PyObject>, py: Python) -> PyResult<()> {
        let mut sequence = ChoiceSequence::new();
        let mut constraints_vec = Vec::new();
        
        for choice_py in choices {
            let (choice_value, choice_type) = python_to_choice_value_with_type(choice_py, py)?;
            let constraints = default_constraints_for_type(&choice_type);
            
            sequence.push(choice_value);
            constraints_vec.push(constraints);
        }
        
        self.tree.record_sequence(&sequence, &constraints_vec);
        Ok(())
    }

    fn is_exhausted(&mut self) -> bool {
        self.tree.is_exhausted()
    }

    fn get_stats(&self) -> PyResult<PyObject> {
        let stats = self.tree.stats();
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("node_count", stats.node_count)?;
            dict.set_item("max_depth", stats.max_depth)?;
            dict.set_item("cached_sequences", stats.cached_sequences)?;
            Ok(dict.into())
        })
    }
}

/// PyO3 wrapper for PrefixSelector to test FFI integration
#[pyclass]
struct PrefixSelectorWrapper {
    selector: PrefixSelector,
}

#[pymethods]
impl PrefixSelectorWrapper {
    #[new]
    fn new(prefix_choices: Vec<PyObject>, total_choices: usize, py: Python) -> PyResult<Self> {
        let mut prefix = ChoiceSequence::new();
        
        for choice_py in prefix_choices {
            let (choice_value, _) = python_to_choice_value_with_type(choice_py, py)?;
            prefix.push(choice_value);
        }
        
        Ok(Self {
            selector: PrefixSelector::new(prefix, total_choices),
        })
    }

    fn selection_order(&self, start_index: usize) -> Vec<usize> {
        self.selector.selection_order(start_index)
    }

    fn random_selection_order(&self, seed: u64) -> Vec<usize> {
        self.selector.random_selection_order(seed)
    }
}

/// PyO3 wrapper for ChoiceIndexer to test FFI integration
#[pyclass]
struct ChoiceIndexerWrapper {
    indexer: ChoiceIndexer,
}

#[pymethods]
impl ChoiceIndexerWrapper {
    #[new]
    fn new() -> Self {
        Self {
            indexer: ChoiceIndexer::new(),
        }
    }

    fn choice_to_index(&mut self, choice: PyObject, choice_type: &str, py: Python) -> PyResult<u128> {
        let choice_value = python_to_choice_value(choice_type, choice, py)?;
        let constraints = default_constraints_for_type(choice_type);
        
        self.indexer.choice_to_index(&choice_value, &constraints)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
    }

    fn index_to_choice(&mut self, index: u128, choice_type: &str) -> PyResult<PyObject> {
        let choice_type_enum = choice_type_from_str(choice_type);
        let constraints = default_constraints_for_type(choice_type);
        
        let choice_value = self.indexer.index_to_choice(index, choice_type_enum, &constraints)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
        
        Python::with_gil(|py| choice_value_to_python(&choice_value, py))
    }
}

/// FFI Test Functions

#[pyfunction]
fn test_navigation_tree_workflow(py: Python) -> PyResult<PyObject> {
    let mut tree = NavigationTree::new();
    
    // Create a root node with integer constraints
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(10), Some(5)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(5),
        constraints.clone(),
        false,
    );
    tree.set_root(root);
    
    // Generate a novel prefix
    let prefix_result = tree.generate_novel_prefix();
    let success = prefix_result.is_ok();
    let prefix_length = if let Ok(ref prefix) = prefix_result {
        prefix.length
    } else {
        0
    };
    
    // Record a sequence
    let mut sequence = ChoiceSequence::new();
    sequence.push(ChoiceValue::Integer(5));
    sequence.push(ChoiceValue::Integer(3));
    tree.record_sequence(&sequence, &[constraints.clone(), constraints.clone()]);
    
    // Get tree statistics
    let stats = tree.stats();
    
    // Return results as Python dict
    let result = pyo3::types::PyDict::new(py);
    result.set_item("prefix_generation_success", success)?;
    result.set_item("prefix_length", prefix_length)?;
    result.set_item("tree_node_count", stats.node_count)?;
    result.set_item("tree_max_depth", stats.max_depth)?;
    result.set_item("cached_sequences", stats.cached_sequences)?;
    result.set_item("is_exhausted", tree.is_exhausted())?;
    
    Ok(result.into())
}

#[pyfunction]
fn test_prefix_selector_strategies(py: Python) -> PyResult<PyObject> {
    // Create prefix with integer choices
    let prefix = ChoiceSequence::from_choices(vec![
        ChoiceValue::Integer(42),
        ChoiceValue::Boolean(true),
    ]);
    
    let selector = PrefixSelector::new(prefix, 10);
    
    // Test deterministic selection order
    let order_from_start = selector.selection_order(0);
    let order_from_middle = selector.selection_order(5);
    let order_from_end = selector.selection_order(9);
    
    // Test random selection order
    let random_order_1 = selector.random_selection_order(12345);
    let random_order_2 = selector.random_selection_order(67890);
    
    // Verify orders have correct length
    let all_correct_length = 
        order_from_start.len() == 10 &&
        order_from_middle.len() == 10 &&
        order_from_end.len() == 10 &&
        random_order_1.len() == 10 &&
        random_order_2.len() == 10;
    
    // Verify orders contain all indices 0-9
    let contains_all_indices = |order: &[usize]| -> bool {
        let mut found = vec![false; 10];
        for &i in order {
            if i < 10 {
                found[i] = true;
            }
        }
        found.iter().all(|&x| x)
    };
    
    let all_complete = 
        contains_all_indices(&order_from_start) &&
        contains_all_indices(&order_from_middle) &&
        contains_all_indices(&order_from_end) &&
        contains_all_indices(&random_order_1) &&
        contains_all_indices(&random_order_2);
    
    // Return results
    let result = pyo3::types::PyDict::new(py);
    result.set_item("all_correct_length", all_correct_length)?;
    result.set_item("all_complete", all_complete)?;
    result.set_item("order_from_start", order_from_start)?;
    result.set_item("order_from_middle", order_from_middle)?;
    result.set_item("order_from_end", order_from_end)?;
    result.set_item("random_order_1", random_order_1)?;
    result.set_item("random_order_2", random_order_2)?;
    result.set_item("random_orders_different", random_order_1 != random_order_2)?;
    
    Ok(result.into())
}

#[pyfunction]
fn test_choice_indexer_bidirectional(py: Python) -> PyResult<PyObject> {
    let mut indexer = ChoiceIndexer::new();
    let mut test_results = Vec::new();
    
    // Test integer choices
    let integer_constraints = Constraints::Integer(IntegerConstraints::new(None, None, Some(0)));
    let integer_test_values = vec![0, 1, -1, 2, -2, 42, -100];
    
    for value in integer_test_values {
        let choice = ChoiceValue::Integer(value);
        let index = indexer.choice_to_index(&choice, &integer_constraints)?;
        let recovered = indexer.index_to_choice(index, ChoiceType::Integer, &integer_constraints)?;
        
        let success = choice == recovered;
        test_results.push((format!("integer_{}", value), success, index));
    }
    
    // Test boolean choices
    let boolean_constraints = Constraints::Boolean(BooleanConstraints::new());
    let boolean_test_values = vec![true, false];
    
    for value in boolean_test_values {
        let choice = ChoiceValue::Boolean(value);
        let index = indexer.choice_to_index(&choice, &boolean_constraints)?;
        let recovered = indexer.index_to_choice(index, ChoiceType::Boolean, &boolean_constraints)?;
        
        let success = choice == recovered;
        test_results.push((format!("boolean_{}", value), success, index));
    }
    
    // Test float choices
    let float_constraints = Constraints::Float(FloatConstraints::default());
    let float_test_values = vec![0.0, 1.0, -1.0, 3.14159, -2.718];
    
    for value in float_test_values {
        let choice = ChoiceValue::Float(value);
        let index = indexer.choice_to_index(&choice, &float_constraints)?;
        let recovered = indexer.index_to_choice(index, ChoiceType::Float, &float_constraints)?;
        
        let success = if let ChoiceValue::Float(recovered_val) = recovered {
            (value - recovered_val).abs() < 1e-10 || (value.is_nan() && recovered_val.is_nan())
        } else {
            false
        };
        test_results.push((format!("float_{}", value), success, index));
    }
    
    // Test string choices
    let string_constraints = Constraints::String(StringConstraints {
        min_size: 0,
        max_size: 10,
        intervals: IntervalSet {
            intervals: vec![(b'a' as u32, b'z' as u32)],
        },
    });
    let string_test_values = vec!["", "a", "hello", "test"];
    
    for value in string_test_values {
        let choice = ChoiceValue::String(value.to_string());
        let index = indexer.choice_to_index(&choice, &string_constraints)?;
        let recovered = indexer.index_to_choice(index, ChoiceType::String, &string_constraints)?;
        
        let success = choice == recovered;
        test_results.push((format!("string_{}", value), success, index));
    }
    
    // Count successful roundtrips
    let total_tests = test_results.len();
    let successful_tests = test_results.iter().filter(|(_, success, _)| *success).count();
    
    // Return results
    let result = pyo3::types::PyDict::new(py);
    result.set_item("total_tests", total_tests)?;
    result.set_item("successful_tests", successful_tests)?;
    result.set_item("success_rate", successful_tests as f64 / total_tests as f64)?;
    result.set_item("all_successful", successful_tests == total_tests)?;
    
    // Add detailed results
    let details = pyo3::types::PyList::new(py, 
        test_results.iter().map(|(name, success, index)| {
            let detail = pyo3::types::PyDict::new(py);
            detail.set_item("test_name", name).unwrap();
            detail.set_item("success", *success).unwrap();
            detail.set_item("index", index.to_string()).unwrap(); // Convert u128 to string for Python
            detail
        })
    );
    result.set_item("test_details", details)?;
    
    Ok(result.into())
}

#[pyfunction]
fn test_complete_navigation_workflow(py: Python) -> PyResult<PyObject> {
    // Create a complete navigation workflow test
    let mut tree = NavigationTree::new();
    let mut indexer = ChoiceIndexer::new();
    
    // Setup: Create root node
    let integer_constraints = Constraints::Integer(IntegerConstraints::new(Some(-5), Some(5), Some(0)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(0),
        integer_constraints.clone(),
        false,
    );
    tree.set_root(root);
    
    // Step 1: Generate novel prefixes
    let mut generated_prefixes = Vec::new();
    let mut generation_count = 0;
    
    while generation_count < 5 {
        match tree.generate_novel_prefix() {
            Ok(prefix) => {
                generated_prefixes.push(prefix.clone());
                generation_count += 1;
                
                // Record the prefix to build the tree
                let constraints_vec = vec![integer_constraints.clone(); prefix.length];
                tree.record_sequence(&prefix, &constraints_vec);
            }
            Err(NavigationError::TreeExhausted) => break,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Navigation error: {}", e))),
        }
    }
    
    // Step 2: Test prefix selector with generated choices
    if !generated_prefixes.is_empty() {
        let first_prefix = &generated_prefixes[0];
        let selector = PrefixSelector::new(first_prefix.clone(), 10);
        let selection_order = selector.selection_order(2);
        
        // Verify selection order properties
        let order_valid = selection_order.len() == 10 && 
            selection_order.iter().all(|&i| i < 10);
        
        // Step 3: Test choice indexing with generated choices
        let mut indexing_successful = true;
        let mut total_indexed = 0;
        
        for prefix in &generated_prefixes {
            for choice in &prefix.choices {
                if let Ok(index) = indexer.choice_to_index(choice, &integer_constraints) {
                    if let Ok(recovered) = indexer.index_to_choice(index, ChoiceType::Integer, &integer_constraints) {
                        if choice != &recovered {
                            indexing_successful = false;
                        }
                        total_indexed += 1;
                    } else {
                        indexing_successful = false;
                    }
                } else {
                    indexing_successful = false;
                }
            }
        }
        
        // Get final tree stats
        let final_stats = tree.stats();
        
        // Return comprehensive results
        let result = pyo3::types::PyDict::new(py);
        result.set_item("prefixes_generated", generated_prefixes.len())?;
        result.set_item("tree_node_count", final_stats.node_count)?;
        result.set_item("tree_max_depth", final_stats.max_depth)?;
        result.set_item("selection_order_valid", order_valid)?;
        result.set_item("indexing_successful", indexing_successful)?;
        result.set_item("total_choices_indexed", total_indexed)?;
        result.set_item("tree_exhausted", tree.is_exhausted())?;
        result.set_item("workflow_complete", true)?;
        
        Ok(result.into())
    } else {
        let result = pyo3::types::PyDict::new(py);
        result.set_item("prefixes_generated", 0)?;
        result.set_item("workflow_complete", false)?;
        result.set_item("error", "No prefixes could be generated")?;
        Ok(result.into())
    }
}

/// Integration Test for Navigation Capability Errors
#[pyfunction]
fn test_navigation_error_handling(py: Python) -> PyResult<PyObject> {
    let mut results = HashMap::new();
    
    // Test 1: Tree exhaustion
    let mut empty_tree = NavigationTree::new();
    let exhaustion_result = empty_tree.generate_novel_prefix();
    results.insert("tree_exhaustion_handled", exhaustion_result.is_err());
    
    // Test 2: Invalid choice indexing
    let mut indexer = ChoiceIndexer::new();
    let invalid_constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(5), Some(0)));
    let out_of_bounds_choice = ChoiceValue::Integer(100);
    let index_result = indexer.choice_to_index(&out_of_bounds_choice, &invalid_constraints);
    results.insert("invalid_indexing_handled", index_result.is_ok()); // Should still work but may clamp
    
    // Test 3: Inconsistent constraints
    let boolean_choice = ChoiceValue::Boolean(true);
    let integer_constraints = Constraints::Integer(IntegerConstraints::new(None, None, Some(0)));
    let constraint_mismatch = indexer.choice_to_index(&boolean_choice, &integer_constraints);
    results.insert("constraint_mismatch_handled", constraint_mismatch.is_ok()); // Should handle gracefully
    
    // Convert results to Python dict
    let result = pyo3::types::PyDict::new(py);
    for (key, value) in results {
        result.set_item(key, value)?;
    }
    
    Ok(result.into())
}

/// Integration Test for Navigation Performance
#[pyfunction]
fn test_navigation_performance(py: Python) -> PyResult<PyObject> {
    use std::time::Instant;
    
    let start = Instant::now();
    
    // Performance test: Generate many novel prefixes
    let mut tree = NavigationTree::new();
    let constraints = Constraints::Integer(IntegerConstraints::new(Some(0), Some(100), Some(50)));
    let root = NavigationChoiceNode::new(
        ChoiceType::Integer,
        ChoiceValue::Integer(50),
        constraints.clone(),
        false,
    );
    tree.set_root(root);
    
    let mut generation_count = 0;
    let max_generations = 50;
    
    while generation_count < max_generations {
        match tree.generate_novel_prefix() {
            Ok(prefix) => {
                generation_count += 1;
                let constraints_vec = vec![constraints.clone(); prefix.length];
                tree.record_sequence(&prefix, &constraints_vec);
            }
            Err(_) => break,
        }
    }
    
    let generation_time = start.elapsed();
    
    // Performance test: Index many choices
    let indexing_start = Instant::now();
    let mut indexer = ChoiceIndexer::new();
    let mut indexing_count = 0;
    
    for i in 0..1000 {
        let choice = ChoiceValue::Integer(i % 101); // 0-100
        if indexer.choice_to_index(&choice, &constraints).is_ok() {
            indexing_count += 1;
        }
    }
    
    let indexing_time = indexing_start.elapsed();
    
    // Return performance metrics
    let result = pyo3::types::PyDict::new(py);
    result.set_item("generations_completed", generation_count)?;
    result.set_item("generation_time_ms", generation_time.as_millis() as u64)?;
    result.set_item("indexing_operations", indexing_count)?;
    result.set_item("indexing_time_ms", indexing_time.as_millis() as u64)?;
    result.set_item("generations_per_second", generation_count as f64 / generation_time.as_secs_f64())?;
    result.set_item("indexing_ops_per_second", indexing_count as f64 / indexing_time.as_secs_f64())?;
    
    Ok(result.into())
}

// Helper functions for Python/Rust conversion

fn python_to_choice_value(choice_type: &str, value: PyObject, py: Python) -> PyResult<ChoiceValue> {
    match choice_type {
        "integer" => {
            let int_val: i128 = value.extract(py)?;
            Ok(ChoiceValue::Integer(int_val))
        }
        "boolean" => {
            let bool_val: bool = value.extract(py)?;
            Ok(ChoiceValue::Boolean(bool_val))
        }
        "float" => {
            let float_val: f64 = value.extract(py)?;
            Ok(ChoiceValue::Float(float_val))
        }
        "string" => {
            let string_val: String = value.extract(py)?;
            Ok(ChoiceValue::String(string_val))
        }
        "bytes" => {
            let bytes_val: Vec<u8> = value.extract(py)?;
            Ok(ChoiceValue::Bytes(bytes_val))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unknown choice type")),
    }
}

fn python_to_choice_value_with_type(value: PyObject, py: Python) -> PyResult<(ChoiceValue, String)> {
    // Try to infer type from Python object
    if value.is_instance_of::<pyo3::types::PyInt>(py) {
        let int_val: i128 = value.extract(py)?;
        Ok((ChoiceValue::Integer(int_val), "integer".to_string()))
    } else if value.is_instance_of::<pyo3::types::PyBool>(py) {
        let bool_val: bool = value.extract(py)?;
        Ok((ChoiceValue::Boolean(bool_val), "boolean".to_string()))
    } else if value.is_instance_of::<pyo3::types::PyFloat>(py) {
        let float_val: f64 = value.extract(py)?;
        Ok((ChoiceValue::Float(float_val), "float".to_string()))
    } else if value.is_instance_of::<pyo3::types::PyString>(py) {
        let string_val: String = value.extract(py)?;
        Ok((ChoiceValue::String(string_val), "string".to_string()))
    } else if value.is_instance_of::<pyo3::types::PyBytes>(py) {
        let bytes_val: Vec<u8> = value.extract(py)?;
        Ok((ChoiceValue::Bytes(bytes_val), "bytes".to_string()))
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Cannot infer choice type from Python object"))
    }
}

fn choice_value_to_python(choice: &ChoiceValue, py: Python) -> PyResult<PyObject> {
    match choice {
        ChoiceValue::Integer(val) => Ok(val.to_object(py)),
        ChoiceValue::Boolean(val) => Ok(val.to_object(py)),
        ChoiceValue::Float(val) => Ok(val.to_object(py)),
        ChoiceValue::String(val) => Ok(val.to_object(py)),
        ChoiceValue::Bytes(val) => Ok(val.to_object(py)),
    }
}

fn choice_type_from_str(choice_type: &str) -> ChoiceType {
    match choice_type {
        "integer" => ChoiceType::Integer,
        "boolean" => ChoiceType::Boolean,
        "float" => ChoiceType::Float,
        "string" => ChoiceType::String,
        "bytes" => ChoiceType::Bytes,
        _ => ChoiceType::Integer, // Default fallback
    }
}

fn default_constraints_for_type(choice_type: &str) -> Constraints {
    match choice_type {
        "integer" => Constraints::Integer(IntegerConstraints::new(None, None, Some(0))),
        "boolean" => Constraints::Boolean(BooleanConstraints::new()),
        "float" => Constraints::Float(FloatConstraints::default()),
        "string" => Constraints::String(StringConstraints {
            min_size: 0,
            max_size: 100,
            intervals: IntervalSet {
                intervals: vec![(b'a' as u32, b'z' as u32)],
            },
        }),
        "bytes" => Constraints::Bytes(BytesConstraints {
            min_size: 0,
            max_size: 100,
        }),
        _ => Constraints::Integer(IntegerConstraints::new(None, None, Some(0))),
    }
}

/// PyO3 module for FFI integration tests
#[pymodule]
fn navigation_ffi_tests(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<NavigationTreeWrapper>()?;
    m.add_class::<PrefixSelectorWrapper>()?;
    m.add_class::<ChoiceIndexerWrapper>()?;
    m.add_function(wrap_pyfunction!(test_navigation_tree_workflow, m)?)?;
    m.add_function(wrap_pyfunction!(test_prefix_selector_strategies, m)?)?;
    m.add_function(wrap_pyfunction!(test_choice_indexer_bidirectional, m)?)?;
    m.add_function(wrap_pyfunction!(test_complete_navigation_workflow, m)?)?;
    m.add_function(wrap_pyfunction!(test_navigation_error_handling, m)?)?;
    m.add_function(wrap_pyfunction!(test_navigation_performance, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navigation_tree_wrapper_creation() {
        let wrapper = NavigationTreeWrapper::new();
        assert!(!wrapper.tree.is_exhausted()); // Should work with empty tree initially
    }

    #[test]
    fn test_prefix_selector_wrapper_basic() {
        Python::with_gil(|py| {
            let choices = vec![42i128.to_object(py), true.to_object(py)];
            let wrapper = PrefixSelectorWrapper::new(choices, 5, py).unwrap();
            let order = wrapper.selection_order(2);
            assert_eq!(order.len(), 5);
            assert!(order.contains(&2)); // Should include start index
        });
    }

    #[test]
    fn test_choice_indexer_wrapper_basic() {
        Python::with_gil(|py| {
            let mut wrapper = ChoiceIndexerWrapper::new();
            let choice = 42i128.to_object(py);
            let index = wrapper.choice_to_index(choice, "integer", py).unwrap();
            let recovered = wrapper.index_to_choice(index, "integer").unwrap();
            let recovered_val: i128 = recovered.extract(py).unwrap();
            assert_eq!(recovered_val, 42);
        });
    }

    #[test]
    fn test_python_conversion_helpers() {
        Python::with_gil(|py| {
            // Test integer conversion
            let int_obj = 42i128.to_object(py);
            let choice = python_to_choice_value("integer", int_obj, py).unwrap();
            assert_eq!(choice, ChoiceValue::Integer(42));
            
            // Test boolean conversion  
            let bool_obj = true.to_object(py);
            let choice = python_to_choice_value("boolean", bool_obj, py).unwrap();
            assert_eq!(choice, ChoiceValue::Boolean(true));
            
            // Test float conversion
            let float_obj = 3.14.to_object(py);
            let choice = python_to_choice_value("float", float_obj, py).unwrap();
            assert_eq!(choice, ChoiceValue::Float(3.14));
        });
    }

    #[test]
    fn test_choice_value_to_python_conversion() {
        Python::with_gil(|py| {
            // Test integer conversion back to Python
            let choice = ChoiceValue::Integer(42);
            let py_obj = choice_value_to_python(&choice, py).unwrap();
            let recovered: i128 = py_obj.extract(py).unwrap();
            assert_eq!(recovered, 42);
            
            // Test string conversion back to Python
            let choice = ChoiceValue::String("hello".to_string());
            let py_obj = choice_value_to_python(&choice, py).unwrap();
            let recovered: String = py_obj.extract(py).unwrap();
            assert_eq!(recovered, "hello");
        });
    }

    #[test]
    fn test_default_constraints_generation() {
        let int_constraints = default_constraints_for_type("integer");
        if let Constraints::Integer(ic) = int_constraints {
            assert_eq!(ic.shrink_towards, Some(0));
        } else {
            panic!("Expected integer constraints");
        }
        
        let bool_constraints = default_constraints_for_type("boolean");
        if let Constraints::Boolean(bc) = bool_constraints {
            assert_eq!(bc.p, 0.5); // Default probability
        } else {
            panic!("Expected boolean constraints");
        }
    }

    #[test]
    fn test_choice_type_conversion() {
        assert_eq!(choice_type_from_str("integer"), ChoiceType::Integer);
        assert_eq!(choice_type_from_str("boolean"), ChoiceType::Boolean);
        assert_eq!(choice_type_from_str("float"), ChoiceType::Float);
        assert_eq!(choice_type_from_str("string"), ChoiceType::String);
        assert_eq!(choice_type_from_str("bytes"), ChoiceType::Bytes);
        assert_eq!(choice_type_from_str("unknown"), ChoiceType::Integer); // Default fallback
    }
}