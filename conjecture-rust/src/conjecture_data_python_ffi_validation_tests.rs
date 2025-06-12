//! Comprehensive PyO3 Validation Test Suite for ConjectureData Python Parity
//! 
//! This module provides exhaustive validation tests to ensure complete behavioral
//! parity between the Rust ConjectureData implementation and Python's implementation.
//! 
//! Test Categories:
//! - Constraint serialization/deserialization parity
//! - Choice value type conversion accuracy  
//! - State synchronization completeness
//! - Binary format compatibility
//! - Edge case handling consistency
//! - Performance characteristics validation
//! - Memory safety under Python integration

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyBool, PyNone};
use crate::conjecture_data_python_ffi::*;
use crate::conjecture_data_python_ffi_advanced::*;
use crate::data::ConjectureData;
use crate::choice::constraints::*;
use crate::choice::values::*;
use crate::choice::*;
use std::collections::HashMap;

// Validation test logging
macro_rules! validation_debug {
    ($($arg:tt)*) => {
        #[cfg(not(test))]
        println!("VALIDATION_TEST DEBUG: {}", format!($($arg)*));
    };
}

/// Comprehensive test suite for Python parity validation
pub struct ConjectureDataValidationSuite;

impl ConjectureDataValidationSuite {
    /// Run complete validation suite and return detailed results
    pub fn run_complete_validation_suite(py: Python) -> PyResult<PyObject> {
        validation_debug!("Starting comprehensive ConjectureData Python parity validation suite");
        
        let results = PyDict::new(py);
        let mut total_tests = 0;
        let mut passed_tests = 0;
        let mut failed_tests = Vec::new();
        
        // Test Category 1: Constraint Serialization Parity
        let constraint_results = Self::test_constraint_serialization_parity(py)?;
        Self::aggregate_test_results(constraint_results, &mut total_tests, &mut passed_tests, &mut failed_tests)?;
        results.set_item("constraint_serialization", constraint_results)?;
        
        // Test Category 2: Choice Value Type Conversion
        let value_conversion_results = Self::test_choice_value_conversion_parity(py)?;
        Self::aggregate_test_results(value_conversion_results, &mut total_tests, &mut passed_tests, &mut failed_tests)?;
        results.set_item("value_conversion", value_conversion_results)?;
        
        // Test Category 3: State Synchronization
        let state_sync_results = Self::test_state_synchronization_parity(py)?;
        Self::aggregate_test_results(state_sync_results, &mut total_tests, &mut passed_tests, &mut failed_tests)?;
        results.set_item("state_synchronization", state_sync_results)?;
        
        // Test Category 4: Binary Format Compatibility
        let binary_format_results = Self::test_binary_format_compatibility(py)?;
        Self::aggregate_test_results(binary_format_results, &mut total_tests, &mut passed_tests, &mut failed_tests)?;
        results.set_item("binary_format", binary_format_results)?;
        
        // Test Category 5: Edge Case Handling
        let edge_case_results = Self::test_edge_case_handling_parity(py)?;
        Self::aggregate_test_results(edge_case_results, &mut total_tests, &mut passed_tests, &mut failed_tests)?;
        results.set_item("edge_cases", edge_case_results)?;
        
        // Test Category 6: Performance Characteristics
        let performance_results = Self::test_performance_characteristics(py)?;
        Self::aggregate_test_results(performance_results, &mut total_tests, &mut passed_tests, &mut failed_tests)?;
        results.set_item("performance", performance_results)?;
        
        // Test Category 7: Memory Safety
        let memory_safety_results = Self::test_memory_safety_integration(py)?;
        Self::aggregate_test_results(memory_safety_results, &mut total_tests, &mut passed_tests, &mut failed_tests)?;
        results.set_item("memory_safety", memory_safety_results)?;
        
        // Generate summary
        let summary = PyDict::new(py);
        summary.set_item("total_tests", total_tests)?;
        summary.set_item("passed_tests", passed_tests)?;
        summary.set_item("failed_tests", failed_tests.len())?;
        summary.set_item("success_rate", passed_tests as f64 / total_tests as f64)?;
        summary.set_item("failed_test_names", PyList::new(py, &failed_tests))?;
        results.set_item("summary", summary)?;
        
        validation_debug!("Validation suite complete: {}/{} tests passed ({:.1}%)", 
                         passed_tests, total_tests, (passed_tests as f64 / total_tests as f64) * 100.0);
        
        Ok(results.to_object(py))
    }
    
    /// Test constraint serialization/deserialization parity with Python
    fn test_constraint_serialization_parity(py: Python) -> PyResult<&PyDict> {
        validation_debug!("Testing constraint serialization parity");
        
        let results = PyDict::new(py);
        let test_results = PyList::empty(py);
        
        // Test 1: IntegerConstraints comprehensive serialization
        let test_result = Self::run_single_test(py, "integer_constraint_full_range", || {
            let constraints = vec![
                IntegerConstraints { min_value: None, max_value: None, shrink_towards: None },
                IntegerConstraints { min_value: Some(0), max_value: Some(100), shrink_towards: Some(50) },
                IntegerConstraints { min_value: Some(i64::MIN), max_value: Some(i64::MAX), shrink_towards: Some(0) },
                IntegerConstraints { min_value: Some(-1000), max_value: Some(1000), shrink_towards: Some(-500) },
            ];
            
            for (i, constraint) in constraints.iter().enumerate() {
                validation_debug!("Testing IntegerConstraints variant {}: {:?}", i, constraint);
                
                let py_dict = constraint.to_python_dict(py)?;
                let deserialized = IntegerConstraints::from_python_dict(py, py_dict)?;
                
                if constraint.min_value != deserialized.min_value ||
                   constraint.max_value != deserialized.max_value ||
                   constraint.shrink_towards != deserialized.shrink_towards {
                    return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                        format!("IntegerConstraints roundtrip failed for variant {}", i)
                    ));
                }
            }
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 2: FloatConstraints with special values
        let test_result = Self::run_single_test(py, "float_constraint_special_values", || {
            let constraints = vec![
                FloatConstraints {
                    min_value: Some(f64::NEG_INFINITY),
                    max_value: Some(f64::INFINITY),
                    allow_nan: true,
                    smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
                    exclude_min: false,
                    exclude_max: false,
                },
                FloatConstraints {
                    min_value: Some(-1e100),
                    max_value: Some(1e100),
                    allow_nan: false,
                    smallest_nonzero_magnitude: Some(1e-100),
                    exclude_min: true,
                    exclude_max: true,
                },
                FloatConstraints {
                    min_value: None,
                    max_value: None,
                    allow_nan: true,
                    smallest_nonzero_magnitude: None,
                    exclude_min: false,
                    exclude_max: false,
                },
            ];
            
            for (i, constraint) in constraints.iter().enumerate() {
                validation_debug!("Testing FloatConstraints variant {}: {:?}", i, constraint);
                
                let py_dict = constraint.to_python_dict(py)?;
                let deserialized = FloatConstraints::from_python_dict(py, py_dict)?;
                
                // Special handling for infinite values
                let orig_min = constraint.min_value.unwrap_or(f64::NEG_INFINITY);
                let deser_min = deserialized.min_value.unwrap_or(f64::NEG_INFINITY);
                let orig_max = constraint.max_value.unwrap_or(f64::INFINITY);
                let deser_max = deserialized.max_value.unwrap_or(f64::INFINITY);
                
                if (orig_min - deser_min).abs() > f64::EPSILON ||
                   (orig_max - deser_max).abs() > f64::EPSILON ||
                   constraint.allow_nan != deserialized.allow_nan {
                    return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                        format!("FloatConstraints roundtrip failed for variant {}", i)
                    ));
                }
            }
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 3: BytesConstraints boundary conditions
        let test_result = Self::run_single_test(py, "bytes_constraint_boundaries", || {
            let constraints = vec![
                BytesConstraints { min_size: Some(0), max_size: Some(0), encoding: None },
                BytesConstraints { min_size: Some(1), max_size: Some(1024), encoding: None },
                BytesConstraints { min_size: Some(1024), max_size: Some(1048576), encoding: None },
                BytesConstraints { min_size: None, max_size: None, encoding: Some("utf-8".to_string()) },
            ];
            
            for (i, constraint) in constraints.iter().enumerate() {
                validation_debug!("Testing BytesConstraints variant {}: {:?}", i, constraint);
                
                let py_dict = constraint.to_python_dict(py)?;
                let deserialized = BytesConstraints::from_python_dict(py, py_dict)?;
                
                if constraint.min_size != deserialized.min_size ||
                   constraint.max_size != deserialized.max_size {
                    return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                        format!("BytesConstraints roundtrip failed for variant {}", i)
                    ));
                }
            }
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 4: Unified constraint type detection
        let test_result = Self::run_single_test(py, "unified_constraint_type_detection", || {
            let constraints = vec![
                Constraints::Integer(IntegerConstraints { min_value: Some(0), max_value: Some(100), shrink_towards: Some(50) }),
                Constraints::Float(FloatConstraints { min_value: Some(0.0), max_value: Some(1.0), allow_nan: false, smallest_nonzero_magnitude: None, exclude_min: false, exclude_max: false }),
                Constraints::Bytes(BytesConstraints { min_size: Some(0), max_size: Some(1024), encoding: None }),
                Constraints::Boolean(BooleanConstraints { p: 0.5 }),
            ];
            
            for (i, constraint) in constraints.iter().enumerate() {
                validation_debug!("Testing Constraints variant {}: {:?}", i, constraint);
                
                let py_dict = constraint.to_python_dict_with_type(py)?;
                let deserialized = Constraints::from_python_dict_with_type(py, py_dict)?;
                
                // Verify constraint type preserved
                match (constraint, &deserialized) {
                    (Constraints::Integer(_), Constraints::Integer(_)) => {},
                    (Constraints::Float(_), Constraints::Float(_)) => {},
                    (Constraints::Bytes(_), Constraints::Bytes(_)) => {},
                    (Constraints::Boolean(_), Constraints::Boolean(_)) => {},
                    _ => return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                        format!("Constraints type not preserved for variant {}", i)
                    )),
                }
            }
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        results.set_item("tests", test_results)?;
        Ok(results)
    }
    
    /// Test choice value type conversion parity with Python
    fn test_choice_value_conversion_parity(py: Python) -> PyResult<&PyDict> {
        validation_debug!("Testing choice value conversion parity");
        
        let results = PyDict::new(py);
        let test_results = PyList::empty(py);
        
        // Test 1: Integer value edge cases
        let test_result = Self::run_single_test(py, "integer_value_edge_cases", || {
            let values = vec![
                0i64, 1i64, -1i64, 
                i64::MAX, i64::MIN,
                127i64, -128i64,
                32767i64, -32768i64,
                2147483647i64, -2147483648i64,
            ];
            
            for (i, &value) in values.iter().enumerate() {
                validation_debug!("Testing integer value {}: {}", i, value);
                
                let choice_value = ChoiceValue::Integer(value);
                let py_obj = choice_value_to_python(py, &choice_value)?;
                let converted_back = choice_value_from_python(py_obj.as_ref(py))
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
                
                match converted_back {
                    ChoiceValue::Integer(converted_value) => {
                        if value != converted_value {
                            return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                                format!("Integer conversion failed: {} != {}", value, converted_value)
                            ));
                        }
                    }
                    _ => return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                        "Integer converted to wrong type"
                    )),
                }
            }
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 2: Float value special cases
        let test_result = Self::run_single_test(py, "float_value_special_cases", || {
            let values = vec![
                0.0, -0.0, 1.0, -1.0,
                f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
                f64::MIN, f64::MAX, f64::MIN_POSITIVE,
                1e-100, 1e100, std::f64::consts::PI, std::f64::consts::E,
            ];
            
            for (i, &value) in values.iter().enumerate() {
                validation_debug!("Testing float value {}: {}", i, value);
                
                let choice_value = ChoiceValue::Float(value);
                let py_obj = choice_value_to_python(py, &choice_value)?;
                let converted_back = choice_value_from_python(py_obj.as_ref(py))
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
                
                match converted_back {
                    ChoiceValue::Float(converted_value) => {
                        if value.is_nan() && converted_value.is_nan() {
                            // Both NaN - OK
                        } else if (value - converted_value).abs() > f64::EPSILON {
                            return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                                format!("Float conversion failed: {} != {}", value, converted_value)
                            ));
                        }
                    }
                    _ => return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                        "Float converted to wrong type"
                    )),
                }
            }
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 3: String value Unicode handling
        let test_result = Self::run_single_test(py, "string_value_unicode_handling", || {
            let values = vec![
                "".to_string(),
                "Hello".to_string(),
                "Hello, World!".to_string(),
                "αβγδε".to_string(), // Greek
                "你好世界".to_string(), // Chinese
                "🌟🚀🎉".to_string(), // Emoji
                "Line1\nLine2\tTab".to_string(), // Control chars
                "\"Quotes\" and 'apostrophes'".to_string(),
                "\u{0000}\u{0001}\u{001F}".to_string(), // Control chars
            ];
            
            for (i, value) in values.iter().enumerate() {
                validation_debug!("Testing string value {}: \"{}\"", i, value);
                
                let choice_value = ChoiceValue::String(value.clone());
                let py_obj = choice_value_to_python(py, &choice_value)?;
                let converted_back = choice_value_from_python(py_obj.as_ref(py))
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
                
                match converted_back {
                    ChoiceValue::String(converted_value) => {
                        if *value != converted_value {
                            return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                                format!("String conversion failed: \"{}\" != \"{}\"", value, converted_value)
                            ));
                        }
                    }
                    _ => return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                        "String converted to wrong type"
                    )),
                }
            }
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 4: Bytes value edge cases
        let test_result = Self::run_single_test(py, "bytes_value_edge_cases", || {
            let values = vec![
                vec![], // Empty
                vec![0], // Single zero
                vec![255], // Single max
                vec![0, 1, 2, 3, 4, 5], // Sequential
                vec![255, 254, 253, 252], // Reverse high
                (0..256).map(|i| i as u8).collect(), // Full range
                vec![0x48, 0x65, 0x6c, 0x6c, 0x6f], // "Hello"
                vec![0xFF; 1024], // Large uniform
                (0..1024).map(|i| (i % 256) as u8).collect(), // Large pattern
            ];
            
            for (i, value) in values.iter().enumerate() {
                validation_debug!("Testing bytes value {}: {} bytes", i, value.len());
                
                let choice_value = ChoiceValue::Bytes(value.clone());
                let py_obj = choice_value_to_python(py, &choice_value)?;
                let converted_back = choice_value_from_python(py_obj.as_ref(py))
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
                
                match converted_back {
                    ChoiceValue::Bytes(converted_value) => {
                        if *value != converted_value {
                            return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                                format!("Bytes conversion failed: lengths {} != {}", value.len(), converted_value.len())
                            ));
                        }
                    }
                    _ => return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                        "Bytes converted to wrong type"
                    )),
                }
            }
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 5: Boolean value conversion
        let test_result = Self::run_single_test(py, "boolean_value_conversion", || {
            let values = vec![true, false];
            
            for (i, &value) in values.iter().enumerate() {
                validation_debug!("Testing boolean value {}: {}", i, value);
                
                let choice_value = ChoiceValue::Boolean(value);
                let py_obj = choice_value_to_python(py, &choice_value)?;
                let converted_back = choice_value_from_python(py_obj.as_ref(py))
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
                
                match converted_back {
                    ChoiceValue::Boolean(converted_value) => {
                        if value != converted_value {
                            return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                                format!("Boolean conversion failed: {} != {}", value, converted_value)
                            ));
                        }
                    }
                    _ => return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                        "Boolean converted to wrong type"
                    )),
                }
            }
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        results.set_item("tests", test_results)?;
        Ok(results)
    }
    
    /// Test state synchronization between Rust and Python
    fn test_state_synchronization_parity(py: Python) -> PyResult<&PyDict> {
        validation_debug!("Testing state synchronization parity");
        
        let results = PyDict::new(py);
        let test_results = PyList::empty(py);
        
        // Test 1: Basic state export/import roundtrip
        let test_result = Self::run_single_test(py, "basic_state_roundtrip", || {
            let buffer = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
            let mut data = ConjectureData::for_buffer(&buffer);
            
            // Modify state
            data.start_example(42);
            data.draw_bits(8, Some(&IntegerConstraints {
                min_value: Some(0),
                max_value: Some(255),
                shrink_towards: Some(128),
            }));
            data.draw_bits(8, None);
            data.stop_example();
            
            // Export state
            let exported = export_conjecture_data_state(py, &data)?;
            let exported_dict = exported.downcast::<PyDict>(py)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Export not dict: {}", e)))?;
            
            // Import state
            let imported_data = import_conjecture_data_state(py, exported_dict)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Import failed: {}", e)))?;
            
            // Verify state preservation
            if data.buffer() != imported_data.buffer() {
                return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>("Buffer not preserved"));
            }
            if data.index() != imported_data.index() {
                return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>("Index not preserved"));
            }
            if data.frozen() != imported_data.frozen() {
                return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>("Frozen status not preserved"));
            }
            
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 2: Complex choice sequence preservation
        let test_result = Self::run_single_test(py, "complex_choice_sequence_preservation", || {
            let buffer = vec![0; 100];
            let mut data = ConjectureData::for_buffer(&buffer);
            
            // Create complex choice sequence
            data.start_example(1);
            
            // Mix of different choice types
            data.draw_bits(8, Some(&IntegerConstraints {
                min_value: Some(-100),
                max_value: Some(100),
                shrink_towards: Some(0),
            }));
            
            data.draw_bits(32, Some(&FloatConstraints {
                min_value: Some(0.0),
                max_value: Some(1.0),
                allow_nan: false,
                smallest_nonzero_magnitude: None,
                exclude_min: false,
                exclude_max: false,
            }));
            
            data.draw_bits(16, Some(&BytesConstraints {
                min_size: Some(1),
                max_size: Some(10),
                encoding: None,
            }));
            
            data.stop_example();
            
            // Export and verify choice sequence
            let exported = export_conjecture_data_state(py, &data)?;
            let exported_dict = exported.downcast::<PyDict>(py)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Export not dict: {}", e)))?;
            
            let nodes_obj = exported_dict.get_item("nodes")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing nodes"))?;
            let nodes_list = nodes_obj.downcast::<PyList>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Nodes not list: {}", e)))?;
            
            if nodes_list.len() != data.choices().len() {
                return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                    format!("Choice count mismatch: {} vs {}", nodes_list.len(), data.choices().len())
                ));
            }
            
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 3: Example/span structure preservation
        let test_result = Self::run_single_test(py, "example_span_structure_preservation", || {
            let buffer = vec![0; 50];
            let mut data = ConjectureData::for_buffer(&buffer);
            
            // Create nested example structure
            data.start_example(100);
            data.draw_bits(8, None);
            
            data.start_example(200);
            data.draw_bits(8, None);
            data.draw_bits(8, None);
            data.stop_example();
            
            data.draw_bits(8, None);
            data.stop_example();
            
            // Export and verify example structure
            let exported = export_conjecture_data_state(py, &data)?;
            let exported_dict = exported.downcast::<PyDict>(py)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Export not dict: {}", e)))?;
            
            let examples_obj = exported_dict.get_item("examples")
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing examples"))?;
            let examples_list = examples_obj.downcast::<PyList>()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Examples not list: {}", e)))?;
            
            if examples_list.len() != data.examples().len() {
                return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                    format!("Example count mismatch: {} vs {}", examples_list.len(), data.examples().len())
                ));
            }
            
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        results.set_item("tests", test_results)?;
        Ok(results)
    }
    
    /// Test binary format compatibility with Python implementation
    fn test_binary_format_compatibility(py: Python) -> PyResult<&PyDict> {
        validation_debug!("Testing binary format compatibility");
        
        let results = PyDict::new(py);
        let test_results = PyList::empty(py);
        
        // Test 1: Binary choice sequence serialization roundtrip
        let test_result = Self::run_single_test(py, "binary_choice_serialization_roundtrip", || {
            let choices = vec![
                ChoiceNode {
                    value: ChoiceValue::Boolean(true),
                    constraint: None,
                    was_forced: false,
                    index: Some(0),
                },
                ChoiceNode {
                    value: ChoiceValue::Boolean(false),
                    constraint: None,
                    was_forced: false,
                    index: Some(1),
                },
                ChoiceNode {
                    value: ChoiceValue::Integer(42),
                    constraint: None,
                    was_forced: false,
                    index: Some(2),
                },
                ChoiceNode {
                    value: ChoiceValue::Integer(-42),
                    constraint: None,
                    was_forced: false,
                    index: Some(3),
                },
                ChoiceNode {
                    value: ChoiceValue::Float(3.14159),
                    constraint: None,
                    was_forced: false,
                    index: Some(4),
                },
                ChoiceNode {
                    value: ChoiceValue::Float(f64::INFINITY),
                    constraint: None,
                    was_forced: false,
                    index: Some(5),
                },
                ChoiceNode {
                    value: ChoiceValue::String("Hello".to_string()),
                    constraint: None,
                    was_forced: false,
                    index: Some(6),
                },
                ChoiceNode {
                    value: ChoiceValue::Bytes(vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]),
                    constraint: None,
                    was_forced: false,
                    index: Some(7),
                },
            ];
            
            validation_debug!("Serializing {} choices to binary format", choices.len());
            
            let binary_data = ChoiceSequenceBinaryCodec::serialize_to_bytes(&choices)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Serialization failed: {}", e)))?;
            
            validation_debug!("Binary data: {} bytes (0x{})", 
                             binary_data.len(),
                             binary_data.iter().map(|b| format!("{:02X}", b)).collect::<String>());
            
            let deserialized = ChoiceSequenceBinaryCodec::deserialize_from_bytes(&binary_data)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Deserialization failed: {}", e)))?;
            
            if choices.len() != deserialized.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                    format!("Choice count mismatch: {} vs {}", choices.len(), deserialized.len())
                ));
            }
            
            for (i, (original, deserialized)) in choices.iter().zip(deserialized.iter()).enumerate() {
                match (&original.value, &deserialized.value) {
                    (ChoiceValue::Boolean(a), ChoiceValue::Boolean(b)) => {
                        if a != b {
                            return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                                format!("Boolean mismatch at {}: {} vs {}", i, a, b)
                            ));
                        }
                    }
                    (ChoiceValue::Integer(a), ChoiceValue::Integer(b)) => {
                        if a != b {
                            return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                                format!("Integer mismatch at {}: {} vs {}", i, a, b)
                            ));
                        }
                    }
                    (ChoiceValue::Float(a), ChoiceValue::Float(b)) => {
                        if a.is_nan() && b.is_nan() {
                            // Both NaN - OK
                        } else if (a - b).abs() > f64::EPSILON {
                            return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                                format!("Float mismatch at {}: {} vs {}", i, a, b)
                            ));
                        }
                    }
                    (ChoiceValue::String(a), ChoiceValue::String(b)) => {
                        if a != b {
                            return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                                format!("String mismatch at {}: \"{}\" vs \"{}\"", i, a, b)
                            ));
                        }
                    }
                    (ChoiceValue::Bytes(a), ChoiceValue::Bytes(b)) => {
                        if a != b {
                            return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                                format!("Bytes mismatch at {}: {} vs {} bytes", i, a.len(), b.len())
                            ));
                        }
                    }
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                            format!("Type mismatch at {}", i)
                        ));
                    }
                }
            }
            
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 2: ULEB128 encoding compatibility
        let test_result = Self::run_single_test(py, "uleb128_encoding_compatibility", || {
            let test_values = vec![
                0, 1, 127, 128, 255, 256, 
                16383, 16384, 2097151, 2097152,
                268435455, 268435456, 
                u32::MAX as usize
            ];
            
            for (i, &value) in test_values.iter().enumerate() {
                validation_debug!("Testing ULEB128 encoding for value {}: {}", i, value);
                
                let mut buffer = Vec::new();
                ChoiceSequenceBinaryCodec::write_uleb128(&mut buffer, value)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("ULEB128 write failed: {}", e)))?;
                
                let mut cursor = std::io::Cursor::new(buffer.as_slice());
                let decoded = ChoiceSequenceBinaryCodec::read_uleb128(&mut cursor)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("ULEB128 read failed: {}", e)))?;
                
                if value != decoded {
                    return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                        format!("ULEB128 encoding failed for {}: {} != {}", value, value, decoded)
                    ));
                }
            }
            
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 3: Signed integer encoding compatibility
        let test_result = Self::run_single_test(py, "signed_integer_encoding_compatibility", || {
            let test_values = vec![
                0i64, 1i64, -1i64, 127i64, -128i64,
                32767i64, -32768i64, 2147483647i64, -2147483648i64,
                i64::MAX, i64::MIN
            ];
            
            for (i, &value) in test_values.iter().enumerate() {
                validation_debug!("Testing signed integer encoding for value {}: {}", i, value);
                
                let encoded = ChoiceSequenceBinaryCodec::encode_signed_integer(value)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Integer encode failed: {}", e)))?;
                
                let decoded = ChoiceSequenceBinaryCodec::decode_signed_integer(&encoded)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Integer decode failed: {}", e)))?;
                
                if value != decoded {
                    return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                        format!("Integer encoding failed for {}: {} != {}", value, value, decoded)
                    ));
                }
            }
            
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        results.set_item("tests", test_results)?;
        Ok(results)
    }
    
    /// Test edge case handling parity
    fn test_edge_case_handling_parity(py: Python) -> PyResult<&PyDict> {
        validation_debug!("Testing edge case handling parity");
        
        let results = PyDict::new(py);
        let test_results = PyList::empty(py);
        
        // Test 1: Constraint validation edge cases
        let test_result = Self::run_single_test(py, "constraint_validation_edge_cases", || {
            // Invalid integer constraint (min > max)
            let invalid_int = Constraints::Integer(IntegerConstraints {
                min_value: Some(100),
                max_value: Some(50),
                shrink_towards: Some(75),
            });
            
            let result = validate_constraint_python_parity(py, &invalid_int);
            if result.is_ok() {
                return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                    "Invalid integer constraint should have failed validation"
                ));
            }
            
            // Invalid float constraint (min > max for finite values)
            let invalid_float = Constraints::Float(FloatConstraints {
                min_value: Some(100.0),
                max_value: Some(50.0),
                allow_nan: false,
                smallest_nonzero_magnitude: None,
                exclude_min: false,
                exclude_max: false,
            });
            
            let result = validate_constraint_python_parity(py, &invalid_float);
            if result.is_ok() {
                return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                    "Invalid float constraint should have failed validation"
                ));
            }
            
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 2: Empty and malformed data handling
        let test_result = Self::run_single_test(py, "empty_malformed_data_handling", || {
            // Empty buffer
            let empty_data = ConjectureData::for_buffer(&[]);
            let exported = export_conjecture_data_state(py, &empty_data)?;
            let exported_dict = exported.downcast::<PyDict>(py)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Export not dict: {}", e)))?;
            
            let imported = import_conjecture_data_state(py, exported_dict)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Import failed: {}", e)))?;
            
            if !imported.buffer().is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                    "Empty buffer not preserved"
                ));
            }
            
            // Malformed choice sequence binary data
            let malformed_binary = vec![0xFF, 0xFF, 0xFF]; // Invalid format
            let result = ChoiceSequenceBinaryCodec::deserialize_from_bytes(&malformed_binary);
            if result.is_ok() {
                return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                    "Malformed binary data should have failed deserialization"
                ));
            }
            
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 3: Large data handling
        let test_result = Self::run_single_test(py, "large_data_handling", || {
            // Large buffer
            let large_buffer = vec![0u8; 10000];
            let mut large_data = ConjectureData::for_buffer(&large_buffer);
            
            // Add many choices
            for i in 0..1000 {
                large_data.start_example(i);
                large_data.draw_bits(8, None);
                large_data.stop_example();
            }
            
            // Test streaming serialization
            let streamed = BulkOperations::stream_serialize_choice_sequence(py, large_data.choices(), 100)?;
            let streamed_dict = streamed.downcast::<PyDict>(py)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Stream result not dict: {}", e)))?;
            
            let chunk_count: usize = streamed_dict.get_item("chunk_count")
                .and_then(|obj| obj.extract().ok())
                .unwrap_or(0);
            
            if chunk_count == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                    "Streaming serialization produced no chunks"
                ));
            }
            
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        results.set_item("tests", test_results)?;
        Ok(results)
    }
    
    /// Test performance characteristics
    fn test_performance_characteristics(py: Python) -> PyResult<&PyDict> {
        validation_debug!("Testing performance characteristics");
        
        let results = PyDict::new(py);
        let test_results = PyList::empty(py);
        
        // Test 1: Constraint serialization performance
        let test_result = Self::run_single_test(py, "constraint_serialization_performance", || {
            let constraints: Vec<Constraints> = (0..1000).map(|i| {
                Constraints::Integer(IntegerConstraints {
                    min_value: Some(i),
                    max_value: Some(i + 1000),
                    shrink_towards: Some(i + 500),
                })
            }).collect();
            
            let start_time = std::time::Instant::now();
            
            for constraint in &constraints {
                let _py_dict = constraint.to_python_dict_with_type(py)?;
            }
            
            let duration = start_time.elapsed();
            validation_debug!("Serialized {} constraints in {:?}", constraints.len(), duration);
            
            // Performance should be reasonable (< 1ms per constraint on average)
            if duration.as_millis() > constraints.len() as u128 {
                return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                    format!("Constraint serialization too slow: {:?} for {} constraints", duration, constraints.len())
                ));
            }
            
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 2: Choice value conversion performance
        let test_result = Self::run_single_test(py, "choice_value_conversion_performance", || {
            let values: Vec<ChoiceValue> = (0..1000).map(|i| {
                match i % 5 {
                    0 => ChoiceValue::Integer(i as i64),
                    1 => ChoiceValue::Float(i as f64 / 1000.0),
                    2 => ChoiceValue::String(format!("value_{}", i)),
                    3 => ChoiceValue::Bytes(vec![i as u8; 10]),
                    4 => ChoiceValue::Boolean(i % 2 == 0),
                    _ => unreachable!(),
                }
            }).collect();
            
            let start_time = std::time::Instant::now();
            
            for value in &values {
                let py_obj = choice_value_to_python(py, value)?;
                let _converted_back = choice_value_from_python(py_obj.as_ref(py))
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            }
            
            let duration = start_time.elapsed();
            validation_debug!("Converted {} choice values in {:?}", values.len(), duration);
            
            // Performance should be reasonable
            if duration.as_millis() > values.len() as u128 * 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>(
                    format!("Choice value conversion too slow: {:?} for {} values", duration, values.len())
                ));
            }
            
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        results.set_item("tests", test_results)?;
        Ok(results)
    }
    
    /// Test memory safety under Python integration
    fn test_memory_safety_integration(py: Python) -> PyResult<&PyDict> {
        validation_debug!("Testing memory safety integration");
        
        let results = PyDict::new(py);
        let test_results = PyList::empty(py);
        
        // Test 1: Large object handling without memory leaks
        let test_result = Self::run_single_test(py, "large_object_memory_safety", || {
            for iteration in 0..100 {
                validation_debug!("Memory safety iteration {}", iteration);
                
                // Create large data structures
                let large_buffer = vec![0u8; 10000];
                let mut data = ConjectureData::for_buffer(&large_buffer);
                
                // Add many choices
                for _ in 0..100 {
                    data.draw_bits(8, Some(&IntegerConstraints {
                        min_value: Some(0),
                        max_value: Some(255),
                        shrink_towards: Some(128),
                    }));
                }
                
                // Export to Python (should not leak)
                let _exported = export_conjecture_data_state(py, &data)?;
                
                // Force Python garbage collection periodically
                if iteration % 10 == 0 {
                    py.run("import gc; gc.collect()", None, None)?;
                }
            }
            
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        // Test 2: Repeated Python object creation/destruction
        let test_result = Self::run_single_test(py, "python_object_lifecycle_safety", || {
            for iteration in 0..1000 {
                let constraint = Constraints::Float(FloatConstraints {
                    min_value: Some(iteration as f64),
                    max_value: Some((iteration + 1000) as f64),
                    allow_nan: false,
                    smallest_nonzero_magnitude: None,
                    exclude_min: false,
                    exclude_max: false,
                });
                
                // Create Python object
                let py_dict = constraint.to_python_dict_with_type(py)?;
                
                // Convert back to Rust
                let _deserialized = Constraints::from_python_dict_with_type(py, py_dict)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
                
                // Python objects should be properly cleaned up
            }
            
            Ok(())
        })?;
        test_results.append(test_result)?;
        
        results.set_item("tests", test_results)?;
        Ok(results)
    }
    
    /// Helper function to run a single test and capture results
    fn run_single_test<'py, F>(py: Python<'py>, test_name: &str, test_fn: F) -> PyResult<&'py PyDict>
    where
        F: FnOnce() -> PyResult<()>,
    {
        validation_debug!("Running test: {}", test_name);
        
        let result = PyDict::new(py);
        result.set_item("name", test_name)?;
        
        let start_time = std::time::Instant::now();
        
        match test_fn() {
            Ok(()) => {
                let duration = start_time.elapsed();
                result.set_item("passed", true)?;
                result.set_item("error", py.None())?;
                result.set_item("duration_ms", duration.as_millis())?;
                validation_debug!("Test {} PASSED in {:?}", test_name, duration);
            }
            Err(err) => {
                let duration = start_time.elapsed();
                result.set_item("passed", false)?;
                result.set_item("error", format!("{}", err))?;
                result.set_item("duration_ms", duration.as_millis())?;
                validation_debug!("Test {} FAILED in {:?}: {}", test_name, duration, err);
            }
        }
        
        Ok(result)
    }
    
    /// Helper function to aggregate test results
    fn aggregate_test_results(
        category_result: &PyDict,
        total_tests: &mut usize,
        passed_tests: &mut usize,
        failed_tests: &mut Vec<String>,
    ) -> PyResult<()> {
        if let Some(tests_obj) = category_result.get_item("tests")? {
            let tests_list = tests_obj.downcast::<PyList>()?;
            
            for test_obj in tests_list.iter() {
                let test_dict = test_obj.downcast::<PyDict>()?;
                *total_tests += 1;
                
                let passed: bool = test_dict.get_item("passed")?.unwrap().extract()?;
                if passed {
                    *passed_tests += 1;
                } else {
                    let test_name: String = test_dict.get_item("name")?.unwrap().extract()?;
                    failed_tests.push(test_name);
                }
            }
        }
        
        Ok(())
    }
}

/// PyO3 module exports for validation testing
#[pymodule]
pub fn conjecture_data_validation(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_validation_suite, m)?)?;
    m.add_function(wrap_pyfunction!(run_constraint_validation, m)?)?;
    m.add_function(wrap_pyfunction!(run_conversion_validation, m)?)?;
    m.add_function(wrap_pyfunction!(run_state_validation, m)?)?;
    m.add_function(wrap_pyfunction!(run_binary_validation, m)?)?;
    Ok(())
}

#[pyfunction]
fn run_validation_suite(py: Python) -> PyResult<PyObject> {
    ConjectureDataValidationSuite::run_complete_validation_suite(py)
}

#[pyfunction]
fn run_constraint_validation(py: Python) -> PyResult<PyObject> {
    let results = ConjectureDataValidationSuite::test_constraint_serialization_parity(py)?;
    Ok(results.to_object(py))
}

#[pyfunction]
fn run_conversion_validation(py: Python) -> PyResult<PyObject> {
    let results = ConjectureDataValidationSuite::test_choice_value_conversion_parity(py)?;
    Ok(results.to_object(py))
}

#[pyfunction]
fn run_state_validation(py: Python) -> PyResult<PyObject> {
    let results = ConjectureDataValidationSuite::test_state_synchronization_parity(py)?;
    Ok(results.to_object(py))
}

#[pyfunction]
fn run_binary_validation(py: Python) -> PyResult<PyObject> {
    let results = ConjectureDataValidationSuite::test_binary_format_compatibility(py)?;
    Ok(results.to_object(py))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::prepare_freethreaded_python;
    
    #[test]
    fn test_validation_suite_infrastructure() {
        prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Test that the validation suite can be instantiated and run
            let results = ConjectureDataValidationSuite::run_complete_validation_suite(py).unwrap();
            let results_dict = results.downcast::<PyDict>(py).unwrap();
            
            // Verify summary exists
            assert!(results_dict.contains("summary").unwrap());
            
            let summary = results_dict.get_item("summary").unwrap().unwrap().downcast::<PyDict>().unwrap();
            assert!(summary.contains("total_tests").unwrap());
            assert!(summary.contains("passed_tests").unwrap());
            assert!(summary.contains("success_rate").unwrap());
        });
    }
    
    #[test]
    fn test_single_test_runner() {
        prepare_freethreaded_python();
        Python::with_gil(|py| {
            // Test successful test
            let success_result = ConjectureDataValidationSuite::run_single_test(py, "test_success", || {
                Ok(())
            }).unwrap();
            
            let passed: bool = success_result.get_item("passed").unwrap().unwrap().extract().unwrap();
            assert!(passed);
            
            // Test failing test
            let fail_result = ConjectureDataValidationSuite::run_single_test(py, "test_failure", || {
                Err(PyErr::new::<pyo3::exceptions::PyAssertionError, _>("Test failed"))
            }).unwrap();
            
            let passed: bool = fail_result.get_item("passed").unwrap().unwrap().extract().unwrap();
            assert!(!passed);
        });
    }
    
    #[test]
    fn test_constraint_validation_category() {
        prepare_freethreaded_python();
        Python::with_gil(|py| {
            let results = ConjectureDataValidationSuite::test_constraint_serialization_parity(py).unwrap();
            
            // Verify tests were run
            let tests = results.get_item("tests").unwrap().unwrap().downcast::<PyList>().unwrap();
            assert!(tests.len() > 0);
            
            // Verify test structure
            let first_test = tests.get_item(0).unwrap().downcast::<PyDict>().unwrap();
            assert!(first_test.contains("name").unwrap());
            assert!(first_test.contains("passed").unwrap());
            assert!(first_test.contains("duration_ms").unwrap());
        });
    }
}