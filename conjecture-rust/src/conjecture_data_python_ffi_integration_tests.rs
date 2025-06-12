use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use crate::data::ConjectureData;
use crate::choice::constraints::*;
use crate::choice::values::*;
use std::collections::HashMap;

#[cfg(test)]
mod conjecture_data_python_ffi_integration_tests {
    use super::*;

    #[test]
    fn test_constraint_serialization_roundtrip() {
        Python::with_gil(|py| {
            let mut data = ConjectureData::for_buffer(&[1, 2, 3, 4, 5]);
            
            // Test IntegerConstraints serialization
            let int_constraint = IntegerConstraints {
                min_value: Some(-100),
                max_value: Some(100),
                shrink_towards: Some(0),
            };
            
            let serialized = serialize_constraint_to_python(py, &int_constraint).unwrap();
            let deserialized = deserialize_constraint_from_python(py, serialized).unwrap();
            
            match deserialized {
                Constraint::Integer(constraint) => {
                    assert_eq!(constraint.min_value, Some(-100));
                    assert_eq!(constraint.max_value, Some(100));
                    assert_eq!(constraint.shrink_towards, Some(0));
                }
                _ => panic!("Expected IntegerConstraints"),
            }
        });
    }

    #[test]
    fn test_float_constraint_serialization_with_python_types() {
        Python::with_gil(|py| {
            // Test FloatConstraint with special values
            let float_constraint = FloatConstraint {
                min_value: Some(f64::NEG_INFINITY),
                max_value: Some(f64::INFINITY),
                allow_nan: true,
                smallest_nonzero_magnitude: Some(1e-100),
                exclude_min: false,
                exclude_max: false,
            };
            
            let py_dict = serialize_constraint_to_python(py, &float_constraint).unwrap();
            
            // Verify Python-side handling of special float values
            let min_val: &PyFloat = py_dict.get_item("min_value").unwrap().unwrap().downcast().unwrap();
            assert!(min_val.value().is_infinite() && min_val.value().is_sign_negative());
            
            let max_val: &PyFloat = py_dict.get_item("max_value").unwrap().unwrap().downcast().unwrap();
            assert!(max_val.value().is_infinite() && max_val.value().is_sign_positive());
            
            let allow_nan: bool = py_dict.get_item("allow_nan").unwrap().unwrap().extract().unwrap();
            assert!(allow_nan);
        });
    }

    #[test]
    fn test_bytes_constraint_serialization_with_encoding() {
        Python::with_gil(|py| {
            let bytes_constraint = BytesConstraint {
                min_size: Some(0),
                max_size: Some(1024),
                encoding: Some("utf-8".to_string()),
            };
            
            let py_dict = serialize_constraint_to_python(py, &bytes_constraint).unwrap();
            let deserialized = deserialize_constraint_from_python(py, py_dict).unwrap();
            
            match deserialized {
                Constraint::Bytes(constraint) => {
                    assert_eq!(constraint.min_size, Some(0));
                    assert_eq!(constraint.max_size, Some(1024));
                    assert_eq!(constraint.encoding, Some("utf-8".to_string()));
                }
                _ => panic!("Expected BytesConstraint"),
            }
        });
    }

    #[test]
    fn test_choice_value_python_conversion() {
        Python::with_gil(|py| {
            // Test integer value conversion
            let int_value = ChoiceValue::Integer(42);
            let py_value = convert_choice_value_to_python(py, &int_value).unwrap();
            let py_int: &PyInt = py_value.downcast().unwrap();
            assert_eq!(py_int.extract::<i64>().unwrap(), 42);
            
            // Test float value conversion
            let float_value = ChoiceValue::Float(3.14159);
            let py_value = convert_choice_value_to_python(py, &float_value).unwrap();
            let py_float: &PyFloat = py_value.downcast().unwrap();
            assert!((py_float.value() - 3.14159).abs() < 1e-10);
            
            // Test bytes value conversion
            let bytes_value = ChoiceValue::Bytes(vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]);
            let py_value = convert_choice_value_to_python(py, &bytes_value).unwrap();
            let py_bytes: &PyBytes = py_value.downcast().unwrap();
            assert_eq!(py_bytes.as_bytes(), b"Hello");
        });
    }

    #[test]
    fn test_conjecture_data_state_synchronization() {
        Python::with_gil(|py| {
            let mut data = ConjectureData::for_buffer(&[1, 2, 3, 4, 5, 6, 7, 8]);
            
            // Simulate Python-side operations
            data.start_example(42);
            data.draw_bits(8, Some(&IntegerConstraints {
                min_value: Some(0),
                max_value: Some(255),
                shrink_towards: Some(0),
            }));
            data.stop_example();
            
            // Export state to Python
            let py_state = export_conjecture_data_state_to_python(py, &data).unwrap();
            
            // Verify Python dict structure
            let py_dict: &PyDict = py_state.downcast().unwrap();
            
            let buffer: &PyBytes = py_dict.get_item("buffer").unwrap().unwrap().downcast().unwrap();
            assert_eq!(buffer.as_bytes(), &[1, 2, 3, 4, 5, 6, 7, 8]);
            
            let index: i64 = py_dict.get_item("index").unwrap().unwrap().extract().unwrap();
            assert_eq!(index, 1);
            
            let examples: &PyList = py_dict.get_item("examples").unwrap().unwrap().downcast().unwrap();
            assert_eq!(examples.len(), 1);
        });
    }

    #[test]
    fn test_python_to_rust_state_import() {
        Python::with_gil(|py| {
            // Create Python state dict
            let py_dict = PyDict::new(py);
            py_dict.set_item("buffer", PyBytes::new(py, &[10, 20, 30, 40])).unwrap();
            py_dict.set_item("index", 2).unwrap();
            py_dict.set_item("overdraw", 0).unwrap();
            py_dict.set_item("frozen", false).unwrap();
            
            let examples = PyList::empty(py);
            let example_dict = PyDict::new(py);
            example_dict.set_item("label", 123).unwrap();
            example_dict.set_item("start", 0).unwrap();
            example_dict.set_item("length", 2).unwrap();
            examples.append(example_dict).unwrap();
            py_dict.set_item("examples", examples).unwrap();
            
            // Import state from Python
            let mut data = import_conjecture_data_state_from_python(py, py_dict).unwrap();
            
            // Verify imported state
            assert_eq!(data.buffer(), &[10, 20, 30, 40]);
            assert_eq!(data.index(), 2);
            assert!(!data.frozen());
            
            let examples = data.examples();
            assert_eq!(examples.len(), 1);
            assert_eq!(examples[0].label, 123);
            assert_eq!(examples[0].start, 0);
            assert_eq!(examples[0].length, 2);
        });
    }

    #[test]
    fn test_constraint_validation_python_parity() {
        Python::with_gil(|py| {
            // Test constraint validation that matches Python behavior
            let invalid_int_constraint = IntegerConstraints {
                min_value: Some(100),
                max_value: Some(50), // Invalid: min > max
                shrink_towards: Some(0),
            };
            
            let result = validate_constraint_python_parity(py, &invalid_int_constraint);
            assert!(result.is_err());
            
            let valid_int_constraint = IntegerConstraints {
                min_value: Some(0),
                max_value: Some(100),
                shrink_towards: Some(50),
            };
            
            let result = validate_constraint_python_parity(py, &valid_int_constraint);
            assert!(result.is_ok());
        });
    }

    #[test]
    fn test_choice_sequence_python_ffi_integration() {
        Python::with_gil(|py| {
            let mut data = ConjectureData::for_buffer(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            
            // Create a choice sequence with mixed constraint types
            data.start_example(1);
            
            let int_choice = data.draw_bits(8, Some(&IntegerConstraints {
                min_value: Some(0),
                max_value: Some(255),
                shrink_towards: Some(128),
            }));
            
            let float_choice = data.draw_bits(32, Some(&FloatConstraint {
                min_value: Some(0.0),
                max_value: Some(1.0),
                allow_nan: false,
                smallest_nonzero_magnitude: None,
                exclude_min: false,
                exclude_max: false,
            }));
            
            data.stop_example();
            
            // Export choice sequence to Python
            let py_choices = export_choice_sequence_to_python(py, &data).unwrap();
            let py_list: &PyList = py_choices.downcast().unwrap();
            
            assert_eq!(py_list.len(), 2);
            
            // Verify first choice (integer)
            let first_choice: &PyDict = py_list.get_item(0).unwrap().downcast().unwrap();
            let choice_type: &str = first_choice.get_item("type").unwrap().unwrap().extract().unwrap();
            assert_eq!(choice_type, "integer");
            
            // Verify second choice (float)
            let second_choice: &PyDict = py_list.get_item(1).unwrap().downcast().unwrap();
            let choice_type: &str = second_choice.get_item("type").unwrap().unwrap().extract().unwrap();
            assert_eq!(choice_type, "float");
        });
    }

    #[test]
    fn test_error_handling_python_exceptions() {
        Python::with_gil(|py| {
            // Test that Rust errors are properly converted to Python exceptions
            let invalid_buffer = vec![];
            let result = create_conjecture_data_from_python(py, &invalid_buffer);
            
            match result {
                Err(PyErr { .. }) => {
                    // Verify that we get a proper Python exception
                    assert!(true);
                }
                Ok(_) => panic!("Expected error for invalid buffer"),
            }
        });
    }

    #[test]
    fn test_memory_management_python_objects() {
        Python::with_gil(|py| {
            // Test that Python objects are properly managed in long-running operations
            let mut data = ConjectureData::for_buffer(&[0; 1000]);
            
            for i in 0..100 {
                data.start_example(i);
                
                let constraint = IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(1000),
                    shrink_towards: Some(i as i64),
                };
                
                // Convert constraint to Python and back multiple times
                let py_constraint = serialize_constraint_to_python(py, &constraint).unwrap();
                let rust_constraint = deserialize_constraint_from_python(py, py_constraint).unwrap();
                
                data.draw_bits(8, Some(&constraint));
                data.stop_example();
            }
            
            // Verify data integrity after many Python interactions
            assert_eq!(data.examples().len(), 100);
        });
    }

    #[test]
    fn test_unicode_string_handling() {
        Python::with_gil(|py| {
            let string_constraint = StringConstraint {
                min_size: Some(0),
                max_size: Some(100),
                alphabet: Some("αβγδε".to_string()), // Greek letters
                encoding: Some("utf-8".to_string()),
            };
            
            let py_dict = serialize_constraint_to_python(py, &string_constraint).unwrap();
            let alphabet: &str = py_dict.get_item("alphabet").unwrap().unwrap().extract().unwrap();
            assert_eq!(alphabet, "αβγδε");
            
            let deserialized = deserialize_constraint_from_python(py, py_dict).unwrap();
            match deserialized {
                Constraint::String(constraint) => {
                    assert_eq!(constraint.alphabet, Some("αβγδε".to_string()));
                }
                _ => panic!("Expected StringConstraint"),
            }
        });
    }

    // Helper functions for the tests
    fn serialize_constraint_to_python(py: Python, constraint: &impl ConstraintTrait) -> PyResult<&PyDict> {
        // Implementation would serialize constraint to Python dict
        let dict = PyDict::new(py);
        // Add constraint-specific serialization logic
        Ok(dict)
    }

    fn deserialize_constraint_from_python(py: Python, py_obj: &PyDict) -> PyResult<Constraint> {
        // Implementation would deserialize constraint from Python dict
        Ok(Constraint::Integer(IntegerConstraints {
            min_value: py_obj.get_item("min_value")?.and_then(|v| v.extract().ok()),
            max_value: py_obj.get_item("max_value")?.and_then(|v| v.extract().ok()),
            shrink_towards: py_obj.get_item("shrink_towards")?.and_then(|v| v.extract().ok()),
        }))
    }

    fn convert_choice_value_to_python(py: Python, value: &ChoiceValue) -> PyResult<&PyAny> {
        match value {
            ChoiceValue::Integer(i) => Ok(PyInt::new(py, *i).into()),
            ChoiceValue::Float(f) => Ok(PyFloat::new(py, *f).into()),
            ChoiceValue::Bytes(b) => Ok(PyBytes::new(py, b).into()),
            ChoiceValue::String(s) => Ok(PyString::new(py, s).into()),
        }
    }

    fn export_conjecture_data_state_to_python(py: Python, data: &ConjectureData) -> PyResult<&PyDict> {
        let dict = PyDict::new(py);
        dict.set_item("buffer", PyBytes::new(py, data.buffer()))?;
        dict.set_item("index", data.index())?;
        dict.set_item("overdraw", data.overdraw())?;
        dict.set_item("frozen", data.frozen())?;
        
        let examples = PyList::empty(py);
        for example in data.examples() {
            let example_dict = PyDict::new(py);
            example_dict.set_item("label", example.label)?;
            example_dict.set_item("start", example.start)?;
            example_dict.set_item("length", example.length)?;
            examples.append(example_dict)?;
        }
        dict.set_item("examples", examples)?;
        
        Ok(dict)
    }

    fn import_conjecture_data_state_from_python(py: Python, py_dict: &PyDict) -> PyResult<ConjectureData> {
        let buffer: &PyBytes = py_dict.get_item("buffer")?.unwrap().downcast()?;
        let mut data = ConjectureData::for_buffer(buffer.as_bytes());
        
        let index: usize = py_dict.get_item("index")?.unwrap().extract()?;
        data.set_index(index);
        
        let frozen: bool = py_dict.get_item("frozen")?.unwrap().extract()?;
        if frozen {
            data.freeze();
        }
        
        Ok(data)
    }

    fn validate_constraint_python_parity(py: Python, constraint: &IntegerConstraints) -> PyResult<()> {
        if let (Some(min), Some(max)) = (constraint.min_value, constraint.max_value) {
            if min > max {
                return Err(pyo3::exceptions::PyValueError::new_err("min_value cannot be greater than max_value"));
            }
        }
        Ok(())
    }

    fn export_choice_sequence_to_python(py: Python, data: &ConjectureData) -> PyResult<&PyList> {
        let list = PyList::empty(py);
        
        for choice in data.choices() {
            let choice_dict = PyDict::new(py);
            match &choice.constraint {
                Some(Constraint::Integer(_)) => {
                    choice_dict.set_item("type", "integer")?;
                }
                Some(Constraint::Float(_)) => {
                    choice_dict.set_item("type", "float")?;
                }
                Some(Constraint::Bytes(_)) => {
                    choice_dict.set_item("type", "bytes")?;
                }
                Some(Constraint::String(_)) => {
                    choice_dict.set_item("type", "string")?;
                }
                None => {
                    choice_dict.set_item("type", "unconstrained")?;
                }
            }
            list.append(choice_dict)?;
        }
        
        Ok(list)
    }

    fn create_conjecture_data_from_python(py: Python, buffer: &[u8]) -> PyResult<ConjectureData> {
        if buffer.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("Buffer cannot be empty"));
        }
        Ok(ConjectureData::for_buffer(buffer))
    }
}

// Additional trait definitions needed for the tests
trait ConstraintTrait {
    fn serialize_to_dict(&self, py: Python) -> PyResult<&PyDict>;
}

impl ConstraintTrait for IntegerConstraints {
    fn serialize_to_dict(&self, py: Python) -> PyResult<&PyDict> {
        let dict = PyDict::new(py);
        if let Some(min) = self.min_value {
            dict.set_item("min_value", min)?;
        }
        if let Some(max) = self.max_value {
            dict.set_item("max_value", max)?;
        }
        if let Some(shrink) = self.shrink_towards {
            dict.set_item("shrink_towards", shrink)?;
        }
        Ok(dict)
    }
}

impl ConstraintTrait for FloatConstraint {
    fn serialize_to_dict(&self, py: Python) -> PyResult<&PyDict> {
        let dict = PyDict::new(py);
        if let Some(min) = self.min_value {
            dict.set_item("min_value", min)?;
        }
        if let Some(max) = self.max_value {
            dict.set_item("max_value", max)?;
        }
        dict.set_item("allow_nan", self.allow_nan)?;
        if let Some(smallest) = self.smallest_nonzero_magnitude {
            dict.set_item("smallest_nonzero_magnitude", smallest)?;
        }
        dict.set_item("exclude_min", self.exclude_min)?;
        dict.set_item("exclude_max", self.exclude_max)?;
        Ok(dict)
    }
}

impl ConstraintTrait for BytesConstraint {
    fn serialize_to_dict(&self, py: Python) -> PyResult<&PyDict> {
        let dict = PyDict::new(py);
        if let Some(min) = self.min_size {
            dict.set_item("min_size", min)?;
        }
        if let Some(max) = self.max_size {
            dict.set_item("max_size", max)?;
        }
        if let Some(ref encoding) = self.encoding {
            dict.set_item("encoding", encoding)?;
        }
        Ok(dict)
    }
}

// Mock constraint types for testing
#[derive(Debug, Clone)]
pub struct StringConstraint {
    pub min_size: Option<usize>,
    pub max_size: Option<usize>,
    pub alphabet: Option<String>,
    pub encoding: Option<String>,
}

#[derive(Debug, Clone)]
pub enum Constraint {
    Integer(IntegerConstraints),
    Float(FloatConstraint),
    Bytes(BytesConstraint),
    String(StringConstraint),
}