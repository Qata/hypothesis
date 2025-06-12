//! Comprehensive tests for ConjectureData Python FFI Integration Layer capability
//! Tests constraint serialization and type conversion between Rust and Python
//! with focus on addressing type system mismatches and compilation errors

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyAny, PyFloat, PyInt, PyString, PyBytes, PyBool};
use std::collections::HashMap;
use crate::choice::{
    constraints::{IntegerConstraints, FloatConstraints, BytesConstraints, StringConstraints, BooleanConstraints},
    values::ChoiceValue,
    node::ChoiceNode,
};
use crate::conjecture_data_python_ffi::{
    FfiError, UnifiedConstraint, ConstraintPythonSerializable, ConstraintPythonDeserializable,
    choice_value_to_python, choice_value_from_python, export_conjecture_data_state,
    import_conjecture_data_state, ChoiceSequenceBinaryCodec,
};
use crate::data::ConjectureData;

/// Comprehensive test suite for ConjectureData Python FFI Integration Layer
pub struct ConjectureDataPythonFfiCapabilityTests;

impl ConjectureDataPythonFfiCapabilityTests {
    /// Test constraint serialization type consistency with Python parity
    pub fn test_constraint_type_consistency_capability() -> PyResult<()> {
        Python::with_gil(|py| {
            // Test IntegerConstraints vs IntegerConstraintss naming consistency
            let integer_constraint = IntegerConstraints {
                min_value: Some(-100),
                max_value: Some(100),
                shrink_towards: Some(0),
                weights: None, // Address type mismatch
            };
            
            let py_dict = integer_constraint.to_python_dict(py)?;
            
            // Verify Python dict structure matches expected TypedDict
            assert!(py_dict.contains("min_value")?);
            assert!(py_dict.contains("max_value")?);
            assert!(py_dict.contains("shrink_towards")?);
            
            // Test roundtrip conversion
            let reconstructed = IntegerConstraints::from_python_dict(py, py_dict)?;
            assert_eq!(integer_constraint.min_value, reconstructed.min_value);
            assert_eq!(integer_constraint.max_value, reconstructed.max_value);
            assert_eq!(integer_constraint.shrink_towards, reconstructed.shrink_towards);
            
            Ok(())
        })
    }

    /// Test float constraint special value handling with proper serialization
    pub fn test_float_constraint_special_values_capability() -> PyResult<()> {
        Python::with_gil(|py| {
            // Test infinity handling consistency
            let float_constraint_inf = FloatConstraints {
                min_value: None, // Represents negative infinity
                max_value: None, // Represents positive infinity
                allow_nan: true,
                allow_infinity: true,
                smallest_nonzero_magnitude: Some(1e-100),
                width: 64,
            };
            
            let py_dict = float_constraint_inf.to_python_dict(py)?;
            
            // Verify infinity serialization matches Python expectations
            let min_val = py_dict.get_item("min_value")?;
            let max_val = py_dict.get_item("max_value")?;
            
            // Python should see finite values representing infinity bounds
            if let Some(min_py) = min_val {
                let min_float: f64 = min_py.extract()?;
                assert!(min_float.is_finite() || min_float == f64::NEG_INFINITY);
            }
            
            if let Some(max_py) = max_val {
                let max_float: f64 = max_py.extract()?;
                assert!(max_float.is_finite() || max_float == f64::INFINITY);
            }
            
            // Test NaN handling
            let float_constraint_nan = FloatConstraints {
                min_value: Some(f64::NAN),
                max_value: Some(f64::NAN),
                allow_nan: true,
                allow_infinity: false,
                smallest_nonzero_magnitude: None,
                width: 32,
            };
            
            let py_dict_nan = float_constraint_nan.to_python_dict(py)?;
            let reconstructed_nan = FloatConstraints::from_python_dict(py, py_dict_nan)?;
            
            // Verify NaN is properly handled in roundtrip
            assert!(reconstructed_nan.allow_nan);
            
            Ok(())
        })
    }

    /// Test unified constraint enum serialization capability
    pub fn test_unified_constraint_serialization_capability() -> PyResult<()> {
        Python::with_gil(|py| {
            let constraints = vec![
                UnifiedConstraint::Integer(IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(255),
                    shrink_towards: Some(0),
                    weights: None,
                }),
                UnifiedConstraint::Float(FloatConstraints {
                    min_value: Some(0.0),
                    max_value: Some(1.0),
                    allow_nan: false,
                    allow_infinity: false,
                    smallest_nonzero_magnitude: Some(1e-10),
                    width: 64,
                }),
                UnifiedConstraint::Bytes(BytesConstraints {
                    min_size: Some(0),
                    max_size: Some(1024),
                    encoding: Some("utf-8".to_string()),
                }),
                UnifiedConstraint::String(StringConstraints {
                    intervals: Some("printable".to_string()),
                    min_size: Some(0),
                    max_size: Some(100),
                }),
                UnifiedConstraint::Boolean(BooleanConstraints {
                    p: 0.5,
                }),
            ];
            
            // Test each constraint type serialization
            for constraint in &constraints {
                let py_dict = match constraint {
                    UnifiedConstraint::Integer(c) => c.to_python_dict(py)?,
                    UnifiedConstraint::Float(c) => c.to_python_dict(py)?,
                    UnifiedConstraint::Bytes(c) => c.to_python_dict(py)?,
                    UnifiedConstraint::String(c) => c.to_python_dict(py)?,
                    UnifiedConstraint::Boolean(c) => c.to_python_dict(py)?,
                };
                
                // Verify constraint type is properly identified
                assert!(py_dict.contains("constraint_type")?);
                let constraint_type: String = py_dict.get_item("constraint_type").unwrap().extract()?;
                
                match constraint {
                    UnifiedConstraint::Integer(_) => assert_eq!(constraint_type, "integer"),
                    UnifiedConstraint::Float(_) => assert_eq!(constraint_type, "float"),
                    UnifiedConstraint::Bytes(_) => assert_eq!(constraint_type, "bytes"),
                    UnifiedConstraint::String(_) => assert_eq!(constraint_type, "string"),
                    UnifiedConstraint::Boolean(_) => assert_eq!(constraint_type, "boolean"),
                }
            }
            
            Ok(())
        })
    }

    /// Test choice value conversion with proper lifetime management
    pub fn test_choice_value_conversion_capability() -> PyResult<()> {
        Python::with_gil(|py| {
            let test_values = vec![
                ChoiceValue::Integer(42),
                ChoiceValue::Integer(-1000),
                ChoiceValue::Integer(i64::MAX),
                ChoiceValue::Integer(i64::MIN),
                ChoiceValue::Float(3.14159),
                ChoiceValue::Float(f64::INFINITY),
                ChoiceValue::Float(f64::NEG_INFINITY),
                ChoiceValue::Float(f64::NAN),
                ChoiceValue::String("Hello, 世界!".to_string()),
                ChoiceValue::String("".to_string()),
                ChoiceValue::String("🦀".to_string()),
                ChoiceValue::Bytes(vec![0, 1, 2, 255]),
                ChoiceValue::Bytes(vec![]),
                ChoiceValue::Boolean(true),
                ChoiceValue::Boolean(false),
            ];
            
            for value in &test_values {
                // Test Rust to Python conversion
                let py_obj = choice_value_to_python(py, value)?;
                
                // Test Python to Rust conversion
                let reconstructed = choice_value_from_python(py_obj.as_ref(py))?;
                
                // Verify roundtrip consistency
                match (value, &reconstructed) {
                    (ChoiceValue::Integer(a), ChoiceValue::Integer(b)) => assert_eq!(a, b),
                    (ChoiceValue::Float(a), ChoiceValue::Float(b)) => {
                        if a.is_nan() && b.is_nan() {
                            // NaN != NaN, so we check both are NaN
                            assert!(true);
                        } else {
                            assert_eq!(a, b);
                        }
                    },
                    (ChoiceValue::String(a), ChoiceValue::String(b)) => assert_eq!(a, b),
                    (ChoiceValue::Bytes(a), ChoiceValue::Bytes(b)) => assert_eq!(a, b),
                    (ChoiceValue::Boolean(a), ChoiceValue::Boolean(b)) => assert_eq!(a, b),
                    _ => panic!("Type mismatch in roundtrip conversion"),
                }
            }
            
            Ok(())
        })
    }

    /// Test ConjectureData state export/import capability
    pub fn test_conjecture_data_state_synchronization_capability() -> PyResult<()> {
        Python::with_gil(|py| {
            // Create a ConjectureData instance with complex state
            let mut data = ConjectureData::for_buffer(vec![1, 2, 3, 4, 5, 6, 7, 8]);
            
            // Add some choices to create state
            data.draw_integer(Some(0), Some(100), Some(50));
            data.draw_float(Some(0.0), Some(1.0), false, false, Some(1e-10), 64);
            data.draw_bytes(Some(4), Some(4));
            
            // Export state to Python
            let py_state = export_conjecture_data_state(py, &data)?;
            let py_dict = py_state.downcast::<PyDict>(py)?;
            
            // Verify exported state structure
            assert!(py_dict.contains("buffer")?);
            assert!(py_dict.contains("index")?);
            assert!(py_dict.contains("choices")?);
            assert!(py_dict.contains("status")?);
            
            // Import state back to Rust
            let reconstructed_data = import_conjecture_data_state(py, py_dict)?;
            
            // Verify state consistency
            assert_eq!(data.buffer(), reconstructed_data.buffer());
            assert_eq!(data.index(), reconstructed_data.index());
            
            Ok(())
        })
    }

    /// Test binary codec serialization capability
    pub fn test_binary_codec_serialization_capability() -> PyResult<()> {
        Python::with_gil(|py| {
            let choice_nodes = vec![
                ChoiceNode {
                    value: ChoiceValue::Integer(42),
                    constraint: UnifiedConstraint::Integer(IntegerConstraints {
                        min_value: Some(0),
                        max_value: Some(100),
                        shrink_towards: Some(0),
                        weights: None,
                    }),
                    index: 0,
                    forced: false,
                },
                ChoiceNode {
                    value: ChoiceValue::Float(3.14159),
                    constraint: UnifiedConstraint::Float(FloatConstraints {
                        min_value: Some(0.0),
                        max_value: Some(10.0),
                        allow_nan: false,
                        allow_infinity: false,
                        smallest_nonzero_magnitude: Some(1e-10),
                        width: 64,
                    }),
                    index: 1,
                    forced: false,
                },
                ChoiceNode {
                    value: ChoiceValue::String("test".to_string()),
                    constraint: UnifiedConstraint::String(StringConstraints {
                        intervals: Some("ascii".to_string()),
                        min_size: Some(1),
                        max_size: Some(10),
                    }),
                    index: 2,
                    forced: true,
                },
            ];
            
            // Test binary serialization
            let serialized = ChoiceSequenceBinaryCodec::serialize_to_bytes(&choice_nodes)?;
            assert!(!serialized.is_empty());
            
            // Test binary deserialization
            let deserialized = ChoiceSequenceBinaryCodec::deserialize_from_bytes(&serialized)?;
            assert_eq!(choice_nodes.len(), deserialized.len());
            
            // Verify each choice node roundtrip
            for (original, reconstructed) in choice_nodes.iter().zip(deserialized.iter()) {
                assert_eq!(original.index, reconstructed.index);
                assert_eq!(original.forced, reconstructed.forced);
                
                // Verify value consistency
                match (&original.value, &reconstructed.value) {
                    (ChoiceValue::Integer(a), ChoiceValue::Integer(b)) => assert_eq!(a, b),
                    (ChoiceValue::Float(a), ChoiceValue::Float(b)) => {
                        if a.is_nan() && b.is_nan() {
                            assert!(true);
                        } else {
                            assert_eq!(a, b);
                        }
                    },
                    (ChoiceValue::String(a), ChoiceValue::String(b)) => assert_eq!(a, b),
                    _ => panic!("Value type mismatch in binary codec roundtrip"),
                }
            }
            
            Ok(())
        })
    }

    /// Test error handling and type safety capability
    pub fn test_error_handling_capability() -> PyResult<()> {
        Python::with_gil(|py| {
            // Test invalid constraint deserialization
            let invalid_dict = PyDict::new(py);
            invalid_dict.set_item("constraint_type", "invalid_type")?;
            
            let result = IntegerConstraints::validate_dict_structure(invalid_dict);
            assert!(result.is_err());
            
            match result.unwrap_err() {
                FfiError::ValidationError(_) => assert!(true),
                _ => panic!("Expected ValidationError"),
            }
            
            // Test malformed binary data
            let malformed_data = vec![0xFF, 0xFF, 0xFF, 0xFF];
            let result = ChoiceSequenceBinaryCodec::deserialize_from_bytes(&malformed_data);
            assert!(result.is_err());
            
            match result.unwrap_err() {
                FfiError::DeserializationError(_) => assert!(true),
                _ => panic!("Expected DeserializationError"),
            }
            
            // Test type conversion errors
            let py_string = PyString::new(py, "not a number");
            let result = choice_value_from_python(py_string.as_ref());
            assert!(result.is_err());
            
            match result.unwrap_err() {
                FfiError::TypeConversionError(_) => assert!(true),
                _ => panic!("Expected TypeConversionError"),
            }
            
            Ok(())
        })
    }

    /// Test unicode string handling capability with surrogate escapes
    pub fn test_unicode_string_handling_capability() -> PyResult<()> {
        Python::with_gil(|py| {
            let test_strings = vec![
                "Hello, 世界!",           // Basic Unicode
                "🦀🚀🌟",                 // Emoji
                "\u{0000}\u{FFFF}",       // Null and max BMP
                "\u{10000}\u{10FFFF}",    // Supplementary planes
                "a\u{0308}e\u{0301}",     // Combining characters
                "",                        // Empty string
                "ASCII only",              // ASCII subset
                "\u{FEFF}BOM test",       // Byte Order Mark
            ];
            
            for test_string in &test_strings {
                let choice_value = ChoiceValue::String(test_string.to_string());
                
                // Convert to Python
                let py_obj = choice_value_to_python(py, &choice_value)?;
                let py_str = py_obj.downcast::<PyString>(py)?;
                
                // Verify Python string content
                let py_string_content: String = py_str.extract()?;
                assert_eq!(test_string, &py_string_content);
                
                // Convert back to Rust
                let reconstructed = choice_value_from_python(py_str.as_ref())?;
                
                match reconstructed {
                    ChoiceValue::String(s) => assert_eq!(test_string, &s),
                    _ => panic!("Expected String choice value"),
                }
            }
            
            Ok(())
        })
    }

    /// Test large data handling and memory safety capability
    pub fn test_large_data_handling_capability() -> PyResult<()> {
        Python::with_gil(|py| {
            // Test large integer
            let large_int = i64::MAX;
            let choice_value = ChoiceValue::Integer(large_int);
            let py_obj = choice_value_to_python(py, &choice_value)?;
            let reconstructed = choice_value_from_python(py_obj.as_ref(py))?;
            
            match reconstructed {
                ChoiceValue::Integer(i) => assert_eq!(large_int, i),
                _ => panic!("Expected Integer choice value"),
            }
            
            // Test large byte array
            let large_bytes: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
            let choice_value = ChoiceValue::Bytes(large_bytes.clone());
            let py_obj = choice_value_to_python(py, &choice_value)?;
            let reconstructed = choice_value_from_python(py_obj.as_ref(py))?;
            
            match reconstructed {
                ChoiceValue::Bytes(bytes) => assert_eq!(large_bytes, bytes),
                _ => panic!("Expected Bytes choice value"),
            }
            
            // Test large string
            let large_string: String = "x".repeat(10000);
            let choice_value = ChoiceValue::String(large_string.clone());
            let py_obj = choice_value_to_python(py, &choice_value)?;
            let reconstructed = choice_value_from_python(py_obj.as_ref(py))?;
            
            match reconstructed {
                ChoiceValue::String(s) => assert_eq!(large_string, s),
                _ => panic!("Expected String choice value"),
            }
            
            Ok(())
        })
    }

    /// Test constraint validation parity with Python capability
    pub fn test_constraint_validation_parity_capability() -> PyResult<()> {
        Python::with_gil(|py| {
            // Test integer constraint validation
            let integer_constraint = IntegerConstraints {
                min_value: Some(10),
                max_value: Some(5), // Invalid: min > max
                shrink_towards: Some(0),
                weights: None,
            };
            
            let validation_result = integer_constraint.validate_python_parity();
            assert!(validation_result.is_err());
            
            // Test float constraint validation
            let float_constraint = FloatConstraints {
                min_value: Some(1.0),
                max_value: Some(0.0), // Invalid: min > max
                allow_nan: false,
                allow_infinity: false,
                smallest_nonzero_magnitude: Some(-1.0), // Invalid: negative
                width: 64,
            };
            
            let validation_result = float_constraint.validate_python_parity();
            assert!(validation_result.is_err());
            
            // Test bytes constraint validation
            let bytes_constraint = BytesConstraints {
                min_size: Some(100),
                max_size: Some(10), // Invalid: min > max
                encoding: Some("invalid-encoding".to_string()),
            };
            
            let validation_result = bytes_constraint.validate_python_parity();
            assert!(validation_result.is_err());
            
            Ok(())
        })
    }

    /// Run the complete ConjectureData Python FFI Integration Layer capability test suite
    pub fn run_complete_capability_test_suite() -> PyResult<()> {
        println!("Running ConjectureData Python FFI Integration Layer Capability Tests...");
        
        Self::test_constraint_type_consistency_capability()?;
        println!("✓ Constraint type consistency tests passed");
        
        Self::test_float_constraint_special_values_capability()?;
        println!("✓ Float constraint special values tests passed");
        
        Self::test_unified_constraint_serialization_capability()?;
        println!("✓ Unified constraint serialization tests passed");
        
        Self::test_choice_value_conversion_capability()?;
        println!("✓ Choice value conversion tests passed");
        
        Self::test_conjecture_data_state_synchronization_capability()?;
        println!("✓ ConjectureData state synchronization tests passed");
        
        Self::test_binary_codec_serialization_capability()?;
        println!("✓ Binary codec serialization tests passed");
        
        Self::test_error_handling_capability()?;
        println!("✓ Error handling tests passed");
        
        Self::test_unicode_string_handling_capability()?;
        println!("✓ Unicode string handling tests passed");
        
        Self::test_large_data_handling_capability()?;
        println!("✓ Large data handling tests passed");
        
        Self::test_constraint_validation_parity_capability()?;
        println!("✓ Constraint validation parity tests passed");
        
        println!("🎉 All ConjectureData Python FFI Integration Layer capability tests passed!");
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_type_consistency() {
        ConjectureDataPythonFfiCapabilityTests::test_constraint_type_consistency_capability().unwrap();
    }

    #[test]
    fn test_float_constraint_special_values() {
        ConjectureDataPythonFfiCapabilityTests::test_float_constraint_special_values_capability().unwrap();
    }

    #[test]
    fn test_unified_constraint_serialization() {
        ConjectureDataPythonFfiCapabilityTests::test_unified_constraint_serialization_capability().unwrap();
    }

    #[test]
    fn test_choice_value_conversion() {
        ConjectureDataPythonFfiCapabilityTests::test_choice_value_conversion_capability().unwrap();
    }

    #[test]
    fn test_conjecture_data_state_synchronization() {
        ConjectureDataPythonFfiCapabilityTests::test_conjecture_data_state_synchronization_capability().unwrap();
    }

    #[test]
    fn test_binary_codec_serialization() {
        ConjectureDataPythonFfiCapabilityTests::test_binary_codec_serialization_capability().unwrap();
    }

    #[test]
    fn test_error_handling() {
        ConjectureDataPythonFfiCapabilityTests::test_error_handling_capability().unwrap();
    }

    #[test]
    fn test_unicode_string_handling() {
        ConjectureDataPythonFfiCapabilityTests::test_unicode_string_handling_capability().unwrap();
    }

    #[test]
    fn test_large_data_handling() {
        ConjectureDataPythonFfiCapabilityTests::test_large_data_handling_capability().unwrap();
    }

    #[test]
    fn test_constraint_validation_parity() {
        ConjectureDataPythonFfiCapabilityTests::test_constraint_validation_parity_capability().unwrap();
    }

    #[test]
    fn test_complete_capability_suite() {
        ConjectureDataPythonFfiCapabilityTests::run_complete_capability_test_suite().unwrap();
    }
}

/// PyO3 integration test for the complete capability
#[cfg(feature = "pyo3")]
#[pyo3::prelude::pyfunction]
pub fn run_conjecture_data_ffi_capability_tests() -> PyResult<()> {
    ConjectureDataPythonFfiCapabilityTests::run_complete_capability_test_suite()
}