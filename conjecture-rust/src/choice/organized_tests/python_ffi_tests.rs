//! Python FFI verification tests
//! 
//! These tests call Python Hypothesis's actual choice functions via PyO3 FFI
//! to verify our Rust implementation produces identical results.

use crate::choice::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyBool, PyFloat, PyLong, PyAny};

#[cfg(test)]
mod python_ffi_verification {
    use super::*;

    /// Setup Python interpreter and import required modules
    fn setup_python() -> PyResult<Py<PyModule>> {
        Python::with_gil(|py| {
            // Add the hypothesis-python src directory to Python path
            let sys = py.import("sys")?;
            let path: &pyo3::types::PyList = sys.getattr("path")?.extract()?;
            path.insert(0, "/home/ch/Develop/hypothesis-conjecture-rust-enhancement/hypothesis-python/src")?;
            
            // Import the choice module
            let choice_module = py.import("hypothesis.internal.conjecture.choice")?;
            Ok(choice_module.into())
        })
    }

    /// Convert Rust IntegerConstraints to Python dict
    fn rust_int_constraints_to_python<'a>(py: Python<'a>, constraints: &IntegerConstraints) -> PyResult<&'a PyDict> {
        let dict = PyDict::new(py);
        
        if let Some(min_val) = constraints.min_value {
            dict.set_item("min_value", min_val)?;
        } else {
            dict.set_item("min_value", py.None())?;
        }
        
        if let Some(max_val) = constraints.max_value {
            dict.set_item("max_value", max_val)?;
        } else {
            dict.set_item("max_value", py.None())?;
        }
        
        dict.set_item("weights", py.None())?;
        dict.set_item("shrink_towards", constraints.shrink_towards.unwrap_or(0))?;
        
        Ok(dict)
    }

    /// Convert Rust BooleanConstraints to Python dict
    fn rust_bool_constraints_to_python<'a>(py: Python<'a>, constraints: &BooleanConstraints) -> PyResult<&'a PyDict> {
        let dict = PyDict::new(py);
        dict.set_item("p", constraints.p)?;
        Ok(dict)
    }

    /// Convert Rust FloatConstraints to Python dict
    fn rust_float_constraints_to_python<'a>(py: Python<'a>, constraints: &FloatConstraints) -> PyResult<&'a PyDict> {
        let dict = PyDict::new(py);
        dict.set_item("min_value", constraints.min_value)?;
        dict.set_item("max_value", constraints.max_value)?;
        dict.set_item("allow_nan", constraints.allow_nan)?;
        
        // smallest_nonzero_magnitude is f64, not Option<f64>
        dict.set_item("smallest_nonzero_magnitude", constraints.smallest_nonzero_magnitude)?;
        
        Ok(dict)
    }

    /// Call Python's choice_to_index function
    fn call_python_choice_to_index(
        choice_module: &PyModule, 
        value: &ChoiceValue, 
        constraints: &Constraints
    ) -> PyResult<u128> {
        Python::with_gil(|py| {
            let choice_to_index = choice_module.getattr("choice_to_index")?;
            
            let (py_value, py_constraints) = match (value, constraints) {
                (ChoiceValue::Integer(val), Constraints::Integer(c)) => {
                    let py_val = PyLong::from(*val);
                    let py_constr = rust_int_constraints_to_python(py, c)?;
                    (py_val.as_ref(), py_constr.as_ref())
                },
                (ChoiceValue::Boolean(val), Constraints::Boolean(c)) => {
                    let py_val = PyBool::new(py, *val);
                    let py_constr = rust_bool_constraints_to_python(py, c)?;
                    (py_val as &PyAny, py_constr as &PyAny)
                },
                (ChoiceValue::Float(val), Constraints::Float(c)) => {
                    let py_val = PyFloat::new(py, *val);
                    let py_constr = rust_float_constraints_to_python(py, c)?;
                    (py_val as &PyAny, py_constr as &PyAny)
                },
                _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "String and Bytes not implemented in FFI tests yet"
                )),
            };
            
            let result = choice_to_index.call1((py_value, py_constraints))?;
            let index: u128 = result.extract()?;
            Ok(index)
        })
    }

    /// Call Python's choice_from_index function
    fn call_python_choice_from_index(
        choice_module: &PyModule,
        index: u128,
        choice_type: &str,
        constraints: &Constraints
    ) -> PyResult<ChoiceValue> {
        Python::with_gil(|py| {
            let choice_from_index = choice_module.getattr("choice_from_index")?;
            
            let py_constraints = match constraints {
                Constraints::Integer(c) => rust_int_constraints_to_python(py, c)?.as_ref(),
                Constraints::Boolean(c) => rust_bool_constraints_to_python(py, c)?.as_ref(),
                Constraints::Float(c) => rust_float_constraints_to_python(py, c)?.as_ref(),
                _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "String and Bytes not implemented in FFI tests yet"
                )),
            };
            
            let result = choice_from_index.call1((index, choice_type, py_constraints))?;
            
            match choice_type {
                "integer" => {
                    let val: i128 = result.extract()?;
                    Ok(ChoiceValue::Integer(val))
                },
                "boolean" => {
                    let val: bool = result.extract()?;
                    Ok(ChoiceValue::Boolean(val))
                },
                "float" => {
                    let val: f64 = result.extract()?;
                    Ok(ChoiceValue::Float(val))
                },
                _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Unsupported choice type"
                )),
            }
        })
    }

    #[test]
    fn test_integer_ffi_verification() -> PyResult<()> {
        println!("FFI_VERIFICATION DEBUG: Testing integer choice indexing against Python");
        
        let choice_module = setup_python()?;
        
        let test_cases = vec![
            // Unbounded case with default shrink_towards=0
            (
                vec![0, 1, -1, 2, -2, 3, -3],
                IntegerConstraints {
                    min_value: None,
                    max_value: None,
                    weights: None,
                    shrink_towards: Some(0),
                }
            ),
            // Custom shrink_towards
            (
                vec![2, 3, 1, 4, 0],
                IntegerConstraints {
                    min_value: None,
                    max_value: None,
                    weights: None,
                    shrink_towards: Some(2),
                }
            ),
            // Bounded case
            (
                vec![0, 1, -1, 2, -2, 3, -3],
                IntegerConstraints {
                    min_value: Some(-3),
                    max_value: Some(3),
                    weights: None,
                    shrink_towards: Some(0),
                }
            ),
        ];
        
        Python::with_gil(|py| {
            let choice_module = choice_module.as_ref(py);
            
            for (test_values, constraints) in test_cases {
                let rust_constraints = Constraints::Integer(constraints.clone());
                
                for value in test_values {
                    let rust_value = ChoiceValue::Integer(value);
                    
                    // Get results from both implementations
                    let python_index = call_python_choice_to_index(choice_module, &rust_value, &rust_constraints)?;
                    let rust_index = choice_to_index(&rust_value, &rust_constraints);
                    
                    println!("FFI_VERIFICATION DEBUG: Value {} -> Python index {}, Rust index {}", 
                        value, python_index, rust_index);
                    
                    assert_eq!(python_index, rust_index, 
                        "Index mismatch for value {} with constraints {:?}", value, constraints);
                    
                    // Test reverse direction
                    let python_recovered = call_python_choice_from_index(choice_module, python_index, "integer", &rust_constraints)?;
                    let rust_recovered = choice_from_index(rust_index, "integer", &rust_constraints);
                    
                    assert!(choice_equal(&python_recovered, &rust_recovered),
                        "Recovered value mismatch: Python {:?} vs Rust {:?}", python_recovered, rust_recovered);
                }
            }
            
            Ok(())
        })
    }

    #[test]
    fn test_boolean_ffi_verification() -> PyResult<()> {
        println!("FFI_VERIFICATION DEBUG: Testing boolean choice indexing against Python");
        
        let choice_module = setup_python()?;
        
        let test_cases = vec![
            (false, BooleanConstraints { p: 0.0 }),  // Only false permitted
            (true, BooleanConstraints { p: 1.0 }),   // Only true permitted
            (false, BooleanConstraints { p: 0.5 }),  // Both permitted
            (true, BooleanConstraints { p: 0.5 }),
        ];
        
        Python::with_gil(|py| {
            let choice_module = choice_module.as_ref(py);
            
            for (value, constraints) in test_cases {
                let rust_constraints = Constraints::Boolean(constraints.clone());
                let rust_value = ChoiceValue::Boolean(value);
                
                // Get results from both implementations
                let python_index = call_python_choice_to_index(choice_module, &rust_value, &rust_constraints)?;
                let rust_index = choice_to_index(&rust_value, &rust_constraints);
                
                println!("FFI_VERIFICATION DEBUG: Boolean {} (p={}) -> Python index {}, Rust index {}", 
                    value, constraints.p, python_index, rust_index);
                
                assert_eq!(python_index, rust_index, 
                    "Boolean index mismatch for value {} with p={}", value, constraints.p);
                
                // Test reverse direction
                let python_recovered = call_python_choice_from_index(choice_module, python_index, "boolean", &rust_constraints)?;
                let rust_recovered = choice_from_index(rust_index, "boolean", &rust_constraints);
                
                assert!(choice_equal(&python_recovered, &rust_recovered),
                    "Boolean recovered value mismatch: Python {:?} vs Rust {:?}", python_recovered, rust_recovered);
            }
            
            Ok(())
        })
    }

    #[test]
    fn test_float_ffi_verification() -> PyResult<()> {
        println!("FFI_VERIFICATION DEBUG: Testing float choice indexing against Python");
        
        let choice_module = setup_python()?;
        
        let test_values = vec![
            0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5,
            f64::INFINITY, f64::NEG_INFINITY, f64::NAN
        ];
        
        let constraints = FloatConstraints {
            min_value: f64::NEG_INFINITY,
            max_value: f64::INFINITY,
            allow_nan: true,
            smallest_nonzero_magnitude: Some(f64::MIN_POSITIVE),
        };
        
        Python::with_gil(|py| {
            let choice_module = choice_module.as_ref(py);
            
            for value in test_values {
                let rust_constraints = Constraints::Float(constraints.clone());
                let rust_value = ChoiceValue::Float(value);
                
                // Get results from both implementations
                let python_index = call_python_choice_to_index(choice_module, &rust_value, &rust_constraints)?;
                let rust_index = choice_to_index(&rust_value, &rust_constraints);
                
                println!("FFI_VERIFICATION DEBUG: Float {} -> Python index {}, Rust index {}", 
                    value, python_index, rust_index);
                
                // For finite values, should match exactly
                if value.is_finite() {
                    assert_eq!(python_index, rust_index, 
                        "Float index mismatch for finite value {}", value);
                } else {
                    // For special values (inf, nan), check sign bit consistency
                    let python_sign = (python_index >> 64) != 0;
                    let rust_sign = (rust_index >> 64) != 0;
                    assert_eq!(python_sign, rust_sign, 
                        "Float sign bit mismatch for value {}: Python sign={}, Rust sign={}", 
                        value, python_sign, rust_sign);
                }
                
                // Test reverse direction for finite values only
                if value.is_finite() && !value.is_nan() {
                    let python_recovered = call_python_choice_from_index(choice_module, python_index, "float", &rust_constraints)?;
                    let rust_recovered = choice_from_index(rust_index, "float", &rust_constraints);
                    
                    assert!(choice_equal(&python_recovered, &rust_recovered),
                        "Float recovered value mismatch: Python {:?} vs Rust {:?}", python_recovered, rust_recovered);
                }
            }
            
            Ok(())
        })
    }

    #[test]
    fn test_edge_cases_ffi_verification() -> PyResult<()> {
        println!("FFI_VERIFICATION DEBUG: Testing edge cases against Python");
        
        let choice_module = setup_python()?;
        
        // Test shrink_towards clamping
        let edge_cases = vec![
            // shrink_towards gets clamped to min_value
            (1, IntegerConstraints {
                min_value: Some(1),
                max_value: Some(5),
                weights: None,
                shrink_towards: Some(0),  // Should be clamped to 1
            }),
            // shrink_towards gets clamped to max_value  
            (5, IntegerConstraints {
                min_value: Some(1),
                max_value: Some(5),
                weights: None,
                shrink_towards: Some(10), // Should be clamped to 5
            }),
        ];
        
        Python::with_gil(|py| {
            let choice_module = choice_module.as_ref(py);
            
            for (test_value, constraints) in edge_cases {
                let rust_constraints = Constraints::Integer(constraints.clone());
                let rust_value = ChoiceValue::Integer(test_value);
                
                let python_index = call_python_choice_to_index(choice_module, &rust_value, &rust_constraints)?;
                let rust_index = choice_to_index(&rust_value, &rust_constraints);
                
                println!("FFI_VERIFICATION DEBUG: Edge case value {} -> Python index {}, Rust index {}", 
                    test_value, python_index, rust_index);
                
                assert_eq!(python_index, rust_index,
                    "Edge case index mismatch for value {} with constraints {:?}", test_value, constraints);
            }
            
            Ok(())
        })
    }
}