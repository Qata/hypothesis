//! Python FFI interface for calling Hypothesis choice functions

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyList};
use pyo3::conversion::IntoPyObjectExt;
use conjecture::choice::*;

/// Python interface for calling Hypothesis choice functions
pub struct PythonInterface {
    choice_module: Py<PyModule>,
}

impl PythonInterface {
    /// Initialize Python and import Hypothesis choice module
    pub fn new() -> PyResult<Self> {
        Python::with_gil(|py| {
            // Add the hypothesis-python src directory to Python path
            let sys = PyModule::import(py, "sys")?;
            let path: Bound<'_, PyList> = sys.getattr("path")?.downcast_into()?;
            path.insert(0, "/home/ch/Develop/hypothesis-conjecture-rust-enhancement/hypothesis-python/src")?;
            
            // Import the choice module
            let choice_module = PyModule::import(py, "hypothesis.internal.conjecture.choice")?;
            Ok(Self {
                choice_module: choice_module.into(),
            })
        })
    }

    /// Call Python's choice_to_index function
    pub fn choice_to_index(&self, value: &ChoiceValue, constraints: &Constraints) -> PyResult<u128> {
        Python::with_gil(|py| {
            let choice_module = self.choice_module.bind(py);
            let choice_to_index = choice_module.getattr("choice_to_index")?;
            
            let (py_value, py_constraints) = self.convert_to_python(py, value, constraints)?;
            let result = choice_to_index.call1((py_value, py_constraints))?;
            let index: u128 = result.extract()?;
            Ok(index)
        })
    }

    /// Call Python's choice_from_index function
    pub fn choice_from_index(
        &self,
        index: u128,
        choice_type: &str,
        constraints: &Constraints
    ) -> PyResult<ChoiceValue> {
        Python::with_gil(|py| {
            let choice_module = self.choice_module.bind(py);
            let choice_from_index = choice_module.getattr("choice_from_index")?;
            
            let py_constraints = self.convert_constraints_to_python(py, constraints)?;
            let result = choice_from_index.call1((index, choice_type, py_constraints))?;
            
            self.convert_from_python(choice_type, &result)
        })
    }

    /// Call Python's choice_equal function
    pub fn choice_equal(&self, a: &ChoiceValue, b: &ChoiceValue) -> PyResult<bool> {
        Python::with_gil(|py| {
            let choice_module = self.choice_module.bind(py);
            let choice_equal = choice_module.getattr("choice_equal")?;
            
            let py_a = self.convert_value_to_python(py, a)?;
            let py_b = self.convert_value_to_python(py, b)?;
            
            let result = choice_equal.call1((py_a, py_b))?;
            let equal: bool = result.extract()?;
            Ok(equal)
        })
    }

    /// Convert Rust types to Python equivalents
    fn convert_to_python<'py>(
        &self,
        py: Python<'py>,
        value: &ChoiceValue,
        constraints: &Constraints
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let py_value = self.convert_value_to_python(py, value)?;
        let py_constraints = self.convert_constraints_to_python(py, constraints)?;
        Ok((py_value, py_constraints))
    }

    /// Convert Rust ChoiceValue to Python
    fn convert_value_to_python<'py>(&self, py: Python<'py>, value: &ChoiceValue) -> PyResult<Bound<'py, PyAny>> {
        match value {
            ChoiceValue::Integer(val) => (*val).into_bound_py_any(py),
            ChoiceValue::Boolean(val) => (*val).into_bound_py_any(py),
            ChoiceValue::Float(val) => (*val).into_bound_py_any(py),
            ChoiceValue::String(val) => val.clone().into_bound_py_any(py),
            ChoiceValue::Bytes(val) => val.clone().into_bound_py_any(py),
        }
    }

    /// Convert Rust Constraints to Python dict
    fn convert_constraints_to_python<'py>(&self, py: Python<'py>, constraints: &Constraints) -> PyResult<Bound<'py, PyAny>> {
        match constraints {
            Constraints::Integer(c) => {
                let dict = PyDict::new(py);
                
                if let Some(min_val) = c.min_value {
                    dict.set_item("min_value", min_val)?;
                } else {
                    dict.set_item("min_value", py.None())?;
                }
                
                if let Some(max_val) = c.max_value {
                    dict.set_item("max_value", max_val)?;
                } else {
                    dict.set_item("max_value", py.None())?;
                }
                
                dict.set_item("weights", py.None())?;
                dict.set_item("shrink_towards", c.shrink_towards.unwrap_or(0))?;
                
                Ok(dict.into_any())
            },
            Constraints::Boolean(c) => {
                let dict = PyDict::new(py);
                dict.set_item("p", c.p)?;
                Ok(dict.into_any())
            },
            Constraints::Float(c) => {
                let dict = PyDict::new(py);
                dict.set_item("min_value", c.min_value)?;
                dict.set_item("max_value", c.max_value)?;
                dict.set_item("allow_nan", c.allow_nan)?;
                
                if let Some(smallest) = c.smallest_nonzero_magnitude {
                    dict.set_item("smallest_nonzero_magnitude", smallest)?;
                } else {
                    dict.set_item("smallest_nonzero_magnitude", 0.0)?;
                }
                
                Ok(dict.into_any())
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "String and Bytes constraints not implemented yet"
            )),
        }
    }

    /// Convert Python result back to Rust ChoiceValue
    fn convert_from_python(&self, choice_type: &str, py_value: &Bound<'_, PyAny>) -> PyResult<ChoiceValue> {
        match choice_type {
            "integer" => {
                let val: i128 = py_value.extract()?;
                Ok(ChoiceValue::Integer(val))
            },
            "boolean" => {
                let val: bool = py_value.extract()?;
                Ok(ChoiceValue::Boolean(val))
            },
            "float" => {
                let val: f64 = py_value.extract()?;
                Ok(ChoiceValue::Float(val))
            },
            "string" => {
                let val: String = py_value.extract()?;
                Ok(ChoiceValue::String(val))
            },
            "bytes" => {
                let val: Vec<u8> = py_value.extract()?;
                Ok(ChoiceValue::Bytes(val))
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported choice type: {}", choice_type)
            )),
        }
    }
}