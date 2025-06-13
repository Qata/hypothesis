use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;

use conjecture::choice::{
    ChoiceType, ChoiceValue, Constraints, ChoiceNode,
    IntegerConstraints, BooleanConstraints, FloatConstraints
};
use conjecture::choice_sequence_management::{
    ChoiceSequenceManager, ChoiceSequenceError, MisalignmentInfo
};

#[derive(Debug, PartialEq)]
struct VerificationResult {
    test_name: String,
    python_output: String,
    rust_output: String,
    match_status: bool,
}

pub fn verify_choice_sequence_parity() -> PyResult<Vec<VerificationResult>> {
    let mut results = Vec::new();
    
    Python::with_gil(|py| -> PyResult<()> {
        // Import Python ConjectureData
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", ("/path/to/hypothesis-python/src",))?;
        
        let conjecture_module = py.import("hypothesis.internal.conjecture.data")?;
        let conjecture_data_class = conjecture_module.getattr("ConjectureData")?;
        
        // Test 1: Basic choice recording and replay
        results.push(verify_basic_choice_recording(py, conjecture_data_class)?);
        
        // Test 2: Misalignment detection
        results.push(verify_misalignment_detection(py, conjecture_data_class)?);
        
        // Test 3: Buffer overflow behavior
        results.push(verify_buffer_overflow(py, conjecture_data_class)?);
        
        // Test 4: Choice generation from index
        results.push(verify_choice_from_index(py, conjecture_data_class)?);
        
        Ok(())
    })?;
    
    Ok(results)
}

fn verify_basic_choice_recording(py: Python, conjecture_data_class: &PyAny) -> PyResult<VerificationResult> {
    // Python implementation
    let kwargs = PyDict::new(py);
    kwargs.set_item("max_length", 10000)?;
    let python_data = conjecture_data_class.call((), Some(kwargs))?;
    
    // Record choices in Python
    python_data.call_method1("draw_integer", (0, 100))?;
    python_data.call_method1("draw_boolean", ())?;
    python_data.call_method1("draw_float", ())?;
    
    let python_nodes = python_data.getattr("nodes")?;
    let python_output = format!("{:?}", python_nodes);
    
    // Rust implementation
    let mut rust_manager = ChoiceSequenceManager::new(10000, 1000, None);
    
    let int_constraints = Constraints::Integer(IntegerConstraints {
        min_value: Some(0),
        max_value: Some(100),
    });
    let _int_choice = rust_manager.draw(ChoiceType::Integer, int_constraints, None, true)?;
    
    let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
    let _bool_choice = rust_manager.draw(ChoiceType::Boolean, bool_constraints, None, true)?;
    
    let float_constraints = Constraints::Float(FloatConstraints::default());
    let _float_choice = rust_manager.draw(ChoiceType::Float, float_constraints, None, true)?;
    
    let rust_output = format!("{:?}", rust_manager.get_nodes());
    
    Ok(VerificationResult {
        test_name: "basic_choice_recording".to_string(),
        python_output,
        rust_output,
        match_status: python_output == rust_output,
    })
}

fn verify_misalignment_detection(py: Python, conjecture_data_class: &PyAny) -> PyResult<VerificationResult> {
    // Test misalignment when replaying with different types
    
    // Python: Record then replay with type mismatch
    let kwargs = PyDict::new(py);
    kwargs.set_item("max_length", 10000)?;
    let python_data1 = conjecture_data_class.call((), Some(kwargs))?;
    
    let int_val = python_data1.call_method1("draw_integer", (0, 100))?;
    let choices = python_data1.getattr("choices")?;
    
    // Create new data with prefix from previous run
    let kwargs2 = PyDict::new(py);
    kwargs2.set_item("max_length", 10000)?;
    kwargs2.set_item("prefix", choices)?;
    let python_data2 = conjecture_data_class.call((), Some(kwargs2))?;
    
    // Try to draw boolean where integer was recorded - should misalign
    let _bool_val = python_data2.call_method1("draw_boolean", ())?;
    let python_misaligned = python_data2.getattr("misaligned_at")?;
    let python_output = format!("{:?}", python_misaligned);
    
    // Rust equivalent
    let mut rust_manager1 = ChoiceSequenceManager::new(10000, 1000, None);
    let int_constraints = Constraints::Integer(IntegerConstraints {
        min_value: Some(0), 
        max_value: Some(100),
    });
    let _rust_int = rust_manager1.draw(ChoiceType::Integer, int_constraints, None, true)?;
    
    let prefix = rust_manager1.get_nodes().iter()
        .map(|node| node.value.clone())
        .collect();
    
    let mut rust_manager2 = ChoiceSequenceManager::new(10000, 1000, Some(prefix));
    let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
    let _rust_bool = rust_manager2.draw(ChoiceType::Boolean, bool_constraints, None, true)?;
    
    let rust_misaligned = rust_manager2.get_misalignment();
    let rust_output = format!("{:?}", rust_misaligned);
    
    Ok(VerificationResult {
        test_name: "misalignment_detection".to_string(),
        python_output,
        rust_output,
        match_status: python_output == rust_output,
    })
}

fn verify_buffer_overflow(py: Python, conjecture_data_class: &PyAny) -> PyResult<VerificationResult> {
    // Test buffer overflow behavior
    let small_max = 10;
    
    // Python implementation
    let kwargs = PyDict::new(py);
    kwargs.set_item("max_length", small_max)?;
    let python_data = conjecture_data_class.call((), Some(kwargs))?;
    
    // Try to exceed buffer in Python
    let mut python_overrun = false;
    for _ in 0..15 {
        match python_data.call_method1("draw_integer", (0, 100)) {
            Ok(_) => {},
            Err(_) => {
                python_overrun = true;
                break;
            }
        }
    }
    let python_output = format!("overrun: {}", python_overrun);
    
    // Rust implementation
    let mut rust_manager = ChoiceSequenceManager::new(small_max, 1000, None);
    let int_constraints = Constraints::Integer(IntegerConstraints {
        min_value: Some(0),
        max_value: Some(100),
    });
    
    let mut rust_overrun = false;
    for _ in 0..15 {
        match rust_manager.draw(ChoiceType::Integer, int_constraints.clone(), None, true) {
            Ok(_) => {},
            Err(_) => {
                rust_overrun = true;
                break;
            }
        }
    }
    let rust_output = format!("overrun: {}", rust_overrun);
    
    Ok(VerificationResult {
        test_name: "buffer_overflow".to_string(),
        python_output,
        rust_output,
        match_status: python_output == rust_output,
    })
}

fn verify_choice_from_index(py: Python, conjecture_data_class: &PyAny) -> PyResult<VerificationResult> {
    // Test choice_from_index function for generating simplest choices
    
    // Python: Get the choice_from_index function
    let conjecture_module = py.import("hypothesis.internal.conjecture.data")?;
    let choice_from_index_fn = conjecture_module.getattr("choice_from_index")?;
    
    // Test integer choice
    let python_int = choice_from_index_fn.call1((0, "integer", 0, 100))?;
    let python_bool = choice_from_index_fn.call1((0, "boolean"))?;
    let python_float = choice_from_index_fn.call1((0, "float"))?;
    
    let python_output = format!("int: {:?}, bool: {:?}, float: {:?}", 
                               python_int, python_bool, python_float);
    
    // Rust implementation  
    let manager = ChoiceSequenceManager::new(1000, 100, None);
    
    let int_constraints = Constraints::Integer(IntegerConstraints {
        min_value: Some(0),
        max_value: Some(100),
    });
    let rust_int = manager.choice_from_index(0, ChoiceType::Integer, &int_constraints)?;
    
    let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
    let rust_bool = manager.choice_from_index(0, ChoiceType::Boolean, &bool_constraints)?;
    
    let float_constraints = Constraints::Float(FloatConstraints::default());
    let rust_float = manager.choice_from_index(0, ChoiceType::Float, &float_constraints)?;
    
    let rust_output = format!("int: {:?}, bool: {:?}, float: {:?}", 
                             rust_int, rust_bool, rust_float);
    
    Ok(VerificationResult {
        test_name: "choice_from_index".to_string(),
        python_output,
        rust_output,
        match_status: python_output == rust_output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_choice_sequence_parity() {
        let results = verify_choice_sequence_parity().expect("Verification failed");
        
        for result in &results {
            println!("\n=== {} ===", result.test_name);
            println!("Python: {}", result.python_output);
            println!("Rust:   {}", result.rust_output);
            println!("Match:  {}", result.match_status);
            
            if !result.match_status {
                println!("❌ MISMATCH DETECTED");
            } else {
                println!("✅ OUTPUTS MATCH");
            }
        }
        
        let all_match = results.iter().all(|r| r.match_status);
        assert!(all_match, "Some tests failed parity verification");
    }
}