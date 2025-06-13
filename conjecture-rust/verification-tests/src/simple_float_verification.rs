//! Simple float encoding verification without complex features

use pyo3::prelude::*;
use pyo3::types::PyModule;

pub fn run_float_verification() -> PyResult<()> {
    Python::with_gil(|py| {
        // Add the hypothesis-python src directory to Python path
        let sys = PyModule::import(py, "sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("insert", (0, "/home/ch/Develop/hypothesis/hypothesis-python/src"))?;
        
        // Import the float encoding module directly
        match PyModule::import(py, "hypothesis.internal.conjecture.floats") {
            Ok(float_encoding) => {
                println!("✅ Successfully imported Python float encoding module");
                
                let encode_to_index = float_encoding.getattr("float_to_lex")?;
                
                // Test critical float values (positive values for basic verification)
                let test_floats = [0.0, 1.0, 2.5];
                
                println!("\n🔍 Testing Float Encoding Python-Rust Parity:");
                
                for &value in &test_floats {
                    print!("  Testing {:<15}: ", format!("{}", value));
                    
                    // Get Python's encoding
                    match encode_to_index.call1((value,)) {
                        Ok(py_index) => {
                            let py_index_value: u64 = py_index.extract()?;
                            
                            // Get Rust's encoding
                            let rust_index = conjecture::choice::indexing::float_encoding::float_to_lex(value);
                            
                            if rust_index == py_index_value {
                                println!("✅ MATCH ({})", py_index_value);
                            } else {
                                println!("❌ MISMATCH - Python: {}, Rust: {}", py_index_value, rust_index);
                            }
                        },
                        Err(e) => {
                            println!("💥 Python Error: {}", e);
                        }
                    }
                }
                
                Ok(())
            },
            Err(e) => {
                println!("❌ Failed to import Python float encoding: {}", e);
                println!("   This suggests the Python path or Hypothesis installation has issues");
                Ok(())
            }
        }
    })
}