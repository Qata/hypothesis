//! ConjectureData operation verification between Python and Rust

use pyo3::prelude::*;
use pyo3::types::PyModule;

pub fn run_conjecture_data_verification() -> PyResult<()> {
    Python::with_gil(|py| {
        // Add the hypothesis-python src directory to Python path
        let sys = PyModule::import(py, "sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("insert", (0, "/home/ch/Develop/hypothesis/hypothesis-python/src"))?;
        
        match PyModule::import(py, "hypothesis.internal.conjecture.data") {
            Ok(data_module) => {
                println!("✅ Successfully imported Python ConjectureData module");
                
                // Test basic ConjectureData operations with same seed
                let seed = 12345_u64;
                
                println!("\n🔍 Testing ConjectureData Operations Python-Rust Parity:");
                
                // Create Python ConjectureData with Random object
                let random_module = PyModule::import(py, "random")?;
                let random_class = random_module.getattr("Random")?;
                let py_random = random_class.call1((seed,))?;
                
                let conjecture_data_class = data_module.getattr("ConjectureData")?;
                // Use keyword arguments
                let kwargs = pyo3::types::PyDict::new(py);
                kwargs.set_item("random", py_random)?;
                let py_data = conjecture_data_class.call((), Some(&kwargs))?;
                
                // Create Rust ConjectureData
                let mut rust_data = conjecture::data::ConjectureData::new(seed);
                
                println!("  Seed {}: Python and Rust ConjectureData created", seed);
                
                // Test draw_boolean operations
                println!("  Testing draw_boolean()...");
                match rust_data.draw_boolean(0.5, None, false) {
                    Ok(rust_bool) => {
                        match py_data.call_method0("draw_boolean") {
                            Ok(py_result) => {
                                let py_bool: bool = py_result.extract()?;
                                if rust_bool == py_bool {
                                    println!("    ✅ draw_boolean(): {} (MATCH)", rust_bool);
                                } else {
                                    println!("    ❌ draw_boolean(): Rust={}, Python={}", rust_bool, py_bool);
                                }
                            },
                            Err(e) => println!("    💥 Python draw_boolean() error: {}", e)
                        }
                    },
                    Err(e) => println!("    💥 Rust draw_boolean() error: {:?}", e)
                }
                
                // Test draw_integer operations  
                println!("  Testing draw_integer(0, 10)...");
                match rust_data.draw_integer(Some(0), Some(10), None, 0, None, false) {
                    Ok(rust_int) => {
                        match py_data.call_method1("draw_integer", (0, 10)) {
                            Ok(py_result) => {
                                let py_int: i128 = py_result.extract()?;
                                if rust_int == py_int {
                                    println!("    ✅ draw_integer(0, 10): {} (MATCH)", rust_int);
                                } else {
                                    println!("    ❌ draw_integer(0, 10): Rust={}, Python={}", rust_int, py_int);
                                }
                            },
                            Err(e) => println!("    💥 Python draw_integer() error: {}", e)
                        }
                    },
                    Err(e) => println!("    💥 Rust draw_integer() error: {:?}", e)
                }
                
                println!("  ✅ ConjectureData parity verification completed");
                Ok(())
            },
            Err(e) => {
                println!("❌ Failed to import Python ConjectureData: {}", e);
                println!("   This suggests the Python path or Hypothesis installation has issues");
                Ok(())
            }
        }
    })
}