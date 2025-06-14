// PyO3 verification for direct comparison of Python and Rust float encoding
// Tests byte-for-byte identity between Python Hypothesis and Rust implementations

use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyBytes, PyList};
use conjecture::choice::indexing::float_encoding::{float_to_lex, lex_to_float, FloatWidth};

/// Direct comparison of float encoding between Python and Rust
/// Uses PyO3 to call original Python functions and compare outputs
pub struct FloatEncodingVerifier {
    python_module: PyObject,
}

impl Default for FloatEncodingVerifier {
    fn default() -> Self {
        Self::new().expect("Failed to initialize Python")
    }
}

impl FloatEncodingVerifier {
    pub fn new() -> PyResult<Self> {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            // Import Python Hypothesis float encoding functions
            let python_code = r#"
import sys
sys.path.insert(0, '/home/ch/Develop/hypothesis/hypothesis-python/src')

from hypothesis.internal.conjecture.floats import float_to_lex as python_float_to_lex
from hypothesis.internal.conjecture.floats import lex_to_float as python_lex_to_float

def encode_float_python(value, width=64):
    """Encode float using Python Hypothesis implementation"""
    return python_float_to_lex(value, width)

def decode_float_python(encoded, width=64):
    """Decode float using Python Hypothesis implementation""" 
    return python_lex_to_float(encoded, width)

def test_float_values():
    """Return standard test float values"""
    return [
        0.0, -0.0, 1.0, -1.0, 2.5, -2.5,
        float('inf'), float('-inf'), float('nan'),
        1e-100, 1e100, 1.2345e-50, -9.8765e50,
        2.2250738585072014e-308,  # min normal f64
        1.7976931348623157e308,   # max f64
    ]
"#;
            
            py.run_bound(python_code, None, None)?;
            let module = py.import("__main__")?;
            
            Ok(Self {
                python_module: module.into(),
            })
        })
    }

    /// Test encoding of specific float values
    pub fn verify_float_encoding(&self, value: f64, width: FloatWidth) -> Result<bool, String> {
        Python::with_gil(|py| {
            let module = self.python_module.bind(py);
            
            // Get Python encoding
            let py_encode_fn = module.getattr("encode_float_python")
                .map_err(|e| format!("Failed to get Python function: {}", e))?;
            
            let width_bits = match width {
                FloatWidth::Width32 => 32u8,
                FloatWidth::Width64 => 64u8,
                FloatWidth::Width16 => 16u8,
            };
            
            let python_result = py_encode_fn.call1((value, width_bits))
                .map_err(|e| format!("Python encoding failed: {}", e))?;
            
            let python_bytes: Vec<u8> = python_result.extract()
                .map_err(|e| format!("Failed to extract Python bytes: {}", e))?;
            
            // Get Rust encoding - current functions don't support width parameter
            let rust_lex_value = match width {
                FloatWidth::Width32 => float_to_lex(value as f32 as f64),
                FloatWidth::Width64 => float_to_lex(value),
                FloatWidth::Width16 => float_to_lex(value), // fallback to f64
            };
            
            // Convert to bytes for comparison
            let rust_bytes = rust_lex_value.to_le_bytes().to_vec();
            
            // Compare byte-for-byte
            let matches = python_bytes == rust_bytes;
            
            if !matches {
                return Err(format!(
                    "Encoding mismatch for {}: Python={:?}, Rust={:?}", 
                    value, python_bytes, rust_bytes
                ));
            }
            
            Ok(true)
        })
    }

    /// Test decoding of specific byte sequences  
    pub fn verify_float_decoding(&self, bytes: &[u8], width: FloatWidth) -> Result<bool, String> {
        Python::with_gil(|py| {
            let module = self.python_module.bind(py);
            
            // Get Python decoding
            let py_decode_fn = module.getattr("decode_float_python")
                .map_err(|e| format!("Failed to get Python function: {}", e))?;
                
            let width_bits = match width {
                FloatWidth::Width32 => 32u8,
                FloatWidth::Width64 => 64u8,
                FloatWidth::Width16 => 16u8,
            };
            
            let py_bytes = PyBytes::new(py, bytes);
            let python_result = py_decode_fn.call1((py_bytes, width_bits))
                .map_err(|e| format!("Python decoding failed: {}", e))?;
                
            let python_value: f64 = python_result.extract()
                .map_err(|e| format!("Failed to extract Python float: {}", e))?;
            
            // Get Rust decoding - convert bytes back to u64 first
            if bytes.len() != 8 {
                return Err(format!("Expected 8 bytes for u64, got {}", bytes.len()));
            }
            
            let mut lex_bytes = [0u8; 8];
            lex_bytes.copy_from_slice(bytes);
            let lex_value = u64::from_le_bytes(lex_bytes);
            let rust_value = lex_to_float(lex_value);
            
            // Compare values (handling NaN specially)
            let matches = if python_value.is_nan() && rust_value.is_nan() {
                true
            } else if python_value.is_infinite() && rust_value.is_infinite() {
                python_value.signum() == rust_value.signum()
            } else {
                (python_value - rust_value).abs() < f64::EPSILON
            };
            
            if !matches {
                return Err(format!(
                    "Decoding mismatch for {:?}: Python={}, Rust={}", 
                    bytes, python_value, rust_value
                ));
            }
            
            Ok(true)
        })
    }
    
    /// Get test values from Python 
    pub fn get_test_values(&self) -> Result<Vec<f64>, String> {
        Python::with_gil(|py| {
            let module = self.python_module.bind(py);
            let test_fn = module.getattr("test_float_values")
                .map_err(|e| format!("Failed to get test function: {}", e))?;
                
            let result = test_fn.call0()
                .map_err(|e| format!("Failed to call test function: {}", e))?;
                
            let values: Vec<f64> = result.extract()
                .map_err(|e| format!("Failed to extract test values: {}", e))?;
                
            Ok(values)
        })
    }
}

/// Run comprehensive float encoding verification
pub fn run_float_encoding_verification() -> Result<(), String> {
    println!("Starting PyO3 float encoding verification...");
    
    let verifier = FloatEncodingVerifier::new()
        .map_err(|e| format!("Failed to initialize verifier: {}", e))?;
    
    // Get test values from Python
    let test_values = verifier.get_test_values()?;
    println!("Testing {} float values", test_values.len());
    
    let mut passed = 0;
    let mut failed = 0;
    
    // Test both Width32 and Width64 widths
    for &width in &[FloatWidth::Width32, FloatWidth::Width64] {
        println!("\nTesting {:?} encoding:", width);
        
        for &value in &test_values {
            match verifier.verify_float_encoding(value, width) {
                Ok(_) => {
                    passed += 1;
                    println!("  ✓ {}", value);
                }
                Err(e) => {
                    failed += 1;
                    println!("  ✗ {}", e);
                }
            }
        }
    }
    
    // Test round-trip consistency
    println!("\nTesting round-trip consistency:");
    for &value in &test_values {
        for &width in &[FloatWidth::Width32, FloatWidth::Width64] {
            // Encode then decode
            let encoded = match width {
                FloatWidth::Width32 => float_to_lex(value as f32 as f64),
                FloatWidth::Width64 => float_to_lex(value),
                FloatWidth::Width16 => float_to_lex(value), // fallback
            }.to_le_bytes();
            
            match verifier.verify_float_decoding(&encoded, width) {
                Ok(_) => {
                    passed += 1;
                    println!("  ✓ Round-trip {} ({:?})", value, width);
                }
                Err(e) => {
                    failed += 1;
                    println!("  ✗ {}", e);
                }
            }
        }
    }
    
    println!("\n=== VERIFICATION RESULTS ===");
    println!("Passed: {}", passed);
    println!("Failed: {}", failed);
    
    if failed == 0 {
        println!("✅ All tests passed - Python/Rust parity achieved!");
        Ok(())
    } else {
        Err(format!("❌ {} test(s) failed - parity not achieved", failed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_float_encoding() {
        let verifier = FloatEncodingVerifier::new().expect("Failed to init verifier");
        
        // Test basic values
        assert!(verifier.verify_float_encoding(0.0, FloatWidth::Width64).is_ok());
        assert!(verifier.verify_float_encoding(1.0, FloatWidth::Width64).is_ok());
        assert!(verifier.verify_float_encoding(-1.0, FloatWidth::Width64).is_ok());
    }
    
    #[test]
    fn test_special_float_values() {
        let verifier = FloatEncodingVerifier::new().expect("Failed to init verifier");
        
        // Test special values
        assert!(verifier.verify_float_encoding(f64::INFINITY, FloatWidth::Width64).is_ok());
        assert!(verifier.verify_float_encoding(f64::NEG_INFINITY, FloatWidth::Width64).is_ok());
        assert!(verifier.verify_float_encoding(f64::NAN, FloatWidth::Width64).is_ok());
    }
    
    #[test]
    fn test_comprehensive_verification() {
        // This will run the full verification suite
        assert!(run_float_encoding_verification().is_ok());
    }
}