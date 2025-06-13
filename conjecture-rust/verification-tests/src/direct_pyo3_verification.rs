//! Direct PyO3 verification comparing Rust outputs to Python outputs byte-for-byte

use pyo3::prelude::*;
use pyo3::types::PyModule;
use conjecture::data::ConjectureData;

pub struct DirectPyO3Verifier {
    conjecture_module: Py<PyModule>,
}

impl DirectPyO3Verifier {
    /// Initialize Python and import the exact Hypothesis modules we need
    pub fn new() -> PyResult<Self> {
        Python::with_gil(|py| {
            // Add the hypothesis-python src directory to Python path
            let sys = PyModule::import(py, "sys")?;
            let path = sys.getattr("path")?;
            path.call_method1("insert", (0, "/home/ch/Develop/hypothesis/hypothesis-python/src"))?;
            
            // Import the conjecture data module directly
            let conjecture_module = PyModule::import(py, "hypothesis.internal.conjecture.data")?;
            Ok(Self {
                conjecture_module: conjecture_module.into(),
            })
        })
    }

    /// Direct comparison test for draw_integer operations using same seed
    pub fn verify_draw_integer(&self, seed: u64, min_val: Option<i128>, max_val: Option<i128>) -> PyResult<bool> {
        Python::with_gil(|py| {
            // Create Rust ConjectureData with specific seed
            let mut rust_data = ConjectureData::new(seed);
            let rust_result = rust_data.draw_integer(
                min_val,
                max_val,
                None, // weights
                0,    // shrink_towards
                None, // forced
                false // observe
            );

            // Create Python ConjectureData with same seed
            let conjecture_module = self.conjecture_module.bind(py);
            let conjecture_data_class = conjecture_module.getattr("ConjectureData")?;
            
            // Python ConjectureData constructor takes a Random instance or seed
            let py_data = conjecture_data_class.call1((seed,))?;
            
            // Call Python draw_integer with same constraints
            let py_result = if min_val.is_some() || max_val.is_some() {
                py_data.call_method1("draw_integer", (min_val, max_val))?
            } else {
                py_data.call_method0("draw_integer")?
            };

            // Extract results and compare
            match rust_result {
                Ok(rust_value) => {
                    let py_value: i128 = py_result.extract()?;
                    let values_match = rust_value == py_value;
                    if !values_match {
                        println!("INTEGER MISMATCH: seed={}, constraints=({:?}, {:?})", seed, min_val, max_val);
                        println!("  Rust:   {}", rust_value);
                        println!("  Python: {}", py_value);
                    }
                    Ok(values_match)
                },
                Err(e) => {
                    println!("RUST ERROR: seed={}, error={:?}", seed, e);
                    Ok(false)
                }
            }
        })
    }

    /// Direct comparison test for draw_float operations with lexicographic byte verification
    pub fn verify_draw_float_encoding(&self, value: f64) -> PyResult<bool> {
        Python::with_gil(|py| {
            // Import the float encoding module directly
            let float_encoding = PyModule::import(py, "hypothesis.internal.conjecture.floats")?;
            let encode_to_index = float_encoding.getattr("float_to_index")?;
            let decode_from_index = float_encoding.getattr("float_from_index")?;
            
            // Get Python's encoding
            let py_index = encode_to_index.call1((value,))?;
            let py_index_value: u64 = py_index.extract()?;
            
            // Get Rust's encoding (using lexicographic encoding for ordering)
            let rust_index = conjecture::choice::indexing::float_encoding::float_to_lex(value);
            
            // Compare the indices (this is the critical lexicographic ordering test)
            let indices_match = rust_index == py_index_value;
            
            if !indices_match {
                println!("DISCREPANCY: Float encoding mismatch for {}", value);
                println!("  Python index: {}", py_index_value);
                println!("  Rust index:   {}", rust_index);
                return Ok(false);
            }
            
            // Verify round-trip works the same way
            let py_decoded = decode_from_index.call1((py_index_value,))?;
            let py_decoded_value: f64 = py_decoded.extract()?;
            let rust_decoded = conjecture::choice::indexing::float_encoding::lex_to_float(rust_index);
            
            // For NaN values, both should be NaN
            let round_trip_match = if value.is_nan() {
                py_decoded_value.is_nan() && rust_decoded.is_nan()
            } else {
                py_decoded_value == rust_decoded && py_decoded_value == value
            };
            
            if !round_trip_match {
                println!("DISCREPANCY: Float round-trip mismatch for {}", value);
                println!("  Original:     {}", value);
                println!("  Python:       {}", py_decoded_value);
                println!("  Rust:         {}", rust_decoded);
            }
            
            Ok(indices_match && round_trip_match)
        })
    }

    /// Direct comparison test for draw_boolean operations using same seed
    pub fn verify_draw_boolean(&self, seed: u64, p: Option<f64>) -> PyResult<bool> {
        Python::with_gil(|py| {
            // Create Rust ConjectureData with specific seed
            let mut rust_data = ConjectureData::new(seed);
            let rust_result = rust_data.draw_boolean(
                p.unwrap_or(0.5),
                None, // forced
                false // observe
            );

            // Create Python ConjectureData with same seed
            let conjecture_module = self.conjecture_module.bind(py);
            let conjecture_data_class = conjecture_module.getattr("ConjectureData")?;
            
            let py_data = conjecture_data_class.call1((seed,))?;
            
            // Call Python draw_boolean with same constraints
            let py_result = if let Some(prob) = p {
                py_data.call_method1("draw_boolean", (prob,))?
            } else {
                py_data.call_method0("draw_boolean")?
            };

            // Extract results and compare
            match rust_result {
                Ok(rust_value) => {
                    let py_value: bool = py_result.extract()?;
                    let values_match = rust_value == py_value;
                    if !values_match {
                        println!("BOOLEAN MISMATCH: seed={}, p={:?}", seed, p);
                        println!("  Rust:   {}", rust_value);
                        println!("  Python: {}", py_value);
                    }
                    Ok(values_match)
                },
                Err(e) => {
                    println!("RUST BOOLEAN ERROR: seed={}, error={:?}", seed, e);
                    Ok(false)
                }
            }
        })
    }

    /// Run comprehensive verification tests
    pub fn run_verification_suite(&self) -> PyResult<VerificationResults> {
        let mut results = VerificationResults::new();
        
        println!("Starting direct PyO3 verification...");
        
        // Test 1: Basic integer drawing
        println!("Testing draw_integer operations...");
        let test_seeds = [12345, 67890, 11111, 99999, 42];
        
        for &seed in &test_seeds {
            // Test unbounded integers
            match self.verify_draw_integer(seed, None, None) {
                Ok(true) => results.integer_tests_passed += 1,
                Ok(false) => {
                    results.integer_tests_failed += 1;
                    println!("  FAIL: Unbounded integer mismatch with seed {}", seed);
                },
                Err(e) => {
                    results.integer_tests_failed += 1;
                    println!("  ERROR: {}", e);
                }
            }
            
            // Test bounded integers
            match self.verify_draw_integer(seed, Some(-100), Some(100)) {
                Ok(true) => results.integer_tests_passed += 1,
                Ok(false) => {
                    results.integer_tests_failed += 1;
                    println!("  FAIL: Bounded integer mismatch with seed {}", seed);
                },
                Err(e) => {
                    results.integer_tests_failed += 1;
                    println!("  ERROR: {}", e);
                }
            }
        }
        
        // Test 2: Float encoding verification (this is critical for lexicographic ordering)
        println!("Testing float encoding lexicographic ordering...");
        let test_floats = [
            0.0, -0.0, 1.0, -1.0, 2.5, -2.5,
            f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
            f64::MIN, f64::MAX, f64::EPSILON,
            1.7976931348623157e308, // Near MAX
            2.2250738585072014e-308, // Near MIN_POSITIVE
        ];
        
        for &value in &test_floats {
            match self.verify_draw_float_encoding(value) {
                Ok(true) => results.float_tests_passed += 1,
                Ok(false) => {
                    results.float_tests_failed += 1;
                    // Error already printed in verify_draw_float_encoding
                },
                Err(e) => {
                    results.float_tests_failed += 1;
                    println!("  ERROR encoding {}: {}", value, e);
                }
            }
        }
        
        // Test 3: Boolean drawing
        println!("Testing draw_boolean operations...");
        for &seed in &test_seeds {
            // Test default probability
            match self.verify_draw_boolean(seed, None) {
                Ok(true) => results.boolean_tests_passed += 1,
                Ok(false) => {
                    results.boolean_tests_failed += 1;
                    println!("  FAIL: Boolean mismatch with seed {}", seed);
                },
                Err(e) => {
                    results.boolean_tests_failed += 1;
                    println!("  ERROR: {}", e);
                }
            }
            
            // Test custom probability
            match self.verify_draw_boolean(seed, Some(0.3)) {
                Ok(true) => results.boolean_tests_passed += 1,
                Ok(false) => {
                    results.boolean_tests_failed += 1;
                    println!("  FAIL: Boolean with p=0.3 mismatch with seed {}", seed);
                },
                Err(e) => {
                    results.boolean_tests_failed += 1;
                    println!("  ERROR: {}", e);
                }
            }
        }
        
        Ok(results)
    }
}

#[derive(Debug)]
pub struct VerificationResults {
    pub integer_tests_passed: u32,
    pub integer_tests_failed: u32,
    pub float_tests_passed: u32,
    pub float_tests_failed: u32,
    pub boolean_tests_passed: u32,
    pub boolean_tests_failed: u32,
}

impl VerificationResults {
    fn new() -> Self {
        Self {
            integer_tests_passed: 0,
            integer_tests_failed: 0,
            float_tests_passed: 0,
            float_tests_failed: 0,
            boolean_tests_passed: 0,
            boolean_tests_failed: 0,
        }
    }
    
    pub fn total_tests(&self) -> u32 {
        self.integer_tests_passed + self.integer_tests_failed +
        self.float_tests_passed + self.float_tests_failed +
        self.boolean_tests_passed + self.boolean_tests_failed
    }
    
    pub fn total_passed(&self) -> u32 {
        self.integer_tests_passed + self.float_tests_passed + self.boolean_tests_passed
    }
    
    pub fn total_failed(&self) -> u32 {
        self.integer_tests_failed + self.float_tests_failed + self.boolean_tests_failed
    }
    
    pub fn success_rate(&self) -> f64 {
        if self.total_tests() == 0 {
            0.0
        } else {
            self.total_passed() as f64 / self.total_tests() as f64
        }
    }
}