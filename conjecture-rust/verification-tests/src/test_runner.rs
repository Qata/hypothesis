//! Test runner for verification tests

use crate::python_ffi::PythonInterface;
use crate::test_cases::{get_all_test_suites, TestCase, TestSuite};
use conjecture::choice::{choice_to_index, choice_from_index, choice_equal, ChoiceValue};
use pyo3::PyResult;
use std::fmt;

/// Test execution statistics
#[derive(Debug, Default)]
pub struct TestStats {
    pub total: u32,
    pub passed: u32,
    pub failed: u32,
}

/// Test execution result
#[derive(Debug)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub error: Option<String>,
    pub rust_index: Option<u128>,
    pub python_index: Option<u128>,
    pub roundtrip_passed: bool,
}

impl fmt::Display for TestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.passed {
            write!(f, "✅ {}", self.name)
        } else {
            write!(f, "❌ {} - {}", self.name, self.error.as_ref().unwrap_or(&"Unknown error".to_string()))
        }
    }
}

/// Main test runner
pub struct TestRunner {
    python: PythonInterface,
    verbose: bool,
}

impl TestRunner {
    /// Create new test runner
    pub fn new(verbose: bool) -> Self {
        let python = PythonInterface::new()
            .expect("Failed to initialize Python interface");
        
        Self { python, verbose }
    }

    /// Run all verification tests
    pub fn run_all_tests(&mut self) -> Result<TestStats, Box<dyn std::error::Error>> {
        let mut stats = TestStats::default();
        
        // First run Float Constraint Type System Consistency tests
        crate::test_cases::run_float_constraint_type_consistency_tests();
        println!();
        
        let test_suites = get_all_test_suites();

        println!("Running {} test suites...\n", test_suites.len());

        for suite in test_suites {
            self.run_test_suite(&suite, &mut stats)?;
        }

        Ok(stats)
    }

    /// Run a single named test
    pub fn run_single_test(&mut self, test_name: &str) -> Result<TestStats, Box<dyn std::error::Error>> {
        let mut stats = TestStats::default();
        let test_suites = get_all_test_suites();

        for suite in test_suites {
            for case in suite.cases {
                if case.name == test_name {
                    println!("Running test: {}", test_name);
                    let result = self.run_test_case(&case)?;
                    self.process_result(&result, &mut stats);
                    return Ok(stats);
                }
            }
        }

        return Err(format!("Test '{}' not found", test_name).into());
    }

    /// Run a complete test suite
    fn run_test_suite(&mut self, suite: &TestSuite, stats: &mut TestStats) -> Result<(), Box<dyn std::error::Error>> {
        println!("📋 Test Suite: {}", suite.name);
        println!("   {} tests", suite.cases.len());

        for case in &suite.cases {
            let result = self.run_test_case(case)?;
            self.process_result(&result, stats);
            
            if self.verbose {
                println!("   {}", result);
                if let (Some(rust_idx), Some(python_idx)) = (result.rust_index, result.python_index) {
                    println!("      Rust index: {}, Python index: {}", rust_idx, python_idx);
                }
            } else if !result.passed {
                println!("   {}", result);
            }
        }

        println!();
        Ok(())
    }

    /// Run a single test case
    fn run_test_case(&self, case: &TestCase) -> PyResult<TestResult> {
        if self.verbose {
            println!("🧪 Running: {} - {}", case.name, case.description);
        }

        // Get results from both implementations
        let rust_index = choice_to_index(&case.value, &case.constraints);
        let python_index_result = self.python.choice_to_index(&case.value, &case.constraints);
        
        let python_index = match python_index_result {
            Ok(idx) => idx,
            Err(e) => {
                return Ok(TestResult {
                    name: case.name.clone(),
                    passed: false,
                    error: Some(format!("Python choice_to_index failed: {}", e)),
                    rust_index: Some(rust_index),
                    python_index: None,
                    roundtrip_passed: false,
                });
            }
        };

        // Check if indices match
        let indices_match = rust_index == python_index;
        
        // Test roundtrip properties
        let roundtrip_passed = self.test_roundtrip(case, rust_index, python_index);

        // Check expected properties
        let properties_passed = self.check_expected_properties(case, rust_index, python_index);

        let passed = indices_match && roundtrip_passed && properties_passed;
        let error = if !passed {
            Some(self.build_error_message(indices_match, roundtrip_passed, properties_passed))
        } else {
            None
        };

        Ok(TestResult {
            name: case.name.clone(),
            passed,
            error,
            rust_index: Some(rust_index),
            python_index: Some(python_index),
            roundtrip_passed,
        })
    }

    /// Test roundtrip property: choice_to_index -> choice_from_index should recover original value
    fn test_roundtrip(&self, case: &TestCase, rust_index: u128, python_index: u128) -> bool {
        // Test Rust roundtrip
        let choice_type = self.get_choice_type_str(&case.value);
        let rust_recovered = choice_from_index(rust_index, choice_type, &case.constraints);
        let rust_roundtrip = choice_equal(&case.value, &rust_recovered);

        // Test Python roundtrip
        let python_recovered_result = self.python.choice_from_index(python_index, choice_type, &case.constraints);
        let python_roundtrip = match python_recovered_result {
            Ok(recovered) => {
                match self.python.choice_equal(&case.value, &recovered) {
                    Ok(equal) => equal,
                    Err(_) => false,
                }
            },
            Err(_) => false,
        };

        rust_roundtrip && python_roundtrip
    }

    /// Check expected properties like specific index values or sign bits
    fn check_expected_properties(&self, case: &TestCase, rust_index: u128, python_index: u128) -> bool {
        for property in &case.expected_properties {
            match property.as_str() {
                "index_zero" => {
                    if rust_index != 0 || python_index != 0 {
                        return false;
                    }
                },
                "index_one" => {
                    if rust_index != 1 || python_index != 1 {
                        return false;
                    }
                },
                "index_two" => {
                    if rust_index != 2 || python_index != 2 {
                        return false;
                    }
                },
                "sign_bit_zero" => {
                    if (rust_index >> 64) != 0 || (python_index >> 64) != 0 {
                        return false;
                    }
                },
                "sign_bit_one" => {
                    if (rust_index >> 64) == 0 || (python_index >> 64) == 0 {
                        return false;
                    }
                },
                "roundtrip" => {
                    // Already tested in test_roundtrip
                },
                _ => {
                    if self.verbose {
                        println!("   Warning: Unknown property '{}'", property);
                    }
                }
            }
        }
        true
    }

    /// Get choice type string for choice_from_index calls
    fn get_choice_type_str(&self, value: &ChoiceValue) -> &'static str {
        match value {
            ChoiceValue::Integer(_) => "integer",
            ChoiceValue::Boolean(_) => "boolean",
            ChoiceValue::Float(_) => "float",
            ChoiceValue::String(_) => "string",
            ChoiceValue::Bytes(_) => "bytes",
        }
    }

    /// Build detailed error message
    fn build_error_message(&self, indices_match: bool, roundtrip_passed: bool, properties_passed: bool) -> String {
        let mut errors = Vec::new();
        
        if !indices_match {
            errors.push("Index mismatch");
        }
        if !roundtrip_passed {
            errors.push("Roundtrip failed");
        }
        if !properties_passed {
            errors.push("Expected properties failed");
        }
        
        errors.join(", ")
    }

    /// Process test result and update statistics
    fn process_result(&self, result: &TestResult, stats: &mut TestStats) {
        stats.total += 1;
        if result.passed {
            stats.passed += 1;
        } else {
            stats.failed += 1;
        }
    }
}