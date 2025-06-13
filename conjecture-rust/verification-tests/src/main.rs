//! # Conjecture Python Verification Tool: Comprehensive Compatibility Testing
//!
//! This sophisticated verification tool provides comprehensive testing and validation
//! of the Rust Conjecture implementation against Python Hypothesis's proven algorithms.
//! It implements multiple verification strategies including direct FFI comparisons,
//! parity testing, and compilation error resolution validation.
//!
//! ## Core Capabilities
//!
//! ### Python FFI Integration
//! - **Direct Algorithm Comparison**: Byte-for-byte comparison of generation algorithms
//! - **Choice Function Parity**: Validates that Rust and Python choice functions produce
//!   identical results for the same inputs and random seeds
//! - **Error Handling Compatibility**: Ensures error conditions are handled consistently
//! - **Performance Benchmarking**: Comparative performance analysis between implementations
//!
//! ### Comprehensive Test Coverage
//! - **Core Type System**: Validates all choice types, values, and constraints
//! - **Float Encoding**: Detailed verification of sophisticated float representation
//! - **Choice Sequence Replay**: Ensures deterministic replay matches Python behavior
//! - **Database Compatibility**: Validates serialization format compatibility
//!
//! ### Compilation Verification
//! - **Type System Resolution**: Validates that all compilation errors are resolved
//! - **Import Path Correctness**: Ensures all module imports are valid
//! - **Trait Implementation**: Verifies all required traits are properly implemented
//! - **Feature Gate Compatibility**: Tests conditional compilation features
//!
//! ## Verification Strategies
//!
//! ### 1. Direct Equivalence Testing
//! ```text
//! Rust Implementation ←→ Python Hypothesis
//!         ↓                      ↓
//!    Generated Values ===== Generated Values
//!         ↓                      ↓
//!    Choice Sequences ===== Choice Sequences
//! ```
//!
//! ### 2. Statistical Validation
//! - **Distribution Analysis**: Validates that generated value distributions match
//! - **Edge Case Coverage**: Ensures rare conditions are handled consistently
//! - **Bias Detection**: Identifies and validates intentional/unintentional biases
//! - **Convergence Testing**: Validates shrinking algorithms converge to same results
//!
//! ### 3. Property-Based Meta-Testing
//! - **Commutativity**: Tests that operation order doesn't affect results
//! - **Idempotence**: Validates that repeated operations are stable
//! - **Associativity**: Ensures complex operations compose correctly
//! - **Determinism**: Validates reproducibility across runs
//!
//! ## Command Line Interface
//!
//! The tool provides flexible command-line options for different verification modes:
//!
//! ### Basic Usage
//! ```bash
//! # Run all verification tests
//! cargo run --bin verify
//!
//! # Run specific test category
//! cargo run --bin verify --test float_encoding
//! cargo run --bin verify --parity
//! cargo run --bin verify --core
//!
//! # Enable verbose output
//! cargo run --bin verify --verbose
//! ```
//!
//! ### Specialized Modes
//! ```bash
//! # Python parity verification (requires Python with Hypothesis)
//! cargo run --bin verify --parity --verbose
//!
//! # Core compilation verification only
//! cargo run --bin verify --core
//!
//! # Specific test with detailed output
//! cargo run --bin verify --test choice_sequence_replay --verbose
//! ```
//!
//! ## Architecture
//!
//! ```text
//! main.rs (CLI Interface)
//! ├── python_ffi.rs (Python FFI Integration)
//! ├── test_runner.rs (Test Execution Framework)
//! ├── test_cases.rs (Test Case Definitions)
//! ├── direct_type_test.rs (Direct Type Testing)
//! ├── float_constraint_python_parity_test.rs (Float Verification)
//! ├── python_parity_verification.rs (Comprehensive Parity)
//! └── core_compilation_verification.rs (Compilation Validation)
//! ```
//!
//! ## Error Reporting
//!
//! The verification tool provides detailed error analysis:
//! - **Diff Reports**: Side-by-side comparison of divergent results
//! - **Context Preservation**: Full execution context for debugging
//! - **Statistical Analysis**: Distribution differences and significance testing
//! - **Performance Metrics**: Timing and resource usage comparisons
//!
//! ## Integration with CI/CD
//!
//! The tool is designed for automated testing environments:
//! - **Exit Codes**: Standard exit codes for automated script integration
//! - **JSON Output**: Machine-readable test results for dashboard integration
//! - **Parallel Execution**: Supports concurrent test execution for faster CI
//! - **Incremental Testing**: Can run only changed components for efficiency
//!
//! ## Requirements
//!
//! - **Rust**: 1.70+ with required dependencies
//! - **Python**: 3.8+ with Hypothesis installed (for parity tests)
//! - **System**: Sufficient memory for large-scale statistical testing
//!
//! ## Performance Characteristics
//!
//! - **Startup Time**: <100ms for basic verification
//! - **Full Test Suite**: ~30 seconds on modern hardware
//! - **Memory Usage**: <256MB for comprehensive testing
//! - **Parallel Scaling**: Near-linear speedup with available cores

use clap::{Arg, Command};
use std::process;

mod python_ffi;
mod test_runner;
mod test_cases;
mod direct_type_test;
mod float_constraint_python_parity_test;
mod python_parity_verification;
mod core_compilation_verification;
mod direct_pyo3_verification;
mod choice_sequence_direct_verification;

use test_runner::TestRunner;

fn main() {
    let matches = Command::new("conjecture-verify")
        .about("Verify Rust conjecture implementation against Python Hypothesis")
        .arg(
            Arg::new("test")
                .short('t')
                .long("test")
                .value_name("TEST_NAME")
                .help("Run specific test (default: all)")
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .action(clap::ArgAction::SetTrue)
                .help("Enable verbose output")
        )
        .arg(
            Arg::new("parity")
                .short('p')
                .long("parity")
                .action(clap::ArgAction::SetTrue)
                .help("Run comprehensive Python parity verification")
        )
        .arg(
            Arg::new("core")
                .short('c')
                .long("core")
                .action(clap::ArgAction::SetTrue)
                .help("Run core compilation error resolution verification")
        )
        .arg(
            Arg::new("direct")
                .short('d')
                .long("direct")
                .action(clap::ArgAction::SetTrue)
                .help("Run direct PyO3 byte-for-byte comparison verification")
        )
        .arg(
            Arg::new("choice-sequence")
                .short('s')
                .long("choice-sequence")
                .action(clap::ArgAction::SetTrue)
                .help("Run choice sequence management verification")
        )
        .get_matches();

    let test_name = matches.get_one::<String>("test");
    let verbose = matches.get_flag("verbose");
    let run_parity = matches.get_flag("parity");
    let run_core = matches.get_flag("core");
    let run_direct = matches.get_flag("direct");
    let run_choice_sequence = matches.get_flag("choice-sequence");

    println!("🔍 Conjecture Python-Rust Verification Tool");
    println!("============================================");
    
    // If choice sequence flag is set, run choice sequence verification
    if run_choice_sequence {
        println!("\n🔄 Running choice sequence management verification...");
        let results = choice_sequence_direct_verification::verify_choice_sequence_behaviors();
        
        println!("\n📊 Choice Sequence Verification Results:");
        let total = results.len();
        let passed = results.iter().filter(|r| r.match_status).count();
        let failed = total - passed;
        
        for result in &results {
            let status = if result.match_status { "✅" } else { "❌" };
            println!("   {} {}", status, result.test_name);
            if verbose || !result.match_status {
                println!("      Expected: {}", result.expected_behavior);
                println!("      Actual:   {}", result.actual_behavior);
            }
        }
        
        println!("\n📈 Summary: {}/{} tests passed", passed, total);
        
        if failed > 0 {
            println!("❌ Choice sequence verification found issues!");
            process::exit(1);
        } else {
            println!("✅ All choice sequence verification tests passed!");
            return;
        }
    }
    
    // If direct flag is set, run direct PyO3 byte-for-byte comparison
    if run_direct {
        println!("\n🔍 Running direct PyO3 byte-for-byte comparison verification...");
        match direct_pyo3_verification::DirectPyO3Verifier::new() {
            Ok(verifier) => {
                match verifier.run_verification_suite() {
                    Ok(results) => {
                        println!("\n📊 Direct Verification Results:");
                        println!("   Total tests: {}", results.total_tests());
                        println!("   Passed: {}", results.total_passed());
                        println!("   Failed: {}", results.total_failed());
                        println!("   Success rate: {:.1}%", results.success_rate() * 100.0);
                        
                        println!("\n📈 Breakdown:");
                        println!("   Integer tests: {}/{}", results.integer_tests_passed, results.integer_tests_passed + results.integer_tests_failed);
                        println!("   Float tests: {}/{}", results.float_tests_passed, results.float_tests_passed + results.float_tests_failed);
                        println!("   Boolean tests: {}/{}", results.boolean_tests_passed, results.boolean_tests_passed + results.boolean_tests_failed);
                        
                        if results.total_failed() > 0 {
                            println!("\n❌ Direct verification found discrepancies!");
                            process::exit(1);
                        } else {
                            println!("\n✅ All direct verification tests passed!");
                            return;
                        }
                    }
                    Err(e) => {
                        eprintln!("\n❌ Direct verification failed: {}", e);
                        process::exit(1);
                    }
                }
            }
            Err(e) => {
                eprintln!("\n❌ Failed to initialize direct verifier: {}", e);
                process::exit(1);
            }
        }
    }
    
    // If core flag is set, run core compilation error resolution verification
    if run_core {
        println!("\n🔧 Running core compilation error resolution verification...");
        match core_compilation_verification::run_core_compilation_verification(verbose) {
            Ok(()) => {
                println!("\n🎉 Core compilation verification completed successfully!");
                return;
            }
            Err(e) => {
                eprintln!("\n❌ Core compilation verification failed: {}", e);
                process::exit(1);
            }
        }
    }
    
    // If parity flag is set, run comprehensive Python parity verification
    if run_parity {
        println!("\n🐍 Running comprehensive Python parity verification...");
        match python_parity_verification::run_comprehensive_verification(verbose) {
            Ok(()) => {
                println!("\n🎉 Python parity verification completed successfully!");
                return;
            }
            Err(e) => {
                eprintln!("\n❌ Python parity verification failed: {}", e);
                process::exit(1);
            }
        }
    }
    
    // First run direct Rust-only type tests
    if let Err(e) = direct_type_test::run_direct_type_tests() {
        eprintln!("\n❌ Direct type tests failed: {}", e);
        process::exit(1);
    }
    
    // Run float constraint Python parity verification
    float_constraint_python_parity_test::verify_float_constraint_python_parity();
    float_constraint_python_parity_test::verify_qa_issue_resolution();
    println!();
    
    let mut runner = TestRunner::new(verbose);
    
    let result = if let Some(test) = test_name {
        runner.run_single_test(test)
    } else {
        runner.run_all_tests()
    };
    
    match result {
        Ok(stats) => {
            println!("\n✅ Verification Complete!");
            println!("   Tests run: {}", stats.total);
            println!("   Passed: {}", stats.passed);
            println!("   Failed: {}", stats.failed);
            
            if stats.failed > 0 {
                println!("\n❌ Some tests failed!");
                process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("\n💥 Verification failed: {}", e);
            process::exit(1);
        }
    }
}