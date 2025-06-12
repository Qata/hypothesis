//! Conjecture Python Verification Tool
//! 
//! This tool runs comprehensive verification tests between our Rust conjecture
//! implementation and Python Hypothesis's actual choice functions via FFI.

use clap::{Arg, Command};
use std::process;

mod python_ffi;
mod test_runner;
mod test_cases;
mod direct_type_test;
mod float_constraint_python_parity_test;

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
        .get_matches();

    let test_name = matches.get_one::<String>("test");
    let verbose = matches.get_flag("verbose");

    println!("🔍 Conjecture Python-Rust Verification Tool");
    println!("============================================");
    
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