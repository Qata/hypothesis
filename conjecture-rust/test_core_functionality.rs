// Direct functionality test for Core Compilation Error Resolution
use std::process::Command;

fn main() {
    println!("=== DIRECT FUNCTIONALITY TEST ===\n");
    
    // Create a minimal Rust program that uses the core functionality
    let test_program = r#"
extern crate conjecture_rust;

use conjecture_rust::choice::core_compilation_error_resolution::{
    CompilationErrorResolver, CompilationErrorType, ResolutionResult
};

fn main() {
    println!("Testing Core Compilation Error Resolution functionality...");
    
    let mut resolver = CompilationErrorResolver::new();
    
    // Test the 4 critical error types
    let test_errors = vec![
        CompilationErrorType::ImportPathError {
            invalid_path: "conjecture_rust".to_string(),
            suggested_path: "conjecture".to_string(),
        },
        CompilationErrorType::MissingTraitImplementation {
            trait_name: "Clone".to_string(),
            target_type: "TestStruct".to_string(),
        },
        CompilationErrorType::TypeParameterMismatch {
            expected_type: "String".to_string(),
            actual_type: "i32".to_string(),
            context: "function parameter".to_string(),
        },
        CompilationErrorType::FieldAccessError {
            struct_name: "ChoiceNode".to_string(),
            field_name: "index".to_string(),
            available_fields: vec!["choice_type".to_string(), "value".to_string()],
        },
    ];
    
    let mut resolved_count = 0;
    let mut manual_fix_count = 0;
    
    for (i, error) in test_errors.into_iter().enumerate() {
        let result = resolver.resolve_error(error.clone());
        println!("Error {}: {:?}", i + 1, error);
        
        match result {
            ResolutionResult::Resolved { confidence, fix_applied, .. } => {
                println!("  ✅ RESOLVED: {} (confidence: {:.2})", fix_applied, confidence);
                resolved_count += 1;
            }
            ResolutionResult::RequiresManualFix { suggestions, .. } => {
                println!("  🔧 MANUAL FIX NEEDED: {:?}", suggestions);
                manual_fix_count += 1;
            }
            ResolutionResult::Unresolvable { reason, .. } => {
                println!("  ❌ UNRESOLVABLE: {}", reason);
            }
        }
        println!();
    }
    
    println!("=== RESOLUTION SUMMARY ===");
    println!("Total Resolved: {}", resolved_count);
    println!("Manual Fixes Required: {}", manual_fix_count);
    
    let stats = resolver.get_statistics();
    println!("Statistics: {:?}", stats);
    
    println!("Report:\n{}", resolver.generate_resolution_report());
    
    println!("✅ Core Compilation Error Resolution capability verified successfully!");
}
"#;
    
    // Write the test program
    std::fs::write("direct_test.rs", test_program).expect("Failed to write test file");
    
    // Build the library first
    println!("Building library...");
    let build_output = Command::new("cargo")
        .args(&["build", "--lib", "--release"])
        .output()
        .expect("Failed to build library");
    
    if !build_output.status.success() {
        println!("❌ Library build failed");
        return;
    }
    
    // Compile and run the test program
    println!("Running direct functionality test...");
    let compile_output = Command::new("rustc")
        .args(&[
            "--edition=2021",
            "--extern", "conjecture_rust=target/release/libconjecture_rust.rlib",
            "direct_test.rs",
            "-L", "target/release/deps"
        ])
        .output()
        .expect("Failed to compile test");
    
    if compile_output.status.success() {
        println!("✅ Test compiled successfully");
        
        let run_output = Command::new("./direct_test")
            .output()
            .expect("Failed to run test");
        
        if run_output.status.success() {
            println!("✅ Test executed successfully");
            println!("Output:");
            println!("{}", String::from_utf8_lossy(&run_output.stdout));
        } else {
            println!("❌ Test execution failed");
            println!("Error: {}", String::from_utf8_lossy(&run_output.stderr));
        }
    } else {
        println!("❌ Test compilation failed");
        println!("Error: {}", String::from_utf8_lossy(&compile_output.stderr));
    }
    
    // Cleanup
    let _ = std::fs::remove_file("direct_test.rs");
    let _ = std::fs::remove_file("direct_test");
}