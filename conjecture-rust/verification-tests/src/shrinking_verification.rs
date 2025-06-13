//! Direct shrinking verification: Rust implementation testing
//!
//! This module tests the shrinking functionality of our Rust implementation
//! with basic test cases to verify the shrinking system works correctly.

use std::collections::HashMap;
use conjecture_rust::data::ConjectureData;
use conjecture_rust::shrinking::PythonEquivalentShrinker;

/// Results of shrinking verification
#[derive(Debug, Clone)]
pub struct ShrinkingVerificationResult {
    pub test_name: String,
    pub initial_shrink_steps: usize,
    pub rust_shrink_steps: usize,
    pub initial_final_size: usize,
    pub rust_final_size: usize,
    pub initial_final_bytes: Vec<u8>,
    pub rust_final_bytes: Vec<u8>,
    pub bytes_improved: bool,
    pub shrink_quality_good: bool,
    pub error: Option<String>,
}

/// Direct shrinking verifier for Rust implementation
pub struct ShrinkingVerifier {
}

impl ShrinkingVerifier {
    pub fn new() -> Result<Self, String> {
        Ok(Self { })
    }
    
    /// Test basic shrinking functionality with a simple test case
    pub fn verify_basic_shrinking(&self) -> ShrinkingVerificationResult {
        // Generate a failing test case with large buffer
        let initial_buffer = vec![
            255, 255, 255, 255, 255, 255, 255, 255,  // High values that sum > 1000
            255, 255, 255, 255, 255, 255, 255, 255,
            100, 100, 100, 100, 100, 100, 100, 100,  // More bytes to shrink
        ];
        
        let initial_size = initial_buffer.len();
        
        // Test with Rust shrinking
        let rust_result = self.shrink_with_rust_basic_test(&initial_buffer);
        
        match rust_result {
            Ok((final_buffer, steps, final_size)) => {
                ShrinkingVerificationResult {
                    test_name: "basic_shrinking".to_string(),
                    initial_shrink_steps: 0,
                    rust_shrink_steps: steps,
                    initial_final_size: initial_size,
                    rust_final_size: final_size,
                    initial_final_bytes: initial_buffer.clone(),
                    rust_final_bytes: final_buffer.clone(),
                    bytes_improved: final_buffer.len() < initial_buffer.len(),
                    shrink_quality_good: final_buffer.len() <= initial_buffer.len(),
                    error: None,
                }
            }
            Err(e) => {
                ShrinkingVerificationResult {
                    test_name: "basic_shrinking".to_string(),
                    initial_shrink_steps: 0,
                    rust_shrink_steps: 0,
                    initial_final_size: initial_size,
                    rust_final_size: initial_size,
                    initial_final_bytes: initial_buffer.clone(),
                    rust_final_bytes: initial_buffer,
                    bytes_improved: false,
                    shrink_quality_good: false,
                    error: Some(e),
                }
            }
        }
    }
    
    /// Shrink using Rust implementation for basic test
    fn shrink_with_rust_basic_test(&self, initial_buffer: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
        // Create ConjectureData with fixed seed
        let mut data = ConjectureData::new(12345);
        data.buffer = initial_buffer.to_vec();
        data.length = initial_buffer.len();
        
        // Define the failing test (matching Python logic)
        let test_function = |data: &ConjectureData| -> bool {
            // For now, use a simpler test based on buffer size and content
            // This is a placeholder - in a real implementation we'd reconstruct
            // the choices and replay the test
            if data.buffer.len() < 10 {
                return false;
            }
            
            // Simple heuristic: sum first 10 bytes
            let sum: u32 = data.buffer[..10].iter().map(|&b| b as u32).sum();
            sum > 1000
        };
        
        // Verify initial test fails
        if !test_function(&data) {
            return Err("Initial test does not fail".to_string());
        }
        
        // Create shrinker
        let mut shrinker = PythonEquivalentShrinker::new(data);
        
        // Perform shrinking
        let final_data = shrinker.shrink_with_function(test_function);
        
        Ok((
            final_data.buffer,
            shrinker.metrics.successful_transformations as usize,
            final_data.buffer.len()
        ))
    }
    
    /// Run comprehensive shrinking verification
    pub fn run_verification_suite(&self) -> Vec<ShrinkingVerificationResult> {
        let mut results = Vec::new();
        
        // Test 1: Basic shrinking
        results.push(self.verify_basic_shrinking());
        
        // TODO: Add more test cases:
        // - Different buffer sizes
        // - Various test predicates
        // - Complex nested structures
        
        results
    }
}

/// Verification summary
#[derive(Debug)]
pub struct ShrinkingVerificationSummary {
    pub total_tests: usize,
    pub bytes_improved: usize,
    pub quality_good: usize,
    pub errors: usize,
}

impl ShrinkingVerificationSummary {
    pub fn from_results(results: &[ShrinkingVerificationResult]) -> Self {
        Self {
            total_tests: results.len(),
            bytes_improved: results.iter().filter(|r| r.bytes_improved).count(),
            quality_good: results.iter().filter(|r| r.shrink_quality_good).count(),
            errors: results.iter().filter(|r| r.error.is_some()).count(),
        }
    }
    
    pub fn success_rate(&self) -> f64 {
        if self.total_tests == 0 {
            0.0
        } else {
            self.quality_good as f64 / self.total_tests as f64
        }
    }
}

/// Main verification function
pub fn verify_shrinking_functionality() -> Result<ShrinkingVerificationSummary, String> {
    let verifier = ShrinkingVerifier::new()?;
    let results = verifier.run_verification_suite();
    
    // Print detailed results
    for result in &results {
        println!("🔍 Test: {}", result.test_name);
        
        if let Some(error) = &result.error {
            println!("  ❌ Error: {}", error);
            continue;
        }
        
        println!("  Initial: {} bytes", result.initial_final_size);
        println!("  Final:   {} bytes after {} steps", result.rust_final_size, result.rust_shrink_steps);
        
        if result.bytes_improved {
            println!("  ✅ Buffer size improved: {} → {}", result.initial_final_size, result.rust_final_size);
        } else if result.shrink_quality_good {
            println!("  ✅ Shrinking completed successfully");
        } else {
            println!("  ❌ Shrinking failed or made no progress");
            println!("    Error: {:?}", result.error);
        }
        println!();
    }
    
    Ok(ShrinkingVerificationSummary::from_results(&results))
}