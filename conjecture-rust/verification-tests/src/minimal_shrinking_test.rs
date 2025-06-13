//! Minimal shrinking functionality test
//!
//! This module provides a standalone verification that the core shrinking
//! algorithms work correctly without depending on the full conjecture system.

use std::collections::HashMap;

/// Simple test structure that mimics ConjectureData for basic shrinking tests
#[derive(Debug, Clone)]
pub struct TestData {
    pub buffer: Vec<u8>,
    pub length: usize,
    pub test_passes: bool,
}

impl TestData {
    pub fn new(buffer: Vec<u8>) -> Self {
        Self {
            length: buffer.len(),
            buffer,
            test_passes: false,
        }
    }
}

/// Simple shrinking algorithm for testing
pub struct MinimalShrinker {
    original: TestData,
    best: TestData,
    attempts: usize,
}

impl MinimalShrinker {
    pub fn new(data: TestData) -> Self {
        Self {
            best: data.clone(),
            original: data,
            attempts: 0,
        }
    }
    
    /// Shrink by removing bytes from the end
    pub fn shrink_by_truncation<F>(&mut self, test_fn: F) -> TestData
    where
        F: Fn(&TestData) -> bool,
    {
        let mut current = self.best.clone();
        
        // Try progressively smaller buffers
        while current.buffer.len() > 1 {
            self.attempts += 1;
            
            // Remove one byte from the end
            current.buffer.pop();
            current.length = current.buffer.len();
            
            // Test if this still satisfies the condition
            if test_fn(&current) {
                self.best = current.clone();
                println!("SHRINKING: Found smaller buffer: {} bytes", current.buffer.len());
            } else {
                // This doesn't work, restore previous state
                current = self.best.clone();
                break;
            }
        }
        
        self.best.clone()
    }
    
    /// Shrink by reducing individual byte values
    pub fn shrink_by_reduction<F>(&mut self, test_fn: F) -> TestData
    where
        F: Fn(&TestData) -> bool,
    {
        let mut current = self.best.clone();
        let mut changed = true;
        
        while changed {
            changed = false;
            
            // Try to reduce each byte value
            for i in 0..current.buffer.len() {
                if current.buffer[i] > 0 {
                    self.attempts += 1;
                    
                    let original_val = current.buffer[i];
                    current.buffer[i] = current.buffer[i] / 2; // Try half the value
                    
                    if test_fn(&current) {
                        self.best = current.clone();
                        changed = true;
                        println!("SHRINKING: Reduced byte {} from {} to {}", 
                                i, original_val, current.buffer[i]);
                    } else {
                        // Restore original value
                        current.buffer[i] = original_val;
                    }
                }
            }
        }
        
        self.best.clone()
    }
    
    /// Combined shrinking strategy
    pub fn shrink<F>(&mut self, test_fn: F) -> TestData
    where
        F: Fn(&TestData) -> bool + Clone,
    {
        // First try truncation
        self.shrink_by_truncation(test_fn.clone());
        
        // Then try byte reduction
        self.shrink_by_reduction(test_fn);
        
        self.best.clone()
    }
}

/// Test results for verification
#[derive(Debug)]
pub struct MinimalShrinkingResult {
    pub test_name: String,
    pub original_size: usize,
    pub final_size: usize,
    pub shrink_attempts: usize,
    pub improvement_ratio: f64,
    pub success: bool,
}

/// Run minimal shrinking verification
pub fn verify_minimal_shrinking() -> Vec<MinimalShrinkingResult> {
    let mut results = Vec::new();
    
    // Test 1: Buffer sum > threshold
    {
        let buffer = vec![100, 100, 100, 100, 100, 50, 50, 50]; // Sum = 650
        let data = TestData::new(buffer);
        let mut shrinker = MinimalShrinker::new(data.clone());
        
        let test_fn = |data: &TestData| -> bool {
            let sum: u32 = data.buffer.iter().map(|&b| b as u32).sum();
            sum > 200  // Threshold that should allow shrinking
        };
        
        let final_data = shrinker.shrink(test_fn);
        
        results.push(MinimalShrinkingResult {
            test_name: "buffer_sum_threshold".to_string(),
            original_size: data.buffer.len(),
            final_size: final_data.buffer.len(),
            shrink_attempts: shrinker.attempts,
            improvement_ratio: 1.0 - (final_data.buffer.len() as f64 / data.buffer.len() as f64),
            success: final_data.buffer.len() < data.buffer.len(),
        });
    }
    
    // Test 2: Maximum byte value test
    {
        let buffer = vec![255, 200, 150, 100, 80]; // Max value = 255
        let data = TestData::new(buffer);
        let mut shrinker = MinimalShrinker::new(data.clone());
        
        let test_fn = |data: &TestData| -> bool {
            data.buffer.iter().any(|&b| b > 50)  // Any byte > 50
        };
        
        let final_data = shrinker.shrink(test_fn);
        
        let original_max = data.buffer.iter().max().copied().unwrap_or(0);
        let final_max = final_data.buffer.iter().max().copied().unwrap_or(0);
        
        results.push(MinimalShrinkingResult {
            test_name: "max_byte_value".to_string(),
            original_size: original_max as usize,
            final_size: final_max as usize,
            shrink_attempts: shrinker.attempts,
            improvement_ratio: 1.0 - (final_max as f64 / original_max as f64),
            success: final_max < original_max,
        });
    }
    
    // Test 3: Buffer length test
    {
        let buffer = vec![10; 20]; // 20 bytes of value 10
        let data = TestData::new(buffer);
        let mut shrinker = MinimalShrinker::new(data.clone());
        
        let test_fn = |data: &TestData| -> bool {
            data.buffer.len() > 5  // Length must be > 5
        };
        
        let final_data = shrinker.shrink(test_fn);
        
        results.push(MinimalShrinkingResult {
            test_name: "buffer_length".to_string(),
            original_size: data.buffer.len(),
            final_size: final_data.buffer.len(),
            shrink_attempts: shrinker.attempts,
            improvement_ratio: 1.0 - (final_data.buffer.len() as f64 / data.buffer.len() as f64),
            success: final_data.buffer.len() < data.buffer.len() && final_data.buffer.len() > 5,
        });
    }
    
    results
}

/// Summary of minimal verification results
#[derive(Debug)]
pub struct MinimalVerificationSummary {
    pub total_tests: usize,
    pub successful_shrinks: usize,
    pub average_improvement: f64,
    pub total_attempts: usize,
}

impl MinimalVerificationSummary {
    pub fn from_results(results: &[MinimalShrinkingResult]) -> Self {
        let total_tests = results.len();
        let successful_shrinks = results.iter().filter(|r| r.success).count();
        let average_improvement = if total_tests > 0 {
            results.iter().map(|r| r.improvement_ratio).sum::<f64>() / total_tests as f64
        } else {
            0.0
        };
        let total_attempts = results.iter().map(|r| r.shrink_attempts).sum();
        
        Self {
            total_tests,
            successful_shrinks,
            average_improvement,
            total_attempts,
        }
    }
}

/// Main verification entry point
pub fn run_minimal_verification() -> MinimalVerificationSummary {
    let results = verify_minimal_shrinking();
    
    println!("🧪 Minimal Shrinking Verification Results:");
    println!("==========================================");
    
    for result in &results {
        println!("\n📋 Test: {}", result.test_name);
        println!("   Original size: {}", result.original_size);
        println!("   Final size: {}", result.final_size);
        println!("   Attempts: {}", result.shrink_attempts);
        println!("   Improvement: {:.1}%", result.improvement_ratio * 100.0);
        
        if result.success {
            println!("   ✅ Shrinking successful");
        } else {
            println!("   ❌ Shrinking failed");
        }
    }
    
    let summary = MinimalVerificationSummary::from_results(&results);
    
    println!("\n📊 Summary:");
    println!("   Total tests: {}", summary.total_tests);
    println!("   Successful: {}", summary.successful_shrinks);
    println!("   Success rate: {:.1}%", 
             (summary.successful_shrinks as f64 / summary.total_tests as f64) * 100.0);
    println!("   Average improvement: {:.1}%", summary.average_improvement * 100.0);
    println!("   Total attempts: {}", summary.total_attempts);
    
    summary
}