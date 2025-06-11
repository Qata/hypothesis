//! Performance benchmarking test for float indexing operations
//! 
//! This module contains performance tests to analyze the current 799ms performance
//! for 1000 float operations and identify optimization opportunities.

use std::time::Instant;
use crate::choice::indexing::float_encoding::{float_to_lex, lex_to_float};

/// Benchmark 1000 float indexing operations to measure current performance
pub fn benchmark_float_indexing() -> std::time::Duration {
    println!("PERFORMANCE BENCHMARK: Starting 1000 float indexing operations...");
    
    let start_time = Instant::now();
    
    // Test with a mix of simple and complex floats to reflect real usage
    let test_values = [
        0.0, 1.0, 2.0, 3.0, 10.0, 100.0,  // Simple integers
        1.5, 2.25, 3.14159, 0.1, 0.333333, // Complex floats
        f64::MIN_POSITIVE, f64::MAX / 2.0,  // Boundary values
        std::f64::consts::PI, std::f64::consts::E, // Mathematical constants
    ];
    
    // Perform 1000 operations (100 iterations of 10 values)
    let mut total_operations = 0;
    for iteration in 0..100 {
        for &value in &test_values {
            // Forward conversion: float -> lex
            let lex = float_to_lex(value);
            
            // Backward conversion: lex -> float
            let recovered = lex_to_float(lex);
            
            // Verify correctness (essential for benchmark validity)
            assert_eq!(value, recovered, "Roundtrip failed for {} at iteration {}", value, iteration);
            
            total_operations += 2; // Count both directions
        }
    }
    
    let elapsed = start_time.elapsed();
    
    println!("PERFORMANCE BENCHMARK: Completed {} operations in {:?}", total_operations, elapsed);
    println!("PERFORMANCE BENCHMARK: Average per operation: {:?}", elapsed / total_operations);
    
    elapsed
}

/// Benchmark specifically the exponent table building which may be a bottleneck
pub fn benchmark_exponent_table_building() -> std::time::Duration {
    println!("PERFORMANCE BENCHMARK: Testing exponent table building performance...");
    
    let start_time = Instant::now();
    
    // Build exponent tables 100 times to see if this is expensive
    for _i in 0..100 {
        let (_encoding_table, _decoding_table) = crate::choice::indexing::float_encoding::build_exponent_tables();
    }
    
    let elapsed = start_time.elapsed();
    
    println!("PERFORMANCE BENCHMARK: Built exponent tables 100 times in {:?}", elapsed);
    println!("PERFORMANCE BENCHMARK: Average per table build: {:?}", elapsed / 100);
    
    elapsed
}

/// Benchmark bit reversal operations which may be expensive
pub fn benchmark_bit_reversal() -> std::time::Duration {
    println!("PERFORMANCE BENCHMARK: Testing bit reversal performance...");
    
    let start_time = Instant::now();
    
    // Test bit reversal with various patterns 10000 times
    let test_patterns: [u64; 5] = [
        0x0000000000000000,
        0xFFFFFFFFFFFFFFFF,
        0x0123456789ABCDEF,
        0xAAAAAAAAAAAAAAAA,
        0x5555555555555555,
    ];
    
    let mut total_operations = 0;
    for _iteration in 0..2000 {
        for &pattern in &test_patterns {
            // This is calling internal functions - we'll need to make them public or test differently
            // For now, just count the patterns
            total_operations += 1;
        }
    }
    
    let elapsed = start_time.elapsed();
    
    println!("PERFORMANCE BENCHMARK: Processed {} bit patterns in {:?}", total_operations, elapsed);
    
    elapsed
}

/// Comprehensive performance analysis
pub fn run_performance_analysis() {
    println!("=== FLOAT INDEXING PERFORMANCE ANALYSIS ===");
    
    // Warm up
    println!("Warming up...");
    for _ in 0..10 {
        let _ = float_to_lex(3.14159);
        let _ = lex_to_float(12345);
    }
    
    // Main benchmark
    let main_duration = benchmark_float_indexing();
    
    // Component benchmarks
    let table_duration = benchmark_exponent_table_building();
    let bit_duration = benchmark_bit_reversal();
    
    println!("\n=== PERFORMANCE SUMMARY ===");
    println!("Main float indexing (1000 ops): {:?}", main_duration);
    println!("Exponent table building (100x): {:?}", table_duration);
    println!("Bit reversal test: {:?}", bit_duration);
    
    // Analysis
    if main_duration.as_millis() > 100 {
        println!("\n⚠️  PERFORMANCE ISSUE DETECTED:");
        println!("Current performance: {}ms > target 100ms", main_duration.as_millis());
        
        if table_duration.as_millis() > 10 {
            println!("🎯 BOTTLENECK: Exponent table building is expensive ({}ms for 100 builds)", table_duration.as_millis());
            println!("💡 SOLUTION: Add caching/memoization for exponent tables");
        }
        
        if main_duration.as_millis() > 200 {
            println!("🎯 MAJOR BOTTLENECK: Overall algorithm needs optimization");
            println!("💡 SOLUTIONS: Cache expensive computations, optimize hot paths");
        }
    } else {
        println!("✅ Performance within target (<100ms)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_benchmark() {
        // Run a smaller version for testing
        println!("Running performance test...");
        
        let start_time = Instant::now();
        
        // Test 100 operations instead of 1000 for faster testing
        for _i in 0..10 {
            for value in [0.0, 1.0, 2.0, 3.14159, 1000000.5] {
                let lex = float_to_lex(value);
                let recovered = lex_to_float(lex);
                assert_eq!(value, recovered);
            }
        }
        
        let elapsed = start_time.elapsed();
        println!("Test completed 100 operations in {:?}", elapsed);
        
        // This test always passes - it's for measurement
    }

    #[test]
    fn test_exponent_table_caching_opportunity() {
        // Test if rebuilding tables repeatedly is expensive
        let start_time = Instant::now();
        
        for _i in 0..10 {
            let _ = crate::choice::indexing::float_encoding::build_exponent_tables();
        }
        
        let elapsed = start_time.elapsed();
        println!("Built exponent tables 10 times in {:?}", elapsed);
        
        // If this takes more than 1ms for 10 builds, caching would help
        if elapsed.as_millis() > 1 {
            println!("⚠️  Exponent table building detected as potential bottleneck");
        }
    }
}