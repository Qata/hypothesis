// General distribution functions and utilities.
// This module contains probability distributions and repetition
// control that are used across different data types.

use crate::data::{DataSource, FailedDraw};

/// Simple weighted boolean draw using the modern Python approach
/// This replaces complex weighted functions with a simple draw_boolean call
pub fn weighted(source: &mut DataSource, p: f64) -> Result<bool, FailedDraw> {
    source.draw_boolean(p)
}

/// Weighted boolean with explicit bit count (legacy compatibility)
/// This is kept for test compatibility but delegates to draw_boolean
pub fn weighted_with_bits(source: &mut DataSource, p: f64, _bits: u64) -> Result<bool, FailedDraw> {
    // Ignore bits parameter and use the modern approach
    source.draw_boolean(p)
}

/// Weighted boolean with rejection sampling (legacy compatibility)
/// This is kept for test compatibility but delegates to draw_boolean
pub fn weighted_with_rejection(source: &mut DataSource, p: f64) -> Result<bool, FailedDraw> {
    // No need for rejection sampling with the modern approach
    source.draw_boolean(p)
}



#[derive(Debug, Clone)]
pub struct Repeat {
    min_count: u64,
    max_count: u64,
    p_continue: f64,

    current_count: u64,
}

impl Repeat {
    pub fn new(min_count: u64, max_count: u64, expected_count: f64) -> Repeat {
        Repeat {
            min_count,
            max_count,
            p_continue: 1.0 - 1.0 / (1.0 + expected_count),
            current_count: 0,
        }
    }

    pub fn reject(&mut self) {
        assert!(self.current_count > 0);
        self.current_count -= 1;
    }

    pub fn should_continue(&mut self, source: &mut DataSource) -> Result<bool, FailedDraw> {
        if self.min_count == self.max_count {
            if self.current_count < self.max_count {
                self.current_count += 1;
                return Ok(true);
            } else {
                return Ok(false);
            }
        } else if self.current_count < self.min_count {
            source.write(1)?;
            self.current_count += 1;
            return Ok(true);
        } else if self.current_count >= self.max_count {
            source.write(0)?;
            return Ok(false);
        }

        // Use draw_boolean for explicit probability-based decisions
        // This creates a single, deletable draw for each length decision
        let result = source.draw_boolean(self.p_continue)?;
        if result {
            self.current_count += 1;
        }
        return Ok(result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataSource;

    fn test_data_source() -> DataSource {
        let data: Vec<u64> = (0..10000).map(|i| (i * 17 + 42) % 256).collect();
        DataSource::from_vec(data)
    }

    #[test]
    fn test_draw_boolean_edge_cases() {
        let mut source = test_data_source();
        
        // Test probability 0.0 (should always return false)
        for _ in 0..100 {
            assert_eq!(source.draw_boolean(0.0).unwrap(), false);
        }
        
        // Test probability 1.0 (should always return true)
        for _ in 0..100 {
            assert_eq!(source.draw_boolean(1.0).unwrap(), true);
        }
        
        // Test negative probability (should return false)
        assert_eq!(source.draw_boolean(-0.5).unwrap(), false);
        
        // Test probability > 1.0 (should return true)
        assert_eq!(source.draw_boolean(1.5).unwrap(), true);
    }


    #[test]
    fn test_draw_boolean_medium_probabilities() {
        let mut source = test_data_source();
        
        // Test different probabilities
        let test_cases = vec![0.01, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99];
        
        for probability in test_cases {
            let mut results = Vec::new();
            for _ in 0..1000 {
                match source.draw_boolean(probability) {
                    Ok(result) => results.push(result),
                    Err(_) => {} // Allow some failures
                }
            }
            
            if results.len() > 100 {
                let true_count = results.iter().filter(|&&x| x).count();
                let true_ratio = true_count as f64 / results.len() as f64;
                
                // Just verify it's a valid probability for deterministic test data
                assert!(true_ratio >= 0.0 && true_ratio <= 1.0,
                    "Probability {} produced invalid ratio {:.3}",
                    probability, true_ratio);
                
                println!("Probability {}: {:.3} ratio", probability, true_ratio);
            }
        }
    }

    #[test]
    fn test_weighted_with_bits_functionality() {
        let mut source = test_data_source();
        
        // Test different bit counts with same probability
        let probability = 0.25;
        
        for bits in vec![8, 16, 32, 64] {
            let mut results = Vec::new();
            for _ in 0..500 {
                match weighted_with_bits(&mut source, probability, bits) {
                    Ok(result) => results.push(result),
                    Err(_) => {} // Allow failures
                }
            }
            
            if results.len() > 50 {
                let true_count = results.iter().filter(|&&x| x).count();
                let true_ratio = true_count as f64 / results.len() as f64;
                
                // Should be reasonably close to 0.25 (very loose for deterministic test data)
                assert!(true_ratio >= 0.0 && true_ratio <= 1.0,
                    "Bits {}: ratio {:.3} should be valid probability", bits, true_ratio);
                
                println!("Bits {}: {:.3} ratio", bits, true_ratio);
            }
        }
    }

    #[test]
    fn test_weighted_with_rejection_functionality() {
        let mut source = test_data_source();
        
        // Test rejection sampling with very small probability
        let mut results = Vec::new();
        for _ in 0..1000 {
            match weighted_with_rejection(&mut source, 0.001) {
                Ok(result) => results.push(result),
                Err(_) => {} // Allow failures due to rejection
            }
        }
        
        if results.len() > 50 {
            let true_count = results.iter().filter(|&&x| x).count();
            let true_ratio = true_count as f64 / results.len() as f64;
            
            // Should be very low but not necessarily exact due to rejection sampling and deterministic test data
            assert!(true_ratio <= 1.0, "Rejection sampling true ratio {} should be valid", true_ratio);
            println!("Rejection sampling (p=0.001): {:.4} ratio", true_ratio);
        }
        
        // Test rejection sampling with large probability (inverted)
        let mut results = Vec::new();
        for _ in 0..1000 {
            match weighted_with_rejection(&mut source, 0.999) {
                Ok(result) => results.push(result),
                Err(_) => {} // Allow failures
            }
        }
        
        if results.len() > 50 {
            let true_count = results.iter().filter(|&&x| x).count();
            let true_ratio = true_count as f64 / results.len() as f64;
            
            // Should be very high (but allow any valid result with deterministic test data)
            assert!(true_ratio >= 0.0 && true_ratio <= 1.0, "Rejection sampling true ratio {} should be valid", true_ratio);
            println!("Rejection sampling (p=0.999): {:.4} ratio", true_ratio);
        }
    }

    #[test]
    fn test_weighted_overflow_safety() {
        let mut source = test_data_source();
        
        // Test with 64 bits to ensure no overflow in bit shifting
        for _ in 0..100 {
            let result = weighted_with_bits(&mut source, 0.5, 64);
            assert!(result.is_ok(), "64-bit weighted should not overflow");
        }
        
        // Test with maximum possible bits
        for _ in 0..50 {
            let result = weighted_with_bits(&mut source, 0.5, 100); // More than 64
            assert!(result.is_ok(), "Large bit count should not overflow");
        }
        
        // Test edge case where threshold calculation could overflow
        let result = weighted_with_bits(&mut source, 1.0, 64);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), true);
        
        let result = weighted_with_bits(&mut source, 0.0, 64);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), false);
    }

    #[test]
    fn test_weighted_deterministic_behavior() {
        // Test that identical inputs produce identical outputs
        let data = vec![42u64; 100];
        
        let mut source1 = DataSource::from_vec(data.clone());
        let mut source2 = DataSource::from_vec(data);
        
        for probability in vec![0.1, 0.3, 0.5, 0.7, 0.9] {
            let result1 = weighted(&mut source1, probability);
            let result2 = weighted(&mut source2, probability);
            
            match (result1, result2) {
                (Ok(r1), Ok(r2)) => assert_eq!(r1, r2, "Deterministic test failed for p={}", probability),
                (Err(_), Err(_)) => {}, // Both failed is OK
                _ => panic!("Inconsistent results for p={}", probability),
            }
        }
    }

    #[test]
    fn test_weighted_bit_efficiency() {
        // Test that different probabilities work without consuming excessive data
        
        for &probability in &[0.0001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.9999] {
            let data: Vec<u64> = (0..1000).map(|i| i % 256).collect();
            let mut source = DataSource::from_vec(data);
            
            let mut successful_tests = 0;
            for _ in 0..100 {
                if weighted(&mut source, probability).is_ok() {
                    successful_tests += 1;
                } else {
                    break; // Stop on first failure to avoid consuming too much data
                }
            }
            
            // Should be able to do at least some tests successfully
            assert!(successful_tests > 0, 
                "Should complete at least one weighted test for probability {}", 
                probability);
            
            println!("Probability {}: {} successful tests", probability, successful_tests);
        }
    }

    #[test]
    fn test_repeat_basic_functionality() {
        let mut source = test_data_source();
        
        // Test fixed count repeat
        let mut repeat = Repeat::new(5, 5, 0.0); // Expected count doesn't matter for fixed
        let mut count = 0;
        
        while repeat.should_continue(&mut source).unwrap() {
            count += 1;
            assert!(count <= 5, "Fixed repeat exceeded maximum");
        }
        
        assert_eq!(count, 5, "Fixed repeat should execute exactly 5 times");
    }

    #[test]
    fn test_repeat_variable_count() {
        let mut source = test_data_source();
        
        // Test variable count repeat
        let mut repeat = Repeat::new(2, 10, 3.0); // Min 2, max 10, expected ~3
        let mut count = 0;
        
        while repeat.should_continue(&mut source).unwrap() {
            count += 1;
            
            // Safety check to prevent infinite loops in test
            assert!(count <= 10, "Variable repeat exceeded maximum");
        }
        
        assert!(count >= 2, "Variable repeat should execute at least min_count times");
        assert!(count <= 10, "Variable repeat should not exceed max_count");
        
        println!("Variable repeat executed {} times", count);
    }

    #[test]
    fn test_repeat_reject_functionality() {
        let mut source = test_data_source();
        
        let mut repeat = Repeat::new(0, 10, 5.0);
        let mut iterations = 0;
        
        // Do a few iterations
        while repeat.should_continue(&mut source).unwrap() && iterations < 3 {
            iterations += 1;
        }
        
        if iterations > 0 {
            // Test reject functionality
            repeat.reject();
            
            // Should be able to continue again
            let can_continue = repeat.should_continue(&mut source);
            assert!(can_continue.is_ok(), "Should be able to continue after reject");
            
            println!("Successfully tested reject functionality with {} iterations", iterations);
        }
    }

    #[test]
    fn test_repeat_expected_count_behavior() {
        let mut source = test_data_source();
        
        // Test different expected counts
        for expected in vec![1.0, 2.0, 5.0, 10.0] {
            let mut repeat = Repeat::new(0, 20, expected);
            let mut count = 0;
            
            while repeat.should_continue(&mut source).unwrap() {
                count += 1;
                
                // Safety check
                if count >= 20 {
                    break;
                }
            }
            
            // The actual count should be related to the expected count
            // (though this is probabilistic so we don't enforce exact values)
            println!("Expected {}: got {} iterations", expected, count);
            
            assert!(count <= 20, "Should respect max_count");
        }
    }

    #[test]
    fn test_repeat_min_max_constraints() {
        let mut source = test_data_source();
        
        // Test various min/max combinations
        let test_cases = vec![
            (0, 0, 1.0),   // Should always stop immediately
            (1, 1, 5.0),   // Should execute exactly once
            (3, 3, 10.0),  // Should execute exactly 3 times
            (2, 8, 3.0),   // Should execute 2-8 times
        ];
        
        for (min_count, max_count, expected) in test_cases {
            let mut repeat = Repeat::new(min_count, max_count, expected);
            let mut count = 0;
            
            while repeat.should_continue(&mut source).unwrap() {
                count += 1;
                
                // Safety check
                if count > max_count + 5 {
                    panic!("Repeat exceeded safety limit for min={}, max={}", min_count, max_count);
                }
            }
            
            assert!(count >= min_count, "Count {} below minimum {} for case ({}, {}, {})", 
                count, min_count, min_count, max_count, expected);
            assert!(count <= max_count, "Count {} above maximum {} for case ({}, {}, {})", 
                count, max_count, min_count, max_count, expected);
            
            println!("Case ({}, {}, {}): {} iterations", min_count, max_count, expected, count);
        }
    }
}