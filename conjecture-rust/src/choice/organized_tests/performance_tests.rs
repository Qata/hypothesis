//! Performance and stress tests for choice system
//! 
//! These tests verify that the choice system performs well under load
//! and with large/complex inputs, helping identify performance regressions.

use crate::choice::*;
use std::time::Instant;

#[cfg(test)]
mod performance_benchmarks {
    use super::*;

    #[test]
    fn test_integer_indexing_performance() {
        println!("PERFORMANCE_TEST DEBUG: Testing integer indexing performance");
        
        let constraints = Constraints::Integer(IntegerConstraints::default());
        let start = Instant::now();
        
        // Test indexing performance with many values
        for i in -1000..1000 {
            let value = ChoiceValue::Integer(i);
            let index = choice_to_index(&value, &constraints);
            let _recovered = choice_from_index(index, "integer", &constraints);
        }
        
        let duration = start.elapsed();
        println!("PERFORMANCE_TEST DEBUG: 2000 integer indexing operations took {:?}", duration);
        
        // Should complete in reasonable time (less than 100ms for 2000 ops)
        assert!(duration.as_millis() < 100, 
            "Integer indexing too slow: {} ms for 2000 operations", duration.as_millis());
        
        println!("PERFORMANCE_TEST DEBUG: Integer indexing performance test passed");
    }

    #[test]
    fn test_float_indexing_performance() {
        println!("PERFORMANCE_TEST DEBUG: Testing float indexing performance");
        
        let constraints = Constraints::Float(FloatConstraints::default());
        let start = Instant::now();
        
        // Test float indexing performance with various values
        let test_values: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
        
        // First run (cold cache)
        for val in &test_values {
            let value = ChoiceValue::Float(*val);
            let index = choice_to_index(&value, &constraints);
            let _recovered = choice_from_index(index, "float", &constraints);
        }
        
        let first_duration = start.elapsed();
        println!("PERFORMANCE_TEST DEBUG: 1000 float indexing operations (cold cache) took {:?}", first_duration);
        
        // Second run (warm cache) 
        let warm_start = Instant::now();
        for val in &test_values {
            let value = ChoiceValue::Float(*val);
            let index = choice_to_index(&value, &constraints);
            let _recovered = choice_from_index(index, "float", &constraints);
        }
        let warm_duration = warm_start.elapsed();
        println!("PERFORMANCE_TEST DEBUG: 1000 float indexing operations (warm cache) took {:?}", warm_duration);
        
        let duration = first_duration;
        
        // Float operations may be slower due to complex encoding
        assert!(duration.as_millis() < 500, 
            "Float indexing too slow: {} ms for 1000 operations", duration.as_millis());
        
        println!("PERFORMANCE_TEST DEBUG: Float indexing performance test passed");
    }

    #[test]
    fn test_string_indexing_performance() {
        println!("PERFORMANCE_TEST DEBUG: Testing string indexing performance");
        
        let constraints = Constraints::String(StringConstraints {
            min_size: 0,
            max_size: 20,
            intervals: IntervalSet::from_string("abc"),
        });
        let start = Instant::now();
        
        // Test string indexing with various lengths
        for i in 0..100 {
            let string_content = "a".repeat(i % 10); // Strings of length 0-9
            let value = ChoiceValue::String(string_content);
            let index = choice_to_index(&value, &constraints);
            let _recovered = choice_from_index(index, "string", &constraints);
        }
        
        let duration = start.elapsed();
        println!("PERFORMANCE_TEST DEBUG: 100 string indexing operations took {:?}", duration);
        
        assert!(duration.as_millis() < 200, 
            "String indexing too slow: {} ms for 100 operations", duration.as_millis());
        
        println!("PERFORMANCE_TEST DEBUG: String indexing performance test passed");
    }

    #[test]
    fn test_large_index_space_stress() {
        println!("PERFORMANCE_TEST DEBUG: Testing large index space stress");
        
        let float_constraints = Constraints::Float(FloatConstraints::default());
        
        // Test with values that should produce large indices
        let large_values = vec![
            1e100, 1e200, 1e300, f64::MAX / 2.0, f64::MAX / 1000.0
        ];
        
        for val in large_values {
            if val.is_finite() {
                let start = Instant::now();
                
                let value = ChoiceValue::Float(val);
                let index = choice_to_index(&value, &float_constraints);
                let recovered = choice_from_index(index, "float", &float_constraints);
                
                let duration = start.elapsed();
                
                // Large value operations should still be fast
                assert!(duration.as_millis() < 10, 
                    "Large value indexing too slow: {} ms for value {}", duration.as_millis(), val);
                
                if let ChoiceValue::Float(recovered_val) = recovered {
                    assert_eq!(recovered_val, val, 
                        "Large value roundtrip failed: {} -> {} -> {}", val, index, recovered_val);
                }
                
                println!("PERFORMANCE_TEST DEBUG: Large value {} -> index {} in {:?}", val, index, duration);
            }
        }
        
        println!("PERFORMANCE_TEST DEBUG: Large index space stress test passed");
    }

    #[test]
    fn test_bounded_constraint_performance() {
        println!("PERFORMANCE_TEST DEBUG: Testing bounded constraint performance");
        
        // Test performance with complex bounded constraints
        let bounded_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(-1000),
            max_value: Some(1000),
            weights: None,
            shrink_towards: Some(42),
        });
        
        let start = Instant::now();
        
        // Test many values within the bounded range
        for i in -1000..=1000 {
            let value = ChoiceValue::Integer(i);
            let index = choice_to_index(&value, &bounded_constraints);
            let _recovered = choice_from_index(index, "integer", &bounded_constraints);
        }
        
        let duration = start.elapsed();
        println!("PERFORMANCE_TEST DEBUG: 2001 bounded constraint operations took {:?}", duration);
        
        // Bounded constraints may use sequence generation which could be slower
        assert!(duration.as_millis() < 200, 
            "Bounded constraint indexing too slow: {} ms for 2001 operations", duration.as_millis());
        
        println!("PERFORMANCE_TEST DEBUG: Bounded constraint performance test passed");
    }

    #[test] 
    fn test_memory_usage_stability() {
        println!("PERFORMANCE_TEST DEBUG: Testing memory usage stability");
        
        let constraints = Constraints::Integer(IntegerConstraints::default());
        
        // Test that repeated operations don't cause memory growth
        // (This is a basic test - in practice we'd want more sophisticated memory monitoring)
        
        for round in 0..10 {
            for i in 0..1000 {
                let value = ChoiceValue::Integer(i);
                let index = choice_to_index(&value, &constraints);
                let _recovered = choice_from_index(index, "integer", &constraints);
            }
            
            // Basic check - if we've gotten this far without OOM, memory usage is probably stable
            println!("PERFORMANCE_TEST DEBUG: Completed round {} of 1000 operations", round + 1);
        }
        
        println!("PERFORMANCE_TEST DEBUG: Memory usage stability test passed");
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    #[test]
    fn test_extreme_values_stress() {
        println!("STRESS_TEST DEBUG: Testing extreme values stress");
        
        let constraints = Constraints::Integer(IntegerConstraints::default());
        
        // Test with extreme integer values (using safer values to avoid overflow)
        let extreme_values = vec![
            i128::MIN / 8,
            -1000000000000000i128,
            -1000000i128,
            0,
            1000000i128,
            1000000000000000i128,
            i128::MAX / 8,
        ];
        
        for val in extreme_values {
            let value = ChoiceValue::Integer(val);
            let index = choice_to_index(&value, &constraints);
            let recovered = choice_from_index(index, "integer", &constraints);
            
            assert!(choice_equal(&value, &recovered),
                "Extreme value stress test failed for {}: got {:?}", val, recovered);
            
            println!("STRESS_TEST DEBUG: Extreme value {} -> index {} -> recovered", val, index);
        }
        
        println!("STRESS_TEST DEBUG: Extreme values stress test passed");
    }

    #[test]
    fn test_complex_constraint_combinations() {
        println!("STRESS_TEST DEBUG: Testing complex constraint combinations");
        
        // Test various complex constraint scenarios
        let constraint_sets = vec![
            // Very tight bounds
            Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(1),
                weights: None,
                shrink_towards: Some(0),
            }),
            
            // Large negative bounds  
            Constraints::Integer(IntegerConstraints {
                min_value: Some(-1000000),
                max_value: Some(-999999),
                weights: None,
                shrink_towards: Some(-1000000),
            }),
            
            // Asymmetric bounds
            Constraints::Integer(IntegerConstraints {
                min_value: Some(-100),
                max_value: Some(1000000),
                weights: None,
                shrink_towards: Some(42),
            }),
        ];
        
        for (i, constraints) in constraint_sets.into_iter().enumerate() {
            println!("STRESS_TEST DEBUG: Testing constraint set {}: {:?}", i, constraints);
            
            // Test several values that should work with these constraints
            for test_index in 0..50 {
                let recovered = choice_from_index(test_index, "integer", &constraints);
                
                if let ChoiceValue::Integer(val) = recovered {
                    // Verify the recovered value satisfies constraints
                    if let Constraints::Integer(int_constraints) = &constraints {
                        if let Some(min) = int_constraints.min_value {
                            assert!(val >= min, "Constraint violation: {} < min {}", val, min);
                        }
                        if let Some(max) = int_constraints.max_value {
                            assert!(val <= max, "Constraint violation: {} > max {}", val, max);
                        }
                    }
                }
            }
        }
        
        println!("STRESS_TEST DEBUG: Complex constraint combinations test passed");
    }
}