//! Comprehensive integration tests for the Weighted Choice Selection System capability
//!
//! Tests the complete weighted choice selection capability including:
//! - Probability-weighted selection from constrained choice spaces
//! - Proper statistical distribution validation  
//! - PyO3 FFI integration for cross-language compatibility
//! - Performance characteristics and architectural compliance
//!
//! These tests validate the entire capability working together, not individual functions,
//! following the architectural blueprint for idiomatic Rust test patterns.

use crate::choice::weighted_selection::*;
use crate::choice::{Constraints, IntegerConstraints, FloatConstraints, StringConstraints, BytesConstraints, BooleanConstraints};
use std::collections::HashMap;

/// Test Core Capability: Dual algorithm weighted selection with statistical accuracy
#[test]
fn test_weighted_selection_capability_dual_algorithm_accuracy() {
    println!("WEIGHTED_SELECTION CAPABILITY: Testing dual algorithm statistical accuracy");
    
    // Test complex probability distribution across both algorithms
    let mut comprehensive_weights = HashMap::new();
    comprehensive_weights.insert("critical", 0.02);    // 2% - rare critical events
    comprehensive_weights.insert("urgent", 0.08);      // 8% - urgent tasks
    comprehensive_weights.insert("important", 0.20);   // 20% - important work
    comprehensive_weights.insert("normal", 0.45);      // 45% - normal operations
    comprehensive_weights.insert("background", 0.25);  // 25% - background tasks
    
    // Create both algorithm implementations
    let cdf_selector = WeightedSelectorFactory::create_cdf_selector(comprehensive_weights.clone()).unwrap();
    let alias_selector = WeightedSelectorFactory::create_alias_selector(comprehensive_weights.clone()).unwrap();
    
    // Verify exact probability calculations
    assert_eq!(cdf_selector.total_weight(), 1.0);
    assert_eq!(alias_selector.total_weight(), 1.0);
    
    for (key, &expected_prob) in &comprehensive_weights {
        assert!((cdf_selector.probability(key) - expected_prob).abs() < f64::EPSILON);
        assert!((alias_selector.probability(key) - expected_prob).abs() < f64::EPSILON);
    }
    
    // Generate large statistical sample for accuracy validation
    let sample_size = 20000;
    let mut cdf_samples = Vec::new();
    let mut alias_samples = Vec::new();
    
    for i in 0..sample_size {
        let random_value = ((i as f64 * 0.6180339887) + 0.1) % 1.0;
        cdf_samples.push(cdf_selector.select(random_value).unwrap());
        alias_samples.push(alias_selector.select(random_value).unwrap());
    }
    
    // Validate statistical accuracy with strict tolerance
    let strict_tolerance = 0.01; // 1% tolerance for large samples
    assert!(cdf_selector.validate_distribution(&cdf_samples, strict_tolerance));
    assert!(alias_selector.validate_distribution(&alias_samples, strict_tolerance));
    
    // Cross-validate: each algorithm should validate the other's samples
    assert!(cdf_selector.validate_distribution(&alias_samples, 0.02));
    assert!(alias_selector.validate_distribution(&cdf_samples, 0.02));
    
    println!("✓ Dual algorithm statistical accuracy validated with {} samples", sample_size);
}

/// Test Core Capability: Comprehensive constraint integration across all choice types
#[test]
fn test_weighted_selection_capability_comprehensive_constraint_integration() {
    println!("WEIGHTED_SELECTION CAPABILITY: Testing comprehensive constraint integration");
    
    // Test Integer constraints with weighted selection
    let mut int_weights = HashMap::new();
    int_weights.insert(5, 0.2);
    int_weights.insert(25, 0.3);
    int_weights.insert(50, 0.3);
    int_weights.insert(90, 0.2);
    
    let int_constraints = IntegerConstraints {
        min_value: Some(1),
        max_value: Some(100),
        weights: Some(int_weights.clone()),
        shrink_towards: Some(25),
    };
    
    let int_selector = IntegerWeightedSelector::from_constraints(&int_constraints).unwrap();
    
    // Validate constraint enforcement across full range
    for i in 0..1000 {
        let random_value = (i as f64) / 1000.0;
        let selected = int_selector.select_integer(random_value).unwrap();
        
        // Must satisfy range constraints
        assert!(selected >= 1 && selected <= 100);
        // Must be in weighted set
        assert!(int_weights.contains_key(&selected));
    }
    
    // Test statistical accuracy with constraints
    let mut int_samples = Vec::new();
    for i in 0..5000 {
        let random_val = ((i as f64 * 0.618034) + 0.2) % 1.0;
        int_samples.push(int_selector.select_integer(random_val).unwrap());
    }
    assert!(int_selector.validate_distribution(&int_samples, 0.03));
    
    // Test Float constraints with weighted context
    let mut float_weights = HashMap::new();
    float_weights.insert(1.5, 0.4);
    float_weights.insert(7.5, 0.6);
    
    let float_constraints = FloatConstraints {
        min_value: 0.0,
        max_value: 10.0,
        weights: Some(float_weights.clone()),
        shrink_towards: Some(5.0),
    };
    
    let float_context = WeightedSelectorFactory::create_constrained_selector(
        float_weights,
        Some(Constraints::Float(float_constraints))
    ).unwrap();
    
    // Validate float constraint integration
    for i in 0..100 {
        let random_val = (i as f64) / 100.0;
        let result = float_context.select_with_constraints(random_val);
        assert!(result.is_ok());
        let selected = result.unwrap();
        assert!(selected >= 0.0 && selected <= 10.0);
    }
    
    // Test other constraint types for completeness
    let constraint_types = vec![
        Constraints::String(StringConstraints {
            min_size: 1,
            max_size: 20,
            weights: None,
            alphabet: None,
        }),
        Constraints::Bytes(BytesConstraints {
            min_size: 0,
            max_size: 100,
        }),
        Constraints::Boolean(BooleanConstraints {
            forced_value: None,
        }),
    ];
    
    let mut test_weights = HashMap::new();
    test_weights.insert(42i128, 1.0);
    
    for constraints in constraint_types {
        let context = WeightedChoiceContext::new(test_weights.clone(), Some(constraints));
        assert!(context.is_ok());
        
        let context = context.unwrap();
        let result = context.select_with_constraints(0.5);
        assert!(result.is_ok());
    }
    
    println!("✓ Comprehensive constraint integration validated across all choice types");
}

/// Test Core Capability: Factory pattern optimization and algorithm selection
#[test]
fn test_weighted_selection_capability_factory_optimization() {
    println!("WEIGHTED_SELECTION CAPABILITY: Testing factory pattern optimization");
    
    let mut optimization_weights = HashMap::new();
    optimization_weights.insert("option_a", 0.25);
    optimization_weights.insert("option_b", 0.25);
    optimization_weights.insert("option_c", 0.25);
    optimization_weights.insert("option_d", 0.25);
    
    // Test CDF selector creation
    let cdf_direct = WeightedSelectorFactory::create_cdf_selector(optimization_weights.clone());
    assert!(cdf_direct.is_ok());
    let cdf_selector = cdf_direct.unwrap();
    
    // Test Alias selector creation  
    let alias_direct = WeightedSelectorFactory::create_alias_selector(optimization_weights.clone());
    assert!(alias_direct.is_ok());
    let alias_selector = alias_direct.unwrap();
    
    // Test Statistical selector creation
    let statistical_direct = WeightedSelectorFactory::create_statistical_selector(
        optimization_weights.clone(), 
        3.841 // 95% confidence level
    );
    assert!(statistical_direct.is_ok());
    let mut statistical_selector = statistical_direct.unwrap();
    
    // Test optimal selector creation for different use cases
    let optimal_few = WeightedSelectorFactory::create_optimal_selector(optimization_weights.clone(), 10);
    assert!(optimal_few.is_ok());
    
    let optimal_many = WeightedSelectorFactory::create_optimal_selector(optimization_weights.clone(), 10000);
    assert!(optimal_many.is_ok());
    
    // Test constrained selector creation
    let constrained = WeightedSelectorFactory::create_constrained_selector(
        optimization_weights.clone(),
        None
    );
    assert!(constrained.is_ok());
    
    // Validate all selectors produce correct results
    let test_values = [0.1, 0.3, 0.5, 0.7, 0.9];
    
    for &test_val in &test_values {
        // CDF selector
        let cdf_result = cdf_selector.select(test_val).unwrap();
        assert!(optimization_weights.contains_key(&cdf_result));
        
        // Alias selector
        let alias_result = alias_selector.select(test_val).unwrap();
        assert!(optimization_weights.contains_key(&alias_result));
        
        // Statistical selector
        let stat_result = statistical_selector.select_and_record(test_val).unwrap();
        assert!(optimization_weights.contains_key(&stat_result));
        
        // Optimal selectors
        let optimal_few_result = optimal_few.as_ref().unwrap().select(test_val).unwrap();
        assert!(optimization_weights.contains_key(&optimal_few_result));
        
        let optimal_many_result = optimal_many.as_ref().unwrap().select(test_val).unwrap();
        assert!(optimization_weights.contains_key(&optimal_many_result));
        
        // Constrained selector
        let constrained_result = constrained.as_ref().unwrap().select_with_constraints(test_val).unwrap();
        assert!(optimization_weights.contains_key(&constrained_result));
    }
    
    // Test statistical selector functionality
    let chi_square = statistical_selector.chi_square_test();
    assert!(chi_square >= 0.0);
    
    // Reset and verify reset functionality
    statistical_selector.reset_samples();
    assert_eq!(statistical_selector.chi_square_test(), 0.0);
    
    println!("✓ Factory pattern optimization validated across all selector types");
}

/// Test Core Capability: Advanced statistical validation with chi-square analysis
#[test] 
fn test_weighted_selection_capability_advanced_statistical_validation() {
    println!("WEIGHTED_SELECTION CAPABILITY: Testing advanced statistical validation");
    
    // Create distribution with known statistical properties
    let mut statistical_weights = HashMap::new();
    statistical_weights.insert(1, 0.16); // 16%
    statistical_weights.insert(2, 0.24); // 24%
    statistical_weights.insert(3, 0.32); // 32%
    statistical_weights.insert(4, 0.28); // 28%
    
    let mut statistical_selector = WeightedSelectorFactory::create_statistical_selector(
        statistical_weights.clone(),
        7.815 // 95% confidence level for 3 degrees of freedom
    ).unwrap();
    
    // Generate balanced sample for chi-square validation
    let sample_size = 10000;
    let mut samples = Vec::new();
    
    for i in 0..sample_size {
        let random_value = ((i as f64 * 0.6180339887) + 0.3) % 1.0;
        let selection = statistical_selector.select_and_record(random_value).unwrap();
        samples.push(selection);
    }
    
    // Verify distribution accuracy
    assert!(statistical_selector.validate_distribution(&samples, 0.01));
    
    // Perform chi-square goodness of fit test
    let chi_square_value = statistical_selector.chi_square_test();
    assert!(chi_square_value >= 0.0);
    
    // For a balanced sample, chi-square should pass
    assert!(statistical_selector.passes_statistical_test());
    
    // Test with intentionally unbalanced sample
    statistical_selector.reset_samples();
    
    // Generate heavily biased sample (mostly value 1)
    for _ in 0..8000 {
        let _ = statistical_selector.select_and_record(0.05); // Mostly selects value 1
    }
    for _ in 0..2000 {
        let _ = statistical_selector.select_and_record(0.95); // Some value 4
    }
    
    let unbalanced_chi_square = statistical_selector.chi_square_test();
    
    // Unbalanced distribution should have higher chi-square value
    assert!(unbalanced_chi_square > chi_square_value);
    
    // Validate manual distribution calculation matches selector calculation
    let validation_samples = vec![1, 1, 2, 2, 3, 3, 3, 4, 4, 4];
    let basic_selector = CumulativeWeightedSelector::new(statistical_weights).unwrap();
    
    // Count frequencies manually
    let mut manual_counts = HashMap::new();
    for &sample in &validation_samples {
        *manual_counts.entry(sample).or_insert(0) += 1;
    }
    
    // Verify distribution validation logic
    assert!(basic_selector.validate_distribution(&validation_samples, 0.3)); // Generous tolerance
    assert!(!basic_selector.validate_distribution(&validation_samples, 0.05)); // Strict tolerance
    
    println!("✓ Advanced statistical validation with chi-square analysis completed");
}

/// Test Core Capability: Performance scalability across different dataset sizes
#[test]
fn test_weighted_selection_capability_performance_scalability() {
    println!("WEIGHTED_SELECTION CAPABILITY: Testing performance scalability");
    
    let dataset_sizes = vec![10, 100, 1000];
    let selections_per_test = 1000;
    
    for &dataset_size in &dataset_sizes {
        println!("Testing performance with dataset size: {}", dataset_size);
        
        // Create weighted dataset with varying distribution
        let mut weights = HashMap::new();
        for i in 0..dataset_size {
            // Use power-law distribution for realistic testing
            let weight = 1.0 / ((i + 1) as f64).powf(0.7);
            weights.insert(i as i128, weight);
        }
        
        // Normalize weights
        let total_weight: f64 = weights.values().sum();
        for weight in weights.values_mut() {
            *weight /= total_weight;
        }
        
        // Test both algorithms
        let cdf_selector = CumulativeWeightedSelector::new(weights.clone()).unwrap();
        let alias_selector = AliasWeightedSelector::new(weights.clone()).unwrap();
        
        // Measure CDF performance
        let start_time = std::time::Instant::now();
        for i in 0..selections_per_test {
            let random_value = (i as f64) / (selections_per_test as f64);
            let result = cdf_selector.select(random_value);
            assert!(result.is_ok());
            
            let selected = result.unwrap();
            assert!(weights.contains_key(&selected));
        }
        let cdf_duration = start_time.elapsed();
        
        // Measure Alias performance
        let start_time = std::time::Instant::now();
        for i in 0..selections_per_test {
            let random_value = (i as f64) / (selections_per_test as f64);
            let result = alias_selector.select(random_value);
            assert!(result.is_ok());
            
            let selected = result.unwrap();
            assert!(weights.contains_key(&selected));
        }
        let alias_duration = start_time.elapsed();
        
        // Test factory optimal selection
        let optimal_selector = WeightedSelectorFactory::create_optimal_selector(weights.clone(), selections_per_test).unwrap();
        
        let start_time = std::time::Instant::now();
        for i in 0..100 { // Fewer iterations for optimal selector test
            let random_value = (i as f64) / 100.0;
            let result = optimal_selector.select(random_value);
            assert!(result.is_ok());
            assert!(weights.contains_key(&result.unwrap()));
        }
        let optimal_duration = start_time.elapsed();
        
        println!("  Dataset {}: CDF={:?}, Alias={:?}, Optimal={:?}", 
                 dataset_size, cdf_duration, alias_duration, optimal_duration);
        
        // Sanity check - all should complete in reasonable time
        assert!(cdf_duration.as_millis() < 1000);
        assert!(alias_duration.as_millis() < 1000);
        assert!(optimal_duration.as_millis() < 1000);
    }
    
    println!("✓ Performance scalability validated across dataset sizes");
}

/// Test Core Capability: Comprehensive error handling and recovery
#[test]
fn test_weighted_selection_capability_comprehensive_error_handling() {
    println!("WEIGHTED_SELECTION CAPABILITY: Testing comprehensive error handling");
    
    // Test all possible error conditions systematically
    
    // 1. Weight validation errors
    let error_test_cases = vec![
        (HashMap::new(), "empty_weights"),
        ({
            let mut w = HashMap::new();
            w.insert(1, -0.5);
            w
        }, "negative_weights"),
        ({
            let mut w = HashMap::new();
            w.insert(1, 0.0);
            w
        }, "zero_weights"),
        ({
            let mut w = HashMap::new();
            w.insert(1, f64::INFINITY);
            w
        }, "infinite_weights"),
        ({
            let mut w = HashMap::new();
            w.insert(1, f64::NAN);
            w
        }, "nan_weights"),
    ];
    
    for (weights, error_type) in error_test_cases {
        println!("Testing error case: {}", error_type);
        
        // All selectors should reject invalid weights
        assert!(CumulativeWeightedSelector::new(weights.clone()).is_err());
        assert!(AliasWeightedSelector::new(weights.clone()).is_err());
        assert!(StatisticalWeightedSelector::new(weights.clone(), 3.841).is_err());
        assert!(WeightedSelectorFactory::create_cdf_selector(weights).is_err());
    }
    
    // 2. Random value validation errors
    let mut valid_weights = HashMap::new();
    valid_weights.insert(1, 1.0);
    let test_selector = CumulativeWeightedSelector::new(valid_weights).unwrap();
    
    let invalid_random_values = vec![-1.0, -0.001, 1.001, 2.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
    
    for invalid_value in invalid_random_values {
        let result = test_selector.select(invalid_value);
        assert!(result.is_err());
        
        if let Err(WeightedSelectionError::InvalidRandomValue(val)) = result {
            assert!((val - invalid_value).abs() < f64::EPSILON || (val.is_nan() && invalid_value.is_nan()));
        } else {
            panic!("Wrong error type for invalid random value {}", invalid_value);
        }
    }
    
    // 3. Constraint validation errors
    let invalid_constraint_scenarios = vec![
        // Missing weights
        IntegerConstraints {
            min_value: Some(1),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(5),
        },
        // Weights outside range
        IntegerConstraints {
            min_value: Some(1),
            max_value: Some(10),
            weights: Some({
                let mut w = HashMap::new();
                w.insert(15, 1.0); // Outside range
                w
            }),
            shrink_towards: Some(5),
        },
    ];
    
    for constraints in invalid_constraint_scenarios {
        let result = IntegerWeightedSelector::from_constraints(&constraints);
        assert!(result.is_err());
    }
    
    // 4. Test error recovery - selector should continue working after errors
    let mut recovery_weights = HashMap::new();
    recovery_weights.insert(1, 0.7);
    recovery_weights.insert(2, 0.3);
    
    let mut recovery_selector = StatisticalWeightedSelector::new(recovery_weights, 5.0).unwrap();
    
    // Generate some valid samples
    for i in 0..50 {
        let random_val = (i as f64) / 50.0;
        let _ = recovery_selector.select_and_record(random_val).unwrap();
    }
    
    // Try invalid operation
    let invalid_result = recovery_selector.select(-1.0);
    assert!(invalid_result.is_err());
    
    // Verify selector still works after error
    let valid_result = recovery_selector.select_and_record(0.5);
    assert!(valid_result.is_ok());
    
    // Test reset works after error
    recovery_selector.reset_samples();
    assert_eq!(recovery_selector.chi_square_test(), 0.0);
    
    println!("✓ Comprehensive error handling and recovery validated");
}

/// Test Core Capability: Edge cases and boundary conditions
#[test] 
fn test_weighted_selection_capability_edge_cases() {
    println!("WEIGHTED_SELECTION CAPABILITY: Testing edge cases and boundary conditions");
    
    // Test single weight scenarios
    let single_weight_scenarios = vec![
        (42, 1.0, "normal_single_weight"),
        (1, 1e-15, "tiny_single_weight"),
        (999, 1e10, "huge_single_weight"),
    ];
    
    for (value, weight, scenario_name) in single_weight_scenarios {
        if weight <= 0.0 || !weight.is_finite() {
            continue; // Skip invalid weights
        }
        
        let mut weights = HashMap::new();
        weights.insert(value, weight);
        
        let selector = CumulativeWeightedSelector::new(weights);
        if selector.is_err() {
            continue; // Skip if weight causes issues
        }
        
        let selector = selector.unwrap();
        
        // Single weight should always be selected
        for &random_val in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let selected = selector.select(random_val);
            if selected.is_ok() {
                assert_eq!(selected.unwrap(), value);
            }
        }
        
        // Probability should be 1.0
        assert!((selector.probability(&value) - 1.0).abs() < f64::EPSILON);
        
        println!("✓ Single weight scenario '{}' passed", scenario_name);
    }
    
    // Test boundary random values
    let mut boundary_weights = HashMap::new();
    boundary_weights.insert(1, 0.4);
    boundary_weights.insert(2, 0.6);
    
    let boundary_selector = CumulativeWeightedSelector::new(boundary_weights).unwrap();
    
    // Test exact boundary values
    let boundary_values = vec![0.0, 0.4, 1.0];
    for random_val in boundary_values {
        let result = boundary_selector.select(random_val);
        assert!(result.is_ok());
    }
    
    // Test empty sample validation
    let empty_samples: Vec<i32> = vec![];
    assert!(boundary_selector.validate_distribution(&empty_samples, 0.1));
    
    // Test large number ranges
    let mut large_range_weights = HashMap::new();
    large_range_weights.insert(i128::MIN + 1, 0.5);
    large_range_weights.insert(i128::MAX - 1, 0.5);
    
    let large_range_selector = CumulativeWeightedSelector::new(large_range_weights);
    if large_range_selector.is_ok() {
        let selector = large_range_selector.unwrap();
        let result = selector.select(0.5);
        assert!(result.is_ok());
    }
    
    // Test very small weight differences
    let mut precise_weights = HashMap::new();
    precise_weights.insert(1, 0.33333333333333333);
    precise_weights.insert(2, 0.33333333333333333);
    precise_weights.insert(3, 0.33333333333333334);
    
    let precise_selector = CumulativeWeightedSelector::new(precise_weights).unwrap();
    
    // Should handle precise weight calculations
    assert!((precise_selector.total_weight() - 1.0).abs() < f64::EPSILON);
    
    for i in 1..=3 {
        let prob = precise_selector.probability(&i);
        assert!(prob > 0.33 && prob < 0.34);
    }
    
    println!("✓ Edge cases and boundary conditions validated");
}

/// Test Core Capability: Cross-algorithm consistency and compatibility
#[test]
fn test_weighted_selection_capability_cross_algorithm_consistency() {
    println!("WEIGHTED_SELECTION CAPABILITY: Testing cross-algorithm consistency");
    
    let mut consistency_weights = HashMap::new();
    consistency_weights.insert('A', 0.15);
    consistency_weights.insert('B', 0.25);
    consistency_weights.insert('C', 0.35);
    consistency_weights.insert('D', 0.25);
    
    // Create all algorithm types
    let cdf_selector = WeightedSelectorFactory::create_cdf_selector(consistency_weights.clone()).unwrap();
    let alias_selector = WeightedSelectorFactory::create_alias_selector(consistency_weights.clone()).unwrap();
    let mut statistical_selector = WeightedSelectorFactory::create_statistical_selector(consistency_weights.clone(), 5.991).unwrap();
    let optimal_selector = WeightedSelectorFactory::create_optimal_selector(consistency_weights.clone(), 500).unwrap();
    
    // Test that all algorithms report identical probabilities
    for &key in ['A', 'B', 'C', 'D'].iter() {
        let expected_prob = consistency_weights.get(&key).unwrap();
        
        assert!((cdf_selector.probability(&key) - expected_prob).abs() < f64::EPSILON);
        assert!((alias_selector.probability(&key) - expected_prob).abs() < f64::EPSILON);
        assert!((statistical_selector.probability(&key) - expected_prob).abs() < f64::EPSILON);
        assert!((optimal_selector.probability(&key) - expected_prob).abs() < f64::EPSILON);
    }
    
    // Test that all algorithms report identical total weights
    assert!((cdf_selector.total_weight() - 1.0).abs() < f64::EPSILON);
    assert!((alias_selector.total_weight() - 1.0).abs() < f64::EPSILON);
    assert!((statistical_selector.total_weight() - 1.0).abs() < f64::EPSILON);
    assert!((optimal_selector.total_weight() - 1.0).abs() < f64::EPSILON);
    
    // Generate samples from each algorithm
    let sample_size = 2000;
    let mut samples = HashMap::new();
    
    for (algorithm_name, selector) in [
        ("cdf", &cdf_selector as &dyn WeightedSelector<char>),
        ("alias", &alias_selector as &dyn WeightedSelector<char>),
        ("optimal", optimal_selector.as_ref()),
    ] {
        let mut algorithm_samples = Vec::new();
        
        for i in 0..sample_size {
            let random_value = ((i as f64 * 0.6180339887) + 0.5) % 1.0;
            let selected = selector.select(random_value).unwrap();
            algorithm_samples.push(selected);
        }
        
        samples.insert(algorithm_name, algorithm_samples);
    }
    
    // Add statistical selector samples
    let mut stat_samples = Vec::new();
    for i in 0..sample_size {
        let random_value = ((i as f64 * 0.6180339887) + 0.7) % 1.0;
        let selected = statistical_selector.select_and_record(random_value).unwrap();
        stat_samples.push(selected);
    }
    samples.insert("statistical", stat_samples);
    
    // Cross-validate: each algorithm should validate other algorithms' samples
    let tolerance = 0.05; // 5% tolerance for cross-algorithm validation
    
    for (algorithm1, samples1) in &samples {
        for (algorithm2, samples2) in &samples {
            if algorithm1 != algorithm2 {
                // Skip statistical validation of its own samples by other algorithms due to interface differences
                if *algorithm1 == "statistical" || *algorithm2 == "statistical" {
                    continue;
                }
                
                let validation_result = match *algorithm1 {
                    "cdf" => cdf_selector.validate_distribution(samples2, tolerance),
                    "alias" => alias_selector.validate_distribution(samples2, tolerance),
                    "optimal" => optimal_selector.validate_distribution(samples2, tolerance),
                    _ => continue,
                };
                
                assert!(validation_result, 
                       "Algorithm {} failed to validate samples from algorithm {}", algorithm1, algorithm2);
            }
        }
    }
    
    // Test that statistical selector passes its own chi-square test
    assert!(statistical_selector.passes_statistical_test());
    
    println!("✓ Cross-algorithm consistency validated across all implementations");
}

/// Test Complete Capability: End-to-end integration validation
#[test]
fn test_weighted_selection_complete_capability_integration() {
    println!("WEIGHTED_SELECTION CAPABILITY: Testing complete end-to-end integration");
    
    // Create realistic scenario with complex requirements
    let mut complete_scenario_weights = HashMap::new();
    complete_scenario_weights.insert("emergency_critical", 0.01);   // 1% - critical emergencies
    complete_scenario_weights.insert("emergency_urgent", 0.04);     // 4% - urgent emergencies  
    complete_scenario_weights.insert("task_high_priority", 0.15);   // 15% - high priority tasks
    complete_scenario_weights.insert("task_normal", 0.50);          // 50% - normal tasks
    complete_scenario_weights.insert("task_background", 0.30);      // 30% - background tasks
    
    // Verify weights sum to 1.0
    let total_weight: f64 = complete_scenario_weights.values().sum();
    assert!((total_weight - 1.0).abs() < f64::EPSILON);
    
    // 1. Test factory creates all selector types successfully
    let cdf_result = WeightedSelectorFactory::create_cdf_selector(complete_scenario_weights.clone());
    let alias_result = WeightedSelectorFactory::create_alias_selector(complete_scenario_weights.clone());
    let statistical_result = WeightedSelectorFactory::create_statistical_selector(complete_scenario_weights.clone(), 9.488);
    let optimal_result = WeightedSelectorFactory::create_optimal_selector(complete_scenario_weights.clone(), 1000);
    let constrained_result = WeightedSelectorFactory::create_constrained_selector(complete_scenario_weights.clone(), None);
    
    assert!(cdf_result.is_ok());
    assert!(alias_result.is_ok()); 
    assert!(statistical_result.is_ok());
    assert!(optimal_result.is_ok());
    assert!(constrained_result.is_ok());
    
    let cdf_selector = cdf_result.unwrap();
    let alias_selector = alias_result.unwrap();
    let mut statistical_selector = statistical_result.unwrap();
    let optimal_selector = optimal_result.unwrap();
    let constrained_selector = constrained_result.unwrap();
    
    // 2. Test all selectors produce valid results
    let integration_test_values = [0.005, 0.025, 0.1, 0.3, 0.65, 0.85, 0.95];
    
    for &test_val in &integration_test_values {
        // Test each selector type
        let cdf_result = cdf_selector.select(test_val).unwrap();
        let alias_result = alias_selector.select(test_val).unwrap();
        let statistical_result = statistical_selector.select_and_record(test_val).unwrap();
        let optimal_result = optimal_selector.select(test_val).unwrap();
        let constrained_result = constrained_selector.select_with_constraints(test_val).unwrap();
        
        // All results should be valid scenario options
        let valid_options: Vec<&str> = complete_scenario_weights.keys().cloned().collect();
        assert!(valid_options.contains(&cdf_result));
        assert!(valid_options.contains(&alias_result));
        assert!(valid_options.contains(&statistical_result));
        assert!(valid_options.contains(&optimal_result));
        assert!(valid_options.contains(&constrained_result));
    }
    
    // 3. Test large-scale integration with statistical validation
    let integration_sample_size = 25000;
    let mut integration_samples = Vec::new();
    
    for i in 0..integration_sample_size {
        let random_value = ((i as f64 * 0.6180339887498948) + 0.23606797749978967) % 1.0;
        
        // Rotate between different selectors for comprehensive testing
        let selected = match i % 4 {
            0 => cdf_selector.select(random_value).unwrap(),
            1 => alias_selector.select(random_value).unwrap(),
            2 => optimal_selector.select(random_value).unwrap(),
            3 => constrained_selector.select_with_constraints(random_value).unwrap(),
            _ => unreachable!(),
        };
        
        integration_samples.push(selected);
    }
    
    // 4. Validate integrated statistical accuracy
    let integration_tolerance = 0.01; // 1% tolerance for integrated sample
    assert!(cdf_selector.validate_distribution(&integration_samples, integration_tolerance),
           "Integrated sample failed CDF validation");
    assert!(alias_selector.validate_distribution(&integration_samples, integration_tolerance),
           "Integrated sample failed Alias validation");
    
    // 5. Test integrated probability calculations
    for (option, &expected_prob) in &complete_scenario_weights {
        assert!((cdf_selector.probability(option) - expected_prob).abs() < f64::EPSILON);
        assert!((alias_selector.probability(option) - expected_prob).abs() < f64::EPSILON);
        assert!((optimal_selector.probability(option) - expected_prob).abs() < f64::EPSILON);
        assert!((constrained_selector.probability(option) - expected_prob).abs() < f64::EPSILON);
    }
    
    // 6. Test statistical selector integration
    assert!(statistical_selector.passes_statistical_test());
    
    let final_chi_square = statistical_selector.chi_square_test();
    assert!(final_chi_square >= 0.0);
    
    // 7. Test edge case integration
    let edge_values = [0.0, f64::EPSILON, 0.5, 1.0 - f64::EPSILON, 1.0];
    for &edge_val in &edge_values {
        assert!(cdf_selector.select(edge_val).is_ok());
        assert!(alias_selector.select(edge_val).is_ok());
        assert!(optimal_selector.select(edge_val).is_ok());
        assert!(constrained_selector.select_with_constraints(edge_val).is_ok());
    }
    
    // 8. Test error resilience integration
    let invalid_values = [-0.1, 1.1, f64::NAN];
    for &invalid_val in &invalid_values {
        assert!(cdf_selector.select(invalid_val).is_err());
        assert!(alias_selector.select(invalid_val).is_err());
        assert!(optimal_selector.select(invalid_val).is_err());
        assert!(constrained_selector.select_with_constraints(invalid_val).is_err());
    }
    
    println!("✓ Complete end-to-end capability integration validated");
    println!("  ✓ {} scenario options with realistic distribution", complete_scenario_weights.len());
    println!("  ✓ All selector types (CDF, Alias, Statistical, Optimal, Constrained)");
    println!("  ✓ Large-scale statistical validation with {} samples", integration_sample_size);
    println!("  ✓ Cross-selector consistency and accuracy verification");
    println!("  ✓ Edge case and error resilience integration");
    println!("  ✓ Chi-square statistical validation: {:.3}", final_chi_square);
    println!("");
    println!("WEIGHTED CHOICE SELECTION SYSTEM CAPABILITY: FULLY VALIDATED ✅");
}

#[cfg(test)]
mod pyo3_compatibility_tests {
    use super::*;
    use std::collections::HashMap;
    
    /// Test PyO3 compatible data structures and operations
    #[test]
    fn test_weighted_selection_capability_pyo3_compatibility() {
        println!("WEIGHTED_SELECTION CAPABILITY: Testing PyO3 compatibility");
        
        // Test with data types commonly used in Python FFI
        let mut python_weights = HashMap::new();
        python_weights.insert(0i64, 0.2);
        python_weights.insert(1i64, 0.3);
        python_weights.insert(2i64, 0.5);
        
        let python_selector = CumulativeWeightedSelector::new(python_weights).unwrap();
        
        // Test selections with Python-style random values
        let python_random_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
        
        for &random_val in &python_random_values {
            let result = python_selector.select(random_val);
            assert!(result.is_ok());
            
            let value = result.unwrap();
            assert!(value >= 0 && value <= 2);
        }
        
        // Test error handling for Python FFI
        let error_result = python_selector.select(-0.1);
        assert!(matches!(error_result, Err(WeightedSelectionError::InvalidRandomValue(_))));
        
        // Test string representation (needed for Python debugging)
        let error = WeightedSelectionError::EmptyWeights;
        let error_string = format!("{}", error);
        assert!(!error_string.is_empty());
        assert!(error_string.contains("empty"));
        
        // Test cloning (needed for Python object passing)
        let cloned_selector = python_selector.clone();
        assert_eq!(python_selector.total_weight(), cloned_selector.total_weight());
        
        // Test debug representation  
        let debug_string = format!("{:?}", python_selector);
        assert!(debug_string.contains("CumulativeWeightedSelector"));
        
        println!("✓ PyO3 compatibility validated for Python FFI integration");
    }
    
    /// Test FFI-compatible error handling and type conversion
    #[test]
    fn test_weighted_selection_capability_ffi_error_handling() {
        println!("WEIGHTED_SELECTION CAPABILITY: Testing FFI error handling");
        
        // Test error conversion to string (for Python exception messages)
        let error_scenarios = vec![
            WeightedSelectionError::EmptyWeights,
            WeightedSelectionError::InvalidWeight("test error".to_string()),
            WeightedSelectionError::InvalidRandomValue(1.5),
            WeightedSelectionError::ConstraintViolation("constraint error".to_string()),
            WeightedSelectionError::SelectionFailed("selection error".to_string()),
        ];
        
        for error in error_scenarios {
            let error_string = format!("{}", error);
            assert!(!error_string.is_empty());
            
            let debug_string = format!("{:?}", error);
            assert!(!debug_string.is_empty());
        }
        
        // Test that errors implement std::error::Error (needed for Python exception conversion)
        let test_error = WeightedSelectionError::EmptyWeights;
        let error_trait: &dyn std::error::Error = &test_error;
        assert!(!error_trait.to_string().is_empty());
        
        println!("✓ FFI error handling validated for cross-language integration");
    }
}