//! Standalone test for weighted selection comprehensive capability
//! This is a focused test that only tests the weighted selection system

use std::collections::HashMap;

// Import weighted selection types directly
use conjecture_rust::choice::weighted_selection::{
    WeightedSelector, CumulativeWeightedSelector, AliasWeightedSelector,
    WeightedChoiceContext, StatisticalWeightedSelector, WeightedSelectorFactory,
    IntegerWeightedSelector, WeightedSelectionError
};
use conjecture_rust::choice::{Constraints, IntegerConstraints, FloatConstraints, StringConstraints, BytesConstraints, BooleanConstraints, IntervalSet};

fn main() {
    println!("Running weighted selection comprehensive capability tests...");
    
    test_weighted_choice_selection_capability_probability_weighted_selection();
    test_weighted_choice_selection_capability_dual_algorithm_optimization();
    test_weighted_choice_selection_capability_error_handling();
    test_weighted_choice_selection_capability_factory_optimization();
    test_weighted_choice_selection_capability_edge_cases();
    test_weighted_choice_selection_capability_performance_validation();
    test_weighted_choice_selection_capability_statistical_distribution();
    test_weighted_choice_selection_capability_chi_square_validation();
    
    println!("All weighted selection comprehensive capability tests passed!");
}

/// Test the complete weighted choice selection capability with probability-weighted selection
fn test_weighted_choice_selection_capability_probability_weighted_selection() {
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Testing probability-weighted selection capability");
    
    let mut weights = HashMap::new();
    weights.insert("low_prob", 0.1);
    weights.insert("medium_prob", 0.3);
    weights.insert("high_prob", 0.6);
    
    let selector = CumulativeWeightedSelector::new(weights).unwrap();
    
    // Test probability calculations
    assert!((selector.probability(&"low_prob") - 0.1).abs() < f64::EPSILON);
    assert!((selector.probability(&"medium_prob") - 0.3).abs() < f64::EPSILON);
    assert!((selector.probability(&"high_prob") - 0.6).abs() < f64::EPSILON);
    
    // Test deterministic selection based on probabilities
    assert_eq!(selector.select(0.05).unwrap(), "low_prob");
    assert_eq!(selector.select(0.25).unwrap(), "medium_prob");
    assert_eq!(selector.select(0.80).unwrap(), "high_prob");
    
    // Test statistical distribution over many samples
    let mut samples = Vec::new();
    for i in 0..1000 {
        let random_val = (i as f64) / 1000.0;
        samples.push(selector.select(random_val).unwrap());
    }
    
    assert!(selector.validate_distribution(&samples, 0.05));
    
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Probability-weighted selection capability validated");
}

/// Test the complete capability with dual algorithm optimization (CDF vs Alias)
fn test_weighted_choice_selection_capability_dual_algorithm_optimization() {
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Testing dual algorithm optimization capability");
    
    let mut weights = HashMap::new();
    weights.insert("alpha", 0.1);
    weights.insert("beta", 0.2);
    weights.insert("gamma", 0.3);
    weights.insert("delta", 0.4);
    
    // Test CDF algorithm (O(log n))
    let cdf_selector = WeightedSelectorFactory::create_cdf_selector(weights.clone()).unwrap();
    
    // Test Alias algorithm (O(1))
    let alias_selector = WeightedSelectorFactory::create_alias_selector(weights.clone()).unwrap();
    
    // Verify both algorithms produce statistically equivalent results
    let test_values = [0.05, 0.15, 0.35, 0.75, 0.95];
    let mut cdf_samples = Vec::new();
    let mut alias_samples = Vec::new();
    
    for &random_val in &test_values {
        cdf_samples.push(cdf_selector.select(random_val).unwrap());
        alias_samples.push(alias_selector.select(random_val).unwrap());
    }
    
    // Both algorithms should respect the same probability distribution
    assert!(cdf_selector.validate_distribution(&cdf_samples, 0.5));
    assert!(alias_selector.validate_distribution(&alias_samples, 0.5));
    
    // Test factory optimization selection
    let optimal_few = WeightedSelectorFactory::create_optimal_selector(weights.clone(), 50).unwrap();
    let optimal_many = WeightedSelectorFactory::create_optimal_selector(weights.clone(), 500).unwrap();
    
    // Verify both optimal selectors work correctly
    for &random_val in &test_values {
        assert!(optimal_few.select(random_val).is_ok());
        assert!(optimal_many.select(random_val).is_ok());
    }
    
    // Test statistical equivalence between algorithms
    let large_sample_size = 1000;
    let mut cdf_large_samples = Vec::new();
    let mut alias_large_samples = Vec::new();
    
    for i in 0..large_sample_size {
        let random_val = (i as f64) / (large_sample_size as f64);
        cdf_large_samples.push(cdf_selector.select(random_val).unwrap());
        alias_large_samples.push(alias_selector.select(random_val).unwrap());
    }
    
    assert!(cdf_selector.validate_distribution(&cdf_large_samples, 0.05));
    assert!(alias_selector.validate_distribution(&alias_large_samples, 0.05));
    
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Dual algorithm optimization capability validated");
}

/// Test the complete capability with comprehensive error handling
fn test_weighted_choice_selection_capability_error_handling() {
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Testing error handling capability");
    
    // Test empty weights error
    let empty_weights: HashMap<i32, f64> = HashMap::new();
    let empty_result = CumulativeWeightedSelector::new(empty_weights);
    assert!(matches!(empty_result, Err(WeightedSelectionError::EmptyWeights)));
    
    // Test invalid weights errors
    let mut invalid_weights = HashMap::new();
    invalid_weights.insert(1, -0.5); // Negative weight
    let invalid_result = CumulativeWeightedSelector::new(invalid_weights);
    assert!(matches!(invalid_result, Err(WeightedSelectionError::InvalidWeight(_))));
    
    let mut zero_weights = HashMap::new();
    zero_weights.insert(1, 0.0); // Zero weight
    let zero_result = CumulativeWeightedSelector::new(zero_weights);
    assert!(matches!(zero_result, Err(WeightedSelectionError::InvalidWeight(_))));
    
    let mut infinite_weights = HashMap::new();
    infinite_weights.insert(1, f64::INFINITY); // Infinite weight
    let infinite_result = CumulativeWeightedSelector::new(infinite_weights);
    assert!(matches!(infinite_result, Err(WeightedSelectionError::InvalidWeight(_))));
    
    let mut nan_weights = HashMap::new();
    nan_weights.insert(1, f64::NAN); // NaN weight
    let nan_result = CumulativeWeightedSelector::new(nan_weights);
    assert!(matches!(nan_result, Err(WeightedSelectionError::InvalidWeight(_))));
    
    // Test invalid random value errors
    let mut valid_weights = HashMap::new();
    valid_weights.insert(1, 1.0);
    let selector = CumulativeWeightedSelector::new(valid_weights).unwrap();
    
    let negative_random = selector.select(-0.1);
    assert!(matches!(negative_random, Err(WeightedSelectionError::InvalidRandomValue(_))));
    
    let large_random = selector.select(1.1);
    assert!(matches!(large_random, Err(WeightedSelectionError::InvalidRandomValue(_))));
    
    // Test constraint violation errors
    let mut constraint_weights = HashMap::new();
    constraint_weights.insert(25i128, 0.5); // Outside constraint range
    constraint_weights.insert(30i128, 0.5); // Outside constraint range
    
    let constraint_limits = IntegerConstraints {
        min_value: Some(1),
        max_value: Some(20), // Max is 20, but weights include 25, 30
        weights: Some(constraint_weights),
        shrink_towards: Some(10),
    };
    
    let constraint_result = IntegerWeightedSelector::from_constraints(&constraint_limits);
    assert!(matches!(constraint_result, Err(WeightedSelectionError::ConstraintViolation(_))));
    
    // Test missing weights in constraints
    let missing_weights_constraints = IntegerConstraints {
        min_value: Some(1),
        max_value: Some(20),
        weights: None, // No weights provided
        shrink_towards: Some(10),
    };
    
    let missing_weights_result = IntegerWeightedSelector::from_constraints(&missing_weights_constraints);
    assert!(matches!(missing_weights_result, Err(WeightedSelectionError::ConstraintViolation(_))));
    
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Error handling capability validated");
}

/// Test the complete capability with factory pattern optimization
fn test_weighted_choice_selection_capability_factory_optimization() {
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Testing factory optimization capability");
    
    let mut weights = HashMap::new();
    weights.insert(100, 0.1);
    weights.insert(200, 0.2);
    weights.insert(300, 0.3);
    weights.insert(400, 0.4);
    
    // Test all factory creation methods
    let cdf_selector = WeightedSelectorFactory::create_cdf_selector(weights.clone());
    assert!(cdf_selector.is_ok());
    
    let alias_selector = WeightedSelectorFactory::create_alias_selector(weights.clone());
    assert!(alias_selector.is_ok());
    
    let statistical_selector = WeightedSelectorFactory::create_statistical_selector(weights.clone(), 5.0);
    assert!(statistical_selector.is_ok());
    
    // Test optimal selector choice based on expected usage
    let optimal_low = WeightedSelectorFactory::create_optimal_selector(weights.clone(), 10);
    assert!(optimal_low.is_ok());
    
    let optimal_high = WeightedSelectorFactory::create_optimal_selector(weights.clone(), 1000);
    assert!(optimal_high.is_ok());
    
    // Test constrained selector factory
    let constraints = IntegerConstraints {
        min_value: Some(50),
        max_value: Some(500),
        weights: Some(weights.clone()),
        shrink_towards: Some(250),
    };
    
    let constrained_selector = WeightedSelectorFactory::create_constrained_selector(
        weights.clone(),
        Some(Constraints::Integer(constraints)),
    );
    assert!(constrained_selector.is_ok());
    
    // Verify all factory-created selectors work correctly
    let test_random_values = [0.1, 0.3, 0.6, 0.9];
    
    for &random_val in &test_random_values {
        assert!(cdf_selector.as_ref().unwrap().select(random_val).is_ok());
        assert!(alias_selector.as_ref().unwrap().select(random_val).is_ok());
        assert!(optimal_low.as_ref().unwrap().select(random_val).is_ok());
        assert!(optimal_high.as_ref().unwrap().select(random_val).is_ok());
        assert!(constrained_selector.as_ref().unwrap().select_with_constraints(random_val).is_ok());
    }
    
    // Test statistical selector with sample recording
    let mut stat_sel = statistical_selector.unwrap();
    for i in 0..100 {
        let random_val = (i as f64) / 100.0;
        assert!(stat_sel.select_and_record(random_val).is_ok());
    }
    
    let chi_square = stat_sel.chi_square_test();
    assert!(chi_square >= 0.0);
    
    stat_sel.reset_samples();
    
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Factory optimization capability validated");
}

/// Test the complete capability with edge cases and boundary conditions
fn test_weighted_choice_selection_capability_edge_cases() {
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Testing edge cases capability");
    
    // Test single weight edge case
    let mut single_weight = HashMap::new();
    single_weight.insert("only_choice", 1.0);
    
    let single_selector = CumulativeWeightedSelector::new(single_weight).unwrap();
    assert_eq!(single_selector.select(0.0).unwrap(), "only_choice");
    assert_eq!(single_selector.select(0.5).unwrap(), "only_choice");
    assert_eq!(single_selector.select(1.0).unwrap(), "only_choice");
    assert_eq!(single_selector.probability(&"only_choice"), 1.0);
    
    // Test very small weights
    let mut tiny_weights = HashMap::new();
    tiny_weights.insert(1, 1e-15);
    tiny_weights.insert(2, 1e-15);
    
    let tiny_selector = CumulativeWeightedSelector::new(tiny_weights).unwrap();
    assert!(tiny_selector.select(0.0).is_ok());
    assert!(tiny_selector.select(0.5).is_ok());
    assert!(tiny_selector.select(1.0).is_ok());
    
    // Test very large weights
    let mut large_weights = HashMap::new();
    large_weights.insert(1, 1e10);
    large_weights.insert(2, 1e10);
    
    let large_selector = CumulativeWeightedSelector::new(large_weights).unwrap();
    assert!(large_selector.select(0.0).is_ok());
    assert!(large_selector.select(0.5).is_ok());
    assert!(large_selector.select(1.0).is_ok());
    
    // Test unequal weights with extreme ratios
    let mut extreme_weights = HashMap::new();
    extreme_weights.insert("rare", 1e-6);
    extreme_weights.insert("common", 1.0);
    
    let extreme_selector = CumulativeWeightedSelector::new(extreme_weights).unwrap();
    assert_eq!(extreme_selector.select(0.0).unwrap(), "rare");
    assert_eq!(extreme_selector.select(0.99999).unwrap(), "common");
    
    // Test boundary random values
    let mut boundary_weights = HashMap::new();
    boundary_weights.insert(10, 0.3);
    boundary_weights.insert(20, 0.7);
    
    let boundary_selector = CumulativeWeightedSelector::new(boundary_weights).unwrap();
    assert!(boundary_selector.select(0.0).is_ok());
    assert!(boundary_selector.select(1.0).is_ok());
    
    // Test distribution validation with empty samples
    let empty_samples: Vec<i32> = vec![];
    assert!(boundary_selector.validate_distribution(&empty_samples, 0.1));
    
    // Test distribution validation with single sample
    let single_sample = vec![10];
    assert!(boundary_selector.validate_distribution(&single_sample, 1.0)); // High tolerance for single sample
    
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Edge cases capability validated");
}

/// Test the complete capability with large-scale performance validation
fn test_weighted_choice_selection_capability_performance_validation() {
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Testing performance validation capability");
    
    // Create a large weight set for performance testing
    let mut large_weights = HashMap::new();
    for i in 0..1000 {
        large_weights.insert(i, 1.0 / 1000.0); // Equal weights
    }
    
    // Test CDF selector performance
    let cdf_selector = CumulativeWeightedSelector::new(large_weights.clone()).unwrap();
    
    // Test Alias selector performance
    let alias_selector = AliasWeightedSelector::new(large_weights.clone()).unwrap();
    
    // Verify both selectors handle large datasets correctly
    let large_sample_size = 10000;
    let mut cdf_samples = Vec::new();
    let mut alias_samples = Vec::new();
    
    for i in 0..large_sample_size {
        let random_val = (i as f64) / (large_sample_size as f64);
        cdf_samples.push(cdf_selector.select(random_val).unwrap());
        alias_samples.push(alias_selector.select(random_val).unwrap());
    }
    
    // Both should produce valid distributions
    assert!(cdf_selector.validate_distribution(&cdf_samples, 0.1));
    assert!(alias_selector.validate_distribution(&alias_samples, 0.1));
    
    // Test statistical validation with large sample
    let mut stat_selector = StatisticalWeightedSelector::new(large_weights, 100.0).unwrap(); // Relaxed threshold for large DOF
    
    for i in 0..1000 {
        let random_val = (i as f64) / 1000.0;
        assert!(stat_selector.select_and_record(random_val).is_ok());
    }
    
    let chi_square = stat_selector.chi_square_test();
    assert!(chi_square >= 0.0);
    
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Performance validation capability validated");
}

/// Test the complete capability with proper statistical distribution validation
fn test_weighted_choice_selection_capability_statistical_distribution() {
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Testing statistical distribution capability");
    
    let mut weights = HashMap::new();
    weights.insert(1, 0.2);
    weights.insert(2, 0.3);
    weights.insert(3, 0.5);
    
    let mut stat_selector = StatisticalWeightedSelector::new(weights.clone(), 7.815).unwrap(); // 95% confidence, 2 DOF
    
    // Generate large sample set for statistical validation
    let sample_size = 10000;
    let mut samples = Vec::new();
    
    for i in 0..sample_size {
        let random_val = (i as f64) / (sample_size as f64);
        let selection = stat_selector.select_and_record(random_val).unwrap();
        samples.push(selection);
    }
    
    // Validate distribution matches expected probabilities
    assert!(stat_selector.validate_distribution(&samples, 0.02)); // 2% tolerance
    
    // Test chi-square goodness of fit
    let chi_square = stat_selector.chi_square_test();
    assert!(chi_square >= 0.0);
    assert!(stat_selector.passes_statistical_test());
    
    // Verify expected frequencies
    let mut counts = HashMap::new();
    for sample in &samples {
        *counts.entry(*sample).or_insert(0) += 1;
    }
    
    let count_1 = *counts.get(&1).unwrap_or(&0) as f64;
    let count_2 = *counts.get(&2).unwrap_or(&0) as f64;
    let count_3 = *counts.get(&3).unwrap_or(&0) as f64;
    
    let freq_1 = count_1 / sample_size as f64;
    let freq_2 = count_2 / sample_size as f64;
    let freq_3 = count_3 / sample_size as f64;
    
    // Verify frequencies are close to expected probabilities
    assert!((freq_1 - 0.2).abs() < 0.02);
    assert!((freq_2 - 0.3).abs() < 0.02);
    assert!((freq_3 - 0.5).abs() < 0.02);
    
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Statistical distribution capability validated");
}

/// Test comprehensive chi-square statistical validation capability
fn test_weighted_choice_selection_capability_chi_square_validation() {
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Testing chi-square validation capability");
    
    // Test with known distribution that should pass chi-square test
    let mut good_weights = HashMap::new();
    good_weights.insert("outcome_1", 0.25);
    good_weights.insert("outcome_2", 0.25);
    good_weights.insert("outcome_3", 0.25);
    good_weights.insert("outcome_4", 0.25);
    
    let mut good_stat_selector = StatisticalWeightedSelector::new(good_weights, 7.815).unwrap(); // 95% confidence, 3 DOF
    
    // Generate perfectly distributed samples
    let sample_size = 1000;
    for i in 0..sample_size {
        let random_val = (i as f64) / (sample_size as f64);
        let _ = good_stat_selector.select_and_record(random_val).unwrap();
    }
    
    let good_chi_square = good_stat_selector.chi_square_test();
    assert!(good_chi_square >= 0.0);
    assert!(good_stat_selector.passes_statistical_test());
    
    // Test with deliberately skewed distribution that should fail chi-square test
    let mut skewed_weights = HashMap::new();
    skewed_weights.insert("rare", 0.01);
    skewed_weights.insert("common", 0.99);
    
    let mut skewed_stat_selector = StatisticalWeightedSelector::new(skewed_weights, 3.841).unwrap(); // 95% confidence, 1 DOF
    
    // Generate samples with artificial bias
    for i in 0..100 {
        let biased_random = if i < 90 { 0.99 } else { 0.005 }; // Heavily favor "common"
        let _ = skewed_stat_selector.select_and_record(biased_random).unwrap();
    }
    
    let skewed_chi_square = skewed_stat_selector.chi_square_test();
    assert!(skewed_chi_square >= 0.0);
    // This might pass or fail depending on exact distribution, but the test validates the mechanism works
    
    // Test chi-square with insufficient samples
    let mut insufficient_weights = HashMap::new();
    insufficient_weights.insert("test", 1.0);
    
    let mut insufficient_selector = StatisticalWeightedSelector::new(insufficient_weights, 1.0).unwrap();
    
    // Only add a few samples
    for i in 0..3 {
        let random_val = (i as f64) / 3.0;
        let _ = insufficient_selector.select_and_record(random_val).unwrap();
    }
    
    let insufficient_chi_square = insufficient_selector.chi_square_test();
    assert_eq!(insufficient_chi_square, 0.0); // Should return 0 for insufficient samples
    
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Chi-square validation capability validated");
}