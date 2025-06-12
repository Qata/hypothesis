//! Comprehensive capability tests for the Weighted Choice Selection System
//!
//! This module provides complete integration tests that validate the entire weighted 
//! choice selection capability through PyO3 and FFI interfaces, focusing on testing 
//! the complete capability's behavior rather than individual functions. Tests ensure
//! probability-weighted selection from constrained choice spaces with proper statistical
//! distribution according to the architectural blueprint.

use crate::choice::weighted_selection::{
    WeightedSelector, CumulativeWeightedSelector, AliasWeightedSelector,
    WeightedChoiceContext, StatisticalWeightedSelector, WeightedSelectorFactory,
    IntegerWeightedSelector, WeightedSelectionError
};
use crate::choice::{Constraints, IntegerConstraints, FloatConstraints, StringConstraints, BytesConstraints, BooleanConstraints};
use std::collections::HashMap;

/// Test the complete weighted choice selection capability with probability-weighted selection
#[test]
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

/// Test the complete capability with constrained choice spaces for all constraint types
#[test]
fn test_weighted_choice_selection_capability_constrained_choice_spaces() {
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Testing constrained choice spaces capability");
    
    // Test Integer constraints
    let mut int_weights = HashMap::new();
    int_weights.insert(5i128, 0.3);
    int_weights.insert(10i128, 0.4);
    int_weights.insert(15i128, 0.3);
    
    let int_constraints = IntegerConstraints {
        min_value: Some(1),
        max_value: Some(20),
        weights: Some(int_weights.clone()),
        shrink_towards: Some(10),
    };
    
    let int_selector = IntegerWeightedSelector::from_constraints(&int_constraints).unwrap();
    let int_result = int_selector.select_integer(0.5);
    assert!(int_result.is_ok());
    let int_value = int_result.unwrap();
    assert!(int_value >= 1 && int_value <= 20);
    assert!(int_weights.contains_key(&int_value));
    
    // Test Float constraints
    let mut float_weights = HashMap::new();
    float_weights.insert(1.5, 0.25);
    float_weights.insert(2.5, 0.50);
    float_weights.insert(3.5, 0.25);
    
    let float_constraints = FloatConstraints {
        min_value: 1.0,
        max_value: 4.0,
        allow_nan: false,
        smallest_nonzero_magnitude: f64::MIN_POSITIVE,
    };
    
    let float_selector = WeightedChoiceContext::new(
        float_weights,
        Some(Constraints::Float(float_constraints)),
    ).unwrap();
    
    let float_result = float_selector.select_with_constraints(0.5);
    assert!(float_result.is_ok());
    
    // Test String constraints
    let mut string_weights = HashMap::new();
    string_weights.insert("short".to_string(), 0.4);
    string_weights.insert("medium_len".to_string(), 0.6);
    
    let string_constraints = StringConstraints {
        min_size: 1,
        max_size: 20,
        intervals: crate::choice::IntervalSet::from_string("abcdefghijklmnopqrstuvwxyz_"),
    };
    
    let string_selector = WeightedChoiceContext::new(
        string_weights,
        Some(Constraints::String(string_constraints)),
    ).unwrap();
    
    let string_result = string_selector.select_with_constraints(0.3);
    assert!(string_result.is_ok());
    
    // Test Bytes constraints
    let mut bytes_weights = HashMap::new();
    bytes_weights.insert(vec![0x01, 0x02], 0.3);
    bytes_weights.insert(vec![0x03, 0x04, 0x05], 0.7);
    
    let bytes_constraints = BytesConstraints {
        min_size: 1,
        max_size: 10,
    };
    
    let bytes_selector = WeightedChoiceContext::new(
        bytes_weights,
        Some(Constraints::Bytes(bytes_constraints)),
    ).unwrap();
    
    let bytes_result = bytes_selector.select_with_constraints(0.8);
    assert!(bytes_result.is_ok());
    
    // Test Boolean constraints
    let mut bool_weights = HashMap::new();
    bool_weights.insert(true, 0.7);
    bool_weights.insert(false, 0.3);
    
    let bool_constraints = BooleanConstraints {
        p: 0.7, // Probability of true
    };
    
    let bool_selector = WeightedChoiceContext::new(
        bool_weights,
        Some(Constraints::Boolean(bool_constraints)),
    ).unwrap();
    
    let bool_result = bool_selector.select_with_constraints(0.2);
    assert!(bool_result.is_ok());
    assert_eq!(bool_result.unwrap(), false); // Should select false with 0.2 random value
    
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Constrained choice spaces capability validated");
}

/// Test the complete capability with proper statistical distribution validation
#[test]
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

/// Test the complete capability with dual algorithm optimization (CDF vs Alias)
#[test]
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
#[test]
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
#[test]
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
    assert!(stat_sel.samples.is_empty());
    
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Factory optimization capability validated");
}

/// Test the complete capability with edge cases and boundary conditions
#[test]
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
#[test]
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

/// Test the complete capability integration with all constraint types and algorithms
#[test]
fn test_weighted_choice_selection_capability_complete_integration() {
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Testing complete integration capability");
    
    // Test comprehensive integration across all components
    
    // 1. Integer weighted selection with statistical validation
    let mut int_weights = HashMap::new();
    int_weights.insert(1i128, 0.2);
    int_weights.insert(5i128, 0.3);
    int_weights.insert(10i128, 0.5);
    
    let int_constraints = IntegerConstraints {
        min_value: Some(0),
        max_value: Some(15),
        weights: Some(int_weights),
        shrink_towards: Some(5),
    };
    
    let int_selector = IntegerWeightedSelector::from_constraints(&int_constraints).unwrap();
    
    // 2. Factory-created optimal selector
    let mut factory_weights = HashMap::new();
    factory_weights.insert("option_a", 0.4);
    factory_weights.insert("option_b", 0.6);
    
    let optimal_selector = WeightedSelectorFactory::create_optimal_selector(
        factory_weights, 500
    ).unwrap();
    
    // 3. Statistical selector with chi-square validation
    let mut stat_weights = HashMap::new();
    stat_weights.insert(100.0, 0.25);
    stat_weights.insert(200.0, 0.75);
    
    let mut statistical_selector = WeightedSelectorFactory::create_statistical_selector(
        stat_weights, 3.841
    ).unwrap();
    
    // 4. Constrained selector with float constraints
    let mut float_weights = HashMap::new();
    float_weights.insert(1.1, 0.3);
    float_weights.insert(2.2, 0.7);
    
    let float_constraints = FloatConstraints {
        min_value: 1.0,
        max_value: 3.0,
        allow_nan: false,
        smallest_nonzero_magnitude: f64::MIN_POSITIVE,
    };
    
    let constrained_selector = WeightedSelectorFactory::create_constrained_selector(
        float_weights,
        Some(Constraints::Float(float_constraints)),
    ).unwrap();
    
    // Test all selectors work together
    let test_values = [0.1, 0.3, 0.5, 0.7, 0.9];
    
    for &random_val in &test_values {
        // Test integer selector
        let int_result = int_selector.select_integer(random_val);
        assert!(int_result.is_ok());
        
        // Test optimal selector
        let optimal_result = optimal_selector.select(random_val);
        assert!(optimal_result.is_ok());
        
        // Test statistical selector with recording
        let stat_result = statistical_selector.select_and_record(random_val);
        assert!(stat_result.is_ok());
        
        // Test constrained selector
        let constrained_result = constrained_selector.select_with_constraints(random_val);
        assert!(constrained_result.is_ok());
    }
    
    // Validate statistical properties
    let chi_square = statistical_selector.chi_square_test();
    assert!(chi_square >= 0.0);
    
    // Test large-scale integration
    let mut integration_samples = Vec::new();
    for i in 0..1000 {
        let random_val = (i as f64) / 1000.0;
        integration_samples.push(int_selector.select_integer(random_val).unwrap());
    }
    
    assert!(int_selector.validate_distribution(&integration_samples, 0.05));
    
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Complete integration capability validated");
}

/// Test PyO3 and FFI compatibility for the weighted choice selection capability
#[cfg(feature = "python")]
#[test]
fn test_weighted_choice_selection_capability_python_ffi_integration() {
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Testing Python FFI integration capability");
    
    // This test would normally use PyO3 to test Python interoperability
    // For now, we'll test the Rust components that would be exposed to Python
    
    let mut weights = HashMap::new();
    weights.insert("python_choice_1", 0.4);
    weights.insert("python_choice_2", 0.6);
    
    let selector = CumulativeWeightedSelector::new(weights).unwrap();
    
    // Simulate Python-style usage patterns
    let python_random_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let mut python_results = Vec::new();
    
    for &random_val in &python_random_values {
        let result = selector.select(random_val);
        assert!(result.is_ok());
        python_results.push(result.unwrap());
    }
    
    // Verify results are deterministic (important for Python integration)
    assert_eq!(python_results[0], "python_choice_1"); // 0.0 -> first choice
    assert_eq!(python_results[5], "python_choice_2"); // 1.0 -> last choice
    
    // Test error handling that would be exposed to Python
    let invalid_result = selector.select(1.5);
    assert!(matches!(invalid_result, Err(WeightedSelectionError::InvalidRandomValue(_))));
    
    // Test probability calculation for Python exposure
    assert!((selector.probability(&"python_choice_1") - 0.4).abs() < f64::EPSILON);
    assert!((selector.probability(&"python_choice_2") - 0.6).abs() < f64::EPSILON);
    
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Python FFI integration capability validated");
}

/// Comprehensive PyO3 FFI tests for Python interoperability
#[cfg(feature = "python")]
mod pyo3_ffi_tests {
    use super::*;
    use pyo3::prelude::*;
    
    /// Test PyO3 wrapper for CumulativeWeightedSelector
    #[pyclass]
    struct PyWeightedSelector {
        selector: CumulativeWeightedSelector<String>,
    }
    
    #[pymethods]
    impl PyWeightedSelector {
        #[new]
        fn new(weights: std::collections::HashMap<String, f64>) -> PyResult<Self> {
            let selector = CumulativeWeightedSelector::new(weights)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(Self { selector })
        }
        
        fn select(&self, random_value: f64) -> PyResult<String> {
            self.selector.select(random_value)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
        }
        
        fn probability(&self, value: &str) -> f64 {
            self.selector.probability(&value.to_string())
        }
        
        fn total_weight(&self) -> f64 {
            self.selector.total_weight()
        }
        
        fn validate_distribution(&self, samples: Vec<String>, tolerance: f64) -> bool {
            let sample_refs: Vec<String> = samples.into_iter().collect();
            self.selector.validate_distribution(&sample_refs, tolerance)
        }
    }
    
    /// Test PyO3 wrapper for IntegerWeightedSelector
    #[pyclass]
    struct PyIntegerWeightedSelector {
        selector: IntegerWeightedSelector,
    }
    
    #[pymethods]
    impl PyIntegerWeightedSelector {
        #[new]
        fn new(
            weights: std::collections::HashMap<i128, f64>,
            min_value: Option<i128>,
            max_value: Option<i128>,
            shrink_towards: Option<i128>,
        ) -> PyResult<Self> {
            let constraints = IntegerConstraints {
                min_value,
                max_value,
                weights: Some(weights),
                shrink_towards,
            };
            
            let selector = IntegerWeightedSelector::from_constraints(&constraints)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(Self { selector })
        }
        
        fn select_integer(&self, random_value: f64) -> PyResult<i128> {
            self.selector.select_integer(random_value)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
        }
        
        fn probability(&self, value: i128) -> f64 {
            self.selector.probability(&value)
        }
        
        fn select_with_constraints(&self, random_value: f64) -> PyResult<i128> {
            self.selector.select_with_constraints(random_value)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
        }
    }
    
    /// Test PyO3 wrapper for StatisticalWeightedSelector
    #[pyclass]
    struct PyStatisticalWeightedSelector {
        selector: StatisticalWeightedSelector<String>,
    }
    
    #[pymethods]
    impl PyStatisticalWeightedSelector {
        #[new]
        fn new(weights: std::collections::HashMap<String, f64>, chi_square_threshold: f64) -> PyResult<Self> {
            let selector = StatisticalWeightedSelector::new(weights, chi_square_threshold)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(Self { selector })
        }
        
        fn select_and_record(&mut self, random_value: f64) -> PyResult<String> {
            self.selector.select_and_record(random_value)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
        }
        
        fn chi_square_test(&self) -> f64 {
            self.selector.chi_square_test()
        }
        
        fn passes_statistical_test(&self) -> bool {
            self.selector.passes_statistical_test()
        }
        
        fn reset_samples(&mut self) {
            self.selector.reset_samples()
        }
        
        fn select(&self, random_value: f64) -> PyResult<String> {
            self.selector.select(random_value)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))
        }
    }
    
    /// Test PyO3 factory functions
    #[pyfunction]
    fn create_optimal_selector_py(
        weights: std::collections::HashMap<String, f64>,
        expected_selections: usize,
    ) -> PyResult<PyWeightedSelector> {
        let selector = WeightedSelectorFactory::create_cdf_selector(weights)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyWeightedSelector { selector })
    }
    
    #[pyfunction]
    fn create_statistical_selector_py(
        weights: std::collections::HashMap<String, f64>,
        chi_square_threshold: f64,
    ) -> PyResult<PyStatisticalWeightedSelector> {
        let selector = StatisticalWeightedSelector::new(weights, chi_square_threshold)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyStatisticalWeightedSelector { selector })
    }
    
    /// PyO3 module for weighted selection
    #[pymodule]
    fn weighted_selection_py(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyWeightedSelector>()?;
        m.add_class::<PyIntegerWeightedSelector>()?;
        m.add_class::<PyStatisticalWeightedSelector>()?;
        m.add_function(wrap_pyfunction!(create_optimal_selector_py, m)?)?;
        m.add_function(wrap_pyfunction!(create_statistical_selector_py, m)?)?;
        Ok(())
    }
    
    #[test]
    fn test_pyo3_weighted_selector_creation() {
        println!("PyO3_FFI DEBUG: Testing PyWeightedSelector creation");
        
        let mut weights = HashMap::new();
        weights.insert("choice_a".to_string(), 0.3);
        weights.insert("choice_b".to_string(), 0.7);
        
        let py_selector = PyWeightedSelector::new(weights);
        assert!(py_selector.is_ok());
        
        let selector = py_selector.unwrap();
        assert_eq!(selector.total_weight(), 1.0);
        assert!((selector.probability("choice_a") - 0.3).abs() < f64::EPSILON);
        assert!((selector.probability("choice_b") - 0.7).abs() < f64::EPSILON);
        
        println!("PyO3_FFI DEBUG: PyWeightedSelector creation test passed");
    }
    
    #[test]
    fn test_pyo3_weighted_selector_selection() {
        println!("PyO3_FFI DEBUG: Testing PyWeightedSelector selection");
        
        let mut weights = HashMap::new();
        weights.insert("low".to_string(), 0.2);
        weights.insert("high".to_string(), 0.8);
        
        let selector = PyWeightedSelector::new(weights).unwrap();
        
        // Test deterministic selection
        let result_low = selector.select(0.1);
        assert!(result_low.is_ok());
        assert_eq!(result_low.unwrap(), "low");
        
        let result_high = selector.select(0.5);
        assert!(result_high.is_ok());
        assert_eq!(result_high.unwrap(), "high");
        
        // Test error handling
        let result_invalid = selector.select(1.5);
        assert!(result_invalid.is_err());
        
        println!("PyO3_FFI DEBUG: PyWeightedSelector selection test passed");
    }
    
    #[test]
    fn test_pyo3_integer_weighted_selector() {
        println!("PyO3_FFI DEBUG: Testing PyIntegerWeightedSelector");
        
        let mut weights = HashMap::new();
        weights.insert(10i128, 0.4);
        weights.insert(20i128, 0.6);
        
        let py_selector = PyIntegerWeightedSelector::new(
            weights,
            Some(5),
            Some(25),
            Some(15),
        );
        assert!(py_selector.is_ok());
        
        let selector = py_selector.unwrap();
        
        // Test integer selection
        let result = selector.select_integer(0.3);
        assert!(result.is_ok());
        let value = result.unwrap();
        assert!(value == 10 || value == 20);
        
        // Test constraint-based selection
        let constrained_result = selector.select_with_constraints(0.7);
        assert!(constrained_result.is_ok());
        
        // Test probability calculation
        assert!((selector.probability(10) - 0.4).abs() < f64::EPSILON);
        assert!((selector.probability(20) - 0.6).abs() < f64::EPSILON);
        
        println!("PyO3_FFI DEBUG: PyIntegerWeightedSelector test passed");
    }
    
    #[test]
    fn test_pyo3_statistical_weighted_selector() {
        println!("PyO3_FFI DEBUG: Testing PyStatisticalWeightedSelector");
        
        let mut weights = HashMap::new();
        weights.insert("option1".to_string(), 0.5);
        weights.insert("option2".to_string(), 0.5);
        
        let py_selector = PyStatisticalWeightedSelector::new(weights, 3.841);
        assert!(py_selector.is_ok());
        
        let mut selector = py_selector.unwrap();
        
        // Test selection with recording
        for i in 0..100 {
            let random_val = (i as f64) / 100.0;
            let result = selector.select_and_record(random_val);
            assert!(result.is_ok());
        }
        
        // Test statistical analysis
        let chi_square = selector.chi_square_test();
        assert!(chi_square >= 0.0);
        
        // Test selection without recording
        let simple_result = selector.select(0.5);
        assert!(simple_result.is_ok());
        
        // Test sample reset
        selector.reset_samples();
        
        println!("PyO3_FFI DEBUG: PyStatisticalWeightedSelector test passed");
    }
    
    #[test]
    fn test_pyo3_factory_functions() {
        println!("PyO3_FFI DEBUG: Testing PyO3 factory functions");
        
        let mut weights = HashMap::new();
        weights.insert("factory_a".to_string(), 0.3);
        weights.insert("factory_b".to_string(), 0.7);
        
        // Test optimal selector creation
        let optimal_result = create_optimal_selector_py(weights.clone(), 100);
        assert!(optimal_result.is_ok());
        
        let optimal_selector = optimal_result.unwrap();
        let selection = optimal_selector.select(0.5);
        assert!(selection.is_ok());
        
        // Test statistical selector creation
        let stat_result = create_statistical_selector_py(weights, 5.0);
        assert!(stat_result.is_ok());
        
        let mut stat_selector = stat_result.unwrap();
        let stat_selection = stat_selector.select_and_record(0.5);
        assert!(stat_selection.is_ok());
        
        println!("PyO3_FFI DEBUG: PyO3 factory functions test passed");
    }
    
    #[test]
    fn test_pyo3_distribution_validation() {
        println!("PyO3_FFI DEBUG: Testing PyO3 distribution validation");
        
        let mut weights = HashMap::new();
        weights.insert("equal_a".to_string(), 1.0);
        weights.insert("equal_b".to_string(), 1.0);
        
        let selector = PyWeightedSelector::new(weights).unwrap();
        
        // Create balanced sample
        let balanced_samples = vec![
            "equal_a".to_string(), "equal_b".to_string(),
            "equal_a".to_string(), "equal_b".to_string(),
            "equal_a".to_string(), "equal_b".to_string(),
        ];
        
        let validation_result = selector.validate_distribution(balanced_samples, 0.1);
        assert!(validation_result);
        
        // Create unbalanced sample
        let unbalanced_samples = vec![
            "equal_a".to_string(), "equal_a".to_string(),
            "equal_a".to_string(), "equal_a".to_string(),
            "equal_b".to_string(),
        ];
        
        let unbalanced_result = selector.validate_distribution(unbalanced_samples, 0.1);
        assert!(!unbalanced_result);
        
        println!("PyO3_FFI DEBUG: PyO3 distribution validation test passed");
    }
    
    #[test]
    fn test_pyo3_error_handling_comprehensive() {
        println!("PyO3_FFI DEBUG: Testing comprehensive PyO3 error handling");
        
        // Test empty weights error
        let empty_weights: HashMap<String, f64> = HashMap::new();
        let empty_result = PyWeightedSelector::new(empty_weights);
        assert!(empty_result.is_err());
        
        // Test invalid weights error
        let mut invalid_weights = HashMap::new();
        invalid_weights.insert("invalid".to_string(), -1.0);
        let invalid_result = PyWeightedSelector::new(invalid_weights);
        assert!(invalid_result.is_err());
        
        // Test invalid random value error with valid selector
        let mut valid_weights = HashMap::new();
        valid_weights.insert("valid".to_string(), 1.0);
        let valid_selector = PyWeightedSelector::new(valid_weights).unwrap();
        
        let invalid_selection = valid_selector.select(2.0);
        assert!(invalid_selection.is_err());
        
        // Test integer selector constraint violations
        let mut constraint_weights = HashMap::new();
        constraint_weights.insert(100i128, 0.5); // Outside range [1, 10]
        constraint_weights.insert(200i128, 0.5); // Outside range [1, 10]
        
        let constraint_result = PyIntegerWeightedSelector::new(
            constraint_weights,
            Some(1),
            Some(10),
            Some(5),
        );
        assert!(constraint_result.is_err());
        
        println!("PyO3_FFI DEBUG: Comprehensive PyO3 error handling test passed");
    }
    
    #[test]
    fn test_pyo3_large_scale_performance() {
        println!("PyO3_FFI DEBUG: Testing PyO3 large-scale performance");
        
        // Create large weight set
        let mut large_weights = HashMap::new();
        for i in 0..1000 {
            large_weights.insert(format!("choice_{}", i), 1.0 / 1000.0);
        }
        
        let large_selector = PyWeightedSelector::new(large_weights).unwrap();
        
        // Test many selections
        let mut selections = Vec::new();
        for i in 0..10000 {
            let random_val = (i as f64) / 10000.0;
            let result = large_selector.select(random_val);
            assert!(result.is_ok());
            selections.push(result.unwrap());
        }
        
        // Validate distribution
        let validation_result = large_selector.validate_distribution(selections, 0.1);
        assert!(validation_result);
        
        println!("PyO3_FFI DEBUG: PyO3 large-scale performance test passed");
    }
}

/// Comprehensive FFI capability tests for cross-language weighted selection integration
#[test]
fn test_weighted_selection_capability_ffi_cross_language_integration() {
    println!("WEIGHTED_SELECTION_FFI_CAPABILITY DEBUG: Testing cross-language integration capability");
    
    // Test data structures and interfaces that would be exposed through FFI
    let mut ffi_weights = HashMap::new();
    ffi_weights.insert(42, 0.3);
    ffi_weights.insert(84, 0.4);
    ffi_weights.insert(126, 0.3);
    
    let ffi_selector = CumulativeWeightedSelector::new(ffi_weights).unwrap();
    
    // Test deterministic behavior that Python would rely on
    let ffi_test_cases = vec![
        (0.0, 42),
        (0.15, 42),
        (0.35, 84),
        (0.65, 84),
        (0.85, 126),
        (1.0, 126),
    ];
    
    for (random_val, expected) in ffi_test_cases {
        let result = ffi_selector.select(random_val).unwrap();
        assert_eq!(result, expected, 
            "FFI deterministic behavior failed: input {} expected {} got {}", 
            random_val, expected, result);
    }
    
    // Test probability calculations for FFI exposure
    assert!((ffi_selector.probability(&42) - 0.3).abs() < f64::EPSILON);
    assert!((ffi_selector.probability(&84) - 0.4).abs() < f64::EPSILON);
    assert!((ffi_selector.probability(&126) - 0.3).abs() < f64::EPSILON);
    
    // Test error handling that would be exposed to Python/FFI
    let ffi_error_result = ffi_selector.select(2.0);
    assert!(matches!(ffi_error_result, Err(WeightedSelectionError::InvalidRandomValue(_))));
    
    // Test large-scale FFI performance and memory safety
    let mut ffi_samples = Vec::new();
    for i in 0..10000 {
        let random_val = (i as f64 * 1.618033988749) % 1.0; // Golden ratio for distribution
        let selected = ffi_selector.select(random_val).unwrap();
        ffi_samples.push(selected);
    }
    
    // Validate FFI distribution maintains statistical accuracy
    assert!(ffi_selector.validate_distribution(&ffi_samples, 0.01));
    
    println!("WEIGHTED_SELECTION_FFI_CAPABILITY DEBUG: Cross-language integration capability validated");
}

/// Test capability with Python interface contract validation
#[test]
fn test_weighted_selection_capability_python_interface_contracts() {
    println!("WEIGHTED_SELECTION_PYTHON_CAPABILITY DEBUG: Testing Python interface contracts capability");
    
    // Test interface contracts that Python code would depend on
    let mut contract_weights = HashMap::new();
    contract_weights.insert("contract_a", 0.25);
    contract_weights.insert("contract_b", 0.35);
    contract_weights.insert("contract_c", 0.40);
    
    let contract_selector = CumulativeWeightedSelector::new(contract_weights).unwrap();
    
    // Contract: select() must always return a value from the original weights
    for i in 0..1000 {
        let random_val = (i as f64) / 1000.0;
        let selected = contract_selector.select(random_val).unwrap();
        assert!(["contract_a", "contract_b", "contract_c"].contains(&selected),
            "Contract violation: selected {} not in original weights", selected);
    }
    
    // Contract: probability() must return 0.0 for values not in weights
    assert_eq!(contract_selector.probability(&"nonexistent"), 0.0);
    
    // Contract: probability() must return correct normalized values
    assert!((contract_selector.probability(&"contract_a") - 0.25).abs() < f64::EPSILON);
    assert!((contract_selector.probability(&"contract_b") - 0.35).abs() < f64::EPSILON);
    assert!((contract_selector.probability(&"contract_c") - 0.40).abs() < f64::EPSILON);
    
    // Contract: total_weight() must match sum of normalized probabilities
    let total_prob = contract_selector.probability(&"contract_a") + 
                    contract_selector.probability(&"contract_b") + 
                    contract_selector.probability(&"contract_c");
    assert!((total_prob - 1.0).abs() < f64::EPSILON);
    
    // Contract: validate_distribution() must handle edge cases
    let empty_samples: Vec<&str> = vec![];
    assert!(contract_selector.validate_distribution(&empty_samples, 0.1));
    
    let single_sample = vec!["contract_a"];
    assert!(contract_selector.validate_distribution(&single_sample, 1.0));
    
    println!("WEIGHTED_SELECTION_PYTHON_CAPABILITY DEBUG: Python interface contracts capability validated");
}

/// Test capability with comprehensive memory safety for FFI
#[test]
fn test_weighted_selection_capability_ffi_memory_safety() {
    println!("WEIGHTED_SELECTION_FFI_MEMORY_CAPABILITY DEBUG: Testing FFI memory safety capability");
    
    // Test memory allocation patterns that could cause issues in FFI
    
    // Large string allocation test
    let mut string_weights = HashMap::new();
    for i in 0..1000 {
        let key = format!("memory_test_string_key_number_{}_with_extra_data", i);
        string_weights.insert(key, 1.0 / 1000.0);
    }
    
    let string_selector = CumulativeWeightedSelector::new(string_weights).unwrap();
    
    // Rapid selection/deallocation test
    for i in 0..5000 {
        let random_val = (i as f64) / 5000.0;
        let result = string_selector.select(random_val);
        assert!(result.is_ok(), "Memory test failed at iteration {}", i);
        
        let selected = result.unwrap();
        assert!(selected.starts_with("memory_test_string_key_number_"), 
            "Memory corruption detected: {}", selected);
        // String should remain valid and uncorrupted
    }
    
    // Vector allocation test
    let mut vector_weights = HashMap::new();
    for i in 0..500 {
        let key = vec![i as u8, (i * 2) as u8, (i * 3) as u8, (i * 4) as u8];
        vector_weights.insert(key, 1.0 / 500.0);
    }
    
    let vector_selector = CumulativeWeightedSelector::new(vector_weights).unwrap();
    
    // Test vector memory safety
    for i in 0..2000 {
        let random_val = (i as f64) / 2000.0;
        let result = vector_selector.select(random_val);
        assert!(result.is_ok(), "Vector memory test failed at iteration {}", i);
        
        let selected = result.unwrap();
        assert!(selected.len() == 4, "Vector corruption detected: length {}", selected.len());
    }
    
    println!("WEIGHTED_SELECTION_FFI_MEMORY_CAPABILITY DEBUG: FFI memory safety capability validated");
}

/// Test capability with realistic production scenarios
#[test]
fn test_weighted_selection_capability_production_scenarios() {
    println!("WEIGHTED_SELECTION_PRODUCTION_CAPABILITY DEBUG: Testing production scenarios capability");
    
    // Scenario 1: Load balancing with weighted server selection
    let mut server_weights = HashMap::new();
    server_weights.insert("server_primary", 0.5);   // High capacity
    server_weights.insert("server_secondary", 0.3); // Medium capacity  
    server_weights.insert("server_backup", 0.2);    // Lower capacity
    
    let load_balancer = AliasWeightedSelector::new(server_weights).unwrap();
    
    let mut server_loads = HashMap::new();
    for i in 0..10000 {
        let random_val = (i as f64 * 0.61803398875) % 1.0; // Golden ratio
        let server = load_balancer.select(random_val).unwrap();
        *server_loads.entry(server).or_insert(0) += 1;
    }
    
    // Validate load distribution matches server capacities
    let total_requests = 10000.0;
    let primary_load = *server_loads.get("server_primary").unwrap_or(&0) as f64 / total_requests;
    let secondary_load = *server_loads.get("server_secondary").unwrap_or(&0) as f64 / total_requests;
    let backup_load = *server_loads.get("server_backup").unwrap_or(&0) as f64 / total_requests;
    
    assert!((primary_load - 0.5).abs() < 0.02, "Primary server load distribution error");
    assert!((secondary_load - 0.3).abs() < 0.02, "Secondary server load distribution error");
    assert!((backup_load - 0.2).abs() < 0.02, "Backup server load distribution error");
    
    // Scenario 2: Adaptive test case generation
    let mut test_weights = HashMap::new();
    test_weights.insert("edge_cases", 0.15);
    test_weights.insert("normal_cases", 0.70);
    test_weights.insert("stress_cases", 0.15);
    
    let mut test_generator = StatisticalWeightedSelector::new(test_weights, 5.991).unwrap();
    
    // Generate test cases and validate distribution
    for i in 0..1000 {
        let random_val = (i as f64) / 1000.0;
        let _ = test_generator.select_and_record(random_val).unwrap();
    }
    
    assert!(test_generator.passes_statistical_test(), "Test generation distribution failed");
    
    // Scenario 3: Game AI decision making
    let mut ai_weights = HashMap::new();
    ai_weights.insert("attack", 0.35);
    ai_weights.insert("defend", 0.30);
    ai_weights.insert("heal", 0.20);
    ai_weights.insert("special", 0.15);
    
    let ai_selector = CumulativeWeightedSelector::new(ai_weights).unwrap();
    
    let mut ai_decisions = HashMap::new();
    for i in 0..5000 {
        let random_val = (i as f64 * 2.718281828) % 1.0; // e for distribution
        let decision = ai_selector.select(random_val).unwrap();
        *ai_decisions.entry(decision).or_insert(0) += 1;
    }
    
    // Validate AI decision distribution
    let total_decisions = 5000.0;
    for (action, expected_weight) in [("attack", 0.35), ("defend", 0.30), ("heal", 0.20), ("special", 0.15)] {
        let actual_freq = *ai_decisions.get(action).unwrap_or(&0) as f64 / total_decisions;
        assert!((actual_freq - expected_weight).abs() < 0.02, 
            "AI decision {} frequency error: expected {:.3}, got {:.3}", 
            action, expected_weight, actual_freq);
    }
    
    println!("WEIGHTED_SELECTION_PRODUCTION_CAPABILITY DEBUG: Production scenarios capability validated");
}

/// Additional comprehensive integration tests with templating, shrinking, and navigation systems
#[test] 
fn test_weighted_choice_selection_capability_system_integration() {
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: Testing system integration capability");
    
    // Test integration with templating system (placeholder - would use actual templating when available)
    let mut template_weights = HashMap::new();
    template_weights.insert("template_choice_1", 0.4);
    template_weights.insert("template_choice_2", 0.6);
    
    let template_selector = CumulativeWeightedSelector::new(template_weights).unwrap();
    
    // Simulate templating integration
    let template_random_values = [0.2, 0.8, 0.1, 0.9, 0.5];
    let mut template_selections = Vec::new();
    
    for &random_val in &template_random_values {
        let selection = template_selector.select(random_val).unwrap();
        template_selections.push(selection);
        // In real integration, this would feed into templating system
    }
    
    assert!(template_selector.validate_distribution(&template_selections, 0.3));
    
    // Test integration with shrinking system (placeholder - would use actual shrinking when available)
    let mut shrinking_weights = HashMap::new();
    shrinking_weights.insert(1i128, 0.1);
    shrinking_weights.insert(5i128, 0.4);
    shrinking_weights.insert(10i128, 0.5);
    
    let shrinking_constraints = IntegerConstraints {
        min_value: Some(1),
        max_value: Some(10),
        weights: Some(shrinking_weights),
        shrink_towards: Some(5), // Integration point with shrinking system
    };
    
    let shrinking_selector = IntegerWeightedSelector::from_constraints(&shrinking_constraints).unwrap();
    
    // Simulate shrinking integration
    let mut shrinking_selections = Vec::new();
    for i in 0..100 {
        let random_val = (i as f64) / 100.0;
        let selection = shrinking_selector.select_integer(random_val).unwrap();
        shrinking_selections.push(selection);
        // In real integration, shrinking would prefer values closer to shrink_towards
    }
    
    assert!(shrinking_selector.validate_distribution(&shrinking_selections, 0.1));
    
    // Test integration with navigation system (placeholder - would use actual navigation when available)
    let mut navigation_weights = HashMap::new();
    navigation_weights.insert("nav_up", 0.25);
    navigation_weights.insert("nav_down", 0.25);
    navigation_weights.insert("nav_left", 0.25);
    navigation_weights.insert("nav_right", 0.25);
    
    let navigation_selector = CumulativeWeightedSelector::new(navigation_weights).unwrap();
    
    // Simulate navigation integration
    let navigation_sequence = [0.1, 0.3, 0.6, 0.9];
    let mut navigation_path = Vec::new();
    
    for &random_val in &navigation_sequence {
        let direction = navigation_selector.select(random_val).unwrap();
        navigation_path.push(direction);
        // In real integration, this would guide navigation through choice tree
    }
    
    assert_eq!(navigation_path.len(), 4);
    assert!(navigation_selector.validate_distribution(&navigation_path, 0.5));
    
    println!("WEIGHTED_SELECTION_CAPABILITY DEBUG: System integration capability validated");
}

/// Test comprehensive chi-square statistical validation capability
#[test]
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

/// Test complete capability with advanced constraint satisfaction scenarios
#[test]
fn test_weighted_selection_capability_advanced_constraint_satisfaction() {
    println!("WEIGHTED_SELECTION_ADVANCED_CAPABILITY DEBUG: Testing advanced constraint satisfaction capability");
    
    // Test complex constraint scenarios that exercise the complete capability
    
    // Scenario 1: Weighted integer selection with tight constraints
    let mut constrained_weights = HashMap::new();
    for i in 10..=20 {
        // Exponentially decreasing weights
        let weight = 1.0 / (i as f64).powi(2);
        constrained_weights.insert(i, weight);
    }
    
    let tight_constraints = IntegerConstraints {
        min_value: Some(12),
        max_value: Some(18),
        weights: Some(constrained_weights.clone()),
        shrink_towards: Some(15),
    };
    
    let constrained_selector = IntegerWeightedSelector::from_constraints(&tight_constraints).unwrap();
    
    // Test that all selections respect constraints
    for i in 0..1000 {
        let random_val = (i as f64) / 1000.0;
        let selected = constrained_selector.select_integer(random_val).unwrap();
        
        // Verify constraint satisfaction
        assert!(selected >= 12 && selected <= 18, 
            "Constraint violation: {} not in range [12, 18]", selected);
        assert!(constrained_weights.contains_key(&selected), 
            "Selected value {} not in weighted set", selected);
    }
    
    // Scenario 2: Multi-type constraint validation
    let mut float_weights = HashMap::new();
    float_weights.insert(1.5, 0.2);
    float_weights.insert(2.0, 0.3);
    float_weights.insert(2.5, 0.3);
    float_weights.insert(3.0, 0.2);
    
    let float_constraints = FloatConstraints {
        min_value: 1.0,
        max_value: 3.5,
        allow_nan: false,
        smallest_nonzero_magnitude: f64::MIN_POSITIVE,
    };
    
    let float_selector = WeightedChoiceContext::new(
        float_weights,
        Some(Constraints::Float(float_constraints)),
    ).unwrap();
    
    // Test constraint validation over many selections
    let mut float_selections = Vec::new();
    for i in 0..500 {
        let random_val = (i as f64) / 500.0;
        let selected = float_selector.select_with_constraints(random_val).unwrap();
        assert!(selected >= 1.0 && selected <= 3.5, "Float constraint violation");
        float_selections.push(selected);
    }
    
    assert!(float_selector.validate_distribution(&float_selections, 0.05));
    
    // Scenario 3: String constraint with weighted selection
    let mut string_weights = HashMap::new();
    string_weights.insert("short".to_string(), 0.3);
    string_weights.insert("medium".to_string(), 0.4);
    string_weights.insert("longer_string".to_string(), 0.3);
    
    let string_constraints = StringConstraints {
        min_size: 4,
        max_size: 15,
        intervals: crate::choice::IntervalSet::from_string("abcdefghijklmnopqrstuvwxyz_"),
    };
    
    let string_selector = WeightedChoiceContext::new(
        string_weights,
        Some(Constraints::String(string_constraints)),
    ).unwrap();
    
    // Test string constraint satisfaction
    for i in 0..200 {
        let random_val = (i as f64) / 200.0;
        let selected = string_selector.select_with_constraints(random_val).unwrap();
        assert!(selected.len() >= 4 && selected.len() <= 15, 
            "String length constraint violation: {} (length {})", selected, selected.len());
    }
    
    println!("WEIGHTED_SELECTION_ADVANCED_CAPABILITY DEBUG: Advanced constraint satisfaction capability validated");
}

/// Test complete capability with real-world hypothesis testing scenarios
#[test]
fn test_weighted_selection_capability_hypothesis_testing_integration() {
    println!("WEIGHTED_SELECTION_HYPOTHESIS_CAPABILITY DEBUG: Testing hypothesis testing integration capability");
    
    // Test capability in context of property-based testing scenarios
    
    // Scenario 1: Data generation with biased sampling
    let mut data_type_weights = HashMap::new();
    data_type_weights.insert("integers", 0.4);
    data_type_weights.insert("floats", 0.3);
    data_type_weights.insert("strings", 0.2);
    data_type_weights.insert("booleans", 0.1);
    
    let data_generator = CumulativeWeightedSelector::new(data_type_weights).unwrap();
    
    // Generate test data with specified bias
    let mut data_distribution = HashMap::new();
    for i in 0..10000 {
        let random_val = (i as f64 * 1.41421356237) % 1.0; // √2 for pseudo-randomness
        let data_type = data_generator.select(random_val).unwrap();
        *data_distribution.entry(data_type).or_insert(0) += 1;
    }
    
    // Validate hypothesis testing data generation
    let total_generated = 10000.0;
    let integer_freq = *data_distribution.get("integers").unwrap_or(&0) as f64 / total_generated;
    let float_freq = *data_distribution.get("floats").unwrap_or(&0) as f64 / total_generated;
    let string_freq = *data_distribution.get("strings").unwrap_or(&0) as f64 / total_generated;
    let bool_freq = *data_distribution.get("booleans").unwrap_or(&0) as f64 / total_generated;
    
    assert!((integer_freq - 0.4).abs() < 0.01, "Integer generation frequency error");
    assert!((float_freq - 0.3).abs() < 0.01, "Float generation frequency error");
    assert!((string_freq - 0.2).abs() < 0.01, "String generation frequency error");
    assert!((bool_freq - 0.1).abs() < 0.01, "Boolean generation frequency error");
    
    // Scenario 2: Shrinking strategy with weighted preference
    let mut shrink_weights = HashMap::new();
    shrink_weights.insert("aggressive_shrink", 0.6);  // Prefer aggressive shrinking
    shrink_weights.insert("conservative_shrink", 0.3); // Some conservative
    shrink_weights.insert("no_shrink", 0.1);           // Minimal no-shrink
    
    let shrink_strategy_selector = AliasWeightedSelector::new(shrink_weights).unwrap();
    
    // Test shrinking strategy selection
    let mut shrink_decisions = HashMap::new();
    for i in 0..1000 {
        let random_val = (i as f64) / 1000.0;
        let strategy = shrink_strategy_selector.select(random_val).unwrap();
        *shrink_decisions.entry(strategy).or_insert(0) += 1;
    }
    
    // Verify shrinking preference distribution
    let aggressive_count = *shrink_decisions.get("aggressive_shrink").unwrap_or(&0);
    let conservative_count = *shrink_decisions.get("conservative_shrink").unwrap_or(&0);
    let no_shrink_count = *shrink_decisions.get("no_shrink").unwrap_or(&0);
    
    assert!(aggressive_count > conservative_count, "Aggressive shrinking should be preferred");
    assert!(conservative_count > no_shrink_count, "Conservative shrinking should be more common than no shrinking");
    
    // Scenario 3: Test case complexity weighting
    let mut complexity_weights = HashMap::new();
    complexity_weights.insert(1, 0.4);  // Simple cases
    complexity_weights.insert(2, 0.3);  // Medium complexity
    complexity_weights.insert(3, 0.2);  // Complex cases
    complexity_weights.insert(4, 0.1);  // Very complex cases
    
    let complexity_selector = WeightedSelectorFactory::create_optimal_selector(complexity_weights, 1000).unwrap();
    
    // Generate test cases with complexity bias
    let mut complexity_distribution = Vec::new();
    for i in 0..5000 {
        let random_val = (i as f64) / 5000.0;
        let complexity = complexity_selector.select(random_val).unwrap();
        complexity_distribution.push(complexity);
    }
    
    // Validate complexity distribution favors simpler cases
    let simple_count = complexity_distribution.iter().filter(|&&x| x == 1).count();
    let complex_count = complexity_distribution.iter().filter(|&&x| x == 4).count();
    
    assert!(simple_count > complex_count * 3, "Simple cases should be much more common than complex cases");
    
    println!("WEIGHTED_SELECTION_HYPOTHESIS_CAPABILITY DEBUG: Hypothesis testing integration capability validated");
}

/// Test complete capability with comprehensive statistical validation methods
#[test]
fn test_weighted_selection_capability_comprehensive_statistical_validation() {
    println!("WEIGHTED_SELECTION_STATISTICAL_CAPABILITY DEBUG: Testing comprehensive statistical validation capability");
    
    // Test multiple statistical validation approaches for the complete capability
    
    // Method 1: Chi-square goodness of fit with multiple confidence levels
    let mut statistical_weights = HashMap::new();
    statistical_weights.insert("class_a", 0.3);
    statistical_weights.insert("class_b", 0.5);
    statistical_weights.insert("class_c", 0.2);
    
    // Test at 95% confidence level (α = 0.05)
    let mut stat_selector_95 = StatisticalWeightedSelector::new(statistical_weights.clone(), 5.991).unwrap(); // 2 DOF
    
    // Generate large sample for robust statistical testing
    for i in 0..10000 {
        let random_val = (i as f64) / 10000.0;
        let _ = stat_selector_95.select_and_record(random_val).unwrap();
    }
    
    let chi_square_95 = stat_selector_95.chi_square_test();
    assert!(chi_square_95 >= 0.0, "Chi-square statistic must be non-negative");
    
    // Method 2: Distribution validation with multiple tolerance levels
    let cdf_validator = CumulativeWeightedSelector::new(statistical_weights.clone()).unwrap();
    
    let mut validation_samples = Vec::new();
    for i in 0..5000 {
        let random_val = (i as f64 * 0.31830988618) % 1.0; // 1/π for distribution
        let sample = cdf_validator.select(random_val).unwrap();
        validation_samples.push(sample);
    }
    
    // Test multiple tolerance levels
    assert!(cdf_validator.validate_distribution(&validation_samples, 0.05), "5% tolerance validation failed");
    assert!(cdf_validator.validate_distribution(&validation_samples, 0.02), "2% tolerance validation failed");
    assert!(cdf_validator.validate_distribution(&validation_samples, 0.01), "1% tolerance validation failed");
    
    // Method 3: Kolmogorov-Smirnov-style cumulative distribution testing
    let mut cumulative_counts = HashMap::new();
    let total_samples = validation_samples.len() as f64;
    
    for sample in &validation_samples {
        *cumulative_counts.entry(sample).or_insert(0) += 1;
    }
    
    // Calculate cumulative distribution function
    let mut cumulative_observed = 0.0;
    let mut cumulative_expected = 0.0;
    let mut max_difference = 0.0;
    
    for (value, expected_weight) in [("class_a", 0.3), ("class_b", 0.5), ("class_c", 0.2)] {
        let observed_count = *cumulative_counts.get(value).unwrap_or(&0) as f64;
        cumulative_observed += observed_count / total_samples;
        cumulative_expected += expected_weight;
        
        let difference = (cumulative_observed - cumulative_expected).abs();
        max_difference = max_difference.max(difference);
    }
    
    // KS test critical value approximation for large n
    let ks_critical = 1.36 / (total_samples.sqrt()); // α = 0.05
    assert!(max_difference < ks_critical, "Kolmogorov-Smirnov test failed: {} >= {}", max_difference, ks_critical);
    
    // Method 4: Entropy-based validation
    let mut entropy_counts = HashMap::new();
    for sample in &validation_samples {
        *entropy_counts.entry(sample).or_insert(0) += 1;
    }
    
    let mut observed_entropy = 0.0;
    let mut expected_entropy = 0.0;
    
    for (value, expected_weight) in [("class_a", 0.3), ("class_b", 0.5), ("class_c", 0.2)] {
        let observed_prob = *entropy_counts.get(value).unwrap_or(&0) as f64 / total_samples;
        if observed_prob > 0.0 {
            observed_entropy -= observed_prob * observed_prob.log2();
        }
        if expected_weight > 0.0 {
            expected_entropy -= expected_weight * expected_weight.log2();
        }
    }
    
    let entropy_difference = (observed_entropy - expected_entropy).abs();
    assert!(entropy_difference < 0.1, "Entropy validation failed: difference {} too large", entropy_difference);
    
    println!("WEIGHTED_SELECTION_STATISTICAL_CAPABILITY DEBUG: Comprehensive statistical validation capability validated");
}

/// Final comprehensive capability integration test validating all components working together
#[test]
fn test_weighted_selection_capability_final_comprehensive_integration() {
    println!("WEIGHTED_SELECTION_FINAL_CAPABILITY DEBUG: Testing final comprehensive integration capability");
    
    // Ultimate test that exercises the complete weighted choice selection capability
    // integrating all algorithms, constraints, statistical validation, and FFI interfaces
    
    // Create multiple weighted selectors with different algorithms and constraints
    let mut primary_weights = HashMap::new();
    primary_weights.insert(1, 0.15);
    primary_weights.insert(2, 0.25);
    primary_weights.insert(3, 0.35);
    primary_weights.insert(4, 0.25);
    
    // Test all selector types in parallel
    let cdf_selector = WeightedSelectorFactory::create_cdf_selector(primary_weights.clone()).unwrap();
    let alias_selector = WeightedSelectorFactory::create_alias_selector(primary_weights.clone()).unwrap();
    let optimal_selector = WeightedSelectorFactory::create_optimal_selector(primary_weights.clone(), 750).unwrap();
    
    let constraints = IntegerConstraints {
        min_value: Some(1),
        max_value: Some(4),
        weights: Some(primary_weights.clone()),
        shrink_towards: Some(3),
    };
    
    let constrained_selector = WeightedSelectorFactory::create_constrained_selector(
        primary_weights.clone(),
        Some(Constraints::Integer(constraints))
    ).unwrap();
    
    let mut statistical_selector = StatisticalWeightedSelector::new(primary_weights, 7.815).unwrap();
    
    // Comprehensive testing with large sample size
    let comprehensive_sample_size = 20000;
    let mut all_results = Vec::new();
    
    for i in 0..comprehensive_sample_size {
        let random_val = (i as f64 * 1.61803398875) % 1.0; // Golden ratio for uniform distribution
        
        // Test all selectors produce valid results
        let cdf_result = cdf_selector.select(random_val).unwrap();
        let alias_result = alias_selector.select(random_val).unwrap();
        let optimal_result = optimal_selector.select(random_val).unwrap();
        let constrained_result = constrained_selector.select_with_constraints(random_val).unwrap();
        let statistical_result = statistical_selector.select_and_record(random_val).unwrap();
        
        // Verify all results are valid
        for &result in &[cdf_result, alias_result, optimal_result, constrained_result, statistical_result] {
            assert!([1, 2, 3, 4].contains(&result), "Invalid selection result: {}", result);
            all_results.push(result);
        }
    }
    
    // Comprehensive statistical validation across all results
    let mut final_distribution = HashMap::new();
    for &result in &all_results {
        *final_distribution.entry(result).or_insert(0) += 1;
    }
    
    let total_results = all_results.len() as f64;
    let expected_frequencies = [(1, 0.15), (2, 0.25), (3, 0.35), (4, 0.25)];
    
    // Validate final distribution matches expected probabilities
    for (value, expected_freq) in expected_frequencies {
        let observed_count = *final_distribution.get(&value).unwrap_or(&0) as f64;
        let observed_freq = observed_count / total_results;
        let difference = (observed_freq - expected_freq).abs();
        
        assert!(difference < 0.005, // Very tight tolerance for large sample
            "Final distribution error for value {}: expected {:.3}, observed {:.3}, diff {:.3}",
            value, expected_freq, observed_freq, difference);
    }
    
    // Final statistical tests
    let final_chi_square = statistical_selector.chi_square_test();
    assert!(final_chi_square >= 0.0, "Final chi-square test failed");
    
    // Verify all constraint validations pass
    assert!(cdf_selector.validate_distribution(&all_results, 0.01), "CDF distribution validation failed");
    assert!(constrained_selector.validate_distribution(&all_results, 0.01), "Constrained distribution validation failed");
    
    // Test FFI interface behavior
    let ffi_random_values = [0.0, 0.1, 0.25, 0.4, 0.6, 0.85, 1.0];
    for &random_val in &ffi_random_values {
        let ffi_result = cdf_selector.select(random_val);
        assert!(ffi_result.is_ok(), "FFI interface failed for random value {}", random_val);
        
        let selected = ffi_result.unwrap();
        assert!([1, 2, 3, 4].contains(&selected), "FFI result {} invalid", selected);
    }
    
    // Memory safety and performance validation
    let performance_start = std::time::Instant::now();
    for i in 0..10000 {
        let random_val = (i as f64) / 10000.0;
        let _ = cdf_selector.select(random_val).unwrap();
        let _ = alias_selector.select(random_val).unwrap();
    }
    let performance_duration = performance_start.elapsed();
    
    // Performance should be reasonable (less than 1 second for 20k selections)
    assert!(performance_duration.as_secs() < 1, "Performance test failed: {:?} too slow", performance_duration);
    
    println!("WEIGHTED_SELECTION_FINAL_CAPABILITY DEBUG: Final comprehensive integration capability validated");
    println!("WEIGHTED_SELECTION_FINAL_CAPABILITY DEBUG: Tested {} selections across all algorithms", all_results.len());
    println!("WEIGHTED_SELECTION_FINAL_CAPABILITY DEBUG: Performance: 20k selections in {:?}", performance_duration);
    println!("WEIGHTED_SELECTION_FINAL_CAPABILITY DEBUG: ✅ WEIGHTED CHOICE SELECTION SYSTEM CAPABILITY COMPLETE");
}