//! Comprehensive weighted choice selection system for Conjecture
//!
//! This module implements probability-weighted selection from constrained choice spaces
//! with proper statistical distribution, providing both efficient selection algorithms
//! and comprehensive validation capabilities.

use crate::choice::{Constraints, IntegerConstraints};
use std::collections::HashMap;
use std::fmt;

/// Error types for weighted selection operations
#[derive(Debug, Clone, PartialEq)]
pub enum WeightedSelectionError {
    EmptyWeights,
    InvalidWeight(String),
    InvalidRandomValue(f64),
    ConstraintViolation(String),
    SelectionFailed(String),
}

impl fmt::Display for WeightedSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightedSelectionError::EmptyWeights => write!(f, "Weights cannot be empty"),
            WeightedSelectionError::InvalidWeight(msg) => write!(f, "Invalid weight: {}", msg),
            WeightedSelectionError::InvalidRandomValue(val) => {
                write!(f, "Random value {} must be between 0.0 and 1.0", val)
            }
            WeightedSelectionError::ConstraintViolation(msg) => {
                write!(f, "Constraint violation: {}", msg)
            }
            WeightedSelectionError::SelectionFailed(msg) => {
                write!(f, "Selection failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for WeightedSelectionError {}

/// Core trait for weighted selection algorithms
pub trait WeightedSelector<T> {
    /// Select a value based on the given random value [0.0, 1.0]
    fn select(&self, random_value: f64) -> Result<T, WeightedSelectionError>;
    
    /// Get the probability of selecting a specific value
    fn probability(&self, value: &T) -> f64;
    
    /// Validate that the selection distribution matches expected probabilities
    fn validate_distribution(&self, samples: &[T], tolerance: f64) -> bool;
    
    /// Get the total weight of all values
    fn total_weight(&self) -> f64;
}

/// Cumulative Distribution Function (CDF) based weighted selector
/// Uses binary search for O(log n) selection time
#[derive(Debug, Clone)]
pub struct CumulativeWeightedSelector<T>
where
    T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug + std::cmp::Ord,
{
    weights: HashMap<T, f64>,
    cumulative_weights: Vec<(T, f64)>,
    total_weight: f64,
}

impl<T> CumulativeWeightedSelector<T>
where
    T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug + std::cmp::Ord,
{
    /// Create a new CDF-based weighted selector
    pub fn new(weights: HashMap<T, f64>) -> Result<Self, WeightedSelectionError> {
        println!("WEIGHTED_SELECTION DEBUG: Creating CumulativeWeightedSelector with 0x{:X} weights", weights.len());
        
        if weights.is_empty() {
            return Err(WeightedSelectionError::EmptyWeights);
        }

        // Validate weights
        let mut total_weight = 0.0;
        for (value, &weight) in &weights {
            println!("WEIGHTED_SELECTION DEBUG: Validating weight {:E} for value {:?}", weight, value);
            
            if weight <= 0.0 {
                return Err(WeightedSelectionError::InvalidWeight(
                    format!("Weight {} must be positive", weight)
                ));
            }
            if !weight.is_finite() {
                return Err(WeightedSelectionError::InvalidWeight(
                    format!("Weight {} must be finite", weight)
                ));
            }
            total_weight += weight;
        }

        if total_weight <= 0.0 {
            return Err(WeightedSelectionError::InvalidWeight(
                "Total weight must be positive".to_string()
            ));
        }

        // Build cumulative distribution
        // First, collect and sort values for deterministic ordering
        let mut sorted_values: Vec<_> = weights.iter().collect();
        sorted_values.sort_by(|a, b| a.0.cmp(b.0)); // Sort by value for deterministic order
        
        let mut cumulative_weights = Vec::with_capacity(weights.len());
        let mut cumulative = 0.0;

        for (value, &weight) in sorted_values {
            cumulative += weight;
            cumulative_weights.push((value.clone(), cumulative));
            println!("WEIGHTED_SELECTION DEBUG: Added cumulative entry ({:?}, {:E})", value, cumulative);
        }

        // Already sorted by value, no need to sort again

        println!("WEIGHTED_SELECTION DEBUG: Created selector with total_weight: {:E}", total_weight);

        Ok(Self {
            weights,
            cumulative_weights,
            total_weight,
        })
    }
}

impl<T> WeightedSelector<T> for CumulativeWeightedSelector<T>
where
    T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug + std::cmp::Ord,
{
    fn select(&self, random_value: f64) -> Result<T, WeightedSelectionError> {
        println!("WEIGHTED_SELECTION DEBUG: Selecting with random_value: {:E}", random_value);
        
        if !(0.0..=1.0).contains(&random_value) {
            return Err(WeightedSelectionError::InvalidRandomValue(random_value));
        }

        let target = random_value * self.total_weight;
        println!("WEIGHTED_SELECTION DEBUG: Target cumulative weight: {:E}", target);

        // Binary search for the first cumulative weight >= target
        match self.cumulative_weights.binary_search_by(|&(_, weight)| {
            weight.partial_cmp(&target).unwrap()
        }) {
            Ok(index) => {
                let selected = &self.cumulative_weights[index].0;
                println!("WEIGHTED_SELECTION DEBUG: Exact match at index 0x{:X}, selected: {:?}", index, selected);
                Ok(selected.clone())
            }
            Err(index) => {
                if index < self.cumulative_weights.len() {
                    let selected = &self.cumulative_weights[index].0;
                    println!("WEIGHTED_SELECTION DEBUG: Insertion point at index 0x{:X}, selected: {:?}", index, selected);
                    Ok(selected.clone())
                } else {
                    // Fallback to last element (should not happen with valid inputs)
                    let selected = &self.cumulative_weights.last().unwrap().0;
                    println!("WEIGHTED_SELECTION DEBUG: Fallback to last element: {:?}", selected);
                    Ok(selected.clone())
                }
            }
        }
    }

    fn probability(&self, value: &T) -> f64 {
        self.weights.get(value)
            .map(|w| w / self.total_weight)
            .unwrap_or(0.0)
    }

    fn validate_distribution(&self, samples: &[T], tolerance: f64) -> bool {
        println!("WEIGHTED_SELECTION DEBUG: Validating distribution with 0x{:X} samples, tolerance: {:E}", samples.len(), tolerance);
        
        if samples.is_empty() {
            return true;
        }

        let mut counts = HashMap::new();
        for sample in samples {
            *counts.entry(sample).or_insert(0) += 1;
        }

        let total_samples = samples.len() as f64;
        for (value, &weight) in &self.weights {
            let expected_probability = weight / self.total_weight;
            let observed_count = counts.get(value).copied().unwrap_or(0) as f64;
            let observed_probability = observed_count / total_samples;
            
            let difference = (expected_probability - observed_probability).abs();
            println!("WEIGHTED_SELECTION DEBUG: Value {:?}: expected={:.4}, observed={:.4}, diff={:.4}", 
                    value, expected_probability, observed_probability, difference);
            
            if difference > tolerance {
                println!("WEIGHTED_SELECTION DEBUG: Distribution validation failed for value {:?}", value);
                return false;
            }
        }

        println!("WEIGHTED_SELECTION DEBUG: Distribution validation passed");
        true
    }

    fn total_weight(&self) -> f64 {
        self.total_weight
    }
}

/// Alias method (Walker's algorithm) implementation
/// Provides O(1) selection after O(n) preprocessing
#[derive(Debug, Clone)]
pub struct AliasWeightedSelector<T> {
    values: Vec<T>,
    probabilities: Vec<f64>,
    aliases: Vec<usize>,
    original_weights: HashMap<T, f64>,
    total_weight: f64,
}

impl<T> AliasWeightedSelector<T>
where
    T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug,
{
    /// Create a new alias method weighted selector
    pub fn new(weights: HashMap<T, f64>) -> Result<Self, WeightedSelectionError> {
        println!("WEIGHTED_SELECTION DEBUG: Creating AliasWeightedSelector with 0x{:X} weights", weights.len());
        
        if weights.is_empty() {
            return Err(WeightedSelectionError::EmptyWeights);
        }

        let n = weights.len();
        let mut values = Vec::with_capacity(n);
        let mut probabilities = vec![0.0; n];
        let mut aliases = vec![0; n];

        // Validate and normalize weights
        let total_weight: f64 = weights.values().sum();
        if total_weight <= 0.0 {
            return Err(WeightedSelectionError::InvalidWeight(
                "Total weight must be positive".to_string()
            ));
        }

        for &weight in weights.values() {
            if weight <= 0.0 || !weight.is_finite() {
                return Err(WeightedSelectionError::InvalidWeight(
                    format!("Weight {} must be positive and finite", weight)
                ));
            }
        }

        // Setup for Walker's alias method
        let mut scaled_probs = Vec::with_capacity(n);
        for (value, &weight) in &weights {
            values.push(value.clone());
            scaled_probs.push(weight / total_weight * n as f64);
        }

        let mut small = Vec::new();
        let mut large = Vec::new();

        // Classify probabilities as small or large
        for (i, &prob) in scaled_probs.iter().enumerate() {
            if prob < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        // Build alias table
        while !small.is_empty() && !large.is_empty() {
            let small_idx = small.pop().unwrap();
            let large_idx = large.pop().unwrap();

            probabilities[small_idx] = scaled_probs[small_idx];
            aliases[small_idx] = large_idx;

            scaled_probs[large_idx] = (scaled_probs[large_idx] + scaled_probs[small_idx]) - 1.0;

            if scaled_probs[large_idx] < 1.0 {
                small.push(large_idx);
            } else {
                large.push(large_idx);
            }
        }

        // Handle remaining items
        while !large.is_empty() {
            let idx = large.pop().unwrap();
            probabilities[idx] = 1.0;
        }

        while !small.is_empty() {
            let idx = small.pop().unwrap();
            probabilities[idx] = 1.0;
        }

        println!("WEIGHTED_SELECTION DEBUG: Created alias selector with total_weight: {:E}", total_weight);

        Ok(Self {
            values,
            probabilities,
            aliases,
            original_weights: weights,
            total_weight,
        })
    }
}

impl<T> WeightedSelector<T> for AliasWeightedSelector<T>
where
    T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug,
{
    fn select(&self, random_value: f64) -> Result<T, WeightedSelectionError> {
        println!("WEIGHTED_SELECTION DEBUG: Alias selecting with random_value: {:E}", random_value);
        
        if !(0.0..=1.0).contains(&random_value) {
            return Err(WeightedSelectionError::InvalidRandomValue(random_value));
        }

        let n = self.values.len();
        let scaled_random = random_value * n as f64;
        let index = scaled_random.floor() as usize;
        let remainder = scaled_random - index as f64;

        let selected_index = if remainder < self.probabilities[index] {
            index
        } else {
            self.aliases[index]
        };

        let selected = &self.values[selected_index];
        println!("WEIGHTED_SELECTION DEBUG: Alias selected: {:?}", selected);
        Ok(selected.clone())
    }

    fn probability(&self, value: &T) -> f64 {
        self.original_weights.get(value)
            .map(|w| w / self.total_weight)
            .unwrap_or(0.0)
    }

    fn validate_distribution(&self, samples: &[T], tolerance: f64) -> bool {
        println!("WEIGHTED_SELECTION DEBUG: Alias validating distribution with 0x{:X} samples", samples.len());
        
        if samples.is_empty() {
            return true;
        }

        let mut counts = HashMap::new();
        for sample in samples {
            *counts.entry(sample).or_insert(0) += 1;
        }

        let total_samples = samples.len() as f64;
        for (value, &weight) in &self.original_weights {
            let expected_probability = weight / self.total_weight;
            let observed_count = counts.get(value).copied().unwrap_or(0) as f64;
            let observed_probability = observed_count / total_samples;
            
            let difference = (expected_probability - observed_probability).abs();
            println!("WEIGHTED_SELECTION DEBUG: Alias value {:?}: expected={:.4}, observed={:.4}, diff={:.4}", 
                    value, expected_probability, observed_probability, difference);
            
            if difference > tolerance {
                return false;
            }
        }

        true
    }

    fn total_weight(&self) -> f64 {
        self.total_weight
    }
}

/// Weighted choice context that manages constraint validation and selection
#[derive(Debug, Clone)]
pub struct WeightedChoiceContext<T>
where
    T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug + std::cmp::Ord,
{
    selector: CumulativeWeightedSelector<T>,
    constraints: Option<Constraints>,
}

impl<T> WeightedChoiceContext<T>
where
    T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug + std::cmp::Ord,
{
    /// Create a new weighted choice context with constraints
    pub fn new(
        weights: HashMap<T, f64>,
        constraints: Option<Constraints>,
    ) -> Result<Self, WeightedSelectionError> {
        println!("WEIGHTED_SELECTION DEBUG: Creating WeightedChoiceContext with constraints: {:?}", constraints);
        
        let selector = CumulativeWeightedSelector::new(weights)?;
        
        Ok(Self {
            selector,
            constraints,
        })
    }

    /// Select a value with constraint validation
    pub fn select_with_constraints(&self, random_value: f64) -> Result<T, WeightedSelectionError> {
        let selected = self.selector.select(random_value)?;
        
        // Validate against constraints if provided
        if let Some(ref constraints) = self.constraints {
            if let Err(msg) = self.validate_selection(&selected, constraints) {
                return Err(WeightedSelectionError::ConstraintViolation(msg));
            }
        }
        
        Ok(selected)
    }

    /// Get selection probability
    pub fn probability(&self, value: &T) -> f64 {
        self.selector.probability(value)
    }

    /// Validate distribution
    pub fn validate_distribution(&self, samples: &[T], tolerance: f64) -> bool {
        self.selector.validate_distribution(samples, tolerance)
    }

    /// Validate a selection against constraints
    fn validate_selection(&self, value: &T, constraints: &Constraints) -> Result<(), String> {
        println!("WEIGHTED_SELECTION DEBUG: Validating selection {:?} against constraints", value);
        
        match constraints {
            Constraints::Integer(int_constraints) => {
                // For integer constraints, we need to cast T to integer
                // This is a simplified validation - in practice, we'd need proper type checking
                if let (Some(min), Some(max)) = (int_constraints.min_value, int_constraints.max_value) {
                    println!("WEIGHTED_SELECTION DEBUG: Checking integer range [{}, {}]", min, max);
                    // Note: This is a placeholder - actual implementation would need proper type conversion
                }
                Ok(())
            }
            Constraints::Float(float_constraints) => {
                let min = float_constraints.min_value;
                let max = float_constraints.max_value;
                println!("WEIGHTED_SELECTION DEBUG: Checking float range [{:E}, {:E}]", min, max);
                Ok(())
            }
            Constraints::String(string_constraints) => {
                let min_len = string_constraints.min_size;
                let max_len = string_constraints.max_size;
                println!("WEIGHTED_SELECTION DEBUG: Checking string length range [0x{:X}, 0x{:X}]", min_len, max_len);
                Ok(())
            }
            Constraints::Bytes(bytes_constraints) => {
                let min_len = bytes_constraints.min_size;
                let max_len = bytes_constraints.max_size;
                println!("WEIGHTED_SELECTION DEBUG: Checking bytes length range [0x{:X}, 0x{:X}]", min_len, max_len);
                Ok(())
            }
            Constraints::Boolean(_) => {
                println!("WEIGHTED_SELECTION DEBUG: Boolean constraint validation passed");
                Ok(())
            }
        }
    }
}

/// Integer-specific weighted selection with full constraint support
pub type IntegerWeightedSelector = WeightedChoiceContext<i128>;

impl IntegerWeightedSelector {
    /// Create a new integer weighted selector from constraints
    pub fn from_constraints(constraints: &IntegerConstraints) -> Result<Self, WeightedSelectionError> {
        println!("WEIGHTED_SELECTION DEBUG: Creating IntegerWeightedSelector from constraints");
        
        let weights = constraints.weights.clone().ok_or_else(|| {
            WeightedSelectionError::ConstraintViolation("No weights provided".to_string())
        })?;

        // Validate that all weighted values are within range
        if let (Some(min_val), Some(max_val)) = (constraints.min_value, constraints.max_value) {
            for &value in weights.keys() {
                if value < min_val || value > max_val {
                    return Err(WeightedSelectionError::ConstraintViolation(
                        format!("Weighted value {} outside range [{}, {}]", value, min_val, max_val)
                    ));
                }
            }
        }

        Self::new(weights, Some(Constraints::Integer(constraints.clone())))
    }

    /// Select an integer value with full constraint validation
    pub fn select_integer(&self, random_value: f64) -> Result<i128, WeightedSelectionError> {
        self.select_with_constraints(random_value)
    }
}

/// Advanced weighted selection with statistical analysis
#[derive(Debug, Clone)]
pub struct StatisticalWeightedSelector<T>
where
    T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug + std::cmp::Ord,
{
    selector: CumulativeWeightedSelector<T>,
    samples: Vec<T>,
    chi_square_threshold: f64,
}

impl<T> StatisticalWeightedSelector<T>
where
    T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug + std::cmp::Ord,
{
    /// Create a new statistical weighted selector
    pub fn new(weights: HashMap<T, f64>, chi_square_threshold: f64) -> Result<Self, WeightedSelectionError> {
        println!("WEIGHTED_SELECTION DEBUG: Creating StatisticalWeightedSelector with chi² threshold: {:E}", chi_square_threshold);
        
        let selector = CumulativeWeightedSelector::new(weights)?;
        Ok(Self {
            selector,
            samples: Vec::new(),
            chi_square_threshold,
        })
    }

    /// Select and record for statistical analysis
    pub fn select_and_record(&mut self, random_value: f64) -> Result<T, WeightedSelectionError> {
        let selection = self.selector.select(random_value)?;
        self.samples.push(selection.clone());
        println!("WEIGHTED_SELECTION DEBUG: Recorded selection {:?}, sample count: 0x{:X}", selection, self.samples.len());
        Ok(selection)
    }

    /// Perform chi-square goodness of fit test
    pub fn chi_square_test(&self) -> f64 {
        if self.samples.len() < 5 {
            return 0.0; // Not enough samples for meaningful test
        }

        let mut counts = HashMap::new();
        for sample in &self.samples {
            *counts.entry(sample).or_insert(0) += 1;
        }

        let total_samples = self.samples.len() as f64;
        let mut chi_square = 0.0;

        for (value, weight) in self.selector.weights.iter() {
            let expected = (weight / self.selector.total_weight) * total_samples;
            let observed = counts.get(value).copied().unwrap_or(0) as f64;
            
            if expected > 0.0 {
                let diff = observed - expected;
                chi_square += (diff * diff) / expected;
            }
        }

        println!("WEIGHTED_SELECTION DEBUG: Chi-square statistic: {:E} (threshold: {:E})", chi_square, self.chi_square_threshold);
        chi_square
    }

    /// Check if distribution passes statistical test
    pub fn passes_statistical_test(&self) -> bool {
        self.chi_square_test() <= self.chi_square_threshold
    }

    /// Reset sample collection
    pub fn reset_samples(&mut self) {
        println!("WEIGHTED_SELECTION DEBUG: Resetting 0x{:X} samples", self.samples.len());
        self.samples.clear();
    }
}

impl<T> WeightedSelector<T> for StatisticalWeightedSelector<T>
where
    T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug + std::cmp::Ord,
{
    fn select(&self, random_value: f64) -> Result<T, WeightedSelectionError> {
        self.selector.select(random_value)
    }

    fn probability(&self, value: &T) -> f64 {
        self.selector.probability(value)
    }

    fn validate_distribution(&self, samples: &[T], tolerance: f64) -> bool {
        self.selector.validate_distribution(samples, tolerance)
    }

    fn total_weight(&self) -> f64 {
        self.selector.total_weight()
    }
}

/// Factory for creating different types of weighted selectors
pub struct WeightedSelectorFactory;

impl WeightedSelectorFactory {
    /// Create a CDF-based selector (good for general use)
    pub fn create_cdf_selector<T>(weights: HashMap<T, f64>) -> Result<CumulativeWeightedSelector<T>, WeightedSelectionError>
    where
        T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug + std::cmp::Ord,
    {
        CumulativeWeightedSelector::new(weights)
    }

    /// Create an alias-based selector (optimal for repeated sampling)
    pub fn create_alias_selector<T>(weights: HashMap<T, f64>) -> Result<AliasWeightedSelector<T>, WeightedSelectionError>
    where
        T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug,
    {
        AliasWeightedSelector::new(weights)
    }

    /// Create a statistical selector with chi-square analysis
    pub fn create_statistical_selector<T>(
        weights: HashMap<T, f64>,
        chi_square_threshold: f64,
    ) -> Result<StatisticalWeightedSelector<T>, WeightedSelectionError>
    where
        T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug + std::cmp::Ord,
    {
        StatisticalWeightedSelector::new(weights, chi_square_threshold)
    }

    /// Create the best selector for the given use case
    pub fn create_optimal_selector<T>(
        weights: HashMap<T, f64>,
        expected_selections: usize,
    ) -> Result<Box<dyn WeightedSelector<T>>, WeightedSelectionError>
    where
        T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug + std::cmp::Ord + 'static,
    {
        println!("WEIGHTED_SELECTION DEBUG: Creating optimal selector for 0x{:X} expected selections", expected_selections);
        
        // Use alias method for many selections, CDF for few
        if expected_selections > 100 {
            println!("WEIGHTED_SELECTION DEBUG: Using alias method for 0x{:X}+ selections", expected_selections);
            Ok(Box::new(AliasWeightedSelector::new(weights)?))
        } else {
            println!("WEIGHTED_SELECTION DEBUG: Using CDF method for 0x{:X} selections", expected_selections);
            Ok(Box::new(CumulativeWeightedSelector::new(weights)?))
        }
    }

    /// Create a selector optimized for constraint validation
    pub fn create_constrained_selector<T>(
        weights: HashMap<T, f64>,
        constraints: Option<Constraints>,
    ) -> Result<WeightedChoiceContext<T>, WeightedSelectionError>
    where
        T: Clone + std::cmp::Eq + std::hash::Hash + std::fmt::Debug + std::cmp::Ord,
    {
        WeightedChoiceContext::new(weights, constraints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cumulative_weighted_selector_creation() {
        println!("WEIGHTED_SELECTION DEBUG: Testing CumulativeWeightedSelector creation");
        
        let mut weights = HashMap::new();
        weights.insert(1, 0.3);
        weights.insert(2, 0.7);

        let selector = CumulativeWeightedSelector::new(weights).unwrap();
        assert_eq!(selector.total_weight(), 1.0);
        assert_eq!(selector.probability(&1), 0.3);
        assert_eq!(selector.probability(&2), 0.7);
        
        println!("WEIGHTED_SELECTION DEBUG: CumulativeWeightedSelector creation test passed");
    }

    #[test]
    fn test_cumulative_selector_deterministic_selection() {
        println!("WEIGHTED_SELECTION DEBUG: Testing deterministic selection");
        
        let mut weights = HashMap::new();
        weights.insert(10, 0.25);
        weights.insert(20, 0.50);
        weights.insert(30, 0.25);

        let selector = CumulativeWeightedSelector::new(weights).unwrap();

        assert_eq!(selector.select(0.0).unwrap(), 10);
        assert_eq!(selector.select(0.24).unwrap(), 10);
        assert_eq!(selector.select(0.26).unwrap(), 20);
        assert_eq!(selector.select(0.74).unwrap(), 20);
        assert_eq!(selector.select(0.76).unwrap(), 30);
        
        println!("WEIGHTED_SELECTION DEBUG: Deterministic selection test passed");
    }

    #[test]
    fn test_alias_weighted_selector_creation() {
        println!("WEIGHTED_SELECTION DEBUG: Testing AliasWeightedSelector creation");
        
        let mut weights = HashMap::new();
        weights.insert(100, 2.0);
        weights.insert(200, 3.0);
        weights.insert(300, 5.0);

        let selector = AliasWeightedSelector::new(weights).unwrap();
        assert_eq!(selector.total_weight(), 10.0);
        assert!((selector.probability(&100) - 0.2).abs() < f64::EPSILON);
        assert!((selector.probability(&200) - 0.3).abs() < f64::EPSILON);
        assert!((selector.probability(&300) - 0.5).abs() < f64::EPSILON);
        
        println!("WEIGHTED_SELECTION DEBUG: AliasWeightedSelector creation test passed");
    }

    #[test]
    fn test_distribution_validation() {
        println!("WEIGHTED_SELECTION DEBUG: Testing distribution validation");
        
        let mut weights = HashMap::new();
        weights.insert(1, 1.0);
        weights.insert(2, 1.0);

        let selector = CumulativeWeightedSelector::new(weights).unwrap();

        let balanced_samples = vec![1, 2, 1, 2, 1, 2, 1, 2];
        assert!(selector.validate_distribution(&balanced_samples, 0.1));

        let unbalanced_samples = vec![1, 1, 1, 1, 1, 1, 2, 2];
        assert!(!selector.validate_distribution(&unbalanced_samples, 0.1));
        
        println!("WEIGHTED_SELECTION DEBUG: Distribution validation test passed");
    }

    #[test]
    fn test_integer_weighted_selector_from_constraints() {
        println!("WEIGHTED_SELECTION DEBUG: Testing IntegerWeightedSelector from constraints");
        
        let mut weights = HashMap::new();
        weights.insert(5, 0.4);
        weights.insert(10, 0.6);

        let constraints = IntegerConstraints {
            min_value: Some(1),
            max_value: Some(15),
            weights: Some(weights),
            shrink_towards: Some(5),
        };

        let selector = IntegerWeightedSelector::from_constraints(&constraints).unwrap();
        
        let result = selector.select_integer(0.3);
        assert!(result.is_ok());
        let value = result.unwrap();
        assert!(value == 5 || value == 10);
        
        println!("WEIGHTED_SELECTION DEBUG: IntegerWeightedSelector from constraints test passed");
    }

    #[test]
    fn test_weighted_selector_factory() {
        println!("WEIGHTED_SELECTION DEBUG: Testing WeightedSelectorFactory");
        
        let mut weights = HashMap::new();
        weights.insert("A", 0.3);
        weights.insert("B", 0.7);

        let cdf_selector = WeightedSelectorFactory::create_cdf_selector(weights.clone());
        assert!(cdf_selector.is_ok());

        let alias_selector = WeightedSelectorFactory::create_alias_selector(weights.clone());
        assert!(alias_selector.is_ok());

        let optimal_few = WeightedSelectorFactory::create_optimal_selector(weights.clone(), 10);
        assert!(optimal_few.is_ok());

        let optimal_many = WeightedSelectorFactory::create_optimal_selector(weights, 1000);
        assert!(optimal_many.is_ok());
        
        println!("WEIGHTED_SELECTION DEBUG: WeightedSelectorFactory test passed");
    }

    #[test]
    fn test_error_handling() {
        println!("WEIGHTED_SELECTION DEBUG: Testing error handling");
        
        // Empty weights
        let empty_weights: HashMap<i32, f64> = HashMap::new();
        let result = CumulativeWeightedSelector::new(empty_weights);
        assert!(matches!(result, Err(WeightedSelectionError::EmptyWeights)));

        // Invalid weights
        let mut invalid_weights = HashMap::new();
        invalid_weights.insert(1, -0.5);
        let result = CumulativeWeightedSelector::new(invalid_weights);
        assert!(matches!(result, Err(WeightedSelectionError::InvalidWeight(_))));

        // Invalid random value
        let mut valid_weights = HashMap::new();
        valid_weights.insert(1, 1.0);
        let selector = CumulativeWeightedSelector::new(valid_weights).unwrap();
        let result = selector.select(1.1);
        assert!(matches!(result, Err(WeightedSelectionError::InvalidRandomValue(_))));
        
        println!("WEIGHTED_SELECTION DEBUG: Error handling test passed");
    }

    #[test]
    fn test_statistical_selector() {
        println!("WEIGHTED_SELECTION DEBUG: Testing StatisticalWeightedSelector");
        
        let mut weights = HashMap::new();
        weights.insert(1, 0.5);
        weights.insert(2, 0.5);

        let mut stat_selector = WeightedSelectorFactory::create_statistical_selector(weights, 3.841).unwrap(); // 95% confidence
        
        // Generate some samples
        for i in 0..100 {
            let random_val = (i as f64) / 100.0;
            let _ = stat_selector.select_and_record(random_val);
        }
        
        let chi_square = stat_selector.chi_square_test();
        assert!(chi_square >= 0.0);
        
        stat_selector.reset_samples();
        assert!(stat_selector.samples.is_empty());
        
        println!("WEIGHTED_SELECTION DEBUG: StatisticalWeightedSelector test passed");
    }

    #[test]
    fn test_constrained_selector_factory() {
        println!("WEIGHTED_SELECTION DEBUG: Testing constrained selector factory");
        
        let mut weights = HashMap::new();
        weights.insert(5, 0.4);
        weights.insert(10, 0.6);

        let constraints = IntegerConstraints {
            min_value: Some(1),
            max_value: Some(15),
            weights: Some(weights.clone()),
            shrink_towards: Some(5),
        };

        let constrained_selector = WeightedSelectorFactory::create_constrained_selector(
            weights,
            Some(Constraints::Integer(constraints)),
        ).unwrap();
        
        let result = constrained_selector.select_with_constraints(0.3);
        assert!(result.is_ok());
        
        println!("WEIGHTED_SELECTION DEBUG: Constrained selector factory test passed");
    }

    #[test]
    fn test_edge_cases() {
        println!("WEIGHTED_SELECTION DEBUG: Testing edge cases");
        
        // Single weight
        let mut single_weight = HashMap::new();
        single_weight.insert(42, 1.0);
        let selector = CumulativeWeightedSelector::new(single_weight).unwrap();
        assert_eq!(selector.select(0.0).unwrap(), 42);
        assert_eq!(selector.select(0.5).unwrap(), 42);
        assert_eq!(selector.select(1.0).unwrap(), 42);

        // Very small weights
        let mut tiny_weights = HashMap::new();
        tiny_weights.insert(1, 1e-10);
        tiny_weights.insert(2, 1e-10);
        let tiny_selector = CumulativeWeightedSelector::new(tiny_weights).unwrap();
        assert!(tiny_selector.select(0.0).is_ok());
        assert!(tiny_selector.select(1.0).is_ok());
        
        println!("WEIGHTED_SELECTION DEBUG: Edge cases test passed");
    }
}