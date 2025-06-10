//! Choice-aware shrinking implementation
//! 
//! This module implements modern shrinking algorithms that leverage choice metadata
//! to produce high-quality minimal examples. Unlike byte-stream shrinking, choice-aware
//! shrinking understands the semantic structure of the data being shrunk.

use crate::choice::{ChoiceNode, ChoiceValue, Constraints, ChoiceType};
use crate::data::{ConjectureData, ConjectureResult, Status, DrawError};
use std::collections::HashMap;

/// A shrinking transformation that can be applied to a choice sequence
#[derive(Debug)]
pub struct ShrinkingTransformation {
    /// Human-readable description of this transformation
    pub description: String,
    
    /// The transformation function that modifies a choice sequence
    pub transform: fn(&[ChoiceNode]) -> Vec<ChoiceNode>,
}

impl Clone for ShrinkingTransformation {
    fn clone(&self) -> Self {
        Self {
            description: self.description.clone(),
            transform: self.transform,
        }
    }
}

/// Core shrinker that applies transformations to minimize test cases
#[derive(Debug)]
pub struct ChoiceShrinker {
    /// Original failing test result
    pub original_result: ConjectureResult,
    
    /// Best (smallest) result found so far
    pub best_result: ConjectureResult,
    
    /// Number of shrinking attempts made
    pub attempts: u32,
    
    /// Maximum number of shrinking attempts allowed
    pub max_attempts: u32,
    
    /// Available shrinking transformations
    pub transformations: Vec<ShrinkingTransformation>,
}

impl ChoiceShrinker {
    /// Create a new shrinker for the given failing test result
    pub fn new(original_result: ConjectureResult) -> Self {
        Self {
            best_result: original_result.clone(),
            original_result,
            attempts: 0,
            max_attempts: 10000, // Match Python's default
            transformations: Self::default_transformations(),
        }
    }
    
    /// Get the default set of shrinking transformations
    fn default_transformations() -> Vec<ShrinkingTransformation> {
        vec![
            ShrinkingTransformation {
                description: "minimize_integer_values".to_string(),
                transform: minimize_integer_values,
            },
            ShrinkingTransformation {
                description: "minimize_to_false".to_string(),
                transform: minimize_booleans_to_false,
            },
            ShrinkingTransformation {
                description: "delete_choices".to_string(),
                transform: delete_trailing_choices,
            },
        ]
    }
    
    /// Apply all available transformations to find minimal example
    pub fn shrink<F>(&mut self, test_function: F) -> ConjectureResult 
    where
        F: Fn(&ConjectureResult) -> bool,
    {
        println!("SHRINKING DEBUG: Starting shrink with {} choices", self.original_result.choices.len());
        
        // Try each transformation until no more improvements
        let mut improved = true;
        while improved && self.attempts < self.max_attempts {
            improved = false;
            
            // Clone transformations to avoid borrow conflicts
            let transformations = self.transformations.clone();
            for transformation in &transformations {
                if self.attempts >= self.max_attempts {
                    break;
                }
                
                if self.apply_transformation(transformation, &test_function) {
                    improved = true;
                    println!("SHRINKING DEBUG: {} succeeded, new best has {} choices", 
                             transformation.description, self.best_result.choices.len());
                }
            }
        }
        
        println!("SHRINKING DEBUG: Shrinking complete after {} attempts", self.attempts);
        println!("SHRINKING DEBUG: Original: {} choices, Final: {} choices", 
                 self.original_result.choices.len(), self.best_result.choices.len());
        
        self.best_result.clone()
    }
    
    /// Apply a specific transformation and test if it improves the result
    fn apply_transformation<F>(&mut self, transformation: &ShrinkingTransformation, test_function: F) -> bool 
    where
        F: Fn(&ConjectureResult) -> bool,
    {
        self.attempts += 1;
        
        println!("SHRINKING DEBUG: Applying transformation: {}", transformation.description);
        
        // Apply transformation to current best choices
        let transformed_choices = (transformation.transform)(&self.best_result.choices);
        
        println!("SHRINKING DEBUG: Original choices: {}", self.best_result.choices.len());
        println!("SHRINKING DEBUG: Transformed choices: {}", transformed_choices.len());
        
        // Print choice values for debugging
        for (i, (orig, trans)) in self.best_result.choices.iter().zip(transformed_choices.iter()).enumerate() {
            println!("SHRINKING DEBUG: Choice {}: {:?} -> {:?}", i, orig.value, trans.value);
        }
        
        // Skip if transformation didn't change anything
        if transformed_choices == self.best_result.choices {
            println!("SHRINKING DEBUG: No changes detected, skipping");
            return false;
        }
        
        // Create new result with transformed choices
        let length = self.calculate_length(&transformed_choices);
        let test_result = ConjectureResult {
            status: Status::Valid,
            choices: transformed_choices,
            length,
            events: HashMap::new(),
            buffer: Vec::new(),
            examples: Vec::new(),
        };
        
        println!("SHRINKING DEBUG: Testing if transformation still fails");
        
        // Test if the transformation still produces a failing test
        if test_function(&test_result) {
            println!("SHRINKING DEBUG: Transformation still fails - checking if better");
            // This is a valid shrinking - smaller and still fails
            if self.is_better(&test_result, &self.best_result) {
                println!("SHRINKING DEBUG: Found better result, updating best");
                self.best_result = test_result;
                return true;
            } else {
                println!("SHRINKING DEBUG: Not better than current best");
            }
        } else {
            println!("SHRINKING DEBUG: Transformation passes test - not a valid shrinking");
        }
        
        false
    }
    
    /// Check if one result is better (smaller) than another
    fn is_better(&self, candidate: &ConjectureResult, current: &ConjectureResult) -> bool {
        println!("SHRINKING DEBUG: Comparing candidates:");
        println!("SHRINKING DEBUG: Candidate: {} choices, length {}", candidate.choices.len(), candidate.length);
        println!("SHRINKING DEBUG: Current: {} choices, length {}", current.choices.len(), current.length);
        
        // Primarily prefer fewer choices
        if candidate.choices.len() != current.choices.len() {
            let result = candidate.choices.len() < current.choices.len();
            println!("SHRINKING DEBUG: Different choice counts -> {}", result);
            return result;
        }
        
        // Secondary: prefer smaller total length
        if candidate.length != current.length {
            let result = candidate.length < current.length;
            println!("SHRINKING DEBUG: Different lengths -> {}", result);
            return result;
        }
        
        // Tertiary: prefer lexicographically smaller choice values
        // This handles value minimization within same structure
        for (cand_choice, curr_choice) in candidate.choices.iter().zip(current.choices.iter()) {
            let comparison = self.compare_choice_values(&cand_choice.value, &curr_choice.value);
            if comparison != std::cmp::Ordering::Equal {
                let result = comparison == std::cmp::Ordering::Less;
                println!("SHRINKING DEBUG: Value comparison: {:?} vs {:?} -> {}", 
                         cand_choice.value, curr_choice.value, result);
                return result;
            }
        }
        
        println!("SHRINKING DEBUG: No difference found -> false");
        false
    }
    
    /// Compare two choice values for shrinking purposes
    fn compare_choice_values(&self, a: &ChoiceValue, b: &ChoiceValue) -> std::cmp::Ordering {
        match (a, b) {
            (ChoiceValue::Integer(a), ChoiceValue::Integer(b)) => {
                // Prefer smaller absolute values (closer to 0 is "smaller" for shrinking)
                let abs_a = a.abs();
                let abs_b = b.abs();
                if abs_a != abs_b {
                    abs_a.cmp(&abs_b)
                } else {
                    // If absolute values are equal, prefer positive over negative
                    a.cmp(b)
                }
            },
            (ChoiceValue::Boolean(a), ChoiceValue::Boolean(b)) => {
                // false < true for shrinking purposes
                a.cmp(b)
            },
            (ChoiceValue::Float(a), ChoiceValue::Float(b)) => {
                // Prefer smaller absolute values, handle NaN specially
                if a.is_nan() && b.is_nan() {
                    std::cmp::Ordering::Equal
                } else if a.is_nan() {
                    std::cmp::Ordering::Greater // NaN is "larger" for shrinking
                } else if b.is_nan() {
                    std::cmp::Ordering::Less
                } else {
                    let abs_a = a.abs();
                    let abs_b = b.abs();
                    abs_a.partial_cmp(&abs_b).unwrap_or(std::cmp::Ordering::Equal)
                }
            },
            (ChoiceValue::String(a), ChoiceValue::String(b)) => {
                // Prefer shorter strings, then lexicographic order
                let len_cmp = a.len().cmp(&b.len());
                if len_cmp != std::cmp::Ordering::Equal {
                    len_cmp
                } else {
                    a.cmp(b)
                }
            },
            (ChoiceValue::Bytes(a), ChoiceValue::Bytes(b)) => {
                // Prefer shorter byte arrays, then lexicographic order
                let len_cmp = a.len().cmp(&b.len());
                if len_cmp != std::cmp::Ordering::Equal {
                    len_cmp
                } else {
                    a.cmp(b)
                }
            },
            _ => std::cmp::Ordering::Equal, // Different types, consider equal
        }
    }
    
    /// Calculate total length for a choice sequence
    fn calculate_length(&self, choices: &[ChoiceNode]) -> usize {
        choices.iter().map(|choice| match choice.choice_type {
            ChoiceType::Integer => 2,
            ChoiceType::Boolean => 1,
            ChoiceType::Float => 8,
            ChoiceType::String => {
                if let ChoiceValue::String(s) = &choice.value {
                    s.len()
                } else {
                    0
                }
            },
            ChoiceType::Bytes => {
                if let ChoiceValue::Bytes(b) = &choice.value {
                    b.len()
                } else {
                    0
                }
            },
        }).sum()
    }
}

/// Minimize integer values towards their shrink_towards target
/// Use a more aggressive approach that tries to make larger jumps
fn minimize_integer_values(choices: &[ChoiceNode]) -> Vec<ChoiceNode> {
    choices.iter().map(|choice| {
        // Only modify non-forced choices
        if choice.was_forced {
            return choice.clone();
        }
        
        if let (ChoiceType::Integer, ChoiceValue::Integer(value), Constraints::Integer(constraints)) = 
            (&choice.choice_type, &choice.value, &choice.constraints) {
            
            let min_val = constraints.min_value.unwrap_or(i128::MIN);
            let max_val = constraints.max_value.unwrap_or(i128::MAX);
            let shrink_target = constraints.shrink_towards.unwrap_or(0).max(min_val).min(max_val);
            
            // If already at target, no change needed
            if *value == shrink_target {
                return choice.clone();
            }
            
            // Calculate the distance to the target
            let distance = (*value - shrink_target).abs();
            
            // Try to make a larger jump towards the target
            // Start with half the distance, then fall back to smaller steps
            let reduction_candidates = [
                distance,         // Jump all the way to target if possible
                distance / 2,     // Half the distance  
                distance / 4,     // Quarter distance
                distance / 8,     // Eighth distance
                (distance / 10).max(1), // 10% or at least 1
                1,                // Single step
            ];
            
            for &reduction in &reduction_candidates {
                let new_value = if *value > shrink_target {
                    (*value - reduction).max(shrink_target)
                } else {
                    (*value + reduction).min(shrink_target)
                };
                
                // Check if the new value is within bounds and closer to target
                if new_value >= min_val && new_value <= max_val && new_value != *value {
                    let mut new_choice = choice.clone();
                    new_choice.value = ChoiceValue::Integer(new_value);
                    return new_choice;
                }
            }
        }
        
        choice.clone()
    }).collect()
}

/// Minimize boolean values to false where possible
fn minimize_booleans_to_false(choices: &[ChoiceNode]) -> Vec<ChoiceNode> {
    choices.iter().map(|choice| {
        // Only modify non-forced boolean choices that are true
        if let (ChoiceType::Boolean, ChoiceValue::Boolean(true)) = (&choice.choice_type, &choice.value) {
            if !choice.was_forced {
                let mut new_choice = choice.clone();
                new_choice.value = ChoiceValue::Boolean(false);
                return new_choice;
            }
        }
        choice.clone()
    }).collect()
}

/// Delete trailing choices to reduce sequence length
fn delete_trailing_choices(choices: &[ChoiceNode]) -> Vec<ChoiceNode> {
    if choices.is_empty() {
        return choices.to_vec();
    }
    
    // Remove the last choice
    choices[..choices.len() - 1].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::{IntegerConstraints, BooleanConstraints};

    #[test]
    fn test_minimize_integer_values() {
        let choices = vec![
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(50),
                Constraints::Integer(IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(100),
                    weights: None,
                    shrink_towards: Some(0),
                }),
                false,
            ),
        ];
        
        let minimized = minimize_integer_values(&choices);
        
        if let ChoiceValue::Integer(value) = &minimized[0].value {
            assert_eq!(*value, 49); // Should shrink towards 0
        } else {
            panic!("Expected integer value");
        }
    }
    
    #[test]
    fn test_minimize_booleans_to_false() {
        let choices = vec![
            ChoiceNode::new(
                ChoiceType::Boolean,
                ChoiceValue::Boolean(true),
                Constraints::Boolean(BooleanConstraints { p: 0.5 }),
                false,
            ),
        ];
        
        let minimized = minimize_booleans_to_false(&choices);
        
        if let ChoiceValue::Boolean(value) = &minimized[0].value {
            assert_eq!(*value, false); // Should shrink to false
        } else {
            panic!("Expected boolean value");
        }
    }
    
    #[test]
    fn test_delete_trailing_choices() {
        let choices = vec![
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(1),
                Constraints::Integer(IntegerConstraints::default()),
                false,
            ),
            ChoiceNode::new(
                ChoiceType::Boolean,
                ChoiceValue::Boolean(true),
                Constraints::Boolean(BooleanConstraints { p: 0.5 }),
                false,
            ),
        ];
        
        let result = delete_trailing_choices(&choices);
        assert_eq!(result.len(), 1);
        
        if let ChoiceValue::Integer(value) = &result[0].value {
            assert_eq!(*value, 1);
        } else {
            panic!("Expected first choice to remain");
        }
    }
}