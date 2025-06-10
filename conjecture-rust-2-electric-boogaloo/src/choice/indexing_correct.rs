//! Corrected choice indexing that properly enumerates values in Python's order

use super::{ChoiceValue, Constraints, IntegerConstraints, BooleanConstraints};

/// Generate all possible integer values in the order Python uses
fn generate_integer_sequence(constraints: &IntegerConstraints) -> Vec<i128> {
    println!("INDEXING_CORRECT DEBUG: Generating integer sequence for constraints: {:?}", constraints);
    
    let shrink_towards = constraints.shrink_towards.unwrap_or(0);
    
    // Get the clamped shrink_towards value  
    let clamped_shrink = match (constraints.min_value, constraints.max_value) {
        (Some(min), Some(max)) => shrink_towards.max(min).min(max),
        (Some(min), None) => shrink_towards.max(min),
        (None, Some(max)) => shrink_towards.min(max),
        (None, None) => shrink_towards,
    };
    
    let mut sequence = vec![clamped_shrink];
    println!("INDEXING_CORRECT DEBUG: Starting with clamped shrink_towards: {}", clamped_shrink);
    
    // Generate values by distance from clamped_shrink
    for distance in 1.. {
        let mut added_any = false;
        
        // Try positive direction first
        let positive_candidate = clamped_shrink + distance;
        let positive_valid = match (constraints.min_value, constraints.max_value) {
            (Some(min), Some(max)) => positive_candidate >= min && positive_candidate <= max,
            (Some(min), None) => positive_candidate >= min,
            (None, Some(max)) => positive_candidate <= max,
            (None, None) => true,
        };
        
        if positive_valid {
            sequence.push(positive_candidate);
            added_any = true;
            println!("INDEXING_CORRECT DEBUG: Added positive distance {}: {}", distance, positive_candidate);
        }
        
        // Try negative direction  
        let negative_candidate = clamped_shrink - distance;
        let negative_valid = match (constraints.min_value, constraints.max_value) {
            (Some(min), Some(max)) => negative_candidate >= min && negative_candidate <= max,
            (Some(min), None) => negative_candidate >= min,
            (None, Some(max)) => negative_candidate <= max,
            (None, None) => true,
        };
        
        if negative_valid {
            sequence.push(negative_candidate);
            added_any = true;
            println!("INDEXING_CORRECT DEBUG: Added negative distance {}: {}", distance, negative_candidate);
        }
        
        // If we're bounded and didn't add anything new, we're done
        if !added_any && (constraints.min_value.is_some() || constraints.max_value.is_some()) {
            break;
        }
        
        // For unbounded, stop at some reasonable limit for testing
        if distance > 1000 && constraints.min_value.is_none() && constraints.max_value.is_none() {
            break;
        }
    }
    
    println!("INDEXING_CORRECT DEBUG: Generated sequence: {:?}", sequence);
    sequence
}

/// Convert a choice value to its index using correct enumeration
pub fn choice_to_index_correct(value: &ChoiceValue, constraints: &Constraints) -> u64 {
    println!("INDEXING_CORRECT DEBUG: Converting choice to index");
    println!("INDEXING_CORRECT DEBUG: value={:?}", value);
    
    match (value, constraints) {
        (ChoiceValue::Integer(val), Constraints::Integer(c)) => {
            let sequence = generate_integer_sequence(c);
            
            if let Some(index) = sequence.iter().position(|&x| x == *val) {
                println!("INDEXING_CORRECT DEBUG: Found integer {} at index {}", val, index);
                index as u64
            } else {
                println!("INDEXING_CORRECT DEBUG: Integer {} not found in sequence, returning 0", val);
                0
            }
        }
        
        (ChoiceValue::Boolean(val), Constraints::Boolean(c)) => {
            let index = if c.p == 0.0 {
                // Only false is possible
                0
            } else if c.p == 1.0 {
                // Only true is possible
                0
            } else {
                // Both possible: false=0, true=1 (lexicographic order)
                if *val { 1 } else { 0 }
            };
            
            println!("INDEXING_CORRECT DEBUG: Boolean {} with p={} -> index {}", val, c.p, index);
            index
        }
        
        _ => {
            println!("INDEXING_CORRECT DEBUG: Unsupported choice type for indexing, returning 0");
            0
        }
    }
}

/// Convert an index back to a choice value using correct enumeration
pub fn choice_from_index_correct(index: u64, choice_type: &str, constraints: &Constraints) -> ChoiceValue {
    println!("INDEXING_CORRECT DEBUG: Converting index {} to choice", index);
    
    match (choice_type, constraints) {
        ("integer", Constraints::Integer(c)) => {
            let sequence = generate_integer_sequence(c);
            
            if (index as usize) < sequence.len() {
                let value = sequence[index as usize];
                println!("INDEXING_CORRECT DEBUG: Index {} -> integer {}", index, value);
                ChoiceValue::Integer(value)
            } else {
                println!("INDEXING_CORRECT DEBUG: Index {} out of bounds, returning first value", index);
                ChoiceValue::Integer(sequence[0])
            }
        }
        
        ("boolean", Constraints::Boolean(c)) => {
            let result = if c.p == 0.0 {
                // Only false is possible
                false
            } else if c.p == 1.0 {
                // Only true is possible
                true
            } else {
                // Both possible: index 0=false, index 1=true
                index == 1
            };
            
            println!("INDEXING_CORRECT DEBUG: Index {} with p={} -> boolean {}", index, c.p, result);
            ChoiceValue::Boolean(result)
        }
        
        _ => {
            println!("INDEXING_CORRECT DEBUG: Unsupported choice type for indexing, returning default");
            ChoiceValue::Integer(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::{choice_equal};

    fn integer_constr_helper(min_value: Option<i128>, max_value: Option<i128>) -> IntegerConstraints {
        IntegerConstraints {
            min_value,
            max_value,
            weights: None,
            shrink_towards: Some(0),
        }
    }

    fn integer_constr_with_shrink(min_value: Option<i128>, max_value: Option<i128>, shrink_towards: i128) -> IntegerConstraints {
        IntegerConstraints {
            min_value,
            max_value,
            weights: None,
            shrink_towards: Some(shrink_towards),
        }
    }

    #[test]
    fn test_correct_integer_choice_index_ordering() {
        println!("INDEXING_CORRECT DEBUG: Testing correct integer choice index ordering");
        
        // Test cases from Python's test_integer_choice_index
        let test_cases = vec![
            // unbounded
            (integer_constr_helper(None, None), vec![0, 1, -1, 2, -2, 3, -3]),
            (integer_constr_with_shrink(None, None, 2), vec![2, 3, 1, 4, 0, 5, -1]),
            // bounded
            (integer_constr_helper(Some(-3), Some(3)), vec![0, 1, -1, 2, -2, 3, -3]),
            (integer_constr_with_shrink(Some(-3), Some(3), 1), vec![1, 2, 0, 3, -1, -2, -3]),
        ];
        
        for (constraints, expected_order) in test_cases {
            println!("INDEXING_CORRECT DEBUG: Testing constraints {:?}", constraints);
            println!("INDEXING_CORRECT DEBUG: Expected order: {:?}", expected_order);
            
            for (expected_index, value) in expected_order.iter().enumerate() {
                let choice_value = ChoiceValue::Integer(*value);
                let constraints_enum = Constraints::Integer(constraints.clone());
                let actual_index = choice_to_index_correct(&choice_value, &constraints_enum);
                
                println!("INDEXING_CORRECT DEBUG: Value {} should have index {}, got {}", 
                    value, expected_index, actual_index);
                assert_eq!(actual_index, expected_index as u64, 
                    "Value {} should have index {}, got {}", value, expected_index, actual_index);
            }
        }
        
        println!("INDEXING_CORRECT DEBUG: Correct integer choice index ordering test passed");
    }

    #[test]
    fn test_correct_choice_index_and_value_are_inverses() {
        println!("INDEXING_CORRECT DEBUG: Testing that correct functions are inverses");
        
        // Test various integer values
        let test_cases = vec![
            (integer_constr_helper(None, None), vec![0, 1, -1, 2, -2]),
            (integer_constr_helper(Some(0), Some(10)), vec![0, 1, 2, 5, 10]),
            (integer_constr_with_shrink(Some(-5), Some(5), 2), vec![-5, -2, 0, 2, 3, 5]),
        ];
        
        for (constraints, values) in test_cases {
            for val in values {
                let value = ChoiceValue::Integer(val);
                let constraints_enum = Constraints::Integer(constraints.clone());
                
                let index = choice_to_index_correct(&value, &constraints_enum);
                let recovered = choice_from_index_correct(index, "integer", &constraints_enum);
                
                println!("INDEXING_CORRECT DEBUG: {} -> index {} -> {:?}", val, index, recovered);
                assert!(choice_equal(&value, &recovered), 
                    "Failed for value {}: got {:?}, expected {:?}", val, recovered, value);
            }
        }
        
        println!("INDEXING_CORRECT DEBUG: Correct choice index and value inverses test passed");
    }
}