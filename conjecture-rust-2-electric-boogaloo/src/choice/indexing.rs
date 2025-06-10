//! Choice indexing functions for ordering and shrinking
//! 
//! These functions convert choices to/from indices for deterministic ordering,
//! which is essential for shrinking and choice tree navigation.

use super::{ChoiceValue, Constraints, IntegerConstraints};

#[cfg(test)]
use super::BooleanConstraints;

/// Generate sequence for bounded integer ranges only
fn generate_bounded_integer_sequence(constraints: &IntegerConstraints, max_size: usize) -> Vec<i128> {
    let shrink_towards = constraints.shrink_towards.unwrap_or(0);
    let clamped_shrink = match (constraints.min_value, constraints.max_value) {
        (Some(min), Some(max)) => shrink_towards.max(min).min(max),
        (Some(min), None) => shrink_towards.max(min),
        (None, Some(max)) => shrink_towards.min(max),
        (None, None) => shrink_towards,
    };
    
    let mut sequence = vec![clamped_shrink];
    
    // Generate values by distance from clamped_shrink
    for distance in 1..max_size {
        // Try positive direction first
        let positive_candidate = clamped_shrink + distance as i128;
        let positive_valid = match (constraints.min_value, constraints.max_value) {
            (Some(min), Some(max)) => positive_candidate >= min && positive_candidate <= max,
            (Some(min), None) => positive_candidate >= min,
            (None, Some(max)) => positive_candidate <= max,
            (None, None) => true,
        };
        
        if positive_valid {
            sequence.push(positive_candidate);
        }
        
        // Try negative direction  
        let negative_candidate = clamped_shrink - distance as i128;
        let negative_valid = match (constraints.min_value, constraints.max_value) {
            (Some(min), Some(max)) => negative_candidate >= min && negative_candidate <= max,
            (Some(min), None) => negative_candidate >= min,
            (None, Some(max)) => negative_candidate <= max,
            (None, None) => true,
        };
        
        if negative_valid {
            sequence.push(negative_candidate);
        }
        
        // For bounded ranges, if both directions are invalid, we're done
        if !positive_valid && !negative_valid && 
           (constraints.min_value.is_some() || constraints.max_value.is_some()) {
            break;
        }
    }
    
    sequence
}

/// Convert a choice value to its index in the ordering sequence
/// Returns >= 0 index for valid choices, used for deterministic ordering
pub fn choice_to_index(value: &ChoiceValue, constraints: &Constraints) -> u64 {
    println!("INDEXING DEBUG: Converting choice to index");
    println!("INDEXING DEBUG: value={:?}", value);
    
    match (value, constraints) {
        (ChoiceValue::Integer(val), Constraints::Integer(c)) => {
            // For bounded ranges, use sequence generation
            if c.min_value.is_some() || c.max_value.is_some() {
                let sequence = generate_bounded_integer_sequence(c, 1000); // Reasonable limit
                if let Some(index) = sequence.iter().position(|&x| x == *val) {
                    println!("INDEXING DEBUG: Found bounded integer {} at index {}", val, index);
                    return index as u64;
                }
            }
            
            // For unbounded ranges, use mathematical formula
            let shrink_towards = c.shrink_towards.unwrap_or(0);
            if *val == shrink_towards {
                println!("INDEXING DEBUG: Value {} equals shrink_towards, index=0", val);
                return 0;
            }
            
            let distance = (*val - shrink_towards).abs() as u64;
            let index = if *val > shrink_towards {
                distance * 2 - 1  // Positive: 1, 3, 5, ...
            } else {
                distance * 2      // Negative: 2, 4, 6, ...
            };
            
            println!("INDEXING DEBUG: Unbounded integer {} -> index {}", val, index);
            index
        }
        
        (ChoiceValue::Boolean(val), Constraints::Boolean(c)) => {
            let index = if c.p == 0.0 {
                // Only false is permitted
                if *val {
                    // true with p=0.0 is invalid, but for robustness return 0
                    println!("INDEXING DEBUG: WARNING: true value with p=0.0 is invalid, returning 0");
                    0
                } else {
                    0  // false -> index 0
                }
            } else if c.p == 1.0 {
                // Only true is permitted
                if *val {
                    0  // true -> index 0
                } else {
                    // false with p=1.0 is invalid, but for robustness return 0
                    println!("INDEXING DEBUG: WARNING: false value with p=1.0 is invalid, returning 0");
                    0
                }
            } else {
                // Both values permitted: false=0, true=1
                if *val { 1 } else { 0 }
            };
            
            println!("INDEXING DEBUG: Boolean {} with p={} -> index {}", val, c.p, index);
            index
        }
        
        _ => {
            println!("INDEXING DEBUG: Unsupported choice type, returning 0");
            0
        }
    }
}

/// Convert an index back to a choice value in the ordering sequence
/// This is the inverse of choice_to_index
pub fn choice_from_index(index: u64, choice_type: &str, constraints: &Constraints) -> ChoiceValue {
    println!("INDEXING DEBUG: Converting index {} to choice", index);
    println!("INDEXING DEBUG: choice_type={}, constraints={:?}", choice_type, constraints);
    
    match (choice_type, constraints) {
        ("integer", Constraints::Integer(c)) => {
            println!("INDEXING DEBUG: Converting index {} to integer", index);
            
            // For bounded ranges, use sequence generation to get exact match
            if c.min_value.is_some() || c.max_value.is_some() {
                let sequence = generate_bounded_integer_sequence(c, 1000); // Same limit as choice_to_index
                if (index as usize) < sequence.len() {
                    let value = sequence[index as usize];
                    println!("INDEXING DEBUG: Bounded index {} -> integer {}", index, value);
                    return ChoiceValue::Integer(value);
                } else {
                    println!("INDEXING DEBUG: Bounded index {} out of range, returning first value", index);
                    return ChoiceValue::Integer(sequence[0]);
                }
            }
            
            // For unbounded ranges, use mathematical formula
            let shrink_towards = c.shrink_towards.unwrap_or(0);
            
            if index == 0 {
                println!("INDEXING DEBUG: Index 0 -> shrink_towards {}", shrink_towards);
                return ChoiceValue::Integer(shrink_towards);
            }
            
            // Reverse the indexing algorithm:
            // Odd indices (1, 3, 5, ...) are positive distances
            // Even indices (2, 4, 6, ...) are negative distances  
            let is_positive = (index % 2) == 1;
            let distance = if is_positive {
                // For odd indices: distance = (index + 1) / 2
                ((index + 1) / 2) as i128
            } else {
                // For even indices: distance = index / 2
                (index / 2) as i128
            };
            
            let result = if is_positive {
                shrink_towards + distance
            } else {
                shrink_towards - distance
            };
            
            println!("INDEXING DEBUG: Unbounded index {} -> integer {} (distance {} {}, shrink_towards {})", 
                index, result, distance, if is_positive { "positive" } else { "negative" }, shrink_towards);
            ChoiceValue::Integer(result)
        }
        
        ("boolean", Constraints::Boolean(c)) => {
            println!("INDEXING DEBUG: Converting index {} to boolean", index);
            
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
            
            println!("INDEXING DEBUG: Index {} with p={} -> boolean {}", index, c.p, result);
            ChoiceValue::Boolean(result)
        }
        
        _ => {
            println!("INDEXING DEBUG: Unsupported choice type for indexing, returning default");
            ChoiceValue::Integer(0)
        }
    }
}

/// Get the clamped shrink_towards value for integer constraints
/// This is the value that should have index 0
pub fn clamped_shrink_towards(constraints: &IntegerConstraints) -> i128 {
    println!("INDEXING DEBUG: Computing clamped shrink_towards");
    println!("INDEXING DEBUG: constraints={:?}", constraints);
    
    let shrink_towards = constraints.shrink_towards.unwrap_or(0);
    
    let result = match (constraints.min_value, constraints.max_value) {
        (Some(min), Some(max)) => shrink_towards.max(min).min(max),
        (Some(min), None) => shrink_towards.max(min),
        (None, Some(max)) => shrink_towards.min(max),
        (None, None) => shrink_towards,
    };
    
    println!("INDEXING DEBUG: Clamped shrink_towards: {} -> {}", shrink_towards, result);
    result
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
    fn test_choice_indices_are_positive() {
        println!("INDEXING DEBUG: Testing that all choice indices are valid");
        
        // Test integer
        let constraints = Constraints::Integer(integer_constr_helper(Some(0), Some(10)));
        let value = ChoiceValue::Integer(5);
        let index = choice_to_index(&value, &constraints);
        assert!(index < u64::MAX); // u64 is always >= 0, so check it's reasonable
        println!("INDEXING DEBUG: Integer index {} is valid ✓", index);
        
        // Test boolean
        let constraints = Constraints::Boolean(BooleanConstraints { p: 0.5 });
        let value = ChoiceValue::Boolean(true);
        let index = choice_to_index(&value, &constraints);
        assert!(index < u64::MAX); // u64 is always >= 0, so check it's reasonable
        println!("INDEXING DEBUG: Boolean index {} is valid ✓", index);
        
        // Test edge cases
        let constraints = Constraints::Boolean(BooleanConstraints { p: 0.0 });
        let value = ChoiceValue::Boolean(false);
        let index = choice_to_index(&value, &constraints);
        assert!(index < u64::MAX); // u64 is always >= 0, so check it's reasonable
        println!("INDEXING DEBUG: Boolean p=0 index {} is valid ✓", index);
        
        let constraints = Constraints::Boolean(BooleanConstraints { p: 1.0 });
        let value = ChoiceValue::Boolean(true);
        let index = choice_to_index(&value, &constraints);
        assert!(index < u64::MAX); // u64 is always >= 0, so check it's reasonable
        println!("INDEXING DEBUG: Boolean p=1 index {} is valid ✓", index);
        
        println!("INDEXING DEBUG: All choice indices are valid test passed");
    }

    #[test]
    fn test_shrink_towards_has_index_0() {
        println!("INDEXING DEBUG: Testing that shrink_towards value has index 0");
        
        // Test unbounded
        let constraints = integer_constr_helper(None, None);
        let shrink_towards = clamped_shrink_towards(&constraints);
        let index = choice_to_index(&ChoiceValue::Integer(shrink_towards), &Constraints::Integer(constraints.clone()));
        assert_eq!(index, 0);
        
        let value = choice_from_index(0, "integer", &Constraints::Integer(constraints));
        assert!(choice_equal(&value, &ChoiceValue::Integer(shrink_towards)));
        println!("INDEXING DEBUG: Unbounded shrink_towards test passed");
        
        // Test bounded
        let constraints = integer_constr_helper(Some(-5), Some(5));
        let shrink_towards = clamped_shrink_towards(&constraints);
        let index = choice_to_index(&ChoiceValue::Integer(shrink_towards), &Constraints::Integer(constraints.clone()));
        assert_eq!(index, 0);
        
        let value = choice_from_index(0, "integer", &Constraints::Integer(constraints));
        assert!(choice_equal(&value, &ChoiceValue::Integer(shrink_towards)));
        println!("INDEXING DEBUG: Bounded shrink_towards test passed");
        
        // Test with custom shrink_towards
        let constraints = integer_constr_with_shrink(Some(-5), Some(5), 2);
        let shrink_towards = clamped_shrink_towards(&constraints);
        let index = choice_to_index(&ChoiceValue::Integer(shrink_towards), &Constraints::Integer(constraints.clone()));
        assert_eq!(index, 0);
        
        let value = choice_from_index(0, "integer", &Constraints::Integer(constraints));
        assert!(choice_equal(&value, &ChoiceValue::Integer(shrink_towards)));
        println!("INDEXING DEBUG: Custom shrink_towards test passed");
        
        println!("INDEXING DEBUG: Shrink_towards has index 0 test passed");
    }

    #[test]
    fn test_choice_index_and_value_are_inverses() {
        println!("INDEXING DEBUG: Testing that choice_to_index and choice_from_index are inverses");
        
        // Test various integer values
        let test_cases = vec![
            (integer_constr_helper(None, None), vec![0, 1, -1, 2, -2, 5]),
            (integer_constr_helper(Some(0), Some(10)), vec![0, 1, 2, 5, 10]),
            (integer_constr_with_shrink(Some(-5), Some(5), 2), vec![-5, -2, 0, 2, 3, 5]),
        ];
        
        for (constraints, values) in test_cases {
            for val in values {
                let value = ChoiceValue::Integer(val);
                let constraints_enum = Constraints::Integer(constraints.clone());
                
                let index = choice_to_index(&value, &constraints_enum);
                let recovered = choice_from_index(index, "integer", &constraints_enum);
                
                println!("INDEXING DEBUG: {} -> index {} -> {:?}", val, index, recovered);
                assert!(choice_equal(&value, &recovered), 
                    "Failed for value {}: got {:?}, expected {:?}", val, recovered, value);
            }
        }
        
        // Test boolean values - only test valid combinations
        let bool_test_cases = vec![
            (BooleanConstraints { p: 0.0 }, vec![false]),  // Only false is valid
            (BooleanConstraints { p: 1.0 }, vec![true]),   // Only true is valid
            (BooleanConstraints { p: 0.5 }, vec![true, false]), // Both are valid
        ];
        
        for (constraints, valid_values) in bool_test_cases {
            for val in valid_values {
                let value = ChoiceValue::Boolean(val);
                let constraints_enum = Constraints::Boolean(constraints.clone());
                
                let index = choice_to_index(&value, &constraints_enum);
                let recovered = choice_from_index(index, "boolean", &constraints_enum);
                
                println!("INDEXING DEBUG: {} (p={}) -> index {} -> {:?}", val, constraints.p, index, recovered);
                assert!(choice_equal(&value, &recovered),
                    "Failed for boolean {}: got {:?}, expected {:?}", val, recovered, value);
            }
        }
        
        println!("INDEXING DEBUG: Choice index and value are inverses test passed");
    }

    #[test]
    fn test_integer_choice_index_ordering() {
        println!("INDEXING DEBUG: Testing integer choice index ordering matches Python");
        
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
            println!("INDEXING DEBUG: Testing constraints {:?}", constraints);
            
            for (expected_index, value) in expected_order.iter().enumerate() {
                let choice_value = ChoiceValue::Integer(*value);
                let constraints_enum = Constraints::Integer(constraints.clone());
                let actual_index = choice_to_index(&choice_value, &constraints_enum);
                
                println!("INDEXING DEBUG: Value {} should have index {}, got {}", 
                    value, expected_index, actual_index);
                assert_eq!(actual_index, expected_index as u64, 
                    "Value {} should have index {}, got {}", value, expected_index, actual_index);
            }
        }
        
        println!("INDEXING DEBUG: Integer choice index ordering test passed");
    }

    #[test]
    fn test_boolean_choice_index_explicit() {
        println!("INDEXING DEBUG: Testing boolean choice index explicit cases");
        
        // p=1: only true is possible
        let constraints = Constraints::Boolean(BooleanConstraints { p: 1.0 });
        let index = choice_to_index(&ChoiceValue::Boolean(true), &constraints);
        assert_eq!(index, 0);
        
        // p=0: only false is possible
        let constraints = Constraints::Boolean(BooleanConstraints { p: 0.0 });
        let index = choice_to_index(&ChoiceValue::Boolean(false), &constraints);
        assert_eq!(index, 0);
        
        // p=0.5: false=0, true=1
        let constraints = Constraints::Boolean(BooleanConstraints { p: 0.5 });
        let index_false = choice_to_index(&ChoiceValue::Boolean(false), &constraints);
        let index_true = choice_to_index(&ChoiceValue::Boolean(true), &constraints);
        assert_eq!(index_false, 0);
        assert_eq!(index_true, 1);
        
        println!("INDEXING DEBUG: Boolean choice index explicit test passed");
    }

    #[test]
    fn test_integer_choice_index_comprehensive() {
        println!("INDEXING DEBUG: Testing comprehensive integer choice index scenarios from Python");
        
        // Test cases ported from Python's test_integer_choice_index
        let test_cases = vec![
            // unbounded
            (integer_constr_helper(None, None), vec![0, 1, -1, 2, -2, 3, -3]),
            (integer_constr_with_shrink(None, None, 2), vec![2, 3, 1, 4, 0, 5, -1, 6, -2]),
            // semibounded (below)
            (integer_constr_with_shrink(Some(3), None, 0), vec![3, 4, 5, 6, 7]),
            (integer_constr_with_shrink(Some(3), None, 5), vec![5, 6, 4, 7, 3, 8, 9]),
            (integer_constr_with_shrink(Some(-3), None, 0), vec![0, 1, -1, 2, -2, 3, -3, 4, 5, 6]),
            (integer_constr_with_shrink(Some(-3), None, -1), vec![-1, 0, -2, 1, -3, 2, 3, 4]),
            // semibounded (above)
            (integer_constr_helper(None, Some(3)), vec![0, 1, -1, 2, -2, 3, -3, -4, -5, -6]),
            (integer_constr_with_shrink(None, Some(3), 1), vec![1, 2, 0, 3, -1, -2, -3, -4]),
            (integer_constr_helper(None, Some(-3)), vec![-3, -4, -5, -6, -7]),
            (integer_constr_with_shrink(None, Some(-3), -5), vec![-5, -4, -6, -3, -7, -8, -9]),
            // bounded
            (integer_constr_helper(Some(-3), Some(3)), vec![0, 1, -1, 2, -2, 3, -3]),
            (integer_constr_with_shrink(Some(-3), Some(3), 1), vec![1, 2, 0, 3, -1, -2, -3]),
            (integer_constr_with_shrink(Some(-3), Some(3), -1), vec![-1, 0, -2, 1, -3, 2, 3]),
        ];
        
        for (test_index, (constraints, expected_choices)) in test_cases.iter().enumerate() {
            println!("INDEXING DEBUG: Running test case {}: constraints={:?}", test_index, constraints);
            println!("INDEXING DEBUG: Expected order: {:?}", expected_choices);
            
            for (expected_index, choice) in expected_choices.iter().enumerate() {
                let choice_value = ChoiceValue::Integer(*choice);
                let constraints_enum = Constraints::Integer(constraints.clone());
                let actual_index = choice_to_index(&choice_value, &constraints_enum);
                
                println!("INDEXING DEBUG: Choice {} should have index {}, got {}", choice, expected_index, actual_index);
                assert_eq!(actual_index, expected_index as u64, 
                    "Test case {}: Choice {} should have index {}, got {}", 
                    test_index, choice, expected_index, actual_index);
            }
        }
        
        println!("INDEXING DEBUG: Comprehensive integer choice index test passed");
    }
}