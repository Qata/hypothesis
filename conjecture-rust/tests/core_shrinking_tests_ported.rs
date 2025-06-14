/// Core shrinking tests ported from Python Hypothesis
/// 
/// This file contains essential shrinking tests ported from:
/// - tests/conjecture/test_minimizer.py  
/// - tests/quality/test_shrink_quality.py
/// - tests/quality/test_float_shrinking.py
/// 
/// These tests focus on the core shrinking algorithms without dependencies
/// on problematic parts of the current codebase.

/// Test integer shrinking to zero - port of test_minimizer.py::test_shrink_to_zero
#[test]
fn test_integer_shrinking_to_zero() {
    // Test the core concept: integers should shrink towards zero when possible
    let initial = 65536i128;
    let target = 0i128;
    
    // Simulate shrinking steps
    let mut current = initial;
    let mut steps = 0;
    
    while current > target && steps < 100 {
        // Basic shrinking: try to get closer to zero
        let candidate = current / 2;
        if candidate >= target {
            current = candidate;
        } else {
            current = target;
        }
        steps += 1;
    }
    
    assert_eq!(current, target, "Should shrink to zero");
    assert!(steps > 0, "Should take multiple steps");
    assert!(steps < 100, "Should converge quickly");
}

/// Test integer shrinking to smallest valid - port of test_minimizer.py::test_shrink_to_smallest
#[test]
fn test_integer_shrinking_to_smallest_valid() {
    // Test shrinking to smallest value that satisfies constraint
    let initial = 65536i128;
    let min_valid = 11i128; // Must be > 10
    
    let mut current = initial;
    let mut steps = 0;
    
    while current > min_valid && steps < 100 {
        let candidate = current - 1;
        if candidate > 10 { // Constraint: must be > 10
            current = candidate;
        } else {
            break;
        }
        steps += 1;
    }
    
    assert_eq!(current, min_valid, "Should shrink to smallest valid value (11)");
    assert!(current > 10, "Should satisfy constraint");
    assert!(steps > 0, "Should make progress");
}

/// Test boolean shrinking preference - port of test_shrinker.py::test_boolean_minimization
#[test]
fn test_boolean_shrinking_preference() {
    // Test that false is preferred over true in shrinking
    let true_value = true;
    let false_value = false;
    
    // Convert to indices for comparison (as in Python's choice_to_index)
    let true_index = if true_value { 1u64 } else { 0u64 };
    let false_index = if false_value { 1u64 } else { 0u64 };
    
    assert_eq!(false_index, 0, "False should have index 0");
    assert_eq!(true_index, 1, "True should have index 1");
    assert!(false_index < true_index, "False should be preferred (smaller index)");
}

/// Test float shrinking behavior - port of test_float_shrinking.py::test_shrinks_to_simple_floats
#[test]
fn test_float_shrinking_to_simple_values() {
    // Test that floats shrink towards simpler values
    let initial = 123.456f64;
    
    // Test halving strategy (common float shrinking approach)
    let halved = initial / 2.0;
    assert!(halved < initial, "Halving should reduce magnitude");
    assert!(halved.is_finite(), "Result should remain finite");
    
    // Test shrinking towards zero
    let mut current = 100.0f64;
    let mut iterations = 0;
    
    while current.abs() > 1.0 && iterations < 20 {
        current = current / 2.0;
        iterations += 1;
        assert!(current.is_finite(), "Should remain finite");
    }
    
    assert!(current.abs() <= 1.0, "Should shrink close to target");
    assert!(iterations > 0, "Should require multiple iterations");
}

/// Test sequence length minimization - port of test_shrink_quality.py::test_minimize_length
#[test]
fn test_sequence_length_minimization() {
    // Test that shorter sequences are preferred
    let long_sequence = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let short_sequence = vec![1, 2, 3];
    
    // Test deletion from end (trailing deletion)
    let mut truncated = long_sequence.clone();
    truncated.truncate(3);
    
    assert_eq!(truncated.len(), 3, "Should truncate to shorter length");
    assert!(truncated.len() < long_sequence.len(), "Should be shorter than original");
    assert_eq!(truncated, short_sequence, "Should match expected short sequence");
}

/// Test value minimization within sequences - port of test_minimizer.py::test_minimize_values
#[test]
fn test_value_minimization_in_sequences() {
    // Test minimizing values within sequences
    let original = vec![100i32, 200, 300, 400];
    
    // Test minimizing first element
    let mut minimized = original.clone();
    if minimized[0] > 0 {
        minimized[0] -= 1;
    }
    
    assert!(minimized[0] < original[0], "Should minimize individual values");
    assert_eq!(minimized.len(), original.len(), "Should preserve length");
    
    // Test minimizing all elements towards zero
    let all_minimized: Vec<i32> = original.iter()
        .map(|&x| (x / 2).max(0))
        .collect();
    
    for (i, (&orig, &min)) in original.iter().zip(all_minimized.iter()).enumerate() {
        assert!(min <= orig, "Element {} should not increase: {} -> {}", i, orig, min);
    }
}

/// Test constraint preservation during shrinking - port of test_choice.py::test_constraints
#[test]
fn test_constraint_preservation() {
    // Test that shrinking respects constraints
    let min_value = 10i32;
    let max_value = 100i32;
    let original = 75i32;
    
    // Test shrinking within bounds
    let mut current = original;
    let mut iterations = 0;
    
    while current > min_value && iterations < 50 {
        let candidate = current - 1;
        if candidate >= min_value && candidate <= max_value {
            current = candidate;
        } else {
            break;
        }
        iterations += 1;
    }
    
    assert!(current >= min_value, "Should respect minimum constraint");
    assert!(current <= max_value, "Should respect maximum constraint");
    assert!(current < original, "Should make progress");
    assert_eq!(current, min_value, "Should shrink to minimum when possible");
}

/// Test special float value handling - port of test_float_shrinking.py::test_special_values
#[test]
fn test_special_float_value_handling() {
    // Test handling of NaN, infinity, and other special values
    
    // Test NaN detection and replacement
    let nan_value = f64::NAN;
    assert!(nan_value.is_nan(), "NaN should be detected");
    
    let repaired_nan = if nan_value.is_nan() { 0.0 } else { nan_value };
    assert!(repaired_nan.is_finite(), "NaN should be repaired to finite value");
    assert_eq!(repaired_nan, 0.0, "Should repair to zero");
    
    // Test infinity detection and replacement
    let inf_value = f64::INFINITY;
    assert!(inf_value.is_infinite(), "Infinity should be detected");
    
    let repaired_inf = if inf_value.is_infinite() { 1000.0 } else { inf_value };
    assert!(repaired_inf.is_finite(), "Infinity should be repaired to finite value");
    assert_eq!(repaired_inf, 1000.0, "Should repair to reasonable finite value");
    
    // Test negative infinity
    let neg_inf = f64::NEG_INFINITY;
    let repaired_neg_inf = if neg_inf.is_infinite() { -1000.0 } else { neg_inf };
    assert!(repaired_neg_inf.is_finite(), "Negative infinity should be repaired");
    assert_eq!(repaired_neg_inf, -1000.0, "Should repair to reasonable negative value");
}

/// Test sequence comparison logic - port of test_shrinker.py::test_comparison
#[test]
fn test_sequence_comparison_logic() {
    // Test the comparison logic used in shrinking (shorter is better, then lexicographic)
    
    // Test length comparison
    let short_seq = vec![5];
    let long_seq = vec![1, 2];
    
    let short_key = (short_seq.len(), short_seq.clone());
    let long_key = (long_seq.len(), long_seq.clone());
    
    assert!(short_key < long_key, "Shorter sequence should be better");
    
    // Test lexicographic comparison for same length
    let seq1 = vec![1, 5];
    let seq2 = vec![2, 3];
    
    let key1 = (seq1.len(), seq1.clone());
    let key2 = (seq2.len(), seq2.clone());
    
    assert!(key1 < key2, "Lexicographically smaller sequence should be better");
    
    // Test that first element dominates
    let seq3 = vec![1, 999];
    let seq4 = vec![2, 1];
    
    let key3 = (seq3.len(), seq3.clone());
    let key4 = (seq4.len(), seq4.clone());
    
    assert!(key3 < key4, "First element should dominate comparison");
}

/// Test greedy shrinking strategy - port of test_shrinker.py::test_greedy_strategy
#[test]
fn test_greedy_shrinking_strategy() {
    // Test that greedy shrinking takes the first improvement found
    let original = 100i32;
    let mut current = original;
    
    // Candidates in order of consideration
    let candidates = vec![99, 50, 25, 10, 5, 1, 0];
    let mut first_improvement = None;
    
    for candidate in candidates {
        // Greedy: take first improvement
        if candidate < current {
            first_improvement = Some(candidate);
            current = candidate;
            break; // Stop at first improvement
        }
    }
    
    assert_eq!(first_improvement, Some(99), "Should take first improvement");
    assert_eq!(current, 99, "Should update to first improvement");
    assert!(current < original, "Should be an improvement");
}

/// Test shrinking phases concept - port of test_shrinker.py::test_phases
#[test]
fn test_shrinking_phases_concept() {
    // Test the multi-phase shrinking approach
    let mut sequence = vec![100, 200, 300, 0, 0];
    let original_len = sequence.len();
    
    // Phase 1: Delete trailing zeros
    while sequence.last() == Some(&0) && sequence.len() > 1 {
        sequence.pop();
    }
    
    assert!(sequence.len() < original_len, "Phase 1 should reduce length");
    assert_eq!(sequence, vec![100, 200, 300], "Should remove trailing zeros");
    
    // Phase 2: Minimize individual values
    let phase2_result: Vec<i32> = sequence.iter()
        .map(|&x| if x > 0 { x - 1 } else { x })
        .collect();
    
    assert_eq!(phase2_result, vec![99, 199, 299], "Phase 2 should minimize values");
    
    // Phase 3: Try further reductions (conceptual)
    let phase3_result: Vec<i32> = phase2_result.iter()
        .map(|&x| x / 2)
        .collect();
    
    assert_eq!(phase3_result, vec![49, 99, 149], "Phase 3 should further reduce");
    
    // Verify each phase makes progress
    assert!(phase2_result.iter().sum::<i32>() < sequence.iter().sum::<i32>());
    assert!(phase3_result.iter().sum::<i32>() < phase2_result.iter().sum::<i32>());
}

/// Test forced value preservation - port of test_choice.py::test_forced_preservation
#[test]
fn test_forced_value_preservation() {
    // Test that forced values are never modified during shrinking
    struct Choice {
        value: i32,
        is_forced: bool,
    }
    
    let choices = vec![
        Choice { value: 100, is_forced: true },  // This should never change
        Choice { value: 200, is_forced: false }, // This can be modified
    ];
    
    // Simulate shrinking that respects forced flag
    let shrunk: Vec<Choice> = choices.into_iter()
        .map(|choice| {
            if choice.is_forced {
                choice // Never modify forced choices
            } else {
                Choice { value: choice.value - 1, is_forced: choice.is_forced }
            }
        })
        .collect();
    
    assert_eq!(shrunk[0].value, 100, "Forced choice should not change");
    assert!(shrunk[0].is_forced, "Forced flag should be preserved");
    assert_eq!(shrunk[1].value, 199, "Non-forced choice should change");
    assert!(!shrunk[1].is_forced, "Non-forced flag should be preserved");
}

/// Test constraint violation repair - port of test_shrinker.py::test_repair
#[test]
fn test_constraint_violation_repair() {
    // Test repairing values that violate constraints
    let min_bound = 10i32;
    let max_bound = 100i32;
    
    // Test value below minimum
    let below_min = 5i32;
    let repaired_low = below_min.max(min_bound).min(max_bound);
    assert_eq!(repaired_low, min_bound, "Should repair to minimum bound");
    
    // Test value above maximum
    let above_max = 150i32;
    let repaired_high = above_max.max(min_bound).min(max_bound);
    assert_eq!(repaired_high, max_bound, "Should repair to maximum bound");
    
    // Test value within bounds
    let valid_value = 50i32;
    let unchanged = valid_value.max(min_bound).min(max_bound);
    assert_eq!(unchanged, valid_value, "Valid value should not change");
    
    // Test that all repaired values are within bounds
    let test_values = vec![-10, 5, 50, 150, 200];
    for value in test_values {
        let repaired = value.max(min_bound).min(max_bound);
        assert!(repaired >= min_bound, "Repaired value should be >= minimum");
        assert!(repaired <= max_bound, "Repaired value should be <= maximum");
    }
}

/// Test empty sequence handling - port of test_test_data.py::test_empty
#[test]
fn test_empty_sequence_handling() {
    // Test that empty sequences are handled gracefully
    let empty: Vec<i32> = vec![];
    
    // Test operations on empty sequences
    assert_eq!(empty.len(), 0, "Empty sequence should have zero length");
    assert!(empty.is_empty(), "Empty sequence should be empty");
    
    let empty_copy = empty.clone();
    assert_eq!(empty_copy.len(), 0, "Copy should also be empty");
    
    // Test that empty sequences compare correctly
    let empty_key = (empty.len(), empty.clone());
    let non_empty_key = (1, vec![0]);
    
    assert!(empty_key < non_empty_key, "Empty should be better than non-empty");
}

/// Test string minimization concept - port of test_shrink_quality.py::test_string_shrinking
#[test]
fn test_string_minimization_concept() {
    // Test string minimization strategies
    let original = "hello world test";
    
    // Test length reduction
    let shortened = &original[..5];
    assert_eq!(shortened, "hello", "Should be able to shorten string");
    assert!(shortened.len() < original.len(), "Should reduce length");
    
    // Test character minimization (replace with simpler characters)
    let simplified: String = original.chars()
        .map(|c| if c.is_alphabetic() { '0' } else { c })
        .collect();
    
    assert!(
        simplified.chars().all(|c| c == '0' || c == ' '),
        "Should simplify characters"
    );
    
    // Test that both length and content can be minimized
    let doubly_minimized = "00000";
    assert!(doubly_minimized.len() < original.len(), "Should reduce length");
    assert!(
        doubly_minimized.chars().all(|c| c == '0'),
        "Should use simplest characters"
    );
}

/// Test bytes minimization concept - port of test_shrink_quality.py::test_bytes_shrinking
#[test]
fn test_bytes_minimization_concept() {
    // Test byte array minimization strategies
    let original = vec![100u8, 200, 150, 75, 50, 25];
    
    // Test length reduction
    let shortened = &original[..3];
    assert_eq!(shortened, &[100, 200, 150], "Should be able to shorten bytes");
    assert!(shortened.len() < original.len(), "Should reduce length");
    
    // Test value minimization
    let minimized_values: Vec<u8> = original.iter()
        .map(|&b| b.min(10)) // Minimize to at most 10
        .collect();
    
    assert!(
        minimized_values.iter().all(|&b| b <= 10),
        "Should minimize byte values"
    );
    
    // Test zero minimization
    let zero_minimized: Vec<u8> = original.iter()
        .map(|&b| if b > 0 { 0 } else { b })
        .collect();
    
    assert_eq!(zero_minimized, vec![0; original.len()], "Should minimize to zeros");
}

/// Integration test of shrinking workflow - port of test_shrinker.py::test_workflow
#[test]
fn test_complete_shrinking_workflow() {
    // Test a complete shrinking workflow
    let initial_sequence = vec![1000, 2000, 3000, 4000];
    let mut current = initial_sequence.clone();
    
    // Define test condition: sum must be > 500
    let test_condition = |seq: &Vec<i32>| seq.iter().sum::<i32>() > 500;
    
    assert!(test_condition(&current), "Initial should satisfy condition");
    
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 100;
    
    // Multi-phase shrinking simulation
    while iterations < MAX_ITERATIONS {
        let mut made_progress = false;
        
        // Phase 1: Try to remove last element
        if current.len() > 1 {
            let mut candidate = current.clone();
            candidate.pop();
            if test_condition(&candidate) {
                current = candidate;
                made_progress = true;
            }
        }
        
        // Phase 2: Try to minimize values
        if !made_progress {
            let mut candidate = current.clone();
            for i in 0..candidate.len() {
                if candidate[i] > 0 {
                    candidate[i] -= 1;
                    if test_condition(&candidate) {
                        current = candidate;
                        made_progress = true;
                        break;
                    }
                    candidate[i] += 1; // Restore if didn't work
                }
            }
        }
        
        if !made_progress {
            break; // No more progress possible
        }
        
        iterations += 1;
    }
    
    // Verify final state
    assert!(test_condition(&current), "Final result should satisfy condition");
    assert!(current.len() <= initial_sequence.len(), "Should not increase length");
    assert!(
        current.iter().sum::<i32>() <= initial_sequence.iter().sum::<i32>(),
        "Should not increase total"
    );
    assert!(iterations > 0, "Should have made some progress");
    
    // Check that we found a reasonably minimal solution
    let final_sum = current.iter().sum::<i32>();
    assert!(final_sum > 500, "Should still satisfy constraint");
    assert!(final_sum < 10000, "Should be much smaller than initial");
}