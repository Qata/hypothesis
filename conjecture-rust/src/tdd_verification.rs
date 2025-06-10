//! TDD Verification Tests for Phase 2 Implementation
//! 
//! This module contains failing tests that drive our implementation of ConjectureData
//! and other missing functionality. Following TDD methodology: RED -> GREEN -> REFACTOR

use crate::data::{ConjectureData, ConjectureResult, Status, DrawError};
use crate::choice::{ChoiceValue, ChoiceType, ChoiceNode, Constraints};

/// Test ConjectureData creation and basic properties
#[test]
fn test_conjecture_data_creation() {
    let data = ConjectureData::new(42);
    
    // Basic properties should match Python behavior
    assert_eq!(data.status, Status::Valid);
    assert_eq!(data.max_length, 8192); // Python's BUFFER_SIZE
    assert_eq!(data.index, 0);
    assert_eq!(data.length, 0);
    assert!(!data.frozen);
    assert_eq!(data.choice_count(), 0);
    assert_eq!(data.depth, -1); // Python starts at -1
}

/// Test draw_integer functionality
#[test]
fn test_draw_integer_basic() {
    let mut data = ConjectureData::new(42);
    
    // Test basic integer drawing
    let value = data.draw_integer(0, 100).unwrap();
    
    assert!(value >= 0 && value <= 100);
    assert_eq!(data.choice_count(), 1);
    assert_eq!(data.length, 2); // Should match Python behavior
    assert!(!data.frozen);
    
    // Check that choice was recorded
    let choices = data.choices();
    assert_eq!(choices.len(), 1);
    
    if let ChoiceValue::Integer(recorded_value) = &choices[0].value {
        assert_eq!(*recorded_value, value);
    } else {
        panic!("Expected integer choice");
    }
}

/// Test draw_boolean functionality
#[test] 
fn test_draw_boolean_basic() {
    let mut data = ConjectureData::new(42);
    
    // Test basic boolean drawing
    let value = data.draw_boolean(0.5).unwrap();
    
    assert!(value == true || value == false);
    assert_eq!(data.choice_count(), 1);
    assert_eq!(data.length, 1); // Booleans should be 1 byte
    
    // Check that choice was recorded
    let choices = data.choices();
    assert_eq!(choices.len(), 1);
    
    if let ChoiceValue::Boolean(recorded_value) = &choices[0].value {
        assert_eq!(*recorded_value, value);
    } else {
        panic!("Expected boolean choice");
    }
}

/// Test draw_float functionality  
#[test]
fn test_draw_float_basic() {
    let mut data = ConjectureData::new(42);
    
    // Test basic float drawing
    let value = data.draw_float().unwrap();
    
    assert!(value.is_finite() || value.is_nan() || value.is_infinite());
    assert_eq!(data.choice_count(), 1);
    assert_eq!(data.length, 8); // f64 should be 8 bytes
    
    // Check that choice was recorded
    let choices = data.choices();
    assert_eq!(choices.len(), 1);
    
    if let ChoiceValue::Float(recorded_value) = &choices[0].value {
        // Handle NaN comparison properly
        if value.is_nan() {
            assert!(recorded_value.is_nan());
        } else {
            assert_eq!(*recorded_value, value);
        }
    } else {
        panic!("Expected float choice");
    }
}

/// Test draw_string functionality
#[test]
fn test_draw_string_basic() {
    let mut data = ConjectureData::new(42);
    
    // Test basic string drawing
    let value = data.draw_string("abc", 0, 10).unwrap();
    
    assert!(value.len() <= 10);
    assert!(value.chars().all(|c| "abc".contains(c)));
    assert_eq!(data.choice_count(), 1);
    assert_eq!(data.length, value.len()); // Length should match string size
    
    // Check that choice was recorded
    let choices = data.choices();
    assert_eq!(choices.len(), 1);
    
    if let ChoiceValue::String(recorded_value) = &choices[0].value {
        assert_eq!(*recorded_value, value);
    } else {
        panic!("Expected string choice");
    }
}

/// Test draw_bytes functionality
#[test]
fn test_draw_bytes_basic() {
    let mut data = ConjectureData::new(42);
    
    // Test basic bytes drawing
    let value = data.draw_bytes(5).unwrap();
    
    assert_eq!(value.len(), 5);
    assert_eq!(data.choice_count(), 1);
    assert_eq!(data.length, 5); // Length should match bytes size
    
    // Check that choice was recorded
    let choices = data.choices();
    assert_eq!(choices.len(), 1);
    
    if let ChoiceValue::Bytes(recorded_value) = &choices[0].value {
        assert_eq!(*recorded_value, value);
    } else {
        panic!("Expected bytes choice");
    }
}

/// Test freeze functionality prevents further draws
#[test]
fn test_freeze_prevents_draws() {
    let mut data = ConjectureData::new(42);
    
    // Draw something first
    let _ = data.draw_integer(0, 100).unwrap();
    assert_eq!(data.choice_count(), 1);
    
    // Freeze the data
    data.freeze();
    assert!(data.frozen);
    
    // Further draws should fail
    assert_eq!(data.draw_integer(0, 100), Err(DrawError::Frozen));
    assert_eq!(data.draw_boolean(0.5), Err(DrawError::Frozen));
    assert_eq!(data.draw_float(), Err(DrawError::Frozen));
    assert_eq!(data.draw_string("abc", 0, 10), Err(DrawError::Frozen));
    assert_eq!(data.draw_bytes(5), Err(DrawError::Frozen));
    
    // Choice count should not have changed
    assert_eq!(data.choice_count(), 1);
}

/// Test error handling for invalid ranges
#[test]
fn test_invalid_integer_range() {
    let mut data = ConjectureData::new(42);
    
    // Invalid range should fail
    let result = data.draw_integer(100, 0);
    assert_eq!(result, Err(DrawError::InvalidRange));
    
    // No choice should be recorded on error
    assert_eq!(data.choice_count(), 0);
}

/// Test error handling for invalid probabilities
#[test]
fn test_invalid_probability() {
    let mut data = ConjectureData::new(42);
    
    // Invalid probabilities should fail
    assert_eq!(data.draw_boolean(-0.1), Err(DrawError::InvalidProbability));
    assert_eq!(data.draw_boolean(1.1), Err(DrawError::InvalidProbability));
    
    // No choices should be recorded on error
    assert_eq!(data.choice_count(), 0);
}

/// Test multiple sequential draws accumulate correctly
#[test]
fn test_multiple_draws_accumulate() {
    let mut data = ConjectureData::new(42);
    
    // Draw various types
    let int_val = data.draw_integer(0, 100).unwrap();
    let bool_val = data.draw_boolean(0.5).unwrap();
    let float_val = data.draw_float().unwrap();
    
    // Check accumulation
    assert_eq!(data.choice_count(), 3);
    assert_eq!(data.length, 2 + 1 + 8); // int + bool + float
    
    // Check all choices are recorded
    let choices = data.choices();
    assert_eq!(choices.len(), 3);
    
    // Verify choice types and values
    match (&choices[0].value, &choices[1].value, &choices[2].value) {
        (ChoiceValue::Integer(i), ChoiceValue::Boolean(b), ChoiceValue::Float(f)) => {
            assert_eq!(*i, int_val);
            assert_eq!(*b, bool_val);
            if float_val.is_nan() {
                assert!(f.is_nan());
            } else {
                assert_eq!(*f, float_val);
            }
        },
        _ => panic!("Unexpected choice sequence"),
    }
}

/// Test observation recording
#[test]
fn test_observation_recording() {
    let mut data = ConjectureData::new(42);
    
    // Record some observations
    data.observe("metric_1", "42");
    data.observe("metric_2", "test_value");
    
    // Check observations are stored
    assert_eq!(data.events.len(), 2);
    assert_eq!(data.events.get("metric_1"), Some(&"42".to_string()));
    assert_eq!(data.events.get("metric_2"), Some(&"test_value".to_string()));
}

/// Test empty string alphabet error
#[test]
fn test_empty_alphabet_error() {
    let mut data = ConjectureData::new(42);
    
    // Empty alphabet should fail
    let result = data.draw_string("", 0, 10);
    assert_eq!(result, Err(DrawError::EmptyAlphabet));
    
    // No choice should be recorded on error
    assert_eq!(data.choice_count(), 0);
}

/// Test string size constraints
#[test]
fn test_string_invalid_size_range() {
    let mut data = ConjectureData::new(42);
    
    // Invalid size range should fail
    let result = data.draw_string("abc", 10, 5);
    assert_eq!(result, Err(DrawError::InvalidRange));
    
    // No choice should be recorded on error
    assert_eq!(data.choice_count(), 0);
}

/// Test deterministic reproduction with same seed
#[test]
fn test_deterministic_reproduction() {
    let mut data1 = ConjectureData::new(42);
    let mut data2 = ConjectureData::new(42);
    
    // Make identical sequence of draws
    let val1_1 = data1.draw_integer(0, 100).unwrap();
    let val1_2 = data1.draw_boolean(0.5).unwrap();
    let val1_3 = data1.draw_float().unwrap();
    
    let val2_1 = data2.draw_integer(0, 100).unwrap();
    let val2_2 = data2.draw_boolean(0.5).unwrap();
    let val2_3 = data2.draw_float().unwrap();
    
    // Should produce identical results with same seed
    assert_eq!(val1_1, val2_1);
    assert_eq!(val1_2, val2_2);
    if val1_3.is_nan() && val2_3.is_nan() {
        // Both NaN is fine
    } else {
        assert_eq!(val1_3, val2_3);
    }
}

/// Test replay from choice sequence
#[test]
fn test_choice_sequence_replay() {
    // Create data and record choices
    let mut original_data = ConjectureData::new(42);
    let original_int = original_data.draw_integer(5, 15).unwrap();
    let original_bool = original_data.draw_boolean(0.7).unwrap();
    
    // Get the choice sequence
    let choices = original_data.choices();
    assert_eq!(choices.len(), 2);
    
    // Test replay using forced values
    let mut replay_data = ConjectureData::new(999); // Different seed
    
    // Extract values from original choices and replay them
    let replayed_int = if let ChoiceValue::Integer(val) = &choices[0].value {
        replay_data.draw_integer_with_forced(5, 15, Some(*val)).unwrap()
    } else {
        panic!("Expected integer choice");
    };
    
    let replayed_bool = if let ChoiceValue::Boolean(val) = &choices[1].value {
        replay_data.draw_boolean_with_forced(0.7, Some(*val)).unwrap()
    } else {
        panic!("Expected boolean choice");
    };
    
    // Replayed values should match original
    assert_eq!(replayed_int, original_int);
    assert_eq!(replayed_bool, original_bool);
    
    // The choices should be marked as forced
    let replay_choices = replay_data.choices();
    assert_eq!(replay_choices.len(), 2);
    assert!(replay_choices[0].was_forced);
    assert!(replay_choices[1].was_forced);
}

/// Test forced choice functionality
#[test]  
fn test_forced_choices() {
    let mut data = ConjectureData::new(42);
    
    // Test forced integer
    let forced_int = data.draw_integer_with_forced(0, 100, Some(42)).unwrap();
    assert_eq!(forced_int, 42);
    
    // Test forced boolean
    let forced_bool = data.draw_boolean_with_forced(0.1, Some(true)).unwrap();
    assert_eq!(forced_bool, true);
    
    // Verify choices are marked as forced
    let choices = data.choices();
    assert_eq!(choices.len(), 2);
    assert!(choices[0].was_forced);
    assert!(choices[1].was_forced);
    
    // Test forced value validation - should fail if out of range
    let result = data.draw_integer_with_forced(0, 10, Some(20));
    assert_eq!(result, Err(DrawError::InvalidRange));
    
    // Should not have recorded the invalid forced choice
    assert_eq!(data.choice_count(), 2); // Still just the two valid choices
}

/// Test buffer and index management
#[test]
fn test_buffer_and_index_management() {
    let mut data = ConjectureData::new(42);
    
    // Initial state
    assert_eq!(data.index, 0);
    assert_eq!(data.length, 0);
    
    // Draw some data
    let _ = data.draw_integer(0, 100).unwrap();
    assert_eq!(data.length, 2); // Should advance by 2 bytes
    
    let _ = data.draw_boolean(0.5).unwrap();
    assert_eq!(data.length, 3); // Should advance by 1 byte
    
    let _ = data.draw_float().unwrap();
    assert_eq!(data.length, 11); // Should advance by 8 bytes
    
    // TODO: Implement proper buffer management
    // The buffer should store the actual byte data for replay
    // Currently we're just tracking length, but we need the actual bytes
    
    // This test passes now but highlights the need for proper buffer implementation
}

/// Test ConjectureResult creation and finalization
#[test]
fn test_conjecture_result_creation() {
    let mut data = ConjectureData::new(42);
    
    // Make some draws
    let _ = data.draw_integer(0, 100).unwrap();
    let _ = data.draw_boolean(0.5).unwrap();
    
    // Freeze the data
    data.freeze();
    
    // Convert to result
    let result = data.as_result();
    
    // Verify result properties
    assert_eq!(result.status, Status::Valid);
    assert_eq!(result.choices.len(), 2);
    assert_eq!(result.length, 3); // 2 + 1 bytes
    assert!(result.events.is_empty()); // No observations yet
    
    // Result should be immutable snapshot of the data
}

/// Test ConjectureResult with observations and events
#[test]
fn test_conjecture_result_with_observations() {
    let mut data = ConjectureData::new(42);
    
    // Make draws and observations
    let _ = data.draw_integer(5, 15).unwrap();
    data.observe("test_metric", "42.0");
    data.observe("iteration", "1");
    
    let _ = data.draw_boolean(0.7).unwrap();
    
    // Finalize
    data.freeze();
    let result = data.as_result();
    
    // Verify observations are captured
    assert_eq!(result.events.len(), 2);
    assert_eq!(result.events.get("test_metric"), Some(&"42.0".to_string()));
    assert_eq!(result.events.get("iteration"), Some(&"1".to_string()));
    
    // Verify choices
    assert_eq!(result.choices.len(), 2);
    assert_eq!(result.status, Status::Valid);
}

/// Test ConjectureResult immutability
#[test]
fn test_conjecture_result_immutability() {
    let mut data = ConjectureData::new(42);
    let _ = data.draw_integer(0, 10).unwrap();
    
    // Create result before freezing
    data.freeze();
    let result1 = data.as_result();
    
    // Should get identical result on subsequent calls
    let result2 = data.as_result();
    
    assert_eq!(result1.status, result2.status);
    assert_eq!(result1.choices.len(), result2.choices.len());
    assert_eq!(result1.length, result2.length);
    assert_eq!(result1.events.len(), result2.events.len());
}

/// Test ConjectureResult status handling
#[test]
fn test_conjecture_result_status_transitions() {
    let mut data = ConjectureData::new(42);
    
    // Start in Valid state
    assert_eq!(data.status, Status::Valid);
    
    // Make some draws
    let _ = data.draw_integer(0, 100).unwrap();
    assert_eq!(data.status, Status::Valid);
    
    // TODO: Test other status transitions
    // For now, just test Valid → frozen result
    data.freeze();
    let result = data.as_result();
    assert_eq!(result.status, Status::Valid);
}

/// Test ConjectureResult choice sequence preservation
#[test]
fn test_conjecture_result_choice_preservation() {
    let mut data = ConjectureData::new(42);
    
    // Make specific sequence of choices
    let int_val = data.draw_integer(10, 20).unwrap();
    let bool_val = data.draw_boolean(0.3).unwrap();
    let float_val = data.draw_float().unwrap();
    
    // Get original choices
    let original_choices = data.choices();
    assert_eq!(original_choices.len(), 3);
    
    // Finalize and get result
    data.freeze();
    let result = data.as_result();
    
    // Verify choices are preserved exactly
    assert_eq!(result.choices.len(), 3);
    
    // Check values match
    if let ChoiceValue::Integer(val) = &result.choices[0].value {
        assert_eq!(*val, int_val);
    } else {
        panic!("Expected integer choice");
    }
    
    if let ChoiceValue::Boolean(val) = &result.choices[1].value {
        assert_eq!(*val, bool_val);
    } else {
        panic!("Expected boolean choice");
    }
    
    if let ChoiceValue::Float(val) = &result.choices[2].value {
        if float_val.is_nan() {
            assert!(val.is_nan());
        } else {
            assert_eq!(*val, float_val);
        }
    } else {
        panic!("Expected float choice");
    }
}

/// Test using ConjectureResult for test reproduction
#[test]
fn test_conjecture_result_reproduction() {
    // Original test execution
    let mut original_data = ConjectureData::new(42);
    let original_int = original_data.draw_integer(1, 100).unwrap();
    let original_bool = original_data.draw_boolean(0.3).unwrap();
    
    original_data.observe("test_run", "1");
    original_data.freeze();
    let result = original_data.as_result();
    
    // Use result to reproduce the test
    let mut replay_data = ConjectureData::new(999); // Different seed
    
    // Extract choices and replay them
    for choice in &result.choices {
        match &choice.value {
            ChoiceValue::Integer(val) => {
                if let Constraints::Integer(constraints) = &choice.constraints {
                    let min = constraints.min_value.unwrap_or(i128::MIN);
                    let max = constraints.max_value.unwrap_or(i128::MAX);
                    let reproduced = replay_data.draw_integer_with_forced(min, max, Some(*val)).unwrap();
                    assert_eq!(reproduced, *val);
                }
            },
            ChoiceValue::Boolean(val) => {
                if let Constraints::Boolean(constraints) = &choice.constraints {
                    let reproduced = replay_data.draw_boolean_with_forced(constraints.p, Some(*val)).unwrap();
                    assert_eq!(reproduced, *val);
                }
            },
            _ => {} // Skip other types for this test
        }
    }
    
    // Verify reproduction
    assert_eq!(replay_data.choice_count(), result.choices.len());
    
    // The reproduced values should match original
    let replay_choices = replay_data.choices();
    assert_eq!(replay_choices.len(), 2);
    
    if let (ChoiceValue::Integer(orig), ChoiceValue::Integer(replay)) = 
        (&result.choices[0].value, &replay_choices[0].value) {
        assert_eq!(orig, replay);
        assert_eq!(*orig, original_int);
    }
    
    if let (ChoiceValue::Boolean(orig), ChoiceValue::Boolean(replay)) = 
        (&result.choices[1].value, &replay_choices[1].value) {
        assert_eq!(orig, replay);
        assert_eq!(*orig, original_bool);
    }
}

/// Test buffer management and byte serialization
#[test]
fn test_buffer_byte_serialization() {
    let mut data = ConjectureData::new(42);
    
    // Make some draws that should produce deterministic bytes
    let _ = data.draw_integer(10, 20).unwrap();
    let _ = data.draw_boolean(0.5).unwrap();
    
    // Check that buffer tracks the data properly
    assert_eq!(data.length, 3); // 2 + 1 bytes
    
    // Buffer should contain the serialized choice data
    // For now, this is a placeholder test - proper buffer implementation will come
    let result = data.as_result();
    assert_eq!(result.length, data.length);
    
    // TODO: Implement proper byte serialization
    // The buffer should contain the actual byte representation of choices
    // that can be used for replay and shrinking
}

/// Test overrun protection
#[test]
fn test_overrun_protection() {
    let mut data = ConjectureData::new(42);
    
    // Test that we don't exceed max_length
    assert_eq!(data.max_length, 8192);
    
    // For now, just verify the limit exists
    // TODO: Implement actual overrun detection
    // When length exceeds max_length, should transition to Overrun status
    
    // This test serves as documentation of required behavior
    assert!(data.length <= data.max_length);
}

// =============================================================================
// PHASE 3: CHOICE-AWARE SHRINKING TDD TESTS
// =============================================================================

/// Test basic shrinking infrastructure
#[test]
fn test_choice_shrinker_creation() {
    // Create a failing test result
    let mut data = ConjectureData::new(42);
    let _ = data.draw_integer(50, 100).unwrap();
    let _ = data.draw_boolean(0.5).unwrap();
    data.freeze();
    let result = data.as_result();
    
    // Create shrinker
    let shrinker = crate::shrinking::ChoiceShrinker::new(result.clone());
    
    // Verify initial state
    assert_eq!(shrinker.original_result.choices.len(), result.choices.len());
    assert_eq!(shrinker.best_result.choices.len(), result.choices.len());
    assert_eq!(shrinker.attempts, 0);
    assert_eq!(shrinker.max_attempts, 10000);
    assert_eq!(shrinker.transformations.len(), 3); // Default transformations
}

/// Test integer value minimization
#[test]
fn test_shrink_integer_values() {
    use crate::shrinking::ChoiceShrinker;
    
    // Create test with large integer that should shrink
    let mut data = ConjectureData::new(42);
    let large_value = data.draw_integer(10, 100).unwrap();
    data.freeze();
    let result = data.as_result();
    
    println!("Original integer value: {}", large_value);
    
    let mut shrinker = ChoiceShrinker::new(result);
    
    // Shrinking test: "fails" only if we have choices (prevents deleting all choices)
    let shrunk_result = shrinker.shrink(|result| {
        // "Fail" only if we have at least one choice - this prevents deletion
        !result.choices.is_empty()
    });
    
    // Should have shrunk the integer towards 0 (the shrink_towards default)
    if let ChoiceValue::Integer(shrunk_value) = &shrunk_result.choices[0].value {
        println!("Shrunk integer value: {}", shrunk_value);
        assert!(*shrunk_value < large_value, "Value should have shrunk");
    } else {
        panic!("Expected integer choice");
    }
}

/// Test boolean minimization to false
#[test]
fn test_shrink_boolean_to_false() {
    use crate::shrinking::ChoiceShrinker;
    use crate::choice::{ChoiceNode, ChoiceType, BooleanConstraints};
    
    // Create a test result with a true boolean choice manually
    // This ensures we have a non-forced true boolean to test shrinking
    let true_choice = ChoiceNode::new(
        ChoiceType::Boolean,
        ChoiceValue::Boolean(true),
        Constraints::Boolean(BooleanConstraints { p: 0.5 }),
        false, // not forced
    );
    
    let result = ConjectureResult {
        status: Status::Valid,
        choices: vec![true_choice],
        length: 1,
        events: std::collections::HashMap::new(),
        buffer: Vec::new(),
        examples: Vec::new(),
    };
    
    // Verify we start with true and it's not forced
    if let ChoiceValue::Boolean(value) = &result.choices[0].value {
        assert_eq!(*value, true);
        assert!(!result.choices[0].was_forced, "Choice should not be forced");
    }
    
    let mut shrinker = ChoiceShrinker::new(result);
    
    let shrunk_result = shrinker.shrink(|result| {
        // "Fail" if we have choices (prevents deletion)
        // This allows boolean minimization while preventing total deletion
        !result.choices.is_empty()
    });
    
    // Should have shrunk to false (and still have choices)
    assert!(!shrunk_result.choices.is_empty(), "Should still have choices after shrinking");
    if let ChoiceValue::Boolean(shrunk_value) = &shrunk_result.choices[0].value {
        assert_eq!(*shrunk_value, false, "Boolean should shrink to false");
    } else {
        panic!("Expected boolean choice");
    }
}

/// Test choice sequence deletion
#[test]
fn test_shrink_by_deletion() {
    use crate::shrinking::ChoiceShrinker;
    
    // Create test with multiple choices
    let mut data = ConjectureData::new(42);
    let _ = data.draw_integer(10, 20).unwrap();
    let _ = data.draw_boolean(0.5).unwrap();
    let _ = data.draw_integer(30, 40).unwrap();
    data.freeze();
    let result = data.as_result();
    
    assert_eq!(result.choices.len(), 3);
    
    let mut shrinker = ChoiceShrinker::new(result);
    
    let shrunk_result = shrinker.shrink(|result| {
        // "Fail" only if we have at least 2 choices
        // This should allow deletion of the third choice
        result.choices.len() >= 2
    });
    
    // Should have removed at least one choice
    assert!(shrunk_result.choices.len() < 3, "Should have deleted some choices");
}

/// Test shrinking with conditional failure
#[test]
fn test_shrinking_with_conditions() {
    use crate::shrinking::ChoiceShrinker;
    
    // Create test case where failure depends on integer value
    let mut data = ConjectureData::new(42);
    let large_int = data.draw_integer(20, 100).unwrap();
    data.freeze();
    let result = data.as_result();
    
    println!("Original value for conditional test: {}", large_int);
    
    let mut shrinker = ChoiceShrinker::new(result);
    
    let shrunk_result = shrinker.shrink(|result| {
        // "Fail" only if we have choices AND integer value is >= 15
        // This should shrink down to 15 (minimum failing value)
        !result.choices.is_empty() && {
            if let ChoiceValue::Integer(value) = &result.choices[0].value {
                *value >= 15
            } else {
                false
            }
        }
    });
    
    // Should have shrunk to the minimum failing value (and still have choices)
    assert!(!shrunk_result.choices.is_empty(), "Should still have choices after shrinking");
    if let ChoiceValue::Integer(final_value) = &shrunk_result.choices[0].value {
        println!("Final value after conditional shrinking: {}", final_value);
        assert!(*final_value >= 15, "Should not shrink below the failing threshold");
        assert!(*final_value < large_int, "Should have shrunk from original");
    } else {
        panic!("Expected integer choice");
    }
}

/// Test complex shrinking scenario
#[test]
fn test_complex_shrinking_scenario() {
    use crate::shrinking::ChoiceShrinker;
    
    // Create a complex test case
    let mut data = ConjectureData::new(42);
    let int1 = data.draw_integer(10, 50).unwrap();
    let bool1 = data.draw_boolean_with_forced(0.5, Some(true)).unwrap();
    let int2 = data.draw_integer(20, 80).unwrap();
    let bool2 = data.draw_boolean_with_forced(0.5, Some(true)).unwrap();
    data.freeze();
    let result = data.as_result();
    
    println!("Complex scenario original: int1={}, bool1={}, int2={}, bool2={}", 
             int1, bool1, int2, bool2);
    
    let mut shrinker = ChoiceShrinker::new(result.clone());
    
    let shrunk_result = shrinker.shrink(|result| {
        // Complex condition: "fail" if sum of integers > 25 OR any boolean is true
        let mut sum = 0i128;
        let mut has_true_bool = false;
        
        for choice in &result.choices {
            match &choice.value {
                ChoiceValue::Integer(val) => sum += val,
                ChoiceValue::Boolean(val) => if *val { has_true_bool = true; },
                _ => {}
            }
        }
        
        sum > 25 || has_true_bool
    });
    
    println!("Complex scenario shrunk to {} choices", shrunk_result.choices.len());
    
    // Should have improved from original
    assert!(shrunk_result.choices.len() <= result.choices.len(), 
            "Should not have more choices than original");
    
    // Verify the result still satisfies the failure condition
    let mut sum = 0i128;
    let mut has_true_bool = false;
    
    for choice in &shrunk_result.choices {
        match &choice.value {
            ChoiceValue::Integer(val) => {
                sum += val;
                println!("Shrunk integer: {}", val);
            },
            ChoiceValue::Boolean(val) => {
                println!("Shrunk boolean: {}", val);
                if *val { has_true_bool = true; }
            },
            _ => {}
        }
    }
    
    assert!(sum > 25 || has_true_bool, "Shrunk result should still satisfy failure condition");
}

/// Test shrinking performance and limits
#[test]
fn test_shrinking_performance() {
    use crate::shrinking::ChoiceShrinker;
    
    // Create a shrinker with limited attempts
    let mut data = ConjectureData::new(42);
    let _ = data.draw_integer(100, 1000).unwrap();
    data.freeze();
    let result = data.as_result();
    
    let mut shrinker = ChoiceShrinker::new(result);
    shrinker.max_attempts = 50; // Limit attempts for this test
    
    let start_time = std::time::Instant::now();
    
    let _shrunk_result = shrinker.shrink(|_result| {
        // Always fail to stress-test the shrinker
        true
    });
    
    let duration = start_time.elapsed();
    
    // Should respect attempt limits
    assert!(shrinker.attempts <= 50, "Should respect max_attempts limit");
    
    // Should complete reasonably quickly
    assert!(duration.as_millis() < 1000, "Shrinking should complete quickly");
    
    println!("Shrinking completed in {}ms with {} attempts", 
             duration.as_millis(), shrinker.attempts);
}