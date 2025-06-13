use conjecture::choice::{
    ChoiceType, ChoiceValue, Constraints, 
    IntegerConstraints, BooleanConstraints, FloatConstraints, StringConstraints
};
use conjecture::choice_sequence_management::{
    ChoiceSequenceManager, ChoiceSequenceError
};

#[derive(Debug, PartialEq)]
pub struct VerificationResult {
    pub test_name: String,
    pub expected_behavior: String,
    pub actual_behavior: String,
    pub match_status: bool,
}

pub fn verify_choice_sequence_behaviors() -> Vec<VerificationResult> {
    let mut results = Vec::new();
    
    // Test 1: Basic choice recording functionality
    results.push(verify_basic_choice_recording());
    
    // Test 2: Misalignment detection and recovery
    results.push(verify_misalignment_behavior());
    
    // Test 3: Buffer overflow handling
    results.push(verify_buffer_overflow_behavior());
    
    // Test 4: Choice generation consistency
    results.push(verify_choice_generation_consistency());
    
    // Test 5: Prefix replay functionality
    results.push(verify_prefix_replay());
    
    results
}

fn verify_basic_choice_recording() -> VerificationResult {
    let mut manager = ChoiceSequenceManager::new(10000, 1000, None);
    
    // Record multiple choices
    let int_constraints = Constraints::Integer(IntegerConstraints {
        min_value: Some(0),
        max_value: Some(100),
        weights: None,
        shrink_towards: Some(0),
    });
    let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
    let float_constraints = Constraints::Float(FloatConstraints::default());
    
    let int_result = manager.draw(ChoiceType::Integer, int_constraints, None, true);
    let bool_result = manager.draw(ChoiceType::Boolean, bool_constraints, None, true);
    let float_result = manager.draw(ChoiceType::Float, float_constraints, None, true);
    
    let expected = "3 choices recorded with correct types and sequence";
    let actual = format!("{} choices: {:?}, {:?}, {:?}", 
                        manager.num_choices(),
                        int_result.is_ok(),
                        bool_result.is_ok(), 
                        float_result.is_ok());
    
    let nodes = manager.get_nodes();
    let types_match = nodes.len() == 3 &&
                     nodes[0].choice_type == ChoiceType::Integer &&
                     nodes[1].choice_type == ChoiceType::Boolean &&
                     nodes[2].choice_type == ChoiceType::Float;
    
    VerificationResult {
        test_name: "basic_choice_recording".to_string(),
        expected_behavior: expected.to_string(),
        actual_behavior: actual,
        match_status: types_match && int_result.is_ok() && bool_result.is_ok() && float_result.is_ok(),
    }
}

fn verify_misalignment_behavior() -> VerificationResult {
    // Setup: Record an integer, then replay as boolean (should misalign)
    let mut manager1 = ChoiceSequenceManager::new(10000, 1000, None);
    let int_constraints = Constraints::Integer(IntegerConstraints {
        min_value: Some(10),
        max_value: Some(20),
        weights: None,
        shrink_towards: Some(10),
    });
    
    let _int_choice = manager1.draw(ChoiceType::Integer, int_constraints.clone(), None, true).unwrap();
    
    // Extract prefix for replay
    let prefix: Vec<ChoiceValue> = manager1.get_nodes().iter()
        .map(|node| node.value.clone())
        .collect();
    
    // Replay with different type - should detect misalignment
    let mut manager2 = ChoiceSequenceManager::new(10000, 1000, Some(prefix));
    let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
    
    let bool_result = manager2.draw(ChoiceType::Boolean, bool_constraints, None, true);
    let misalignment = manager2.get_misalignment();
    
    let expected = "Misalignment detected at index 0, boolean choice generated as fallback";
    let actual = format!("Misaligned: {}, Result: {:?}, Index: {:?}", 
                        misalignment.is_some(),
                        bool_result,
                        misalignment.map(|m| m.index));
    
    let correct_behavior = misalignment.is_some() && 
                          misalignment.unwrap().index == 0 &&
                          bool_result.is_ok() &&
                          bool_result.unwrap() == ChoiceValue::Boolean(false); // Simplest choice
    
    VerificationResult {
        test_name: "misalignment_detection".to_string(),
        expected_behavior: expected.to_string(),
        actual_behavior: actual,
        match_status: correct_behavior,
    }
}

fn verify_buffer_overflow_behavior() -> VerificationResult {
    let small_buffer = 50; // Very small buffer
    let mut manager = ChoiceSequenceManager::new(small_buffer, 1000, None);
    
    let int_constraints = Constraints::Integer(IntegerConstraints::default());
    
    // Try to fill buffer beyond capacity
    let mut successful_draws = 0;
    let mut hit_overflow = false;
    
    for _ in 0..20 { // Attempt more draws than buffer can handle
        match manager.draw(ChoiceType::Integer, int_constraints.clone(), None, true) {
            Ok(_) => successful_draws += 1,
            Err(ChoiceSequenceError::BufferOverflow { .. }) => {
                hit_overflow = true;
                break;
            },
            Err(_) => break,
        }
    }
    
    let expected = "Buffer overflow detected, manager marked as overrun";
    let actual = format!("Successful draws: {}, Hit overflow: {}, Is overrun: {}", 
                        successful_draws, hit_overflow, manager.is_overrun());
    
    // Should have hit overflow and marked overrun state
    let correct_behavior = hit_overflow && manager.is_overrun();
    
    VerificationResult {
        test_name: "buffer_overflow".to_string(),
        expected_behavior: expected.to_string(),
        actual_behavior: actual,
        match_status: correct_behavior,
    }
}

fn verify_choice_generation_consistency() -> VerificationResult {
    let manager = ChoiceSequenceManager::new(1000, 100, None);
    
    // Test that choice_from_index(0) generates consistent "simplest" choices
    let int_constraints = Constraints::Integer(IntegerConstraints {
        min_value: Some(5),
        max_value: Some(15),
        weights: None,
        shrink_towards: Some(5),
    });
    let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
    let string_constraints = Constraints::String(StringConstraints {
        min_size: 3,
        max_size: 10,
        intervals: conjecture::choice::IntervalSet::default(),
    });
    
    let int_choice = manager.choice_from_index(0, ChoiceType::Integer, &int_constraints).unwrap();
    let bool_choice = manager.choice_from_index(0, ChoiceType::Boolean, &bool_constraints).unwrap();
    let string_choice = manager.choice_from_index(0, ChoiceType::String, &string_constraints).unwrap();
    
    let expected = "Integer: 5 (min), Boolean: false, String: 'aaa' (min length)";
    let actual = format!("Integer: {:?}, Boolean: {:?}, String: {:?}", 
                        int_choice, bool_choice, string_choice);
    
    let correct_choices = int_choice == ChoiceValue::Integer(5) &&
                         bool_choice == ChoiceValue::Boolean(false) &&
                         string_choice == ChoiceValue::String("aaa".to_string());
    
    VerificationResult {
        test_name: "choice_generation_consistency".to_string(),
        expected_behavior: expected.to_string(),
        actual_behavior: actual,
        match_status: correct_choices,
    }
}

fn verify_prefix_replay() -> VerificationResult {
    // Setup: Record a sequence of choices
    let mut manager1 = ChoiceSequenceManager::new(10000, 1000, None);
    
    let int_constraints = Constraints::Integer(IntegerConstraints {
        min_value: Some(0),
        max_value: Some(100),
        weights: None,
        shrink_towards: Some(0),
    });
    let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
    
    let original_int = manager1.draw(ChoiceType::Integer, int_constraints.clone(), Some(ChoiceValue::Integer(42)), true).unwrap();
    let original_bool = manager1.draw(ChoiceType::Boolean, bool_constraints.clone(), Some(ChoiceValue::Boolean(true)), true).unwrap();
    
    // Extract prefix
    let prefix: Vec<ChoiceValue> = manager1.get_nodes().iter()
        .map(|node| node.value.clone())
        .collect();
    
    // Replay the exact sequence
    let mut manager2 = ChoiceSequenceManager::new(10000, 1000, Some(prefix));
    
    let replayed_int = manager2.draw(ChoiceType::Integer, int_constraints, None, true).unwrap();
    let replayed_bool = manager2.draw(ChoiceType::Boolean, bool_constraints, None, true).unwrap();
    
    let expected = "Original: (42, true), Replayed: (42, true) - exact match";
    let actual = format!("Original: ({:?}, {:?}), Replayed: ({:?}, {:?})", 
                        original_int, original_bool, replayed_int, replayed_bool);
    
    let exact_replay = original_int == replayed_int && 
                      original_bool == replayed_bool &&
                      manager2.get_misalignment().is_none();
    
    VerificationResult {
        test_name: "prefix_replay".to_string(),
        expected_behavior: expected.to_string(),
        actual_behavior: actual,
        match_status: exact_replay,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_choice_sequence_verification() {
        let results = verify_choice_sequence_behaviors();
        
        println!("\n=== Choice Sequence Management Verification ===");
        
        for result in &results {
            println!("\n--- {} ---", result.test_name);
            println!("Expected: {}", result.expected_behavior);
            println!("Actual:   {}", result.actual_behavior);
            println!("Status:   {}", if result.match_status { "✅ PASS" } else { "❌ FAIL" });
        }
        
        let all_passed = results.iter().all(|r| r.match_status);
        println!("\n=== Summary ===");
        println!("Tests passed: {}/{}", results.iter().filter(|r| r.match_status).count(), results.len());
        
        if !all_passed {
            println!("❌ Some verification tests failed");
            for result in results.iter().filter(|r| !r.match_status) {
                println!("  - {}: {}", result.test_name, result.actual_behavior);
            }
        } else {
            println!("✅ All verification tests passed");
        }
        
        assert!(all_passed, "Verification tests failed");
    }
}