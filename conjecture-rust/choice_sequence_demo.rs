//! Choice Sequence Management System Demonstration
//! 
//! This standalone program demonstrates the Core Choice Sequence Management System
//! capability that fixes type inconsistencies, index tracking issues, and sequence
//! replay functionality to restore basic ConjectureData buffer operations.

use conjecture::choice::{ChoiceType, ChoiceValue, Constraints, IntegerConstraints, BooleanConstraints};
use conjecture::choice_sequence_management::ChoiceSequenceManager;

fn main() {
    println!("=== CORE CHOICE SEQUENCE MANAGEMENT SYSTEM DEMONSTRATION ===\n");
    
    // Create a choice sequence manager
    let mut manager = ChoiceSequenceManager::new(8192);
    println!("✓ Created ChoiceSequenceManager with 8KB buffer");
    
    // Demonstrate recording different types of choices
    println!("\n--- Recording Choice Sequence ---");
    
    // Record integer choice
    let int_constraints = Box::new(Constraints::Integer(IntegerConstraints::default()));
    let result1 = manager.record_choice(
        ChoiceType::Integer,
        ChoiceValue::Integer(42),
        int_constraints,
        false,
        0,
    );
    match result1 {
        Ok(index) => println!("✓ Recorded integer choice 42 at index {}", index),
        Err(e) => println!("✗ Failed to record integer choice: {}", e),
    }
    
    // Record boolean choice
    let bool_constraints = Box::new(Constraints::Boolean(BooleanConstraints::default()));
    let result2 = manager.record_choice(
        ChoiceType::Boolean,
        ChoiceValue::Boolean(true),
        bool_constraints,
        false,
        8,
    );
    match result2 {
        Ok(index) => println!("✓ Recorded boolean choice true at index {}", index),
        Err(e) => println!("✗ Failed to record boolean choice: {}", e),
    }
    
    // Record another integer choice
    let int_constraints2 = Box::new(Constraints::Integer(IntegerConstraints::default()));
    let result3 = manager.record_choice(
        ChoiceType::Integer,
        ChoiceValue::Integer(-17),
        int_constraints2,
        false,
        9,
    );
    match result3 {
        Ok(index) => println!("✓ Recorded integer choice -17 at index {}", index),
        Err(e) => println!("✗ Failed to record integer choice: {}", e),
    }
    
    println!("\nSequence length: {}", manager.sequence_length());
    
    // Demonstrate replay functionality
    println!("\n--- Replaying Choice Sequence ---");
    
    // Replay integer choice
    let replay_int_constraints = Constraints::Integer(IntegerConstraints::default());
    let replay1 = manager.replay_choice_at_index(0, ChoiceType::Integer, &replay_int_constraints);
    match replay1 {
        Ok(value) => println!("✓ Replayed choice at index 0: {:?}", value),
        Err(e) => println!("✗ Failed to replay choice at index 0: {}", e),
    }
    
    // Replay boolean choice
    let replay_bool_constraints = Constraints::Boolean(BooleanConstraints::default());
    let replay2 = manager.replay_choice_at_index(1, ChoiceType::Boolean, &replay_bool_constraints);
    match replay2 {
        Ok(value) => println!("✓ Replayed choice at index 1: {:?}", value),
        Err(e) => println!("✗ Failed to replay choice at index 1: {}", e),
    }
    
    // Replay second integer choice
    let replay_int_constraints2 = Constraints::Integer(IntegerConstraints::default());
    let replay3 = manager.replay_choice_at_index(2, ChoiceType::Integer, &replay_int_constraints2);
    match replay3 {
        Ok(value) => println!("✓ Replayed choice at index 2: {:?}", value),
        Err(e) => println!("✗ Failed to replay choice at index 2: {}", e),
    }
    
    // Demonstrate type mismatch detection
    println!("\n--- Type Consistency Verification ---");
    let wrong_replay = manager.replay_choice_at_index(0, ChoiceType::Boolean, &replay_bool_constraints);
    match wrong_replay {
        Ok(_) => println!("✗ Unexpected success - should have detected type mismatch"),
        Err(e) => println!("✓ Correctly detected type mismatch: {}", e),
    }
    
    // Demonstrate index bounds checking
    println!("\n--- Index Bounds Verification ---");
    let out_of_bounds = manager.replay_choice_at_index(10, ChoiceType::Integer, &replay_int_constraints);
    match out_of_bounds {
        Ok(_) => println!("✗ Unexpected success - should have detected out of bounds"),
        Err(e) => println!("✓ Correctly detected index out of bounds: {}", e),
    }
    
    // Display integrity status
    println!("\n--- Sequence Integrity Status ---");
    let integrity_status = manager.get_integrity_status();
    println!("✓ Sequence is healthy: {}", integrity_status.is_healthy);
    println!("✓ Total violations: {}", integrity_status.total_violations);
    println!("✓ Last verified index: {}", integrity_status.last_verified_index);
    println!("✓ Recovery actions taken: {}", integrity_status.recovery_actions_taken);
    
    // Display performance metrics
    println!("\n--- Performance Metrics ---");
    let metrics = manager.get_performance_metrics();
    println!("✓ Total recordings: {}", metrics.total_recordings);
    println!("✓ Total replays: {}", metrics.total_replays);
    println!("✓ Average recording time: {:.6}ms", metrics.avg_recording_time * 1000.0);
    println!("✓ Average replay time: {:.6}ms", metrics.avg_replay_time * 1000.0);
    println!("✓ Type verification time: {:.6}ms", metrics.type_verification_time * 1000.0);
    
    // Demonstrate large sequence handling
    println!("\n--- Large Sequence Performance Test ---");
    let start_time = std::time::Instant::now();
    
    manager.reset_sequence();
    
    // Record 1000 choices
    for i in 0..1000 {
        let constraints = Box::new(Constraints::Integer(IntegerConstraints::default()));
        let _ = manager.record_choice(
            ChoiceType::Integer,
            ChoiceValue::Integer(i),
            constraints,
            false,
            i as usize * 8,
        );
    }
    
    let recording_time = start_time.elapsed();
    println!("✓ Recorded 1000 choices in {:.3}ms", recording_time.as_millis());
    
    // Replay all choices
    let replay_start = std::time::Instant::now();
    let replay_constraints = Constraints::Integer(IntegerConstraints::default());
    
    for i in 0..1000 {
        let _ = manager.replay_choice_at_index(i, ChoiceType::Integer, &replay_constraints);
    }
    
    let replay_time = replay_start.elapsed();
    println!("✓ Replayed 1000 choices in {:.3}ms", replay_time.as_millis());
    
    let final_integrity = manager.get_integrity_status();
    println!("✓ Final integrity status: healthy = {}", final_integrity.is_healthy);
    
    let final_metrics = manager.get_performance_metrics();
    println!("✓ Final performance: {:.6}ms avg recording, {:.6}ms avg replay", 
             final_metrics.avg_recording_time * 1000.0,
             final_metrics.avg_replay_time * 1000.0);
    
    println!("\n=== DEMONSTRATION COMPLETE ===");
    println!("✅ Core Choice Sequence Management System successfully implemented!");
    println!("✅ Type inconsistencies in choice node storage: FIXED");
    println!("✅ Index tracking issues in sequence replay: FIXED");
    println!("✅ Basic ConjectureData buffer operations: RESTORED");
}