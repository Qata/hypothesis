//! Test the ConjectureData Lifecycle Management implementation
//! This is a standalone test to verify the core functionality works correctly

use conjecture::conjecture_data_lifecycle_management::*;
use conjecture::choice::*;
use conjecture::data::ConjectureData;

fn main() {
    println!("Testing ConjectureData Lifecycle Management implementation...");
    
    // Test 1: Basic lifecycle manager creation
    println!("\n1. Testing lifecycle manager creation...");
    let config = LifecycleConfig::default();
    let mut manager = ConjectureDataLifecycleManager::new(config);
    assert_eq!(manager.active_instance_count(), 0);
    println!("✓ Lifecycle manager created successfully");
    
    // Test 2: Create regular instance
    println!("\n2. Testing regular instance creation...");
    let instance_id = manager.create_instance(42, None, None).unwrap();
    assert_eq!(manager.active_instance_count(), 1);
    assert!(manager.is_instance_valid(instance_id));
    println!("✓ Regular instance created: ID {}", instance_id);
    
    // Test 3: Create replay instance and verify max_choices preservation
    println!("\n3. Testing replay instance creation with max_choices preservation...");
    let choices = vec![
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(100),
                weights: None,
                shrink_towards: Some(0),
            }),
            false, // was_forced
        ),
        ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints { p: 0.5 }),
            false, // was_forced
        ),
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(123),
            Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(200),
                weights: None,
                shrink_towards: Some(0),
            }),
            false, // was_forced
        ),
    ];
    
    let replay_id = manager.create_for_replay(&choices, None, None, None).unwrap();
    assert_eq!(manager.active_instance_count(), 2);
    assert_eq!(manager.get_state(replay_id), Some(LifecycleState::Replaying));
    
    // CRITICAL TEST: Verify max_choices preservation
    if let Some(data) = manager.get_instance(replay_id) {
        assert_eq!(data.max_choices, Some(3), 
                  "Replay instance max_choices should be {} (choices.len()), not lifecycle config max_choices", 
                  choices.len());
        println!("✓ Replay instance created with correct max_choices: {:?}", data.max_choices);
    } else {
        panic!("Failed to get replay instance");
    }
    
    // Test 4: Forced values integration
    println!("\n4. Testing forced values integration...");
    let forced_values = vec![
        (0, ChoiceValue::Integer(999)),
        (1, ChoiceValue::Boolean(false)),
    ];
    
    let result = manager.integrate_forced_values(replay_id, forced_values.clone());
    assert!(result.is_ok(), "Failed to integrate forced values: {:?}", result);
    
    assert_eq!(manager.get_forced_values(replay_id), Some(&forced_values));
    println!("✓ Forced values integrated successfully");
    
    // Test 5: Invalid forced values validation
    println!("\n5. Testing forced values validation...");
    let invalid_forced = vec![(3, ChoiceValue::Integer(123))]; // index 3 >= max_choices (3)
    let result = manager.integrate_forced_values(replay_id, invalid_forced);
    assert!(result.is_err(), "Should reject invalid forced value index");
    if let Err(LifecycleError::ForcedValueError { details }) = result {
        assert!(details.contains("exceeds max_choices"));
        println!("✓ Invalid forced values correctly rejected: {}", details);
    } else {
        panic!("Expected ForcedValueError for invalid index");
    }
    
    // Test 6: State transitions
    println!("\n6. Testing state transitions...");
    assert_eq!(manager.get_state(replay_id), Some(LifecycleState::Replaying));
    
    manager.transition_state(replay_id, LifecycleState::Executing).unwrap();
    assert_eq!(manager.get_state(replay_id), Some(LifecycleState::Executing));
    
    manager.transition_state(replay_id, LifecycleState::Completed).unwrap();
    assert_eq!(manager.get_state(replay_id), Some(LifecycleState::Completed));
    println!("✓ State transitions working correctly");
    
    // Test 7: Metrics tracking
    println!("\n7. Testing metrics tracking...");
    let metrics = manager.get_metrics();
    assert_eq!(metrics.instances_created, 2);
    assert_eq!(metrics.forced_value_integrations, 1);
    println!("✓ Metrics tracked correctly: {} instances, {} forced integrations", 
             metrics.instances_created, metrics.forced_value_integrations);
    
    // Test 8: Cleanup
    println!("\n8. Testing cleanup...");
    manager.cleanup_instance(instance_id).unwrap();
    manager.cleanup_instance(replay_id).unwrap();
    assert_eq!(manager.active_instance_count(), 0);
    
    let final_metrics = manager.get_metrics();
    assert_eq!(final_metrics.cleanup_operations, 2);
    println!("✓ Cleanup completed successfully");
    
    // Test 9: Status report
    println!("\n9. Testing status report generation...");
    let report = manager.generate_status_report();
    assert!(report.contains("Total instances created: 2"));
    assert!(report.contains("Cleanup operations: 2"));
    println!("✓ Status report generated successfully");
    
    println!("\n🎉 All tests passed! ConjectureData Lifecycle Management implementation is working correctly.");
    println!("\nKey fixes implemented:");
    println!("• Fixed max_choices override issue in create_for_replay()");
    println!("• Enhanced forced value system integration with validation");
    println!("• Comprehensive replay mechanism with state management");
    println!("• Proper lifecycle tracking and cleanup");
}