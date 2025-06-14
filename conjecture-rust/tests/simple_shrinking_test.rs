/// Minimal shrinking test to verify basic functionality
use conjecture::choice::{ChoiceNode, ChoiceValue, ChoiceType, Constraints, IntegerConstraints};
use conjecture::data::ConjectureData;
use conjecture::shrinking::Shrinker;

#[test]
fn test_basic_shrinking_functionality() {
    // Create simple test data using draw operations to populate nodes
    let mut data = ConjectureData::new(1);
    
    // Draw an integer value to create a choice node
    let _value = data.draw_integer(
        Some(0),    // min_value
        Some(100),  // max_value
        None,       // weights
        0,          // shrink_towards
        Some(50),   // forced value
        false       // endpoint
    ).unwrap();

    // Simple test function
    let test_fn = |data: &ConjectureData| -> bool {
        if let Some(ChoiceValue::Integer(val)) = data.get_nodes().get(0).map(|n| &n.value) {
            val >= &20
        } else {
            false
        }
    };

    // Test shrinking
    let mut shrinker = Shrinker::new(data, Box::new(test_fn));
    let result = shrinker.shrink();

    // Verify basic functionality
    assert!(!result.get_nodes().is_empty(), "Should have at least one node");
    
    if let Some(ChoiceValue::Integer(val)) = result.get_nodes().get(0).map(|n| &n.value) {
        assert!(val >= &20, "Should satisfy test condition");
        println!("Shrinking test passed: 50 -> {}", val);
    }
}