/// Minimal shrinking test to verify basic functionality
use conjecture::choice::{ChoiceNode, ChoiceValue, ChoiceType, Constraints, IntegerConstraints};
use conjecture::data::ConjectureData;
use conjecture::shrinking::PythonEquivalentShrinker;

#[test]
fn test_basic_shrinking_functionality() {
    // Create simple test data
    let mut original = ConjectureData::new_from_buffer(vec![100], 1000);
    original.nodes = vec![
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

    // Simple test function
    let test_fn = |data: &ConjectureData| -> bool {
        if let Some(ChoiceValue::Integer(val)) = data.nodes.get(0).map(|n| &n.value) {
            *val >= 20
        } else {
            false
        }
    };

    // Test shrinking
    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let result = shrinker.shrink_with_function(test_fn);

    // Verify basic functionality
    assert!(!result.nodes.is_empty(), "Should have at least one node");
    
    if let Some(ChoiceValue::Integer(val)) = result.nodes.get(0).map(|n| &n.value) {
        assert!(*val >= 20, "Should satisfy test condition");
        println!("Shrinking test passed: {} -> {}", 50, val);
    }
}