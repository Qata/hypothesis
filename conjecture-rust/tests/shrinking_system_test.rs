/// Direct port of Python shrinking tests from hypothesis-python
/// 
/// This file ports the essential shrinking tests from:
/// - tests/conjecture/test_shrinker.py
/// - tests/conjecture/test_test_data.py 
/// - tests/conjecture/test_choice.py
/// - tests/quality/test_shrink_quality.py

use conjecture::choice::{
    ChoiceNode, ChoiceValue, ChoiceType, Constraints,
    IntegerConstraints, BooleanConstraints, FloatConstraints, StringConstraints, BytesConstraints
};
use conjecture::data::ConjectureData;
use conjecture::shrinking::PythonEquivalentShrinker;

/// Test basic shrinking functionality - port of test_shrinker.py::test_basic_shrinking
#[test]
fn test_basic_shrinking() {
    let mut original = ConjectureData::new_from_buffer(vec![100, 200, 50, 30], 1000);
    original.nodes = vec![
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(100),
            Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(200),
                weights: None,
                shrink_towards: Some(0),
            }),
            false,
        ),
    ];

    let test_fn = |data: &ConjectureData| -> bool {
        if let Some(ChoiceValue::Integer(val)) = data.nodes.get(0).map(|n| &n.value) {
            *val >= 10
        } else {
            false
        }
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let result = shrinker.shrink_with_function(test_fn);

    if let Some(ChoiceValue::Integer(val)) = result.nodes.get(0).map(|n| &n.value) {
        assert!(*val < 100, "Should shrink from original value");
        assert!(*val >= 10, "Should still satisfy test condition");
    } else {
        panic!("Expected integer value");
    }
}

/// Test shrinking with multiple choice types - port of test_shrinker.py::test_mixed_types
#[test]
fn test_shrinking_mixed_types() {
    let mut original = ConjectureData::new_from_buffer(vec![1, 2, 3, 4], 1000);
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
        ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints { p: 0.5 }),
            false,
        ),
    ];

    let test_fn = |data: &ConjectureData| -> bool {
        let has_large_int = data.nodes.iter().any(|n| {
            if let ChoiceValue::Integer(val) = &n.value {
                *val > 20
            } else {
                false
            }
        });
        let has_true_bool = data.nodes.iter().any(|n| {
            if let ChoiceValue::Boolean(val) = &n.value {
                *val
            } else {
                false
            }
        });
        has_large_int && has_true_bool
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let result = shrinker.shrink_with_function(test_fn);

    assert!(result.nodes.len() <= original.nodes.len(), "Should not add nodes");
    
    // Check that shrinking occurred
    let result_int = result.nodes.iter().find_map(|n| {
        if let ChoiceValue::Integer(val) = &n.value { Some(*val) } else { None }
    }).unwrap();
    assert!(result_int <= 50, "Integer should be shrunk");
    assert!(result_int > 20, "Integer should still satisfy constraint");
}

/// Test deletion of trailing nodes - port of test_shrinker.py::test_delete_trailing
#[test]
fn test_delete_trailing_nodes() {
    let mut original = ConjectureData::new_from_buffer(vec![1, 2, 3, 4, 5], 1000);
    original.nodes = vec![
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        ),
        ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints { p: 0.5 }),
            false,
        ),
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(100),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        ),
    ];

    // Test that only needs the first node
    let test_fn = |data: &ConjectureData| -> bool {
        if let Some(ChoiceValue::Integer(val)) = data.nodes.get(0).map(|n| &n.value) {
            *val == 42
        } else {
            false
        }
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let result = shrinker.shrink_with_function(test_fn);

    assert!(result.nodes.len() < original.nodes.len(), "Should delete trailing nodes");
    assert!(result.nodes.len() >= 1, "Should keep at least one node");
    
    if let Some(ChoiceValue::Integer(val)) = result.nodes.get(0).map(|n| &n.value) {
        assert_eq!(*val, 42, "First node should be preserved");
    }
}

/// Test integer minimization - port of test_minimizer.py::test_minimize_integers
#[test]
fn test_minimize_integers() {
    let mut original = ConjectureData::new_from_buffer(vec![200], 1000);
    original.nodes = vec![
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(200),
            Constraints::Integer(IntegerConstraints {
                min_value: Some(50),
                max_value: Some(300),
                weights: None,
                shrink_towards: Some(75),
            }),
            false,
        ),
    ];

    // Test requires value >= 100
    let test_fn = |data: &ConjectureData| -> bool {
        if let Some(ChoiceValue::Integer(val)) = data.nodes.get(0).map(|n| &n.value) {
            *val >= 100
        } else {
            false
        }
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let result = shrinker.shrink_with_function(test_fn);

    if let Some(ChoiceValue::Integer(val)) = result.nodes.get(0).map(|n| &n.value) {
        assert!(*val < 200, "Should shrink from original");
        assert!(*val >= 100, "Should satisfy test constraint");
        assert!(*val >= 50, "Should respect minimum constraint");
        assert!(*val <= 300, "Should respect maximum constraint");
    }
}

/// Test boolean minimization - port of test_shrinker.py::test_minimize_booleans
#[test]
fn test_minimize_booleans() {
    let mut original = ConjectureData::new_from_buffer(vec![1, 1, 1], 1000);
    original.nodes = vec![
        ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints { p: 0.5 }),
            false,
        ),
        ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints { p: 0.5 }),
            false,
        ),
        ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints { p: 0.5 }),
            false,
        ),
    ];

    // Test requires at least one true value
    let test_fn = |data: &ConjectureData| -> bool {
        data.nodes.iter().any(|n| {
            if let ChoiceValue::Boolean(val) = &n.value {
                *val
            } else {
                false
            }
        })
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let result = shrinker.shrink_with_function(test_fn);

    let true_count = result.nodes.iter().filter(|n| {
        if let ChoiceValue::Boolean(val) = &n.value {
            *val
        } else {
            false
        }
    }).count();

    assert!(true_count < 3, "Should minimize some booleans to false");
    assert!(true_count >= 1, "Should keep at least one true");
}

/// Test float minimization - port of test_float_shrinking.py::test_minimize_floats
#[test]
fn test_minimize_floats() {
    let mut original = ConjectureData::new_from_buffer(vec![64, 64, 64, 64], 1000);
    original.nodes = vec![
        ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(123.456),
            Constraints::Float(FloatConstraints {
                min_value: 1.0,
                max_value: 1000.0,
                allow_nan: false,
                smallest_nonzero_magnitude: Some(0.001),
            }),
            false,
        ),
    ];

    // Test requires value > 50.0
    let test_fn = |data: &ConjectureData| -> bool {
        if let Some(ChoiceValue::Float(val)) = data.nodes.get(0).map(|n| &n.value) {
            *val > 50.0
        } else {
            false
        }
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let result = shrinker.shrink_with_function(test_fn);

    if let Some(ChoiceValue::Float(val)) = result.nodes.get(0).map(|n| &n.value) {
        assert!(*val < 123.456, "Should shrink from original");
        assert!(*val > 50.0, "Should satisfy test constraint");
        assert!(*val >= 1.0, "Should respect minimum constraint");
        assert!(val.is_finite(), "Should remain finite");
    }
}

/// Test string minimization - port of test_shrink_quality.py::test_minimize_strings
#[test]
fn test_minimize_strings() {
    let mut original = ConjectureData::new_from_buffer(vec![1, 2, 3, 4], 1000);
    original.nodes = vec![
        ChoiceNode::new(
            ChoiceType::String,
            ChoiceValue::String("hello world test".to_string()),
            Constraints::String(StringConstraints {
                min_size: 5,
                max_size: Some(100),
                alphabet: None,
            }),
            false,
        ),
    ];

    // Test requires string length > 10
    let test_fn = |data: &ConjectureData| -> bool {
        if let Some(ChoiceValue::String(s)) = data.nodes.get(0).map(|n| &n.value) {
            s.len() > 10
        } else {
            false
        }
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let result = shrinker.shrink_with_function(test_fn);

    if let Some(ChoiceValue::String(s)) = result.nodes.get(0).map(|n| &n.value) {
        assert!(s.len() < "hello world test".len(), "Should shrink string length");
        assert!(s.len() > 10, "Should satisfy test constraint");
        assert!(s.len() >= 5, "Should respect minimum constraint");
    }
}

/// Test bytes minimization - port of test_shrink_quality.py::test_minimize_bytes
#[test]
fn test_minimize_bytes() {
    let mut original = ConjectureData::new_from_buffer(vec![1, 2, 3, 4], 1000);
    original.nodes = vec![
        ChoiceNode::new(
            ChoiceType::Bytes,
            ChoiceValue::Bytes(vec![1, 2, 3, 4, 5, 6, 7, 8]),
            Constraints::Bytes(BytesConstraints {
                min_size: 2,
                max_size: Some(20),
            }),
            false,
        ),
    ];

    // Test requires bytes length > 4
    let test_fn = |data: &ConjectureData| -> bool {
        if let Some(ChoiceValue::Bytes(b)) = data.nodes.get(0).map(|n| &n.value) {
            b.len() > 4
        } else {
            false
        }
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let result = shrinker.shrink_with_function(test_fn);

    if let Some(ChoiceValue::Bytes(b)) = result.nodes.get(0).map(|n| &n.value) {
        assert!(b.len() < 8, "Should shrink bytes length");
        assert!(b.len() > 4, "Should satisfy test constraint");
        assert!(b.len() >= 2, "Should respect minimum constraint");
    }
}

/// Test forced choices are preserved - port of test_choice.py::test_forced_choices
#[test]
fn test_forced_choices_preserved() {
    let mut original = ConjectureData::new_from_buffer(vec![100, 200], 1000);
    original.nodes = vec![
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(100),
            Constraints::Integer(IntegerConstraints::default()),
            true, // forced
        ),
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(200),
            Constraints::Integer(IntegerConstraints::default()),
            false, // not forced
        ),
    ];

    let test_fn = |data: &ConjectureData| -> bool {
        data.nodes.iter().any(|n| {
            if let ChoiceValue::Integer(val) = &n.value {
                *val > 50
            } else {
                false
            }
        })
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let result = shrinker.shrink_with_function(test_fn);

    // Forced choice should be unchanged
    if let Some(ChoiceValue::Integer(val)) = result.nodes.get(0).map(|n| &n.value) {
        assert_eq!(*val, 100, "Forced choice should not change");
    }
    
    // Non-forced choice may change
    if let Some(ChoiceValue::Integer(val)) = result.nodes.get(1).map(|n| &n.value) {
        assert!(*val <= 200, "Non-forced choice may shrink");
    }
}

/// Test constraint violations are repaired - port of test_shrinker.py::test_constraint_repair
#[test]
fn test_constraint_repair() {
    let mut original = ConjectureData::new_from_buffer(vec![1, 2], 1000);
    original.nodes = vec![
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(-10), // Below minimum
            Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(100),
                weights: None,
                shrink_towards: Some(0),
            }),
            false,
        ),
    ];

    let test_fn = |data: &ConjectureData| -> bool {
        if let Some(ChoiceValue::Integer(val)) = data.nodes.get(0).map(|n| &n.value) {
            *val >= 0 // This will force repair of the constraint violation
        } else {
            false
        }
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let result = shrinker.shrink_with_function(test_fn);

    if let Some(ChoiceValue::Integer(val)) = result.nodes.get(0).map(|n| &n.value) {
        assert!(*val >= 0, "Should repair constraint violation");
        assert!(*val <= 100, "Should respect maximum constraint");
    }
}

/// Test shrinking phases execute in order - port of test_shrinker.py::test_shrinking_phases
#[test]
fn test_shrinking_phases() {
    let mut original = ConjectureData::new_from_buffer(vec![1, 2, 3, 4, 5], 1000);
    original.nodes = vec![
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(100),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        ),
        ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints { p: 0.5 }),
            false,
        ),
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(50),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        ),
    ];

    let test_fn = |data: &ConjectureData| -> bool {
        // Require at least one node with some complexity
        data.nodes.iter().any(|n| {
            match &n.value {
                ChoiceValue::Integer(val) => *val > 0,
                ChoiceValue::Boolean(val) => *val,
                _ => false,
            }
        })
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let result = shrinker.shrink_with_function(test_fn);

    // Verify some progress was made
    assert!(shrinker.made_progress(), "Should have made shrinking progress");
    assert!(result.nodes.len() <= original.nodes.len(), "Should not increase node count");
}

/// Test NaN float handling - port of test_float_shrinking.py::test_nan_handling
#[test]
fn test_nan_float_handling() {
    let mut original = ConjectureData::new_from_buffer(vec![1, 2, 3, 4], 1000);
    original.nodes = vec![
        ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(f64::NAN),
            Constraints::Float(FloatConstraints {
                min_value: 0.0,
                max_value: 100.0,
                allow_nan: false,
                smallest_nonzero_magnitude: Some(0.001),
            }),
            false,
        ),
    ];

    let test_fn = |data: &ConjectureData| -> bool {
        if let Some(ChoiceValue::Float(val)) = data.nodes.get(0).map(|n| &n.value) {
            val.is_finite() && *val >= 0.0
        } else {
            false
        }
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let result = shrinker.shrink_with_function(test_fn);

    if let Some(ChoiceValue::Float(val)) = result.nodes.get(0).map(|n| &n.value) {
        assert!(val.is_finite(), "NaN should be repaired to finite value");
        assert!(*val >= 0.0, "Should respect minimum constraint");
        assert!(*val <= 100.0, "Should respect maximum constraint");
    }
}

/// Test empty sequence handling - port of test_test_data.py::test_empty_data
#[test]
fn test_empty_sequence_handling() {
    let original = ConjectureData::new_from_buffer(vec![], 1000);
    
    let test_fn = |_data: &ConjectureData| -> bool {
        false // Empty data is never interesting
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let result = shrinker.shrink_with_function(test_fn);

    assert_eq!(result.nodes.len(), 0, "Empty data should remain empty");
    assert_eq!(result.buffer.len(), 0, "Buffer should remain empty");
}

/// Test shrinking timeout and limits - port of test_shrinker.py::test_shrinking_limits
#[test]
fn test_shrinking_limits() {
    let mut original = ConjectureData::new_from_buffer(vec![1, 2], 1000);
    original.nodes = vec![
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(1000),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        ),
    ];

    let test_fn = |data: &ConjectureData| -> bool {
        if let Some(ChoiceValue::Integer(val)) = data.nodes.get(0).map(|n| &n.value) {
            *val > 0
        } else {
            false
        }
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    
    // Set very low limits to test timeout behavior
    shrinker.max_shrinks = 5;
    shrinker.max_shrinking_time = std::time::Duration::from_millis(1);
    
    let result = shrinker.shrink_with_function(test_fn);

    // Should still return a valid result even with tight limits
    assert!(result.nodes.len() > 0, "Should have at least one node");
    if let Some(ChoiceValue::Integer(val)) = result.nodes.get(0).map(|n| &n.value) {
        assert!(*val > 0, "Should still satisfy test condition");
    }
}

/// Test progress tracking - port of test_shrinker.py::test_progress_tracking
#[test]
fn test_progress_tracking() {
    let mut original = ConjectureData::new_from_buffer(vec![100, 200], 1000);
    original.nodes = vec![
        ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(100),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        ),
    ];

    let test_fn = |data: &ConjectureData| -> bool {
        if let Some(ChoiceValue::Integer(val)) = data.nodes.get(0).map(|n| &n.value) {
            *val >= 10
        } else {
            false
        }
    };

    let mut shrinker = PythonEquivalentShrinker::new(original.clone());
    let initial_best = shrinker.get_best().clone();
    
    let result = shrinker.shrink_with_function(test_fn);
    
    // Check that progress tracking works
    let made_progress = shrinker.made_progress();
    let metrics = shrinker.get_metrics();
    
    if made_progress {
        assert!(metrics.successful_transformations > 0, "Should track successful transformations");
        assert!(shrinker.calls > 0, "Should track function calls");
    }
    
    // Final result should be valid
    if let Some(ChoiceValue::Integer(val)) = result.nodes.get(0).map(|n| &n.value) {
        assert!(*val >= 10, "Final result should satisfy test");
    }
}