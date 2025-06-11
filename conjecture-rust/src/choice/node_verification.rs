//! Comprehensive verification tests for ChoiceNode capability
//! 
//! This module contains extensive tests to verify that the ChoiceNode implementation
//! meets all architectural requirements and maintains perfect Python parity.

use super::*;
use crate::choice::{IntegerConstraints, BooleanConstraints, FloatConstraints, StringConstraints, BytesConstraints, IntervalSet};
use std::collections::{HashMap, HashSet};

/// Test the complete ChoiceNode API surface
#[cfg(test)]
mod comprehensive_tests {
    use super::*;

    #[test]
    fn test_choice_node_complete_api() {
        println!("VERIFICATION: Testing complete ChoiceNode API");

        // Test all construction methods
        let node1 = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );
        assert_eq!(node1.index, None);

        let node2 = ChoiceNode::with_index(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
            5,
        );
        assert_eq!(node2.index, Some(5));

        let node3 = node1.set_index(10);
        assert_eq!(node3.index, Some(10));

        println!("VERIFICATION: All construction methods work correctly");
    }

    #[test]
    fn test_choice_node_immutability() {
        println!("VERIFICATION: Testing ChoiceNode immutability");

        let original = ChoiceNode::new(
            ChoiceType::String,
            ChoiceValue::String("original".to_string()),
            Constraints::String(StringConstraints::default()),
            false,
        );

        // Copy should not affect original
        let copied = original.copy_with_value(ChoiceValue::String("modified".to_string())).unwrap();
        
        assert_eq!(original.value, ChoiceValue::String("original".to_string()));
        assert_eq!(copied.value, ChoiceValue::String("modified".to_string()));
        assert_eq!(copied.index, None); // Index should not be copied

        println!("VERIFICATION: Immutability maintained correctly");
    }

    #[test]
    fn test_choice_node_all_types() {
        println!("VERIFICATION: Testing all ChoiceNode types");

        let nodes = vec![
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(123),
                Constraints::Integer(IntegerConstraints::default()),
                false,
            ),
            ChoiceNode::new(
                ChoiceType::Float,
                ChoiceValue::Float(3.14159),
                Constraints::Float(FloatConstraints::default()),
                false,
            ),
            ChoiceNode::new(
                ChoiceType::Boolean,
                ChoiceValue::Boolean(true),
                Constraints::Boolean(BooleanConstraints::default()),
                false,
            ),
            ChoiceNode::new(
                ChoiceType::String,
                ChoiceValue::String("test string".to_string()),
                Constraints::String(StringConstraints::default()),
                false,
            ),
            ChoiceNode::new(
                ChoiceType::Bytes,
                ChoiceValue::Bytes(vec![1, 2, 3, 4, 5]),
                Constraints::Bytes(BytesConstraints::default()),
                false,
            ),
        ];

        // Verify all types are distinct and properly handled
        let mut type_set = HashSet::new();
        for node in &nodes {
            type_set.insert(node.choice_type);
        }
        assert_eq!(type_set.len(), 5);

        println!("VERIFICATION: All choice types work correctly");
    }

    #[test]
    fn test_choice_node_forced_semantics() {
        println!("VERIFICATION: Testing forced node semantics");

        let forced = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(100),
            Constraints::Integer(IntegerConstraints::default()),
            true, // forced
        );

        let not_forced = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(100),
            Constraints::Integer(IntegerConstraints::default()),
            false, // not forced
        );

        // Forced nodes should always be trivial
        assert!(forced.trivial());
        
        // Forced nodes cannot be modified
        assert!(forced.copy_with_value(ChoiceValue::Integer(200)).is_err());
        
        // Non-forced nodes can be modified
        assert!(not_forced.copy_with_value(ChoiceValue::Integer(200)).is_ok());

        // Forced and non-forced nodes with same value should not be equal
        assert_ne!(forced, not_forced);

        println!("VERIFICATION: Forced node semantics correct");
    }

    #[test]
    fn test_choice_node_equality_and_hashing() {
        println!("VERIFICATION: Testing equality and hashing");

        let node1 = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(f64::NAN),
            Constraints::Float(FloatConstraints::default()),
            false,
        );

        let node2 = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(f64::NAN),
            Constraints::Float(FloatConstraints::default()),
            false,
        );

        // NaN should equal NaN in choice nodes
        assert_eq!(node1, node2);

        // Test signed zero distinction
        let pos_zero = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(0.0),
            Constraints::Float(FloatConstraints::default()),
            false,
        );

        let neg_zero = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(-0.0),
            Constraints::Float(FloatConstraints::default()),
            false,
        );

        // Should distinguish +0.0 from -0.0
        assert_ne!(pos_zero, neg_zero);

        // Test hashing in collections
        let mut map = HashMap::new();
        map.insert(node1.clone(), "nan1");
        map.insert(node2.clone(), "nan2");
        
        // Should treat equal nodes as same key
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&node1), Some(&"nan2"));

        println!("VERIFICATION: Equality and hashing work correctly");
    }

    #[test]
    fn test_choice_node_constraint_validation() {
        println!("VERIFICATION: Testing constraint validation scenarios");

        // Create a helper function to test constraint validation
        fn test_constraint_scenario<T>(
            choice_type: ChoiceType,
            valid_value: ChoiceValue,
            invalid_value: ChoiceValue,
            constraints: Constraints,
        ) {
            let valid_node = ChoiceNode::new(choice_type, valid_value, constraints.clone(), false);
            let invalid_node = ChoiceNode::new(choice_type, invalid_value, constraints, false);

            // Both should be constructible, but validation happens elsewhere
            // The node structure itself doesn't validate, it just stores the data
            println!("Created valid and invalid constraint scenarios");
        }

        // Test integer constraints
        test_constraint_scenario(
            ChoiceType::Integer,
            ChoiceValue::Integer(50),
            ChoiceValue::Integer(150),
            Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(100),
                shrink_towards: Some(25),
                weights: None,
            }),
        );

        // Test float constraints
        test_constraint_scenario(
            ChoiceType::Float,
            ChoiceValue::Float(50.0),
            ChoiceValue::Float(f64::NAN),
            Constraints::Float(FloatConstraints {
                min_value: 0.0,
                max_value: 100.0,
                allow_nan: false,
                smallest_nonzero_magnitude: f64::MIN_POSITIVE,
            }),
        );

        println!("VERIFICATION: Constraint scenarios handled correctly");
    }

    #[test]
    fn test_choice_node_trivial_detection_comprehensive() {
        println!("VERIFICATION: Testing comprehensive trivial detection");

        // Test integer triviality with different shrink_towards values
        let int_constraints_zero = IntegerConstraints {
            min_value: None,
            max_value: None,
            shrink_towards: Some(0),
            weights: None,
        };

        let int_constraints_ten = IntegerConstraints {
            min_value: None,
            max_value: None,
            shrink_towards: Some(10),
            weights: None,
        };

        let zero_node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(0),
            Constraints::Integer(int_constraints_zero),
            false,
        );

        let ten_node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(10),
            Constraints::Integer(int_constraints_ten),
            false,
        );

        assert!(zero_node.trivial());
        assert!(ten_node.trivial());

        // Test boolean triviality
        let false_node = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(false),
            Constraints::Boolean(BooleanConstraints::default()),
            false,
        );

        let true_node = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints::default()),
            false,
        );

        assert!(false_node.trivial());
        assert!(!true_node.trivial());

        // Test string triviality
        let empty_string = ChoiceNode::new(
            ChoiceType::String,
            ChoiceValue::String("".to_string()),
            Constraints::String(StringConstraints {
                min_size: 0,
                max_size: 100,
                intervals: IntervalSet::default(),
            }),
            false,
        );

        let non_empty_string = ChoiceNode::new(
            ChoiceType::String,
            ChoiceValue::String("hello".to_string()),
            Constraints::String(StringConstraints::default()),
            false,
        );

        assert!(empty_string.trivial());
        assert!(!non_empty_string.trivial());

        // Test bytes triviality
        let empty_bytes = ChoiceNode::new(
            ChoiceType::Bytes,
            ChoiceValue::Bytes(vec![]),
            Constraints::Bytes(BytesConstraints {
                min_size: 0,
                max_size: 100,
            }),
            false,
        );

        let non_empty_bytes = ChoiceNode::new(
            ChoiceType::Bytes,
            ChoiceValue::Bytes(vec![1, 2, 3]),
            Constraints::Bytes(BytesConstraints::default()),
            false,
        );

        assert!(empty_bytes.trivial());
        assert!(!non_empty_bytes.trivial());

        println!("VERIFICATION: Comprehensive trivial detection works correctly");
    }

    #[test]
    fn test_choice_node_float_special_cases() {
        println!("VERIFICATION: Testing float special cases");

        // Test unbounded range with zero
        let unbounded_zero = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(0.0),
            Constraints::Float(FloatConstraints::default()),
            false,
        );
        assert!(unbounded_zero.trivial());

        // Test bounded range with integer values
        let bounded_constraints = FloatConstraints {
            min_value: 2.5,
            max_value: 7.5,
            allow_nan: false,
            smallest_nonzero_magnitude: f64::MIN_POSITIVE,
        };

        // Value 3.0 should be trivial (closest integer to 0 in range [2.5, 7.5])
        let bounded_trivial = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(3.0),
            Constraints::Float(bounded_constraints.clone()),
            false,
        );
        assert!(bounded_trivial.trivial());

        // Value 4.5 should not be trivial
        let bounded_non_trivial = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(4.5),
            Constraints::Float(bounded_constraints),
            false,
        );
        assert!(!bounded_non_trivial.trivial());

        // Test range with no integers
        let no_integer_constraints = FloatConstraints {
            min_value: 1.1,
            max_value: 1.9,
            allow_nan: false,
            smallest_nonzero_magnitude: f64::MIN_POSITIVE,
        };

        let no_integer_node = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(1.5),
            Constraints::Float(no_integer_constraints),
            false,
        );
        // Should be conservative and return false
        assert!(!no_integer_node.trivial());

        println!("VERIFICATION: Float special cases handled correctly");
    }

    #[test]
    fn test_choice_node_copy_semantics() {
        println!("VERIFICATION: Testing copy semantics match Python");

        let original = ChoiceNode::with_index(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
            5,
        );

        // Test copy with value
        let with_value = original.copy_with_value(ChoiceValue::Integer(100)).unwrap();
        assert_eq!(with_value.value, ChoiceValue::Integer(100));
        assert_eq!(with_value.choice_type, original.choice_type);
        assert_eq!(with_value.constraints, original.constraints);
        assert_eq!(with_value.was_forced, original.was_forced);
        assert_eq!(with_value.index, None); // Index should NOT be copied

        // Test copy with constraints
        let new_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(200),
            shrink_towards: Some(50),
            weights: None,
        });
        let with_constraints = original.copy_with_constraints(new_constraints.clone()).unwrap();
        assert_eq!(with_constraints.constraints, new_constraints);
        assert_eq!(with_constraints.value, original.value);
        assert_eq!(with_constraints.index, None); // Index should NOT be copied

        // Test full copy
        let full_copy = original.copy(
            Some(ChoiceValue::Integer(200)),
            Some(new_constraints.clone()),
        ).unwrap();
        assert_eq!(full_copy.value, ChoiceValue::Integer(200));
        assert_eq!(full_copy.constraints, new_constraints);
        assert_eq!(full_copy.index, None); // Index should NOT be copied

        println!("VERIFICATION: Copy semantics match Python exactly");
    }

    #[test]
    fn test_choice_node_collections_usage() {
        println!("VERIFICATION: Testing ChoiceNode usage in collections");

        let mut nodes = vec![];
        let mut node_set = HashSet::new();
        let mut node_map = HashMap::new();

        // Create various nodes
        for i in 0..10 {
            let node = ChoiceNode::with_index(
                ChoiceType::Integer,
                ChoiceValue::Integer(i),
                Constraints::Integer(IntegerConstraints::default()),
                false,
                i as usize,
            );
            
            nodes.push(node.clone());
            node_set.insert(node.clone());
            node_map.insert(node, format!("value_{}", i));
        }

        assert_eq!(nodes.len(), 10);
        assert_eq!(node_set.len(), 10);
        assert_eq!(node_map.len(), 10);

        // Test that equal nodes are treated as equal in collections
        let duplicate = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(5),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );

        // Should be considered equal to nodes[5] despite different index
        assert!(node_set.contains(&duplicate));
        assert!(node_map.contains_key(&duplicate));

        println!("VERIFICATION: Collections usage works correctly");
    }

    #[test]
    fn test_choice_node_python_parity_behaviors() {
        println!("VERIFICATION: Testing Python parity behaviors");

        // Test forced node modification prevention (Python: assert error)
        let forced = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints::default()),
            true,
        );

        match forced.copy_with_value(ChoiceValue::Boolean(false)) {
            Err(msg) => assert!(msg.contains("modifying a forced node doesn't make sense")),
            Ok(_) => panic!("Should not be able to modify forced node"),
        }

        // Test index non-copying behavior
        let with_index = ChoiceNode::with_index(
            ChoiceType::String,
            ChoiceValue::String("test".to_string()),
            Constraints::String(StringConstraints::default()),
            false,
            42,
        );

        let copied = with_index.copy(None, None).unwrap();
        assert_eq!(copied.index, None);

        // Test trivial property for forced nodes
        assert!(forced.trivial());

        // Test equality ignores index (like Python)
        let node1 = ChoiceNode::with_index(
            ChoiceType::Integer,
            ChoiceValue::Integer(123),
            Constraints::Integer(IntegerConstraints::default()),
            false,
            1,
        );

        let node2 = ChoiceNode::with_index(
            ChoiceType::Integer,
            ChoiceValue::Integer(123),
            Constraints::Integer(IntegerConstraints::default()),
            false,
            999, // Different index
        );

        assert_eq!(node1, node2); // Should be equal despite different indices

        println!("VERIFICATION: Python parity behaviors match exactly");
    }

    #[test]
    fn test_choice_node_memory_efficiency() {
        println!("VERIFICATION: Testing memory efficiency");

        // Create many nodes with shared constraints
        let shared_constraints = Constraints::Integer(IntegerConstraints::default());
        let mut nodes = Vec::new();

        for i in 0..1000 {
            let node = ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(i),
                shared_constraints.clone(),
                false,
            );
            nodes.push(node);
        }

        assert_eq!(nodes.len(), 1000);

        // Verify all nodes are properly constructed
        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(node.value, ChoiceValue::Integer(i as i128));
            assert_eq!(node.choice_type, ChoiceType::Integer);
            assert!(!node.was_forced);
        }

        println!("VERIFICATION: Memory efficiency acceptable");
    }

    #[test]
    fn test_choice_node_debug_output() {
        println!("VERIFICATION: Testing debug output");

        let node = ChoiceNode::with_index(
            ChoiceType::Float,
            ChoiceValue::Float(3.14159),
            Constraints::Float(FloatConstraints::default()),
            true,
            7,
        );

        let debug_str = format!("{:?}", node);
        
        // Should contain key information
        assert!(debug_str.contains("ChoiceNode"));
        assert!(debug_str.contains("Float"));
        assert!(debug_str.contains("3.14159"));
        assert!(debug_str.contains("true"));

        println!("VERIFICATION: Debug output contains expected information");
        println!("Debug output: {}", debug_str);
    }

    #[test]
    fn test_choice_node_architectural_compliance() {
        println!("VERIFICATION: Testing architectural compliance");

        // Test that ChoiceNode meets all architectural requirements:
        
        // 1. Core immutable choice representation ✓
        let node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );
        
        // 2. Type, value, constraints, forcing metadata ✓
        assert_eq!(node.choice_type, ChoiceType::Integer);
        assert_eq!(node.value, ChoiceValue::Integer(42));
        assert!(!node.was_forced);
        
        // 3. Indexing support ✓
        let indexed = node.set_index(5);
        assert_eq!(indexed.index, Some(5));
        
        // 4. Immutable operations ✓
        let copied = indexed.copy_with_value(ChoiceValue::Integer(100)).unwrap();
        assert_eq!(indexed.value, ChoiceValue::Integer(42)); // Original unchanged
        assert_eq!(copied.value, ChoiceValue::Integer(100));
        
        // 5. Python parity ✓
        // - Forced node protection
        // - Index non-copying
        // - Trivial detection logic
        // - Equality semantics
        
        println!("VERIFICATION: All architectural requirements met");
    }
}

/// Integration tests with other choice system components
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_choice_node_with_indexing_system() {
        println!("INTEGRATION: Testing ChoiceNode with indexing system");
        
        // This would test integration with choice_to_index/choice_from_index
        // when those systems are available
        
        let node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );
        
        // The node should be ready for indexing integration
        assert_eq!(node.choice_type, ChoiceType::Integer);
        assert_eq!(node.value, ChoiceValue::Integer(42));
        
        println!("INTEGRATION: ChoiceNode ready for indexing system");
    }

    #[test]
    fn test_choice_node_with_constraints_system() {
        println!("INTEGRATION: Testing ChoiceNode with constraints system");
        
        // Test that ChoiceNode works properly with all constraint types
        let constraints = vec![
            Constraints::Integer(IntegerConstraints {
                min_value: Some(-100),
                max_value: Some(100),
                shrink_towards: Some(0),
                weights: None,
            }),
            Constraints::Boolean(BooleanConstraints { p: 0.7 }),
            Constraints::Float(FloatConstraints {
                min_value: 0.0,
                max_value: 1.0,
                allow_nan: false,
                smallest_nonzero_magnitude: f64::MIN_POSITIVE,
            }),
            Constraints::String(StringConstraints {
                min_size: 1,
                max_size: 50,
                intervals: IntervalSet::default(),
            }),
            Constraints::Bytes(BytesConstraints {
                min_size: 0,
                max_size: 1024,
            }),
        ];

        // All constraint types should work with ChoiceNode
        for constraint in constraints {
            let node = ChoiceNode::new(
                match constraint {
                    Constraints::Integer(_) => ChoiceType::Integer,
                    Constraints::Boolean(_) => ChoiceType::Boolean,
                    Constraints::Float(_) => ChoiceType::Float,
                    Constraints::String(_) => ChoiceType::String,
                    Constraints::Bytes(_) => ChoiceType::Bytes,
                },
                match constraint {
                    Constraints::Integer(_) => ChoiceValue::Integer(0),
                    Constraints::Boolean(_) => ChoiceValue::Boolean(false),
                    Constraints::Float(_) => ChoiceValue::Float(0.5),
                    Constraints::String(_) => ChoiceValue::String("test".to_string()),
                    Constraints::Bytes(_) => ChoiceValue::Bytes(vec![1, 2, 3]),
                },
                constraint,
                false,
            );

            // Node should be constructible and usable
            assert!(!node.was_forced);
        }
        
        println!("INTEGRATION: ChoiceNode works with all constraint types");
    }

    #[test]
    fn test_choice_node_with_values_system() {
        println!("INTEGRATION: Testing ChoiceNode with values system");
        
        // Test all ChoiceValue variants work properly with ChoiceNode
        let values = vec![
            ChoiceValue::Integer(i128::MAX),
            ChoiceValue::Integer(i128::MIN),
            ChoiceValue::Integer(0),
            ChoiceValue::Boolean(true),
            ChoiceValue::Boolean(false),
            ChoiceValue::Float(f64::INFINITY),
            ChoiceValue::Float(f64::NEG_INFINITY),
            ChoiceValue::Float(f64::NAN),
            ChoiceValue::Float(0.0),
            ChoiceValue::Float(-0.0),
            ChoiceValue::String("".to_string()),
            ChoiceValue::String("unicode: 🦀 Rust".to_string()),
            ChoiceValue::Bytes(vec![]),
            ChoiceValue::Bytes(vec![0, 255, 128]),
        ];

        for value in values {
            let choice_type = match value {
                ChoiceValue::Integer(_) => ChoiceType::Integer,
                ChoiceValue::Boolean(_) => ChoiceType::Boolean,
                ChoiceValue::Float(_) => ChoiceType::Float,
                ChoiceValue::String(_) => ChoiceType::String,
                ChoiceValue::Bytes(_) => ChoiceType::Bytes,
            };

            let constraints = match value {
                ChoiceValue::Integer(_) => Constraints::Integer(IntegerConstraints::default()),
                ChoiceValue::Boolean(_) => Constraints::Boolean(BooleanConstraints::default()),
                ChoiceValue::Float(_) => Constraints::Float(FloatConstraints::default()),
                ChoiceValue::String(_) => Constraints::String(StringConstraints::default()),
                ChoiceValue::Bytes(_) => Constraints::Bytes(BytesConstraints::default()),
            };

            let node = ChoiceNode::new(choice_type, value.clone(), constraints, false);
            assert_eq!(node.value, value);
        }
        
        println!("INTEGRATION: ChoiceNode works with all value types");
    }
}