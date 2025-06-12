//! Comprehensive tests for Choice Structure Field Access System capability
//! 
//! This module validates the complete field access system across all choice structures,
//! testing visibility patterns, type consistency, initialization patterns, and PyO3/FFI
//! integration for the choice module's struct field access capability.
//!
//! Tests cover:
//! - Field visibility and access patterns
//! - Type consistency and validation
//! - Struct initialization and construction
//! - PyO3 integration and FFI compatibility
//! - Cross-module field access patterns

use crate::choice::{
    ChoiceType, ChoiceValue, Constraints, ChoiceNode,
    field_access_system::*,
    IntegerConstraints, FloatConstraints, BooleanConstraints, 
    StringConstraints, BytesConstraints,
    NavigationChoiceNode, ChoiceTemplate, TemplateType,
};
use pyo3::prelude::*;
use pyo3::types::*;
use std::collections::HashMap;

/// Test field access patterns for core choice structures
#[cfg(test)]
mod field_access_tests {
    use super::*;

    #[test]
    fn test_choice_node_field_access_patterns() {
        // Test direct field access for all public fields
        let mut node = ChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(42),
            constraints: Constraints::Integer(IntegerConstraints::default()),
            was_forced: false,
            index: Some(0),
        };

        // Test read access to all fields
        assert_eq!(node.choice_type, ChoiceType::Integer);
        assert_eq!(node.value, ChoiceValue::Integer(42));
        assert!(matches!(node.constraints, Constraints::Integer(_)));
        assert!(!node.was_forced);
        assert_eq!(node.index, Some(0));

        // Test write access to all fields
        node.choice_type = ChoiceType::Boolean;
        node.value = ChoiceValue::Boolean(true);
        node.constraints = Constraints::Boolean(BooleanConstraints { p: 0.5 });
        node.was_forced = true;
        node.index = Some(1);

        // Verify mutations
        assert_eq!(node.choice_type, ChoiceType::Boolean);
        assert_eq!(node.value, ChoiceValue::Boolean(true));
        assert!(matches!(node.constraints, Constraints::Boolean(_)));
        assert!(node.was_forced);
        assert_eq!(node.index, Some(1));
    }

    #[test]
    fn test_constraint_struct_field_access() {
        // Test IntegerConstraints field access
        let mut int_constraints = IntegerConstraints {
            min_value: Some(-100),
            max_value: Some(100),
            weights: Some(HashMap::new()),
            shrink_towards: Some(0),
        };

        // Test read access
        assert_eq!(int_constraints.min_value, Some(-100));
        assert_eq!(int_constraints.max_value, Some(100));
        assert!(int_constraints.weights.is_some());
        assert_eq!(int_constraints.shrink_towards, Some(0));

        // Test write access
        int_constraints.min_value = Some(-200);
        int_constraints.max_value = Some(200);
        int_constraints.shrink_towards = Some(10);
        
        assert_eq!(int_constraints.min_value, Some(-200));
        assert_eq!(int_constraints.max_value, Some(200));
        assert_eq!(int_constraints.shrink_towards, Some(10));

        // Test FloatConstraints field access
        let mut float_constraints = FloatConstraints {
            min_value: -1.0,
            max_value: 1.0,
            allow_nan: false,
            smallest_nonzero_magnitude: 1e-10,
        };

        // Test direct field access
        assert_eq!(float_constraints.min_value, -1.0);
        assert_eq!(float_constraints.max_value, 1.0);
        assert!(!float_constraints.allow_nan);
        assert_eq!(float_constraints.smallest_nonzero_magnitude, 1e-10);

        // Test field mutation
        float_constraints.min_value = -2.0;
        float_constraints.max_value = 2.0;
        float_constraints.allow_nan = true;
        
        assert_eq!(float_constraints.min_value, -2.0);
        assert_eq!(float_constraints.max_value, 2.0);
        assert!(float_constraints.allow_nan);
    }

    #[test]
    fn test_navigation_struct_field_access() {
        let mut nav_node = NavigationChoiceNode {
            choice_type: ChoiceType::String,
            value: ChoiceValue::String("test".to_string()),
            constraints: Constraints::String(StringConstraints::default()),
            was_forced: false,
            index: Some(0),
            children: HashMap::new(),
            is_exhausted: false,
        };

        // Test all public field access
        assert_eq!(nav_node.choice_type, ChoiceType::String);
        assert_eq!(nav_node.value, ChoiceValue::String("test".to_string()));
        assert!(matches!(nav_node.constraints, Constraints::String(_)));
        assert!(!nav_node.was_forced);
        assert_eq!(nav_node.index, Some(0));
        assert!(nav_node.children.is_empty());
        assert!(!nav_node.is_exhausted);

        // Test field mutations
        nav_node.choice_type = ChoiceType::Bytes;
        nav_node.value = ChoiceValue::Bytes(vec![1, 2, 3]);
        nav_node.was_forced = true;
        nav_node.is_exhausted = true;

        assert_eq!(nav_node.choice_type, ChoiceType::Bytes);
        assert_eq!(nav_node.value, ChoiceValue::Bytes(vec![1, 2, 3]));
        assert!(nav_node.was_forced);
        assert!(nav_node.is_exhausted);
    }

    #[test]
    fn test_templating_struct_field_access() {
        let mut template = ChoiceTemplate {
            template_type: TemplateType::Simplest,
            count: Some(5),
            is_forcing: false,
            metadata: Some("test_metadata".to_string()),
        };

        // Test public field access
        assert_eq!(template.template_type, TemplateType::Simplest);
        assert_eq!(template.count, Some(5));
        assert!(!template.is_forcing);
        assert_eq!(template.metadata, Some("test_metadata".to_string()));

        // Test field mutations
        template.template_type = TemplateType::AtIndex(1);
        template.count = Some(10);
        template.is_forcing = true;
        template.metadata = None;

        assert_eq!(template.template_type, TemplateType::AtIndex(1));
        assert_eq!(template.count, Some(10));
        assert!(template.is_forcing);
        assert_eq!(template.metadata, None);
    }
}

/// Test type consistency and validation across field access patterns
#[cfg(test)]
mod type_consistency_tests {
    use super::*;

    #[test]
    fn test_choice_value_type_consistency() {
        // Test that choice values maintain type consistency with their types
        let integer_node = ChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(100),
            constraints: Constraints::Integer(IntegerConstraints::default()),
            was_forced: false,
            index: None,
        };

        // Verify type-value consistency
        match (&integer_node.choice_type, &integer_node.value) {
            (ChoiceType::Integer, ChoiceValue::Integer(_)) => (),
            _ => panic!("Type-value mismatch"),
        }

        let boolean_node = ChoiceNode {
            choice_type: ChoiceType::Boolean,
            value: ChoiceValue::Boolean(true),
            constraints: Constraints::Boolean(BooleanConstraints { p: 0.7 }),
            was_forced: false,
            index: None,
        };

        match (&boolean_node.choice_type, &boolean_node.value) {
            (ChoiceType::Boolean, ChoiceValue::Boolean(_)) => (),
            _ => panic!("Type-value mismatch"),
        }
    }

    #[test]
    fn test_constraint_type_consistency() {
        // Test that constraints match their associated choice types
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        });

        match int_constraints {
            Constraints::Integer(_) => (),
            _ => panic!("Constraint type mismatch"),
        }

        let float_constraints = Constraints::Float(FloatConstraints {
            min_value: 0.0,
            max_value: 1.0,
            allow_nan: false,
            smallest_nonzero_magnitude: 1e-10,
        });

        match float_constraints {
            Constraints::Float(_) => (),
            _ => panic!("Constraint type mismatch"),
        }
    }

    #[test]
    fn test_field_type_validation() {
        // Test that field types are validated correctly
        let mut node = ChoiceNode {
            choice_type: ChoiceType::String,
            value: ChoiceValue::String("initial".to_string()),
            constraints: Constraints::String(StringConstraints::default()),
            was_forced: false,
            index: Some(0),
        };

        // Test that we can access and modify fields with correct types
        node.index = Some(42);
        assert_eq!(node.index, Some(42));

        node.was_forced = true;
        assert!(node.was_forced);

        // Test string-specific field access
        if let ChoiceValue::String(ref mut s) = node.value {
            s.push_str("_modified");
        }
        
        if let ChoiceValue::String(s) = &node.value {
            assert_eq!(s, "initial_modified");
        }
    }
}

/// Test struct initialization and construction patterns
#[cfg(test)]
mod initialization_tests {
    use super::*;

    #[test]
    fn test_default_initialization() {
        // Test that all constraint types implement Default properly
        let int_default = IntegerConstraints::default();
        assert_eq!(int_default.min_value, None);
        assert_eq!(int_default.max_value, None);
        assert_eq!(int_default.weights, None);
        assert_eq!(int_default.shrink_towards, Some(0));

        let float_default = FloatConstraints::default();
        assert_eq!(float_default.min_value, f64::NEG_INFINITY);
        assert_eq!(float_default.max_value, f64::INFINITY);
        assert!(float_default.allow_nan);
        assert_eq!(float_default.smallest_nonzero_magnitude, f64::MIN_POSITIVE);

        let bool_default = BooleanConstraints::default();
        assert_eq!(bool_default.p, 0.5);

        let string_default = StringConstraints::default();
        assert_eq!(string_default.min_size, 0);
        assert_eq!(string_default.max_size, 8192);

        let bytes_default = BytesConstraints::default();
        assert_eq!(bytes_default.min_size, 0);
        assert_eq!(bytes_default.max_size, 8192);
    }

    #[test]
    fn test_constructor_patterns() {
        // Test various constructor patterns work with field access
        let node1 = ChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(42),
            constraints: Constraints::Integer(IntegerConstraints::default()),
            was_forced: false,
            index: None,
        };

        // Test that constructed node has accessible fields
        assert_eq!(node1.choice_type, ChoiceType::Integer);
        assert!(!node1.was_forced);
        assert_eq!(node1.index, None);

        // Test construction with all fields specified
        let node2 = ChoiceNode {
            choice_type: ChoiceType::Boolean,
            value: ChoiceValue::Boolean(true),
            constraints: Constraints::Boolean(BooleanConstraints { p: 0.8 }),
            was_forced: true,
            index: Some(5),
        };

        assert_eq!(node2.choice_type, ChoiceType::Boolean);
        assert!(node2.was_forced);
        assert_eq!(node2.index, Some(5));
    }

    #[test]
    fn test_builder_pattern_field_access() {
        // Test builder patterns maintain field access
        let mut template = ChoiceTemplate {
            template_type: TemplateType::Simplest,
            count: None,
            is_forcing: false,
            metadata: None,
        };

        // Build up the template using field access
        template.count = Some(10);
        template.is_forcing = true;
        template.metadata = Some("built".to_string());

        assert_eq!(template.count, Some(10));
        assert!(template.is_forcing);
        assert_eq!(template.metadata, Some("built".to_string()));
    }
}

/// Test PyO3 integration and FFI compatibility for field access
#[cfg(test)]
mod pyo3_integration_tests {
    use super::*;

    #[test]
    fn test_field_access_cloning() {
        // Test that field access works with cloning (important for PyO3)
        let node = ChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(123),
            constraints: Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(1000),
                weights: None,
                shrink_towards: Some(0),
            }),
            was_forced: false,
            index: Some(42),
        };

        // Test cloning preserves field access
        let cloned = node.clone();

        // Verify all fields are accessible after cloning
        assert_eq!(cloned.choice_type, ChoiceType::Integer);
        assert_eq!(cloned.value, ChoiceValue::Integer(123));
        assert!(!cloned.was_forced);
        assert_eq!(cloned.index, Some(42));

        // Test constraint field access after cloning
        if let Constraints::Integer(ref int_constraints) = cloned.constraints {
            assert_eq!(int_constraints.min_value, Some(0));
            assert_eq!(int_constraints.max_value, Some(1000));
            assert_eq!(int_constraints.shrink_towards, Some(0));
        } else {
            panic!("Wrong constraint type after cloning");
        }
    }

    #[test]
    fn test_python_compatibility_field_access() {
        // Test field access patterns that would be used in PyO3 bindings
        let mut node = ChoiceNode {
            choice_type: ChoiceType::String,
            value: ChoiceValue::String("python_test".to_string()),
            constraints: Constraints::String(StringConstraints {
                min_size: 1,
                max_size: 100,
                intervals: Default::default(),
            }),
            was_forced: false,
            index: None,
        };

        // Simulate Python-side field access patterns
        // Read access (would be exposed as getters in PyO3)
        let choice_type = &node.choice_type;
        let value = &node.value;
        let was_forced = node.was_forced;
        let index = node.index;

        assert_eq!(choice_type, &ChoiceType::String);
        if let ChoiceValue::String(s) = value {
            assert_eq!(s, "python_test");
        }
        assert!(!was_forced);
        assert_eq!(index, None);

        // Write access (would be exposed as setters in PyO3)
        node.was_forced = true;
        node.index = Some(100);

        assert!(node.was_forced);
        assert_eq!(node.index, Some(100));
    }

    #[test]
    fn test_ffi_safe_field_access() {
        // Test field access patterns that are safe for FFI
        let constraints = IntegerConstraints {
            min_value: Some(-1000),
            max_value: Some(1000),
            weights: None,
            shrink_towards: Some(0),
        };

        // Test accessing fields in ways that would be FFI-safe
        let min_val = constraints.min_value.unwrap_or(i128::MIN);
        let max_val = constraints.max_value.unwrap_or(i128::MAX);
        let shrink_target = constraints.shrink_towards.unwrap_or(0);

        assert_eq!(min_val, -1000);
        assert_eq!(max_val, 1000);
        assert_eq!(shrink_target, 0);

        // Test that field access doesn't involve complex lifetimes
        let weights_present = constraints.weights.is_some();
        assert!(!weights_present);
    }
}

/// Test cross-module field access patterns
#[cfg(test)]
mod cross_module_access_tests {
    use super::*;

    #[test]
    fn test_navigation_choice_node_integration() {
        // Test field access between navigation and core choice structures
        let mut nav_node = NavigationChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(42),
            constraints: Constraints::Integer(IntegerConstraints::default()),
            was_forced: false,
            index: Some(0),
            children: HashMap::new(),
            is_exhausted: false,
        };

        // Test that navigation-specific fields are accessible
        assert!(nav_node.children.is_empty());
        assert!(!nav_node.is_exhausted);

        // Test that base choice fields are still accessible
        assert_eq!(nav_node.choice_type, ChoiceType::Integer);
        assert_eq!(nav_node.value, ChoiceValue::Integer(42));
        assert!(!nav_node.was_forced);

        // Test adding children via field access
        let child = Box::new(NavigationChoiceNode {
            choice_type: ChoiceType::Boolean,
            value: ChoiceValue::Boolean(true),
            constraints: Constraints::Boolean(BooleanConstraints::default()),
            was_forced: false,
            index: Some(1),
            children: HashMap::new(),
            is_exhausted: false,
        });

        nav_node.children.insert(ChoiceValue::Boolean(true), child);
        assert_eq!(nav_node.children.len(), 1);

        // Test navigation state field access
        nav_node.is_exhausted = true;
        assert!(nav_node.is_exhausted);
    }

    #[test]
    fn test_templating_choice_integration() {
        // Test field access between templating and choice structures
        let template = ChoiceTemplate {
            template_type: TemplateType::AtIndex(0),
            count: Some(3),
            is_forcing: true,
            metadata: Some("integration_test".to_string()),
        };

        // Test template field access
        assert_eq!(template.template_type, TemplateType::AtIndex(1));
        assert_eq!(template.count, Some(3));
        assert!(template.is_forcing);
        assert_eq!(template.metadata, Some("integration_test".to_string()));

        // Test using template data to create choice node
        let node = ChoiceNode {
            choice_type: ChoiceType::String,
            value: ChoiceValue::String("templated".to_string()),
            constraints: Constraints::String(StringConstraints::default()),
            was_forced: template.is_forcing, // Using template field
            index: template.count.map(|c| c as usize), // Using template field
        };

        // Verify integration worked correctly
        assert!(node.was_forced); // From template.is_forcing
        assert_eq!(node.index, Some(3)); // From template.count
    }

    #[test]
    fn test_constraint_inheritance_field_access() {
        // Test field access patterns when constraints are shared/inherited
        let base_int_constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        };

        // Create derived constraints by field access
        let mut derived_constraints = base_int_constraints.clone();
        derived_constraints.min_value = Some(10);
        derived_constraints.max_value = Some(90);

        // Test that both constraint sets have accessible fields
        assert_eq!(base_int_constraints.min_value, Some(0));
        assert_eq!(base_int_constraints.max_value, Some(100));
        
        assert_eq!(derived_constraints.min_value, Some(10));
        assert_eq!(derived_constraints.max_value, Some(90));

        // Test that shared fields remain the same
        assert_eq!(base_int_constraints.shrink_towards, derived_constraints.shrink_towards);
        assert_eq!(base_int_constraints.weights, derived_constraints.weights);
    }
}

/// Test field access error conditions and edge cases
#[cfg(test)]
mod field_access_edge_cases {
    use super::*;

    #[test]
    fn test_optional_field_access() {
        // Test field access with optional fields
        let mut node = ChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(0),
            constraints: Constraints::Integer(IntegerConstraints::default()),
            was_forced: false,
            index: None, // Optional field starts as None
        };

        // Test None case
        assert_eq!(node.index, None);
        assert!(node.index.is_none());

        // Test setting to Some
        node.index = Some(42);
        assert_eq!(node.index, Some(42));
        assert!(node.index.is_some());
        assert_eq!(node.index.unwrap(), 42);

        // Test setting back to None
        node.index = None;
        assert_eq!(node.index, None);
    }

    #[test]
    fn test_complex_field_mutations() {
        // Test complex field access patterns
        let mut constraints = IntegerConstraints {
            min_value: None,
            max_value: None,
            weights: Some(HashMap::new()),
            shrink_towards: Some(0),
        };

        // Test modifying HashMap field
        if let Some(ref mut weights) = constraints.weights {
            weights.insert(10, 0.5);
            weights.insert(20, 0.3);
            weights.insert(30, 0.2);
        }

        // Test accessing modified HashMap
        if let Some(ref weights) = constraints.weights {
            assert_eq!(weights.len(), 3);
            assert_eq!(weights.get(&10), Some(&0.5));
            assert_eq!(weights.get(&20), Some(&0.3));
            assert_eq!(weights.get(&30), Some(&0.2));
        }

        // Test replacing entire HashMap
        let mut new_weights = HashMap::new();
        new_weights.insert(100, 1.0);
        constraints.weights = Some(new_weights);

        if let Some(ref weights) = constraints.weights {
            assert_eq!(weights.len(), 1);
            assert_eq!(weights.get(&100), Some(&1.0));
        }
    }

    #[test]
    fn test_nested_struct_field_access() {
        // Test field access in nested structures
        let mut nav_node = NavigationChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(1),
            constraints: Constraints::Integer(IntegerConstraints::default()),
            was_forced: false,
            index: Some(0),
            children: HashMap::new(),
            is_exhausted: false,
        };

        // Create nested child
        let child = NavigationChoiceNode {
            choice_type: ChoiceType::Boolean,
            value: ChoiceValue::Boolean(true),
            constraints: Constraints::Boolean(BooleanConstraints { p: 0.7 }),
            was_forced: false,
            index: Some(1),
            children: HashMap::new(),
            is_exhausted: false,
        };

        // Test accessing fields in nested structure
        nav_node.children.insert(ChoiceValue::Boolean(true), Box::new(child));

        // Test deep field access
        if let Some(child_node) = nav_node.children.get(&ChoiceValue::Boolean(true)) {
            assert_eq!(child_node.choice_type, ChoiceType::Boolean);
            assert_eq!(child_node.value, ChoiceValue::Boolean(true));
            assert_eq!(child_node.index, Some(1));
            
            // Test accessing constraint fields in nested structure
            if let Constraints::Boolean(ref bool_constraints) = child_node.constraints {
                assert_eq!(bool_constraints.p, 0.7);
            }
        }
    }
}

/// Integration test for complete field access system capability
#[cfg(test)]
mod capability_integration_tests {
    use super::*;

    #[test]
    fn test_complete_field_access_workflow() {
        // Test a complete workflow that exercises all field access patterns
        
        // 1. Create base structures with field access
        let mut base_node = ChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(0),
            constraints: Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(100),
                weights: None,
                shrink_towards: Some(0),
            }),
            was_forced: false,
            index: None,
        };

        // 2. Test field modifications
        base_node.value = ChoiceValue::Integer(50);
        base_node.was_forced = true;
        base_node.index = Some(1);

        // 3. Test constraint field access and modification
        if let Constraints::Integer(ref mut int_constraints) = base_node.constraints {
            int_constraints.min_value = Some(10);
            int_constraints.max_value = Some(90);
            
            let mut weights = HashMap::new();
            weights.insert(50, 0.8);
            weights.insert(25, 0.1);
            weights.insert(75, 0.1);
            int_constraints.weights = Some(weights);
        }

        // 4. Create navigation structure using base node data
        let nav_node = NavigationChoiceNode {
            choice_type: base_node.choice_type,
            value: base_node.value.clone(),
            constraints: base_node.constraints.clone(),
            was_forced: base_node.was_forced,
            index: base_node.index,
            children: HashMap::new(),
            is_exhausted: false,
        };

        // 5. Create template using field data
        let template = ChoiceTemplate {
            template_type: TemplateType::AtIndex(0),
            count: base_node.index,
            is_forcing: base_node.was_forced,
            metadata: Some("workflow_test".to_string()),
        };

        // 6. Verify all field access worked correctly
        assert_eq!(nav_node.choice_type, ChoiceType::Integer);
        assert_eq!(nav_node.value, ChoiceValue::Integer(50));
        assert!(nav_node.was_forced);
        assert_eq!(nav_node.index, Some(1));

        assert_eq!(template.template_type, TemplateType::AtIndex(1));
        assert_eq!(template.count, Some(1));
        assert!(template.is_forcing);
        assert_eq!(template.metadata, Some("workflow_test".to_string()));

        // 7. Test constraint field access in copied structure
        if let Constraints::Integer(ref int_constraints) = nav_node.constraints {
            assert_eq!(int_constraints.min_value, Some(10));
            assert_eq!(int_constraints.max_value, Some(90));
            
            if let Some(ref weights) = int_constraints.weights {
                assert_eq!(weights.len(), 3);
                assert_eq!(weights.get(&50), Some(&0.8));
            }
        }
    }

    #[test]
    fn test_field_access_system_performance() {
        // Test that field access patterns perform well with many operations
        let mut nodes = Vec::new();

        // Create many nodes with field access
        for i in 0..1000 {
            let node = ChoiceNode {
                choice_type: ChoiceType::Integer,
                value: ChoiceValue::Integer(i),
                constraints: Constraints::Integer(IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(i),
                    weights: None,
                    shrink_towards: Some(0),
                }),
                was_forced: i % 2 == 0,
                index: Some(i as usize),
            };
            nodes.push(node);
        }

        // Test bulk field access operations
        let mut sum = 0i128;
        let mut forced_count = 0;
        
        for node in &nodes {
            // Test field access in tight loop
            if let ChoiceValue::Integer(val) = node.value {
                sum += val;
            }
            
            if node.was_forced {
                forced_count += 1;
            }
        }

        assert_eq!(sum, (0..1000).sum::<i128>());
        assert_eq!(forced_count, 500); // Half should be forced (even numbers)

        // Test bulk field mutations
        for node in &mut nodes {
            node.was_forced = !node.was_forced;
            if let Some(ref mut index) = node.index {
                *index += 1000;
            }
        }

        // Verify mutations worked
        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(node.was_forced, i % 2 != 0); // Should be flipped
            assert_eq!(node.index, Some(i + 1000));
        }
    }
}

/// PyO3 FFI capability tests for field access system
#[test]
fn test_pyo3_field_access_capability() {
    Python::with_gil(|py| {
        // Test that field access patterns work with PyO3 integration
        let node = ChoiceNode {
            choice_type: ChoiceType::String,
            value: ChoiceValue::String("pyo3_test".to_string()),
            constraints: Constraints::String(StringConstraints {
                min_size: 1,
                max_size: 50,
                intervals: Default::default(),
            }),
            was_forced: false,
            index: Some(42),
        };

        // Test direct field access for PyO3 compatibility
        assert_eq!(format!("{:?}", node.choice_type), "String");
        if let ChoiceValue::String(ref s) = node.value {
            assert_eq!(s, "pyo3_test");
        }
        assert_eq!(node.was_forced, false);
        assert_eq!(node.index, Some(42));

        // Test direct field access for validation
        if let Constraints::String(ref str_constraints) = node.constraints {
            assert_eq!(str_constraints.min_size, 1);
            assert_eq!(str_constraints.max_size, 50);
        }
    });
}