//! Comprehensive Choice Type System Integration Capability Tests
//!
//! This module provides comprehensive PyO3/FFI integration tests that validate the complete
//! Choice Type System Integration capability for ChoiceNode representation across all type
//! variants (Integer, Float, String, Bytes, Boolean). Tests focus on capability-level behavior
//! validation, proper type handling, constraint enforcement, and forced flag handling.
//!
//! Key capabilities tested:
//! - Complete ChoiceNode representation with proper type variants
//! - Type/value/constraints/forced flag consistency and validation
//! - Cross-variant integration and compatibility
//! - PyO3/FFI marshaling and Python compatibility
//! - Architectural compliance with idiomatic Rust patterns

use crate::choice::{
    ChoiceType, ChoiceValue, Constraints, ChoiceNode,
    IntegerConstraints, BooleanConstraints, FloatConstraints, 
    StringConstraints, BytesConstraints, IntervalSet,
    choice_equal, choice_permitted
};
use crate::choice::indexing::{choice_to_index, choice_from_index};
use pyo3::prelude::*;
use pyo3::types::*;
use std::collections::HashMap;

/// PyO3-compatible wrapper for the Choice Type System Integration capability
/// This struct simulates what would be exposed through PyO3 bindings to Python
#[derive(Debug, Clone)]
pub struct PyChoiceTypeSystemCapability {
    /// Storage for created choice nodes for validation and testing
    nodes: Vec<ChoiceNode>,
    /// Metadata for Python integration tracking
    python_metadata: HashMap<String, String>,
    /// Debug mode flag for enhanced logging
    debug_mode: bool,
}

impl PyChoiceTypeSystemCapability {
    /// Create new PyO3-compatible choice type system capability
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            python_metadata: HashMap::new(),
            debug_mode: false,
        }
    }

    /// Enable debug mode for comprehensive logging
    pub fn enable_debug(&mut self) -> &mut Self {
        self.debug_mode = true;
        self.python_metadata.insert("debug_enabled".to_string(), "true".to_string());
        self
    }

    /// Create ChoiceNode from Python-compatible parameters
    pub fn py_create_choice_node(
        &mut self,
        choice_type: &str,
        value_str: &str,
        constraints_dict: HashMap<String, String>,
        was_forced: bool,
    ) -> Result<usize, String> {
        if self.debug_mode {
            println!("CHOICE_TYPE_SYSTEM DEBUG: Creating choice node - type: {}, value: {}, forced: {}", 
                choice_type, value_str, was_forced);
        }

        let (choice_type_enum, choice_value, constraints) = 
            self.parse_choice_components(choice_type, value_str, constraints_dict)?;

        let node = ChoiceNode::new(choice_type_enum, choice_value, constraints, was_forced);
        
        self.nodes.push(node);
        let node_id = self.nodes.len() - 1;
        
        self.python_metadata.insert(
            format!("node_{}_type", node_id), 
            choice_type.to_string()
        );
        
        if self.debug_mode {
            println!("CHOICE_TYPE_SYSTEM DEBUG: Created node {} successfully", node_id);
        }
        
        Ok(node_id)
    }

    /// Validate a choice node's internal consistency
    pub fn py_validate_choice_node(&self, node_id: usize) -> Result<bool, String> {
        let node = self.nodes.get(node_id)
            .ok_or_else(|| format!("Node {} not found", node_id))?;

        if self.debug_mode {
            println!("CHOICE_TYPE_SYSTEM DEBUG: Validating node {} - {:?}", node_id, node);
        }

        // Check type/value consistency
        let type_matches = match (&node.choice_type, &node.value) {
            (ChoiceType::Integer, ChoiceValue::Integer(_)) => true,
            (ChoiceType::Boolean, ChoiceValue::Boolean(_)) => true,
            (ChoiceType::Float, ChoiceValue::Float(_)) => true,
            (ChoiceType::String, ChoiceValue::String(_)) => true,
            (ChoiceType::Bytes, ChoiceValue::Bytes(_)) => true,
            _ => false,
        };

        if !type_matches {
            return Ok(false);
        }

        // Check type/constraints consistency
        let constraints_match = match (&node.choice_type, &node.constraints) {
            (ChoiceType::Integer, Constraints::Integer(_)) => true,
            (ChoiceType::Boolean, Constraints::Boolean(_)) => true,
            (ChoiceType::Float, Constraints::Float(_)) => true,
            (ChoiceType::String, Constraints::String(_)) => true,
            (ChoiceType::Bytes, Constraints::Bytes(_)) => true,
            _ => false,
        };

        if !constraints_match {
            return Ok(false);
        }

        // Check value satisfies constraints
        let value_permitted = choice_permitted(&node.value, &node.constraints);
        
        if self.debug_mode {
            println!("CHOICE_TYPE_SYSTEM DEBUG: Node {} validation - type_matches: {}, constraints_match: {}, value_permitted: {}", 
                node_id, type_matches, constraints_match, value_permitted);
        }

        Ok(value_permitted)
    }

    /// Test indexing capability for a choice node
    pub fn py_test_indexing(&self, node_id: usize) -> Result<(u128, bool), String> {
        let node = self.nodes.get(node_id)
            .ok_or_else(|| format!("Node {} not found", node_id))?;

        if self.debug_mode {
            println!("CHOICE_TYPE_SYSTEM DEBUG: Testing indexing for node {} - {:?}", node_id, node);
        }

        // Convert to index
        let index = choice_to_index(&node.value, &node.constraints);
        
        // Convert back from index
        let choice_type_str = match node.choice_type {
            ChoiceType::Integer => "integer",
            ChoiceType::Boolean => "boolean", 
            ChoiceType::Float => "float",
            ChoiceType::String => "string",
            ChoiceType::Bytes => "bytes",
        };

        let recovered_value = choice_from_index(index, choice_type_str, &node.constraints);
        let roundtrip_success = choice_equal(&node.value, &recovered_value);

        if self.debug_mode {
            println!("CHOICE_TYPE_SYSTEM DEBUG: Node {} indexing - index: {}, roundtrip_success: {}", 
                node_id, index, roundtrip_success);
        }

        Ok((index, roundtrip_success))
    }

    /// Test choice node copying capability
    pub fn py_test_copying(&self, node_id: usize, new_value_str: Option<String>) -> Result<bool, String> {
        let node = self.nodes.get(node_id)
            .ok_or_else(|| format!("Node {} not found", node_id))?;

        if self.debug_mode {
            println!("CHOICE_TYPE_SYSTEM DEBUG: Testing copying for node {} with new_value: {:?}", 
                node_id, new_value_str);
        }

        if let Some(value_str) = new_value_str {
            let new_value = self.parse_choice_value(&node.choice_type, &value_str)?;
            let copy_result = node.copy_with_value(new_value);
            
            match copy_result {
                Ok(copied_node) => {
                    let consistent = match (&node.choice_type, &copied_node.choice_type) {
                        (t1, t2) => t1 == t2,
                    } && copied_node.was_forced == node.was_forced
                      && copied_node.constraints == node.constraints
                      && copied_node.index.is_none(); // Index should not be copied
                    
                    if self.debug_mode {
                        println!("CHOICE_TYPE_SYSTEM DEBUG: Copy successful, consistency: {}", consistent);
                    }
                    Ok(consistent)
                },
                Err(e) => {
                    if node.was_forced {
                        // Expected error for forced nodes
                        Ok(true)
                    } else {
                        Err(format!("Unexpected copy error: {}", e))
                    }
                }
            }
        } else {
            // Test copying with same value
            let copy_result = node.copy(None, None);
            match copy_result {
                Ok(copied_node) => {
                    let identical = choice_equal(&node.value, &copied_node.value)
                        && node.choice_type == copied_node.choice_type
                        && node.constraints == copied_node.constraints
                        && node.was_forced == copied_node.was_forced
                        && copied_node.index.is_none(); // Index should not be copied
                    
                    if self.debug_mode {
                        println!("CHOICE_TYPE_SYSTEM DEBUG: Same-value copy successful, identical: {}", identical);
                    }
                    Ok(identical)
                },
                Err(e) => Err(format!("Copy error: {}", e))
            }
        }
    }

    /// Test forced flag behavior
    pub fn py_test_forced_behavior(&self, node_id: usize) -> Result<HashMap<String, bool>, String> {
        let node = self.nodes.get(node_id)
            .ok_or_else(|| format!("Node {} not found", node_id))?;

        if self.debug_mode {
            println!("CHOICE_TYPE_SYSTEM DEBUG: Testing forced behavior for node {} (forced: {})", 
                node_id, node.was_forced);
        }

        let mut results = HashMap::new();

        // Test 1: Forced nodes should be trivial
        results.insert("forced_implies_trivial".to_string(), 
            !node.was_forced || node.trivial());

        // Test 2: Cannot modify forced node value
        if node.was_forced {
            let dummy_value = match node.choice_type {
                ChoiceType::Integer => ChoiceValue::Integer(999),
                ChoiceType::Boolean => ChoiceValue::Boolean(!matches!(node.value, ChoiceValue::Boolean(true))),
                ChoiceType::Float => ChoiceValue::Float(999.99),
                ChoiceType::String => ChoiceValue::String("dummy".to_string()),
                ChoiceType::Bytes => ChoiceValue::Bytes(vec![255]),
            };
            
            let modify_result = node.copy_with_value(dummy_value);
            results.insert("forced_modification_fails".to_string(), modify_result.is_err());
        } else {
            results.insert("forced_modification_fails".to_string(), true);
        }

        // Test 3: Forced flag preservation in copying
        if let Ok(copied) = node.copy(None, None) {
            results.insert("forced_flag_preserved".to_string(), 
                copied.was_forced == node.was_forced);
        } else {
            results.insert("forced_flag_preserved".to_string(), false);
        }

        if self.debug_mode {
            println!("CHOICE_TYPE_SYSTEM DEBUG: Forced behavior results: {:?}", results);
        }

        Ok(results)
    }

    /// Get statistics about created nodes
    pub fn py_get_statistics(&self) -> HashMap<String, i64> {
        let mut stats = HashMap::new();
        
        stats.insert("total_nodes".to_string(), self.nodes.len() as i64);
        
        for choice_type in [ChoiceType::Integer, ChoiceType::Boolean, ChoiceType::Float, 
                           ChoiceType::String, ChoiceType::Bytes] {
            let count = self.nodes.iter()
                .filter(|n| n.choice_type == choice_type)
                .count() as i64;
            stats.insert(format!("{}_nodes", choice_type), count);
        }
        
        let forced_count = self.nodes.iter()
            .filter(|n| n.was_forced)
            .count() as i64;
        stats.insert("forced_nodes".to_string(), forced_count);
        
        if self.debug_mode {
            println!("CHOICE_TYPE_SYSTEM DEBUG: Statistics: {:?}", stats);
        }
        
        stats
    }

    /// Helper: Parse choice components from Python-compatible formats
    fn parse_choice_components(
        &self,
        choice_type: &str,
        value_str: &str,
        constraints_dict: HashMap<String, String>,
    ) -> Result<(ChoiceType, ChoiceValue, Constraints), String> {
        let choice_type_enum = match choice_type {
            "integer" => ChoiceType::Integer,
            "boolean" => ChoiceType::Boolean,
            "float" => ChoiceType::Float,
            "string" => ChoiceType::String,
            "bytes" => ChoiceType::Bytes,
            _ => return Err(format!("Unknown choice type: {}", choice_type)),
        };

        let choice_value = self.parse_choice_value(&choice_type_enum, value_str)?;
        let constraints = self.parse_constraints(&choice_type_enum, constraints_dict)?;

        Ok((choice_type_enum, choice_value, constraints))
    }

    /// Helper: Parse choice value from string representation
    fn parse_choice_value(&self, choice_type: &ChoiceType, value_str: &str) -> Result<ChoiceValue, String> {
        match choice_type {
            ChoiceType::Integer => {
                value_str.parse::<i128>()
                    .map(ChoiceValue::Integer)
                    .map_err(|_| format!("Invalid integer: {}", value_str))
            },
            ChoiceType::Boolean => {
                match value_str.to_lowercase().as_str() {
                    "true" | "1" | "yes" => Ok(ChoiceValue::Boolean(true)),
                    "false" | "0" | "no" => Ok(ChoiceValue::Boolean(false)),
                    _ => Err(format!("Invalid boolean: {}", value_str)),
                }
            },
            ChoiceType::Float => {
                if value_str == "nan" {
                    Ok(ChoiceValue::Float(f64::NAN))
                } else if value_str == "inf" {
                    Ok(ChoiceValue::Float(f64::INFINITY))
                } else if value_str == "-inf" {
                    Ok(ChoiceValue::Float(f64::NEG_INFINITY))
                } else {
                    value_str.parse::<f64>()
                        .map(ChoiceValue::Float)
                        .map_err(|_| format!("Invalid float: {}", value_str))
                }
            },
            ChoiceType::String => {
                Ok(ChoiceValue::String(value_str.to_string()))
            },
            ChoiceType::Bytes => {
                // Parse as comma-separated byte values or hex string
                if value_str.starts_with("0x") {
                    // Hex format
                    let hex_str = &value_str[2..];
                    if hex_str.len() % 2 != 0 {
                        return Err("Hex string must have even length".to_string());
                    }
                    let mut bytes = Vec::new();
                    for i in (0..hex_str.len()).step_by(2) {
                        let byte_str = &hex_str[i..i+2];
                        let byte = u8::from_str_radix(byte_str, 16)
                            .map_err(|_| format!("Invalid hex byte: {}", byte_str))?;
                        bytes.push(byte);
                    }
                    Ok(ChoiceValue::Bytes(bytes))
                } else if value_str.contains(',') {
                    // Comma-separated format
                    let bytes: Result<Vec<u8>, _> = value_str
                        .split(',')
                        .map(|s| s.trim().parse::<u8>())
                        .collect();
                    bytes.map(ChoiceValue::Bytes)
                        .map_err(|_| format!("Invalid bytes format: {}", value_str))
                } else if value_str.is_empty() {
                    Ok(ChoiceValue::Bytes(vec![]))
                } else {
                    // Single byte
                    value_str.parse::<u8>()
                        .map(|b| ChoiceValue::Bytes(vec![b]))
                        .map_err(|_| format!("Invalid byte: {}", value_str))
                }
            },
        }
    }

    /// Helper: Parse constraints from dictionary
    fn parse_constraints(
        &self,
        choice_type: &ChoiceType,
        constraints_dict: HashMap<String, String>,
    ) -> Result<Constraints, String> {
        match choice_type {
            ChoiceType::Integer => {
                let min_value = constraints_dict.get("min_value")
                    .and_then(|s| if s == "none" { None } else { s.parse().ok() });
                let max_value = constraints_dict.get("max_value")
                    .and_then(|s| if s == "none" { None } else { s.parse().ok() });
                let shrink_towards = constraints_dict.get("shrink_towards")
                    .and_then(|s| if s == "none" { None } else { s.parse().ok() });
                
                Ok(Constraints::Integer(IntegerConstraints {
                    min_value,
                    max_value,
                    weights: None,
                    shrink_towards,
                }))
            },
            ChoiceType::Boolean => {
                let p = constraints_dict.get("p")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.5);
                
                Ok(Constraints::Boolean(BooleanConstraints { p }))
            },
            ChoiceType::Float => {
                let min_value = constraints_dict.get("min_value")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(f64::NEG_INFINITY);
                let max_value = constraints_dict.get("max_value")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(f64::INFINITY);
                let allow_nan = constraints_dict.get("allow_nan")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(true);
                let smallest_nonzero_magnitude = constraints_dict.get("smallest_nonzero_magnitude")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(f64::MIN_POSITIVE);
                
                Ok(Constraints::Float(FloatConstraints {
                    min_value,
                    max_value,
                    allow_nan,
                    smallest_nonzero_magnitude,
                }))
            },
            ChoiceType::String => {
                let min_size = constraints_dict.get("min_size")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let max_size = constraints_dict.get("max_size")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(8192);
                
                Ok(Constraints::String(StringConstraints {
                    min_size,
                    max_size,
                    intervals: IntervalSet::default(),
                }))
            },
            ChoiceType::Bytes => {
                let min_size = constraints_dict.get("min_size")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let max_size = constraints_dict.get("max_size")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(8192);
                
                Ok(Constraints::Bytes(BytesConstraints {
                    min_size,
                    max_size,
                }))
            },
        }
    }
}

#[cfg(test)]
mod choice_type_system_integration_capability_tests {
    use super::*;

    /// Test complete integer choice type system integration
    #[test]
    fn test_integer_choice_type_system_integration() {
        let mut capability = PyChoiceTypeSystemCapability::new();
        capability.enable_debug();

        // Test case 1: Basic integer with default constraints
        let mut constraints = HashMap::new();
        constraints.insert("shrink_towards".to_string(), "0".to_string());
        
        let node_id = capability.py_create_choice_node(
            "integer", "42", constraints, false
        ).expect("Should create integer node");

        assert!(capability.py_validate_choice_node(node_id).unwrap());

        let (index, roundtrip_ok) = capability.py_test_indexing(node_id).unwrap();
        assert!(roundtrip_ok, "Integer indexing roundtrip should succeed");
        assert!(index > 0, "Non-zero integer should have non-zero index");

        assert!(capability.py_test_copying(node_id, None).unwrap());
        assert!(capability.py_test_copying(node_id, Some("100".to_string())).unwrap());

        // Test case 2: Integer with bounds
        let mut bounded_constraints = HashMap::new();
        bounded_constraints.insert("min_value".to_string(), "-10".to_string());
        bounded_constraints.insert("max_value".to_string(), "10".to_string());
        bounded_constraints.insert("shrink_towards".to_string(), "0".to_string());

        let bounded_node_id = capability.py_create_choice_node(
            "integer", "5", bounded_constraints, false
        ).expect("Should create bounded integer node");

        assert!(capability.py_validate_choice_node(bounded_node_id).unwrap());

        let (_bounded_index, bounded_roundtrip_ok) = capability.py_test_indexing(bounded_node_id).unwrap();
        assert!(bounded_roundtrip_ok, "Bounded integer indexing roundtrip should succeed");

        // Test case 3: Forced integer
        let forced_node_id = capability.py_create_choice_node(
            "integer", "42", HashMap::new(), true
        ).expect("Should create forced integer node");

        let forced_behavior = capability.py_test_forced_behavior(forced_node_id).unwrap();
        assert_eq!(forced_behavior.get("forced_implies_trivial"), Some(&true));
        assert_eq!(forced_behavior.get("forced_modification_fails"), Some(&true));
        assert_eq!(forced_behavior.get("forced_flag_preserved"), Some(&true));
    }

    /// Test complete boolean choice type system integration
    #[test]
    fn test_boolean_choice_type_system_integration() {
        let mut capability = PyChoiceTypeSystemCapability::new();
        capability.enable_debug();

        // Test case 1: Boolean with default probability
        let mut constraints = HashMap::new();
        constraints.insert("p".to_string(), "0.5".to_string());

        let node_id = capability.py_create_choice_node(
            "boolean", "true", constraints, false
        ).expect("Should create boolean node");

        assert!(capability.py_validate_choice_node(node_id).unwrap());

        let (_index, roundtrip_ok) = capability.py_test_indexing(node_id).unwrap();
        assert!(roundtrip_ok, "Boolean indexing roundtrip should succeed");

        assert!(capability.py_test_copying(node_id, None).unwrap());
        assert!(capability.py_test_copying(node_id, Some("false".to_string())).unwrap());

        // Test case 2: Boolean with extreme probabilities
        let mut p0_constraints = HashMap::new();
        p0_constraints.insert("p".to_string(), "0.0".to_string());

        let p0_node_id = capability.py_create_choice_node(
            "boolean", "false", p0_constraints, false
        ).expect("Should create p=0.0 boolean node");

        assert!(capability.py_validate_choice_node(p0_node_id).unwrap());

        let mut p1_constraints = HashMap::new();
        p1_constraints.insert("p".to_string(), "1.0".to_string());

        let p1_node_id = capability.py_create_choice_node(
            "boolean", "true", p1_constraints, false
        ).expect("Should create p=1.0 boolean node");

        assert!(capability.py_validate_choice_node(p1_node_id).unwrap());

        // Test case 3: Forced boolean
        let forced_node_id = capability.py_create_choice_node(
            "boolean", "true", HashMap::new(), true
        ).expect("Should create forced boolean node");

        let forced_behavior = capability.py_test_forced_behavior(forced_node_id).unwrap();
        assert_eq!(forced_behavior.get("forced_implies_trivial"), Some(&true));
    }

    /// Test complete float choice type system integration
    #[test]
    fn test_float_choice_type_system_integration() {
        let mut capability = PyChoiceTypeSystemCapability::new();
        capability.enable_debug();

        // Test case 1: Regular float
        let mut constraints = HashMap::new();
        constraints.insert("min_value".to_string(), "-1000.0".to_string());
        constraints.insert("max_value".to_string(), "1000.0".to_string());
        constraints.insert("allow_nan".to_string(), "true".to_string());

        let node_id = capability.py_create_choice_node(
            "float", "3.14", constraints.clone(), false
        ).expect("Should create float node");

        assert!(capability.py_validate_choice_node(node_id).unwrap());

        let (_index, roundtrip_ok) = capability.py_test_indexing(node_id).unwrap();
        assert!(roundtrip_ok, "Float indexing roundtrip should succeed");

        assert!(capability.py_test_copying(node_id, None).unwrap());
        assert!(capability.py_test_copying(node_id, Some("2.71".to_string())).unwrap());

        // Test case 2: Special float values
        let special_values = ["0.0", "-0.0", "inf", "-inf", "nan"];
        for value in special_values {
            let special_node_id = capability.py_create_choice_node(
                "float", value, constraints.clone(), false
            ).expect(&format!("Should create float node for {}", value));

            assert!(capability.py_validate_choice_node(special_node_id).unwrap());

            if value != "nan" { // NaN doesn't roundtrip exactly
                let (_, special_roundtrip_ok) = capability.py_test_indexing(special_node_id).unwrap();
                assert!(special_roundtrip_ok, "Special float {} indexing roundtrip should succeed", value);
            }
        }

        // Test case 3: Float with smallest_nonzero_magnitude constraint
        let mut magnitude_constraints = HashMap::new();
        magnitude_constraints.insert("min_value".to_string(), "-10.0".to_string());
        magnitude_constraints.insert("max_value".to_string(), "10.0".to_string());
        magnitude_constraints.insert("allow_nan".to_string(), "false".to_string());
        magnitude_constraints.insert("smallest_nonzero_magnitude".to_string(), "1e-6".to_string());

        let magnitude_node_id = capability.py_create_choice_node(
            "float", "1e-3", magnitude_constraints, false
        ).expect("Should create magnitude-constrained float node");

        assert!(capability.py_validate_choice_node(magnitude_node_id).unwrap());
    }

    /// Test complete string choice type system integration
    #[test]
    fn test_string_choice_type_system_integration() {
        let mut capability = PyChoiceTypeSystemCapability::new();
        capability.enable_debug();

        // Test case 1: Basic string
        let mut constraints = HashMap::new();
        constraints.insert("min_size".to_string(), "0".to_string());
        constraints.insert("max_size".to_string(), "100".to_string());

        let node_id = capability.py_create_choice_node(
            "string", "hello", constraints.clone(), false
        ).expect("Should create string node");

        assert!(capability.py_validate_choice_node(node_id).unwrap());

        let (_index, roundtrip_ok) = capability.py_test_indexing(node_id).unwrap();
        assert!(roundtrip_ok, "String indexing roundtrip should succeed");

        assert!(capability.py_test_copying(node_id, None).unwrap());
        assert!(capability.py_test_copying(node_id, Some("world".to_string())).unwrap());

        // Test case 2: Empty string
        let empty_node_id = capability.py_create_choice_node(
            "string", "", constraints.clone(), false
        ).expect("Should create empty string node");

        assert!(capability.py_validate_choice_node(empty_node_id).unwrap());

        // Test case 3: Unicode string
        let unicode_node_id = capability.py_create_choice_node(
            "string", "🦀♥️", constraints.clone(), false
        ).expect("Should create unicode string node");

        assert!(capability.py_validate_choice_node(unicode_node_id).unwrap());

        // Test case 4: String size constraints
        let mut size_constraints = HashMap::new();
        size_constraints.insert("min_size".to_string(), "5".to_string());
        size_constraints.insert("max_size".to_string(), "10".to_string());

        let sized_node_id = capability.py_create_choice_node(
            "string", "medium", size_constraints, false
        ).expect("Should create size-constrained string node");

        assert!(capability.py_validate_choice_node(sized_node_id).unwrap());
    }

    /// Test complete bytes choice type system integration
    #[test]
    fn test_bytes_choice_type_system_integration() {
        let mut capability = PyChoiceTypeSystemCapability::new();
        capability.enable_debug();

        // Test case 1: Basic bytes
        let mut constraints = HashMap::new();
        constraints.insert("min_size".to_string(), "0".to_string());
        constraints.insert("max_size".to_string(), "100".to_string());

        let node_id = capability.py_create_choice_node(
            "bytes", "1,2,3,4,5", constraints.clone(), false
        ).expect("Should create bytes node");

        assert!(capability.py_validate_choice_node(node_id).unwrap());

        let (index, roundtrip_ok) = capability.py_test_indexing(node_id).unwrap();
        assert!(roundtrip_ok, "Bytes indexing roundtrip should succeed");

        assert!(capability.py_test_copying(node_id, None).unwrap());
        assert!(capability.py_test_copying(node_id, Some("10,20,30".to_string())).unwrap());

        // Test case 2: Empty bytes
        let empty_node_id = capability.py_create_choice_node(
            "bytes", "", constraints.clone(), false
        ).expect("Should create empty bytes node");

        assert!(capability.py_validate_choice_node(empty_node_id).unwrap());

        // Test case 3: Hex format bytes
        let hex_node_id = capability.py_create_choice_node(
            "bytes", "0x48656c6c6f", constraints.clone(), false
        ).expect("Should create hex bytes node");

        assert!(capability.py_validate_choice_node(hex_node_id).unwrap());

        // Test case 4: Single byte
        let single_node_id = capability.py_create_choice_node(
            "bytes", "255", constraints.clone(), false
        ).expect("Should create single byte node");

        assert!(capability.py_validate_choice_node(single_node_id).unwrap());
    }

    /// Test cross-variant integration and compatibility
    #[test]
    fn test_cross_variant_integration() {
        let mut capability = PyChoiceTypeSystemCapability::new();
        capability.enable_debug();

        // Create one node of each type
        let integer_id = capability.py_create_choice_node(
            "integer", "42", HashMap::new(), false
        ).unwrap();

        let boolean_id = capability.py_create_choice_node(
            "boolean", "true", HashMap::new(), false
        ).unwrap();

        let float_id = capability.py_create_choice_node(
            "float", "3.14", HashMap::new(), false
        ).unwrap();

        let string_id = capability.py_create_choice_node(
            "string", "test", HashMap::new(), false
        ).unwrap();

        let bytes_id = capability.py_create_choice_node(
            "bytes", "1,2,3", HashMap::new(), false
        ).unwrap();

        // Verify all nodes validate correctly
        for &node_id in &[integer_id, boolean_id, float_id, string_id, bytes_id] {
            assert!(capability.py_validate_choice_node(node_id).unwrap(),
                "Node {} should validate", node_id);
        }

        // Test indexing consistency across all types
        for &node_id in &[integer_id, boolean_id, float_id, string_id, bytes_id] {
            let (_index, roundtrip_ok) = capability.py_test_indexing(node_id).unwrap();
            assert!(roundtrip_ok, "Node {} indexing should roundtrip", node_id);
        }

        // Verify statistics are consistent
        let stats = capability.py_get_statistics();
        assert_eq!(stats.get("total_nodes"), Some(&5));
        assert_eq!(stats.get("integer_nodes"), Some(&1));
        assert_eq!(stats.get("boolean_nodes"), Some(&1));
        assert_eq!(stats.get("float_nodes"), Some(&1));
        assert_eq!(stats.get("string_nodes"), Some(&1));
        assert_eq!(stats.get("bytes_nodes"), Some(&1));
        assert_eq!(stats.get("forced_nodes"), Some(&0));
    }

    /// Test forced flag behavior across all types
    #[test]
    fn test_forced_flag_integration() {
        let mut capability = PyChoiceTypeSystemCapability::new();
        capability.enable_debug();

        let test_cases = [
            ("integer", "42"),
            ("boolean", "true"),
            ("float", "3.14"),
            ("string", "test"),
            ("bytes", "1,2,3"),
        ];

        for (choice_type, value) in test_cases {
            // Test forced node
            let forced_id = capability.py_create_choice_node(
                choice_type, value, HashMap::new(), true
            ).unwrap();

            let forced_behavior = capability.py_test_forced_behavior(forced_id).unwrap();
            assert_eq!(forced_behavior.get("forced_implies_trivial"), Some(&true),
                "Forced {} node should be trivial", choice_type);
            assert_eq!(forced_behavior.get("forced_modification_fails"), Some(&true),
                "Forced {} node modification should fail", choice_type);
            assert_eq!(forced_behavior.get("forced_flag_preserved"), Some(&true),
                "Forced {} node flag should be preserved", choice_type);

            // Test non-forced node
            let non_forced_id = capability.py_create_choice_node(
                choice_type, value, HashMap::new(), false
            ).unwrap();

            let non_forced_behavior = capability.py_test_forced_behavior(non_forced_id).unwrap();
            assert_eq!(non_forced_behavior.get("forced_flag_preserved"), Some(&true),
                "Non-forced {} node flag should be preserved", choice_type);
        }

        // Verify forced node statistics
        let stats = capability.py_get_statistics();
        assert_eq!(stats.get("forced_nodes"), Some(&5));
        assert_eq!(stats.get("total_nodes"), Some(&10));
    }

    /// Test constraint validation across all types
    #[test]
    fn test_constraint_validation_integration() {
        let mut capability = PyChoiceTypeSystemCapability::new();
        capability.enable_debug();

        // Test invalid integer constraint combinations
        let mut invalid_int_constraints = HashMap::new();
        invalid_int_constraints.insert("min_value".to_string(), "10".to_string());
        invalid_int_constraints.insert("max_value".to_string(), "5".to_string()); // max < min

        // This should create the node but validation should respect the constraints
        let invalid_int_id = capability.py_create_choice_node(
            "integer", "7", invalid_int_constraints, false
        ).unwrap();

        // Node creation succeeds but value may not satisfy constraints
        let _ = capability.py_validate_choice_node(invalid_int_id);

        // Test valid float constraints with smallest_nonzero_magnitude
        let mut float_constraints = HashMap::new();
        float_constraints.insert("min_value".to_string(), "0.0".to_string());
        float_constraints.insert("max_value".to_string(), "1.0".to_string());
        float_constraints.insert("smallest_nonzero_magnitude".to_string(), "1e-6".to_string());
        float_constraints.insert("allow_nan".to_string(), "false".to_string());

        let valid_float_id = capability.py_create_choice_node(
            "float", "0.5", float_constraints, false
        ).unwrap();

        assert!(capability.py_validate_choice_node(valid_float_id).unwrap());

        // Test string size constraints
        let mut string_constraints = HashMap::new();
        string_constraints.insert("min_size".to_string(), "3".to_string());
        string_constraints.insert("max_size".to_string(), "10".to_string());

        let valid_string_id = capability.py_create_choice_node(
            "string", "hello", string_constraints, false
        ).unwrap();

        assert!(capability.py_validate_choice_node(valid_string_id).unwrap());

        // Test bytes size constraints
        let mut bytes_constraints = HashMap::new();
        bytes_constraints.insert("min_size".to_string(), "2".to_string());
        bytes_constraints.insert("max_size".to_string(), "5".to_string());

        let valid_bytes_id = capability.py_create_choice_node(
            "bytes", "1,2,3", bytes_constraints, false
        ).unwrap();

        assert!(capability.py_validate_choice_node(valid_bytes_id).unwrap());
    }

    /// Test edge cases and error handling
    #[test]
    fn test_edge_cases_and_error_handling() {
        let mut capability = PyChoiceTypeSystemCapability::new();
        capability.enable_debug();

        // Test invalid choice type
        let invalid_type_result = capability.py_create_choice_node(
            "invalid_type", "42", HashMap::new(), false
        );
        assert!(invalid_type_result.is_err());

        // Test invalid value format
        let invalid_value_result = capability.py_create_choice_node(
            "integer", "not_a_number", HashMap::new(), false
        );
        assert!(invalid_value_result.is_err());

        // Test accessing non-existent node
        let invalid_access_result = capability.py_validate_choice_node(9999);
        assert!(invalid_access_result.is_err());

        // Test invalid hex bytes format
        let invalid_hex_result = capability.py_create_choice_node(
            "bytes", "0x123", HashMap::new(), false // Odd length hex
        );
        assert!(invalid_hex_result.is_err());

        // Test invalid boolean format
        let invalid_bool_result = capability.py_create_choice_node(
            "boolean", "maybe", HashMap::new(), false
        );
        assert!(invalid_bool_result.is_err());

        // Test extreme float values
        let extreme_float_id = capability.py_create_choice_node(
            "float", "1.7976931348623157e+308", HashMap::new(), false // Near f64::MAX
        ).unwrap();

        assert!(capability.py_validate_choice_node(extreme_float_id).unwrap());
    }

    /// Test PyO3 compatibility simulation
    #[test]
    fn test_pyo3_compatibility_simulation() {
        let mut capability = PyChoiceTypeSystemCapability::new();
        capability.enable_debug();

        // Simulate Python dictionary-style constraint passing
        let python_style_constraints: HashMap<String, String> = [
            ("min_value".to_string(), "-100".to_string()),
            ("max_value".to_string(), "100".to_string()),
            ("shrink_towards".to_string(), "0".to_string()),
        ].iter().cloned().collect();

        let node_id = capability.py_create_choice_node(
            "integer", "42", python_style_constraints, false
        ).unwrap();

        // Simulate Python-style boolean returns
        assert!(capability.py_validate_choice_node(node_id).unwrap());

        // Simulate Python-style tuple returns
        let (index, success) = capability.py_test_indexing(node_id).unwrap();
        assert!(success);
        assert!(index > 0);

        // Simulate Python-style dictionary returns
        let behavior_dict = capability.py_test_forced_behavior(node_id).unwrap();
        assert!(behavior_dict.contains_key("forced_implies_trivial"));
        assert!(behavior_dict.contains_key("forced_modification_fails"));

        let stats_dict = capability.py_get_statistics();
        assert!(stats_dict.contains_key("total_nodes"));
        assert!(stats_dict.contains_key("integer_nodes"));

        // Simulate Python exception handling equivalents
        let error_result = capability.py_create_choice_node(
            "nonexistent", "value", HashMap::new(), false
        );
        assert!(error_result.is_err());

        println!("PyO3 compatibility simulation completed successfully");
    }

    /// Comprehensive integration test covering all capabilities
    #[test]
    fn test_comprehensive_choice_type_system_integration() {
        let mut capability = PyChoiceTypeSystemCapability::new();
        capability.enable_debug();

        // Create comprehensive test matrix
        let test_matrix = [
            // (type, value, forced, should_validate)
            ("integer", "0", false, true),
            ("integer", "42", true, true),
            ("boolean", "false", false, true),
            ("boolean", "true", true, true),
            ("float", "0.0", false, true),
            ("float", "nan", false, true),
            ("float", "inf", true, true),
            ("string", "", false, true),
            ("string", "hello", true, true),
            ("bytes", "", false, true),
            ("bytes", "0xff", true, true),
        ];

        let mut created_nodes = Vec::new();

        // Create all test nodes
        for (choice_type, value, forced, _should_validate) in test_matrix {
            let node_id = capability.py_create_choice_node(
                choice_type, value, HashMap::new(), forced
            ).expect(&format!("Should create {} node with value {}", choice_type, value));
            
            created_nodes.push((node_id, choice_type, value, forced));
        }

        // Validate all nodes
        for (node_id, choice_type, value, _forced) in &created_nodes {
            let is_valid = capability.py_validate_choice_node(*node_id).unwrap();
            assert!(is_valid, "Node {} ({}: {}) should validate", node_id, choice_type, value);
        }

        // Test indexing for all nodes
        for (node_id, choice_type, value, _forced) in &created_nodes {
            if *choice_type != "string" && *choice_type != "bytes" { // Skip complex types for indexing
                let (_index, roundtrip_ok) = capability.py_test_indexing(*node_id).unwrap();
                if *value != "nan" { // NaN doesn't roundtrip exactly
                    assert!(roundtrip_ok, "Node {} ({}: {}) indexing should roundtrip", 
                        node_id, choice_type, value);
                }
            }
        }

        // Test copying behavior for all nodes
        for (node_id, choice_type, value, forced) in &created_nodes {
            let copy_result = capability.py_test_copying(*node_id, None).unwrap();
            assert!(copy_result, "Node {} ({}: {}) should copy correctly", node_id, choice_type, value);

            // Test forced behavior
            let forced_behavior = capability.py_test_forced_behavior(*node_id).unwrap();
            if *forced {
                assert_eq!(forced_behavior.get("forced_implies_trivial"), Some(&true));
                assert_eq!(forced_behavior.get("forced_modification_fails"), Some(&true));
            }
        }

        // Verify final statistics
        let final_stats = capability.py_get_statistics();
        assert_eq!(final_stats.get("total_nodes"), Some(&(test_matrix.len() as i64)));

        let forced_count = test_matrix.iter().filter(|(_, _, _, forced)| *forced).count() as i64;
        assert_eq!(final_stats.get("forced_nodes"), Some(&forced_count));

        println!("Comprehensive choice type system integration test completed successfully");
        println!("Created and validated {} nodes across all choice types", test_matrix.len());
        println!("Statistics: {:?}", final_stats);
    }
}