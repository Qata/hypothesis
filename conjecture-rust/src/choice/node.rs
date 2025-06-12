//! ChoiceNode implementation - the core data structure for representing choices
//! 
//! This module provides the core immutable choice representation with type, value, 
//! constraints, forcing metadata, and indexing support. It closely mirrors the Python
//! ChoiceNode class but with Rust's ownership and type safety benefits.

use super::{ChoiceType, ChoiceValue, Constraints};
use serde::{Serialize, Deserialize};

/// A single choice made during test generation
/// 
/// This closely mirrors Python's ChoiceNode class. Represents an immutable choice
/// with complete metadata about how it was generated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceNode {
    /// The type of choice that was made (integer, float, boolean, string, bytes)
    pub choice_type: ChoiceType,
    
    /// The actual value that was chosen
    pub value: ChoiceValue,
    
    /// The constraints that were applied when making this choice
    pub constraints: Constraints,
    
    /// Whether this choice was forced to a specific value (not randomly generated)
    pub was_forced: bool,
    
    /// Optional index for tracking position in the choice sequence
    /// Note: This is None by default and only assigned via ExampleRecord
    pub index: Option<usize>,
}

impl ChoiceNode {
    /// Create a new choice node without an index
    /// 
    /// This matches the Python ChoiceNode constructor. Index is None by default
    /// and only set via ExampleRecord to avoid stale indices after copying.
    pub fn new(
        choice_type: ChoiceType,
        value: ChoiceValue,
        constraints: Constraints,
        was_forced: bool,
    ) -> Self {
        Self {
            choice_type,
            value,
            constraints,
            was_forced,
            index: None,
        }
    }
    
    /// Create a new choice node with explicit index
    /// 
    /// Use this when you need to assign an index during creation (e.g., from ExampleRecord)
    pub fn with_index(
        choice_type: ChoiceType,
        value: ChoiceValue,
        constraints: Constraints,
        was_forced: bool,
        index: usize,
    ) -> Self {
        Self {
            choice_type,
            value,
            constraints,
            was_forced,
            index: Some(index),
        }
    }
    
    /// Set the index on this node, consuming self and returning a new node
    pub fn set_index(mut self, index: usize) -> Self {
        self.index = Some(index);
        self
    }

    /// Copy this node, optionally replacing the value and/or constraints
    /// 
    /// This matches Python's ChoiceNode.copy() method exactly. Note that forced nodes
    /// cannot have their values changed, and the index is explicitly NOT copied to
    /// prevent stale index issues.
    pub fn copy(
        &self,
        with_value: Option<ChoiceValue>,
        with_constraints: Option<Constraints>,
    ) -> Result<Self, String> {
        // Prevent modifying forced nodes with new values as this doesn't make sense
        if self.was_forced && with_value.is_some() {
            return Err("modifying a forced node doesn't make sense".to_string());
        }

        let new_value = with_value.unwrap_or_else(|| self.value.clone());
        let new_constraints = with_constraints.unwrap_or_else(|| self.constraints.clone());

        Ok(Self {
            choice_type: self.choice_type,
            value: new_value,
            constraints: new_constraints,
            was_forced: self.was_forced,
            index: None, // Explicitly not copying index as per Python implementation
        })
    }

    /// Copy this node with a new value only
    /// 
    /// Convenience method that matches common usage pattern
    pub fn copy_with_value(&self, new_value: ChoiceValue) -> Result<Self, String> {
        self.copy(Some(new_value), None)
    }

    /// Copy this node with new constraints only
    /// 
    /// Convenience method for updating constraints
    pub fn copy_with_constraints(&self, new_constraints: Constraints) -> Result<Self, String> {
        self.copy(None, Some(new_constraints))
    }

    /// Check if this node is trivial (would shrink to itself)
    /// 
    /// A node is trivial if it cannot be simplified any further. This does not
    /// mean that modifying a trivial node can't produce simpler test cases when
    /// viewing the tree as a whole. Just that when viewing this node in
    /// isolation, this is the simplest the node can get.
    /// 
    /// This implements Python's ChoiceNode.trivial property logic exactly.
    pub fn trivial(&self) -> bool {
        if self.was_forced {
            return true;
        }

        match self.choice_type {
            ChoiceType::Float => {
                if let (ChoiceValue::Float(float_value), Constraints::Float(constraints)) = 
                    (&self.value, &self.constraints) {
                    
                    let min_value = constraints.min_value;
                    let max_value = constraints.max_value;
                    let shrink_towards = 0.0;

                    // Case 1: Unbounded range (-inf, +inf)
                    if min_value == f64::NEG_INFINITY && max_value == f64::INFINITY {
                        return self.choice_equal_float(*float_value, shrink_towards);
                    }

                    // Case 2: Bounded range that contains an integer
                    if !min_value.is_infinite() && !max_value.is_infinite() {
                        let ceil_min = min_value.ceil();
                        let floor_max = max_value.floor();
                        
                        if ceil_min <= floor_max {
                            // The interval contains an integer. The simplest integer is the
                            // one closest to shrink_towards
                            let clamped_shrink = shrink_towards.max(ceil_min).min(floor_max);
                            return self.choice_equal_float(*float_value, clamped_shrink);
                        }
                    }

                    // Case 3: Conservative case - return false
                    // The real answer here is "the value in [min_value, max_value] with
                    // the lowest denominator when represented as a fraction".
                    // It would be good to compute this correctly in the future, but it's
                    // also not incorrect to be conservative here.
                    return false;
                }
            }
            _ => {
                // For non-float types: check if value equals the zero-index choice
                // For now, implement simple checks for common trivial cases
                match &self.value {
                    ChoiceValue::Integer(0) => {
                        // Check if shrink_towards is 0
                        if let Constraints::Integer(constraints) = &self.constraints {
                            return constraints.shrink_towards == Some(0);
                        }
                    }
                    ChoiceValue::Boolean(false) => {
                        // False is typically the trivial boolean value
                        return true;
                    }
                    ChoiceValue::String(s) if s.is_empty() => {
                        // Empty string is trivial if min_size allows it
                        if let Constraints::String(constraints) = &self.constraints {
                            return constraints.min_size == 0;
                        }
                    }
                    ChoiceValue::Bytes(b) if b.is_empty() => {
                        // Empty bytes is trivial if min_size allows it
                        if let Constraints::Bytes(constraints) = &self.constraints {
                            return constraints.min_size == 0;
                        }
                    }
                    _ => {}
                }
            }
        }

        false
    }

    /// Helper method for float equality comparison that handles NaN and signed zero
    fn choice_equal_float(&self, f1: f64, f2: f64) -> bool {
        // Handle NaN case - NaN == NaN for choice comparison
        if f1.is_nan() && f2.is_nan() {
            return true;
        }
        if f1.is_nan() || f2.is_nan() {
            return false;
        }
        // Use bit-level comparison to distinguish -0.0 from 0.0
        f1.to_bits() == f2.to_bits()
    }
}

impl PartialEq for ChoiceNode {
    fn eq(&self, other: &Self) -> bool {
        use crate::choice::choice_equal;
        
        self.choice_type == other.choice_type
            && self.was_forced == other.was_forced
            && choice_equal(&self.value, &other.value)
            && self.constraints == other.constraints
    }
}

impl Eq for ChoiceNode {}

impl std::hash::Hash for ChoiceNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash based on type, value, and constraints for uniqueness
        self.choice_type.hash(state);
        self.was_forced.hash(state);
        
        // Hash the value based on its type
        match &self.value {
            ChoiceValue::Integer(i) => i.hash(state),
            ChoiceValue::Boolean(b) => b.hash(state),
            ChoiceValue::Float(f) => f.to_bits().hash(state), // Use bits for float hashing
            ChoiceValue::String(s) => s.hash(state),
            ChoiceValue::Bytes(b) => b.hash(state),
        }
        
        // Hash constraints (simplified for now)
        match &self.constraints {
            Constraints::Integer(c) => {
                c.min_value.hash(state);
                c.max_value.hash(state);
                c.shrink_towards.hash(state);
            }
            Constraints::Boolean(c) => {
                c.p.to_bits().hash(state);
            }
            Constraints::Float(c) => {
                c.min_value.to_bits().hash(state);
                c.max_value.to_bits().hash(state);
                c.allow_nan.hash(state);
            }
            Constraints::String(c) => {
                c.min_size.hash(state);
                c.max_size.hash(state);
                c.intervals.hash(state);
            }
            Constraints::Bytes(c) => {
                c.min_size.hash(state);
                c.max_size.hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::{IntegerConstraints, BooleanConstraints};

    #[test]
    fn test_choice_node_creation() {
        let node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );

        assert_eq!(node.choice_type, ChoiceType::Integer);
        assert_eq!(node.value, ChoiceValue::Integer(42));
        assert!(!node.was_forced);
    }

    #[test]
    fn test_choice_node_copy_with_value() {
        let node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );

        let copied = node.copy_with_value(ChoiceValue::Integer(100)).unwrap();
        
        assert_eq!(copied.value, ChoiceValue::Integer(100));
        assert_eq!(copied.choice_type, node.choice_type);
        assert_eq!(copied.was_forced, node.was_forced);
    }

    #[test]
    fn test_cannot_modify_forced_node() {
        let forced_node = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints::default()),
            true, // forced
        );

        let result = forced_node.copy_with_value(ChoiceValue::Boolean(false));
        
        assert!(result.is_err());
    }

    #[test]
    fn test_forced_nodes_are_trivial() {
        let forced_node = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints::default()),
            true, // forced
        );

        assert!(forced_node.trivial());
    }

    #[test]
    fn test_trivial_integer_nodes() {
        // Integer with shrink_towards=0 should be trivial when value=0
        let trivial_node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(0),
            Constraints::Integer(IntegerConstraints::default()), // shrink_towards=0
            false,
        );
        
        assert!(trivial_node.trivial());
        
        // Integer with value != shrink_towards should not be trivial
        let non_trivial_node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );
        
        assert!(!non_trivial_node.trivial());
    }

    #[test]
    fn test_trivial_float_nodes() {
        use crate::choice::FloatConstraints;
        
        // Unbounded float should be trivial when value=0.0
        let trivial_float = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(0.0),
            Constraints::Float(FloatConstraints::default()), // unbounded
            false,
        );
        
        assert!(trivial_float.trivial());
        
        // Non-zero unbounded float should not be trivial
        let non_trivial_float = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(3.14),
            Constraints::Float(FloatConstraints::default()),
            false,
        );
        
        assert!(!non_trivial_float.trivial());
        
        // Bounded float containing integer - should be trivial when value equals the clamped integer
        let bounded_constraints = FloatConstraints {
            min_value: 1.5,
            max_value: 3.5,
            allow_nan: false,
            smallest_nonzero_magnitude: f64::MIN_POSITIVE,
        };
        
        let bounded_trivial = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(2.0), // closest integer to 0 in range [1.5, 3.5]
            Constraints::Float(bounded_constraints.clone()),
            false,
        );
        
        assert!(bounded_trivial.trivial());
    }

    #[test]
    fn test_choice_node_equality() {
        let node1 = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );

        let node2 = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );

        assert_eq!(node1, node2);
    }

    #[test]
    fn test_choice_node_is_hashable() {
        let node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );

        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(node.clone(), "test");
        
        assert_eq!(map.get(&node), Some(&"test"));
    }
}