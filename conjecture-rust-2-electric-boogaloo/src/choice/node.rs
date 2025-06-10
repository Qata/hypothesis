//! ChoiceNode implementation - the core data structure for representing choices

use super::{ChoiceType, ChoiceValue, Constraints};

/// A single choice made during test generation
/// 
/// This closely mirrors Python's ChoiceNode class
#[derive(Debug, Clone)]
pub struct ChoiceNode {
    pub choice_type: ChoiceType,
    pub value: ChoiceValue,
    pub constraints: Constraints,
    pub was_forced: bool,
}

impl ChoiceNode {
    /// Create a new choice node
    pub fn new(
        choice_type: ChoiceType,
        value: ChoiceValue,
        constraints: Constraints,
        was_forced: bool,
    ) -> Self {
        println!("CHOICE_NODE DEBUG: Creating new ChoiceNode");
        println!("CHOICE_NODE DEBUG: type={}, was_forced={}", choice_type, was_forced);
        println!("CHOICE_NODE DEBUG: value={:?}", value);
        println!("CHOICE_NODE DEBUG: constraints={:?}", constraints);
        
        Self {
            choice_type,
            value,
            constraints,
            was_forced,
        }
    }

    /// Copy this node with a new value
    /// Cannot modify forced nodes
    pub fn copy_with_value(&self, new_value: ChoiceValue) -> Result<Self, String> {
        println!("CHOICE_NODE DEBUG: Attempting to copy node with new value");
        println!("CHOICE_NODE DEBUG: original was_forced={}", self.was_forced);
        println!("CHOICE_NODE DEBUG: new_value={:?}", new_value);
        
        if self.was_forced {
            println!("CHOICE_NODE DEBUG: ERROR - Cannot modify forced node");
            return Err("Cannot modify forced nodes".to_string());
        }

        println!("CHOICE_NODE DEBUG: Creating copy with new value");
        Ok(Self {
            choice_type: self.choice_type.clone(),
            value: new_value,
            constraints: self.constraints.clone(),
            was_forced: self.was_forced,
        })
    }

    /// Check if this node is trivial (would shrink to itself)
    /// For now, just return true for forced nodes
    pub fn trivial(&self) -> bool {
        println!("CHOICE_NODE DEBUG: Checking if node is trivial");
        println!("CHOICE_NODE DEBUG: was_forced={}", self.was_forced);
        
        if self.was_forced {
            println!("CHOICE_NODE DEBUG: Forced node is trivial");
            return true;
        }

        // TODO: Implement proper triviality check based on constraints
        // For now, conservatively return false for non-forced nodes
        println!("CHOICE_NODE DEBUG: Non-forced node is not trivial (conservative)");
        false
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
        println!("CHOICE_NODE DEBUG: Testing ChoiceNode creation");
        
        let node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );

        println!("CHOICE_NODE DEBUG: Created node: {:?}", node);
        assert_eq!(node.choice_type, ChoiceType::Integer);
        assert_eq!(node.value, ChoiceValue::Integer(42));
        assert!(!node.was_forced);
        println!("CHOICE_NODE DEBUG: ChoiceNode creation test passed");
    }

    #[test]
    fn test_choice_node_copy_with_value() {
        println!("CHOICE_NODE DEBUG: Testing ChoiceNode copy with value");
        
        let node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );

        let copied = node.copy_with_value(ChoiceValue::Integer(100)).unwrap();
        println!("CHOICE_NODE DEBUG: Copied node: {:?}", copied);
        
        assert_eq!(copied.value, ChoiceValue::Integer(100));
        assert_eq!(copied.choice_type, node.choice_type);
        assert_eq!(copied.was_forced, node.was_forced);
        println!("CHOICE_NODE DEBUG: Copy with value test passed");
    }

    #[test]
    fn test_cannot_modify_forced_node() {
        println!("CHOICE_NODE DEBUG: Testing forced node modification prevention");
        
        let forced_node = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints::default()),
            true, // forced
        );

        let result = forced_node.copy_with_value(ChoiceValue::Boolean(false));
        println!("CHOICE_NODE DEBUG: Modification result: {:?}", result);
        
        assert!(result.is_err());
        println!("CHOICE_NODE DEBUG: Forced node modification prevention test passed");
    }

    #[test]
    fn test_forced_nodes_are_trivial() {
        println!("CHOICE_NODE DEBUG: Testing forced nodes triviality");
        
        let forced_node = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints::default()),
            true, // forced
        );

        println!("CHOICE_NODE DEBUG: Forced node trivial: {}", forced_node.trivial());
        assert!(forced_node.trivial());
        println!("CHOICE_NODE DEBUG: Forced nodes triviality test passed");
    }

    #[test]
    fn test_choice_node_equality() {
        println!("CHOICE_NODE DEBUG: Testing ChoiceNode equality");
        
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

        println!("CHOICE_NODE DEBUG: node1 == node2: {}", node1 == node2);
        assert_eq!(node1, node2);
        println!("CHOICE_NODE DEBUG: ChoiceNode equality test passed");
    }

    #[test]
    fn test_choice_node_is_hashable() {
        println!("CHOICE_NODE DEBUG: Testing ChoiceNode hashability");
        
        let node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );

        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(node.clone(), "test");
        
        println!("CHOICE_NODE DEBUG: Successfully stored node in HashMap");
        assert_eq!(map.get(&node), Some(&"test"));
        println!("CHOICE_NODE DEBUG: ChoiceNode hashability test passed");
    }
}