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
    pub index: usize,
}

impl ChoiceNode {
    /// Create a new choice node
    pub fn new(
        choice_type: ChoiceType,
        value: ChoiceValue,
        constraints: Constraints,
        was_forced: bool,
    ) -> Self {
        Self::new_with_index(choice_type, value, constraints, was_forced, 0)
    }
    
    /// Create a new choice node with explicit index
    pub fn new_with_index(
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
            index,
        }
    }

    /// Copy this node with a new value
    /// Cannot modify forced nodes
    pub fn copy_with_value(&self, new_value: ChoiceValue) -> Result<Self, String> {
        if self.was_forced {
            return Err("Cannot modify forced nodes".to_string());
        }
        Ok(Self {
            choice_type: self.choice_type.clone(),
            value: new_value,
            constraints: self.constraints.clone(),
            was_forced: self.was_forced,
            index: self.index,
        })
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
        use crate::choice::indexing::choice_from_index;
        use crate::choice::choice_equal;
        
        if self.was_forced {
            return true;
        }

        if self.choice_type != ChoiceType::Float {
            // For non-float types: check if value equals choice_from_index(0, ...)
            let zero_value = choice_from_index(0, &self.choice_type.to_string(), &self.constraints);
            return choice_equal(&self.value, &zero_value);
        } else {
            // Float case: complex logic from Python
            if let (ChoiceValue::Float(float_value), Constraints::Float(constraints)) = (&self.value, &self.constraints) {
                let min_value = constraints.min_value;
                let max_value = constraints.max_value;
                let shrink_towards = 0.0;

                // Case 1: Unbounded range (-inf, +inf)
                if min_value == f64::NEG_INFINITY && max_value == f64::INFINITY {
                    return choice_equal(&self.value, &ChoiceValue::Float(shrink_towards));
                }

                // Case 2: Bounded range that contains an integer
                if !min_value.is_infinite() && !max_value.is_infinite() {
                    let ceil_min = min_value.ceil();
                    let floor_max = max_value.floor();
                    
                    if ceil_min <= floor_max {
                        // The interval contains an integer. The simplest integer is the
                        // one closest to shrink_towards
                        let clamped_shrink = shrink_towards.max(ceil_min).min(floor_max);
                        return choice_equal(&self.value, &ChoiceValue::Float(clamped_shrink));
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

        // Fallback: not trivial
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
    fn test_trivial_integer_nodes() {
        println!("CHOICE_NODE DEBUG: Testing trivial integer nodes");
        
        // Integer with shrink_towards=0 should be trivial when value=0
        let trivial_node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(0),
            Constraints::Integer(IntegerConstraints::default()), // shrink_towards=0
            false,
        );
        
        println!("CHOICE_NODE DEBUG: Integer 0 trivial: {}", trivial_node.trivial());
        assert!(trivial_node.trivial());
        
        // Integer with value != shrink_towards should not be trivial
        let non_trivial_node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );
        
        println!("CHOICE_NODE DEBUG: Integer 42 trivial: {}", non_trivial_node.trivial());
        assert!(!non_trivial_node.trivial());
        
        println!("CHOICE_NODE DEBUG: Trivial integer nodes test passed");
    }

    #[test]
    fn test_trivial_float_nodes() {
        use crate::choice::FloatConstraints;
        println!("CHOICE_NODE DEBUG: Testing trivial float nodes");
        
        // Unbounded float should be trivial when value=0.0
        let trivial_float = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(0.0),
            Constraints::Float(FloatConstraints::default()), // unbounded
            false,
        );
        
        println!("CHOICE_NODE DEBUG: Float 0.0 trivial: {}", trivial_float.trivial());
        assert!(trivial_float.trivial());
        
        // Non-zero unbounded float should not be trivial
        let non_trivial_float = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(3.14),
            Constraints::Float(FloatConstraints::default()),
            false,
        );
        
        println!("CHOICE_NODE DEBUG: Float 3.14 trivial: {}", non_trivial_float.trivial());
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
        
        println!("CHOICE_NODE DEBUG: Bounded float 2.0 trivial: {}", bounded_trivial.trivial());
        assert!(bounded_trivial.trivial());
        
        println!("CHOICE_NODE DEBUG: Trivial float nodes test passed");
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