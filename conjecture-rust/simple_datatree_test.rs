//! Simple DataTree navigation verification test
//! This test validates the core functionality is working without requiring dependencies

mod datatree;
mod choice;
mod data;

use datatree::{DataTree, TreeNode};
use choice::{ChoiceType, ChoiceValue, Constraints, IntegerConstraints, BooleanConstraints};
use data::Status;
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_datatree_functionality() {
        let mut tree = DataTree::new();
        
        // Test initial state
        let initial_stats = tree.get_stats();
        assert_eq!(initial_stats.total_nodes, 0);
        
        // Create a simple path
        let simple_path = vec![
            (
                ChoiceType::Integer,
                ChoiceValue::Integer(42),
                Box::new(Constraints::Integer(IntegerConstraints::default())),
                false
            ),
        ];
        
        tree.record_path(&simple_path, Status::Valid, HashMap::new());
        let stats = tree.get_stats();
        assert!(stats.total_nodes > 0);
        
        println!("DataTree basic functionality test passed!");
    }

    #[test]
    fn test_node_splitting() {
        let mut node = TreeNode::new(1);
        let mut next_id = 2;
        
        // Add choices
        node.add_choice(
            ChoiceType::Integer,
            ChoiceValue::Integer(10),
            DataTree::create_integer_constraints(Some(0), Some(100)),
            false
        );
        node.add_choice(
            ChoiceType::Integer,
            ChoiceValue::Integer(20),
            DataTree::create_integer_constraints(Some(0), Some(100)),
            false
        );
        
        assert_eq!(node.values.len(), 2);
        
        // Split at index 1
        let suffix = node.split_at(1, &mut next_id);
        
        // Verify split
        assert_eq!(node.values.len(), 1);
        assert_eq!(suffix.values.len(), 1);
        
        println!("Node splitting test passed!");
    }
}

fn main() {
    println!("Running simple DataTree navigation tests...");
    
    // This would normally run the tests
    println!("Use 'cargo test' to run the actual tests");
}