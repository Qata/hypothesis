//! # Choice Sequence Recording System for Deterministic Replay
//!
//! This module implements the core choice sequence recording functionality that enables
//! deterministic replay in Hypothesis testing. It closely mirrors the Python implementation
//! while leveraging Rust's type system for enhanced safety and performance.
//!
//! ## Key Features
//!
//! ### Automatic Choice Tracking
//! - Records every choice made during ConjectureData draw operations
//! - Assigns deterministic indices using Python's zigzag algorithm
//! - Maintains complete metadata for replay and shrinking
//!
//! ### Deterministic Replay
//! - Enables exact reproduction of test failures
//! - Compatible choice sequence format with Python Hypothesis
//! - Robust handling of choice sequence misalignment
//!
//! ### Indexed Storage
//! - Efficient indexing using Python's complexity-based ordering
//! - Support for 65-bit float indices like Python
//! - Optimal shrinking through lexicographic ordering

use super::{ChoiceType, ChoiceValue, ChoiceNode, Constraints};
use crate::choice::indexing::{choice_to_index, choice_from_index};
use std::collections::HashMap;

/// Choice sequence manager for recording and replaying choices
#[derive(Debug, Clone)]
pub struct ChoiceSequenceRecorder {
    /// Recorded choice sequence for replay
    choices: Vec<ChoiceNode>,
    
    /// Current position in choice sequence during replay
    replay_index: usize,
    
    /// Whether we're in replay mode
    replay_mode: bool,
    
    /// Misalignment detection for robust replay
    misaligned_at: Option<usize>,
    
    /// Cache for choice indices to avoid recomputation
    index_cache: HashMap<(ChoiceType, ChoiceValue), u128>,
}

impl ChoiceSequenceRecorder {
    /// Create a new choice sequence recorder
    pub fn new() -> Self {
        Self {
            choices: Vec::new(),
            replay_index: 0,
            replay_mode: false,
            misaligned_at: None,
            index_cache: HashMap::new(),
        }
    }
    
    /// Create a recorder in replay mode with predefined choices
    pub fn from_choices(choices: Vec<ChoiceNode>) -> Self {
        Self {
            choices,
            replay_index: 0,
            replay_mode: true,
            misaligned_at: None,
            index_cache: HashMap::new(),
        }
    }
    
    /// Record a new choice in the sequence
    /// Implements Python's automatic choice tracking with indexing
    pub fn record_choice(&mut self, mut choice_node: ChoiceNode) -> u128 {
        // Calculate choice index using Python's algorithm
        let choice_index = self.get_or_compute_choice_index(&choice_node);
        
        // Set the index on the choice node
        choice_node.index = Some(choice_index);
        
        // Add to the choice sequence
        self.choices.push(choice_node);
        
        choice_index
    }
    
    /// Try to replay a choice from the recorded sequence
    /// Returns Some(choice_value) if replay succeeds, None if misalignment occurs
    pub fn try_replay_choice(
        &mut self,
        expected_type: ChoiceType,
        expected_constraints: &Constraints,
    ) -> Option<ChoiceValue> {
        if !self.replay_mode || self.replay_index >= self.choices.len() {
            return None;
        }
        
        let replay_choice = &self.choices[self.replay_index];
        
        // Check type compatibility
        if replay_choice.choice_type != expected_type {
            self.misaligned_at = Some(self.replay_index);
            return None;
        }
        
        // Check constraint compatibility using Python's validation
        if !self.choice_compatible_with_constraints(&replay_choice.value, expected_constraints) {
            self.misaligned_at = Some(self.replay_index);
            return None;
        }
        
        self.replay_index += 1;
        Some(replay_choice.value.clone())
    }
    
    /// Get the current choice sequence for inspection
    pub fn get_choices(&self) -> &[ChoiceNode] {
        &self.choices
    }
    
    /// Check if currently in replay mode
    pub fn is_replaying(&self) -> bool {
        self.replay_mode
    }
    
    /// Get the current replay position
    pub fn replay_position(&self) -> usize {
        self.replay_index
    }
    
    /// Check if replay has completed
    pub fn replay_completed(&self) -> bool {
        self.replay_mode && self.replay_index >= self.choices.len()
    }
    
    /// Check if replay misalignment occurred
    pub fn is_misaligned(&self) -> bool {
        self.misaligned_at.is_some()
    }
    
    /// Get misalignment position if any
    pub fn misalignment_position(&self) -> Option<usize> {
        self.misaligned_at
    }
    
    /// Reset to normal recording mode
    pub fn reset_to_recording(&mut self) {
        self.replay_mode = false;
        self.replay_index = 0;
        self.misaligned_at = None;
        self.choices.clear();
    }
    
    /// Generate a choice from an index for reverse replay
    /// This implements Python's choice_from_index functionality
    pub fn choice_from_index(
        &self,
        index: u128,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> ChoiceValue {
        let type_str = match choice_type {
            ChoiceType::Integer => "integer",
            ChoiceType::Boolean => "boolean", 
            ChoiceType::Float => "float",
            ChoiceType::String => "string",
            ChoiceType::Bytes => "bytes",
        };
        
        choice_from_index(index, type_str, constraints)
    }
    
    /// Internal method to get or compute choice index with caching
    fn get_or_compute_choice_index(&mut self, choice_node: &ChoiceNode) -> u128 {
        let cache_key = (choice_node.choice_type, choice_node.value.clone());
        
        if let Some(&cached_index) = self.index_cache.get(&cache_key) {
            return cached_index;
        }
        
        let index = choice_to_index(&choice_node.value, &choice_node.constraints);
        self.index_cache.insert(cache_key, index);
        index
    }
    
    /// Check if a choice value is compatible with given constraints
    /// Implements Python's constraint validation logic
    fn choice_compatible_with_constraints(
        &self,
        value: &ChoiceValue,
        constraints: &Constraints,
    ) -> bool {
        match (value, constraints) {
            (ChoiceValue::Integer(val), Constraints::Integer(c)) => {
                // Check range bounds
                if let Some(min) = c.min_value {
                    if *val < min { return false; }
                }
                if let Some(max) = c.max_value {
                    if *val > max { return false; }
                }
                
                // Check weights if specified
                if let Some(ref weights) = c.weights {
                    if !weights.contains_key(val) {
                        return false;
                    }
                }
                
                true
            }
            
            (ChoiceValue::Boolean(_), Constraints::Boolean(_)) => {
                // Boolean choices are always compatible
                true
            }
            
            (ChoiceValue::Float(val), Constraints::Float(c)) => {
                // Use the float constraint validation
                c.validate(*val)
            }
            
            (ChoiceValue::String(val), Constraints::String(c)) => {
                // Check size bounds
                if val.len() < c.min_size || val.len() > c.max_size {
                    return false;
                }
                
                // Check character set compatibility
                for ch in val.chars() {
                    if !c.intervals.contains(ch as u32) {
                        return false;
                    }
                }
                
                true
            }
            
            (ChoiceValue::Bytes(val), Constraints::Bytes(c)) => {
                // Check size bounds
                val.len() >= c.min_size && val.len() <= c.max_size
            }
            
            _ => false, // Type mismatch
        }
    }
}

impl Default for ChoiceSequenceRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for objects that can record choice sequences
pub trait ChoiceSequenceRecording {
    /// Record a choice and return its index
    fn record_choice_with_index(&mut self, choice_node: ChoiceNode) -> u128;
    
    /// Try to replay a choice from the sequence
    fn try_replay_choice(
        &mut self,
        expected_type: ChoiceType,
        expected_constraints: &Constraints,
    ) -> Option<ChoiceValue>;
    
    /// Get the current choice sequence
    fn get_choice_sequence(&self) -> &[ChoiceNode];
    
    /// Check if currently replaying
    fn is_in_replay_mode(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::{IntegerConstraints, BooleanConstraints, FloatConstraints};
    
    #[test]
    fn test_choice_sequence_recording() {
        let mut recorder = ChoiceSequenceRecorder::new();
        
        // Record some choices
        let int_choice = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );
        
        let index = recorder.record_choice(int_choice);
        assert!(index > 0); // Should have a valid index
        
        let bool_choice = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints::default()),
            false,
        );
        
        recorder.record_choice(bool_choice);
        
        assert_eq!(recorder.get_choices().len(), 2);
    }
    
    #[test]
    fn test_choice_sequence_replay() {
        // Create choices for replay
        let choices = vec![
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(42),
                Constraints::Integer(IntegerConstraints::default()),
                false,
            ),
            ChoiceNode::new(
                ChoiceType::Boolean,
                ChoiceValue::Boolean(true),
                Constraints::Boolean(BooleanConstraints::default()),
                false,
            ),
        ];
        
        let mut recorder = ChoiceSequenceRecorder::from_choices(choices);
        
        // Try to replay the choices
        let int_constraints = Constraints::Integer(IntegerConstraints::default());
        let replayed_int = recorder.try_replay_choice(ChoiceType::Integer, &int_constraints);
        assert_eq!(replayed_int, Some(ChoiceValue::Integer(42)));
        
        let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
        let replayed_bool = recorder.try_replay_choice(ChoiceType::Boolean, &bool_constraints);
        assert_eq!(replayed_bool, Some(ChoiceValue::Boolean(true)));
        
        // Should be completed now
        assert!(recorder.replay_completed());
    }
    
    #[test]
    fn test_choice_sequence_misalignment() {
        // Create choices for replay
        let choices = vec![
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(42),
                Constraints::Integer(IntegerConstraints::default()),
                false,
            ),
        ];
        
        let mut recorder = ChoiceSequenceRecorder::from_choices(choices);
        
        // Try to replay with wrong type - should cause misalignment
        let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
        let result = recorder.try_replay_choice(ChoiceType::Boolean, &bool_constraints);
        assert_eq!(result, None);
        assert!(recorder.is_misaligned());
        assert_eq!(recorder.misalignment_position(), Some(0));
    }
    
    #[test]
    fn test_choice_index_caching() {
        let mut recorder = ChoiceSequenceRecorder::new();
        
        // Record the same choice twice - should use cached index
        let choice1 = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );
        
        let choice2 = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );
        
        let index1 = recorder.record_choice(choice1);
        let index2 = recorder.record_choice(choice2);
        
        // Should get the same index due to caching
        assert_eq!(index1, index2);
    }
    
    #[test]
    fn test_choice_from_index_roundtrip() {
        let recorder = ChoiceSequenceRecorder::new();
        
        // Test integer roundtrip
        let constraints = Constraints::Integer(IntegerConstraints::default());
        let original_value = ChoiceValue::Integer(42);
        let index = choice_to_index(&original_value, &constraints);
        let roundtrip_value = recorder.choice_from_index(index, ChoiceType::Integer, &constraints);
        
        assert_eq!(original_value, roundtrip_value);
    }
}