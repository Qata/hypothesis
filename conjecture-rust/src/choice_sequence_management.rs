//! Choice Sequence Management System for ConjectureData
//! 
//! This module provides a direct port of Python's choice sequence management,
//! including choice recording, replay, misalignment detection, and buffer management.
//! It preserves the same logic and behavior as the Python implementation.

use crate::choice::{ChoiceType, ChoiceValue, Constraints, ChoiceNode};
use std::fmt;

/// Choice Sequence Manager - Direct port of Python ConjectureData choice management
/// Handles choice recording, replay from prefix, and misalignment detection
#[derive(Debug, Clone)]
pub struct ChoiceSequenceManager {
    /// Recorded choice nodes (equivalent to Python's nodes tuple)
    nodes: Vec<ChoiceNode>,
    
    /// Current position in prefix during replay (equivalent to Python's index)
    index: usize,
    
    /// Prefix choices for replay (equivalent to Python's prefix)
    prefix: Option<Vec<ChoiceValue>>,
    
    /// Misalignment tracking (equivalent to Python's misaligned_at)
    misaligned_at: Option<MisalignmentInfo>,
    
    /// Current buffer length (equivalent to Python's length)
    length: usize,
    
    /// Maximum buffer size (equivalent to Python's max_length)
    max_length: usize,
    
    /// Maximum number of choices (equivalent to Python's max_choices)
    max_choices: usize,
    
    /// Whether we've hit an overrun condition
    overrun: bool,
}

/// Misalignment information - tracks where replay diverged from prefix
/// Direct port of Python's MisalignedAt tuple
#[derive(Debug, Clone, PartialEq)]
pub struct MisalignmentInfo {
    /// Index where misalignment occurred
    pub index: usize,
    /// Expected choice type
    pub choice_type: ChoiceType,
    /// Expected constraints
    pub constraints: Constraints,
    /// Forced value if any
    pub forced: Option<ChoiceValue>,
}

/// Choice template for advanced replay scenarios
/// Direct port of Python's ChoiceTemplate
#[derive(Debug, Clone, PartialEq)]
pub enum ChoiceTemplate {
    /// Generate simplest choice (equivalent to choice_from_index(0))
    Simplest {
        choice_type: ChoiceType,
        constraints: Constraints,
        count: Option<i32>,
    },
}

impl ChoiceSequenceManager {
    /// Create a new choice sequence manager with optional prefix for replay
    /// Direct port of Python ConjectureData.__init__ choice-related parameters
    pub fn new(
        max_length: usize,
        max_choices: usize,
        prefix: Option<Vec<ChoiceValue>>,
    ) -> Self {
        Self {
            nodes: Vec::new(),
            index: 0,
            prefix,
            misaligned_at: None,
            length: 0,
            max_length,
            max_choices,
            overrun: false,
        }
    }
    
    /// Draw a choice - main entry point that handles both replay and generation
    /// Direct port of Python ConjectureData._draw method
    pub fn draw(
        &mut self,
        choice_type: ChoiceType,
        constraints: Constraints,
        forced: Option<ChoiceValue>,
        observe: bool,
    ) -> Result<ChoiceValue, ChoiceSequenceError> {
        // Check for overrun conditions (port of Python buffer checks)
        if self.length == self.max_length {
            self.mark_overrun();
            return Err(ChoiceSequenceError::BufferOverflow {
                required: self.length + 1,
                available: self.max_length,
            });
        }
        
        if self.nodes.len() == self.max_choices {
            self.mark_overrun();
            return Err(ChoiceSequenceError::TooManyChoices {
                count: self.nodes.len(),
                max_choices: self.max_choices,
            });
        }
        
        // Handle prefixed choices (replay mechanism)
        let was_forced = forced.is_some();
        let value = if observe && self.prefix.is_some() && self.index < self.prefix.as_ref().unwrap().len() {
            self.pop_choice(choice_type, &constraints, forced)?
        } else if let Some(forced_value) = forced {
            forced_value
        } else {
            // Generate new choice (would call provider in full implementation)
            self.generate_choice(choice_type, &constraints)?
        };
        
        // Record the choice if observable
        if observe {
            let size = self.calculate_choice_size(&value);
            
            if self.length + size > self.max_length {
                self.mark_overrun();
                return Err(ChoiceSequenceError::BufferOverflow {
                    required: self.length + size,
                    available: self.max_length,
                });
            }
            
            // Create and record choice node
            let node = ChoiceNode::with_index(
                choice_type,
                value.clone(),
                constraints,
                was_forced,
                self.nodes.len().try_into().unwrap(),
            );
            
            self.nodes.push(node);
            self.length += size;
        }
        
        Ok(value)
    }
    
    /// Pop a choice from prefix during replay with misalignment detection
    /// Direct port of Python ConjectureData._pop_choice method
    fn pop_choice(
        &mut self,
        choice_type: ChoiceType,
        constraints: &Constraints,
        forced: Option<ChoiceValue>,
    ) -> Result<ChoiceValue, ChoiceSequenceError> {
        let prefix = self.prefix.as_ref().unwrap();
        
        if self.index >= prefix.len() {
            return Err(ChoiceSequenceError::IndexOutOfBounds {
                index: self.index,
                max_index: prefix.len().saturating_sub(1),
            });
        }
        
        let value = prefix[self.index].clone();
        
        // Determine the choice type from the value
        let node_choice_type = match &value {
            ChoiceValue::String(_) => ChoiceType::String,
            ChoiceValue::Float(_) => ChoiceType::Float,
            ChoiceValue::Integer(_) => ChoiceType::Integer,
            ChoiceValue::Boolean(_) => ChoiceType::Boolean,
            ChoiceValue::Bytes(_) => ChoiceType::Bytes,
        };
        
        // MISALIGNMENT DETECTION
        // Check if type differs or constraints don't permit the value
        let is_misaligned = node_choice_type != choice_type || !self.choice_permitted(&value, constraints);
        
        let final_value = if is_misaligned {
            // Track first misalignment only
            if self.misaligned_at.is_none() {
                self.misaligned_at = Some(MisalignmentInfo {
                    index: self.index,
                    choice_type,
                    constraints: constraints.clone(),
                    forced,
                });
            }
            
            // Generate replacement choice using simplest strategy
            self.choice_from_index(0, choice_type, constraints)?
        } else {
            value
        };
        
        self.index += 1;
        Ok(final_value)
    }
    
    /// Get the number of recorded choices
    pub fn num_choices(&self) -> usize {
        self.nodes.len()
    }
    
    /// Check if we're currently replaying from a prefix
    pub fn is_replaying(&self) -> bool {
        self.prefix.is_some() && self.index < self.prefix.as_ref().unwrap().len()
    }
    
    /// Get misalignment information if any occurred
    pub fn get_misalignment(&self) -> Option<&MisalignmentInfo> {
        self.misaligned_at.as_ref()
    }
    
    /// Check if we've hit an overrun condition
    pub fn is_overrun(&self) -> bool {
        self.overrun
    }
    
    /// Reset for a new test case
    pub fn reset(&mut self, prefix: Option<Vec<ChoiceValue>>) {
        self.nodes.clear();
        self.index = 0;
        self.prefix = prefix;
        self.misaligned_at = None;
        self.length = 0;
        self.overrun = false;
    }
    
    /// Get recorded choice nodes
    pub fn get_nodes(&self) -> &[ChoiceNode] {
        &self.nodes
    }
    
    /// Check if a choice value is permitted by the given constraints
    /// Port of Python's choice_permitted function
    fn choice_permitted(&self, value: &ChoiceValue, constraints: &Constraints) -> bool {
        match (value, constraints) {
            (ChoiceValue::Integer(val), Constraints::Integer(int_constraints)) => {
                let min_ok = int_constraints.min_value.map_or(true, |min| *val >= min);
                let max_ok = int_constraints.max_value.map_or(true, |max| *val <= max);
                min_ok && max_ok
            },
            (ChoiceValue::Boolean(_), Constraints::Boolean(_)) => true,
            (ChoiceValue::Float(val), Constraints::Float(float_constraints)) => {
                let min_ok = *val >= float_constraints.min_value;
                let max_ok = *val <= float_constraints.max_value;
                let nan_ok = float_constraints.allow_nan || !val.is_nan();
                min_ok && max_ok && nan_ok
            },
            (ChoiceValue::String(val), Constraints::String(string_constraints)) => {
                val.len() >= string_constraints.min_size && 
                val.len() <= string_constraints.max_size
            },
            (ChoiceValue::Bytes(val), Constraints::Bytes(bytes_constraints)) => {
                val.len() >= bytes_constraints.min_size && 
                val.len() <= bytes_constraints.max_size
            },
            _ => false, // Type mismatch between value and constraints
        }
    }
    
    /// Generate a choice from index (simplest choice strategy)
    /// Port of Python's choice_from_index function
    pub fn choice_from_index(
        &self,
        index: usize,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<ChoiceValue, ChoiceSequenceError> {
        // For index 0, generate the simplest valid choice
        if index != 0 {
            return Err(ChoiceSequenceError::UnsupportedIndex { index });
        }
        
        match (choice_type, constraints) {
            (ChoiceType::Integer, Constraints::Integer(int_constraints)) => {
                let value = int_constraints.min_value.unwrap_or(0);
                Ok(ChoiceValue::Integer(value))
            },
            (ChoiceType::Boolean, Constraints::Boolean(_)) => {
                Ok(ChoiceValue::Boolean(false))
            },
            (ChoiceType::Float, Constraints::Float(float_constraints)) => {
                let value = float_constraints.min_value;
                Ok(ChoiceValue::Float(value))
            },
            (ChoiceType::String, Constraints::String(string_constraints)) => {
                let min_len = string_constraints.min_size;
                let value = "a".repeat(min_len);
                Ok(ChoiceValue::String(value))
            },
            (ChoiceType::Bytes, Constraints::Bytes(bytes_constraints)) => {
                let min_len = bytes_constraints.min_size;
                let value = vec![0u8; min_len];
                Ok(ChoiceValue::Bytes(value))
            },
            _ => Err(ChoiceSequenceError::TypeConstraintMismatch {
                choice_type,
                constraints_type: format!("{:?}", constraints),
            }),
        }
    }
    
    /// Calculate the size of a choice value for buffer management
    fn calculate_choice_size(&self, value: &ChoiceValue) -> usize {
        match value {
            ChoiceValue::Integer(_) => 8,
            ChoiceValue::Boolean(_) => 1,
            ChoiceValue::Float(_) => 8,
            ChoiceValue::String(s) => s.len(),
            ChoiceValue::Bytes(b) => b.len(),
        }
    }
    
    /// Generate a new choice when not replaying from prefix
    /// This would call the provider in a full implementation
    fn generate_choice(
        &self,
        choice_type: ChoiceType,
        constraints: &Constraints,
    ) -> Result<ChoiceValue, ChoiceSequenceError> {
        // For this port, we'll generate deterministic choices
        // In the full implementation, this would delegate to a provider
        self.choice_from_index(0, choice_type, constraints)
    }
    
    /// Mark that we've hit an overrun condition
    fn mark_overrun(&mut self) {
        self.overrun = true;
    }
}

/// Errors that can occur in choice sequence management
#[derive(Debug, Clone)]
pub enum ChoiceSequenceError {
    /// Index out of bounds
    IndexOutOfBounds {
        index: usize,
        max_index: usize,
    },
    /// Buffer overflow
    BufferOverflow {
        required: usize,
        available: usize,
    },
    /// Too many choices recorded
    TooManyChoices {
        count: usize,
        max_choices: usize,
    },
    /// Unsupported choice index
    UnsupportedIndex {
        index: usize,
    },
    /// Type and constraint mismatch
    TypeConstraintMismatch {
        choice_type: ChoiceType,
        constraints_type: String,
    },
}


impl fmt::Display for ChoiceSequenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChoiceSequenceError::IndexOutOfBounds { index, max_index } => {
                write!(f, "Index {} out of bounds (max: {})", index, max_index)
            },
            ChoiceSequenceError::BufferOverflow { required, available } => {
                write!(f, "Buffer overflow: required {} bytes but only {} available", required, available)
            },
            ChoiceSequenceError::TooManyChoices { count, max_choices } => {
                write!(f, "Too many choices: {} exceeds maximum {}", count, max_choices)
            },
            ChoiceSequenceError::UnsupportedIndex { index } => {
                write!(f, "Unsupported choice index: {}", index)
            },
            ChoiceSequenceError::TypeConstraintMismatch { choice_type, constraints_type } => {
                write!(f, "Type {:?} does not match constraints type {}", choice_type, constraints_type)
            },
        }
    }
}

impl std::error::Error for ChoiceSequenceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::{IntegerConstraints, BooleanConstraints, FloatConstraints, StringConstraints};

    #[test]
    fn test_choice_sequence_manager_creation() {
        let manager = ChoiceSequenceManager::new(8192, 1000, None);
        assert_eq!(manager.num_choices(), 0);
        assert!(!manager.is_replaying());
        assert!(!manager.is_overrun());
    }

    #[test]
    fn test_draw_and_record_choice() {
        let mut manager = ChoiceSequenceManager::new(8192, 1000, None);
        
        // Draw an integer choice
        let constraints = Constraints::Integer(IntegerConstraints::default());
        let result = manager.draw(
            ChoiceType::Integer,
            constraints,
            None,
            true, // observe = true to record
        );
        
        assert!(result.is_ok());
        assert_eq!(manager.num_choices(), 1);
        let nodes = manager.get_nodes();
        assert_eq!(nodes[0].choice_type, ChoiceType::Integer);
    }

    #[test]
    fn test_replay_from_prefix() {
        let prefix = vec![
            ChoiceValue::Integer(42),
            ChoiceValue::Boolean(true),
        ];
        let mut manager = ChoiceSequenceManager::new(8192, 1000, Some(prefix));
        
        // Replay first choice
        let constraints = Constraints::Integer(IntegerConstraints::default());
        let result = manager.draw(
            ChoiceType::Integer,
            constraints,
            None,
            true,
        );
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ChoiceValue::Integer(42));
        assert_eq!(manager.num_choices(), 1);
        
        // Replay second choice
        let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
        let bool_result = manager.draw(
            ChoiceType::Boolean,
            bool_constraints,
            None,
            true,
        );
        
        assert!(bool_result.is_ok());
        assert_eq!(bool_result.unwrap(), ChoiceValue::Boolean(true));
        assert_eq!(manager.num_choices(), 2);
    }

    #[test]
    fn test_misalignment_detection() {
        let prefix = vec![ChoiceValue::Integer(42)];
        let mut manager = ChoiceSequenceManager::new(8192, 1000, Some(prefix));
        
        // Try to draw a boolean when prefix has integer - should detect misalignment
        let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
        let result = manager.draw(
            ChoiceType::Boolean,
            bool_constraints,
            None,
            true,
        );
        
        assert!(result.is_ok()); // Should succeed with replacement choice
        assert_eq!(result.unwrap(), ChoiceValue::Boolean(false)); // Simplest choice
        
        // Check that misalignment was detected
        let misalignment = manager.get_misalignment();
        assert!(misalignment.is_some());
        let misalign_info = misalignment.unwrap();
        assert_eq!(misalign_info.index, 0);
        assert_eq!(misalign_info.choice_type, ChoiceType::Boolean);
    }

    #[test]
    fn test_choice_permitted() {
        let manager = ChoiceSequenceManager::new(8192, 1000, None);
        
        // Test integer constraints
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(10),
            max_value: Some(20),
            weights: None,
            shrink_towards: Some(10),
        });
        
        assert!(manager.choice_permitted(&ChoiceValue::Integer(15), &int_constraints));
        assert!(!manager.choice_permitted(&ChoiceValue::Integer(5), &int_constraints));
        assert!(!manager.choice_permitted(&ChoiceValue::Integer(25), &int_constraints));
        
        // Test type mismatch
        assert!(!manager.choice_permitted(&ChoiceValue::Boolean(true), &int_constraints));
    }

    #[test]
    fn test_choice_from_index() {
        let manager = ChoiceSequenceManager::new(8192, 1000, None);
        
        // Test integer simplest choice
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(10),
            max_value: Some(20),
            weights: None,
            shrink_towards: Some(10),
        });
        let result = manager.choice_from_index(0, ChoiceType::Integer, &int_constraints);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ChoiceValue::Integer(10));
        
        // Test boolean simplest choice
        let bool_constraints = Constraints::Boolean(BooleanConstraints::default());
        let bool_result = manager.choice_from_index(0, ChoiceType::Boolean, &bool_constraints);
        assert!(bool_result.is_ok());
        assert_eq!(bool_result.unwrap(), ChoiceValue::Boolean(false));
    }

    #[test]
    fn test_reset() {
        let mut manager = ChoiceSequenceManager::new(8192, 1000, None);
        
        // Draw some choices
        let constraints = Constraints::Boolean(BooleanConstraints::default());
        let _ = manager.draw(ChoiceType::Boolean, constraints, None, true);
        assert_eq!(manager.num_choices(), 1);
        
        // Reset with new prefix
        let new_prefix = vec![ChoiceValue::Integer(99)];
        manager.reset(Some(new_prefix));
        
        assert_eq!(manager.num_choices(), 0);
        assert!(manager.is_replaying());
        assert_eq!(manager.index, 0);
        assert!(manager.get_misalignment().is_none());
    }
}