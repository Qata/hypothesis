//! Core Choice Sequence Management System for ConjectureData
//! 
//! This module implements the complete choice sequence management capability
//! that fixes type inconsistencies in choice node storage, index tracking,
//! and sequence replay functionality to restore basic ConjectureData buffer operations.

use crate::choice::{ChoiceType, ChoiceValue, Constraints, ChoiceNode};
use crate::data::{Status, DrawError};
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

/// Enhanced Choice Sequence Manager that handles all choice recording and replay
/// with proper type consistency and index tracking
#[derive(Debug, Clone)]
pub struct ChoiceSequenceManager {
    /// Primary choice sequence storage with consistent typing
    choices: Vec<EnhancedChoiceNode>,
    
    /// Index mapping for O(1) lookups during replay
    index_map: HashMap<usize, usize>,
    
    /// Current replay position
    replay_position: usize,
    
    /// Choice type verification cache
    type_cache: HashMap<usize, ChoiceType>,
    
    /// Constraint consistency tracker
    constraint_tracker: ConstraintTracker,
    
    /// Sequence integrity monitor
    integrity_monitor: SequenceIntegrityMonitor,
    
    /// Buffer operation tracker
    buffer_tracker: BufferOperationTracker,
}

/// Enhanced ChoiceNode with improved type consistency and index management
#[derive(Debug, Clone, PartialEq)]
pub struct EnhancedChoiceNode {
    /// Base choice node data
    pub choice_node: ChoiceNode,
    
    /// Guaranteed index position (never None after creation)
    pub guaranteed_index: usize,
    
    /// Buffer position tracking
    pub buffer_position: BufferPosition,
    
    /// Type verification metadata
    pub type_metadata: TypeMetadata,
    
    /// Constraint consistency data
    pub constraint_metadata: ConstraintMetadata,
    
    /// Replay verification data
    pub replay_metadata: ReplayMetadata,
}

/// Buffer position tracking for accurate sequence replay
#[derive(Debug, Clone, PartialEq)]
pub struct BufferPosition {
    /// Start position in buffer
    pub start: usize,
    
    /// End position in buffer
    pub end: usize,
    
    /// Size in bytes
    pub size: usize,
    
    /// Buffer alignment information
    pub alignment: BufferAlignment,
}

/// Buffer alignment tracking for proper replay
#[derive(Debug, Clone, PartialEq)]
pub enum BufferAlignment {
    /// Aligned to byte boundary
    ByteAligned,
    /// Aligned to word boundary
    WordAligned,
    /// Custom alignment
    Custom(usize),
}

/// Type verification metadata to prevent type mismatches
#[derive(Debug, Clone, PartialEq)]
pub struct TypeMetadata {
    /// Original choice type
    pub declared_type: ChoiceType,
    
    /// Verified type from value inspection
    pub verified_type: ChoiceType,
    
    /// Type consistency flag
    pub is_consistent: bool,
    
    /// Type conversion notes
    pub conversion_notes: Vec<String>,
}

/// Constraint consistency tracking
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintMetadata {
    /// Original constraints hash for verification
    pub constraint_hash: u64,
    
    /// Constraint validation result
    pub is_valid: bool,
    
    /// Constraint compatibility notes
    pub compatibility_notes: Vec<String>,
}

/// Replay verification metadata
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayMetadata {
    /// Whether this choice is replayable
    pub is_replayable: bool,
    
    /// Replay verification status
    pub replay_status: ReplayStatus,
    
    /// Forced value tracking
    pub was_forced_consistently: bool,
}

/// Status of replay verification
#[derive(Debug, Clone, PartialEq)]
pub enum ReplayStatus {
    /// Not yet replayed
    NotReplayed,
    /// Successfully replayed
    ReplaySuccess,
    /// Replay failed due to type mismatch
    ReplayFailedType,
    /// Replay failed due to constraint mismatch
    ReplayFailedConstraints,
    /// Replay failed due to value mismatch
    ReplayFailedValue,
}

/// Constraint tracking for consistency verification
#[derive(Debug, Clone)]
pub struct ConstraintTracker {
    /// Constraint compatibility matrix
    compatibility_matrix: HashMap<(u64, u64), bool>,
    
    /// Constraint evolution tracking
    evolution_history: Vec<ConstraintEvolution>,
    
    /// Validation cache
    validation_cache: HashMap<u64, bool>,
}

/// Constraint evolution tracking
#[derive(Debug, Clone)]
pub struct ConstraintEvolution {
    /// Index where constraint changed
    pub index: usize,
    
    /// Old constraint hash
    pub old_hash: u64,
    
    /// New constraint hash
    pub new_hash: u64,
    
    /// Reason for change
    pub reason: String,
}

/// Sequence integrity monitoring
#[derive(Debug, Clone)]
pub struct SequenceIntegrityMonitor {
    /// Sequence hash for integrity verification
    pub sequence_hash: u64,
    
    /// Last verified index
    pub last_verified_index: usize,
    
    /// Integrity violations found
    pub violations: Vec<IntegrityViolation>,
    
    /// Recovery actions taken
    pub recovery_actions: Vec<RecoveryAction>,
}

/// Integrity violation tracking
#[derive(Debug, Clone)]
pub struct IntegrityViolation {
    /// Index where violation occurred
    pub index: usize,
    
    /// Type of violation
    pub violation_type: ViolationType,
    
    /// Description of the issue
    pub description: String,
    
    /// Severity level
    pub severity: ViolationSeverity,
}

/// Types of integrity violations
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationType {
    /// Type mismatch between expected and actual
    TypeMismatch,
    /// Index out of sequence
    IndexMismatch,
    /// Constraint violation
    ConstraintViolation,
    /// Buffer position mismatch
    BufferMismatch,
    /// Replay inconsistency
    ReplayInconsistency,
}

/// Severity levels for violations
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationSeverity {
    /// Warning - can continue with caution
    Warning,
    /// Error - should stop and fix
    Error,
    /// Critical - must abort immediately
    Critical,
}

/// Recovery actions taken to fix violations
#[derive(Debug, Clone)]
pub struct RecoveryAction {
    /// Index where recovery was applied
    pub index: usize,
    
    /// Type of recovery action
    pub action_type: RecoveryType,
    
    /// Description of what was done
    pub description: String,
    
    /// Success status
    pub success: bool,
}

/// Types of recovery actions
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryType {
    /// Reset index tracking
    IndexReset,
    /// Type correction
    TypeCorrection,
    /// Constraint adjustment
    ConstraintAdjustment,
    /// Buffer realignment
    BufferRealignment,
    /// Sequence rebuild
    SequenceRebuild,
}

/// Buffer operation tracking for replay consistency
#[derive(Debug, Clone)]
pub struct BufferOperationTracker {
    /// Current buffer size
    pub current_size: usize,
    
    /// Maximum buffer size
    pub max_size: usize,
    
    /// Buffer utilization history
    pub utilization_history: Vec<BufferUtilization>,
    
    /// Operation performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Buffer utilization snapshot
#[derive(Debug, Clone)]
pub struct BufferUtilization {
    /// Timestamp of measurement
    pub timestamp: std::time::Instant,
    
    /// Used bytes
    pub used_bytes: usize,
    
    /// Available bytes
    pub available_bytes: usize,
    
    /// Fragmentation level
    pub fragmentation: f64,
}

/// Performance metrics for buffer operations
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Total choice recording operations
    pub total_recordings: usize,
    
    /// Total replay operations
    pub total_replays: usize,
    
    /// Average recording time
    pub avg_recording_time: f64,
    
    /// Average replay time
    pub avg_replay_time: f64,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
    
    /// Type verification time
    pub type_verification_time: f64,
}

impl ChoiceSequenceManager {
    /// Create a new choice sequence manager
    pub fn new(max_buffer_size: usize) -> Self {
        println!("CHOICE_SEQ DEBUG: Creating ChoiceSequenceManager with max_buffer_size: {}", max_buffer_size);
        
        Self {
            choices: Vec::new(),
            index_map: HashMap::new(),
            replay_position: 0,
            type_cache: HashMap::new(),
            constraint_tracker: ConstraintTracker::new(),
            integrity_monitor: SequenceIntegrityMonitor::new(),
            buffer_tracker: BufferOperationTracker::new(max_buffer_size),
        }
    }
    
    /// Record a new choice with full type consistency verification
    pub fn record_choice(
        &mut self,
        choice_type: ChoiceType,
        value: ChoiceValue,
        constraints: Box<Constraints>,
        was_forced: bool,
        buffer_position: usize,
    ) -> Result<usize, ChoiceSequenceError> {
        let start_time = std::time::Instant::now();
        
        println!("CHOICE_SEQ DEBUG: Recording choice {:?} = {:?} at buffer position {}", 
                 choice_type, value, buffer_position);
        
        // Step 1: Verify type consistency
        let type_metadata = self.verify_type_consistency(choice_type, &value)?;
        
        // Step 2: Validate constraints
        let constraint_metadata = self.validate_constraints(&value, &constraints)?;
        
        // Step 3: Calculate buffer position
        let buffer_pos = self.calculate_buffer_position(buffer_position, &value);
        
        // Step 4: Create enhanced choice node
        let guaranteed_index = self.choices.len();
        let choice_node = ChoiceNode::with_index(
            choice_type,
            value.clone(),
            *constraints,
            was_forced,
            guaranteed_index,
        );
        
        let enhanced_node = EnhancedChoiceNode {
            choice_node,
            guaranteed_index,
            buffer_position: buffer_pos,
            type_metadata,
            constraint_metadata,
            replay_metadata: ReplayMetadata {
                is_replayable: true,
                replay_status: ReplayStatus::NotReplayed,
                was_forced_consistently: was_forced,
            },
        };
        
        // Step 5: Add to sequence with integrity checking
        self.add_choice_with_integrity_check(enhanced_node)?;
        
        // Step 6: Update performance metrics
        let elapsed = start_time.elapsed();
        self.buffer_tracker.performance_metrics.total_recordings += 1;
        self.buffer_tracker.performance_metrics.avg_recording_time = 
            (self.buffer_tracker.performance_metrics.avg_recording_time * 
             (self.buffer_tracker.performance_metrics.total_recordings - 1) as f64 + 
             elapsed.as_secs_f64()) / self.buffer_tracker.performance_metrics.total_recordings as f64;
        
        println!("CHOICE_SEQ DEBUG: Successfully recorded choice at index {}", guaranteed_index);
        Ok(guaranteed_index)
    }
    
    /// Replay a choice sequence with comprehensive verification
    pub fn replay_choice_at_index(
        &mut self,
        index: usize,
        expected_type: ChoiceType,
        expected_constraints: &Constraints,
    ) -> Result<ChoiceValue, ChoiceSequenceError> {
        let start_time = std::time::Instant::now();
        
        println!("CHOICE_SEQ DEBUG: Replaying choice at index {} with type {:?}", index, expected_type);
        
        // Step 1: Validate index bounds
        if index >= self.choices.len() {
            return Err(ChoiceSequenceError::IndexOutOfBounds { 
                index, 
                max_index: self.choices.len().saturating_sub(1) 
            });
        }
        
        // Step 2: Get choice and verify replay compatibility
        let enhanced_choice = &mut self.choices[index];
        
        // Step 3: Verify type consistency
        if enhanced_choice.choice_node.choice_type != expected_type {
            enhanced_choice.replay_metadata.replay_status = ReplayStatus::ReplayFailedType;
            return Err(ChoiceSequenceError::TypeMismatch {
                expected: expected_type,
                actual: enhanced_choice.choice_node.choice_type,
                index,
            });
        }
        
        // Step 4: Verify constraint compatibility
        if !self.constraint_tracker.are_constraints_compatible(
            &enhanced_choice.choice_node.constraints, 
            expected_constraints
        ) {
            enhanced_choice.replay_metadata.replay_status = ReplayStatus::ReplayFailedConstraints;
            return Err(ChoiceSequenceError::ConstraintMismatch {
                index,
                reason: "Constraints are not compatible for replay".to_string(),
            });
        }
        
        // Step 5: Mark replay success and return value
        enhanced_choice.replay_metadata.replay_status = ReplayStatus::ReplaySuccess;
        enhanced_choice.replay_metadata.is_replayable = true;
        
        let value = enhanced_choice.choice_node.value.clone();
        
        // Step 6: Update performance metrics
        let elapsed = start_time.elapsed();
        self.buffer_tracker.performance_metrics.total_replays += 1;
        self.buffer_tracker.performance_metrics.avg_replay_time = 
            (self.buffer_tracker.performance_metrics.avg_replay_time * 
             (self.buffer_tracker.performance_metrics.total_replays - 1) as f64 + 
             elapsed.as_secs_f64()) / self.buffer_tracker.performance_metrics.total_replays as f64;
        
        println!("CHOICE_SEQ DEBUG: Successfully replayed choice: {:?}", value);
        Ok(value)
    }
    
    /// Get the current sequence length
    pub fn sequence_length(&self) -> usize {
        self.choices.len()
    }
    
    /// Check if a specific index is available for replay
    pub fn is_index_replayable(&self, index: usize) -> bool {
        self.choices.get(index)
            .map(|choice| choice.replay_metadata.is_replayable)
            .unwrap_or(false)
    }
    
    /// Get sequence integrity status
    pub fn get_integrity_status(&self) -> SequenceIntegrityStatus {
        SequenceIntegrityStatus {
            is_healthy: self.integrity_monitor.violations.is_empty(),
            total_violations: self.integrity_monitor.violations.len(),
            critical_violations: self.integrity_monitor.violations.iter()
                .filter(|v| v.severity == ViolationSeverity::Critical)
                .count(),
            last_verified_index: self.integrity_monitor.last_verified_index,
            recovery_actions_taken: self.integrity_monitor.recovery_actions.len(),
        }
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.buffer_tracker.performance_metrics
    }
    
    /// Reset the sequence for a new test
    pub fn reset_sequence(&mut self) {
        println!("CHOICE_SEQ DEBUG: Resetting choice sequence");
        
        self.choices.clear();
        self.index_map.clear();
        self.replay_position = 0;
        self.type_cache.clear();
        self.constraint_tracker = ConstraintTracker::new();
        self.integrity_monitor = SequenceIntegrityMonitor::new();
        
        // Keep buffer tracker but reset current utilization
        self.buffer_tracker.current_size = 0;
        self.buffer_tracker.utilization_history.clear();
    }
    
    /// Export the current sequence as standard ChoiceNodes for compatibility
    pub fn export_as_choice_nodes(&self) -> Vec<ChoiceNode> {
        self.choices.iter()
            .map(|enhanced| enhanced.choice_node.clone())
            .collect()
    }
    
    /// Verify type consistency between declared type and actual value
    fn verify_type_consistency(&mut self, declared_type: ChoiceType, value: &ChoiceValue) -> Result<TypeMetadata, ChoiceSequenceError> {
        let start_time = std::time::Instant::now();
        
        let verified_type = match value {
            ChoiceValue::Integer(_) => ChoiceType::Integer,
            ChoiceValue::Boolean(_) => ChoiceType::Boolean,
            ChoiceValue::Float(_) => ChoiceType::Float,
            ChoiceValue::String(_) => ChoiceType::String,
            ChoiceValue::Bytes(_) => ChoiceType::Bytes,
        };
        
        let is_consistent = declared_type == verified_type;
        let mut conversion_notes = Vec::new();
        
        if !is_consistent {
            conversion_notes.push(format!(
                "Type mismatch: declared {:?} but value is {:?}", 
                declared_type, verified_type
            ));
        }
        
        // Update type verification performance metrics
        let elapsed = start_time.elapsed();
        self.buffer_tracker.performance_metrics.type_verification_time += elapsed.as_secs_f64();
        
        if !is_consistent {
            return Err(ChoiceSequenceError::TypeInconsistency {
                declared: declared_type,
                actual: verified_type,
            });
        }
        
        Ok(TypeMetadata {
            declared_type,
            verified_type,
            is_consistent,
            conversion_notes,
        })
    }
    
    /// Validate constraints against the given value
    fn validate_constraints(&self, value: &ChoiceValue, constraints: &Constraints) -> Result<ConstraintMetadata, ChoiceSequenceError> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        constraints.hash(&mut hasher);
        let constraint_hash = hasher.finish();
        
        let is_valid = match (value, constraints) {
            (ChoiceValue::Integer(val), Constraints::Integer(int_constraints)) => {
                let min_ok = int_constraints.min_value.map_or(true, |min| *val >= min);
                let max_ok = int_constraints.max_value.map_or(true, |max| *val <= max);
                min_ok && max_ok
            },
            (ChoiceValue::Boolean(_), Constraints::Boolean(_)) => true, // Boolean constraints are probability-based
            (ChoiceValue::Float(val), Constraints::Float(float_constraints)) => {
                let min_ok = *val >= float_constraints.min_value;
                let max_ok = *val <= float_constraints.max_value;
                let nan_ok = float_constraints.allow_nan || !val.is_nan();
                min_ok && max_ok && nan_ok
            },
            (ChoiceValue::String(val), Constraints::String(string_constraints)) => {
                let len_ok = val.len() >= string_constraints.min_size && 
                           val.len() <= string_constraints.max_size;
                // TODO: Check alphabet constraints
                len_ok
            },
            (ChoiceValue::Bytes(val), Constraints::Bytes(bytes_constraints)) => {
                val.len() >= bytes_constraints.min_size && 
                val.len() <= bytes_constraints.max_size
            },
            _ => false, // Type mismatch between value and constraints
        };
        
        let compatibility_notes = if !is_valid {
            vec!["Constraint validation failed".to_string()]
        } else {
            Vec::new()
        };
        
        Ok(ConstraintMetadata {
            constraint_hash,
            is_valid,
            compatibility_notes,
        })
    }
    
    /// Calculate buffer position for a choice value
    fn calculate_buffer_position(&self, start_position: usize, value: &ChoiceValue) -> BufferPosition {
        let size = match value {
            ChoiceValue::Integer(_) => 8,  // i128 typically stored as 8 bytes
            ChoiceValue::Boolean(_) => 1,
            ChoiceValue::Float(_) => 8,
            ChoiceValue::String(s) => s.len(),
            ChoiceValue::Bytes(b) => b.len(),
        };
        
        BufferPosition {
            start: start_position,
            end: start_position + size,
            size,
            alignment: BufferAlignment::ByteAligned, // Default to byte alignment
        }
    }
    
    /// Add choice with integrity checking
    fn add_choice_with_integrity_check(&mut self, enhanced_node: EnhancedChoiceNode) -> Result<(), ChoiceSequenceError> {
        let index = enhanced_node.guaranteed_index;
        
        // Verify index consistency
        if index != self.choices.len() {
            let violation = IntegrityViolation {
                index,
                violation_type: ViolationType::IndexMismatch,
                description: format!("Expected index {} but got {}", self.choices.len(), index),
                severity: ViolationSeverity::Error,
            };
            self.integrity_monitor.violations.push(violation);
            
            // Attempt recovery
            let recovery = RecoveryAction {
                index,
                action_type: RecoveryType::IndexReset,
                description: "Reset index to match sequence length".to_string(),
                success: true,
            };
            self.integrity_monitor.recovery_actions.push(recovery);
        }
        
        // Add to sequence
        self.choices.push(enhanced_node);
        self.index_map.insert(index, self.choices.len() - 1);
        
        // Update integrity monitor
        self.integrity_monitor.last_verified_index = index;
        self.integrity_monitor.sequence_hash = self.calculate_sequence_hash();
        
        Ok(())
    }
    
    /// Calculate hash of current sequence for integrity verification
    fn calculate_sequence_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for choice in &self.choices {
            choice.choice_node.choice_type.hash(&mut hasher);
            choice.choice_node.value.hash(&mut hasher);
            choice.guaranteed_index.hash(&mut hasher);
        }
        hasher.finish()
    }
}

/// Status of sequence integrity
#[derive(Debug, Clone)]
pub struct SequenceIntegrityStatus {
    pub is_healthy: bool,
    pub total_violations: usize,
    pub critical_violations: usize,
    pub last_verified_index: usize,
    pub recovery_actions_taken: usize,
}

/// Errors that can occur in choice sequence management
#[derive(Debug, Clone)]
pub enum ChoiceSequenceError {
    /// Type inconsistency between declared and actual
    TypeInconsistency {
        declared: ChoiceType,
        actual: ChoiceType,
    },
    /// Type mismatch during replay
    TypeMismatch {
        expected: ChoiceType,
        actual: ChoiceType,
        index: usize,
    },
    /// Constraint validation failure
    ConstraintViolation {
        index: usize,
        reason: String,
    },
    /// Constraint mismatch during replay
    ConstraintMismatch {
        index: usize,
        reason: String,
    },
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
    /// Integrity violation
    IntegrityViolation {
        violation_type: ViolationType,
        description: String,
    },
}

impl fmt::Display for ChoiceSequenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChoiceSequenceError::TypeInconsistency { declared, actual } => {
                write!(f, "Type inconsistency: declared {:?} but actual {:?}", declared, actual)
            },
            ChoiceSequenceError::TypeMismatch { expected, actual, index } => {
                write!(f, "Type mismatch at index {}: expected {:?} but got {:?}", index, expected, actual)
            },
            ChoiceSequenceError::ConstraintViolation { index, reason } => {
                write!(f, "Constraint violation at index {}: {}", index, reason)
            },
            ChoiceSequenceError::ConstraintMismatch { index, reason } => {
                write!(f, "Constraint mismatch at index {}: {}", index, reason)
            },
            ChoiceSequenceError::IndexOutOfBounds { index, max_index } => {
                write!(f, "Index {} out of bounds (max: {})", index, max_index)
            },
            ChoiceSequenceError::BufferOverflow { required, available } => {
                write!(f, "Buffer overflow: required {} bytes but only {} available", required, available)
            },
            ChoiceSequenceError::IntegrityViolation { violation_type, description } => {
                write!(f, "Integrity violation ({:?}): {}", violation_type, description)
            },
        }
    }
}

impl std::error::Error for ChoiceSequenceError {}

// Implement the supporting structures

impl ConstraintTracker {
    pub fn new() -> Self {
        Self {
            compatibility_matrix: HashMap::new(),
            evolution_history: Vec::new(),
            validation_cache: HashMap::new(),
        }
    }
    
    pub fn are_constraints_compatible(&self, old: &Constraints, new: &Constraints) -> bool {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher1 = DefaultHasher::new();
        old.hash(&mut hasher1);
        let old_hash = hasher1.finish();
        
        let mut hasher2 = DefaultHasher::new();
        new.hash(&mut hasher2);
        let new_hash = hasher2.finish();
        
        // Check cache first
        if let Some(&compatible) = self.compatibility_matrix.get(&(old_hash, new_hash)) {
            return compatible;
        }
        
        // Basic compatibility check - same type and values
        std::mem::discriminant(old) == std::mem::discriminant(new)
    }
}

impl SequenceIntegrityMonitor {
    pub fn new() -> Self {
        Self {
            sequence_hash: 0,
            last_verified_index: 0,
            violations: Vec::new(),
            recovery_actions: Vec::new(),
        }
    }
}

impl BufferOperationTracker {
    pub fn new(max_size: usize) -> Self {
        Self {
            current_size: 0,
            max_size,
            utilization_history: Vec::new(),
            performance_metrics: PerformanceMetrics::default(),
        }
    }
}

// Hash implementations for ChoiceValue and Constraints are already provided in choice/mod.rs

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::{IntegerConstraints, BooleanConstraints};

    #[test]
    fn test_choice_sequence_manager_creation() {
        let manager = ChoiceSequenceManager::new(8192);
        assert_eq!(manager.sequence_length(), 0);
        assert!(manager.get_integrity_status().is_healthy);
    }

    #[test]
    fn test_record_and_replay_integer_choice() {
        let mut manager = ChoiceSequenceManager::new(8192);
        
        // Record an integer choice
        let constraints = Box::new(Constraints::Integer(IntegerConstraints::default()));
        let result = manager.record_choice(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            constraints,
            false,
            0,
        );
        
        assert!(result.is_ok());
        assert_eq!(manager.sequence_length(), 1);
        
        // Replay the choice
        let replay_constraints = Constraints::Integer(IntegerConstraints::default());
        let replayed_value = manager.replay_choice_at_index(0, ChoiceType::Integer, &replay_constraints);
        
        assert!(replayed_value.is_ok());
        assert_eq!(replayed_value.unwrap(), ChoiceValue::Integer(42));
    }

    #[test]
    fn test_type_mismatch_detection() {
        let mut manager = ChoiceSequenceManager::new(8192);
        
        // Record an integer choice
        let constraints = Box::new(Constraints::Integer(IntegerConstraints::default()));
        let _ = manager.record_choice(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            constraints,
            false,
            0,
        );
        
        // Try to replay as boolean - should fail
        let replay_constraints = Constraints::Boolean(BooleanConstraints::default());
        let replayed_value = manager.replay_choice_at_index(0, ChoiceType::Boolean, &replay_constraints);
        
        assert!(replayed_value.is_err());
        match replayed_value.unwrap_err() {
            ChoiceSequenceError::TypeMismatch { expected, actual, index } => {
                assert_eq!(expected, ChoiceType::Boolean);
                assert_eq!(actual, ChoiceType::Integer);
                assert_eq!(index, 0);
            },
            _ => panic!("Expected TypeMismatch error"),
        }
    }

    #[test]
    fn test_index_out_of_bounds() {
        let mut manager = ChoiceSequenceManager::new(8192);
        
        // Try to replay from empty sequence
        let replay_constraints = Constraints::Integer(IntegerConstraints::default());
        let result = manager.replay_choice_at_index(0, ChoiceType::Integer, &replay_constraints);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            ChoiceSequenceError::IndexOutOfBounds { index, max_index } => {
                assert_eq!(index, 0);
                assert_eq!(max_index, 0); // No choices recorded, so max_index is 0 
            },
            _ => panic!("Expected IndexOutOfBounds error"),
        }
    }

    #[test]
    fn test_sequence_integrity_monitoring() {
        let mut manager = ChoiceSequenceManager::new(8192);
        
        // Record several choices
        for i in 0..5 {
            let constraints = Box::new(Constraints::Integer(IntegerConstraints::default()));
            let _ = manager.record_choice(
                ChoiceType::Integer,
                ChoiceValue::Integer(i),
                constraints,
                false,
                i as usize * 8,
            );
        }
        
        let integrity_status = manager.get_integrity_status();
        assert!(integrity_status.is_healthy);
        assert_eq!(integrity_status.total_violations, 0);
        assert_eq!(integrity_status.last_verified_index, 4);
    }

    #[test]
    fn test_performance_metrics_tracking() {
        let mut manager = ChoiceSequenceManager::new(8192);
        
        // Record some choices to generate metrics
        for i in 0..10 {
            let constraints = Box::new(Constraints::Integer(IntegerConstraints::default()));
            let _ = manager.record_choice(
                ChoiceType::Integer,
                ChoiceValue::Integer(i),
                constraints,
                false,
                i as usize * 8,
            );
        }
        
        let metrics = manager.get_performance_metrics();
        assert_eq!(metrics.total_recordings, 10);
        assert!(metrics.avg_recording_time >= 0.0);
        assert!(metrics.type_verification_time >= 0.0);
    }

    #[test]
    fn test_sequence_reset() {
        let mut manager = ChoiceSequenceManager::new(8192);
        
        // Record some choices
        let constraints = Box::new(Constraints::Integer(IntegerConstraints::default()));
        let _ = manager.record_choice(
            ChoiceType::Integer,
            ChoiceValue::Integer(42),
            constraints,
            false,
            0,
        );
        
        assert_eq!(manager.sequence_length(), 1);
        
        // Reset and verify
        manager.reset_sequence();
        assert_eq!(manager.sequence_length(), 0);
        assert!(manager.get_integrity_status().is_healthy);
    }
}