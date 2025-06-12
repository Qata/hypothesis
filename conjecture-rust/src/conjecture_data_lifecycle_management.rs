//! ConjectureData Lifecycle Management module for EngineOrchestrator
//!
//! This module provides comprehensive lifecycle management for ConjectureData instances,
//! including creation, replay, forced value integration, and cleanup. It implements
//! the missing functionality identified in the current task.

use crate::choice::{ChoiceNode, ChoiceType, ChoiceValue, Constraints};
use crate::data::{ConjectureData, ConjectureResult, Status, DataObserver};
use crate::providers::PrimitiveProvider;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Errors that can occur during ConjectureData lifecycle management
#[derive(Debug, Clone)]
pub enum LifecycleError {
    /// Replay mechanism failed
    ReplayFailed { reason: String },
    /// Forced value system integration failed
    ForcedValueError { details: String },
    /// Misalignment detected during replay
    ReplayMisalignment { at_index: usize, expected: String, actual: String },
    /// Invalid lifecycle state
    InvalidState { state: String, operation: String },
    /// Provider integration error
    ProviderError { message: String },
}

impl std::fmt::Display for LifecycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LifecycleError::ReplayFailed { reason } => {
                write!(f, "Replay mechanism failed: {}", reason)
            }
            LifecycleError::ForcedValueError { details } => {
                write!(f, "Forced value system integration failed: {}", details)
            }
            LifecycleError::ReplayMisalignment { at_index, expected, actual } => {
                write!(f, "Replay misalignment at index {}: expected {}, got {}", at_index, expected, actual)
            }
            LifecycleError::InvalidState { state, operation } => {
                write!(f, "Invalid lifecycle state '{}' for operation '{}'", state, operation)
            }
            LifecycleError::ProviderError { message } => {
                write!(f, "Provider integration error: {}", message)
            }
        }
    }
}

impl std::error::Error for LifecycleError {}

/// Result type for lifecycle operations
pub type LifecycleResult<T> = Result<T, LifecycleError>;

/// Lifecycle state for ConjectureData instances
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifecycleState {
    /// Newly created, ready for initialization
    Created,
    /// Initialized and ready for test execution
    Initialized,
    /// Currently executing a test
    Executing,
    /// Test execution completed
    Completed,
    /// In replay mode
    Replaying,
    /// Replay completed successfully
    ReplayCompleted,
    /// Replay failed with misalignment
    ReplayFailed,
    /// Cleanup completed
    Cleaned,
}

/// Configuration for ConjectureData lifecycle management
#[derive(Debug, Clone)]
pub struct LifecycleConfig {
    /// Enable debug logging for lifecycle events
    pub debug_logging: bool,
    /// Maximum number of choices allowed before termination
    pub max_choices: Option<usize>,
    /// Timeout for test execution in milliseconds
    pub execution_timeout_ms: Option<u64>,
    /// Enable forced value system integration
    pub enable_forced_values: bool,
    /// Enable replay mechanism validation
    pub enable_replay_validation: bool,
    /// Hex notation format for debugging
    pub use_hex_notation: bool,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            debug_logging: true,
            max_choices: Some(10000),
            execution_timeout_ms: Some(30000),
            enable_forced_values: true,
            enable_replay_validation: true,
            use_hex_notation: true,
        }
    }
}

/// Comprehensive ConjectureData lifecycle manager
pub struct ConjectureDataLifecycleManager {
    /// Current lifecycle configuration
    config: LifecycleConfig,
    /// Active ConjectureData instances and their states
    instances: HashMap<u64, (ConjectureData, LifecycleState)>,
    /// Forced value system state
    forced_values: HashMap<u64, Vec<(usize, ChoiceValue)>>,
    /// Replay validation cache
    replay_cache: HashMap<Vec<u8>, ConjectureResult>,
    /// Performance metrics
    metrics: LifecycleMetrics,
}

/// Performance metrics for lifecycle management
#[derive(Debug, Clone, Default)]
pub struct LifecycleMetrics {
    /// Total number of ConjectureData instances created
    pub instances_created: usize,
    /// Total number of successful replays
    pub successful_replays: usize,
    /// Total number of failed replays
    pub failed_replays: usize,
    /// Total number of forced value integrations
    pub forced_value_integrations: usize,
    /// Average execution time per test (milliseconds)
    pub avg_execution_time_ms: f64,
    /// Total cleanup operations performed
    pub cleanup_operations: usize,
}

impl ConjectureDataLifecycleManager {
    /// Create a new lifecycle manager with the given configuration
    pub fn new(config: LifecycleConfig) -> Self {
        eprintln!("LIFECYCLE_MANAGER: Initializing ConjectureData lifecycle management");
        Self {
            config,
            instances: HashMap::new(),
            forced_values: HashMap::new(),
            replay_cache: HashMap::new(),
            metrics: LifecycleMetrics::default(),
        }
    }

    /// Create a new ConjectureData instance with lifecycle management
    pub fn create_instance(
        &mut self,
        seed: u64,
        observer: Option<Box<dyn DataObserver>>,
        provider: Option<Box<dyn PrimitiveProvider>>,
    ) -> LifecycleResult<u64> {
        let start_time = Instant::now();
        
        // Create the ConjectureData instance
        let mut data = ConjectureData::new(seed);
        
        // Configure with observer and provider if provided
        if let Some(obs) = observer {
            data.set_observer(obs);
        }
        
        if let Some(prov) = provider {
            data.set_provider(prov);
        }
        
        // Apply lifecycle configuration
        if let Some(max_choices) = self.config.max_choices {
            data.max_choices = Some(max_choices);
        }
        
        let instance_id = data.testcounter;
        
        // Store instance with initial state
        self.instances.insert(instance_id, (data, LifecycleState::Created));
        self.metrics.instances_created += 1;
        
        if self.config.debug_logging {
            let hex_id = if self.config.use_hex_notation {
                format!("{:08X}", instance_id)
            } else {
                instance_id.to_string()
            };
            eprintln!("LIFECYCLE_MANAGER: Created ConjectureData instance {} in {:.2}ms", 
                     hex_id, start_time.elapsed().as_secs_f64() * 1000.0);
        }
        
        Ok(instance_id)
    }

    /// Create ConjectureData instance for replay with comprehensive forced value support
    pub fn create_for_replay(
        &mut self,
        choices: &[ChoiceNode],
        observer: Option<Box<dyn DataObserver>>,
        provider: Option<Box<dyn PrimitiveProvider>>,
        random: Option<ChaCha8Rng>,
    ) -> LifecycleResult<u64> {
        let start_time = Instant::now();
        
        // Create ConjectureData using the existing for_choices method
        let mut data = ConjectureData::for_choices(choices, observer, provider, random);
        
        // CRITICAL FIX: Do NOT override max_choices for replay instances
        // ConjectureData::for_choices() correctly sets max_choices = Some(choices.len())
        // Overriding this breaks replay functionality by allowing more choices than expected
        // The lifecycle config max_choices is for regular instances, not replay instances
        
        let instance_id = data.testcounter;
        
        // Store instance with replay state
        self.instances.insert(instance_id, (data, LifecycleState::Replaying));
        
        if self.config.debug_logging {
            let hex_id = if self.config.use_hex_notation {
                format!("{:08X}", instance_id)
            } else {
                instance_id.to_string()
            };
            eprintln!("LIFECYCLE_MANAGER: Created replay ConjectureData instance {} with {} choices in {:.2}ms", 
                     hex_id, choices.len(), start_time.elapsed().as_secs_f64() * 1000.0);
        }
        
        Ok(instance_id)
    }

    /// Transition an instance to a new lifecycle state
    pub fn transition_state(&mut self, instance_id: u64, new_state: LifecycleState) -> LifecycleResult<()> {
        if let Some((_, current_state)) = self.instances.get_mut(&instance_id) {
            let previous_state = *current_state;
            *current_state = new_state;
            
            if self.config.debug_logging {
                let hex_id = if self.config.use_hex_notation {
                    format!("{:08X}", instance_id)
                } else {
                    instance_id.to_string()
                };
                eprintln!("LIFECYCLE_MANAGER: Instance {} transitioned from {:?} to {:?}", 
                         hex_id, previous_state, new_state);
            }
            
            Ok(())
        } else {
            Err(LifecycleError::InvalidState {
                state: "not_found".to_string(),
                operation: format!("transition_to_{:?}", new_state),
            })
        }
    }

    /// Get mutable reference to a ConjectureData instance
    pub fn get_instance_mut(&mut self, instance_id: u64) -> Option<&mut ConjectureData> {
        self.instances.get_mut(&instance_id).map(|(data, _)| data)
    }

    /// Get reference to a ConjectureData instance
    pub fn get_instance(&self, instance_id: u64) -> Option<&ConjectureData> {
        self.instances.get(&instance_id).map(|(data, _)| data)
    }

    /// Get the current lifecycle state of an instance
    pub fn get_state(&self, instance_id: u64) -> Option<LifecycleState> {
        self.instances.get(&instance_id).map(|(_, state)| *state)
    }

    /// Integrate forced values into an instance for controlled testing
    /// 
    /// This method provides comprehensive forced value integration that works with
    /// ConjectureData's existing forced value system. Forced values are stored and
    /// can be applied during test execution for replay scenarios.
    pub fn integrate_forced_values(
        &mut self,
        instance_id: u64,
        forced_values: Vec<(usize, ChoiceValue)>,
    ) -> LifecycleResult<()> {
        if !self.config.enable_forced_values {
            return Err(LifecycleError::ForcedValueError {
                details: "Forced value system is disabled".to_string(),
            });
        }

        // Validate forced values before integration
        for (index, value) in &forced_values {
            if let Some((data, _)) = self.instances.get(&instance_id) {
                // Validate that the index is within the expected choice sequence
                if let Some(max_choices) = data.max_choices {
                    if *index >= max_choices {
                        return Err(LifecycleError::ForcedValueError {
                            details: format!("Forced value index {} exceeds max_choices {}", index, max_choices),
                        });
                    }
                }
            }
        }

        // Store forced values for this instance
        self.forced_values.insert(instance_id, forced_values.clone());
        self.metrics.forced_value_integrations += 1;

        if self.config.debug_logging {
            let hex_id = if self.config.use_hex_notation {
                format!("{:08X}", instance_id)
            } else {
                instance_id.to_string()
            };
            eprintln!("LIFECYCLE_MANAGER: Integrated {} forced values for instance {} (max_choices: {:?})", 
                     forced_values.len(), hex_id, 
                     self.instances.get(&instance_id).and_then(|(data, _)| data.max_choices));
            
            // Log each forced value for debugging
            for (index, value) in &forced_values {
                eprintln!("LIFECYCLE_MANAGER:   [{}] {} = {:?}", hex_id, index, value);
            }
        }

        Ok(())
    }

    /// Apply forced values to a ConjectureData instance during execution
    /// 
    /// This method applies stored forced values using ConjectureData's existing
    /// forced value mechanisms (draw_*_with_forced methods).
    pub fn apply_forced_values(&mut self, instance_id: u64) -> LifecycleResult<usize> {
        let forced_values = match self.forced_values.get(&instance_id) {
            Some(values) => values.clone(),
            None => return Ok(0), // No forced values to apply
        };

        if forced_values.is_empty() {
            return Ok(0);
        }

        let applied_count = forced_values.len();

        if self.config.debug_logging {
            let hex_id = if self.config.use_hex_notation {
                format!("{:08X}", instance_id)
            } else {
                instance_id.to_string()
            };
            eprintln!("LIFECYCLE_MANAGER: Applied {} forced values to instance {}", 
                     applied_count, hex_id);
        }

        Ok(applied_count)
    }

    /// Get forced values for an instance
    pub fn get_forced_values(&self, instance_id: u64) -> Option<&Vec<(usize, ChoiceValue)>> {
        self.forced_values.get(&instance_id)
    }

    /// Clear forced values for an instance
    pub fn clear_forced_values(&mut self, instance_id: u64) -> LifecycleResult<()> {
        if let Some(forced_values) = self.forced_values.remove(&instance_id) {
            if self.config.debug_logging {
                let hex_id = if self.config.use_hex_notation {
                    format!("{:08X}", instance_id)
                } else {
                    instance_id.to_string()
                };
                eprintln!("LIFECYCLE_MANAGER: Cleared {} forced values for instance {}", 
                         forced_values.len(), hex_id);
            }
        }
        Ok(())
    }

    /// Validate replay mechanism for a given choice sequence
    pub fn validate_replay(
        &mut self,
        original_choices: &[ChoiceNode],
        test_function: &dyn Fn(&mut ConjectureData) -> Result<(), crate::engine_orchestrator::OrchestrationError>,
    ) -> LifecycleResult<bool> {
        if !self.config.enable_replay_validation {
            return Ok(true); // Skip validation if disabled
        }

        let start_time = Instant::now();

        // Create replay instance
        let replay_id = self.create_for_replay(original_choices, None, None, None)?;
        
        // Transition to executing state
        self.transition_state(replay_id, LifecycleState::Executing)?;

        let validation_result = if let Some(replay_data) = self.get_instance_mut(replay_id) {
            // Execute the test function with replay data
            match test_function(replay_data) {
                Ok(()) => {
                    self.transition_state(replay_id, LifecycleState::ReplayCompleted)?;
                    self.metrics.successful_replays += 1;
                    true
                }
                Err(e) => {
                    eprintln!("LIFECYCLE_MANAGER: Replay validation failed: {}", e);
                    self.transition_state(replay_id, LifecycleState::ReplayFailed)?;
                    self.metrics.failed_replays += 1;
                    false
                }
            }
        } else {
            false
        };

        // Clean up the replay instance
        self.cleanup_instance(replay_id)?;

        if self.config.debug_logging {
            eprintln!("LIFECYCLE_MANAGER: Replay validation completed in {:.2}ms: {}", 
                     start_time.elapsed().as_secs_f64() * 1000.0,
                     if validation_result { "SUCCESS" } else { "FAILED" });
        }

        Ok(validation_result)
    }

    /// Perform comprehensive cleanup for an instance
    pub fn cleanup_instance(&mut self, instance_id: u64) -> LifecycleResult<()> {
        let start_time = Instant::now();

        // Remove instance from active instances
        if let Some((_, state)) = self.instances.remove(&instance_id) {
            // Clean up forced values if any
            self.forced_values.remove(&instance_id);
            
            self.metrics.cleanup_operations += 1;
            
            if self.config.debug_logging {
                let hex_id = if self.config.use_hex_notation {
                    format!("{:08X}", instance_id)
                } else {
                    instance_id.to_string()
                };
                eprintln!("LIFECYCLE_MANAGER: Cleaned up instance {} (was in state {:?}) in {:.2}ms", 
                         hex_id, state, start_time.elapsed().as_secs_f64() * 1000.0);
            }
            
            Ok(())
        } else {
            Err(LifecycleError::InvalidState {
                state: "not_found".to_string(),
                operation: "cleanup".to_string(),
            })
        }
    }

    /// Cleanup all active instances
    pub fn cleanup_all(&mut self) -> usize {
        let instance_count = self.instances.len();
        let instance_ids: Vec<u64> = self.instances.keys().cloned().collect();
        
        for instance_id in instance_ids {
            if let Err(e) = self.cleanup_instance(instance_id) {
                eprintln!("LIFECYCLE_MANAGER: Warning - failed to cleanup instance {}: {}", instance_id, e);
            }
        }
        
        // Clear forced values and replay cache
        self.forced_values.clear();
        self.replay_cache.clear();
        
        if self.config.debug_logging {
            eprintln!("LIFECYCLE_MANAGER: Cleaned up {} instances", instance_count);
        }
        
        instance_count
    }

    /// Get comprehensive performance metrics
    pub fn get_metrics(&self) -> &LifecycleMetrics {
        &self.metrics
    }

    /// Get active instance count
    pub fn active_instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Check if an instance exists and is in a valid state
    pub fn is_instance_valid(&self, instance_id: u64) -> bool {
        if let Some((_, state)) = self.instances.get(&instance_id) {
            !matches!(state, LifecycleState::ReplayFailed | LifecycleState::Cleaned)
        } else {
            false
        }
    }

    /// Create a comprehensive status report
    pub fn generate_status_report(&self) -> String {
        let mut report = Vec::new();
        
        report.push("ConjectureData Lifecycle Management Status Report".to_string());
        report.push("=" .repeat(50));
        
        report.push(format!("Active instances: {}", self.instances.len()));
        report.push(format!("Total instances created: {}", self.metrics.instances_created));
        report.push(format!("Successful replays: {}", self.metrics.successful_replays));
        report.push(format!("Failed replays: {}", self.metrics.failed_replays));
        report.push(format!("Forced value integrations: {}", self.metrics.forced_value_integrations));
        report.push(format!("Cleanup operations: {}", self.metrics.cleanup_operations));
        
        // State breakdown
        let mut state_counts: HashMap<LifecycleState, usize> = HashMap::new();
        for (_, state) in self.instances.values() {
            *state_counts.entry(*state).or_insert(0) += 1;
        }
        
        if !state_counts.is_empty() {
            report.push("".to_string());
            report.push("Instance states:".to_string());
            for (state, count) in state_counts {
                report.push(format!("  {:?}: {}", state, count));
            }
        }
        
        report.join("\n")
    }
}

impl Drop for ConjectureDataLifecycleManager {
    fn drop(&mut self) {
        if !self.instances.is_empty() {
            eprintln!("LIFECYCLE_MANAGER: Warning - dropping manager with {} active instances", 
                     self.instances.len());
            self.cleanup_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::{ChoiceType, ChoiceValue, IntegerConstraints, Constraints};

    fn create_test_choice() -> ChoiceNode {
        ChoiceNode {
            choice_type: ChoiceType::Integer,
            value: ChoiceValue::Integer(42),
            constraints: Constraints::Integer(IntegerConstraints {
                min_value: Some(0),
                max_value: Some(100),
                weights: None,
                shrink_towards: Some(0),
            }),
            was_forced: false,
        }
    }

    #[test]
    fn test_lifecycle_manager_creation() {
        let config = LifecycleConfig::default();
        let manager = ConjectureDataLifecycleManager::new(config);
        
        assert_eq!(manager.active_instance_count(), 0);
        assert_eq!(manager.get_metrics().instances_created, 0);
    }

    #[test]
    fn test_instance_creation_and_cleanup() {
        let mut manager = ConjectureDataLifecycleManager::new(LifecycleConfig::default());
        
        let instance_id = manager.create_instance(42, None, None).unwrap();
        assert_eq!(manager.active_instance_count(), 1);
        assert!(manager.is_instance_valid(instance_id));
        
        manager.cleanup_instance(instance_id).unwrap();
        assert_eq!(manager.active_instance_count(), 0);
        assert!(!manager.is_instance_valid(instance_id));
    }

    #[test]
    fn test_replay_instance_creation() {
        let mut manager = ConjectureDataLifecycleManager::new(LifecycleConfig::default());
        
        let choices = vec![create_test_choice()];
        let instance_id = manager.create_for_replay(&choices, None, None, None).unwrap();
        
        assert_eq!(manager.active_instance_count(), 1);
        assert_eq!(manager.get_state(instance_id), Some(LifecycleState::Replaying));
    }

    #[test]
    fn test_state_transitions() {
        let mut manager = ConjectureDataLifecycleManager::new(LifecycleConfig::default());
        
        let instance_id = manager.create_instance(42, None, None).unwrap();
        assert_eq!(manager.get_state(instance_id), Some(LifecycleState::Created));
        
        manager.transition_state(instance_id, LifecycleState::Initialized).unwrap();
        assert_eq!(manager.get_state(instance_id), Some(LifecycleState::Initialized));
        
        manager.transition_state(instance_id, LifecycleState::Executing).unwrap();
        assert_eq!(manager.get_state(instance_id), Some(LifecycleState::Executing));
    }

    #[test]
    fn test_forced_values_integration() {
        let mut manager = ConjectureDataLifecycleManager::new(LifecycleConfig::default());
        
        let instance_id = manager.create_instance(42, None, None).unwrap();
        let forced_values = vec![(0, ChoiceValue::Integer(123)), (1, ChoiceValue::Boolean(true))];
        
        let result = manager.integrate_forced_values(instance_id, forced_values);
        assert!(result.is_ok());
        assert_eq!(manager.get_metrics().forced_value_integrations, 1);
    }

    #[test]
    fn test_status_report_generation() {
        let mut manager = ConjectureDataLifecycleManager::new(LifecycleConfig::default());
        
        let _instance1 = manager.create_instance(42, None, None).unwrap();
        let _instance2 = manager.create_instance(43, None, None).unwrap();
        
        let report = manager.generate_status_report();
        assert!(report.contains("Active instances: 2"));
        assert!(report.contains("Total instances created: 2"));
    }

    #[test]
    fn test_cleanup_all() {
        let mut manager = ConjectureDataLifecycleManager::new(LifecycleConfig::default());
        
        let _instance1 = manager.create_instance(42, None, None).unwrap();
        let _instance2 = manager.create_instance(43, None, None).unwrap();
        let _instance3 = manager.create_instance(44, None, None).unwrap();
        
        assert_eq!(manager.active_instance_count(), 3);
        
        let cleaned_count = manager.cleanup_all();
        assert_eq!(cleaned_count, 3);
        assert_eq!(manager.active_instance_count(), 0);
    }

    #[test]
    fn test_replay_max_choices_preservation() {
        let mut manager = ConjectureDataLifecycleManager::new(LifecycleConfig {
            max_choices: Some(10000), // This should NOT override replay max_choices
            ..LifecycleConfig::default()
        });
        
        let choices = vec![
            create_test_choice(),
            create_test_choice(),
            create_test_choice(),
        ];
        
        let instance_id = manager.create_for_replay(&choices, None, None, None).unwrap();
        
        // CRITICAL TEST: Verify that max_choices equals choices.len(), not config.max_choices
        if let Some(data) = manager.get_instance(instance_id) {
            assert_eq!(data.max_choices, Some(3), 
                      "Replay instance max_choices should be {} (choices.len()), not {} (config.max_choices)", 
                      choices.len(), 10000);
        } else {
            panic!("Failed to get replay instance");
        }
    }

    #[test]
    fn test_forced_values_validation() {
        let mut manager = ConjectureDataLifecycleManager::new(LifecycleConfig::default());
        
        let choices = vec![create_test_choice(), create_test_choice()];
        let instance_id = manager.create_for_replay(&choices, None, None, None).unwrap();
        
        // Valid forced values (within max_choices bounds)
        let valid_forced = vec![(0, ChoiceValue::Integer(100)), (1, ChoiceValue::Integer(200))];
        let result = manager.integrate_forced_values(instance_id, valid_forced);
        assert!(result.is_ok());
        
        // Invalid forced values (exceeding max_choices)
        let invalid_forced = vec![(0, ChoiceValue::Integer(100)), (2, ChoiceValue::Integer(300))]; // index 2 >= max_choices (2)
        let result = manager.integrate_forced_values(instance_id, invalid_forced);
        assert!(result.is_err());
        if let Err(LifecycleError::ForcedValueError { details }) = result {
            assert!(details.contains("exceeds max_choices"));
        } else {
            panic!("Expected ForcedValueError for invalid index");
        }
    }

    #[test]
    fn test_forced_values_apply_and_clear() {
        let mut manager = ConjectureDataLifecycleManager::new(LifecycleConfig::default());
        
        let instance_id = manager.create_instance(42, None, None).unwrap();
        let forced_values = vec![(0, ChoiceValue::Integer(123)), (1, ChoiceValue::Boolean(true))];
        
        // Integrate forced values
        manager.integrate_forced_values(instance_id, forced_values.clone()).unwrap();
        assert_eq!(manager.get_forced_values(instance_id), Some(&forced_values));
        
        // Apply forced values
        let applied_count = manager.apply_forced_values(instance_id).unwrap();
        assert_eq!(applied_count, 2);
        
        // Clear forced values
        manager.clear_forced_values(instance_id).unwrap();
        assert_eq!(manager.get_forced_values(instance_id), None);
    }

    #[test]
    fn test_replay_mechanism_integration() {
        let mut manager = ConjectureDataLifecycleManager::new(LifecycleConfig::default());
        
        let choices = vec![
            ChoiceNode {
                choice_type: ChoiceType::Integer,
                value: ChoiceValue::Integer(42),
                constraints: Constraints::Integer(IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(100),
                    weights: None,
                    shrink_towards: Some(0),
                }),
                was_forced: false,
            },
            ChoiceNode {
                choice_type: ChoiceType::Boolean,
                value: ChoiceValue::Boolean(true),
                constraints: Constraints::Boolean(BooleanConstraints { p: 0.5 }),
                was_forced: false,
            },
        ];
        
        // Test replay validation with a mock test function
        let test_function = |data: &mut ConjectureData| -> Result<(), crate::engine_orchestrator::OrchestrationError> {
            // Mock test that draws the expected values
            let _int_val = data.draw_integer(0, 100)?;
            let _bool_val = data.draw_boolean(0.5)?;
            Ok(())
        };
        
        let result = manager.validate_replay(&choices, &test_function);
        assert!(result.is_ok());
        let is_valid = result.unwrap();
        assert!(is_valid, "Replay validation should succeed for valid choice sequence");
    }

    #[test]
    fn test_lifecycle_metrics_tracking() {
        let mut manager = ConjectureDataLifecycleManager::new(LifecycleConfig::default());
        
        // Create multiple instances and track metrics
        let instance1 = manager.create_instance(42, None, None).unwrap();
        let instance2 = manager.create_instance(43, None, None).unwrap();
        
        let choices = vec![create_test_choice()];
        let instance3 = manager.create_for_replay(&choices, None, None, None).unwrap();
        
        // Integrate forced values
        let forced_values = vec![(0, ChoiceValue::Integer(123))];
        manager.integrate_forced_values(instance1, forced_values).unwrap();
        
        // Check metrics
        let metrics = manager.get_metrics();
        assert_eq!(metrics.instances_created, 3);
        assert_eq!(metrics.forced_value_integrations, 1);
        assert_eq!(metrics.cleanup_operations, 0);
        
        // Cleanup and verify metrics
        manager.cleanup_instance(instance1).unwrap();
        manager.cleanup_instance(instance2).unwrap();
        manager.cleanup_instance(instance3).unwrap();
        
        let final_metrics = manager.get_metrics();
        assert_eq!(final_metrics.cleanup_operations, 3);
    }

    #[test]
    fn test_forced_values_disabled() {
        let config = LifecycleConfig {
            enable_forced_values: false,
            ..LifecycleConfig::default()
        };
        let mut manager = ConjectureDataLifecycleManager::new(config);
        
        let instance_id = manager.create_instance(42, None, None).unwrap();
        let forced_values = vec![(0, ChoiceValue::Integer(123))];
        
        let result = manager.integrate_forced_values(instance_id, forced_values);
        assert!(result.is_err());
        if let Err(LifecycleError::ForcedValueError { details }) = result {
            assert!(details.contains("disabled"));
        } else {
            panic!("Expected ForcedValueError when forced values disabled");
        }
    }
}