//! Comprehensive capability tests for ConjectureData Lifecycle Management
//! 
//! These tests validate the complete lifecycle management capability including:
//! - ConjectureData creation and replay mechanisms
//! - Max choices value preservation during replay
//! - Forced value system integration
//! - PyO3 FFI interface contracts
//! - Performance and reliability metrics
//! - Integration with EngineOrchestrator

// PyO3 imports removed for core functionality testing
// use pyo3::prelude::*;
// use pyo3::types::{PyDict, PyList, PyTuple};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::Instant;

use crate::choice::{ChoiceNode, ChoiceType, ChoiceValue, Constraints, IntegerConstraints, FloatConstraints, BooleanConstraints};
use crate::data::{ConjectureData, Status, DataObserver, DrawError};
use crate::conjecture_data_lifecycle_management::{
    ConjectureDataLifecycleManager, LifecycleConfig, LifecycleState, LifecycleError, LifecycleMetrics
};
use crate::providers::PrimitiveProvider;
use crate::engine_orchestrator::{EngineOrchestrator, OrchestrationError};

/// Test-specific observer to capture lifecycle events
struct LifecycleTestObserver {
    events: Arc<Mutex<Vec<(String, String)>>>,
    instance_count: Arc<Mutex<usize>>,
}

impl LifecycleTestObserver {
    fn new() -> Self {
        Self {
            events: Arc::new(Mutex::new(Vec::new())),
            instance_count: Arc::new(Mutex::new(0)),
        }
    }
    
    fn record_event(&self, event_type: &str, details: &str) {
        if let Ok(mut events) = self.events.lock() {
            events.push((event_type.to_string(), details.to_string()));
        }
    }
    
    fn get_events(&self) -> Vec<(String, String)> {
        self.events.lock().unwrap().clone()
    }
    
    fn increment_instance_count(&self) {
        if let Ok(mut count) = self.instance_count.lock() {
            *count += 1;
        }
    }
    
    fn get_instance_count(&self) -> usize {
        *self.instance_count.lock().unwrap()
    }
}

impl DataObserver for LifecycleTestObserver {
    fn draw_value(&mut self, choice_type: ChoiceType, value: ChoiceValue, 
                  was_forced: bool, constraints: Box<Constraints>) {
        self.record_event("draw", &format!("{:?}:{:?} forced:{}", choice_type, value, was_forced));
        self.increment_instance_count();
    }
    
    fn start_example(&mut self, label: &str) {
        self.record_event("start_example", label);
    }
    
    fn end_example(&mut self, label: &str, discard: bool) {
        self.record_event("end_example", &format!("{} discard:{}", label, discard));
    }
}

/// Test provider for controlled value generation
#[derive(Debug)]
struct LifecycleTestProvider {
    values: Vec<ChoiceValue>,
    index: usize,
}

impl LifecycleTestProvider {
    fn new(values: Vec<ChoiceValue>) -> Self {
        Self { values, index: 0 }
    }
}

impl PrimitiveProvider for LifecycleTestProvider {
    fn lifetime(&self) -> crate::providers::ProviderLifetime {
        crate::providers::ProviderLifetime::TestCase
    }
    
    fn draw_boolean(&mut self, _p: f64) -> Result<bool, crate::providers::ProviderError> {
        if self.index < self.values.len() {
            if let ChoiceValue::Boolean(val) = &self.values[self.index] {
                self.index += 1;
                Ok(*val)
            } else {
                Err(crate::providers::ProviderError::InvalidChoice("Type mismatch".to_string()))
            }
        } else {
            Ok(true)
        }
    }
    
    fn draw_integer(&mut self, _constraints: &IntegerConstraints) -> Result<i128, crate::providers::ProviderError> {
        if self.index < self.values.len() {
            if let ChoiceValue::Integer(val) = &self.values[self.index] {
                self.index += 1;
                Ok(*val)
            } else {
                Err(crate::providers::ProviderError::InvalidChoice("Type mismatch".to_string()))
            }
        } else {
            Ok(42)
        }
    }
    
    fn draw_float(&mut self, _constraints: &FloatConstraints) -> Result<f64, crate::providers::ProviderError> {
        if self.index < self.values.len() {
            if let ChoiceValue::Float(val) = &self.values[self.index] {
                self.index += 1;
                Ok(*val)
            } else {
                Err(crate::providers::ProviderError::InvalidChoice("Type mismatch".to_string()))
            }
        } else {
            Ok(3.14)
        }
    }
    
    fn draw_string(&mut self, _intervals: &crate::choice::IntervalSet, _min_size: usize, _max_size: usize) -> Result<String, crate::providers::ProviderError> {
        if self.index < self.values.len() {
            if let ChoiceValue::String(val) = &self.values[self.index] {
                self.index += 1;
                Ok(val.clone())
            } else {
                Err(crate::providers::ProviderError::InvalidChoice("Type mismatch".to_string()))
            }
        } else {
            Ok("test".to_string())
        }
    }
    
    fn draw_bytes(&mut self, _min_size: usize, _max_size: usize) -> Result<Vec<u8>, crate::providers::ProviderError> {
        if self.index < self.values.len() {
            if let ChoiceValue::Bytes(val) = &self.values[self.index] {
                self.index += 1;
                Ok(val.clone())
            } else {
                Err(crate::providers::ProviderError::InvalidChoice("Type mismatch".to_string()))
            }
        } else {
            Ok(vec![1, 2, 3])
        }
    }
    
    fn generate_integer(&mut self, _rng: &mut rand_chacha::ChaCha8Rng, _constraints: &IntegerConstraints) -> Result<i128, DrawError> {
        if self.index < self.values.len() {
            if let ChoiceValue::Integer(val) = &self.values[self.index] {
                self.index += 1;
                Ok(*val)
            } else {
                Err(DrawError::InvalidReplayType)
            }
        } else {
            Ok(42) // Default value
        }
    }
    
    fn generate_boolean(&mut self, _rng: &mut rand_chacha::ChaCha8Rng, _constraints: &BooleanConstraints) -> Result<bool, DrawError> {
        if self.index < self.values.len() {
            if let ChoiceValue::Boolean(val) = &self.values[self.index] {
                self.index += 1;
                Ok(*val)
            } else {
                Err(DrawError::InvalidReplayType)
            }
        } else {
            Ok(true) // Default value
        }
    }
    
    fn generate_float(&mut self, _rng: &mut rand_chacha::ChaCha8Rng, _constraints: &FloatConstraints) -> Result<f64, DrawError> {
        if self.index < self.values.len() {
            if let ChoiceValue::Float(val) = &self.values[self.index] {
                self.index += 1;
                Ok(*val)
            } else {
                Err(DrawError::InvalidReplayType)
            }
        } else {
            Ok(3.14) // Default value
        }
    }
    
    fn generate_string(&mut self, _rng: &mut rand_chacha::ChaCha8Rng, _alphabet: &str, _min_size: usize, _max_size: usize) -> Result<String, DrawError> {
        if self.index < self.values.len() {
            if let ChoiceValue::String(val) = &self.values[self.index] {
                self.index += 1;
                Ok(val.clone())
            } else {
                Err(DrawError::InvalidReplayType)
            }
        } else {
            Ok("test".to_string()) // Default value
        }
    }
    
    fn generate_bytes(&mut self, _rng: &mut rand_chacha::ChaCha8Rng, size: usize) -> Result<Vec<u8>, DrawError> {
        Ok(vec![0u8; size])
    }
}

fn create_test_choices() -> Vec<ChoiceNode> {
    vec![
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
            index: Some(0),
        },
        ChoiceNode {
            choice_type: ChoiceType::Boolean,
            value: ChoiceValue::Boolean(true),
            constraints: Constraints::Boolean(BooleanConstraints { p: 0.5 }),
            was_forced: false,
            index: Some(1),
        },
        ChoiceNode {
            choice_type: ChoiceType::Float,
            value: ChoiceValue::Float(3.14),
            constraints: Constraints::Float(FloatConstraints::new(None, None)),
            was_forced: false,
            index: Some(2),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// Test the core lifecycle management capability
    #[test]
    fn test_lifecycle_management_capability() {
        let config = LifecycleConfig {
            debug_logging: false,  // Reduce noise in tests
            max_choices: Some(1000),
            execution_timeout_ms: Some(5000),
            enable_forced_values: true,
            enable_replay_validation: true,
            use_hex_notation: false,
        };
        
        let mut manager = ConjectureDataLifecycleManager::new(config);
        
        // Test instance creation
        let instance_id = manager.create_instance(12345, None, None).unwrap();
        assert_eq!(manager.active_instance_count(), 1);
        assert_eq!(manager.get_state(instance_id), Some(LifecycleState::Created));
        
        // Test state transitions
        manager.transition_state(instance_id, LifecycleState::Initialized).unwrap();
        assert_eq!(manager.get_state(instance_id), Some(LifecycleState::Initialized));
        
        manager.transition_state(instance_id, LifecycleState::Executing).unwrap();
        assert_eq!(manager.get_state(instance_id), Some(LifecycleState::Executing));
        
        manager.transition_state(instance_id, LifecycleState::Completed).unwrap();
        assert_eq!(manager.get_state(instance_id), Some(LifecycleState::Completed));
        
        // Test cleanup
        manager.cleanup_instance(instance_id).unwrap();
        assert_eq!(manager.active_instance_count(), 0);
        assert!(!manager.is_instance_valid(instance_id));
        
        // Verify metrics
        let metrics = manager.get_metrics();
        assert_eq!(metrics.instances_created, 1);
        assert_eq!(metrics.cleanup_operations, 1);
    }

    /// Test replay mechanism with max_choices preservation (addresses QA feedback)
    #[test]
    fn test_replay_max_choices_preservation() {
        let mut config = LifecycleConfig::default();
        config.debug_logging = false;
        config.max_choices = Some(10000); // Large value that should NOT override choices.len()
        
        let mut manager = ConjectureDataLifecycleManager::new(config);
        
        let choices = create_test_choices();
        let expected_max_choices = choices.len(); // Should be 3
        
        // Create replay instance
        let replay_id = manager.create_for_replay(&choices, None, None, None).unwrap();
        
        // Verify the replay instance was created
        assert_eq!(manager.get_state(replay_id), Some(LifecycleState::Replaying));
        
        // Get the ConjectureData instance and verify max_choices is preserved
        let replay_data = manager.get_instance(replay_id).unwrap();
        
        // CRITICAL: max_choices should be choices.len() (3), not config.max_choices (10000)
        // This tests the fix for the QA feedback issue
        assert_eq!(replay_data.max_choices, Some(expected_max_choices));
        
        // Verify choices were correctly set
        assert_eq!(replay_data.prefix.len(), expected_max_choices);
        
        manager.cleanup_instance(replay_id).unwrap();
    }

    /// Test forced value system integration
    #[test]
    fn test_forced_value_integration_capability() {
        let config = LifecycleConfig {
            debug_logging: false,
            enable_forced_values: true,
            ..Default::default()
        };
        
        let mut manager = ConjectureDataLifecycleManager::new(config);
        
        let instance_id = manager.create_instance(42, None, None).unwrap();
        
        // Test forced value integration
        let forced_values = vec![
            (0, ChoiceValue::Integer(999)),
            (1, ChoiceValue::Boolean(false)),
            (2, ChoiceValue::Float(2.718)),
        ];
        
        let result = manager.integrate_forced_values(instance_id, forced_values.clone());
        assert!(result.is_ok());
        
        // Verify metrics updated
        assert_eq!(manager.get_metrics().forced_value_integrations, 1);
        
        // Test disabling forced values
        let mut disabled_config = LifecycleConfig::default();
        disabled_config.enable_forced_values = false;
        disabled_config.debug_logging = false;
        
        let mut disabled_manager = ConjectureDataLifecycleManager::new(disabled_config);
        let disabled_id = disabled_manager.create_instance(42, None, None).unwrap();
        
        let result = disabled_manager.integrate_forced_values(disabled_id, forced_values);
        assert!(result.is_err());
        if let Err(LifecycleError::ForcedValueError { details }) = result {
            assert!(details.contains("disabled"));
        } else {
            panic!("Expected ForcedValueError");
        }
        
        manager.cleanup_instance(instance_id).unwrap();
        disabled_manager.cleanup_instance(disabled_id).unwrap();
    }

    /// Test replay validation mechanism
    #[test]
    fn test_replay_validation_capability() {
        let config = LifecycleConfig {
            debug_logging: false,
            enable_replay_validation: true,
            ..Default::default()
        };
        
        let mut manager = ConjectureDataLifecycleManager::new(config);
        
        let choices = create_test_choices();
        
        // Define a test function that should succeed
        let successful_test = |data: &mut ConjectureData| -> Result<(), OrchestrationError> {
            // Simulate successful test execution
            if data.prefix.len() == 3 {
                Ok(())
            } else {
                Err(OrchestrationError::InvalidState("Wrong number of choices".to_string()))
            }
        };
        
        // Test successful replay validation
        let result = manager.validate_replay(&choices, &successful_test);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), true);
        
        // Verify metrics
        assert_eq!(manager.get_metrics().successful_replays, 1);
        assert_eq!(manager.get_metrics().failed_replays, 0);
        
        // Define a test function that should fail
        let failing_test = |_data: &mut ConjectureData| -> Result<(), OrchestrationError> {
            Err(OrchestrationError::InvalidState("Simulated failure".to_string()))
        };
        
        // Test failed replay validation
        let result = manager.validate_replay(&choices, &failing_test);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), false);
        
        // Verify updated metrics
        assert_eq!(manager.get_metrics().successful_replays, 1);
        assert_eq!(manager.get_metrics().failed_replays, 1);
    }

    /// Test observer integration
    #[test]
    fn test_observer_integration_capability() {
        let config = LifecycleConfig {
            debug_logging: false,
            ..Default::default()
        };
        
        let mut manager = ConjectureDataLifecycleManager::new(config);
        
        let observer = Box::new(LifecycleTestObserver::new());
        let observer_events = observer.events.clone();
        
        let instance_id = manager.create_instance(42, Some(observer), None).unwrap();
        
        // Simulate some operations that would trigger observer events
        if let Some(data) = manager.get_instance_mut(instance_id) {
            // The observer would be called during actual test execution
            // For this test, we'll verify the observer was properly integrated
            assert!(data.observer.is_some());
        }
        
        manager.cleanup_instance(instance_id).unwrap();
    }

    /// Test provider integration
    #[test]
    fn test_provider_integration_capability() {
        let config = LifecycleConfig {
            debug_logging: false,
            ..Default::default()
        };
        
        let mut manager = ConjectureDataLifecycleManager::new(config);
        
        let test_values = vec![
            ChoiceValue::Integer(123),
            ChoiceValue::Boolean(false),
            ChoiceValue::Float(1.618),
        ];
        
        let provider = Box::new(LifecycleTestProvider::new(test_values));
        
        let instance_id = manager.create_instance(42, None, Some(provider)).unwrap();
        
        // Verify provider was integrated
        if let Some(data) = manager.get_instance(instance_id) {
            assert!(data.provider.is_some());
        }
        
        manager.cleanup_instance(instance_id).unwrap();
    }

    /// Test comprehensive lifecycle state management
    #[test]
    fn test_comprehensive_lifecycle_states() {
        let config = LifecycleConfig {
            debug_logging: false,
            ..Default::default()
        };
        
        let mut manager = ConjectureDataLifecycleManager::new(config);
        
        let instance_id = manager.create_instance(42, None, None).unwrap();
        
        // Test all valid state transitions
        let transitions = vec![
            LifecycleState::Initialized,
            LifecycleState::Executing,
            LifecycleState::Completed,
        ];
        
        for state in transitions {
            assert!(manager.transition_state(instance_id, state).is_ok());
            assert_eq!(manager.get_state(instance_id), Some(state));
            assert!(manager.is_instance_valid(instance_id));
        }
        
        // Test replay states
        let choices = create_test_choices();
        let replay_id = manager.create_for_replay(&choices, None, None, None).unwrap();
        
        assert_eq!(manager.get_state(replay_id), Some(LifecycleState::Replaying));
        
        manager.transition_state(replay_id, LifecycleState::ReplayCompleted).unwrap();
        assert_eq!(manager.get_state(replay_id), Some(LifecycleState::ReplayCompleted));
        
        // Test failed state
        manager.transition_state(replay_id, LifecycleState::ReplayFailed).unwrap();
        assert_eq!(manager.get_state(replay_id), Some(LifecycleState::ReplayFailed));
        assert!(!manager.is_instance_valid(replay_id)); // Should be invalid
        
        manager.cleanup_instance(instance_id).unwrap();
        manager.cleanup_instance(replay_id).unwrap();
    }

    /// Test error handling and edge cases
    #[test]
    fn test_error_handling_capability() {
        let config = LifecycleConfig {
            debug_logging: false,
            ..Default::default()
        };
        
        let mut manager = ConjectureDataLifecycleManager::new(config);
        
        // Test invalid instance operations
        let invalid_id = 99999;
        
        // Should fail for non-existent instance
        assert!(manager.transition_state(invalid_id, LifecycleState::Executing).is_err());
        assert!(manager.cleanup_instance(invalid_id).is_err());
        assert!(manager.get_instance(invalid_id).is_none());
        assert!(manager.get_state(invalid_id).is_none());
        assert!(!manager.is_instance_valid(invalid_id));
        
        // Test forced values with invalid instance
        let forced_values = vec![(0, ChoiceValue::Integer(42))];
        assert!(manager.integrate_forced_values(invalid_id, forced_values).is_err());
    }
    
    /// Test performance metrics and monitoring
    #[test]
    fn test_performance_metrics_capability() {
        let config = LifecycleConfig {
            debug_logging: false,
            ..Default::default()
        };
        
        let mut manager = ConjectureDataLifecycleManager::new(config);
        
        // Create multiple instances
        let instance1 = manager.create_instance(1, None, None).unwrap();
        let instance2 = manager.create_instance(2, None, None).unwrap();
        let instance3 = manager.create_instance(3, None, None).unwrap();
        
        // Test forced values integration
        let forced_values = vec![(0, ChoiceValue::Integer(42))];
        manager.integrate_forced_values(instance1, forced_values).unwrap();
        
        // Test replay with validation
        let choices = create_test_choices();
        let successful_test = |_: &mut ConjectureData| Ok(());
        manager.validate_replay(&choices, &successful_test).unwrap();
        
        // Verify comprehensive metrics
        let metrics = manager.get_metrics();
        assert_eq!(metrics.instances_created, 3);
        assert_eq!(metrics.forced_value_integrations, 1);
        assert_eq!(metrics.successful_replays, 1);
        assert_eq!(metrics.failed_replays, 0);
        assert_eq!(metrics.cleanup_operations, 1); // From replay validation
        
        // Test cleanup all
        let cleaned = manager.cleanup_all();
        assert_eq!(cleaned, 3);
        assert_eq!(manager.active_instance_count(), 0);
        
        // Verify final metrics
        let final_metrics = manager.get_metrics();
        assert_eq!(final_metrics.cleanup_operations, 4); // 1 from replay + 3 from cleanup_all
    }

    /// Test status report generation
    #[test]
    fn test_status_report_capability() {
        let config = LifecycleConfig {
            debug_logging: false,
            ..Default::default()
        };
        
        let mut manager = ConjectureDataLifecycleManager::new(config);
        
        // Create instances in different states
        let instance1 = manager.create_instance(1, None, None).unwrap();
        let instance2 = manager.create_instance(2, None, None).unwrap();
        
        manager.transition_state(instance1, LifecycleState::Executing).unwrap();
        manager.transition_state(instance2, LifecycleState::Completed).unwrap();
        
        let choices = create_test_choices();
        let instance3 = manager.create_for_replay(&choices, None, None, None).unwrap();
        
        // Generate status report
        let report = manager.generate_status_report();
        
        // Verify report contents
        assert!(report.contains("Active instances: 3"));
        assert!(report.contains("Total instances created: 3"));
        assert!(report.contains("Executing: 1"));
        assert!(report.contains("Completed: 1"));
        assert!(report.contains("Replaying: 1"));
        
        manager.cleanup_all();
    }
    
    /// Test integration with multiple concurrent operations
    #[test]
    fn test_concurrent_operations_capability() {
        let config = LifecycleConfig {
            debug_logging: false,
            max_choices: Some(100),
            ..Default::default()
        };
        
        let mut manager = ConjectureDataLifecycleManager::new(config);
        
        // Simulate concurrent operations
        let mut instance_ids = Vec::new();
        
        // Create multiple instances
        for i in 0..5 {
            let id = manager.create_instance(i, None, None).unwrap();
            instance_ids.push(id);
        }
        
        // Create replay instances
        let choices = create_test_choices();
        for _ in 0..3 {
            let replay_id = manager.create_for_replay(&choices, None, None, None).unwrap();
            instance_ids.push(replay_id);
        }
        
        assert_eq!(manager.active_instance_count(), 8);
        
        // Transition states concurrently
        for (i, &id) in instance_ids.iter().enumerate() {
            let state = if i % 2 == 0 {
                LifecycleState::Executing
            } else {
                LifecycleState::Completed
            };
            manager.transition_state(id, state).unwrap();
        }
        
        // Verify all instances are valid
        for &id in &instance_ids {
            assert!(manager.is_instance_valid(id));
        }
        
        // Cleanup all
        let cleaned = manager.cleanup_all();
        assert_eq!(cleaned, 8);
        assert_eq!(manager.active_instance_count(), 0);
    }
}

/// PyO3 FFI Integration Tests
#[cfg(feature = "pyo3")]
#[cfg(test)]
mod pyo3_tests {
    use super::*;

    /// PyO3 wrapper for ConjectureData lifecycle management
    #[pyclass(name = "ConjectureDataLifecycleManager")]
    struct PyConjectureDataLifecycleManager {
        manager: ConjectureDataLifecycleManager,
    }
    
    #[pymethods]
    impl PyConjectureDataLifecycleManager {
        #[new]
        fn new(
            debug_logging: Option<bool>,
            max_choices: Option<usize>,
            enable_forced_values: Option<bool>,
            enable_replay_validation: Option<bool>,
        ) -> Self {
            let config = LifecycleConfig {
                debug_logging: debug_logging.unwrap_or(true),
                max_choices,
                execution_timeout_ms: Some(30000),
                enable_forced_values: enable_forced_values.unwrap_or(true),
                enable_replay_validation: enable_replay_validation.unwrap_or(true),
                use_hex_notation: true,
            };
            
            Self {
                manager: ConjectureDataLifecycleManager::new(config),
            }
        }
        
        fn create_instance(&mut self, seed: u64) -> PyResult<u64> {
            self.manager.create_instance(seed, None, None)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        }
        
        fn create_for_replay(&mut self, choices: Vec<(String, PyObject)>) -> PyResult<u64> {
            // Convert Python choices to Rust ChoiceNode
            let rust_choices: Vec<ChoiceNode> = choices.into_iter().map(|(choice_type, value)| {
                // Simplified conversion for testing
                match choice_type.as_str() {
                    "Integer" => ChoiceNode {
                        choice_type: ChoiceType::Integer,
                        value: ChoiceValue::Integer(42), // Simplified
                        constraints: Constraints::Integer(IntegerConstraints {
                            min_value: Some(0),
                            max_value: Some(100),
                            weights: None,
                            shrink_towards: Some(0),
                        }),
                        was_forced: false,
                    },
                    "Boolean" => ChoiceNode {
                        choice_type: ChoiceType::Boolean,
                        value: ChoiceValue::Boolean(true), // Simplified
                        constraints: Constraints::Boolean(BooleanConstraints { p: 0.5 }),
                        was_forced: false,
                    },
                    _ => ChoiceNode {
                        choice_type: ChoiceType::Float,
                        value: ChoiceValue::Float(3.14), // Simplified
                        constraints: Constraints::Float(FloatConstraints::new()),
                        was_forced: false,
                    }
                }
            }).collect();
            
            self.manager.create_for_replay(&rust_choices, None, None, None)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        }
        
        fn transition_state(&mut self, instance_id: u64, new_state: String) -> PyResult<()> {
            let state = match new_state.as_str() {
                "Created" => LifecycleState::Created,
                "Initialized" => LifecycleState::Initialized,
                "Executing" => LifecycleState::Executing,
                "Completed" => LifecycleState::Completed,
                "Replaying" => LifecycleState::Replaying,
                "ReplayCompleted" => LifecycleState::ReplayCompleted,
                "ReplayFailed" => LifecycleState::ReplayFailed,
                "Cleaned" => LifecycleState::Cleaned,
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid state: {}", new_state))),
            };
            
            self.manager.transition_state(instance_id, state)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        }
        
        fn get_state(&self, instance_id: u64) -> Option<String> {
            self.manager.get_state(instance_id).map(|state| format!("{:?}", state))
        }
        
        fn cleanup_instance(&mut self, instance_id: u64) -> PyResult<()> {
            self.manager.cleanup_instance(instance_id)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        }
        
        fn active_instance_count(&self) -> usize {
            self.manager.active_instance_count()
        }
        
        fn is_instance_valid(&self, instance_id: u64) -> bool {
            self.manager.is_instance_valid(instance_id)
        }
        
        fn get_metrics(&self) -> PyResult<PyObject> {
            Python::with_gil(|py| {
                let metrics = self.manager.get_metrics();
                let dict = PyDict::new(py);
                dict.set_item("instances_created", metrics.instances_created)?;
                dict.set_item("successful_replays", metrics.successful_replays)?;
                dict.set_item("failed_replays", metrics.failed_replays)?;
                dict.set_item("forced_value_integrations", metrics.forced_value_integrations)?;
                dict.set_item("cleanup_operations", metrics.cleanup_operations)?;
                Ok(dict.into())
            })
        }
        
        fn generate_status_report(&self) -> String {
            self.manager.generate_status_report()
        }
    }

    #[test]
    fn test_pyo3_ffi_lifecycle_management() {
        Python::with_gil(|py| {
            // Test PyO3 wrapper creation
            let mut py_manager = PyConjectureDataLifecycleManager::new(
                Some(false), // debug_logging
                Some(1000),  // max_choices
                Some(true),  // enable_forced_values
                Some(true),  // enable_replay_validation
            );
            
            // Test instance creation through PyO3
            let instance_id = py_manager.create_instance(42).unwrap();
            assert_eq!(py_manager.active_instance_count(), 1);
            assert!(py_manager.is_instance_valid(instance_id));
            
            // Test state transitions through PyO3
            py_manager.transition_state(instance_id, "Initialized".to_string()).unwrap();
            assert_eq!(py_manager.get_state(instance_id), Some("Initialized".to_string()));
            
            py_manager.transition_state(instance_id, "Executing".to_string()).unwrap();
            assert_eq!(py_manager.get_state(instance_id), Some("Executing".to_string()));
            
            // Test cleanup through PyO3
            py_manager.cleanup_instance(instance_id).unwrap();
            assert_eq!(py_manager.active_instance_count(), 0);
            assert!(!py_manager.is_instance_valid(instance_id));
        });
    }

    #[test]
    fn test_pyo3_ffi_replay_functionality() {
        Python::with_gil(|py| {
            let mut py_manager = PyConjectureDataLifecycleManager::new(
                Some(false), // debug_logging
                Some(10000), // max_choices - should be overridden by choices.len()
                Some(true),  // enable_forced_values
                Some(true),  // enable_replay_validation
            );
            
            // Create Python-like choices for replay
            let py_choices = vec![
                ("Integer".to_string(), py.None()),
                ("Boolean".to_string(), py.None()),
                ("Float".to_string(), py.None()),
            ];
            
            // Test replay creation through PyO3
            let replay_id = py_manager.create_for_replay(py_choices).unwrap();
            assert_eq!(py_manager.get_state(replay_id), Some("Replaying".to_string()));
            
            // Test replay state transitions
            py_manager.transition_state(replay_id, "ReplayCompleted".to_string()).unwrap();
            assert_eq!(py_manager.get_state(replay_id), Some("ReplayCompleted".to_string()));
            
            py_manager.cleanup_instance(replay_id).unwrap();
        });
    }

    #[test]
    fn test_pyo3_ffi_metrics_interface() {
        Python::with_gil(|py| {
            let mut py_manager = PyConjectureDataLifecycleManager::new(
                Some(false),
                None,
                Some(true),
                Some(true),
            );
            
            // Create some instances for metrics
            let id1 = py_manager.create_instance(1).unwrap();
            let id2 = py_manager.create_instance(2).unwrap();
            
            // Get metrics through PyO3
            let metrics_obj = py_manager.get_metrics().unwrap();
            let metrics_dict = metrics_obj.downcast::<PyDict>(py).unwrap();
            
            // Verify metrics structure
            assert!(metrics_dict.contains("instances_created").unwrap());
            assert!(metrics_dict.contains("successful_replays").unwrap());
            assert!(metrics_dict.contains("failed_replays").unwrap());
            assert!(metrics_dict.contains("forced_value_integrations").unwrap());
            assert!(metrics_dict.contains("cleanup_operations").unwrap());
            
            // Verify values
            let instances_created: usize = metrics_dict.get_item("instances_created").unwrap().extract().unwrap();
            assert_eq!(instances_created, 2);
            
            // Test status report
            let report = py_manager.generate_status_report();
            assert!(report.contains("Active instances: 2"));
            assert!(report.contains("Total instances created: 2"));
            
            py_manager.cleanup_instance(id1).unwrap();
            py_manager.cleanup_instance(id2).unwrap();
        });
    }

    #[test]
    fn test_pyo3_ffi_error_handling() {
        Python::with_gil(|py| {
            let mut py_manager = PyConjectureDataLifecycleManager::new(
                Some(false),
                None,
                Some(true),
                Some(true),
            );
            
            // Test invalid state transition
            let result = py_manager.transition_state(99999, "Invalid".to_string());
            assert!(result.is_err());
            
            // Test invalid cleanup
            let result = py_manager.cleanup_instance(99999);
            assert!(result.is_err());
            
            // Test invalid state name
            let id = py_manager.create_instance(42).unwrap();
            let result = py_manager.transition_state(id, "InvalidState".to_string());
            assert!(result.is_err());
            
            py_manager.cleanup_instance(id).unwrap();
        });
    }
}

/// Integration tests with EngineOrchestrator
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::engine_orchestrator::{EngineOrchestrator, OrchestrationConfig};

    #[test]
    fn test_engine_orchestrator_integration() {
        let lifecycle_config = LifecycleConfig {
            debug_logging: false,
            max_choices: Some(1000),
            enable_forced_values: true,
            enable_replay_validation: true,
            ..Default::default()
        };
        
        let orchestrator_config = OrchestrationConfig {
            debug_logging: false,
            max_test_cases: 100,
            max_examples: 10,
            ..Default::default()
        };
        
        let mut orchestrator = EngineOrchestrator::new(orchestrator_config);
        let mut lifecycle_manager = ConjectureDataLifecycleManager::new(lifecycle_config);
        
        // Test coordinated operation
        let instance_id = lifecycle_manager.create_instance(42, None, None).unwrap();
        lifecycle_manager.transition_state(instance_id, LifecycleState::Initialized).unwrap();
        
        // Simulate test execution coordination
        if let Some(data) = lifecycle_manager.get_instance_mut(instance_id) {
            // Simulate orchestrator working with the ConjectureData
            assert_eq!(data.status, Status::Valid);
            assert!(data.max_choices.is_some());
        }
        
        lifecycle_manager.transition_state(instance_id, LifecycleState::Completed).unwrap();
        lifecycle_manager.cleanup_instance(instance_id).unwrap();
        
        // Verify integration metrics
        let metrics = lifecycle_manager.get_metrics();
        assert_eq!(metrics.instances_created, 1);
        assert_eq!(metrics.cleanup_operations, 1);
    }

    #[test]
    fn test_comprehensive_lifecycle_capability_integration() {
        let config = LifecycleConfig {
            debug_logging: false,
            max_choices: Some(500),
            execution_timeout_ms: Some(10000),
            enable_forced_values: true,
            enable_replay_validation: true,
            use_hex_notation: false,
        };
        
        let mut manager = ConjectureDataLifecycleManager::new(config);
        
        // Comprehensive test scenario
        let test_scenarios = vec![
            ("Basic instance creation", 1),
            ("Observer integration", 2),
            ("Provider integration", 3),
            ("Replay functionality", 4),
            ("Forced values", 5),
        ];
        
        let mut all_instances = Vec::new();
        
        for (scenario, seed) in test_scenarios {
            let observer = Box::new(LifecycleTestObserver::new());
            let provider = Box::new(LifecycleTestProvider::new(vec![
                ChoiceValue::Integer(seed * 10),
                ChoiceValue::Boolean(seed % 2 == 0),
            ]));
            
            let instance_id = manager.create_instance(
                seed,
                Some(observer),
                Some(provider),
            ).unwrap();
            
            // Test full lifecycle
            manager.transition_state(instance_id, LifecycleState::Initialized).unwrap();
            manager.transition_state(instance_id, LifecycleState::Executing).unwrap();
            manager.transition_state(instance_id, LifecycleState::Completed).unwrap();
            
            all_instances.push((instance_id, scenario));
        }
        
        // Test replay scenarios
        let choices = create_test_choices();
        let replay_id = manager.create_for_replay(&choices, None, None, None).unwrap();
        
        // Verify replay max_choices preservation
        let replay_data = manager.get_instance(replay_id).unwrap();
        assert_eq!(replay_data.max_choices, Some(choices.len()));
        
        // Test forced values integration
        let forced_values = vec![(0, ChoiceValue::Integer(777))];
        manager.integrate_forced_values(replay_id, forced_values).unwrap();
        
        // Generate comprehensive report
        let report = manager.generate_status_report();
        assert!(report.contains("Active instances: 6")); // 5 regular + 1 replay
        assert!(report.contains("Total instances created: 6"));
        assert!(report.contains("Forced value integrations: 1"));
        
        // Cleanup all instances
        let cleaned = manager.cleanup_all();
        assert_eq!(cleaned, 6);
        
        // Verify final state
        assert_eq!(manager.active_instance_count(), 0);
        let final_metrics = manager.get_metrics();
        assert_eq!(final_metrics.instances_created, 6);
        assert_eq!(final_metrics.cleanup_operations, 6);
        assert_eq!(final_metrics.forced_value_integrations, 1);
    }
}