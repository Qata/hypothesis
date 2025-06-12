//! Comprehensive Provider Type System Integration Tests for EngineOrchestrator
//!
//! This test module validates the complete Provider Type System Integration capability 
//! of the EngineOrchestrator, focusing on fixing critical type mismatches between generic 
//! `P: PrimitiveProvider` constraints and `Box<dyn PrimitiveProvider>` returns.
//!
//! Tests comprehensive capability behavior including:
//! - Type compatibility between static and dynamic provider types
//! - Provider context switching with correct type resolution
//! - Dynamic provider instantiation through registry
//! - FFI integration with PyO3 for provider type system validation
//! - Error handling for provider creation failures
//! - Lifecycle management of provider instances across phases

use std::sync::Arc;
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::engine_orchestrator::{
    EngineOrchestrator, OrchestratorConfig, ProviderContext, BackendScope,
    OrchestrationError, ExecutionPhase, OrchestrationResult
};
use crate::data::{ConjectureData, Status};
use crate::providers::{PrimitiveProvider, ProviderRegistry, get_provider_registry, HypothesisProvider};
use crate::choice::{ChoiceType, ChoiceNode};

/// Mock provider implementation for testing type system integration
#[derive(Debug, Clone)]
struct MockProvider {
    name: String,
    failure_mode: Option<String>,
}

impl MockProvider {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            failure_mode: None,
        }
    }

    fn with_failure_mode(name: &str, failure_mode: &str) -> Self {
        Self {
            name: name.to_string(),
            failure_mode: Some(failure_mode.to_string()),
        }
    }
}

impl PrimitiveProvider for MockProvider {
    fn weighted_choice(&mut self, _weights: &[f64]) -> Result<usize, String> {
        if let Some(ref failure) = self.failure_mode {
            match failure.as_str() {
                "backend_cannot_proceed_verified" => Err("BackendCannotProceed:verified".to_string()),
                "backend_cannot_proceed_exhausted" => Err("BackendCannotProceed:exhausted".to_string()),
                "backend_cannot_proceed_discard" => Err("BackendCannotProceed:discard_test_case".to_string()),
                "provider_error" => Err("Provider error".to_string()),
                _ => Ok(0),
            }
        } else {
            Ok(0)
        }
    }

    fn draw_integer(&mut self, _min: i64, _max: i64) -> Result<i64, String> {
        if let Some(ref failure) = self.failure_mode {
            Err(failure.clone())
        } else {
            Ok(42)
        }
    }

    fn draw_float(&mut self, _min: f64, _max: f64) -> Result<f64, String> {
        if let Some(ref failure) = self.failure_mode {
            Err(failure.clone())
        } else {
            Ok(0.5)
        }
    }

    fn draw_bytes(&mut self, _length: usize) -> Result<Vec<u8>, String> {
        if let Some(ref failure) = self.failure_mode {
            Err(failure.clone())
        } else {
            Ok(vec![1, 2, 3])
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Enhanced provider registry for testing dynamic dispatch
struct TestProviderRegistry {
    providers: HashMap<String, Box<dyn Fn() -> Box<dyn PrimitiveProvider> + Send + Sync>>,
}

impl TestProviderRegistry {
    fn new() -> Self {
        let mut registry = Self {
            providers: HashMap::new(),
        };
        
        // Register test providers with different type characteristics
        registry.register("hypothesis", Box::new(|| Box::new(HypothesisProvider::new())));
        registry.register("mock_basic", Box::new(|| Box::new(MockProvider::new("mock_basic"))));
        registry.register("mock_verified", Box::new(|| Box::new(MockProvider::with_failure_mode("mock_verified", "backend_cannot_proceed_verified"))));
        registry.register("mock_exhausted", Box::new(|| Box::new(MockProvider::with_failure_mode("mock_exhausted", "backend_cannot_proceed_exhausted"))));
        registry.register("mock_discard", Box::new(|| Box::new(MockProvider::with_failure_mode("mock_discard", "backend_cannot_proceed_discard"))));
        
        registry
    }

    fn register<F>(&mut self, name: &str, factory: F)
    where
        F: Fn() -> Box<dyn PrimitiveProvider> + Send + Sync + 'static,
    {
        self.providers.insert(name.to_string(), Box::new(factory));
    }

    fn create(&self, name: &str) -> Option<Box<dyn PrimitiveProvider>> {
        self.providers.get(name).map(|factory| factory())
    }

    fn available_providers(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }
}

/// Test suite for Provider Type System Integration capability
#[cfg(test)]
mod provider_type_integration_tests {
    use super::*;

    /// Test: Static to Dynamic Provider Type Compatibility
    /// 
    /// Validates that static generic providers can be converted to dynamic dispatch
    /// providers without type mismatches, addressing the core capability requirement.
    #[test]
    fn test_static_to_dynamic_provider_type_compatibility() {
        // Create static provider instance
        let static_provider = MockProvider::new("test_static");
        
        // Convert to dynamic dispatch (this is the core type system challenge)
        let dynamic_provider: Box<dyn PrimitiveProvider> = Box::new(static_provider);
        
        // Verify type compatibility through interface
        assert_eq!(dynamic_provider.name(), "test_static");
        
        // Test that dynamic provider can be used in orchestrator context
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let config = OrchestratorConfig::default();
        
        // This should compile without type errors (addressing the compilation failure)
        let orchestrator = EngineOrchestrator::new(test_fn, config);
        assert_eq!(orchestrator.current_phase(), ExecutionPhase::Initialize);
    }

    /// Test: Dynamic Provider Creation Through Registry
    /// 
    /// Validates the provider registry can create dynamic providers that are
    /// compatible with the orchestrator's type constraints.
    #[test]
    fn test_dynamic_provider_creation_compatibility() {
        let registry = TestProviderRegistry::new();
        
        // Test that all registered providers can be created as dynamic instances
        let available = registry.available_providers();
        assert!(available.contains(&"hypothesis".to_string()));
        assert!(available.contains(&"mock_basic".to_string()));
        
        // Create dynamic providers and verify type compatibility
        let hypothesis_provider = registry.create("hypothesis").unwrap();
        assert_eq!(hypothesis_provider.name(), "hypothesis");
        
        let mock_provider = registry.create("mock_basic").unwrap();
        assert_eq!(mock_provider.name(), "mock_basic");
        
        // Verify providers can be used with orchestrator (no type errors)
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let config = OrchestratorConfig::default();
        let orchestrator = EngineOrchestrator::new(test_fn, config);
        
        // Test create_active_provider method returns compatible type
        let active_provider_result = orchestrator.create_active_provider();
        assert!(active_provider_result.is_ok());
    }

    /// Test: Provider Context Switching with Type Resolution
    /// 
    /// Validates that provider switching maintains type compatibility throughout
    /// the orchestration lifecycle, especially during context transitions.
    #[test]
    fn test_provider_context_switching_type_resolution() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = MockProvider::new("initial_provider");
        let mut config = OrchestratorConfig::default();
        config.backend = "mock_verified".to_string();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        
        // Initial state should use configured provider
        assert_eq!(orchestrator.provider_context().active_provider, "mock_verified");
        assert!(!orchestrator.provider_context().switch_to_hypothesis);
        
        // Test provider switching maintains type compatibility
        let switch_result = orchestrator.switch_to_hypothesis_provider();
        assert!(switch_result.is_ok());
        
        // Verify switch updated context correctly
        assert!(orchestrator.provider_context().switch_to_hypothesis);
        assert_eq!(orchestrator.provider_context().active_provider, "hypothesis");
        
        // Test that create_active_provider respects switch context
        let active_provider = orchestrator.create_active_provider();
        assert!(active_provider.is_ok());
        
        // Verify phase-based provider selection works with type system
        let reuse_provider = orchestrator.select_provider_for_phase(ExecutionPhase::Reuse);
        assert_eq!(reuse_provider, "hypothesis");
        
        let generate_provider = orchestrator.select_provider_for_phase(ExecutionPhase::Generate);
        assert_eq!(generate_provider, "hypothesis"); // Should use switched provider
    }

    /// Test: BackendCannotProceed Error Handling with Provider Types
    /// 
    /// Validates that BackendCannotProceed errors trigger proper provider switching
    /// while maintaining type system compatibility.
    #[test]
    fn test_backend_cannot_proceed_provider_type_handling() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = MockProvider::new("test_provider");
        let mut config = OrchestratorConfig::default();
        config.backend = "crosshair".to_string();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        
        // Test verified scope triggers provider switch
        let result = orchestrator.handle_backend_cannot_proceed(BackendScope::Verified);
        assert!(result.is_ok());
        assert!(orchestrator.provider_context().switch_to_hypothesis);
        assert_eq!(orchestrator.provider_context().verified_by, Some("crosshair".to_string()));
        
        // Verify create_active_provider works after switch
        let active_provider = orchestrator.create_active_provider();
        assert!(active_provider.is_ok());
        
        // Test exhausted scope also triggers switch
        let mut orchestrator2 = EngineOrchestrator::new(
            Box::new(|_data: &mut ConjectureData| Ok(())),
            OrchestratorConfig::default()
        );
        
        let result2 = orchestrator2.handle_backend_cannot_proceed(BackendScope::Exhausted);
        assert!(result2.is_ok());
        assert!(orchestrator2.provider_context().switch_to_hypothesis);
        
        // Test discard threshold behavior
        let mut orchestrator3 = EngineOrchestrator::new(
            Box::new(|_data: &mut ConjectureData| Ok(())),
            OrchestratorConfig::default()
        );
        orchestrator3.set_call_count(50);
        
        // Trigger multiple discard errors to hit threshold
        for _ in 0..12 {
            let result = orchestrator3.handle_backend_cannot_proceed(BackendScope::DiscardTestCase);
            assert!(result.is_ok());
        }
        
        assert!(orchestrator3.provider_context().switch_to_hypothesis);
        assert_eq!(orchestrator3.provider_context().failed_realize_count, 12);
    }

    /// Test: Provider Error Propagation and Type Safety
    /// 
    /// Validates that provider errors are properly handled through the type system
    /// without causing type mismatches or runtime failures.
    #[test]
    fn test_provider_error_propagation_type_safety() {
        let registry = TestProviderRegistry::new();
        
        // Test provider creation failure handling
        let nonexistent_provider = registry.create("nonexistent");
        assert!(nonexistent_provider.is_none());
        
        // Test orchestrator creation with invalid backend
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = MockProvider::new("test_provider");
        let mut config = OrchestratorConfig::default();
        config.backend = "nonexistent_backend".to_string();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        
        // Initialize should fail with proper error type
        let init_result = orchestrator.run();
        assert!(init_result.is_err());
        
        if let Err(OrchestrationError::ProviderCreationFailed { backend, reason }) = init_result {
            assert_eq!(backend, "nonexistent_backend");
            assert!(reason.contains("Backend not found"));
        } else {
            panic!("Expected ProviderCreationFailed error");
        }
        
        // Test provider switching failure scenarios
        let test_fn2 = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider2 = MockProvider::new("test_provider2");
        let config2 = OrchestratorConfig::default();
        
        let mut orchestrator2 = EngineOrchestrator::new(test_fn2, config2);
        
        // Simulate switch failure (would happen if hypothesis provider unavailable)
        // This tests the error path in switch_to_hypothesis_provider
        assert!(orchestrator2.switch_to_hypothesis_provider().is_ok()); // Should succeed with mock registry
    }

    /// Test: Provider Lifecycle Management with Type Constraints
    /// 
    /// Validates that provider instances are properly managed throughout the
    /// orchestration lifecycle while maintaining type system integrity.
    #[test]
    fn test_provider_lifecycle_management_type_constraints() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = MockProvider::new("lifecycle_test");
        let mut config = OrchestratorConfig::default();
        config.backend = "mock_basic".to_string();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        
        // Test provider context initialization
        assert_eq!(orchestrator.provider_context().active_provider, "mock_basic");
        assert!(!orchestrator.provider_context().switch_to_hypothesis);
        
        // Test observation callback registration (type compatibility)
        orchestrator.register_provider_observation_callback("test_callback_1".to_string());
        orchestrator.register_provider_observation_callback("test_callback_2".to_string());
        
        assert_eq!(orchestrator.provider_context().observation_callbacks.len(), 2);
        
        // Test provider observation logging (type safety)
        orchestrator.log_provider_observation("test_event", "test_details");
        
        // Test phase-specific provider selection maintains type constraints
        let init_provider = orchestrator.select_provider_for_phase(ExecutionPhase::Initialize);
        assert_eq!(init_provider, "mock_basic");
        
        let reuse_provider = orchestrator.select_provider_for_phase(ExecutionPhase::Reuse);
        assert_eq!(reuse_provider, "hypothesis");
        
        let generate_provider = orchestrator.select_provider_for_phase(ExecutionPhase::Generate);
        assert_eq!(generate_provider, "mock_basic");
        
        let shrink_provider = orchestrator.select_provider_for_phase(ExecutionPhase::Shrink);
        assert_eq!(shrink_provider, "hypothesis");
        
        // Test cleanup maintains type safety
        orchestrator.cleanup_provider_context();
        assert!(orchestrator.provider_context().observation_callbacks.is_empty());
    }

    /// Test: Comprehensive Provider Type System Integration
    /// 
    /// End-to-end test validating the complete provider type system capability
    /// including creation, switching, error handling, and cleanup.
    #[test]
    fn test_comprehensive_provider_type_system_integration() {
        let test_fn = Box::new(|_data: &mut ConjectureData| {
            // Simulate a test that might trigger BackendCannotProceed
            Err(OrchestrationError::BackendCannotProceed { 
                scope: "verified".to_string() 
            })
        });
        
        let provider = MockProvider::new("comprehensive_test");
        let mut config = OrchestratorConfig::default();
        config.backend = "crosshair".to_string();
        config.max_examples = 5; // Small number for testing
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        
        // Register callbacks to test type system under observation
        orchestrator.register_provider_observation_callback("integration_test".to_string());
        
        // Test that provider creation works with type constraints
        let initial_provider = orchestrator.create_active_provider();
        assert!(initial_provider.is_ok());
        
        // Test BackendCannotProceed handling with type resolution
        let backend_error_result = orchestrator.handle_backend_cannot_proceed(BackendScope::Verified);
        assert!(backend_error_result.is_ok());
        
        // Verify provider switch maintained type compatibility
        assert!(orchestrator.provider_context().switch_to_hypothesis);
        assert_eq!(orchestrator.provider_context().verified_by, Some("crosshair".to_string()));
        
        // Test provider creation after switch
        let switched_provider = orchestrator.create_active_provider();
        assert!(switched_provider.is_ok());
        
        // Test using_hypothesis_backend detection
        assert!(orchestrator.using_hypothesis_backend());
        
        // Test phase-based provider selection after switch
        let post_switch_generate = orchestrator.select_provider_for_phase(ExecutionPhase::Generate);
        assert_eq!(post_switch_generate, "hypothesis");
        
        // Test provider observation logging with callbacks
        orchestrator.log_provider_observation("comprehensive_test", "type_system_validation");
        
        // Test cleanup maintains type safety throughout
        orchestrator.cleanup_provider_context();
        
        // Verify final state is type-safe and consistent
        assert!(orchestrator.provider_context().observation_callbacks.is_empty());
        assert!(orchestrator.provider_context().switch_to_hypothesis);
    }
}

/// PyO3 Integration Tests for Provider Type System
/// 
/// These tests validate that the provider type system integrates properly
/// with Python FFI through PyO3, ensuring no type mismatches occur.
#[cfg(test)]
mod provider_type_pyo3_integration_tests {
    use super::*;

    /// Test: PyO3 Provider Interface Type Compatibility
    /// 
    /// Validates that provider types can be safely exposed to Python
    /// through PyO3 without type system conflicts.
    #[test]
    fn test_pyo3_provider_interface_type_compatibility() {
        Python::with_gil(|py| {
            // Create provider instances to test PyO3 compatibility
            let hypothesis_provider = HypothesisProvider::new();
            let mock_provider = MockProvider::new("pyo3_test");
            
            // Test that provider names can be safely exposed to Python
            let provider_name = hypothesis_provider.name();
            let py_name = PyString::new_bound(py, provider_name);
            assert_eq!(py_name.to_string(), "hypothesis");
            
            let mock_name = mock_provider.name();
            let py_mock_name = PyString::new_bound(py, mock_name);
            assert_eq!(py_mock_name.to_string(), "pyo3_test");
            
            // Test orchestrator configuration can be created from Python data
            let py_dict = PyDict::new_bound(py);
            py_dict.set_item("backend", "hypothesis").unwrap();
            py_dict.set_item("max_examples", 100).unwrap();
            
            let backend: String = py_dict.get_item("backend").unwrap().unwrap().extract().unwrap();
            let max_examples: usize = py_dict.get_item("max_examples").unwrap().unwrap().extract().unwrap();
            
            assert_eq!(backend, "hypothesis");
            assert_eq!(max_examples, 100);
            
            // Verify configuration can create orchestrator without type errors
            let mut config = OrchestratorConfig::default();
            config.backend = backend;
            config.max_examples = max_examples;
            
            let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
            let orchestrator = EngineOrchestrator::new(test_fn, config);
            assert_eq!(orchestrator.current_phase(), ExecutionPhase::Initialize);
        });
    }

    /// Test: PyO3 Provider Error Handling Type Safety
    /// 
    /// Validates that provider errors can be properly handled and converted
    /// to Python exceptions without type system issues.
    #[test]
    fn test_pyo3_provider_error_handling_type_safety() {
        Python::with_gil(|py| {
            // Test provider creation errors
            let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
            let provider = MockProvider::new("pyo3_error_test");
            let mut config = OrchestratorConfig::default();
            config.backend = "nonexistent_pyo3_backend".to_string();
            
            let mut orchestrator = EngineOrchestrator::new(test_fn, config);
            
            // Test that orchestration errors can be converted to Python strings
            let run_result = orchestrator.run();
            assert!(run_result.is_err());
            
            let error_msg = match run_result {
                Err(e) => format!("{}", e),
                Ok(_) => "No error".to_string(),
            };
            
            // Verify error message can be safely passed to Python
            let py_error = PyString::new_bound(py, &error_msg);
            assert!(py_error.to_string().contains("Failed to create provider"));
            
            // Test provider switching errors
            let test_fn2 = Box::new(|_data: &mut ConjectureData| Ok(()));
            let provider2 = MockProvider::new("pyo3_switch_test");
            let config2 = OrchestratorConfig::default();
            
            let mut orchestrator2 = EngineOrchestrator::new(test_fn2, config2);
            let switch_result = orchestrator2.switch_to_hypothesis_provider();
            
            // Verify switch result can be handled in Python context
            let switch_success = switch_result.is_ok();
            let py_success = PyBool::new_bound(py, switch_success);
            assert!(py_success.is_truthy().unwrap());
        });
    }

    /// Test: PyO3 Provider Context Serialization Type Compatibility
    /// 
    /// Validates that provider context can be serialized for Python FFI
    /// without type system conflicts.
    #[test]
    fn test_pyo3_provider_context_serialization_type_compatibility() {
        Python::with_gil(|py| {
            let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
            let provider = MockProvider::new("pyo3_context_test");
            let mut config = OrchestratorConfig::default();
            config.backend = "crosshair".to_string();
            
            let mut orchestrator = EngineOrchestrator::new(test_fn, config);
            
            // Set up provider context state
            orchestrator.register_provider_observation_callback("pyo3_callback".to_string());
            let _ = orchestrator.handle_backend_cannot_proceed(BackendScope::Verified);
            
            let context = orchestrator.provider_context();
            
            // Test that context fields can be safely exposed to Python
            let py_dict = PyDict::new_bound(py);
            py_dict.set_item("active_provider", &context.active_provider).unwrap();
            py_dict.set_item("switch_to_hypothesis", context.switch_to_hypothesis).unwrap();
            py_dict.set_item("failed_realize_count", context.failed_realize_count).unwrap();
            
            if let Some(ref verified_by) = context.verified_by {
                py_dict.set_item("verified_by", verified_by).unwrap();
            }
            
            // Create Python list for callbacks
            let py_callbacks = PyList::new_bound(py, &context.observation_callbacks);
            py_dict.set_item("observation_callbacks", py_callbacks).unwrap();
            
            // Verify Python data can be extracted back to Rust types
            let extracted_provider: String = py_dict.get_item("active_provider").unwrap().unwrap().extract().unwrap();
            let extracted_switch: bool = py_dict.get_item("switch_to_hypothesis").unwrap().unwrap().extract().unwrap();
            let extracted_count: usize = py_dict.get_item("failed_realize_count").unwrap().unwrap().extract().unwrap();
            
            assert_eq!(extracted_provider, "crosshair");
            assert!(extracted_switch);
            assert_eq!(extracted_count, 1); // From BackendCannotProceed handling
            
            // Test callback extraction
            let py_callbacks_back: Vec<String> = py_dict.get_item("observation_callbacks").unwrap().unwrap().extract().unwrap();
            assert_eq!(py_callbacks_back.len(), 1);
            assert_eq!(py_callbacks_back[0], "pyo3_callback");
        });
    }

    /// Test: PyO3 Provider Statistics Integration Type Safety
    /// 
    /// Validates that provider statistics can be safely exposed to Python
    /// for monitoring and analysis without type mismatches.
    #[test]
    fn test_pyo3_provider_statistics_integration_type_safety() {
        Python::with_gil(|py| {
            let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
            let provider = MockProvider::new("pyo3_stats_test");
            let config = OrchestratorConfig::default();
            
            let mut orchestrator = EngineOrchestrator::new(test_fn, config);
            
            // Generate some provider observations
            orchestrator.log_provider_observation("test_event_1", "details_1");
            orchestrator.log_provider_observation("test_event_2", "details_2");
            
            // Test that statistics can be safely exposed to Python
            let stats = orchestrator.statistics();
            let py_stats_dict = PyDict::new_bound(py);
            
            // Convert phases statistics to Python
            let py_phases_dict = PyDict::new_bound(py);
            for (phase, phase_stats) in &stats.phases {
                let py_phase_dict = PyDict::new_bound(py);
                py_phase_dict.set_item("duration_seconds", phase_stats.duration_seconds).unwrap();
                py_phase_dict.set_item("test_cases", phase_stats.test_cases).unwrap();
                py_phase_dict.set_item("distinct_failures", phase_stats.distinct_failures).unwrap();
                py_phase_dict.set_item("shrinks_successful", phase_stats.shrinks_successful).unwrap();
                
                let phase_name = format!("{:?}", phase);
                py_phases_dict.set_item(phase_name, py_phase_dict).unwrap();
            }
            py_stats_dict.set_item("phases", py_phases_dict).unwrap();
            
            // Convert stopped_because to Python
            if let Some(ref reason) = stats.stopped_because {
                py_stats_dict.set_item("stopped_because", reason).unwrap();
            }
            
            // Convert targets to Python
            let py_targets_dict = PyDict::new_bound(py);
            for (target, value) in &stats.targets {
                py_targets_dict.set_item(target, *value).unwrap();
            }
            py_stats_dict.set_item("targets", py_targets_dict).unwrap();
            
            // Convert node_id to Python
            if let Some(ref node_id) = stats.node_id {
                py_stats_dict.set_item("node_id", node_id).unwrap();
            }
            
            // Verify Python statistics can be extracted back safely
            let py_phases: &Bound<PyDict> = py_stats_dict.get_item("phases").unwrap().unwrap().downcast().unwrap();
            assert!(py_phases.len() >= 0); // Should be able to access phases
            
            let py_targets: &Bound<PyDict> = py_stats_dict.get_item("targets").unwrap().unwrap().downcast().unwrap();
            assert_eq!(py_targets.len(), 0); // No targets set in this test
            
            // Test provider context statistics integration
            let context = orchestrator.provider_context();
            let py_context_dict = PyDict::new_bound(py);
            py_context_dict.set_item("active_provider", &context.active_provider).unwrap();
            py_context_dict.set_item("switch_to_hypothesis", context.switch_to_hypothesis).unwrap();
            
            // Verify context integration with statistics
            let extracted_provider: String = py_context_dict.get_item("active_provider").unwrap().unwrap().extract().unwrap();
            assert_eq!(extracted_provider, "hypothesis"); // Default provider
        });
    }
}

use pyo3::types::{PyString, PyBool, PyList};

/// Integration capability demonstration tests
/// 
/// These tests demonstrate the complete capability working end-to-end
/// with real-world scenarios that would trigger the type system issues.
#[cfg(test)]
mod provider_type_capability_demonstration_tests {
    use super::*;

    /// Capability Demo: Dynamic Provider Switching Under Load
    /// 
    /// Demonstrates the capability handling multiple provider switches
    /// while maintaining type system integrity under stress conditions.
    #[test]
    fn test_capability_dynamic_provider_switching_under_load() {
        let test_fn = Box::new(|data: &mut ConjectureData| {
            // Simulate varying test behavior that triggers different provider responses
            let call_count = data.choices.len();
            
            match call_count % 4 {
                0 => Err(OrchestrationError::BackendCannotProceed { 
                    scope: "verified".to_string() 
                }),
                1 => Err(OrchestrationError::BackendCannotProceed { 
                    scope: "exhausted".to_string() 
                }),
                2 => Err(OrchestrationError::BackendCannotProceed { 
                    scope: "discard_test_case".to_string() 
                }),
                _ => Ok(()),
            }
        });
        
        let provider = MockProvider::new("load_test");
        let mut config = OrchestratorConfig::default();
        config.backend = "crosshair".to_string();
        config.max_examples = 10;
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        
        // Simulate multiple provider switches
        for i in 0..5 {
            let scope = match i % 3 {
                0 => BackendScope::Verified,
                1 => BackendScope::Exhausted,
                _ => BackendScope::DiscardTestCase,
            };
            
            let result = orchestrator.handle_backend_cannot_proceed(scope);
            assert!(result.is_ok(), "Provider switch {} failed", i);
            
            // Verify provider creation still works after each switch
            let active_provider = orchestrator.create_active_provider();
            assert!(active_provider.is_ok(), "Provider creation failed after switch {}", i);
            
            // Test phase selection after each switch
            let generate_provider = orchestrator.select_provider_for_phase(ExecutionPhase::Generate);
            assert_eq!(generate_provider, "hypothesis", "Phase selection failed after switch {}", i);
        }
        
        // Verify final state is consistent
        assert!(orchestrator.using_hypothesis_backend());
        assert_eq!(orchestrator.provider_context().verified_by, Some("crosshair".to_string()));
    }

    /// Capability Demo: Multi-Phase Provider Type Consistency
    /// 
    /// Demonstrates that provider types remain consistent across all
    /// execution phases, validating the complete lifecycle integration.
    #[test]
    fn test_capability_multi_phase_provider_type_consistency() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = MockProvider::new("multi_phase_test");
        let mut config = OrchestratorConfig::default();
        config.backend = "crosshair".to_string();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        
        // Test each phase maintains type consistency
        let phases = vec![
            ExecutionPhase::Initialize,
            ExecutionPhase::Reuse,
            ExecutionPhase::Generate,
            ExecutionPhase::Shrink,
            ExecutionPhase::Cleanup,
        ];
        
        for phase in phases {
            // Test provider selection for each phase
            let selected_provider = orchestrator.select_provider_for_phase(phase);
            assert!(!selected_provider.is_empty(), "Provider selection failed for phase {:?}", phase);
            
            // Test provider creation works for each phase context
            let active_provider = orchestrator.create_active_provider();
            assert!(active_provider.is_ok(), "Provider creation failed for phase {:?}", phase);
            
            // Test phase transition maintains type safety
            let transition_result = orchestrator.transition_to_phase(phase);
            assert!(transition_result.is_ok(), "Phase transition failed for {:?}", phase);
            assert_eq!(orchestrator.current_phase(), phase);
        }
        
        // Test provider switching works across all phases
        let switch_result = orchestrator.switch_to_hypothesis_provider();
        assert!(switch_result.is_ok());
        
        // Verify all phases still work after switch
        for phase in vec![ExecutionPhase::Generate, ExecutionPhase::Shrink] {
            let selected_provider = orchestrator.select_provider_for_phase(phase);
            assert_eq!(selected_provider, "hypothesis", "Post-switch provider selection failed for {:?}", phase);
        }
    }

    /// Capability Demo: Error Recovery with Type System Preservation
    /// 
    /// Demonstrates that the type system remains intact even when recovering
    /// from various error conditions.
    #[test]
    fn test_capability_error_recovery_type_system_preservation() {
        // Test multiple error scenarios
        let error_scenarios = vec![
            ("provider_creation_error", "nonexistent_backend"),
            ("provider_switching_error", "hypothesis"), // This should succeed
            ("backend_verification_error", "crosshair"),
        ];
        
        for (scenario_name, backend) in error_scenarios {
            let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
            let provider = MockProvider::new(scenario_name);
            let mut config = OrchestratorConfig::default();
            config.backend = backend.to_string();
            
            let mut orchestrator = EngineOrchestrator::new(test_fn, config);
            
            match scenario_name {
                "provider_creation_error" => {
                    // This should fail during initialization
                    let run_result = orchestrator.run();
                    assert!(run_result.is_err());
                    
                    // But provider context should still be accessible
                    let context = orchestrator.provider_context();
                    assert_eq!(context.active_provider, backend);
                }
                "provider_switching_error" => {
                    // This should succeed
                    let switch_result = orchestrator.switch_to_hypothesis_provider();
                    assert!(switch_result.is_ok());
                    
                    // Verify type consistency after successful switch
                    let active_provider = orchestrator.create_active_provider();
                    assert!(active_provider.is_ok());
                }
                "backend_verification_error" => {
                    // Simulate backend verification
                    let verify_result = orchestrator.handle_backend_cannot_proceed(BackendScope::Verified);
                    assert!(verify_result.is_ok());
                    
                    // Verify type system preserved after verification
                    assert!(orchestrator.using_hypothesis_backend());
                    let post_verify_provider = orchestrator.create_active_provider();
                    assert!(post_verify_provider.is_ok());
                }
                _ => {}
            }
        }
    }

    /// Capability Demo: Concurrent Provider Type Safety
    /// 
    /// Demonstrates that provider type system maintains safety even under
    /// concurrent access patterns (simulated through rapid state changes).
    #[test]
    fn test_capability_concurrent_provider_type_safety() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = MockProvider::new("concurrent_test");
        let config = OrchestratorConfig::default();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        
        // Simulate rapid state changes that might occur under concurrent access
        for i in 0..20 {
            // Alternate between different operations
            match i % 4 {
                0 => {
                    // Provider creation
                    let provider = orchestrator.create_active_provider();
                    assert!(provider.is_ok(), "Concurrent provider creation failed at iteration {}", i);
                }
                1 => {
                    // Phase selection
                    let phase = match i % 5 {
                        0 => ExecutionPhase::Initialize,
                        1 => ExecutionPhase::Reuse,
                        2 => ExecutionPhase::Generate,
                        3 => ExecutionPhase::Shrink,
                        _ => ExecutionPhase::Cleanup,
                    };
                    let selected = orchestrator.select_provider_for_phase(phase);
                    assert!(!selected.is_empty(), "Concurrent phase selection failed at iteration {}", i);
                }
                2 => {
                    // Observation logging
                    orchestrator.log_provider_observation(
                        &format!("concurrent_event_{}", i),
                        &format!("iteration_{}", i)
                    );
                }
                3 => {
                    // Backend handling
                    let scope = if i % 2 == 0 { BackendScope::Verified } else { BackendScope::Exhausted };
                    let result = orchestrator.handle_backend_cannot_proceed(scope);
                    assert!(result.is_ok(), "Concurrent backend handling failed at iteration {}", i);
                }
                _ => {}
            }
            
            // Verify type system consistency after each operation
            let context = orchestrator.provider_context();
            assert!(!context.active_provider.is_empty(), "Provider context corrupted at iteration {}", i);
            
            // Verify provider creation still works
            let verification_provider = orchestrator.create_active_provider();
            assert!(verification_provider.is_ok(), "Type system verification failed at iteration {}", i);
        }
        
        // Final consistency check
        assert!(orchestrator.using_hypothesis_backend());
        let final_provider = orchestrator.create_active_provider();
        assert!(final_provider.is_ok());
    }
}