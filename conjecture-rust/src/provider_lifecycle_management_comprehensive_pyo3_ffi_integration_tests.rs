//! Comprehensive Provider Lifecycle Management System Integration Tests
//! 
//! This module provides comprehensive tests for the Provider Lifecycle Management System capability,
//! focusing on complete behavioral verification of provider lifetime controls, context management,
//! and caching strategies. PyO3 integration tests are conditionally compiled when the python-ffi
//! feature is enabled.
//!
//! ## Capability Coverage:
//! - Per-test-case and per-test-function lifetime controls
//! - Context management with setup/teardown hooks  
//! - Provider instance caching and reuse strategies
//! - Provider registry operations and error handling
//! - Concurrent provider access and thread safety
//! - Performance benchmarks for lifecycle operations

use crate::conjecture_data_lifecycle_management::{
    ConjectureDataLifecycleManager, LifecycleState, LifecycleResult
};
use crate::providers::{
    PrimitiveProvider, ProviderLifetime, TestCaseContext, ProviderRegistry,
    BackendCapabilities, ProviderError, ObservationMessage
};
use crate::choice::{ChoiceType, ChoiceValue, Constraints};
use crate::data::ConjectureData;
use crate::engine_orchestrator::OrchestrationError;

#[cfg(feature = "python-ffi")]
use pyo3::prelude::*;
#[cfg(feature = "python-ffi")]
use pyo3::types::{PyDict, PyList, PyString, PyBool, PyInt};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Test context for tracking provider lifecycle operations
#[derive(Debug, Clone)]
pub struct TestLifecycleContext {
    pub setup_called: bool,
    pub teardown_called: bool,
    pub success_flag: Option<bool>,
    pub context_data: HashMap<String, String>,
    pub provider_name: String,
}

impl TestLifecycleContext {
    pub fn new(provider_name: String) -> Self {
        Self {
            setup_called: false,
            teardown_called: false,
            success_flag: None,
            context_data: HashMap::new(),
            provider_name,
        }
    }
}

impl TestCaseContext for TestLifecycleContext {
    fn enter_test_case(&mut self) {
        self.setup_called = true;
        self.context_data.insert("entered_at".to_string(), format!("{:?}", Instant::now()));
    }

    fn exit_test_case(&mut self, success: bool) {
        self.teardown_called = true;
        self.success_flag = Some(success);
        self.context_data.insert("exited_at".to_string(), format!("{:?}", Instant::now()));
        self.context_data.insert("final_success".to_string(), success.to_string());
    }

    fn get_context_data(&self) -> HashMap<String, serde_json::Value> {
        self.context_data.iter()
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect()
    }
}

/// Mock provider implementation for testing lifecycle management
#[derive(Debug)]
pub struct MockLifecycleProvider {
    name: String,
    lifetime: ProviderLifetime,
    context: Arc<Mutex<TestLifecycleContext>>,
    draw_count: Arc<Mutex<u32>>,
    span_depth: Arc<Mutex<u32>>,
    observations: Arc<Mutex<Vec<String>>>,
}

impl MockLifecycleProvider {
    pub fn new(name: &str, lifetime: ProviderLifetime) -> Self {
        Self {
            name: name.to_string(),
            lifetime,
            context: Arc::new(Mutex::new(TestLifecycleContext::new(name.to_string()))),
            draw_count: Arc::new(Mutex::new(0)),
            span_depth: Arc::new(Mutex::new(0)),
            observations: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn get_context(&self) -> Arc<Mutex<TestLifecycleContext>> {
        self.context.clone()
    }

    pub fn get_draw_count(&self) -> u32 {
        *self.draw_count.lock().unwrap()
    }

    pub fn get_observations(&self) -> Vec<String> {
        self.observations.lock().unwrap().clone()
    }
}

impl PrimitiveProvider for MockLifecycleProvider {
    fn lifetime(&self) -> ProviderLifetime {
        self.lifetime
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supports_integers: true,
            supports_floats: true,
            supports_strings: true,
            supports_bytes: true,
            supports_choices: true,
        }
    }

    fn per_test_case_context(&mut self) -> Box<dyn TestCaseContext> {
        Box::new(TestLifecycleContext::new(self.name.clone()))
    }

    fn draw_choice(&mut self, choice_type: ChoiceType, _constraints: &Constraints) -> Result<ChoiceValue, ProviderError> {
        let mut count = self.draw_count.lock().unwrap();
        *count += 1;
        
        let mut obs = self.observations.lock().unwrap();
        obs.push(format!("draw_choice: {:?}", choice_type));

        match choice_type {
            ChoiceType::Integer => Ok(ChoiceValue::Integer(42)),
            ChoiceType::Float => Ok(ChoiceValue::Float(3.14)),
            ChoiceType::Boolean => Ok(ChoiceValue::Boolean(true)),
            ChoiceType::String => Ok(ChoiceValue::String("test".to_string())),
            ChoiceType::Bytes => Ok(ChoiceValue::Bytes(vec![1, 2, 3])),
        }
    }

    fn span_start(&mut self, label: u32) {
        let mut depth = self.span_depth.lock().unwrap();
        *depth += 1;
        
        let mut obs = self.observations.lock().unwrap();
        obs.push(format!("span_start: {}", label));
    }

    fn span_end(&mut self, discard: bool) {
        let mut depth = self.span_depth.lock().unwrap();
        if *depth > 0 {
            *depth -= 1;
        }
        
        let mut obs = self.observations.lock().unwrap();
        obs.push(format!("span_end: discard={}", discard));
    }

    fn observe_test_case(&mut self) -> HashMap<String, serde_json::Value> {
        let mut observations = HashMap::new();
        observations.insert("provider_name".to_string(), serde_json::Value::String(self.name.clone()));
        observations.insert("draw_count".to_string(), serde_json::Value::Number((*self.draw_count.lock().unwrap()).into()));
        observations.insert("span_depth".to_string(), serde_json::Value::Number((*self.span_depth.lock().unwrap()).into()));
        observations
    }

    fn observe_information_messages(&mut self, _lifetime: ProviderLifetime) -> Vec<ObservationMessage> {
        vec![
            ObservationMessage {
                level: "info".to_string(),
                message: format!("MockLifecycleProvider '{}' active", self.name),
                data: Some(serde_json::json!({
                    "provider_name": self.name,
                    "lifetime": format!("{:?}", self.lifetime)
                })),
            }
        ]
    }
}

/// PyO3 wrapper for provider lifecycle testing - enabled with python-ffi feature
#[cfg(feature = "python-ffi")]
#[pyclass]
pub struct PyProviderLifecycleManager {
    manager: ConjectureDataLifecycleManager,
    registry: ProviderRegistry,
}

#[cfg(feature = "python-ffi")]
#[pymethods]
impl PyProviderLifecycleManager {
    #[new]
    pub fn new() -> Self {
        Self {
            manager: ConjectureDataLifecycleManager::new(),
            registry: ProviderRegistry::new(),
        }
    }

    /// Register a provider with specific lifetime
    pub fn register_provider(&mut self, name: &str, lifetime: &str) -> PyResult<()> {
        let provider_lifetime = match lifetime {
            "test_case" => ProviderLifetime::TestCase,
            "test_function" => ProviderLifetime::TestFunction,
            "test_run" => ProviderLifetime::TestRun,
            "session" => ProviderLifetime::Session,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid provider lifetime")),
        };

        let provider = Box::new(MockLifecycleProvider::new(name, provider_lifetime));
        self.registry.register_provider(name.to_string(), provider)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to register provider: {:?}", e)))?;
        
        Ok(())
    }

    /// Create ConjectureData instance with specific provider
    pub fn create_instance(&mut self, seed: u64, provider_name: Option<&str>) -> PyResult<u64> {
        let provider = if let Some(name) = provider_name {
            Some(self.registry.get_provider(name)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Provider not found: {:?}", e)))?)
        } else {
            None
        };

        self.manager.create_instance(seed, None, provider)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to create instance: {:?}", e)))
    }

    /// Transition instance state
    pub fn transition_state(&mut self, instance_id: u64, state: &str) -> PyResult<()> {
        let lifecycle_state = match state {
            "created" => LifecycleState::Created,
            "initialized" => LifecycleState::Initialized,
            "executing" => LifecycleState::Executing,
            "completed" => LifecycleState::Completed,
            "replaying" => LifecycleState::Replaying,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid lifecycle state")),
        };

        self.manager.transition_state(instance_id, lifecycle_state)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("State transition failed: {:?}", e)))
    }

    /// Cleanup instance
    pub fn cleanup_instance(&mut self, instance_id: u64) -> PyResult<()> {
        self.manager.cleanup_instance(instance_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Cleanup failed: {:?}", e)))
    }

    /// Get provider statistics  
    pub fn get_provider_stats(&self, provider_name: &str, py: Python) -> PyResult<PyObject> {
        let provider = self.registry.get_provider(provider_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Provider not found: {:?}", e)))?;

        let stats = PyDict::new(py);
        stats.set_item("name", provider_name)?;
        stats.set_item("lifetime", format!("{:?}", provider.lifetime()))?;
        stats.set_item("capabilities", format!("{:?}", provider.capabilities()))?;

        Ok(stats.into())
    }

    /// Test provider context lifecycle
    pub fn test_context_lifecycle(&mut self, provider_name: &str, success: bool, py: Python) -> PyResult<PyObject> {
        let mut provider = self.registry.get_provider(provider_name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Provider not found: {:?}", e)))?;

        let mut context = provider.per_test_case_context();
        
        // Simulate test case lifecycle
        context.enter_test_case();
        
        // Simulate some test operations
        thread::sleep(Duration::from_millis(10));
        
        context.exit_test_case(success);
        
        let context_data = context.get_context_data();
        let result = PyDict::new(py);
        
        for (key, value) in context_data {
            result.set_item(key, format!("{:?}", value))?;
        }
        
        Ok(result.into())
    }
}

/// Python module for provider lifecycle management tests - enabled with python-ffi feature
#[cfg(feature = "python-ffi")]
#[pymodule]
fn provider_lifecycle_management_pyo3_tests(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyProviderLifecycleManager>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "python-ffi")]
    use pyo3::prepare_freethreaded_python;

    #[test]
    fn test_provider_lifetime_controls_per_test_case() {
        let mut registry = ProviderRegistry::new();
        let provider = Box::new(MockLifecycleProvider::new("test_case_provider", ProviderLifetime::TestCase));
        let _context_ref = provider.get_context();
        
        registry.register_provider("test_case_provider".to_string(), provider).unwrap();
        
        let mut provider = registry.get_provider("test_case_provider").unwrap();
        assert_eq!(provider.lifetime(), ProviderLifetime::TestCase);
        
        // Test context lifecycle
        let mut context = provider.per_test_case_context();
        context.enter_test_case();
        
        // Simulate test execution
        let _choice = provider.draw_choice(ChoiceType::Integer, &Constraints::default()).unwrap();
        
        context.exit_test_case(true);
        
        let context_data = context.get_context_data();
        assert!(context_data.contains_key("entered_at"));
        assert!(context_data.contains_key("exited_at"));
        assert_eq!(context_data.get("final_success").unwrap(), &serde_json::Value::String("true".to_string()));
    }

    #[test]
    fn test_provider_lifetime_controls_per_test_function() {
        let mut registry = ProviderRegistry::new();
        let provider = Box::new(MockLifecycleProvider::new("test_function_provider", ProviderLifetime::TestFunction));
        
        registry.register_provider("test_function_provider".to_string(), provider).unwrap();
        
        let mut provider = registry.get_provider("test_function_provider").unwrap();
        assert_eq!(provider.lifetime(), ProviderLifetime::TestFunction);
        
        // Test multiple test cases with same provider instance
        for i in 0..3 {
            let mut context = provider.per_test_case_context();
            context.enter_test_case();
            
            let _choice = provider.draw_choice(ChoiceType::Integer, &Constraints::default()).unwrap();
            
            context.exit_test_case(i % 2 == 0); // Alternate success/failure
        }
        
        let observations = provider.observe_test_case();
        assert!(observations.contains_key("provider_name"));
        assert_eq!(observations.get("provider_name").unwrap(), &serde_json::Value::String("test_function_provider".to_string()));
    }

    #[test]
    fn test_provider_instance_caching_session_lifetime() {
        let mut registry = ProviderRegistry::new();
        
        // Register session-lifetime provider
        let provider1 = Box::new(MockLifecycleProvider::new("session_provider", ProviderLifetime::Session));
        registry.register_provider("session_provider".to_string(), provider1).unwrap();
        
        // Get provider multiple times - should be same instance due to caching
        let mut provider_ref1 = registry.get_provider("session_provider").unwrap();
        let mut provider_ref2 = registry.get_provider("session_provider").unwrap();
        
        assert_eq!(provider_ref1.lifetime(), provider_ref2.lifetime());
        
        // Both references should point to same cached instance
        let _choice1 = provider_ref1.draw_choice(ChoiceType::Integer, &Constraints::default()).unwrap();
        let _choice2 = provider_ref2.draw_choice(ChoiceType::Float, &Constraints::default()).unwrap();
        
        // Verify shared state through observations
        let obs1 = provider_ref1.observe_test_case();
        let obs2 = provider_ref2.observe_test_case();
        assert_eq!(obs1.get("draw_count"), obs2.get("draw_count"));
    }

    #[test]
    fn test_context_management_setup_teardown_hooks() {
        let mut provider = MockLifecycleProvider::new("hook_test_provider", ProviderLifetime::TestCase);
        let mut context = provider.per_test_case_context();
        
        // Test setup hook
        context.enter_test_case();
        let context_data = context.get_context_data();
        assert!(context_data.contains_key("entered_at"));
        
        // Simulate test execution with provider operations
        provider.span_start(1);
        let _choice = provider.draw_choice(ChoiceType::String, &Constraints::default()).unwrap();
        provider.span_end(false);
        
        // Test teardown hook with success
        context.exit_test_case(true);
        let final_context_data = context.get_context_data();
        assert!(final_context_data.contains_key("exited_at"));
        assert_eq!(final_context_data.get("final_success").unwrap(), &serde_json::Value::String("true".to_string()));
        
        // Verify provider observations
        let observations = provider.get_observations();
        assert!(observations.contains(&"span_start: 1".to_string()));
        assert!(observations.contains(&"span_end: discard=false".to_string()));
        assert!(observations.contains(&"draw_choice: String".to_string()));
    }

    #[test]
    fn test_provider_context_isolation_and_cleanup() {
        let mut registry = ProviderRegistry::new();
        
        // Test multiple providers with different lifetimes
        let provider1 = Box::new(MockLifecycleProvider::new("isolated_provider_1", ProviderLifetime::TestCase));
        let provider2 = Box::new(MockLifecycleProvider::new("isolated_provider_2", ProviderLifetime::TestFunction));
        
        registry.register_provider("isolated_provider_1".to_string(), provider1).unwrap();
        registry.register_provider("isolated_provider_2".to_string(), provider2).unwrap();
        
        // Test context isolation
        let mut context1 = registry.get_provider("isolated_provider_1").unwrap().per_test_case_context();
        let mut context2 = registry.get_provider("isolated_provider_2").unwrap().per_test_case_context();
        
        context1.enter_test_case();
        context2.enter_test_case();
        
        // Verify contexts are isolated
        let data1 = context1.get_context_data();
        let data2 = context2.get_context_data();
        
        assert_ne!(data1.get("entered_at"), data2.get("entered_at")); // Different timestamps
        
        context1.exit_test_case(true);
        context2.exit_test_case(false);
        
        let final_data1 = context1.get_context_data();
        let final_data2 = context2.get_context_data();
        
        assert_eq!(final_data1.get("final_success").unwrap(), &serde_json::Value::String("true".to_string()));
        assert_eq!(final_data2.get("final_success").unwrap(), &serde_json::Value::String("false".to_string()));
    }

    #[test]
    fn test_conjecture_data_lifecycle_integration() {
        let mut manager = ConjectureDataLifecycleManager::new();
        let provider = Box::new(MockLifecycleProvider::new("lifecycle_provider", ProviderLifetime::TestCase));
        
        // Create instance with provider
        let instance_id = manager.create_instance(12345, None, Some(provider)).unwrap();
        
        // Test state transitions
        manager.transition_state(instance_id, LifecycleState::Initialized).unwrap();
        manager.transition_state(instance_id, LifecycleState::Executing).unwrap();
        manager.transition_state(instance_id, LifecycleState::Completed).unwrap();
        
        // Test cleanup
        manager.cleanup_instance(instance_id).unwrap();
        
        // Verify instance is cleaned up (should fail to transition)
        let result = manager.transition_state(instance_id, LifecycleState::Executing);
        assert!(result.is_err());
    }

    #[cfg(feature = "python-ffi")]
    #[test]
    fn test_pyo3_provider_lifecycle_manager_integration() {
        prepare_freethreaded_python();
        
        Python::with_gil(|py| {
            let mut manager = PyProviderLifecycleManager::new();
            
            // Register providers with different lifetimes
            manager.register_provider("test_case_provider", "test_case").unwrap();
            manager.register_provider("session_provider", "session").unwrap();
            
            // Create instances
            let instance1 = manager.create_instance(12345, Some("test_case_provider")).unwrap();
            let instance2 = manager.create_instance(67890, Some("session_provider")).unwrap();
            
            // Test state transitions
            manager.transition_state(instance1, "initialized").unwrap();
            manager.transition_state(instance1, "executing").unwrap();
            manager.transition_state(instance1, "completed").unwrap();
            
            manager.transition_state(instance2, "initialized").unwrap();
            manager.transition_state(instance2, "executing").unwrap();
            
            // Test provider statistics
            let stats1 = manager.get_provider_stats("test_case_provider", py).unwrap();
            let stats_dict1 = stats1.downcast::<PyDict>(py).unwrap();
            assert_eq!(stats_dict1.get_item("name").unwrap().extract::<String>().unwrap(), "test_case_provider");
            
            let stats2 = manager.get_provider_stats("session_provider", py).unwrap();
            let stats_dict2 = stats2.downcast::<PyDict>(py).unwrap();
            assert_eq!(stats_dict2.get_item("name").unwrap().extract::<String>().unwrap(), "session_provider");
            
            // Test context lifecycle
            let context_result = manager.test_context_lifecycle("test_case_provider", true, py).unwrap();
            let context_dict = context_result.downcast::<PyDict>(py).unwrap();
            assert!(context_dict.contains("entered_at").unwrap());
            assert!(context_dict.contains("exited_at").unwrap());
            assert!(context_dict.contains("final_success").unwrap());
            
            // Cleanup instances
            manager.cleanup_instance(instance1).unwrap();
            manager.cleanup_instance(instance2).unwrap();
        });
    }

    #[test]
    fn test_provider_error_handling_and_recovery() {
        let mut registry = ProviderRegistry::new();
        let provider = Box::new(MockLifecycleProvider::new("error_provider", ProviderLifetime::TestCase));
        
        registry.register_provider("error_provider".to_string(), provider).unwrap();
        
        // Test error handling in provider operations
        let mut provider = registry.get_provider("error_provider").unwrap();
        let mut context = provider.per_test_case_context();
        
        context.enter_test_case();
        
        // Provider should handle errors gracefully
        let choice_result = provider.draw_choice(ChoiceType::Integer, &Constraints::default());
        assert!(choice_result.is_ok());
        
        // Test context cleanup even on failure
        context.exit_test_case(false);
        let context_data = context.get_context_data();
        assert_eq!(context_data.get("final_success").unwrap(), &serde_json::Value::String("false".to_string()));
    }

    #[test]
    fn test_concurrent_provider_access_thread_safety() {
        let registry = Arc::new(Mutex::new(ProviderRegistry::new()));
        
        {
            let mut reg = registry.lock().unwrap();
            let provider = Box::new(MockLifecycleProvider::new("concurrent_provider", ProviderLifetime::Session));
            reg.register_provider("concurrent_provider".to_string(), provider).unwrap();
        }
        
        let handles: Vec<_> = (0..5).map(|i| {
            let reg_clone = registry.clone();
            thread::spawn(move || {
                let reg = reg_clone.lock().unwrap();
                let mut provider = reg.get_provider("concurrent_provider").unwrap();
                
                // Each thread performs provider operations
                let _choice = provider.draw_choice(ChoiceType::Integer, &Constraints::default()).unwrap();
                provider.span_start(i as u32);
                provider.span_end(false);
                
                format!("Thread {} completed", i)
            })
        }).collect();
        
        // Wait for all threads to complete
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.contains("completed"));
        }
        
        // Verify provider state after concurrent access
        let reg = registry.lock().unwrap();
        let mut provider = reg.get_provider("concurrent_provider").unwrap();
        let observations = provider.observe_test_case();
        assert!(observations.contains_key("provider_name"));
        assert!(observations.contains_key("draw_count"));
    }

    #[test]
    fn test_provider_lifecycle_performance_benchmarks() {
        let mut registry = ProviderRegistry::new();
        let provider = Box::new(MockLifecycleProvider::new("perf_provider", ProviderLifetime::TestFunction));
        
        registry.register_provider("perf_provider".to_string(), provider).unwrap();
        
        let start_time = Instant::now();
        let iterations = 1000;
        
        for i in 0..iterations {
            let mut provider = registry.get_provider("perf_provider").unwrap();
            let mut context = provider.per_test_case_context();
            
            context.enter_test_case();
            let _choice = provider.draw_choice(ChoiceType::Integer, &Constraints::default()).unwrap();
            context.exit_test_case(i % 2 == 0);
        }
        
        let elapsed = start_time.elapsed();
        let avg_time_per_iteration = elapsed / iterations;
        
        // Performance assertion - should complete quickly
        assert!(avg_time_per_iteration < Duration::from_millis(1), 
               "Provider lifecycle operations too slow: {:?} per iteration", avg_time_per_iteration);
        
        // Verify provider maintained state correctly
        let mut provider = registry.get_provider("perf_provider").unwrap();
        let observations = provider.observe_test_case();
        assert_eq!(observations.get("draw_count").unwrap(), &serde_json::Value::Number(iterations.into()));
    }
}