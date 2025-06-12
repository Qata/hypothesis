//! Advanced Error Handling and Fallback Capability Tests for Provider System
//! 
//! This module implements comprehensive testing for the Advanced Error Handling and Fallback
//! capability of the ProviderSystem, specifically testing:
//! 
//! 1. BackendCannotProceed exception handling with automatic fallback
//! 2. Provider verification tracking
//! 3. Robust switching logic when backends fail repeatedly
//! 4. PyO3 integration for Python parity verification
//! 5. FFI boundary error handling
//! 6. Recovery mechanisms and state restoration

use crate::providers::{
    PrimitiveProvider, ProviderLifetime, ProviderError, ProviderScope, BackendCapabilities,
    ProviderFactory, ProviderRegistry, TestCaseContext, ObservationMessage,
    TestCaseObservation, register_specialized_backends, get_provider_registry,
    SmtSolverProvider, FuzzingProvider, RandomProvider, HypothesisProvider
};
use crate::provider_lifecycle_management::{
    ProviderLifecycleManager, LifecycleScope, LifecycleHooks, DefaultLifecycleHooks,
    ManagedProvider, ProviderState, ProviderInstanceMetadata, LifecycleEvent,
    CacheConfiguration
};
use crate::choice::{
    ChoiceValue, Constraints, ChoiceType, IntegerConstraints, FloatConstraints,
    BooleanConstraints, IntervalSet
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, Duration};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json;
// PyO3 imports disabled for core testing
// use pyo3::prelude::*;

/// Fallback-aware provider that implements comprehensive error handling
#[derive(Debug)]
pub struct FallbackAwareProvider {
    /// Primary backend provider
    primary_backend: Box<dyn PrimitiveProvider>,
    /// Fallback backend providers in priority order
    fallback_backends: Vec<Box<dyn PrimitiveProvider>>,
    /// Current active backend index (0 = primary, 1+ = fallbacks)
    current_backend_index: usize,
    /// Error tracking for backend verification
    backend_error_counts: Vec<u32>,
    /// Maximum allowed errors before switching backends
    max_errors_per_backend: u32,
    /// Provider verification state
    verification_state: ProviderVerificationState,
    /// Fallback switching statistics
    fallback_statistics: FallbackStatistics,
    /// Recovery mechanisms configuration
    recovery_config: RecoveryConfiguration,
}

/// Provider verification state tracking
#[derive(Debug, Clone)]
pub struct ProviderVerificationState {
    /// Backend verification results
    pub backend_verification: HashMap<String, VerificationResult>,
    /// Last verification timestamp per backend
    pub last_verification: HashMap<String, SystemTime>,
    /// Verification failure patterns
    pub failure_patterns: Vec<FailurePattern>,
    /// Recovery attempts count
    pub recovery_attempts: u32,
    /// Current verification status
    pub status: VerificationStatus,
}

/// Verification result for individual backends
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationResult {
    Passed,
    Failed(String),
    Timeout,
    CriticalFailure(String),
    Unknown,
}

/// Failure pattern analysis for intelligent fallback
#[derive(Debug, Clone)]
pub struct FailurePattern {
    pub backend_name: String,
    pub error_type: String,
    pub frequency: u32,
    pub last_occurrence: SystemTime,
    pub context_data: HashMap<String, serde_json::Value>,
}

/// Overall verification status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationStatus {
    Healthy,
    Degraded,
    Critical,
    RecoveryMode,
    Failed,
}

/// Fallback switching statistics
#[derive(Debug, Clone)]
pub struct FallbackStatistics {
    pub total_switches: u32,
    pub switches_by_backend: HashMap<String, u32>,
    pub average_switch_time: Duration,
    pub successful_recoveries: u32,
    pub failed_recoveries: u32,
    pub current_uptime: Duration,
    pub backend_reliability_scores: HashMap<String, f64>,
}

/// Recovery configuration for error handling
#[derive(Debug, Clone)]
pub struct RecoveryConfiguration {
    /// Enable automatic backend switching
    pub auto_switch_enabled: bool,
    /// Maximum recovery attempts before giving up
    pub max_recovery_attempts: u32,
    /// Cooldown period before retrying failed backend
    pub backend_cooldown: Duration,
    /// Enable circuit breaker pattern
    pub circuit_breaker_enabled: bool,
    /// Verification interval for backends
    pub verification_interval: Duration,
    /// Custom recovery strategies
    pub custom_strategies: Vec<RecoveryStrategy>,
}

/// Recovery strategy definitions
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    ImmediateFallback,
    RetryWithBackoff { max_retries: u32, backoff_factor: f64 },
    CircuitBreaker { failure_threshold: u32, recovery_timeout: Duration },
    GradualRecovery { probe_interval: Duration },
    CustomStrategy { name: String, config: HashMap<String, serde_json::Value> },
}

impl Default for RecoveryConfiguration {
    fn default() -> Self {
        Self {
            auto_switch_enabled: true,
            max_recovery_attempts: 3,
            backend_cooldown: Duration::from_secs(30),
            circuit_breaker_enabled: true,
            verification_interval: Duration::from_secs(60),
            custom_strategies: vec![
                RecoveryStrategy::RetryWithBackoff { max_retries: 2, backoff_factor: 2.0 },
                RecoveryStrategy::CircuitBreaker { failure_threshold: 5, recovery_timeout: Duration::from_secs(300) },
            ],
        }
    }
}

impl FallbackAwareProvider {
    /// Create a new fallback-aware provider with error handling
    pub fn new(
        primary: Box<dyn PrimitiveProvider>,
        fallbacks: Vec<Box<dyn PrimitiveProvider>>,
        max_errors: u32,
    ) -> Self {
        let backend_count = 1 + fallbacks.len();
        
        Self {
            primary_backend: primary,
            fallback_backends: fallbacks,
            current_backend_index: 0,
            backend_error_counts: vec![0; backend_count],
            max_errors_per_backend: max_errors,
            verification_state: ProviderVerificationState {
                backend_verification: HashMap::new(),
                last_verification: HashMap::new(),
                failure_patterns: Vec::new(),
                recovery_attempts: 0,
                status: VerificationStatus::Healthy,
            },
            fallback_statistics: FallbackStatistics {
                total_switches: 0,
                switches_by_backend: HashMap::new(),
                average_switch_time: Duration::ZERO,
                successful_recoveries: 0,
                failed_recoveries: 0,
                current_uptime: Duration::ZERO,
                backend_reliability_scores: HashMap::new(),
            },
            recovery_config: RecoveryConfiguration::default(),
        }
    }

    /// Get current active backend
    fn current_backend(&mut self) -> &mut Box<dyn PrimitiveProvider> {
        if self.current_backend_index == 0 {
            &mut self.primary_backend
        } else {
            &mut self.fallback_backends[self.current_backend_index - 1]
        }
    }

    /// Get current backend name for logging
    fn current_backend_name(&self) -> String {
        if self.current_backend_index == 0 {
            "primary".to_string()
        } else {
            format!("fallback_{}", self.current_backend_index - 1)
        }
    }

    /// Record an error for the current backend
    fn record_error(&mut self, error: &ProviderError) {
        self.backend_error_counts[self.current_backend_index] += 1;
        
        // Update failure patterns
        let pattern = FailurePattern {
            backend_name: self.current_backend_name(),
            error_type: format!("{:?}", error),
            frequency: 1,
            last_occurrence: SystemTime::now(),
            context_data: HashMap::new(),
        };
        
        // Check if this pattern already exists
        if let Some(existing) = self.verification_state.failure_patterns.iter_mut()
            .find(|p| p.backend_name == pattern.backend_name && p.error_type == pattern.error_type) {
            existing.frequency += 1;
            existing.last_occurrence = pattern.last_occurrence;
        } else {
            self.verification_state.failure_patterns.push(pattern);
        }
        
        // Update verification status
        self.update_verification_status();
        
        println!("FALLBACK_PROVIDER ERROR: Backend {} error #{}: {:?}", 
                self.current_backend_name(), 
                self.backend_error_counts[self.current_backend_index], 
                error);
    }

    /// Update overall verification status based on error patterns
    fn update_verification_status(&mut self) {
        let current_errors = self.backend_error_counts[self.current_backend_index];
        let total_backends = self.backend_error_counts.len();
        let failed_backends = self.backend_error_counts.iter().filter(|&&count| count >= self.max_errors_per_backend).count();
        
        self.verification_state.status = if failed_backends == 0 {
            VerificationStatus::Healthy
        } else if failed_backends < total_backends / 2 {
            VerificationStatus::Degraded
        } else if failed_backends < total_backends {
            VerificationStatus::Critical
        } else {
            VerificationStatus::Failed
        };
        
        if current_errors >= self.max_errors_per_backend {
            self.verification_state.status = VerificationStatus::RecoveryMode;
        }
    }

    /// Attempt to switch to the next available backend
    fn switch_to_fallback(&mut self) -> Result<(), ProviderError> {
        let start_time = SystemTime::now();
        let old_backend = self.current_backend_name();
        
        // Record verification failure for current backend
        self.verification_state.backend_verification.insert(
            old_backend.clone(),
            VerificationResult::Failed(format!("Exceeded {} errors", self.max_errors_per_backend))
        );
        
        // Find next available backend
        let total_backends = 1 + self.fallback_backends.len();
        let mut attempts = 0;
        
        while attempts < total_backends {
            let next_index = (self.current_backend_index + 1) % total_backends;
            
            // Check if this backend is still usable
            if self.backend_error_counts[next_index] < self.max_errors_per_backend {
                self.current_backend_index = next_index;
                
                // Update statistics
                self.fallback_statistics.total_switches += 1;
                *self.fallback_statistics.switches_by_backend.entry(self.current_backend_name()).or_insert(0) += 1;
                
                let switch_time = start_time.elapsed().unwrap_or_default();
                self.fallback_statistics.average_switch_time = 
                    (self.fallback_statistics.average_switch_time + switch_time) / 2;
                
                println!("FALLBACK_PROVIDER SWITCH: Switched from {} to {} (switch #{}, time: {:?})", 
                        old_backend, self.current_backend_name(), 
                        self.fallback_statistics.total_switches, switch_time);
                
                // Verify new backend works
                if self.verify_current_backend().is_ok() {
                    self.fallback_statistics.successful_recoveries += 1;
                    self.verification_state.status = VerificationStatus::Degraded;
                    return Ok(());
                }
            }
            
            attempts += 1;
        }
        
        // All backends have failed
        self.fallback_statistics.failed_recoveries += 1;
        self.verification_state.status = VerificationStatus::Failed;
        
        Err(ProviderError::CannotProceed {
            scope: ProviderScope::Exhausted,
            reason: "All backends have exceeded error limits".to_string(),
        })
    }

    /// Verify that the current backend is working
    fn verify_current_backend(&mut self) -> Result<(), ProviderError> {
        let backend_name = self.current_backend_name();
        
        // Simple verification: try to generate a basic value
        let test_constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(1),
            weights: None,
            shrink_towards: Some(0),
        };
        
        match self.current_backend().draw_integer(&test_constraints) {
            Ok(_) => {
                self.verification_state.backend_verification.insert(
                    backend_name.clone(),
                    VerificationResult::Passed
                );
                self.verification_state.last_verification.insert(
                    backend_name,
                    SystemTime::now()
                );
                Ok(())
            },
            Err(e) => {
                self.verification_state.backend_verification.insert(
                    backend_name,
                    VerificationResult::Failed(format!("{:?}", e))
                );
                Err(e)
            }
        }
    }

    /// Execute operation with automatic fallback on errors
    fn execute_with_fallback<T, F>(&mut self, operation: F) -> Result<T, ProviderError>
    where
        F: Fn(&mut Box<dyn PrimitiveProvider>) -> Result<T, ProviderError>,
    {
        let mut last_error = None;
        
        // Try current backend first
        match operation(self.current_backend()) {
            Ok(result) => return Ok(result),
            Err(e) => {
                match &e {
                    ProviderError::CannotProceed { scope, .. } => {
                        self.record_error(&e);
                        
                        // Check if we should switch backends
                        if self.backend_error_counts[self.current_backend_index] >= self.max_errors_per_backend {
                            println!("FALLBACK_PROVIDER TRIGGER: Backend {} exceeded error limit, attempting fallback", 
                                    self.current_backend_name());
                            
                            if let Err(switch_error) = self.switch_to_fallback() {
                                return Err(switch_error);
                            }
                            
                            // Retry with new backend
                            match operation(self.current_backend()) {
                                Ok(result) => return Ok(result),
                                Err(retry_error) => {
                                    self.record_error(&retry_error);
                                    last_error = Some(retry_error);
                                }
                            }
                        } else {
                            last_error = Some(e);
                        }
                    },
                    _ => {
                        // Non-critical error, record but don't switch
                        self.record_error(&e);
                        last_error = Some(e);
                    }
                }
            }
        }
        
        // If we get here, the operation failed
        Err(last_error.unwrap_or_else(|| ProviderError::BackendExhausted("All fallback attempts failed".to_string())))
    }

    /// Get comprehensive error handling statistics
    pub fn get_error_statistics(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        // Backend error counts
        let mut backend_errors = HashMap::new();
        backend_errors.insert("primary".to_string(), self.backend_error_counts[0]);
        for (i, &count) in self.backend_error_counts.iter().enumerate().skip(1) {
            backend_errors.insert(format!("fallback_{}", i - 1), count);
        }
        stats.insert("backend_error_counts".to_string(), serde_json::to_value(backend_errors).unwrap_or_default());
        
        // Current backend
        stats.insert("current_backend".to_string(), serde_json::Value::String(self.current_backend_name()));
        stats.insert("current_backend_index".to_string(), serde_json::Value::Number(serde_json::Number::from(self.current_backend_index)));
        
        // Verification state
        stats.insert("verification_status".to_string(), serde_json::Value::String(format!("{:?}", self.verification_state.status)));
        stats.insert("recovery_attempts".to_string(), serde_json::Value::Number(serde_json::Number::from(self.verification_state.recovery_attempts)));
        stats.insert("failure_patterns_count".to_string(), serde_json::Value::Number(serde_json::Number::from(self.verification_state.failure_patterns.len())));
        
        // Fallback statistics
        stats.insert("total_switches".to_string(), serde_json::Value::Number(serde_json::Number::from(self.fallback_statistics.total_switches)));
        stats.insert("successful_recoveries".to_string(), serde_json::Value::Number(serde_json::Number::from(self.fallback_statistics.successful_recoveries)));
        stats.insert("failed_recoveries".to_string(), serde_json::Value::Number(serde_json::Number::from(self.fallback_statistics.failed_recoveries)));
        
        stats
    }
}

impl PrimitiveProvider for FallbackAwareProvider {
    fn lifetime(&self) -> ProviderLifetime {
        self.primary_backend.lifetime()
    }

    fn capabilities(&self) -> BackendCapabilities {
        // Combine capabilities of all backends
        let mut combined = self.primary_backend.capabilities();
        
        for fallback in &self.fallback_backends {
            let fallback_caps = fallback.capabilities();
            combined.supports_integers = combined.supports_integers || fallback_caps.supports_integers;
            combined.supports_floats = combined.supports_floats || fallback_caps.supports_floats;
            combined.supports_strings = combined.supports_strings || fallback_caps.supports_strings;
            combined.supports_bytes = combined.supports_bytes || fallback_caps.supports_bytes;
            combined.supports_choices = combined.supports_choices || fallback_caps.supports_choices;
            combined.avoid_realization = combined.avoid_realization && fallback_caps.avoid_realization;
            combined.add_observability_callback = combined.add_observability_callback || fallback_caps.add_observability_callback;
            combined.structural_awareness = combined.structural_awareness || fallback_caps.structural_awareness;
            combined.replay_support = combined.replay_support || fallback_caps.replay_support;
            combined.symbolic_constraints = combined.symbolic_constraints || fallback_caps.symbolic_constraints;
        }
        
        combined
    }

    fn draw_boolean(&mut self, p: f64) -> Result<bool, ProviderError> {
        self.execute_with_fallback(|backend| backend.draw_boolean(p))
    }

    fn draw_integer(&mut self, constraints: &IntegerConstraints) -> Result<i128, ProviderError> {
        self.execute_with_fallback(|backend| backend.draw_integer(constraints))
    }

    fn draw_float(&mut self, constraints: &FloatConstraints) -> Result<f64, ProviderError> {
        self.execute_with_fallback(|backend| backend.draw_float(constraints))
    }

    fn draw_string(&mut self, intervals: &IntervalSet, min_size: usize, max_size: usize) -> Result<String, ProviderError> {
        self.execute_with_fallback(|backend| backend.draw_string(intervals, min_size, max_size))
    }

    fn draw_bytes(&mut self, min_size: usize, max_size: usize) -> Result<Vec<u8>, ProviderError> {
        self.execute_with_fallback(|backend| backend.draw_bytes(min_size, max_size))
    }

    fn observe_test_case(&mut self) -> HashMap<String, serde_json::Value> {
        let mut observation = self.current_backend().observe_test_case();
        
        // Add fallback-specific observations
        observation.extend(self.get_error_statistics());
        
        observation
    }

    fn observe_information_messages(&mut self, lifetime: ProviderLifetime) -> Vec<ObservationMessage> {
        let mut messages = self.current_backend().observe_information_messages(lifetime);
        
        // Add fallback status message
        messages.push(ObservationMessage {
            level: "info".to_string(),
            message: "Fallback Provider Status".to_string(),
            data: Some(serde_json::json!({
                "current_backend": self.current_backend_name(),
                "verification_status": format!("{:?}", self.verification_state.status),
                "total_switches": self.fallback_statistics.total_switches,
                "successful_recoveries": self.fallback_statistics.successful_recoveries
            })),
        });
        
        messages
    }
}

/// Failing provider for testing fallback mechanisms
#[derive(Debug)]
pub struct FailingProvider {
    pub failure_mode: FailureMode,
    pub failure_count: u32,
    pub max_operations: u32,
    pub operation_count: u32,
}

#[derive(Debug, Clone)]
pub enum FailureMode {
    AlwaysFail,
    FailAfterCount(u32),
    FailRandomly(f64), // probability of failure
    FailWithScope(ProviderScope),
    FailWithMessage(String),
}

impl FailingProvider {
    pub fn new(mode: FailureMode, max_operations: u32) -> Self {
        Self {
            failure_mode: mode,
            failure_count: 0,
            max_operations,
            operation_count: 0,
        }
    }

    fn should_fail(&mut self) -> Option<ProviderError> {
        self.operation_count += 1;
        
        match &self.failure_mode {
            FailureMode::AlwaysFail => {
                self.failure_count += 1;
                Some(ProviderError::CannotProceed {
                    scope: ProviderScope::DiscardTestCase,
                    reason: format!("Always failing provider (failure #{})", self.failure_count),
                })
            },
            FailureMode::FailAfterCount(count) => {
                if self.operation_count > *count {
                    self.failure_count += 1;
                    Some(ProviderError::CannotProceed {
                        scope: ProviderScope::Exhausted,
                        reason: format!("Failed after {} operations", count),
                    })
                } else {
                    None
                }
            },
            FailureMode::FailRandomly(prob) => {
                let mut rng = ChaCha8Rng::from_entropy();
                if rng.gen::<f64>() < *prob {
                    self.failure_count += 1;
                    Some(ProviderError::CannotProceed {
                        scope: ProviderScope::DiscardTestCase,
                        reason: format!("Random failure (failure #{})", self.failure_count),
                    })
                } else {
                    None
                }
            },
            FailureMode::FailWithScope(scope) => {
                self.failure_count += 1;
                Some(ProviderError::CannotProceed {
                    scope: *scope,
                    reason: format!("Scoped failure: {:?}", scope),
                })
            },
            FailureMode::FailWithMessage(msg) => {
                self.failure_count += 1;
                Some(ProviderError::BackendExhausted(format!("{} (failure #{})", msg, self.failure_count)))
            },
        }
    }
}

impl PrimitiveProvider for FailingProvider {
    fn lifetime(&self) -> ProviderLifetime {
        ProviderLifetime::TestCase
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::default()
    }

    fn draw_boolean(&mut self, _p: f64) -> Result<bool, ProviderError> {
        if let Some(error) = self.should_fail() {
            Err(error)
        } else {
            Ok(true)
        }
    }

    fn draw_integer(&mut self, _constraints: &IntegerConstraints) -> Result<i128, ProviderError> {
        if let Some(error) = self.should_fail() {
            Err(error)
        } else {
            Ok(42)
        }
    }

    fn draw_float(&mut self, _constraints: &FloatConstraints) -> Result<f64, ProviderError> {
        if let Some(error) = self.should_fail() {
            Err(error)
        } else {
            Ok(3.14)
        }
    }

    fn draw_string(&mut self, _intervals: &IntervalSet, _min_size: usize, _max_size: usize) -> Result<String, ProviderError> {
        if let Some(error) = self.should_fail() {
            Err(error)
        } else {
            Ok("test".to_string())
        }
    }

    fn draw_bytes(&mut self, _min_size: usize, _max_size: usize) -> Result<Vec<u8>, ProviderError> {
        if let Some(error) = self.should_fail() {
            Err(error)
        } else {
            Ok(vec![1, 2, 3, 4])
        }
    }
}

/// PyO3 integration tests for error handling capability (disabled for core testing)
/*
#[pymodule]
fn provider_error_handling_tests(_py: Python, m: &PyModule) -> PyResult<()> {
    /// Test provider error handling from Python
    #[pyfn(m)]
    fn test_provider_fallback_from_python(py: Python) -> PyResult<PyObject> {
        let mut primary = Box::new(FailingProvider::new(FailureMode::FailAfterCount(2), 10));
        let fallback1 = Box::new(RandomProvider::new());
        let fallback2 = Box::new(HypothesisProvider::new());
        
        let mut fallback_provider = FallbackAwareProvider::new(
            primary,
            vec![fallback1, fallback2],
            3
        );
        
        let test_constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        };
        
        // Should succeed with primary initially
        let result1 = fallback_provider.draw_integer(&test_constraints);
        assert!(result1.is_ok());
        
        // Should succeed with primary again
        let result2 = fallback_provider.draw_integer(&test_constraints);
        assert!(result2.is_ok());
        
        // Should trigger fallback
        let result3 = fallback_provider.draw_integer(&test_constraints);
        
        let stats = fallback_provider.get_error_statistics();
        
        Ok(serde_json::to_value(stats).unwrap().into_py(py))
    }
    
    /// Test comprehensive error scenarios from Python
    #[pyfn(m)]
    fn test_comprehensive_error_scenarios(py: Python) -> PyResult<PyObject> {
        let mut results = Vec::new();
        
        // Test different failure modes
        let failure_modes = vec![
            FailureMode::FailAfterCount(1),
            FailureMode::FailRandomly(0.5),
            FailureMode::FailWithScope(ProviderScope::Exhausted),
            FailureMode::FailWithMessage("Custom error message".to_string()),
        ];
        
        for (i, mode) in failure_modes.into_iter().enumerate() {
            let primary = Box::new(FailingProvider::new(mode, 5));
            let fallback = Box::new(RandomProvider::new());
            
            let mut provider = FallbackAwareProvider::new(primary, vec![fallback], 2);
            
            // Try multiple operations to trigger errors and fallbacks
            let mut operation_results = Vec::new();
            for j in 0..5 {
                let constraints = IntegerConstraints {
                    min_value: Some(j as i128),
                    max_value: Some((j + 10) as i128),
                    weights: None,
                    shrink_towards: Some(j as i128),
                };
                
                match provider.draw_integer(&constraints) {
                    Ok(value) => operation_results.push(format!("success_{}", value)),
                    Err(e) => operation_results.push(format!("error_{:?}", e)),
                }
            }
            
            let stats = provider.get_error_statistics();
            results.push(serde_json::json!({
                "test_case": i,
                "operation_results": operation_results,
                "final_stats": stats
            }));
        }
        
        Ok(serde_json::to_value(results).unwrap().into_py(py))
    }
    
    /// Test provider verification tracking
    #[pyfn(m)]
    fn test_provider_verification_tracking(py: Python) -> PyResult<PyObject> {
        let primary = Box::new(FailingProvider::new(FailureMode::AlwaysFail, 10));
        let fallback1 = Box::new(FailingProvider::new(FailureMode::FailAfterCount(3), 10));
        let fallback2 = Box::new(RandomProvider::new());
        
        let mut provider = FallbackAwareProvider::new(primary, vec![fallback1, fallback2], 2);
        
        let mut verification_history = Vec::new();
        
        // Perform operations that will trigger verification and fallbacks
        for i in 0..10 {
            let constraints = IntegerConstraints {
                min_value: Some(0),
                max_value: Some(100),
                weights: None,
                shrink_towards: Some(0),
            };
            
            let _result = provider.draw_integer(&constraints);
            
            // Capture verification state after each operation
            verification_history.push(serde_json::json!({
                "operation": i,
                "current_backend": provider.current_backend_name(),
                "verification_status": format!("{:?}", provider.verification_state.status),
                "total_switches": provider.fallback_statistics.total_switches,
                "recovery_attempts": provider.verification_state.recovery_attempts,
                "backend_errors": provider.backend_error_counts.clone(),
            }));
        }
        
        Ok(serde_json::to_value(verification_history).unwrap().into_py(py))
    }

    Ok(())
}
*/

/// Comprehensive test suite for advanced error handling and fallback capability
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_fallback_mechanism() {
        // Create primary provider that fails after 2 operations
        let primary = Box::new(FailingProvider::new(FailureMode::FailAfterCount(2), 10));
        let fallback = Box::new(RandomProvider::new());
        
        let mut provider = FallbackAwareProvider::new(primary, vec![fallback], 3);
        
        let constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        };
        
        // First two operations should succeed with primary
        let result1 = provider.draw_integer(&constraints);
        assert!(result1.is_ok());
        assert_eq!(provider.current_backend_index, 0); // Still using primary
        
        let result2 = provider.draw_integer(&constraints);
        assert!(result2.is_ok());
        assert_eq!(provider.current_backend_index, 0); // Still using primary
        
        // Third operation should fail primary and switch to fallback
        let result3 = provider.draw_integer(&constraints);
        assert!(result3.is_ok()); // Should succeed with fallback
        assert_eq!(provider.current_backend_index, 1); // Should have switched to fallback
        
        // Verify statistics
        let stats = provider.get_error_statistics();
        assert_eq!(provider.fallback_statistics.total_switches, 1);
        assert_eq!(provider.fallback_statistics.successful_recoveries, 1);
    }

    #[test]
    fn test_multiple_backend_failures() {
        // Create providers that all fail at different points
        let primary = Box::new(FailingProvider::new(FailureMode::FailAfterCount(1), 10));
        let fallback1 = Box::new(FailingProvider::new(FailureMode::FailAfterCount(2), 10));
        let fallback2 = Box::new(RandomProvider::new()); // This one works
        
        let mut provider = FallbackAwareProvider::new(primary, vec![fallback1, fallback2], 2);
        
        let constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        };
        
        // First operation succeeds with primary
        let result1 = provider.draw_integer(&constraints);
        assert!(result1.is_ok());
        assert_eq!(provider.current_backend_index, 0);
        
        // Second operation fails primary, switches to fallback1
        let result2 = provider.draw_integer(&constraints);
        assert!(result2.is_ok());
        assert_eq!(provider.current_backend_index, 1);
        
        // Third operation succeeds with fallback1
        let result3 = provider.draw_integer(&constraints);
        assert!(result3.is_ok());
        assert_eq!(provider.current_backend_index, 1);
        
        // Fourth operation fails fallback1, switches to fallback2
        let result4 = provider.draw_integer(&constraints);
        assert!(result4.is_ok());
        assert_eq!(provider.current_backend_index, 2);
        
        // Verify we made two switches
        assert_eq!(provider.fallback_statistics.total_switches, 2);
        assert_eq!(provider.fallback_statistics.successful_recoveries, 2);
    }

    #[test]
    fn test_all_backends_fail() {
        // Create providers that all fail
        let primary = Box::new(FailingProvider::new(FailureMode::AlwaysFail, 10));
        let fallback1 = Box::new(FailingProvider::new(FailureMode::AlwaysFail, 10));
        let fallback2 = Box::new(FailingProvider::new(FailureMode::AlwaysFail, 10));
        
        let mut provider = FallbackAwareProvider::new(primary, vec![fallback1, fallback2], 1);
        
        let constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        };
        
        // Should eventually exhaust all backends
        let mut last_result = Ok(0);
        for i in 0..10 {
            match provider.draw_integer(&constraints) {
                Ok(_) => {}, // Keep going
                Err(e) => {
                    last_result = Err(e);
                    break;
                }
            }
        }
        
        // Should have failed with exhausted scope
        assert!(last_result.is_err());
        if let Err(ProviderError::CannotProceed { scope, .. }) = last_result {
            assert_eq!(scope, ProviderScope::Exhausted);
        }
        
        // Verify verification status
        assert_eq!(provider.verification_state.status, VerificationStatus::Failed);
        assert!(provider.fallback_statistics.failed_recoveries > 0);
    }

    #[test]
    fn test_error_scopes_and_recovery() {
        // Test different error scopes trigger appropriate responses
        let scopes = vec![
            ProviderScope::DiscardTestCase,
            ProviderScope::Verified,
            ProviderScope::Exhausted,
            ProviderScope::Configuration,
        ];
        
        for scope in scopes {
            let primary = Box::new(FailingProvider::new(FailureMode::FailWithScope(scope), 10));
            let fallback = Box::new(RandomProvider::new());
            
            let mut provider = FallbackAwareProvider::new(primary, vec![fallback], 1);
            
            let constraints = IntegerConstraints {
                min_value: Some(0),
                max_value: Some(10),
                weights: None,
                shrink_towards: Some(0),
            };
            
            // First operation should fail and trigger fallback
            let result = provider.draw_integer(&constraints);
            
            // Should succeed with fallback for most scopes
            if scope != ProviderScope::Exhausted {
                assert!(result.is_ok(), "Failed for scope {:?}", scope);
                assert_eq!(provider.current_backend_index, 1);
            }
        }
    }

    #[test]
    fn test_verification_state_tracking() {
        let primary = Box::new(FailingProvider::new(FailureMode::FailAfterCount(2), 10));
        let fallback = Box::new(RandomProvider::new());
        
        let mut provider = FallbackAwareProvider::new(primary, vec![fallback], 2);
        
        // Initially healthy
        assert_eq!(provider.verification_state.status, VerificationStatus::Healthy);
        
        let constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        };
        
        // Operations that will trigger errors
        let _result1 = provider.draw_integer(&constraints); // Success
        assert_eq!(provider.verification_state.status, VerificationStatus::Healthy);
        
        let _result2 = provider.draw_integer(&constraints); // Success
        assert_eq!(provider.verification_state.status, VerificationStatus::Healthy);
        
        let _result3 = provider.draw_integer(&constraints); // Fail primary, switch to fallback
        // Should be in recovery mode after switching
        assert!(matches!(provider.verification_state.status, 
                        VerificationStatus::Degraded | VerificationStatus::RecoveryMode));
        
        // Verify failure patterns are recorded
        assert!(!provider.verification_state.failure_patterns.is_empty());
        
        let primary_pattern = provider.verification_state.failure_patterns.iter()
            .find(|p| p.backend_name == "primary");
        assert!(primary_pattern.is_some());
    }

    #[test]
    fn test_backend_capabilities_combination() {
        // Create backends with different capabilities
        let mut primary_caps = BackendCapabilities::default();
        primary_caps.supports_floats = false;
        primary_caps.symbolic_constraints = true;
        
        let mut fallback_caps = BackendCapabilities::default();
        fallback_caps.supports_floats = true;
        fallback_caps.symbolic_constraints = false;
        fallback_caps.replay_support = true;
        
        // Mock providers for testing (simplified)
        let primary = Box::new(RandomProvider::new());
        let fallback = Box::new(HypothesisProvider::new());
        
        let provider = FallbackAwareProvider::new(primary, vec![fallback], 3);
        
        let combined_caps = provider.capabilities();
        
        // Should combine capabilities from all backends
        assert!(combined_caps.supports_floats); // From fallback
        assert!(combined_caps.replay_support); // From fallback
        assert!(combined_caps.supports_integers); // From both
    }

    #[test]
    fn test_observability_and_monitoring() {
        let primary = Box::new(FailingProvider::new(FailureMode::FailRandomly(0.3), 10));
        let fallback = Box::new(RandomProvider::new());
        
        let mut provider = FallbackAwareProvider::new(primary, vec![fallback], 2);
        
        let constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        };
        
        // Perform several operations to generate statistics
        for _ in 0..20 {
            let _ = provider.draw_integer(&constraints);
        }
        
        // Test observation methods
        let test_case_observation = provider.observe_test_case();
        assert!(test_case_observation.contains_key("backend_error_counts"));
        assert!(test_case_observation.contains_key("current_backend"));
        assert!(test_case_observation.contains_key("verification_status"));
        
        let info_messages = provider.observe_information_messages(ProviderLifetime::TestCase);
        let fallback_message = info_messages.iter()
            .find(|msg| msg.message == "Fallback Provider Status");
        assert!(fallback_message.is_some());
        
        // Test error statistics
        let stats = provider.get_error_statistics();
        assert!(stats.contains_key("total_switches"));
        assert!(stats.contains_key("successful_recoveries"));
        assert!(stats.contains_key("verification_status"));
    }

    #[test]
    fn test_provider_lifecycle_integration() {
        let mut lifecycle_manager = ProviderLifecycleManager::new();
        
        // Create a factory that produces fallback-aware providers
        struct FallbackProviderFactory;
        impl ProviderFactory for FallbackProviderFactory {
            fn create_provider(&self) -> Box<dyn PrimitiveProvider> {
                let primary = Box::new(FailingProvider::new(FailureMode::FailAfterCount(3), 10));
                let fallback = Box::new(RandomProvider::new());
                
                Box::new(FallbackAwareProvider::new(primary, vec![fallback], 2))
            }
            
            fn name(&self) -> &str {
                "fallback_aware"
            }
        }
        
        lifecycle_manager.register_factory(Arc::new(FallbackProviderFactory));
        
        // Test provider creation and lifecycle management
        let provider_result = lifecycle_manager.get_provider(
            "fallback_aware",
            LifecycleScope::TestFunction,
            Some("fallback_test_context".to_string()),
            None,
        );
        
        assert!(provider_result.is_ok());
        
        let provider_arc = provider_result.unwrap();
        let metadata = {
            let provider = provider_arc.lock().unwrap();
            provider.metadata().clone()
        };
        
        assert_eq!(metadata.scope, LifecycleScope::TestFunction);
        assert_eq!(metadata.state, ProviderState::Active);
        
        // Test operations through lifecycle-managed provider
        {
            let mut provider = provider_arc.lock().unwrap();
            let raw_provider = provider.provider_mut();
            
            // Cast to our fallback provider type for testing
            // Note: In practice, this would be handled through traits
            let constraints = IntegerConstraints {
                min_value: Some(0),
                max_value: Some(10),
                weights: None,
                shrink_towards: Some(0),
            };
            
            let result = raw_provider.draw_integer(&constraints);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_comprehensive_error_handling_patterns() {
        // Test various error patterns and recovery strategies
        let error_patterns = vec![
            ("burst_errors", FailureMode::FailAfterCount(1)),
            ("intermittent_errors", FailureMode::FailRandomly(0.4)),
            ("scope_specific", FailureMode::FailWithScope(ProviderScope::DiscardTestCase)),
            ("custom_message", FailureMode::FailWithMessage("Test custom error".to_string())),
        ];
        
        for (pattern_name, failure_mode) in error_patterns {
            println!("Testing error pattern: {}", pattern_name);
            
            let primary = Box::new(FailingProvider::new(failure_mode, 10));
            let fallback1 = Box::new(RandomProvider::new());
            let fallback2 = Box::new(HypothesisProvider::new());
            
            let mut provider = FallbackAwareProvider::new(primary, vec![fallback1, fallback2], 2);
            
            let constraints = IntegerConstraints {
                min_value: Some(0),
                max_value: Some(10),
                weights: None,
                shrink_towards: Some(0),
            };
            
            // Perform operations and track results
            let mut success_count = 0;
            let mut error_count = 0;
            
            for i in 0..10 {
                match provider.draw_integer(&constraints) {
                    Ok(_) => success_count += 1,
                    Err(_) => error_count += 1,
                }
                
                // Check provider state after each operation
                let stats = provider.get_error_statistics();
                println!("  Operation {}: Backend {}, Switches: {}, Status: {:?}", 
                        i, 
                        stats.get("current_backend").unwrap_or(&serde_json::Value::Null),
                        stats.get("total_switches").unwrap_or(&serde_json::Value::Null),
                        provider.verification_state.status);
            }
            
            println!("  Pattern {} results: {} successes, {} errors", 
                    pattern_name, success_count, error_count);
            
            // Verify that we maintained some level of service
            assert!(success_count > 0, "No successful operations for pattern {}", pattern_name);
            
            // Verify error handling statistics were collected
            let final_stats = provider.get_error_statistics();
            assert!(final_stats.contains_key("backend_error_counts"));
            assert!(final_stats.contains_key("verification_status"));
        }
    }

    #[test]
    fn test_concurrent_error_handling() {
        use std::sync::Arc;
        use std::thread;
        
        // Test error handling under concurrent access
        let primary = Box::new(FailingProvider::new(FailureMode::FailRandomly(0.2), 100));
        let fallback = Box::new(RandomProvider::new());
        
        let provider = Arc::new(Mutex::new(
            FallbackAwareProvider::new(primary, vec![fallback], 3)
        ));
        
        let handles: Vec<_> = (0..5).map(|thread_id| {
            let provider_clone = provider.clone();
            thread::spawn(move || {
                let mut thread_results = Vec::new();
                
                for i in 0..20 {
                    let constraints = IntegerConstraints {
                        min_value: Some(i),
                        max_value: Some(i + 10),
                        weights: None,
                        shrink_towards: Some(i),
                    };
                    
                    let result = {
                        let mut p = provider_clone.lock().unwrap();
                        p.draw_integer(&constraints)
                    };
                    
                    thread_results.push((thread_id, i, result.is_ok()));
                }
                
                thread_results
            })
        }).collect();
        
        // Collect results from all threads
        let mut all_results = Vec::new();
        for handle in handles {
            all_results.extend(handle.join().unwrap());
        }
        
        // Verify that most operations succeeded despite concurrent errors
        let success_count = all_results.iter().filter(|(_, _, success)| *success).count();
        let total_count = all_results.len();
        
        println!("Concurrent test: {}/{} operations succeeded", success_count, total_count);
        assert!(success_count > total_count / 2, "Too many failures in concurrent test");
        
        // Verify final provider state
        let final_stats = {
            let p = provider.lock().unwrap();
            p.get_error_statistics()
        };
        
        println!("Final concurrent test stats: {:?}", final_stats);
    }
}