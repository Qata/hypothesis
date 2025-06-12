//! Advanced Error Handling and Fallback Capability for Provider System
//!
//! This module implements a comprehensive error handling and automatic fallback system
//! for provider backends, mirroring Python's BackendCannotProceed exception handling
//! with sophisticated Rust error management patterns.
//!
//! **Core Capabilities:**
//! - `BackendCannotProceed` equivalent error handling with scoped recovery
//! - Automatic fallback to hypothesis provider when backends fail repeatedly
//! - Provider verification tracking with failure threshold monitoring
//! - Robust switching logic for different failure modes
//! - Context-aware error recovery based on test execution phase
//! - Graceful degradation with detailed error reporting

use crate::choice::{ChoiceValue, Constraints, ChoiceType, FloatConstraints, IntegerConstraints, BooleanConstraints, IntervalSet};
use crate::providers::{
    PrimitiveProvider, ProviderError, ProviderScope, ProviderLifetime, ProviderRegistry,
    BackendCapabilities, TestCaseContext, ObservationMessage, TestCaseObservation,
    create_global_provider, get_provider_registry,
};
use crate::data::DrawError;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, Duration};
use serde_json;

/// Advanced error handling scope for provider failures
/// 
/// Maps directly to Python's CannotProceedScopeT with identical semantics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CannotProceedScope {
    /// Backend has verified the property - switch to hypothesis provider
    Verified,
    /// Backend has exhausted search space - switch to hypothesis provider
    Exhausted,
    /// Single test case failed - count failures and potentially switch
    DiscardTestCase,
    /// Generic failure - treat as invalid test case
    Other,
}

impl From<CannotProceedScope> for ProviderScope {
    fn from(scope: CannotProceedScope) -> Self {
        match scope {
            CannotProceedScope::Verified => ProviderScope::Verified,
            CannotProceedScope::Exhausted => ProviderScope::Exhausted,
            CannotProceedScope::DiscardTestCase => ProviderScope::DiscardTestCase,
            CannotProceedScope::Other => ProviderScope::Configuration,
        }
    }
}

impl From<ProviderScope> for CannotProceedScope {
    fn from(scope: ProviderScope) -> Self {
        match scope {
            ProviderScope::Verified => CannotProceedScope::Verified,
            ProviderScope::Exhausted => CannotProceedScope::Exhausted,
            ProviderScope::DiscardTestCase => CannotProceedScope::DiscardTestCase,
            ProviderScope::Configuration => CannotProceedScope::Other,
        }
    }
}

/// Enhanced provider error with automatic fallback capabilities
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackendCannotProceedError {
    /// Scope of the failure for intelligent recovery
    pub scope: CannotProceedScope,
    /// Detailed reason for the failure
    pub reason: String,
    /// Backend that failed
    pub backend_name: String,
    /// Timestamp of the failure
    pub timestamp: SystemTime,
    /// Additional context data
    pub context: HashMap<String, serde_json::Value>,
}

impl BackendCannotProceedError {
    pub fn new(scope: CannotProceedScope, reason: String, backend_name: String) -> Self {
        Self {
            scope,
            reason,
            backend_name,
            timestamp: SystemTime::now(),
            context: HashMap::new(),
        }
    }
    
    pub fn with_context(mut self, key: String, value: serde_json::Value) -> Self {
        self.context.insert(key, value);
        self
    }
}

impl std::fmt::Display for BackendCannotProceedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Backend '{}' cannot proceed (scope: {:?}): {}", 
               self.backend_name, self.scope, self.reason)
    }
}

impl std::error::Error for BackendCannotProceedError {}

/// Provider verification and failure tracking system
/// 
/// Implements sophisticated failure tracking with automatic threshold-based
/// fallback logic mirroring Python's backend switching behavior.
#[derive(Debug)]
pub struct ProviderVerificationTracker {
    /// Total number of provider calls made
    call_count: u64,
    /// Number of failed realization attempts
    failed_realize_count: u64,
    /// Whether to switch to hypothesis provider
    switch_to_hypothesis_provider: bool,
    /// Backend that provided verification (if any)
    verified_by: Option<String>,
    /// Current backend being used
    current_backend: String,
    /// Failure history for detailed analysis
    failure_history: Vec<BackendCannotProceedError>,
    /// Performance metrics per backend
    backend_metrics: HashMap<String, BackendMetrics>,
    /// Configuration for fallback thresholds
    fallback_config: FallbackConfiguration,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BackendMetrics {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub average_duration: Duration,
    pub last_success: Option<SystemTime>,
    pub last_failure: Option<SystemTime>,
    pub consecutive_failures: u64,
}

#[derive(Debug, Clone)]
pub struct FallbackConfiguration {
    /// Minimum failures before considering fallback
    pub min_failures_threshold: u64,
    /// Maximum failure rate before forcing fallback (0.0 to 1.0)
    pub max_failure_rate: f64,
    /// Maximum consecutive failures before fallback
    pub max_consecutive_failures: u64,
    /// Time window for failure rate calculation
    pub failure_rate_window: Duration,
}

impl Default for FallbackConfiguration {
    fn default() -> Self {
        Self {
            min_failures_threshold: 10,
            max_failure_rate: 0.2, // 20% as per Python implementation
            max_consecutive_failures: 5,
            failure_rate_window: Duration::from_secs(60),
        }
    }
}

impl ProviderVerificationTracker {
    pub fn new(initial_backend: String) -> Self {
        Self {
            call_count: 0,
            failed_realize_count: 0,
            switch_to_hypothesis_provider: false,
            verified_by: None,
            current_backend: initial_backend,
            failure_history: Vec::new(),
            backend_metrics: HashMap::new(),
            fallback_config: FallbackConfiguration::default(),
        }
    }
    
    /// Record a provider call attempt
    pub fn record_call(&mut self, backend: &str) {
        self.call_count += 1;
        
        let metrics = self.backend_metrics.entry(backend.to_string()).or_insert_with(|| {
            BackendMetrics {
                total_calls: 0,
                successful_calls: 0,
                failed_calls: 0,
                average_duration: Duration::from_millis(0),
                last_success: None,
                last_failure: None,
                consecutive_failures: 0,
            }
        });
        
        metrics.total_calls += 1;
        
        println!("VERIFICATION_TRACKER DEBUG: Recorded call {} for backend '{}'", 
                metrics.total_calls, backend);
    }
    
    /// Record a successful provider operation
    pub fn record_success(&mut self, backend: &str, duration: Duration) {
        if let Some(metrics) = self.backend_metrics.get_mut(backend) {
            metrics.successful_calls += 1;
            metrics.last_success = Some(SystemTime::now());
            metrics.consecutive_failures = 0;
            
            // Update average duration with exponential moving average
            let alpha = 0.1; // Smoothing factor
            let new_duration_ms = duration.as_millis() as f64;
            let current_avg_ms = metrics.average_duration.as_millis() as f64;
            let updated_avg_ms = alpha * new_duration_ms + (1.0 - alpha) * current_avg_ms;
            metrics.average_duration = Duration::from_millis(updated_avg_ms as u64);
            
            println!("VERIFICATION_TRACKER DEBUG: Success for backend '{}' in {:?}", backend, duration);
        }
    }
    
    /// Record a provider failure and determine if fallback is needed
    pub fn record_failure(&mut self, error: BackendCannotProceedError) -> FallbackDecision {
        let backend = &error.backend_name;
        self.failed_realize_count += 1;
        
        // Update backend metrics
        if let Some(metrics) = self.backend_metrics.get_mut(backend) {
            metrics.failed_calls += 1;
            metrics.last_failure = Some(error.timestamp);
            metrics.consecutive_failures += 1;
        }
        
        // Store failure for analysis
        self.failure_history.push(error.clone());
        
        // Determine fallback decision based on scope
        let decision = match error.scope {
            CannotProceedScope::Verified => {
                println!("VERIFICATION_TRACKER DEBUG: Backend '{}' claims verification complete", backend);
                self.switch_to_hypothesis_provider = true;
                self.verified_by = Some(backend.clone());
                FallbackDecision::SwitchToHypothesis {
                    reason: FallbackReason::BackendVerified,
                    preserve_verification: true,
                }
            },
            CannotProceedScope::Exhausted => {
                println!("VERIFICATION_TRACKER DEBUG: Backend '{}' exhausted search space", backend);
                self.switch_to_hypothesis_provider = true;
                FallbackDecision::SwitchToHypothesis {
                    reason: FallbackReason::SearchSpaceExhausted,
                    preserve_verification: false,
                }
            },
            CannotProceedScope::DiscardTestCase => {
                println!("VERIFICATION_TRACKER DEBUG: Test case discard for backend '{}'", backend);
                
                // Check if we should fallback based on thresholds
                if self.should_fallback_to_hypothesis() {
                    println!("VERIFICATION_TRACKER DEBUG: Fallback threshold exceeded, switching to hypothesis");
                    self.switch_to_hypothesis_provider = true;
                    FallbackDecision::SwitchToHypothesis {
                        reason: FallbackReason::FailureThresholdExceeded,
                        preserve_verification: false,
                    }
                } else {
                    FallbackDecision::DiscardTestCase {
                        retry_count: self.failed_realize_count,
                    }
                }
            },
            CannotProceedScope::Other => {
                println!("VERIFICATION_TRACKER DEBUG: Generic failure for backend '{}'", backend);
                FallbackDecision::TreatAsInvalid {
                    should_retry: true,
                }
            },
        };
        
        println!("VERIFICATION_TRACKER DEBUG: Failure decision: {:?}", decision);
        decision
    }
    
    /// Check if we should fallback to hypothesis provider based on thresholds
    fn should_fallback_to_hypothesis(&self) -> bool {
        let config = &self.fallback_config;
        
        // Must have minimum number of failures
        if self.failed_realize_count < config.min_failures_threshold {
            return false;
        }
        
        // Check failure rate threshold (matches Python's 20% threshold)
        let failure_rate = self.failed_realize_count as f64 / self.call_count as f64;
        if failure_rate > config.max_failure_rate {
            println!("VERIFICATION_TRACKER DEBUG: Failure rate {:.2}% exceeds threshold {:.2}%", 
                    failure_rate * 100.0, config.max_failure_rate * 100.0);
            return true;
        }
        
        // Check consecutive failures for current backend
        if let Some(metrics) = self.backend_metrics.get(&self.current_backend) {
            if metrics.consecutive_failures >= config.max_consecutive_failures {
                println!("VERIFICATION_TRACKER DEBUG: Consecutive failures {} exceeds threshold {}", 
                        metrics.consecutive_failures, config.max_consecutive_failures);
                return true;
            }
        }
        
        false
    }
    
    /// Get current backend selection
    pub fn should_use_hypothesis_backend(&self) -> bool {
        self.switch_to_hypothesis_provider
    }
    
    /// Force switch to hypothesis backend (for testing phases)
    pub fn force_hypothesis_backend(&mut self, reason: &str) {
        println!("VERIFICATION_TRACKER DEBUG: Forcing hypothesis backend: {}", reason);
        self.switch_to_hypothesis_provider = true;
    }
    
    /// Allow backend selection again (for generation phases)
    pub fn allow_backend_selection(&mut self) {
        println!("VERIFICATION_TRACKER DEBUG: Allowing backend selection");
        self.switch_to_hypothesis_provider = false;
    }
    
    /// Get verification status
    pub fn get_verification_status(&self) -> VerificationStatus {
        VerificationStatus {
            verified_by: self.verified_by.clone(),
            call_count: self.call_count,
            failed_realize_count: self.failed_realize_count,
            failure_rate: if self.call_count > 0 { 
                self.failed_realize_count as f64 / self.call_count as f64 
            } else { 
                0.0 
            },
            using_hypothesis_backend: self.switch_to_hypothesis_provider,
            backend_metrics: self.backend_metrics.clone(),
            recent_failures: self.failure_history.iter()
                .rev()
                .take(10)
                .cloned()
                .collect(),
        }
    }
    
    /// Reset tracking state for new test run
    pub fn reset(&mut self) {
        self.call_count = 0;
        self.failed_realize_count = 0;
        self.switch_to_hypothesis_provider = false;
        self.verified_by = None;
        self.failure_history.clear();
        self.backend_metrics.clear();
        println!("VERIFICATION_TRACKER DEBUG: Reset tracking state");
    }
}

/// Decision made after recording a provider failure
#[derive(Debug, Clone, PartialEq)]
pub enum FallbackDecision {
    /// Switch to hypothesis provider
    SwitchToHypothesis {
        reason: FallbackReason,
        preserve_verification: bool,
    },
    /// Discard current test case and continue
    DiscardTestCase {
        retry_count: u64,
    },
    /// Treat as invalid test case
    TreatAsInvalid {
        should_retry: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum FallbackReason {
    BackendVerified,
    SearchSpaceExhausted,
    FailureThresholdExceeded,
    ConsecutiveFailuresExceeded,
    ConfigurationError,
}

/// Current verification and performance status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerificationStatus {
    pub verified_by: Option<String>,
    pub call_count: u64,
    pub failed_realize_count: u64,
    pub failure_rate: f64,
    pub using_hypothesis_backend: bool,
    pub backend_metrics: HashMap<String, BackendMetrics>,
    pub recent_failures: Vec<BackendCannotProceedError>,
}

/// Enhanced provider with automatic fallback capability
/// 
/// Wraps any provider with sophisticated error handling and automatic
/// fallback logic, providing resilient test generation.
#[derive(Debug)]
pub struct FallbackAwareProvider {
    /// Primary provider being used
    primary_provider: Box<dyn PrimitiveProvider>,
    /// Fallback provider (typically hypothesis)
    fallback_provider: Box<dyn PrimitiveProvider>,
    /// Verification and failure tracking
    tracker: Arc<RwLock<ProviderVerificationTracker>>,
    /// Provider name for debugging
    provider_name: String,
    /// Current execution phase
    execution_phase: TestExecutionPhase,
    /// Error recovery configuration
    recovery_config: ErrorRecoveryConfiguration,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TestExecutionPhase {
    /// Database reuse phase - always use hypothesis
    Reuse,
    /// Generation phase - use requested backend
    Generate,
    /// Shrinking phase - always use hypothesis
    Shrink,
    /// Other phase - use current selection
    Other,
}

#[derive(Debug, Clone)]
pub struct ErrorRecoveryConfiguration {
    /// Maximum retries for transient failures
    pub max_retries: u32,
    /// Whether to preserve symbolic values on fallback
    pub preserve_symbolic_values: bool,
    /// Whether to log detailed failure information
    pub detailed_logging: bool,
    /// Timeout for provider operations
    pub operation_timeout: Duration,
}

impl Default for ErrorRecoveryConfiguration {
    fn default() -> Self {
        Self {
            max_retries: 3,
            preserve_symbolic_values: false,
            detailed_logging: true,
            operation_timeout: Duration::from_secs(30),
        }
    }
}

impl FallbackAwareProvider {
    pub fn new(
        primary_provider: Box<dyn PrimitiveProvider>,
        fallback_provider: Box<dyn PrimitiveProvider>,
        provider_name: String,
    ) -> Self {
        let tracker = Arc::new(RwLock::new(
            ProviderVerificationTracker::new(provider_name.clone())
        ));
        
        Self {
            primary_provider,
            fallback_provider,
            tracker,
            provider_name,
            execution_phase: TestExecutionPhase::Other,
            recovery_config: ErrorRecoveryConfiguration::default(),
        }
    }
    
    /// Set the current execution phase
    pub fn set_execution_phase(&mut self, phase: TestExecutionPhase) {
        self.execution_phase = phase;
        
        // Force hypothesis backend for certain phases
        let mut tracker = self.tracker.write().unwrap();
        match phase {
            TestExecutionPhase::Reuse | TestExecutionPhase::Shrink => {
                tracker.force_hypothesis_backend(&format!("{:?} phase", phase));
            },
            TestExecutionPhase::Generate => {
                tracker.allow_backend_selection();
            },
            TestExecutionPhase::Other => {
                // Keep current selection
            },
        }
        
        println!("FALLBACK_PROVIDER DEBUG: Set execution phase to {:?}", phase);
    }
    
    /// Get current verification status
    pub fn get_verification_status(&self) -> VerificationStatus {
        let tracker = self.tracker.read().unwrap();
        tracker.get_verification_status()
    }
    
    /// Execute operation with automatic error handling and fallback
    fn execute_with_fallback<T, F>(&mut self, operation_name: &str, mut operation: F) -> Result<T, ProviderError>
    where
        F: FnMut(&mut dyn PrimitiveProvider) -> Result<T, ProviderError>,
    {
        let start_time = SystemTime::now();
        
        // Record the call attempt
        {
            let mut tracker = self.tracker.write().unwrap();
            tracker.record_call(&self.provider_name);
        }
        
        // Try primary provider first (if selected)
        let result = {
            let should_use_hypothesis = {
                let tracker = self.tracker.read().unwrap();
                tracker.should_use_hypothesis_backend()
            };
            
            let provider_to_use = if should_use_hypothesis || 
                                     matches!(self.execution_phase, TestExecutionPhase::Reuse | TestExecutionPhase::Shrink) {
                "hypothesis"
            } else {
                &self.provider_name
            };
            
            println!("FALLBACK_PROVIDER DEBUG: Using provider '{}' for operation '{}'", 
                    provider_to_use, operation_name);
            
            if provider_to_use == "hypothesis" {
                operation(&mut *self.fallback_provider)
            } else {
                operation(&mut *self.primary_provider)
            }
        };
        
        let duration = start_time.elapsed().unwrap_or_default();
        
        match result {
            Ok(value) => {
                // Record success
                let mut tracker = self.tracker.write().unwrap();
                tracker.record_success(&self.provider_name, duration);
                Ok(value)
            },
            Err(ProviderError::CannotProceed { scope, reason }) => {
                println!("FALLBACK_PROVIDER DEBUG: Provider '{}' cannot proceed in operation '{}': {}", 
                        self.provider_name, operation_name, reason);
                
                // Create error with context
                let error = BackendCannotProceedError::new(
                    scope.into(),
                    reason.clone(),
                    self.provider_name.clone(),
                ).with_context("operation".to_string(), serde_json::Value::String(operation_name.to_string()))
                 .with_context("duration_ms".to_string(), serde_json::Value::Number(
                     serde_json::Number::from(duration.as_millis() as u64)
                 ));
                
                // Record failure and get decision
                let decision = {
                    let mut tracker = self.tracker.write().unwrap();
                    tracker.record_failure(error.clone())
                };
                
                // Handle the decision
                match decision {
                    FallbackDecision::SwitchToHypothesis { reason: fallback_reason, preserve_verification } => {
                        println!("FALLBACK_PROVIDER DEBUG: Switching to hypothesis provider due to {:?}", fallback_reason);
                        
                        if preserve_verification && matches!(error.scope, CannotProceedScope::Verified) {
                            // For verified backends, we might want to note this for later reporting
                            println!("FALLBACK_PROVIDER DEBUG: Backend verification preserved");
                        }
                        
                        // Retry with fallback provider
                        operation(&mut *self.fallback_provider)
                    },
                    FallbackDecision::DiscardTestCase { retry_count } => {
                        println!("FALLBACK_PROVIDER DEBUG: Discarding test case (retry {})", retry_count);
                        Err(ProviderError::CannotProceed { scope, reason: format!("Test case discarded: {}", reason) })
                    },
                    FallbackDecision::TreatAsInvalid { should_retry } => {
                        if should_retry {
                            println!("FALLBACK_PROVIDER DEBUG: Treating as invalid, retrying with fallback");
                            operation(&mut *self.fallback_provider)
                        } else {
                            Err(ProviderError::InvalidChoice(format!("Provider failure: {}", reason)))
                        }
                    },
                }
            },
            Err(other_error) => {
                println!("FALLBACK_PROVIDER DEBUG: Non-recoverable error in operation '{}': {:?}", 
                        operation_name, other_error);
                Err(other_error)
            },
        }
    }
}

impl PrimitiveProvider for FallbackAwareProvider {
    fn lifetime(&self) -> ProviderLifetime {
        // Use the primary provider's lifetime
        self.primary_provider.lifetime()
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        // Combine capabilities from both providers
        let primary_caps = self.primary_provider.capabilities();
        let fallback_caps = self.fallback_provider.capabilities();
        
        BackendCapabilities {
            supports_integers: primary_caps.supports_integers || fallback_caps.supports_integers,
            supports_floats: primary_caps.supports_floats || fallback_caps.supports_floats,
            supports_strings: primary_caps.supports_strings || fallback_caps.supports_strings,
            supports_bytes: primary_caps.supports_bytes || fallback_caps.supports_bytes,
            supports_choices: primary_caps.supports_choices || fallback_caps.supports_choices,
            avoid_realization: primary_caps.avoid_realization, // Primary provider preference
            add_observability_callback: primary_caps.add_observability_callback || fallback_caps.add_observability_callback,
            structural_awareness: primary_caps.structural_awareness || fallback_caps.structural_awareness,
            replay_support: primary_caps.replay_support || fallback_caps.replay_support,
            symbolic_constraints: primary_caps.symbolic_constraints || fallback_caps.symbolic_constraints,
        }
    }
    
    fn draw_boolean(&mut self, p: f64) -> Result<bool, ProviderError> {
        self.execute_with_fallback("draw_boolean", |provider| provider.draw_boolean(p))
    }
    
    fn draw_integer(&mut self, constraints: &IntegerConstraints) -> Result<i128, ProviderError> {
        self.execute_with_fallback("draw_integer", |provider| provider.draw_integer(constraints))
    }
    
    fn draw_float(&mut self, constraints: &FloatConstraints) -> Result<f64, ProviderError> {
        self.execute_with_fallback("draw_float", |provider| provider.draw_float(constraints))
    }
    
    fn draw_string(&mut self, intervals: &IntervalSet, min_size: usize, max_size: usize) -> Result<String, ProviderError> {
        self.execute_with_fallback("draw_string", |provider| provider.draw_string(intervals, min_size, max_size))
    }
    
    fn draw_bytes(&mut self, min_size: usize, max_size: usize) -> Result<Vec<u8>, ProviderError> {
        self.execute_with_fallback("draw_bytes", |provider| provider.draw_bytes(min_size, max_size))
    }
    
    fn observe_test_case(&mut self) -> HashMap<String, serde_json::Value> {
        let mut observation = HashMap::new();
        
        // Add verification status
        let status = self.get_verification_status();
        observation.insert("verification_status".to_string(), 
                         serde_json::to_value(&status).unwrap_or_default());
        
        // Add provider-specific observations from the active provider
        let status = self.get_verification_status();
        let provider_obs = if status.using_hypothesis_backend {
            self.fallback_provider.observe_test_case()
        } else {
            self.primary_provider.observe_test_case()
        };
        observation.extend(provider_obs);
        
        observation
    }
    
    fn observe_information_messages(&mut self, lifetime: ProviderLifetime) -> Vec<ObservationMessage> {
        let mut messages = Vec::new();
        
        // Add verification tracking information
        let status = self.get_verification_status();
        messages.push(ObservationMessage {
            level: "info".to_string(),
            message: "Provider Verification Status".to_string(),
            data: Some(serde_json::json!({
                "verified_by": status.verified_by,
                "call_count": status.call_count,
                "failure_rate": format!("{:.2}%", status.failure_rate * 100.0),
                "using_hypothesis_backend": status.using_hypothesis_backend,
                "execution_phase": format!("{:?}", self.execution_phase)
            })),
        });
        
        // Add messages from active provider
        let mut provider_messages = if status.using_hypothesis_backend {
            self.fallback_provider.observe_information_messages(lifetime)
        } else {
            self.primary_provider.observe_information_messages(lifetime)
        };
        
        messages.append(&mut provider_messages);
        messages
    }
    
    fn span_start(&mut self, label: u32) {
        // Forward to both providers for comprehensive tracking
        self.primary_provider.span_start(label);
        self.fallback_provider.span_start(label);
    }
    
    fn span_end(&mut self, discard: bool) {
        // Forward to both providers for comprehensive tracking
        self.primary_provider.span_end(discard);
        self.fallback_provider.span_end(discard);
    }
    
    fn per_test_case_context(&mut self) -> Box<dyn TestCaseContext> {
        // Use active provider's context
        let status = self.get_verification_status();
        if status.using_hypothesis_backend {
            self.fallback_provider.per_test_case_context()
        } else {
            self.primary_provider.per_test_case_context()
        }
    }
    
    fn can_realize(&self) -> bool {
        // Can realize if either provider can
        self.primary_provider.can_realize() || self.fallback_provider.can_realize()
    }
    
    fn replay_choices(&mut self, choices: &[ChoiceValue]) -> Result<(), ProviderError> {
        // Try with active provider first
        let status = self.get_verification_status();
        if status.using_hypothesis_backend {
            self.fallback_provider.replay_choices(choices)
        } else {
            match self.primary_provider.replay_choices(choices) {
                Ok(()) => Ok(()),
                Err(_) => self.fallback_provider.replay_choices(choices), // Fallback
            }
        }
    }
}

/// Enhanced provider registry with automatic fallback support
pub struct FallbackAwareProviderRegistry {
    /// Base registry for provider management
    base_registry: ProviderRegistry,
    /// Global verification tracker
    global_tracker: Arc<RwLock<ProviderVerificationTracker>>,
    /// Default fallback provider name
    fallback_provider_name: String,
}

impl FallbackAwareProviderRegistry {
    pub fn new() -> Self {
        Self {
            base_registry: ProviderRegistry::new(),
            global_tracker: Arc::new(RwLock::new(
                ProviderVerificationTracker::new("unknown".to_string())
            )),
            fallback_provider_name: "hypothesis".to_string(),
        }
    }
    
    /// Create a fallback-aware provider
    pub fn create_fallback_aware_provider(&mut self, provider_name: &str) -> Result<FallbackAwareProvider, ProviderError> {
        // Create primary provider
        let primary_provider = self.base_registry.create(provider_name)?;
        
        // Create fallback provider
        let fallback_provider = self.base_registry.create(&self.fallback_provider_name)?;
        
        // Create fallback-aware wrapper
        let fallback_aware = FallbackAwareProvider::new(
            primary_provider,
            fallback_provider,
            provider_name.to_string(),
        );
        
        println!("FALLBACK_REGISTRY DEBUG: Created fallback-aware provider for '{}'", provider_name);
        Ok(fallback_aware)
    }
    
    /// Get global verification status
    pub fn get_global_verification_status(&self) -> VerificationStatus {
        let tracker = self.global_tracker.read().unwrap();
        tracker.get_verification_status()
    }
    
    /// Reset all verification tracking
    pub fn reset_verification_tracking(&mut self) {
        let mut tracker = self.global_tracker.write().unwrap();
        tracker.reset();
    }
}

/// Test provider that can simulate various failure modes
#[derive(Debug)]
pub struct FailingTestProvider {
    failure_mode: FailureMode,
    call_count: u32,
    failure_threshold: u32,
}

#[derive(Debug, Clone)]
pub enum FailureMode {
    /// Never fail
    NeverFail,
    /// Always fail with given scope
    AlwaysFail(CannotProceedScope),
    /// Fail after N calls
    FailAfterCalls(u32),
    /// Fail intermittently (every N calls)
    FailIntermittently(u32),
    /// Fail with escalating scopes
    EscalatingFailure,
}

impl FailingTestProvider {
    pub fn new(failure_mode: FailureMode) -> Self {
        Self {
            failure_mode,
            call_count: 0,
            failure_threshold: 5,
        }
    }
    
    fn should_fail(&mut self) -> Option<CannotProceedScope> {
        self.call_count += 1;
        
        match &self.failure_mode {
            FailureMode::NeverFail => None,
            FailureMode::AlwaysFail(scope) => Some(*scope),
            FailureMode::FailAfterCalls(threshold) => {
                if self.call_count > *threshold {
                    Some(CannotProceedScope::Exhausted)
                } else {
                    None
                }
            },
            FailureMode::FailIntermittently(interval) => {
                if self.call_count % interval == 0 {
                    Some(CannotProceedScope::DiscardTestCase)
                } else {
                    None
                }
            },
            FailureMode::EscalatingFailure => {
                match self.call_count {
                    1..=3 => None,
                    4..=6 => Some(CannotProceedScope::DiscardTestCase),
                    7..=9 => Some(CannotProceedScope::Other),
                    _ => Some(CannotProceedScope::Exhausted),
                }
            },
        }
    }
}

impl PrimitiveProvider for FailingTestProvider {
    fn lifetime(&self) -> ProviderLifetime {
        ProviderLifetime::TestCase
    }
    
    fn draw_boolean(&mut self, _p: f64) -> Result<bool, ProviderError> {
        if let Some(scope) = self.should_fail() {
            return Err(ProviderError::CannotProceed {
                scope: scope.into(),
                reason: format!("Simulated failure in draw_boolean (call {})", self.call_count),
            });
        }
        Ok(true)
    }
    
    fn draw_integer(&mut self, _constraints: &IntegerConstraints) -> Result<i128, ProviderError> {
        if let Some(scope) = self.should_fail() {
            return Err(ProviderError::CannotProceed {
                scope: scope.into(),
                reason: format!("Simulated failure in draw_integer (call {})", self.call_count),
            });
        }
        Ok(42)
    }
    
    fn draw_float(&mut self, _constraints: &FloatConstraints) -> Result<f64, ProviderError> {
        if let Some(scope) = self.should_fail() {
            return Err(ProviderError::CannotProceed {
                scope: scope.into(),
                reason: format!("Simulated failure in draw_float (call {})", self.call_count),
            });
        }
        Ok(3.14)
    }
    
    fn draw_string(&mut self, _intervals: &IntervalSet, _min_size: usize, _max_size: usize) -> Result<String, ProviderError> {
        if let Some(scope) = self.should_fail() {
            return Err(ProviderError::CannotProceed {
                scope: scope.into(),
                reason: format!("Simulated failure in draw_string (call {})", self.call_count),
            });
        }
        Ok("test".to_string())
    }
    
    fn draw_bytes(&mut self, _min_size: usize, _max_size: usize) -> Result<Vec<u8>, ProviderError> {
        if let Some(scope) = self.should_fail() {
            return Err(ProviderError::CannotProceed {
                scope: scope.into(),
                reason: format!("Simulated failure in draw_bytes (call {})", self.call_count),
            });
        }
        Ok(vec![1, 2, 3, 4])
    }
}