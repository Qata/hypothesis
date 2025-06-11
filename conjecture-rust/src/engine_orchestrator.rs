// This file is part of the Hypothesis Conjecture Rust implementation.
//
// Copyright (C) 2025 Hypothesis Contributors
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

//! Test execution orchestration module for coordinating the overall test execution lifecycle.
//!
//! This module provides the main `EngineOrchestrator` that coordinates between different phases
//! of test execution including initialization, generation, reuse, shrinking, and cleanup.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::choice::{ChoiceNode, ChoiceType};
use crate::data::{ConjectureData, ConjectureResult, Status, DataObserver};
use crate::providers::{PrimitiveProvider, ProviderRegistry, get_provider_registry};

/// Maximum number of examples to generate before stopping
const DEFAULT_MAX_EXAMPLES: usize = 100;

/// Maximum number of shrinks before giving up
const MAX_SHRINKS: usize = 500;

/// Maximum shrinking time in seconds
const MAX_SHRINKING_SECONDS: u64 = 300;

/// Minimum test calls to make
const MIN_TEST_CALLS: usize = 10;

/// Buffer size for choice sequence storage
const BUFFER_SIZE: usize = 8 * 1024;

/// Execution phases for the test runner
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionPhase {
    /// Initialization phase
    Initialize,
    /// Reusing existing examples from database
    Reuse,
    /// Generating new examples
    Generate,
    /// Shrinking interesting examples
    Shrink,
    /// Cleanup and finalization
    Cleanup,
}

/// Reasons for stopping execution
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitReason {
    /// Reached maximum examples
    MaxExamples,
    /// Reached maximum iterations with too few valid examples
    MaxIterations,
    /// Reached maximum shrinks
    MaxShrinks,
    /// No more work to do
    Finished,
    /// Test was flaky
    Flaky,
    /// Shrinking was very slow
    VerySlowShrinking,
}

impl ExitReason {
    /// Get a human-readable description of the exit reason
    pub fn description(&self, max_examples: usize) -> String {
        match self {
            ExitReason::MaxExamples => format!("max_examples={}", max_examples),
            ExitReason::MaxIterations => format!(
                "max_examples={}, but < 10% of examples satisfied assumptions", 
                max_examples
            ),
            ExitReason::MaxShrinks => format!("shrunk example {} times", MAX_SHRINKS),
            ExitReason::Finished => "nothing left to do".to_string(),
            ExitReason::Flaky => "test was flaky".to_string(),
            ExitReason::VerySlowShrinking => "shrinking was very slow".to_string(),
        }
    }
}

/// Statistics for a single execution phase
#[derive(Debug, Clone, Default)]
pub struct PhaseStatistics {
    /// Duration of the phase in seconds
    pub duration_seconds: f64,
    /// Number of test cases executed
    pub test_cases: usize,
    /// Number of distinct failures found
    pub distinct_failures: usize,
    /// Number of successful shrinks
    pub shrinks_successful: usize,
}

/// Overall execution statistics
#[derive(Debug, Clone, Default)]
pub struct ExecutionStatistics {
    /// Statistics for each phase
    pub phases: HashMap<ExecutionPhase, PhaseStatistics>,
    /// Reason execution stopped
    pub stopped_because: Option<String>,
    /// Target observations
    pub targets: HashMap<String, f64>,
    /// Node identifier for tracking
    pub node_id: Option<String>,
}

/// Health check state for monitoring test execution
#[derive(Debug, Clone, Default)]
pub struct HealthCheckState {
    /// Number of valid examples
    pub valid_examples: usize,
    /// Number of invalid examples
    pub invalid_examples: usize,
    /// Number of overrun examples
    pub overrun_examples: usize,
    /// Draw times for performance monitoring
    pub draw_times: HashMap<String, Vec<f64>>,
}

impl HealthCheckState {
    /// Get total draw time across all operations
    pub fn total_draw_time(&self) -> f64 {
        self.draw_times.values()
            .flat_map(|times| times.iter())
            .sum()
    }

    /// Generate a timing report for slow operations
    pub fn timing_report(&self) -> String {
        if self.draw_times.is_empty() {
            return String::new();
        }

        let mut report = Vec::new();
        report.push(format!("\n  {:<20}   count | fraction |    slowest draws (seconds)", ""));

        let mut operations: Vec<_> = self.draw_times.iter().collect();
        operations.sort_by(|a, b| {
            let sum_a: f64 = a.1.iter().sum();
            let sum_b: f64 = b.1.iter().sum();
            sum_b.partial_cmp(&sum_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_time = self.total_draw_time();
        
        for (i, (op_name, times)) in operations.iter().enumerate() {
            if i >= 5 && times.iter().sum::<f64>() * 20.0 < total_time {
                report.push(format!("  (skipped {} rows of fast draws)", operations.len() - i));
                break;
            }

            let mut sorted_times: Vec<f64> = times.to_vec();
            sorted_times.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            
            let slow_times: Vec<String> = sorted_times.iter()
                .take(5)
                .filter(|&&t| t > 0.0005)
                .map(|t| format!("{:>6.3},", t))
                .collect();
            
            let desc = if slow_times.is_empty() {
                "    -- ".repeat(5).trim_end_matches(',').to_string()
            } else {
                let mut padded = vec!["    -- ".to_string(); 5];
                for (i, time_str) in slow_times.iter().enumerate() {
                    if i < 5 {
                        padded[4 - i] = time_str.clone();
                    }
                }
                padded.join(" ").trim_end_matches(',').to_string()
            };

            let name = op_name.strip_prefix("generate:").unwrap_or(op_name)
                .strip_suffix(": ").unwrap_or(op_name);
            
            report.push(format!(
                "  {:^20} | {:>4}  | {:>7.0}%  |  {}",
                name,
                times.len(),
                times.iter().sum::<f64>() / total_time,
                desc
            ));
        }

        report.join("\n")
    }
}

/// Errors that can occur during test execution orchestration
#[derive(Debug, Clone)]
pub enum OrchestrationError {
    Overrun,
    Invalid { reason: String },
    Provider { message: String },
    BackendCannotProceed { scope: String },
    Interrupted,
    LimitsExceeded { limit_type: String },
    FlakyTest { details: String },
    ResourceError { resource: String },
    ProviderCreationFailed { backend: String, reason: String },
    ProviderSwitchingFailed { from: String, to: String, reason: String },
}

/// Scope values for BackendCannotProceed errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendScope {
    /// Backend has verified the test case exhaustively  
    Verified,
    /// Backend has exhausted its search space
    Exhausted,
    /// Backend cannot proceed with this specific test case
    DiscardTestCase,
}

impl BackendScope {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendScope::Verified => "verified",
            BackendScope::Exhausted => "exhausted", 
            BackendScope::DiscardTestCase => "discard_test_case",
        }
    }
}

/// Provider lifecycle management
#[derive(Debug, Clone)]
pub struct ProviderContext {
    /// Current active provider name
    pub active_provider: String,
    /// Provider instance managed by Box for dynamic dispatch
    pub provider_instance: Option<String>, // We'll store the name for tracking
    /// Whether we should switch to Hypothesis provider
    pub switch_to_hypothesis: bool,
    /// Number of failed realize attempts (for discard threshold)
    pub failed_realize_count: usize,
    /// Backend that verified the test (if any)
    pub verified_by: Option<String>,
    /// Provider observability callbacks
    pub observation_callbacks: Vec<String>, // Simplified for now
}

impl Default for ProviderContext {
    fn default() -> Self {
        Self {
            active_provider: "hypothesis".to_string(),
            provider_instance: None,
            switch_to_hypothesis: false,
            failed_realize_count: 0,
            verified_by: None,
            observation_callbacks: Vec::new(),
        }
    }
}

impl std::fmt::Display for OrchestrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrchestrationError::Overrun => write!(f, "Test case overran buffer limits"),
            OrchestrationError::Invalid { reason } => write!(f, "Invalid test case: {}", reason),
            OrchestrationError::Provider { message } => write!(f, "Provider error: {}", message),
            OrchestrationError::BackendCannotProceed { scope } => write!(f, "Backend cannot proceed: {}", scope),
            OrchestrationError::Interrupted => write!(f, "Test execution was interrupted"),
            OrchestrationError::LimitsExceeded { limit_type } => write!(f, "Execution limits exceeded: {}", limit_type),
            OrchestrationError::FlakyTest { details } => write!(f, "Flaky test detected: {}", details),
            OrchestrationError::ResourceError { resource } => write!(f, "Resource allocation failed: {}", resource),
            OrchestrationError::ProviderCreationFailed { backend, reason } => write!(f, "Failed to create provider '{}': {}", backend, reason),
            OrchestrationError::ProviderSwitchingFailed { from, to, reason } => write!(f, "Failed to switch provider from '{}' to '{}': {}", from, to, reason),
        }
    }
}

impl std::error::Error for OrchestrationError {}

/// Result type for orchestration operations
pub type OrchestrationResult<T> = Result<T, OrchestrationError>;

/// Configuration for the test execution orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Maximum number of examples to generate
    pub max_examples: usize,
    /// Maximum number of iterations before giving up
    pub max_iterations: usize,
    /// Whether to ignore execution limits
    pub ignore_limits: bool,
    /// Database key for example storage
    pub database_key: Option<Vec<u8>>,
    /// Enable verbose debug logging
    pub debug_logging: bool,
    /// Backend provider to use
    pub backend: String,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            max_examples: DEFAULT_MAX_EXAMPLES,
            max_iterations: 1000,
            ignore_limits: false,
            database_key: None,
            debug_logging: false,
            backend: "hypothesis".to_string(),
            random_seed: None,
        }
    }
}

/// The main test execution orchestrator
pub struct EngineOrchestrator<P: PrimitiveProvider> {
    /// Configuration
    config: OrchestratorConfig,
    /// Test function to execute
    test_function: Box<dyn Fn(&mut ConjectureData) -> OrchestrationResult<()> + Send + Sync>,
    /// Primitive provider for generating choices
    provider: P,
    /// Provider integration context for backend management
    provider_context: ProviderContext,
    /// Current execution phase
    current_phase: ExecutionPhase,
    /// Execution statistics
    statistics: ExecutionStatistics,
    /// Health check state
    health_check_state: Option<HealthCheckState>,
    /// Count of test function calls
    call_count: usize,
    /// Count of valid examples
    valid_examples: usize,
    /// Count of invalid examples  
    invalid_examples: usize,
    /// Count of overrun examples
    overrun_examples: usize,
    /// Count of successful shrinks
    shrinks: usize,
    /// Interesting examples found
    interesting_examples: HashMap<String, ConjectureResult>,
    /// Examples that have been shrunk
    shrunk_examples: HashSet<String>,
    /// Start time of execution
    start_time: Instant,
    /// Finish time for shrinking deadline
    finish_shrinking_deadline: Option<Instant>,
    /// Whether execution should be terminated
    should_terminate: bool,
    /// Exit reason when terminating
    exit_reason: Option<ExitReason>,
}

impl<P: PrimitiveProvider> EngineOrchestrator<P> {
    /// Create a new test execution orchestrator
    pub fn new(
        test_function: Box<dyn Fn(&mut ConjectureData) -> OrchestrationResult<()> + Send + Sync>,
        provider: P,
        config: OrchestratorConfig,
    ) -> Self {
        eprintln!("Initializing EngineOrchestrator with config: {:?}", config);
        
        let mut provider_context = ProviderContext::default();
        provider_context.active_provider = config.backend.clone();
        
        Self {
            config,
            test_function,
            provider,
            provider_context,
            current_phase: ExecutionPhase::Initialize,
            statistics: ExecutionStatistics::default(),
            health_check_state: Some(HealthCheckState::default()),
            call_count: 0,
            valid_examples: 0,
            invalid_examples: 0,
            overrun_examples: 0,
            shrinks: 0,
            interesting_examples: HashMap::new(),
            shrunk_examples: HashSet::new(),
            start_time: Instant::now(),
            finish_shrinking_deadline: None,
            should_terminate: false,
            exit_reason: None,
        }
    }

    /// Run the complete test execution lifecycle
    pub fn run(&mut self) -> OrchestrationResult<ExecutionStatistics> {
        eprintln!("Starting test execution orchestration");
        self.start_time = Instant::now();
        
        // Initialize health checking
        self.health_check_state = Some(HealthCheckState::default());
        
        let result = self.run_internal();
        
        // Always cleanup regardless of result
        self.cleanup();
        
        match result {
            Ok(_) => {
                eprintln!("Test execution completed successfully");
                eprintln!("Final statistics: {:?}", self.statistics);
                Ok(self.statistics.clone())
            }
            Err(e) => {
                eprintln!("Test execution failed: {}", e);
                Err(e)
            }
        }
    }

    /// Internal run implementation
    fn run_internal(&mut self) -> OrchestrationResult<()> {
        self.transition_to_phase(ExecutionPhase::Initialize)?;
        self.initialize()?;

        // Provider Integration: Use Hypothesis provider for reuse phase (database interpretation)
        self.transition_to_phase(ExecutionPhase::Reuse)?;
        let reuse_provider = self.select_provider_for_phase(ExecutionPhase::Reuse);
        eprintln!("Provider Integration: Reuse phase using provider '{}'", reuse_provider);
        self.provider_context.switch_to_hypothesis = true; // Force Hypothesis for reuse
        self.run_phase_with_statistics(ExecutionPhase::Reuse, |engine| engine.reuse_existing_examples())?;

        if !self.should_terminate {
            // Provider Integration: Use configured provider for generation (unless switched)
            self.transition_to_phase(ExecutionPhase::Generate)?;
            let generate_provider = self.select_provider_for_phase(ExecutionPhase::Generate);
            eprintln!("Provider Integration: Generate phase using provider '{}'", generate_provider);
            self.provider_context.switch_to_hypothesis = false; // Reset to use configured provider
            self.run_phase_with_statistics(ExecutionPhase::Generate, |engine| engine.generate_new_examples())?;
        }

        if !self.should_terminate && !self.interesting_examples.is_empty() {
            // Provider Integration: Use Hypothesis provider for shrinking (reliable shrinking)
            self.transition_to_phase(ExecutionPhase::Shrink)?;
            let shrink_provider = self.select_provider_for_phase(ExecutionPhase::Shrink);
            eprintln!("Provider Integration: Shrink phase using provider '{}'", shrink_provider);
            self.provider_context.switch_to_hypothesis = true; // Force Hypothesis for shrinking
            self.finish_shrinking_deadline = Some(Instant::now() + Duration::from_secs(MAX_SHRINKING_SECONDS));
            self.run_phase_with_statistics(ExecutionPhase::Shrink, |engine| engine.shrink_interesting_examples())?;
        }

        if self.exit_reason.is_none() {
            self.exit_with(ExitReason::Finished);
        }

        Ok(())
    }

    /// Initialize the orchestrator
    fn initialize(&mut self) -> OrchestrationResult<()> {
        eprintln!("Initializing orchestrator with {} max examples", self.config.max_examples);
        
        // Validate configuration
        if self.config.max_examples == 0 {
            return Err(OrchestrationError::Invalid {
                reason: "max_examples must be greater than 0".to_string(),
            });
        }

        // Initialize provider context
        self.provider_context.active_provider = self.config.backend.clone();
        eprintln!("Provider Integration: Initializing with backend '{}'", self.provider_context.active_provider);
        
        // Validate that the requested backend is available
        let registry = get_provider_registry();
        let available_providers = registry.available_providers();
        if !available_providers.contains(&self.config.backend) {
            return Err(OrchestrationError::ProviderCreationFailed {
                backend: self.config.backend.clone(),
                reason: format!("Backend not found. Available: {:?}", available_providers),
            });
        }
        
        eprintln!("Provider Integration: Backend '{}' validated successfully", self.config.backend);
        Ok(())
    }

    /// Transition to a new execution phase
    fn transition_to_phase(&mut self, phase: ExecutionPhase) -> OrchestrationResult<()> {
        eprintln!("Transitioning from {:?} to {:?}", self.current_phase, phase);
        self.current_phase = phase;
        Ok(())
    }

    /// Run a phase with automatic statistics collection
    fn run_phase_with_statistics<F>(&mut self, phase: ExecutionPhase, f: F) -> OrchestrationResult<()>
    where
        F: FnOnce(&mut Self) -> OrchestrationResult<()>,
    {
        let start_time = Instant::now();
        let initial_call_count = self.call_count;
        let initial_shrinks = self.shrinks;
        let initial_failures = self.interesting_examples.len();

        let result = f(self);

        let phase_stats = PhaseStatistics {
            duration_seconds: start_time.elapsed().as_secs_f64(),
            test_cases: self.call_count - initial_call_count,
            distinct_failures: self.interesting_examples.len() - initial_failures,
            shrinks_successful: self.shrinks - initial_shrinks,
        };

        self.statistics.phases.insert(phase, phase_stats);
        eprintln!("Phase {:?} completed: {:?}", phase, self.statistics.phases[&phase]);

        result
    }

    /// Reuse existing examples from the database
    fn reuse_existing_examples(&mut self) -> OrchestrationResult<()> {
        eprintln!("Reusing existing examples");
        
        if self.config.database_key.is_none() {
            eprintln!("No database key configured, skipping reuse phase");
            return Ok(());
        }

        // TODO: Implement database integration
        // For now, this is a placeholder that would:
        // 1. Fetch examples from the database using the key
        // 2. Replay them through the test function
        // 3. Update interesting_examples with any that still fail
        // 4. Track reuse statistics

        eprintln!("Database reuse not yet implemented");
        Ok(())
    }

    /// Generate new test examples
    fn generate_new_examples(&mut self) -> OrchestrationResult<()> {
        eprintln!("Generating new examples");

        while self.should_generate_more() && !self.should_terminate {
            self.check_limits()?;
            
            // Create a new ConjectureData for this test case
            let mut data = ConjectureData::new(42); // Use a fixed seed for now
            
            // Execute the test function
            match (self.test_function)(&mut data) {
                Ok(_) => {
                    self.process_test_result(data, Status::Valid)?;
                }
                Err(e) => {
                    match e {
                        OrchestrationError::Invalid { .. } => {
                            self.process_test_result(data, Status::Invalid)?;
                        }
                        OrchestrationError::Overrun => {
                            self.process_test_result(data, Status::Overrun)?;
                        }
                        OrchestrationError::BackendCannotProceed { scope } => {
                            // Provider Integration: Handle BackendCannotProceed with provider switching
                            let backend_scope = match scope.as_str() {
                                "verified" => BackendScope::Verified,
                                "exhausted" => BackendScope::Exhausted,
                                "discard_test_case" => BackendScope::DiscardTestCase,
                                _ => BackendScope::DiscardTestCase, // Default fallback
                            };
                            
                            self.handle_backend_cannot_proceed(backend_scope)?;
                            // Skip normal test result processing - BackendCannotProceed is handled specially
                            self.call_count += 1;
                            continue;
                        }
                        _ => {
                            self.process_test_result(data, Status::Interesting)?;
                        }
                    }
                }
            }

            self.call_count += 1;
            
            // Check health periodically
            if self.call_count % 10 == 0 {
                if let Some(ref health) = self.health_check_state {
                    self.check_health(health)?;
                }
            }
        }

        eprintln!("Generated {} examples", self.call_count);
        Ok(())
    }

    /// Determine if we should generate more examples
    fn should_generate_more(&self) -> bool {
        if self.config.ignore_limits {
            return true;
        }

        // Check if we've hit our limits
        if self.valid_examples >= self.config.max_examples {
            return false;
        }

        if self.call_count >= self.config.max_iterations.max(self.config.max_examples * 10) {
            return false;
        }

        // If we have interesting examples and aren't configured for multiple bugs, stop
        if !self.interesting_examples.is_empty() {
            // For now, assume we want to stop after finding the first interesting example
            return false;
        }

        true
    }

    /// Process the result of executing a test case
    fn process_test_result(&mut self, data: ConjectureData, status: Status) -> OrchestrationResult<()> {
        match status {
            Status::Valid => {
                self.valid_examples += 1;
                eprintln!("Valid example #{}", self.valid_examples);
            }
            Status::Invalid => {
                self.invalid_examples += 1;
                eprintln!("Invalid example #{}", self.invalid_examples);
            }
            Status::Overrun => {
                self.overrun_examples += 1;
                eprintln!("Overrun example #{}", self.overrun_examples);
            }
            Status::Interesting => {
                let key = format!("failure_{}", self.interesting_examples.len());
                let result = data.as_result();
                self.interesting_examples.insert(key.clone(), result);
                eprintln!("Found interesting example: {}", key);
                
                // For now, exit after finding the first interesting example
                if !self.config.ignore_limits {
                    self.exit_with(ExitReason::Finished);
                }
            }
        }

        Ok(())
    }

    /// Shrink interesting examples to minimal failing cases
    fn shrink_interesting_examples(&mut self) -> OrchestrationResult<()> {
        eprintln!("Shrinking {} interesting examples", self.interesting_examples.len());

        if self.interesting_examples.is_empty() {
            return Ok(());
        }

        // Set up shrinking deadline
        let deadline = self.finish_shrinking_deadline.unwrap_or_else(|| {
            Instant::now() + Duration::from_secs(MAX_SHRINKING_SECONDS)
        });

        let mut examples_to_shrink: Vec<_> = self.interesting_examples.keys().cloned().collect();
        examples_to_shrink.sort(); // For deterministic ordering

        for example_key in examples_to_shrink {
            if Instant::now() > deadline {
                eprintln!("Shrinking deadline exceeded, stopping shrinking");
                self.exit_with(ExitReason::VerySlowShrinking);
                break;
            }

            if self.shrunk_examples.contains(&example_key) {
                continue;
            }

            if let Some(example) = self.interesting_examples.get(&example_key).cloned() {
                eprintln!("Shrinking example: {}", example_key);
                
                // TODO: Implement actual shrinking logic
                // For now, just mark as shrunk
                self.shrunk_examples.insert(example_key);
                self.shrinks += 1;

                if self.shrinks >= MAX_SHRINKS {
                    self.exit_with(ExitReason::MaxShrinks);
                    break;
                }
            }
        }

        eprintln!("Completed shrinking with {} shrinks", self.shrinks);
        Ok(())
    }

    /// Check execution limits and health
    fn check_limits(&mut self) -> OrchestrationResult<()> {
        if self.config.ignore_limits {
            return Ok(());
        }

        // Check if we should terminate
        if self.should_terminate {
            return Err(OrchestrationError::Interrupted);
        }

        // Check shrinking deadline
        if let Some(deadline) = self.finish_shrinking_deadline {
            if Instant::now() > deadline {
                self.exit_with(ExitReason::VerySlowShrinking);
                return Err(OrchestrationError::LimitsExceeded {
                    limit_type: "shrinking_deadline".to_string(),
                });
            }
        }

        Ok(())
    }

    /// Check health of the test execution
    fn check_health(&self, health: &HealthCheckState) -> OrchestrationResult<()> {
        // Check for too many invalid examples
        if health.invalid_examples >= 50 && health.valid_examples < 10 {
            eprintln!("Too many invalid examples: {} invalid vs {} valid", 
                  health.invalid_examples, health.valid_examples);
        }

        // Check for too many overruns
        if health.overrun_examples >= 20 && health.valid_examples < 10 {
            eprintln!("Too many overrun examples: {} overrun vs {} valid",
                  health.overrun_examples, health.valid_examples);
        }

        // Check timing
        let total_time = health.total_draw_time();
        if total_time > 30.0 && health.valid_examples < 10 {
            eprintln!("Test execution is very slow: {:.2}s for {} valid examples{}",
                  total_time, health.valid_examples, health.timing_report());
        }

        Ok(())
    }

    /// Mark execution for termination with the given reason
    fn exit_with(&mut self, reason: ExitReason) {
        if self.exit_reason.is_none() {
            eprintln!("Exiting with reason: {:?}", reason);
            self.exit_reason = Some(reason.clone());
            self.should_terminate = true;
            self.statistics.stopped_because = Some(reason.description(self.config.max_examples));
        }
    }

    /// Cleanup resources and finalize execution
    fn cleanup(&mut self) {
        eprintln!("Cleaning up orchestrator resources");
        
        self.transition_to_phase(ExecutionPhase::Cleanup)
            .unwrap_or_else(|e| eprintln!("Failed to transition to cleanup phase: {}", e));

        // Provider Integration: Clean up provider context
        self.cleanup_provider_context();

        // Finalize statistics
        if let Some(reason) = &self.exit_reason {
            self.statistics.stopped_because = Some(reason.description(self.config.max_examples));
        }

        // Log final summary
        let duration = self.start_time.elapsed();
        eprintln!(
            "Execution completed: {} calls ({} valid, {} invalid, {} overrun), {} shrinks in {:.2}s",
            self.call_count, self.valid_examples, self.invalid_examples, 
            self.overrun_examples, self.shrinks, duration.as_secs_f64()
        );

        // Provider Integration: Log final provider statistics
        self.log_provider_observation("execution_completed", &format!(
            "final_backend={}, switched={}, verified_by={:?}",
            self.provider_context.active_provider,
            self.provider_context.switch_to_hypothesis,
            self.provider_context.verified_by
        ));
    }

    /// Get current execution statistics
    pub fn statistics(&self) -> &ExecutionStatistics {
        &self.statistics
    }

    /// Get current health check state
    pub fn health_check_state(&self) -> Option<&HealthCheckState> {
        self.health_check_state.as_ref()
    }

    /// Check if execution should terminate
    pub fn should_terminate(&self) -> bool {
        self.should_terminate
    }

    /// Get the current execution phase
    pub fn current_phase(&self) -> ExecutionPhase {
        self.current_phase
    }

    /// Get the number of interesting examples found
    pub fn interesting_examples_count(&self) -> usize {
        self.interesting_examples.len()
    }

    /// Get the current call count
    pub fn call_count(&self) -> usize {
        self.call_count
    }

    /// Set the call count (for testing purposes)
    pub fn set_call_count(&mut self, count: usize) {
        self.call_count = count;
    }

    // =======================================
    // Provider Integration Capability
    // =======================================

    /// Check if we're currently using the Hypothesis backend
    pub fn using_hypothesis_backend(&self) -> bool {
        self.config.backend == "hypothesis" || self.provider_context.switch_to_hypothesis
    }

    /// Handle BackendCannotProceed exception and determine provider switching logic
    /// 
    /// This implements the Python ConjectureRunner's BackendCannotProceed handling
    /// with provider switching logic based on the scope of the error.
    pub fn handle_backend_cannot_proceed(&mut self, scope: BackendScope) -> OrchestrationResult<()> {
        eprintln!("Provider Integration: BackendCannotProceed with scope '{}'", scope.as_str());
        
        match scope {
            BackendScope::Verified | BackendScope::Exhausted => {
                eprintln!("Provider Integration: Switching to Hypothesis provider due to {} scope", scope.as_str());
                self.switch_to_hypothesis_provider()?;
                
                if scope == BackendScope::Verified {
                    self.provider_context.verified_by = Some(self.config.backend.clone());
                    eprintln!("Provider Integration: Test verified by backend '{}'", self.config.backend);
                }
            }
            BackendScope::DiscardTestCase => {
                self.provider_context.failed_realize_count += 1;
                eprintln!("Provider Integration: Failed realize count: {}", self.provider_context.failed_realize_count);
                
                // Switch to Hypothesis if we have too many failed realizes
                if self.provider_context.failed_realize_count > 10 
                    && (self.provider_context.failed_realize_count as f64 / self.call_count as f64) > 0.2 {
                    eprintln!("Provider Integration: Too many failed realizes ({}), switching to Hypothesis", 
                             self.provider_context.failed_realize_count);
                    self.switch_to_hypothesis_provider()?;
                }
            }
        }

        // All BackendCannotProceed exceptions are treated as invalid examples
        self.invalid_examples += 1;
        
        Ok(())
    }

    /// Switch to the Hypothesis provider
    /// 
    /// This implements the Python ConjectureRunner's _switch_to_hypothesis_provider logic.
    /// When backends cannot proceed, we fall back to the Hypothesis provider for more
    /// reliable generation and shrinking.
    pub fn switch_to_hypothesis_provider(&mut self) -> OrchestrationResult<()> {
        if self.provider_context.switch_to_hypothesis {
            eprintln!("Provider Integration: Already using Hypothesis provider");
            return Ok(());
        }

        let previous_provider = self.provider_context.active_provider.clone();
        eprintln!("Provider Integration: Switching from '{}' to 'hypothesis'", previous_provider);
        
        // Attempt to create the Hypothesis provider to validate the switch
        let registry = get_provider_registry();
        if let Some(_hypothesis_provider) = registry.create("hypothesis") {
            self.provider_context.switch_to_hypothesis = true;
            self.provider_context.active_provider = "hypothesis".to_string();
            eprintln!("Provider Integration: Successfully switched to Hypothesis provider");
            
            // Log provider switch in statistics
            self.log_provider_observation("provider_switched", &format!("{}->hypothesis", previous_provider));
            
            Ok(())
        } else {
            Err(OrchestrationError::ProviderSwitchingFailed {
                from: previous_provider,
                to: "hypothesis".to_string(),
                reason: "Failed to create Hypothesis provider instance".to_string(),
            })
        }
    }

    /// Create a new provider instance based on current context
    /// 
    /// This provides dynamic provider instantiation based on the current
    /// provider context and switching state.
    pub fn create_active_provider(&self) -> OrchestrationResult<Box<dyn PrimitiveProvider>> {
        let provider_name = if self.provider_context.switch_to_hypothesis {
            "hypothesis"
        } else {
            &self.provider_context.active_provider
        };

        eprintln!("Provider Integration: Creating provider instance for '{}'", provider_name);
        
        let registry = get_provider_registry();
        match registry.create(provider_name) {
            Some(provider) => {
                eprintln!("Provider Integration: Successfully created provider '{}'", provider_name);
                Ok(provider)
            }
            None => {
                Err(OrchestrationError::ProviderCreationFailed {
                    backend: provider_name.to_string(),
                    reason: "Provider not found in registry".to_string(),
                })
            }
        }
    }

    /// Register a provider observability callback
    /// 
    /// This provides a mechanism for observing provider-specific events
    /// and integrating with external monitoring systems.
    pub fn register_provider_observation_callback(&mut self, callback_id: String) {
        eprintln!("Provider Integration: Registering observation callback '{}'", callback_id);
        self.provider_context.observation_callbacks.push(callback_id);
    }

    /// Log a provider-specific observation event
    /// 
    /// This enables structured logging of provider events for debugging
    /// and observability purposes.
    pub fn log_provider_observation(&self, event_type: &str, details: &str) {
        let hex_id = format!("{:08X}", self.call_count); // Uppercase hex notation
        eprintln!("Provider Integration: [{}] {} - {}", hex_id, event_type, details);
        
        // If observation callbacks are registered, we could invoke them here
        for callback in &self.provider_context.observation_callbacks {
            eprintln!("Provider Integration: Notifying callback '{}' of event '{}'", callback, event_type);
        }
    }

    /// Get the current provider context for inspection
    pub fn provider_context(&self) -> &ProviderContext {
        &self.provider_context
    }

    /// Determine which provider to use for the current phase
    /// 
    /// This implements the Python ConjectureRunner's phase-specific provider selection:
    /// - Reuse phase: Always use Hypothesis provider for database interpretation
    /// - Generate phase: Use configured provider unless switched
    /// - Shrink phase: Always use Hypothesis provider for reliable shrinking
    pub fn select_provider_for_phase(&self, phase: ExecutionPhase) -> String {
        match phase {
            ExecutionPhase::Initialize => {
                self.provider_context.active_provider.clone()
            }
            ExecutionPhase::Reuse => {
                // Always use Hypothesis for database reuse
                eprintln!("Provider Integration: Using Hypothesis provider for reuse phase");
                "hypothesis".to_string()
            }
            ExecutionPhase::Generate => {
                if self.provider_context.switch_to_hypothesis {
                    eprintln!("Provider Integration: Using Hypothesis provider for generation (switched)");
                    "hypothesis".to_string()
                } else {
                    eprintln!("Provider Integration: Using configured provider '{}' for generation", 
                             self.provider_context.active_provider);
                    self.provider_context.active_provider.clone()
                }
            }
            ExecutionPhase::Shrink => {
                // Always use Hypothesis for shrinking
                eprintln!("Provider Integration: Using Hypothesis provider for shrink phase");
                "hypothesis".to_string()
            }
            ExecutionPhase::Cleanup => {
                self.provider_context.active_provider.clone()
            }
        }
    }

    /// Clean up provider resources and contexts
    /// 
    /// This ensures proper resource management with RAII patterns
    /// and provides clean shutdown for provider instances.
    pub fn cleanup_provider_context(&mut self) {
        eprintln!("Provider Integration: Cleaning up provider context");
        
        // Log final provider statistics
        self.log_provider_observation("cleanup_started", &format!(
            "active_provider={}, switch_to_hypothesis={}, failed_realizes={}", 
            self.provider_context.active_provider,
            self.provider_context.switch_to_hypothesis,
            self.provider_context.failed_realize_count
        ));

        // Clear observation callbacks
        let callback_count = self.provider_context.observation_callbacks.len();
        self.provider_context.observation_callbacks.clear();
        eprintln!("Provider Integration: Cleared {} observation callbacks", callback_count);

        // Reset provider context to default state
        if let Some(verified_by) = &self.provider_context.verified_by {
            eprintln!("Provider Integration: Test case was verified by backend '{}'", verified_by);
        }

        eprintln!("Provider Integration: Provider context cleanup completed");
    }
}

impl<P: PrimitiveProvider> Drop for EngineOrchestrator<P> {
    fn drop(&mut self) {
        if !matches!(self.current_phase, ExecutionPhase::Cleanup) {
            eprintln!("EngineOrchestrator dropped without proper cleanup");
            // Ensure provider cleanup happens even in abnormal shutdown
            self.cleanup_provider_context();
            self.cleanup();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::HypothesisProvider;

    #[test]
    fn test_orchestrator_creation() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = HypothesisProvider::new();
        let config = OrchestratorConfig::default();
        
        let orchestrator = EngineOrchestrator::new(test_fn, provider, config);
        assert_eq!(orchestrator.current_phase(), ExecutionPhase::Initialize);
        assert_eq!(orchestrator.interesting_examples_count(), 0);
    }

    #[test]
    fn test_exit_reason_description() {
        assert_eq!(
            ExitReason::MaxExamples.description(100),
            "max_examples=100"
        );
        assert_eq!(
            ExitReason::Finished.description(100),
            "nothing left to do"
        );
    }

    #[test]
    fn test_health_check_state() {
        let mut health = HealthCheckState::default();
        health.draw_times.insert("test_op".to_string(), vec![0.1, 0.2, 0.05]);
        
        assert!((health.total_draw_time() - 0.35).abs() < 1e-10);
        assert!(!health.timing_report().is_empty());
    }

    #[test]
    fn test_phase_statistics() {
        let stats = PhaseStatistics {
            duration_seconds: 1.5,
            test_cases: 50,
            distinct_failures: 2,
            shrinks_successful: 10,
        };
        
        assert_eq!(stats.duration_seconds, 1.5);
        assert_eq!(stats.test_cases, 50);
    }

    // =======================================
    // Provider Integration Tests
    // =======================================

    #[test]
    fn test_provider_context_default() {
        let context = ProviderContext::default();
        assert_eq!(context.active_provider, "hypothesis");
        assert!(!context.switch_to_hypothesis);
        assert_eq!(context.failed_realize_count, 0);
        assert!(context.verified_by.is_none());
        assert!(context.observation_callbacks.is_empty());
    }

    #[test]
    fn test_backend_scope_conversion() {
        assert_eq!(BackendScope::Verified.as_str(), "verified");
        assert_eq!(BackendScope::Exhausted.as_str(), "exhausted");
        assert_eq!(BackendScope::DiscardTestCase.as_str(), "discard_test_case");
    }

    #[test]
    fn test_orchestrator_using_hypothesis_backend() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = HypothesisProvider::new();
        let config = OrchestratorConfig::default();
        
        let orchestrator = EngineOrchestrator::new(test_fn, provider, config);
        assert!(orchestrator.using_hypothesis_backend()); // Default backend is "hypothesis"
    }

    #[test]
    fn test_orchestrator_with_custom_backend() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = HypothesisProvider::new();
        let mut config = OrchestratorConfig::default();
        config.backend = "random".to_string();
        
        let orchestrator = EngineOrchestrator::new(test_fn, provider, config);
        assert!(!orchestrator.using_hypothesis_backend()); // Custom backend, not switched yet
        assert_eq!(orchestrator.provider_context().active_provider, "random");
    }

    #[test]
    fn test_provider_phase_selection() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = HypothesisProvider::new();
        let mut config = OrchestratorConfig::default();
        config.backend = "random".to_string();
        
        let orchestrator = EngineOrchestrator::new(test_fn, provider, config);
        
        // Reuse and Shrink phases always use Hypothesis
        assert_eq!(orchestrator.select_provider_for_phase(ExecutionPhase::Reuse), "hypothesis");
        assert_eq!(orchestrator.select_provider_for_phase(ExecutionPhase::Shrink), "hypothesis");
        
        // Generate phase uses configured provider (unless switched)
        assert_eq!(orchestrator.select_provider_for_phase(ExecutionPhase::Generate), "random");
    }

    #[test]
    fn test_handle_backend_cannot_proceed_verified() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = HypothesisProvider::new();
        let mut config = OrchestratorConfig::default();
        config.backend = "crosshair".to_string();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, provider, config);
        
        // Handle verified scope
        let result = orchestrator.handle_backend_cannot_proceed(BackendScope::Verified);
        assert!(result.is_ok());
        assert!(orchestrator.provider_context.switch_to_hypothesis);
        assert_eq!(orchestrator.provider_context.verified_by, Some("crosshair".to_string()));
        assert_eq!(orchestrator.invalid_examples, 1); // BackendCannotProceed counted as invalid
    }

    #[test]
    fn test_handle_backend_cannot_proceed_discard_threshold() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = HypothesisProvider::new();
        let config = OrchestratorConfig::default();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, provider, config);
        orchestrator.call_count = 50; // Set up call count for threshold calculation
        
        // Trigger multiple discard test case errors to hit threshold
        for _ in 0..12 {
            let result = orchestrator.handle_backend_cannot_proceed(BackendScope::DiscardTestCase);
            assert!(result.is_ok());
        }
        
        // Should switch after crossing 20% threshold with >10 failed realizes
        assert!(orchestrator.provider_context.switch_to_hypothesis);
        assert_eq!(orchestrator.provider_context.failed_realize_count, 12);
    }

    #[test]
    fn test_switch_to_hypothesis_provider() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = HypothesisProvider::new();
        let mut config = OrchestratorConfig::default();
        config.backend = "random".to_string();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, provider, config);
        
        assert!(!orchestrator.provider_context.switch_to_hypothesis);
        assert_eq!(orchestrator.provider_context.active_provider, "random");
        
        let result = orchestrator.switch_to_hypothesis_provider();
        assert!(result.is_ok());
        assert!(orchestrator.provider_context.switch_to_hypothesis);
        assert_eq!(orchestrator.provider_context.active_provider, "hypothesis");
    }

    #[test]
    fn test_provider_observation_callbacks() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = HypothesisProvider::new();
        let config = OrchestratorConfig::default();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, provider, config);
        
        // Register observation callbacks
        orchestrator.register_provider_observation_callback("test_callback_1".to_string());
        orchestrator.register_provider_observation_callback("test_callback_2".to_string());
        
        assert_eq!(orchestrator.provider_context.observation_callbacks.len(), 2);
        assert!(orchestrator.provider_context.observation_callbacks.contains(&"test_callback_1".to_string()));
        assert!(orchestrator.provider_context.observation_callbacks.contains(&"test_callback_2".to_string()));
    }

    #[test]
    fn test_provider_cleanup() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let provider = HypothesisProvider::new();
        let config = OrchestratorConfig::default();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, provider, config);
        
        // Set up some state
        orchestrator.register_provider_observation_callback("test_callback".to_string());
        orchestrator.provider_context.verified_by = Some("test_backend".to_string());
        
        // Cleanup should clear callbacks
        orchestrator.cleanup_provider_context();
        assert!(orchestrator.provider_context.observation_callbacks.is_empty());
    }

    #[test]
    fn test_orchestration_error_display() {
        let error1 = OrchestrationError::ProviderCreationFailed {
            backend: "test_backend".to_string(),
            reason: "not found".to_string(),
        };
        assert_eq!(
            format!("{}", error1),
            "Failed to create provider 'test_backend': not found"
        );

        let error2 = OrchestrationError::ProviderSwitchingFailed {
            from: "crosshair".to_string(),
            to: "hypothesis".to_string(),
            reason: "creation failed".to_string(),
        };
        assert_eq!(
            format!("{}", error2),
            "Failed to switch provider from 'crosshair' to 'hypothesis': creation failed"
        );
    }
}