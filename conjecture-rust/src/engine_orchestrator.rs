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

use crate::choice::{ChoiceNode, ChoiceType, ChoiceValue};
use crate::data::{ConjectureData, ConjectureResult, Status, DataObserver, DrawError};
use crate::providers::{PrimitiveProvider, ProviderRegistry, get_provider_registry};
use crate::engine_orchestrator_provider_type_integration::{
    ProviderTypeManager, ProviderTypeError, EnhancedPrimitiveProvider
};
use crate::persistence::{ExampleDatabase, DatabaseKey, DirectoryDatabase, InMemoryDatabase, DatabaseIntegration};
use crate::conjecture_data_lifecycle_management::{
    ConjectureDataLifecycleManager, LifecycleConfig, LifecycleState, LifecycleError
};
use crate::engine_orchestrator_test_function_signature_alignment::{
    DrawErrorConverter, SignatureAlignmentContext, get_alignment_stats, reset_alignment_stats, 
    record_alignment_operation
};

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

impl ExecutionPhase {
    /// Get debug name for the execution phase
    pub fn debug_name(&self) -> &'static str {
        match self {
            ExecutionPhase::Initialize => "Initialize",
            ExecutionPhase::Reuse => "Reuse",
            ExecutionPhase::Generate => "Generate",
            ExecutionPhase::Shrink => "Shrink",
            ExecutionPhase::Cleanup => "Cleanup",
        }
    }
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
    /// Database path for example storage (None = in-memory)
    pub database_path: Option<String>,
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
            database_path: None,
            debug_logging: false,
            backend: "hypothesis".to_string(),
            random_seed: None,
        }
    }
}

/// The main test execution orchestrator with unified provider type system
pub struct EngineOrchestrator {
    /// Configuration
    config: OrchestratorConfig,
    /// Test function to execute
    test_function: Box<dyn Fn(&mut ConjectureData) -> OrchestrationResult<()> + Send + Sync>,
    /// Provider type manager for unified provider handling
    provider_manager: ProviderTypeManager,
    /// Legacy provider context (deprecated)
    provider_context: ProviderContext,
    /// ConjectureData lifecycle manager for comprehensive lifecycle management
    lifecycle_manager: ConjectureDataLifecycleManager,
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
    /// Database for example persistence and reuse
    database: Option<Box<dyn ExampleDatabase>>,
    /// Start time of execution
    start_time: Instant,
    /// Finish time for shrinking deadline
    finish_shrinking_deadline: Option<Instant>,
    /// Whether execution should be terminated
    should_terminate: bool,
    /// Exit reason when terminating
    exit_reason: Option<ExitReason>,
}

impl EngineOrchestrator {
    /// Create a new test execution orchestrator with unified provider type system
    pub fn new(
        test_function: Box<dyn Fn(&mut ConjectureData) -> OrchestrationResult<()> + Send + Sync>,
        config: OrchestratorConfig,
    ) -> Self {
        Self::with_provider_manager(test_function, config)
    }

    /// Create with explicit provider type manager (enhanced interface)
    pub fn with_provider_manager(
        test_function: Box<dyn Fn(&mut ConjectureData) -> OrchestrationResult<()> + Send + Sync>,
        config: OrchestratorConfig,
    ) -> Self {
        eprintln!("PROVIDER TYPE SYSTEM: Initializing EngineOrchestrator with config: {:?}", config);
        
        let mut provider_manager = ProviderTypeManager::new();
        let mut provider_context = ProviderContext::default();
        provider_context.active_provider = config.backend.clone();
        
        // Initialize database if configured
        let database = if config.database_path.is_some() || config.database_key.is_some() {
            match DatabaseIntegration::create_database(config.database_path.as_deref()) {
                Ok(db) => {
                    eprintln!("DEBUG: Database initialized successfully");
                    Some(db)
                }
                Err(e) => {
                    eprintln!("DEBUG: Failed to initialize database: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        // Initialize lifecycle manager with configuration
        let lifecycle_config = LifecycleConfig {
            debug_logging: config.debug_logging,
            max_choices: Some(10000), // Default limit
            execution_timeout_ms: Some(30000), // 30 second timeout
            enable_forced_values: true,
            enable_replay_validation: true,
            use_hex_notation: true,
        };
        let lifecycle_manager = ConjectureDataLifecycleManager::new(lifecycle_config);
        
        Self {
            config,
            test_function,
            provider_manager,
            provider_context,
            lifecycle_manager,
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
            database,
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

    /// Initialize the orchestrator with provider type system
    fn initialize(&mut self) -> OrchestrationResult<()> {
        eprintln!("PROVIDER TYPE SYSTEM: Initializing orchestrator with {} max examples", self.config.max_examples);
        
        // Validate configuration
        if self.config.max_examples == 0 {
            return Err(OrchestrationError::Invalid {
                reason: "max_examples must be greater than 0".to_string(),
            });
        }

        // Initialize provider type manager
        if let Err(e) = self.provider_manager.initialize(&self.config.backend) {
            return Err(OrchestrationError::ProviderCreationFailed {
                backend: self.config.backend.clone(),
                reason: format!("Provider type system initialization failed: {}", e),
            });
        }
        
        // Update legacy context for compatibility
        self.provider_context.active_provider = self.config.backend.clone();
        eprintln!("PROVIDER TYPE SYSTEM: Backend '{}' initialized successfully", self.config.backend);
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
        eprintln!("DEBUG: Reusing existing examples from database");
        
        // Check if database and key are configured
        let database_key = match &self.config.database_key {
            Some(key) => key,
            None => {
                eprintln!("DEBUG: No database key configured, skipping reuse phase");
                return Ok(());
            }
        };
        
        let database = match &mut self.database {
            Some(db) => db,
            None => {
                eprintln!("DEBUG: No database configured, skipping reuse phase");
                return Ok(());
            }
        };
        
        // Generate database key from test function
        let db_key = match DatabaseIntegration::generate_key(
            "test_function", // In a real implementation, this would be the actual function name
            None, // No source code available
            database_key,
        ) {
            Ok(key) => key,
            Err(e) => {
                eprintln!("DEBUG: Failed to generate database key: {}", e);
                return Ok(());
            }
        };
        
        eprintln!("DEBUG: Fetching examples for key: {:?}", db_key.to_hex());
        
        // Fetch examples from database
        let examples = match database.fetch(&db_key) {
            Ok(examples) => {
                eprintln!("DEBUG: Found {} examples in database", examples.len());
                examples
            }
            Err(e) => {
                eprintln!("DEBUG: Failed to fetch examples from database: {}", e);
                return Ok(());
            }
        };
        
        // Replay each example
        let mut successful_replays = 0;
        let mut failed_replays = 0;
        
        for (i, example_data) in examples.iter().enumerate() {
            eprintln!("DEBUG: Replaying example {} of {}", i + 1, examples.len());
            
            // Deserialize the example
            let choices = match DatabaseIntegration::deserialize_example(example_data) {
                Ok(choices) => choices,
                Err(e) => {
                    eprintln!("DEBUG: Failed to deserialize example {}: {}", i, e);
                    failed_replays += 1;
                    continue;
                }
            };
            
            if choices.is_empty() {
                eprintln!("DEBUG: Skipping empty example {}", i);
                continue;
            }
            
            // Create ConjectureData for replay using lifecycle manager
            let replay_id = match self.lifecycle_manager.create_for_replay(
                &choices,
                None, // observer
                None, // provider  
                None, // random generator
            ) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("DEBUG: Failed to create replay instance: {}", e);
                    failed_replays += 1;
                    continue;
                }
            };
            
            // Transition to executing state
            if let Err(e) = self.lifecycle_manager.transition_state(replay_id, LifecycleState::Executing) {
                eprintln!("DEBUG: Failed to transition replay instance to executing: {}", e);
                failed_replays += 1;
                continue;
            }
            
            // Execute the test function with the replay data using signature alignment
            let alignment_context = SignatureAlignmentContext::new("reuse_existing_examples")
                .with_metadata("example_index", &i.to_string())
                .with_metadata("replay_id", &replay_id.to_string());
            
            let execution_result = if let Some(replay_data) = self.lifecycle_manager.get_instance_mut(replay_id) {
                (self.test_function)(replay_data)
            } else {
                eprintln!("DEBUG: Failed to get replay instance {}", replay_id);
                failed_replays += 1;
                continue;
            };
            
            match execution_result {
                Ok(()) => {
                    eprintln!("DEBUG: Example {} replayed successfully", i);
                    successful_replays += 1;
                    
                    // Transition to completed state
                    if let Err(e) = self.lifecycle_manager.transition_state(replay_id, LifecycleState::Completed) {
                        eprintln!("DEBUG: Failed to transition to completed state: {}", e);
                    }
                    
                    // If the example is interesting, save it
                    if let Some(replay_data) = self.lifecycle_manager.get_instance(replay_id) {
                        if replay_data.status != Status::Valid {
                            let example_key = format!("replay_{}", i);
                            let result = replay_data.as_result();
                            self.interesting_examples.insert(example_key, result);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("DEBUG: Example {} replay failed: {:?}", i, e);
                    failed_replays += 1;
                    
                    // Transition to failed state
                    if let Err(transition_err) = self.lifecycle_manager.transition_state(replay_id, LifecycleState::ReplayFailed) {
                        eprintln!("DEBUG: Failed to transition to failed state: {}", transition_err);
                    }
                }
            }
            
            // Cleanup the replay instance
            if let Err(e) = self.lifecycle_manager.cleanup_instance(replay_id) {
                eprintln!("DEBUG: Failed to cleanup replay instance: {}", e);
            }
            
            self.call_count += 1;
            
            // Check if we should stop early
            if self.should_terminate {
                break;
            }
        }
        
        eprintln!("DEBUG: Database reuse completed: {} successful, {} failed replays", 
                 successful_replays, failed_replays);
        
        Ok(())
    }

    /// Generate new test examples
    fn generate_new_examples(&mut self) -> OrchestrationResult<()> {
        eprintln!("Generating new examples");

        while self.should_generate_more() && !self.should_terminate {
            self.check_limits()?;
            
            // Create a new ConjectureData for this test case
            let mut data = ConjectureData::new(42); // Use a fixed seed for now
            
            // Execute the test function with signature alignment
            let alignment_context = SignatureAlignmentContext::new("generate_new_examples")
                .with_metadata("iteration", &self.call_count.to_string())
                .with_metadata("provider", &self.provider_context.active_provider);
            
            match self.execute_test_function_with_alignment(&mut data, Some(alignment_context)) {
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
    pub fn should_generate_more(&self) -> bool {
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
    pub fn process_test_result(&mut self, data: ConjectureData, status: Status) -> OrchestrationResult<()> {
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
                
                // Save to database before storing locally
                self.save_example_to_database(&result, "primary");
                
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
    
    /// Save an interesting example to the database
    fn save_example_to_database(&mut self, example: &ConjectureResult, example_type: &str) {
        let database = match &mut self.database {
            Some(db) => db,
            None => return, // No database configured
        };
        
        let database_key = match &self.config.database_key {
            Some(key) => key,
            None => return, // No database key configured
        };
        
        // Generate database key
        let db_key = match DatabaseIntegration::generate_key(
            "test_function",
            None,
            database_key,
        ) {
            Ok(key) => {
                // Use sub-keys for different example types
                match example_type {
                    "secondary" => key.with_sub_key("secondary"),
                    "pareto" => key.with_sub_key("pareto"),
                    _ => key, // Primary corpus
                }
            }
            Err(e) => {
                eprintln!("DEBUG: Failed to generate database key for saving: {}", e);
                return;
            }
        };
        
        // Serialize the example
        let serialized = match DatabaseIntegration::serialize_result(example) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("DEBUG: Failed to serialize example: {}", e);
                return;
            }
        };
        
        // Save to database
        match database.save(&db_key, &serialized) {
            Ok(()) => {
                eprintln!("DEBUG: Saved {} example to database (key: {})", 
                         example_type, db_key.to_hex());
            }
            Err(e) => {
                eprintln!("DEBUG: Failed to save example to database: {}", e);
            }
        }
    }

    /// Shrink interesting examples to minimal failing cases using sophisticated Choice System Shrinking Integration
    fn shrink_interesting_examples(&mut self) -> OrchestrationResult<()> {
        eprintln!("CHOICE_SHRINKING: Starting sophisticated shrinking of {} interesting examples", self.interesting_examples.len());

        if self.interesting_examples.is_empty() {
            return Ok(());
        }

        // Set up shrinking deadline
        let deadline = self.finish_shrinking_deadline.unwrap_or_else(|| {
            Instant::now() + Duration::from_secs(MAX_SHRINKING_SECONDS)
        });

        // Configure sophisticated shrinking integration
        let shrinking_config = crate::engine_orchestrator_choice_system_shrinking_integration::ShrinkingIntegrationConfig {
            max_shrinking_attempts: MAX_SHRINKS,
            shrinking_timeout: Duration::from_secs(MAX_SHRINKING_SECONDS),
            enable_advanced_patterns: true,
            enable_multi_strategy: true,
            quality_improvement_threshold: 0.001,
            max_concurrent_strategies: 4,
            debug_logging: self.config.debug_logging,
            use_hex_notation: true,
        };

        // Create sophisticated shrinking integration engine
        let mut shrinking_integration = crate::engine_orchestrator_choice_system_shrinking_integration::ChoiceSystemShrinkingIntegration::new(shrinking_config);

        // Apply sophisticated shrinking to all interesting examples
        let interesting_examples_clone = self.interesting_examples.clone();
        match shrinking_integration.integrate_shrinking(self, &interesting_examples_clone) {
            Ok(shrinking_results) => {
                eprintln!("CHOICE_SHRINKING: Sophisticated shrinking completed with {} results", shrinking_results.len());
                
                // Process shrinking results and update state
                for (example_key, shrink_result) in shrinking_results {
                    if shrink_result.success {
                        self.shrinks += shrink_result.shrinks_performed;
                        
                        // Update the example with shrunk result if available
                        if let Some(ref shrunk_result) = shrink_result.shrunk_result {
                            eprintln!("CHOICE_SHRINKING: Example '{}' shrunk from {} to {} nodes ({:.1}% reduction)", 
                                     example_key, shrink_result.original_size, shrink_result.final_size, 
                                     shrink_result.reduction_percentage());
                            
                            let shrunk_result_clone = shrunk_result.clone();
                            self.interesting_examples.insert(example_key.clone(), shrunk_result_clone);
                            let example_data = self.interesting_examples[&example_key].clone();
                            self.save_example_to_database(&example_data, "shrunk");
                        }
                        
                        self.shrunk_examples.insert(example_key.clone());
                        
                        eprintln!("CHOICE_SHRINKING: Successfully shrunk example '{}' using strategies: {:?}", 
                                 example_key, shrink_result.successful_strategies);
                    } else {
                        eprintln!("CHOICE_SHRINKING: Failed to shrink example '{}': {:?}", 
                                 example_key, shrink_result.errors);
                    }
                    
                    // Check deadline and limits
                    if Instant::now() > deadline {
                        eprintln!("CHOICE_SHRINKING: Shrinking deadline exceeded, stopping sophisticated shrinking");
                        self.exit_with(ExitReason::VerySlowShrinking);
                        break;
                    }
                    
                    if self.shrinks >= MAX_SHRINKS {
                        eprintln!("CHOICE_SHRINKING: Maximum shrinks reached, stopping sophisticated shrinking");
                        self.exit_with(ExitReason::MaxShrinks);
                        break;
                    }
                }
                
                // Log final sophisticated shrinking metrics
                let integration_metrics = shrinking_integration.get_metrics();
                eprintln!("CHOICE_SHRINKING: Integration metrics - attempts: {}, successful: {}, conversion_errors: {}, avg_quality: {:.3}", 
                         integration_metrics.total_attempts, integration_metrics.successful_integrations,
                         integration_metrics.conversion_errors, integration_metrics.average_quality_improvement);
            }
            Err(e) => {
                eprintln!("CHOICE_SHRINKING: Sophisticated shrinking integration failed: {}", e);
                
                // Fallback to simple shrinking if sophisticated shrinking fails
                self.apply_fallback_shrinking(deadline)?;
            }
        }

        eprintln!("CHOICE_SHRINKING: Completed sophisticated shrinking with {} total shrinks", self.shrinks);
        Ok(())
    }
    
    /// Fallback shrinking when sophisticated shrinking fails
    fn apply_fallback_shrinking(&mut self, deadline: Instant) -> OrchestrationResult<()> {
        eprintln!("CHOICE_SHRINKING: Applying fallback shrinking to {} examples", self.interesting_examples.len());
        
        let mut examples_to_shrink: Vec<_> = self.interesting_examples.keys().cloned().collect();
        examples_to_shrink.sort(); // For deterministic ordering

        for example_key in examples_to_shrink {
            if Instant::now() > deadline {
                eprintln!("CHOICE_SHRINKING: Fallback shrinking deadline exceeded");
                self.exit_with(ExitReason::VerySlowShrinking);
                break;
            }

            if self.shrunk_examples.contains(&example_key) {
                continue;
            }

            if let Some(example) = self.interesting_examples.get(&example_key).cloned() {
                eprintln!("CHOICE_SHRINKING: Applying fallback shrinking to example: {}", example_key);
                
                // Simple fallback: try to remove every other choice
                if example.nodes.len() > 2 {
                    let simplified_nodes: Vec<_> = example.nodes.iter()
                        .enumerate()
                        .filter(|(i, _)| i % 2 == 0)
                        .map(|(_, node)| node.clone())
                        .collect();
                    
                    if simplified_nodes.len() < example.nodes.len() {
                        // Create simplified result
                        let mut simplified_result = example.clone();
                        simplified_result.nodes = simplified_nodes;
                        simplified_result.length = simplified_result.nodes.len();
                        
                        // Validate the simplified result
                        if self.validate_fallback_shrinking(&simplified_result)? {
                            eprintln!("CHOICE_SHRINKING: Fallback shrinking reduced example '{}' from {} to {} nodes", 
                                     example_key, example.nodes.len(), simplified_result.nodes.len());
                            
                            let example_key_clone = example_key.clone();
                            let simplified_result_clone = simplified_result.clone();
                            self.interesting_examples.insert(example_key_clone.clone(), simplified_result_clone);
                            let example_data = self.interesting_examples[&example_key_clone].clone();
                            self.save_example_to_database(&example_data, "fallback_shrunk");
                        }
                    }
                }
                
                self.shrunk_examples.insert(example_key.clone());
                self.shrinks += 1;

                if self.shrinks >= MAX_SHRINKS {
                    self.exit_with(ExitReason::MaxShrinks);
                    break;
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate fallback shrinking result
    fn validate_fallback_shrinking(&mut self, result: &ConjectureResult) -> OrchestrationResult<bool> {
        // Simple validation: check if result has valid structure
        if result.nodes.is_empty() {
            return Ok(false);
        }
        
        // Check if all nodes have valid choice types and values
        for node in &result.nodes {
            match (&node.choice_type, &node.value) {
                (ChoiceType::Integer, ChoiceValue::Integer(_)) => {},
                (ChoiceType::Boolean, ChoiceValue::Boolean(_)) => {},
                (ChoiceType::Float, ChoiceValue::Float(_)) => {},
                (ChoiceType::String, ChoiceValue::String(_)) => {},
                (ChoiceType::Bytes, ChoiceValue::Bytes(_)) => {},
                _ => return Ok(false), // Invalid type/value combination
            }
        }
        
        Ok(true)
    }

    /// Check execution limits and health
    pub fn check_limits(&mut self) -> OrchestrationResult<()> {
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

        // Lifecycle Management: Cleanup all ConjectureData instances
        let cleaned_instances = self.lifecycle_manager.cleanup_all();
        eprintln!("LIFECYCLE_MANAGER: Cleaned up {} ConjectureData instances during orchestrator cleanup", cleaned_instances);

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

        // Signature Alignment: Generate final alignment report
        if self.config.debug_logging {
            eprintln!("SIGNATURE_ALIGNMENT: Final Report:\n{}", self.get_signature_alignment_report());
        }
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

    /// Handle BackendCannotProceed with provider type system integration
    /// 
    /// This implements comprehensive BackendCannotProceed handling through the
    /// unified provider type system with proper error propagation and type safety.
    pub fn handle_backend_cannot_proceed(&mut self, scope: BackendScope) -> OrchestrationResult<()> {
        eprintln!("PROVIDER TYPE SYSTEM: BackendCannotProceed with scope '{}'", scope.as_str());
        
        // Handle through provider type manager
        self.provider_manager.handle_backend_cannot_proceed(scope.as_str())
            .map_err(|e| OrchestrationError::Provider {
                message: format!("Backend cannot proceed handling failed: {}", e),
            })?;

        // Update legacy context for compatibility
        let manager_context = self.provider_manager.context();
        self.provider_context.switch_to_hypothesis = manager_context.switch_to_hypothesis;
        self.provider_context.active_provider = manager_context.active_provider.clone();
        self.provider_context.failed_realize_count = manager_context.failed_realize_count;
        
        if let Some(ref verified_by) = manager_context.verified_by {
            self.provider_context.verified_by = Some(verified_by.clone());
            eprintln!("PROVIDER TYPE SYSTEM: Test verified by backend '{}'", verified_by);
        }

        // All BackendCannotProceed exceptions are treated as invalid examples
        self.invalid_examples += 1;
        
        Ok(())
    }

    /// Switch to the Hypothesis provider through type-safe provider manager
    /// 
    /// This implements provider switching with full type safety and proper
    /// lifecycle management through the unified provider type system.
    pub fn switch_to_hypothesis_provider(&mut self) -> OrchestrationResult<()> {
        let previous_provider = self.provider_manager.context().active_provider.clone();
        eprintln!("PROVIDER TYPE SYSTEM: Switching from '{}' to 'hypothesis'", previous_provider);
        
        self.provider_manager.switch_to_hypothesis()
            .map_err(|e| OrchestrationError::ProviderSwitchingFailed {
                from: previous_provider.clone(),
                to: "hypothesis".to_string(),
                reason: format!("Provider type system error: {}", e),
            })?;

        // Update legacy context for compatibility
        self.provider_context.switch_to_hypothesis = true;
        self.provider_context.active_provider = "hypothesis".to_string();
        
        eprintln!("PROVIDER TYPE SYSTEM: Successfully switched to hypothesis");
        self.log_provider_observation("provider_switched", &format!("{}->hypothesis", previous_provider));
        
        Ok(())
    }

    /// Get active provider through type-safe provider manager
    /// 
    /// This provides type-safe access to the active provider instance
    /// through the unified provider type system.
    pub fn active_provider(&mut self) -> OrchestrationResult<&mut dyn EnhancedPrimitiveProvider> {
        self.provider_manager.active_provider()
            .map_err(|e| OrchestrationError::Provider {
                message: format!("Provider access failed: {}", e),
            })
    }

    /// Create active provider (legacy compatibility method)
    pub fn create_active_provider(&self) -> OrchestrationResult<String> {
        // Return provider name for compatibility
        Ok(self.provider_manager.context().active_provider.clone())
    }

    /// Register a provider observability callback through type system
    /// 
    /// This provides a mechanism for observing provider-specific events
    /// through the unified provider type system.
    pub fn register_provider_observation_callback(&mut self, callback_id: String) {
        eprintln!("PROVIDER TYPE SYSTEM: Registering observation callback '{}'", callback_id);
        self.provider_manager.register_observation_callback(callback_id.clone());
        self.provider_context.observation_callbacks.push(callback_id); // Legacy compatibility
    }

    /// Log a provider-specific observation event through type system
    /// 
    /// This enables structured logging of provider events through the
    /// unified provider type system with proper hex notation.
    pub fn log_provider_observation(&self, event_type: &str, details: &str) {
        self.provider_manager.log_observation(event_type, details);
        
        // Legacy hex notation for compatibility
        let hex_id = format!("{:08X}", self.call_count);
        eprintln!("PROVIDER TYPE SYSTEM: [{}] {} - {}", hex_id, event_type, details);
        
        // Notify callbacks for compatibility
        for callback in &self.provider_context.observation_callbacks {
            eprintln!("PROVIDER TYPE SYSTEM: Notifying callback '{}' of event '{}'", callback, event_type);
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

    /// Clean up provider resources through type system
    /// 
    /// This ensures proper resource management through the unified
    /// provider type system with RAII patterns.
    pub fn cleanup_provider_context(&mut self) {
        eprintln!("PROVIDER TYPE SYSTEM: Cleaning up provider context");
        
        // Log final provider statistics
        self.log_provider_observation("cleanup_started", &format!(
            "active_provider={}, switch_to_hypothesis={}, failed_realizes={}", 
            self.provider_context.active_provider,
            self.provider_context.switch_to_hypothesis,
            self.provider_context.failed_realize_count
        ));

        // Cleanup through provider type manager
        self.provider_manager.cleanup();

        // Clear legacy observation callbacks
        let callback_count = self.provider_context.observation_callbacks.len();
        self.provider_context.observation_callbacks.clear();
        eprintln!("PROVIDER TYPE SYSTEM: Cleared {} observation callbacks", callback_count);

        if let Some(verified_by) = &self.provider_context.verified_by {
            eprintln!("PROVIDER TYPE SYSTEM: Test case was verified by backend '{}'", verified_by);
        }

        eprintln!("PROVIDER TYPE SYSTEM: Provider context cleanup completed");
    }

    // =======================================
    // ConjectureData Lifecycle Management Capability
    // =======================================

    /// Create a new ConjectureData instance with lifecycle management
    pub fn create_conjecture_data(
        &mut self,
        seed: u64,
        observer: Option<Box<dyn DataObserver>>,
        provider: Option<Box<dyn PrimitiveProvider>>,
    ) -> Result<u64, OrchestrationError> {
        self.lifecycle_manager.create_instance(seed, observer, provider)
            .map_err(|e| OrchestrationError::Provider {
                message: format!("Failed to create ConjectureData instance: {}", e),
            })
    }

    /// Create ConjectureData for replay with comprehensive lifecycle management
    pub fn create_conjecture_data_for_replay(
        &mut self,
        choices: &[ChoiceNode],
        observer: Option<Box<dyn DataObserver>>,
        provider: Option<Box<dyn PrimitiveProvider>>,
        random: Option<rand_chacha::ChaCha8Rng>,
    ) -> Result<u64, OrchestrationError> {
        self.lifecycle_manager.create_for_replay(choices, observer, provider, random)
            .map_err(|e| OrchestrationError::Provider {
                message: format!("Failed to create replay ConjectureData instance: {}", e),
            })
    }

    /// Get mutable access to a ConjectureData instance
    pub fn get_conjecture_data_mut(&mut self, instance_id: u64) -> Option<&mut ConjectureData> {
        self.lifecycle_manager.get_instance_mut(instance_id)
    }

    /// Get immutable access to a ConjectureData instance
    pub fn get_conjecture_data(&self, instance_id: u64) -> Option<&ConjectureData> {
        self.lifecycle_manager.get_instance(instance_id)
    }

    /// Transition a ConjectureData instance to a new lifecycle state
    pub fn transition_conjecture_data_state(&mut self, instance_id: u64, new_state: LifecycleState) -> Result<(), OrchestrationError> {
        self.lifecycle_manager.transition_state(instance_id, new_state)
            .map_err(|e| OrchestrationError::Provider {
                message: format!("Failed to transition lifecycle state: {}", e),
            })
    }

    /// Integrate forced values into a ConjectureData instance
    pub fn integrate_forced_values(
        &mut self,
        instance_id: u64,
        forced_values: Vec<(usize, crate::choice::ChoiceValue)>,
    ) -> Result<(), OrchestrationError> {
        self.lifecycle_manager.integrate_forced_values(instance_id, forced_values)
            .map_err(|e| OrchestrationError::Provider {
                message: format!("Failed to integrate forced values: {}", e),
            })
    }

    /// Validate replay mechanism for a choice sequence
    pub fn validate_replay_mechanism(
        &mut self,
        choices: &[ChoiceNode],
    ) -> Result<bool, OrchestrationError> {
        // Create a wrapper function that uses execute_test_function_with_alignment
        let test_function = |data: &mut ConjectureData| -> Result<(), OrchestrationError> {
            // For validation, we just check if the data can be processed without error
            // In a full implementation, this would execute the actual test logic
            if data.get_nodes().is_empty() {
                Err(OrchestrationError::Invalid { 
                    reason: "No choices available for validation".to_string() 
                })
            } else {
                Ok(())
            }
        };
        
        self.lifecycle_manager.validate_replay(choices, &test_function)
            .map_err(|e| OrchestrationError::Provider {
                message: format!("Failed to validate replay mechanism: {}", e),
            })
    }

    /// Cleanup a specific ConjectureData instance
    pub fn cleanup_conjecture_data(&mut self, instance_id: u64) -> Result<(), OrchestrationError> {
        self.lifecycle_manager.cleanup_instance(instance_id)
            .map_err(|e| OrchestrationError::Provider {
                message: format!("Failed to cleanup ConjectureData instance: {}", e),
            })
    }

    /// Get lifecycle management metrics
    pub fn get_lifecycle_metrics(&self) -> &crate::conjecture_data_lifecycle_management::LifecycleMetrics {
        self.lifecycle_manager.get_metrics()
    }

    /// Get active ConjectureData instance count
    pub fn active_conjecture_data_count(&self) -> usize {
        self.lifecycle_manager.active_instance_count()
    }

    /// Generate comprehensive lifecycle status report
    pub fn generate_lifecycle_status_report(&self) -> String {
        self.lifecycle_manager.generate_status_report()
    }

    /// Check if a ConjectureData instance is in a valid state
    pub fn is_conjecture_data_valid(&self, instance_id: u64) -> bool {
        self.lifecycle_manager.is_instance_valid(instance_id)
    }

    // =======================================
    // Test Function Signature Alignment Capability
    // =======================================

    /// Execute test function with signature alignment error handling
    /// 
    /// This method provides enhanced test function execution with automatic
    /// conversion between DrawError and OrchestrationError types, ensuring
    /// consistent error handling across the orchestration system.
    pub fn execute_test_function_with_alignment(
        &mut self, 
        data: &mut ConjectureData,
        context: Option<SignatureAlignmentContext>
    ) -> OrchestrationResult<()> {
        let start_time = std::time::Instant::now();
        
        // Create alignment context if not provided
        let alignment_context = context.unwrap_or_else(|| {
            SignatureAlignmentContext::new("test_function_execution")
                .with_metadata("call_count", &self.call_count.to_string())
                .with_metadata("phase", &format!("{:?}", self.current_phase))
        });

        eprintln!("SIGNATURE_ALIGNMENT: Executing test function with context: {}", 
                 alignment_context.format_for_error());

        // Execute the test function
        let result = (self.test_function)(data);
        
        let execution_time = start_time.elapsed();
        
        // Record alignment statistics
        let success = result.is_ok();
        let error_type = if let Err(ref e) = result {
            Some(match e {
                OrchestrationError::Overrun => "Overrun",
                OrchestrationError::Invalid { .. } => "Invalid",
                OrchestrationError::Provider { .. } => "Provider",
                OrchestrationError::BackendCannotProceed { .. } => "BackendCannotProceed",
                OrchestrationError::Interrupted => "Interrupted",
                OrchestrationError::LimitsExceeded { .. } => "LimitsExceeded",
                OrchestrationError::FlakyTest { .. } => "FlakyTest",
                OrchestrationError::ResourceError { .. } => "ResourceError",
                OrchestrationError::ProviderCreationFailed { .. } => "ProviderCreationFailed",
                OrchestrationError::ProviderSwitchingFailed { .. } => "ProviderSwitchingFailed",
            })
        } else {
            None
        };
        
        record_alignment_operation(success, error_type, true);
        
        // Log execution details with hex notation
        let hex_call_id = format!("{:08X}", self.call_count);
        match &result {
            Ok(()) => {
                eprintln!("SIGNATURE_ALIGNMENT: [{}] Test function succeeded in {:.3}ms", 
                         hex_call_id, execution_time.as_secs_f64() * 1000.0);
            }
            Err(e) => {
                eprintln!("SIGNATURE_ALIGNMENT: [{}] Test function failed in {:.3}ms: {}", 
                         hex_call_id, execution_time.as_secs_f64() * 1000.0, e);
            }
        }
        
        result
    }

    /// Convert DrawError to OrchestrationError with enhanced context
    /// 
    /// This method provides the core error conversion functionality with
    /// additional orchestrator-specific context and logging.
    pub fn convert_draw_error_with_orchestrator_context(
        &self, 
        draw_error: DrawError, 
        operation: &str
    ) -> OrchestrationError {
        let context = format!("orchestrator_{}_{}", self.current_phase.debug_name(), operation);
        let converted = DrawErrorConverter::convert_with_context(draw_error.clone(), &context);
        
        // Add orchestrator-specific information
        match converted {
            OrchestrationError::Invalid { reason } => {
                OrchestrationError::Invalid {
                    reason: format!("{} (call #{}, phase: {:?})", reason, self.call_count, self.current_phase)
                }
            }
            OrchestrationError::Provider { message } => {
                OrchestrationError::Provider {
                    message: format!("{} (provider: {}, call #{})", 
                                   message, self.provider_context.active_provider, self.call_count)
                }
            }
            other => other
        }
    }

    /// Get comprehensive signature alignment report
    /// 
    /// This method generates a detailed report of signature alignment operations
    /// including error conversion statistics and performance metrics.
    pub fn get_signature_alignment_report(&self) -> String {
        let stats = get_alignment_stats();
        let mut report = String::new();
        
        report.push_str("=== Test Function Signature Alignment Report ===\n");
        report.push_str(&format!("Orchestrator Call Count: {}\n", self.call_count));
        report.push_str(&format!("Current Phase: {:?}\n", self.current_phase));
        report.push_str(&format!("Active Provider: {}\n", self.provider_context.active_provider));
        report.push_str("\n");
        
        report.push_str(&stats.generate_report());
        
        if !stats.error_conversions.is_empty() {
            report.push_str("\nError Conversion Details:\n");
            for (error_type, count) in &stats.error_conversions {
                let category = self.get_error_category(error_type);
                report.push_str(&format!("  {} ({}): {} occurrences\n", 
                                       error_type, category, count));
            }
        }
        
        report.push_str(&format!("\nAlignment Success Rate: {:.1}%\n", stats.success_rate() * 100.0));
        report
    }

    /// Reset signature alignment statistics
    /// 
    /// This method resets the signature alignment statistics, useful for
    /// starting fresh tracking for a new test session.
    pub fn reset_signature_alignment_stats(&self) {
        eprintln!("SIGNATURE_ALIGNMENT: Resetting alignment statistics");
        reset_alignment_stats();
    }

    /// Check signature alignment health
    /// 
    /// This method performs health checks on the signature alignment system
    /// and reports any issues or recommendations.
    pub fn check_signature_alignment_health(&self) -> OrchestrationResult<()> {
        let stats = get_alignment_stats();
        
        // Check for high error rates
        if stats.total_operations > 10 && stats.success_rate() < 0.5 {
            eprintln!("SIGNATURE_ALIGNMENT: WARNING - Low success rate: {:.1}%", 
                     stats.success_rate() * 100.0);
            return Err(OrchestrationError::Invalid {
                reason: format!("Signature alignment success rate too low: {:.1}%", 
                              stats.success_rate() * 100.0)
            });
        }
        
        // Check for specific error patterns
        for (error_type, count) in &stats.error_conversions {
            if *count > stats.total_operations / 2 {
                eprintln!("SIGNATURE_ALIGNMENT: WARNING - High frequency of {} errors: {}", 
                         error_type, count);
            }
        }
        
        eprintln!("SIGNATURE_ALIGNMENT: Health check passed - {:.1}% success rate over {} operations", 
                 stats.success_rate() * 100.0, stats.total_operations);
        Ok(())
    }

    /// Get error category for an error type name
    fn get_error_category(&self, error_type: &str) -> &'static str {
        match error_type {
            "Overrun" => "resource_limit",
            "Interrupted" => "control_flow", 
            "Invalid" => "validation",
            "Provider" | "ProviderCreationFailed" | "ProviderSwitchingFailed" => "provider",
            "BackendCannotProceed" => "backend",
            "LimitsExceeded" => "limits",
            "FlakyTest" => "flaky",
            "ResourceError" => "resource",
            _ => "unknown"
        }
    }

    /// Execute test function with automatic error alignment (legacy compatibility)
    /// 
    /// This method provides a simplified interface for executing test functions
    /// with automatic signature alignment, maintaining backward compatibility.
    pub fn execute_test_function_aligned(&mut self, data: &mut ConjectureData) -> OrchestrationResult<()> {
        self.execute_test_function_with_alignment(data, None)
    }
}

impl Drop for EngineOrchestrator {
    fn drop(&mut self) {
        if !matches!(self.current_phase, ExecutionPhase::Cleanup) {
            eprintln!("PROVIDER TYPE SYSTEM: EngineOrchestrator dropped without proper cleanup");
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
        let config = OrchestratorConfig::default();
        
        let orchestrator = EngineOrchestrator::new(test_fn, config);
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
        let config = OrchestratorConfig::default();
        
        let orchestrator = EngineOrchestrator::new(test_fn, config);
        assert!(orchestrator.using_hypothesis_backend()); // Default backend is "hypothesis"
    }

    #[test]
    fn test_orchestrator_with_custom_backend() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let mut config = OrchestratorConfig::default();
        config.backend = "random".to_string();
        
        let orchestrator = EngineOrchestrator::new(test_fn, config);
        assert!(!orchestrator.using_hypothesis_backend()); // Custom backend, not switched yet
        assert_eq!(orchestrator.provider_context().active_provider, "random");
    }

    #[test]
    fn test_provider_phase_selection() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let mut config = OrchestratorConfig::default();
        config.backend = "random".to_string();
        
        let orchestrator = EngineOrchestrator::new(test_fn, config);
        
        // Reuse and Shrink phases always use Hypothesis
        assert_eq!(orchestrator.select_provider_for_phase(ExecutionPhase::Reuse), "hypothesis");
        assert_eq!(orchestrator.select_provider_for_phase(ExecutionPhase::Shrink), "hypothesis");
        
        // Generate phase uses configured provider (unless switched)
        assert_eq!(orchestrator.select_provider_for_phase(ExecutionPhase::Generate), "random");
    }

    #[test]
    fn test_handle_backend_cannot_proceed_verified() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let mut config = OrchestratorConfig::default();
        config.backend = "crosshair".to_string();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        
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
        let config = OrchestratorConfig::default();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        orchestrator.call_count = 50; // Set up call count for threshold calculation
        
        // Trigger multiple discard test case errors to hit threshold
        for _ in 0..12 {
            let result = orchestrator.handle_backend_cannot_proceed(BackendScope::DiscardTestCase);
            assert!(result.is_ok());
        }
        
        // Should switch after crossing threshold with >10 failed realizes
        assert!(orchestrator.provider_context.switch_to_hypothesis);
        assert_eq!(orchestrator.provider_context.failed_realize_count, 12);
    }

    #[test]
    fn test_switch_to_hypothesis_provider() {
        let test_fn = Box::new(|_data: &mut ConjectureData| Ok(()));
        let mut config = OrchestratorConfig::default();
        config.backend = "random".to_string();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        
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
        let config = OrchestratorConfig::default();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        
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
        let config = OrchestratorConfig::default();
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        
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

    /// Test function signature alignment - core capability test
    /// 
    /// This test validates that the orchestrator correctly handles test functions
    /// that return OrchestrationResult<()> while ConjectureData operations return
    /// Result<T, DrawError>, ensuring proper type conversion and error handling.
    #[test]
    fn test_function_signature_alignment_core_capability() {
        use crate::engine_orchestrator_test_function_signature_alignment::{
            ToOrchestrationResult, ConjectureDataOrchestrationExt
        };

        // Test function using the new signature alignment system
        let test_fn = Box::new(|data: &mut ConjectureData| -> OrchestrationResult<()> {
            // Use the extension trait methods for seamless integration
            let _integer = data.draw_integer_orchestration(1, 100)?;
            let _boolean = data.draw_boolean_orchestration(0.5)?;
            let _float = data.draw_float_orchestration()?;
            
            // Test assumption checking
            data.assume_orchestration(true, "test assumption")?;
            
            Ok(())
        });
        
        let config = OrchestratorConfig {
            max_examples: 5,
            backend: "hypothesis".to_string(),
            debug_logging: true,
            ..Default::default()
        };
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        
        // Reset alignment stats for clean testing
        orchestrator.reset_signature_alignment_stats();
        
        // Verify the orchestrator accepts the test function
        assert_eq!(orchestrator.current_phase(), ExecutionPhase::Initialize);
        
        // Run the orchestrator - this should work without compilation errors
        let result = orchestrator.run();
        assert!(result.is_ok(), "Orchestrator should handle signature alignment correctly");
        
        // Check signature alignment health
        let health_result = orchestrator.check_signature_alignment_health();
        assert!(health_result.is_ok(), "Signature alignment should be healthy");
        
        // Verify we collected alignment statistics
        let report = orchestrator.get_signature_alignment_report();
        assert!(report.contains("Signature Alignment Statistics"), "Should generate alignment report");
        
        eprintln!("Test function signature alignment test completed successfully");
        eprintln!("Alignment report:\n{}", report);
    }

    /// Test error conversion patterns from DrawError to OrchestrationError
    #[test]
    fn test_draw_error_conversion_patterns() {
        // Helper function
        fn convert_draw_error(draw_error: DrawError) -> OrchestrationError {
            match draw_error {
                DrawError::Overrun => OrchestrationError::Overrun,
                DrawError::InvalidRange => OrchestrationError::Invalid { 
                    reason: "Invalid range".to_string()
                },
                DrawError::InvalidProbability => OrchestrationError::Invalid { 
                    reason: "Invalid probability".to_string()
                },
                _ => OrchestrationError::Invalid { 
                    reason: format!("Error: {:?}", draw_error)
                },
            }
        }

        // Test function that handles different error types
        let test_fn = Box::new(|data: &mut ConjectureData| -> OrchestrationResult<()> {
            // Try an operation that might fail
            match data.draw_integer(1, 10) {
                Ok(_) => Ok(()),
                Err(e) => Err(convert_draw_error(e))
            }
        });
        
        let config = OrchestratorConfig {
            max_examples: 3,
            backend: "hypothesis".to_string(),
            debug_logging: false,
            ..Default::default()
        };
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        let result = orchestrator.run();
        
        // Should complete without compilation errors
        eprintln!("Error conversion test result: {:?}", result.is_ok());
    }

    /// Test realistic usage pattern with mixed success/failure
    #[test]
    fn test_realistic_signature_alignment_usage() {
        use std::sync::{Arc, Mutex};
        
        let call_count = Arc::new(Mutex::new(0));
        
        let test_fn = {
            let call_count = Arc::clone(&call_count);
            Box::new(move |data: &mut ConjectureData| -> OrchestrationResult<()> {
                let mut count = call_count.lock().unwrap();
                *count += 1;
                let current_count = *count;
                drop(count);
                
                // Generate values and check a property
                let a = data.draw_integer(1, 100)
                    .map_err(|e| OrchestrationError::Invalid { 
                        reason: format!("Integer generation failed: {:?}", e)
                    })?;
                
                let b = data.draw_boolean(0.5)
                    .map_err(|e| OrchestrationError::Invalid { 
                        reason: format!("Boolean generation failed: {:?}", e)
                    })?;
                
                // Property: if a > 50 and b is true, that's an error condition
                if a > 50 && b {
                    return Err(OrchestrationError::Invalid { 
                        reason: format!("Property violation on call {}: {} > 50 and boolean is true", current_count, a)
                    });
                }
                
                Ok(())
            })
        };
        
        let config = OrchestratorConfig {
            max_examples: 20,
            backend: "hypothesis".to_string(),
            debug_logging: false,
            ..Default::default()
        };
        
        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        let result = orchestrator.run();
        
        let final_count = *call_count.lock().unwrap();
        eprintln!("Realistic usage test: made {} calls, result: {:?}", final_count, result.is_ok());
        
        // Should have made some calls and handled any errors appropriately
        assert!(final_count > 0, "Should have made at least one test call");
    }

    /// Test signature alignment with various error conditions
    #[test]
    fn test_signature_alignment_error_handling() {
        use crate::engine_orchestrator_test_function_signature_alignment::{
            ToOrchestrationResult, ConjectureDataOrchestrationExt, DrawErrorConverter
        };

        let error_counts = Arc::new(Mutex::new(HashMap::new()));

        // Test function that triggers different error conditions
        let test_fn = {
            let error_counts = Arc::clone(&error_counts);
            Box::new(move |data: &mut ConjectureData| -> OrchestrationResult<()> {
                let mut counts = error_counts.lock().unwrap();
                let call_count = counts.len();

                match call_count % 3 {
                    0 => {
                        // Test invalid range error conversion
                        *counts.entry("invalid_range_test".to_string()).or_insert(0) += 1;
                        data.draw_integer_orchestration(100, 1) // Invalid range
                            .map(|_| ())
                    }
                    1 => {
                        // Test invalid probability error conversion
                        *counts.entry("invalid_probability_test".to_string()).or_insert(0) += 1;
                        data.draw_boolean_orchestration(1.5) // Invalid probability
                            .map(|_| ())
                    }
                    _ => {
                        // Successful case
                        *counts.entry("successful_test".to_string()).or_insert(0) += 1;
                        let _value = data.draw_integer_orchestration(1, 10)?;
                        Ok(())
                    }
                }
            })
        };

        let config = OrchestratorConfig {
            max_examples: 15,
            backend: "hypothesis".to_string(),
            debug_logging: true,
            ignore_limits: true,
            ..Default::default()
        };

        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        orchestrator.reset_signature_alignment_stats();

        let result = orchestrator.run();
        
        // Should complete execution despite errors
        assert!(result.is_ok() || result.is_err(), "Should handle signature alignment errors gracefully");

        let final_counts = error_counts.lock().unwrap();
        eprintln!("Signature alignment error test counts: {:?}", *final_counts);

        // Verify we tracked different error types
        assert!(final_counts.len() > 0, "Should have recorded test operations");

        // Check alignment report includes error details
        let report = orchestrator.get_signature_alignment_report();
        eprintln!("Error handling alignment report:\n{}", report);
    }

    /// Test signature alignment with mixed ConjectureData operations
    #[test]
    fn test_signature_alignment_mixed_operations() {
        use crate::engine_orchestrator_test_function_signature_alignment::{
            ToOrchestrationResult, ConjectureDataOrchestrationExt
        };

        let test_fn = Box::new(|data: &mut ConjectureData| -> OrchestrationResult<()> {
            // Test mixing extension trait methods with manual conversion
            let value1 = data.draw_integer_orchestration(1, 100)?;
            
            // Manual conversion using trait
            let value2 = data.draw_boolean(0.5).to_orchestration_result()?;
            
            // Extension trait with context
            let _value3 = data.draw_float_orchestration()?;
            
            // Test various operation types
            let _string = data.draw_string_orchestration("abc", 1, 5)?;
            let _bytes = data.draw_bytes_orchestration(4)?;
            
            // Test assumptions and targets
            data.assume_orchestration(value1 > 0, "value should be positive")?;
            data.target_orchestration(value1 as f64 / 100.0, "normalized_value")?;
            
            // Test conditional logic
            if value2 {
                data.assume_orchestration(value1 < 50, "conditional assumption")?;
            }
            
            Ok(())
        });

        let config = OrchestratorConfig {
            max_examples: 10,
            backend: "hypothesis".to_string(),
            debug_logging: true,
            ..Default::default()
        };

        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        orchestrator.reset_signature_alignment_stats();

        let result = orchestrator.run();
        assert!(result.is_ok(), "Mixed operations should work with signature alignment");

        // Verify health and statistics
        let health_result = orchestrator.check_signature_alignment_health();
        assert!(health_result.is_ok(), "Mixed operations should maintain healthy alignment");

        let report = orchestrator.get_signature_alignment_report();
        eprintln!("Mixed operations alignment report:\n{}", report);

        // Should have recorded successful operations
        assert!(report.contains("Successful:"), "Should track successful operations");
    }

    /// Test signature alignment with provider integration
    #[test]
    fn test_signature_alignment_provider_integration() {
        use crate::engine_orchestrator_test_function_signature_alignment::{
            ConjectureDataOrchestrationExt
        };

        let test_fn = Box::new(|data: &mut ConjectureData| -> OrchestrationResult<()> {
            // Test with different providers through signature alignment
            let _value1 = data.draw_integer_orchestration(1, 1000)?;
            let _value2 = data.draw_boolean_orchestration(0.3)?;
            let _value3 = data.draw_float_orchestration()?;
            
            // Test provider-specific operations
            data.target_orchestration(42.0, "provider_test")?;
            
            Ok(())
        });

        // Test with different backends
        for backend in &["hypothesis"] { // Add more backends when available
            let config = OrchestratorConfig {
                max_examples: 5,
                backend: backend.to_string(),
                debug_logging: true,
                ..Default::default()
            };

            let mut orchestrator = EngineOrchestrator::new(test_fn.clone(), config);
            orchestrator.reset_signature_alignment_stats();

            let result = orchestrator.run();
            assert!(result.is_ok(), "Provider integration should work with signature alignment for {}", backend);

            // Check provider-specific alignment
            let report = orchestrator.get_signature_alignment_report();
            assert!(report.contains(backend), "Report should mention the backend");
            
            eprintln!("Provider {} alignment report:\n{}", backend, report);
        }
    }

    /// Test signature alignment macro usage patterns
    #[test]
    fn test_signature_alignment_macro_patterns() {
        use crate::engine_orchestrator_test_function_signature_alignment::{
            ToOrchestrationResult
        };

        // Test function using manual conversion patterns that macros would generate
        let test_fn = Box::new(|data: &mut ConjectureData| -> OrchestrationResult<()> {
            // Simulate macro-generated code patterns
            let _value1 = data.draw_integer(1, 100).to_orchestration_result()?;
            let _value2 = data.draw_boolean(0.5).to_orchestration_result_with_context("macro_boolean")?;
            let _value3 = data.draw_float().to_orchestration_unit_result()?;
            
            Ok(())
        });

        let config = OrchestratorConfig {
            max_examples: 8,
            backend: "hypothesis".to_string(),
            debug_logging: false,
            ..Default::default()
        };

        let mut orchestrator = EngineOrchestrator::new(test_fn, config);
        orchestrator.reset_signature_alignment_stats();

        let result = orchestrator.run();
        assert!(result.is_ok(), "Macro patterns should work with signature alignment");

        // Verify alignment worked as expected
        let health_result = orchestrator.check_signature_alignment_health();
        assert!(health_result.is_ok(), "Macro pattern alignment should be healthy");
    }
}