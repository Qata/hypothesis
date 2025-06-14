//! Targeting System & Coverage-Guided Generation
//!
//! This module implements comprehensive targeting capabilities for property-based testing,
//! providing Pareto frontier optimization for multi-objective targeting and coverage-guided
//! test case generation to improve test quality and exploration efficiency.
//!
//! # Architecture
//!
//! The targeting system consists of:
//! - **Target Functions**: Define optimization objectives (minimize/maximize functions)
//! - **Pareto Frontier**: Multi-objective optimization using Pareto dominance
//! - **Coverage Engine**: Branch coverage tracking and guided generation
//! - **Target Observations**: Collect and analyze optimization data
//! - **Integration Interface**: Seamless integration with ChoiceSystem and test engine
//!
//! # Example Usage
//!
//! ```rust
//! use crate::targeting::{TargetingEngine, TargetFunction, OptimizationDirection};
//!
//! // Create targeting engine
//! let mut engine = TargetingEngine::new();
//!
//! // Add target functions
//! engine.add_target(MinimizeFunction::new("minimize_value"));
//! engine.add_target(MaximizeFunction::new("maximize_coverage"));
//!
//! // Update with test results
//! engine.update_with_result(&test_result)?;
//!
//! // Get guided suggestions
//! let suggestions = engine.get_targeting_suggestions()?;
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, Duration};
use std::cmp::Ordering;

use serde::{Serialize, Deserialize};

use crate::choice::{ChoiceNode, ChoiceValue};
use crate::data::{ConjectureData, ConjectureResult, ExtraInformation, Status};

/// Type alias for targeting operation results
pub type TargetingResult<T> = Result<T, TargetingError>;

/// Comprehensive error types for targeting operations
#[derive(Debug, thiserror::Error)]
pub enum TargetingError {
    #[error("Invalid target function: {0}")]
    InvalidTarget(String),
    
    #[error("Pareto computation error: {0}")]
    ParetoError(String),
    
    #[error("Coverage tracking error: {0}")]
    CoverageError(String),
    
    #[error("Insufficient data for targeting: {0}")]
    InsufficientData(String),
    
    #[error("Target observation error: {0}")]
    ObservationError(String),
}

/// Direction for optimization objectives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationDirection {
    /// Minimize the target function value
    Minimize,
    /// Maximize the target function value
    Maximize,
}

/// Target observation for tracking optimization objectives
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetObservation {
    /// The observed value for this target
    pub value: f64,
    /// Human-readable label for this observation
    pub label: String,
    /// Additional metadata about the observation
    pub metadata: HashMap<String, String>,
    /// Timestamp when this observation was made
    pub timestamp: SystemTime,
}

impl TargetObservation {
    /// Create a new target observation
    pub fn new(value: f64, label: String) -> Self {
        TargetObservation {
            value,
            label,
            metadata: HashMap::new(),
            timestamp: SystemTime::now(),
        }
    }
    
    /// Add metadata to this observation
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Core trait for target functions that define optimization objectives
pub trait TargetFunction: Send + Sync {
    /// Evaluate the target function for given test data
    fn evaluate(&self, data: &ConjectureData) -> TargetingResult<f64>;
    
    /// Get the human-readable label for this target
    fn label(&self) -> &str;
    
    /// Get the optimization direction (minimize or maximize)
    fn direction(&self) -> OptimizationDirection;
    
    /// Get additional metadata about this target function
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }
    
    /// Check if this target function is applicable to the given data
    fn is_applicable(&self, _data: &ConjectureData) -> bool {
        true
    }
}

/// Multi-objective optimization point on the Pareto frontier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint {
    /// Objective values for all target functions
    pub objectives: Vec<f64>,
    /// The test case that produced these objectives
    pub test_case: Vec<u8>,
    /// All target observations for this point
    pub observations: Vec<TargetObservation>,
    /// Choice sequence that generated this test case
    pub choice_sequence: Vec<ChoiceNode>,
    /// Quality score for ranking among Pareto-optimal points
    pub quality_score: f64,
    /// Timestamp when this point was discovered
    pub discovered_at: SystemTime,
}

impl ParetoPoint {
    /// Create a new Pareto point
    pub fn new(
        objectives: Vec<f64>,
        test_case: Vec<u8>,
        observations: Vec<TargetObservation>,
        choice_sequence: Vec<ChoiceNode>,
    ) -> Self {
        let quality_score = Self::calculate_quality_score(&objectives, &observations);
        
        ParetoPoint {
            objectives,
            test_case,
            observations,
            choice_sequence,
            quality_score,
            discovered_at: SystemTime::now(),
        }
    }
    
    /// Check if this point dominates another point (Pareto dominance)
    pub fn dominates(&self, other: &ParetoPoint, directions: &[OptimizationDirection]) -> bool {
        if self.objectives.len() != other.objectives.len() || 
           self.objectives.len() != directions.len() {
            return false;
        }
        
        let mut at_least_one_better = false;
        
        for i in 0..self.objectives.len() {
            let self_better = match directions[i] {
                OptimizationDirection::Minimize => self.objectives[i] < other.objectives[i],
                OptimizationDirection::Maximize => self.objectives[i] > other.objectives[i],
            };
            
            let other_better = match directions[i] {
                OptimizationDirection::Minimize => other.objectives[i] < self.objectives[i],
                OptimizationDirection::Maximize => other.objectives[i] > self.objectives[i],
            };
            
            if other_better {
                return false; // Other is better in at least one objective
            }
            
            if self_better {
                at_least_one_better = true;
            }
        }
        
        at_least_one_better
    }
    
    /// Check if this point is Pareto-equivalent to another point
    pub fn equivalent(&self, other: &ParetoPoint) -> bool {
        if self.objectives.len() != other.objectives.len() {
            return false;
        }
        
        const EPSILON: f64 = 1e-10;
        self.objectives.iter()
            .zip(other.objectives.iter())
            .all(|(a, b)| (a - b).abs() < EPSILON)
    }
    
    /// Calculate quality score based on objectives and observations
    fn calculate_quality_score(objectives: &[f64], observations: &[TargetObservation]) -> f64 {
        // Combine objective values with observation metadata quality
        let objective_score: f64 = objectives.iter().map(|v| v.abs()).sum();
        let observation_bonus = observations.len() as f64 * 0.1;
        let metadata_bonus = observations.iter()
            .map(|obs| obs.metadata.len() as f64 * 0.01)
            .sum::<f64>();
        
        objective_score + observation_bonus + metadata_bonus
    }
}

/// Coverage tracking state for guided test generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageState {
    /// Set of covered branch identifiers
    pub covered_branches: HashSet<String>,
    /// Count of how many times each branch has been hit
    pub branch_counts: HashMap<String, u32>,
    /// Interesting test regions that led to new coverage
    pub interesting_regions: Vec<Vec<u8>>,
    /// Coverage evolution over time
    pub coverage_history: VecDeque<CoverageSnapshot>,
    /// Last time coverage was updated
    pub last_updated: SystemTime,
}

/// Snapshot of coverage state at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageSnapshot {
    pub timestamp: SystemTime,
    pub total_branches: usize,
    pub new_branches: Vec<String>,
}

impl CoverageState {
    /// Create new coverage state
    pub fn new() -> Self {
        CoverageState {
            covered_branches: HashSet::new(),
            branch_counts: HashMap::new(),
            interesting_regions: Vec::new(),
            coverage_history: VecDeque::new(),
            last_updated: SystemTime::now(),
        }
    }
    
    /// Update coverage with new branch information
    pub fn update_coverage(&mut self, branches: &[String], test_case: &[u8]) -> bool {
        let mut new_coverage = false;
        let mut new_branches = Vec::new();
        
        for branch in branches {
            *self.branch_counts.entry(branch.clone()).or_insert(0) += 1;
            
            if self.covered_branches.insert(branch.clone()) {
                new_coverage = true;
                new_branches.push(branch.clone());
            }
        }
        
        if new_coverage {
            self.interesting_regions.push(test_case.to_vec());
            
            // Keep only recent interesting regions (max 1000)
            if self.interesting_regions.len() > 1000 {
                self.interesting_regions.drain(0..self.interesting_regions.len() - 1000);
            }
            
            // Add coverage snapshot
            let snapshot = CoverageSnapshot {
                timestamp: SystemTime::now(),
                total_branches: self.covered_branches.len(),
                new_branches,
            };
            
            self.coverage_history.push_back(snapshot);
            
            // Keep only recent history (max 100 snapshots)
            if self.coverage_history.len() > 100 {
                self.coverage_history.pop_front();
            }
        }
        
        self.last_updated = SystemTime::now();
        new_coverage
    }
    
    /// Get coverage statistics
    pub fn get_stats(&self) -> CoverageStats {
        let recent_snapshots = self.coverage_history.iter()
            .rev()
            .take(10)
            .collect::<Vec<_>>();
        
        let coverage_growth_rate = if recent_snapshots.len() >= 2 {
            let recent = recent_snapshots[0].total_branches as f64;
            let older = recent_snapshots[recent_snapshots.len() - 1].total_branches as f64;
            (recent - older) / recent_snapshots.len() as f64
        } else {
            0.0
        };
        
        CoverageStats {
            total_branches: self.covered_branches.len(),
            total_hits: self.branch_counts.values().sum(),
            interesting_regions_count: self.interesting_regions.len(),
            coverage_growth_rate,
            last_updated: self.last_updated,
        }
    }
}

impl Default for CoverageState {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about coverage state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageStats {
    pub total_branches: usize,
    pub total_hits: u32,
    pub interesting_regions_count: usize,
    pub coverage_growth_rate: f64,
    pub last_updated: SystemTime,
}

/// Suggestions for guided test case generation
#[derive(Debug, Clone)]
pub struct TargetingSuggestions {
    /// Test cases from Pareto frontier worth exploring
    pub pareto_suggestions: Vec<Vec<u8>>,
    /// Test cases from coverage-guided generation
    pub coverage_suggestions: Vec<Vec<u8>>,
    /// Choice modifications to improve targeting
    pub choice_modifications: Vec<ChoiceModification>,
    /// Priority score for each suggestion (higher = better)
    pub priorities: Vec<f64>,
}

/// Suggested modification to choice generation
#[derive(Debug, Clone)]
pub struct ChoiceModification {
    /// Index of choice to modify
    pub choice_index: usize,
    /// Suggested new value
    pub suggested_value: ChoiceValue,
    /// Reason for this modification
    pub reason: String,
    /// Expected improvement score
    pub expected_improvement: f64,
}

/// Core targeting engine with Pareto optimization and coverage guidance
pub struct TargetingEngine {
    /// Registered target functions
    target_functions: Vec<Box<dyn TargetFunction>>,
    /// Current Pareto frontier
    pareto_frontier: Arc<RwLock<Vec<ParetoPoint>>>,
    /// Coverage tracking state
    coverage_state: Arc<Mutex<CoverageState>>,
    /// Target observations cache
    observations_cache: Arc<RwLock<HashMap<String, Vec<TargetObservation>>>>,
    /// Configuration for targeting behavior
    config: TargetingConfig,
    /// Performance metrics
    metrics: TargetingMetrics,
}

/// Configuration for targeting engine behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetingConfig {
    /// Maximum size of Pareto frontier to maintain
    pub max_pareto_size: usize,
    /// Maximum number of coverage regions to track
    pub max_coverage_regions: usize,
    /// Minimum improvement threshold for adding to Pareto frontier
    pub min_improvement_threshold: f64,
    /// Enable coverage-guided generation
    pub enable_coverage_guidance: bool,
    /// Update frequency for targeting suggestions
    pub suggestion_update_interval: Duration,
}

impl Default for TargetingConfig {
    fn default() -> Self {
        TargetingConfig {
            max_pareto_size: 1000,
            max_coverage_regions: 10000,
            min_improvement_threshold: 1e-6,
            enable_coverage_guidance: true,
            suggestion_update_interval: Duration::from_millis(100),
        }
    }
}

/// Performance metrics for the targeting engine
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TargetingMetrics {
    pub total_evaluations: u64,
    pub pareto_updates: u64,
    pub coverage_updates: u64,
    pub suggestions_generated: u64,
    pub total_runtime_ms: u64,
    pub avg_evaluation_time_ms: f64,
}

impl TargetingEngine {
    /// Create a new targeting engine
    pub fn new() -> Self {
        TargetingEngine::with_config(TargetingConfig::default())
    }
    
    /// Create a new targeting engine with custom configuration
    pub fn with_config(config: TargetingConfig) -> Self {
        TargetingEngine {
            target_functions: Vec::new(),
            pareto_frontier: Arc::new(RwLock::new(Vec::new())),
            coverage_state: Arc::new(Mutex::new(CoverageState::new())),
            observations_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            metrics: TargetingMetrics::default(),
        }
    }
    
    /// Add a target function to the engine
    pub fn add_target(&mut self, target: Box<dyn TargetFunction>) -> TargetingResult<()> {
        self.target_functions.push(target);
        Ok(())
    }
    
    /// Update the targeting engine with a new test result
    pub fn update_with_result(&mut self, result: &ConjectureResult) -> TargetingResult<()> {
        let start_time = SystemTime::now();
        
        // Evaluate all target functions
        let mut objectives = Vec::new();
        let mut observations = Vec::new();
        
        // Create ConjectureData from result for evaluation
        let test_data = ConjectureData::from_choices(&result.nodes, 12345);
        
        for target in &self.target_functions {
            if target.is_applicable(&test_data) {
                let value = target.evaluate(&test_data)?;
                objectives.push(value);
                
                let observation = TargetObservation::new(value, target.label().to_string());
                observations.push(observation);
            }
        }
        
        // Update Pareto frontier
        if !objectives.is_empty() {
            self.update_pareto_frontier(objectives, &result.buffer, observations, &result.nodes)?;
        }
        
        // Update coverage if enabled
        if self.config.enable_coverage_guidance {
            self.update_coverage(&result.buffer)?;
        }
        
        // Update metrics
        self.metrics.total_evaluations += 1;
        if let Ok(duration) = start_time.elapsed() {
            let eval_time = duration.as_millis() as f64;
            self.metrics.total_runtime_ms += eval_time as u64;
            self.metrics.avg_evaluation_time_ms = 
                (self.metrics.avg_evaluation_time_ms * (self.metrics.total_evaluations - 1) as f64 + eval_time) 
                / self.metrics.total_evaluations as f64;
        }
        
        Ok(())
    }
    
    /// Update the Pareto frontier with new objectives
    fn update_pareto_frontier(
        &mut self,
        objectives: Vec<f64>,
        test_case: &[u8],
        observations: Vec<TargetObservation>,
        choices: &[ChoiceNode],
    ) -> TargetingResult<()> {
        let directions: Vec<OptimizationDirection> = self.target_functions
            .iter()
            .map(|t| t.direction())
            .collect();
        
        let new_point = ParetoPoint::new(
            objectives,
            test_case.to_vec(),
            observations,
            choices.to_vec(),
        );
        
        let mut frontier = self.pareto_frontier.write()
            .map_err(|_| TargetingError::ParetoError("Failed to acquire write lock".to_string()))?;
        
        // Check if new point is dominated by existing points
        let is_dominated = frontier.iter().any(|point| point.dominates(&new_point, &directions));
        
        if !is_dominated {
            // Remove points dominated by new point
            frontier.retain(|point| !new_point.dominates(point, &directions));
            
            // Add new point
            frontier.push(new_point);
            
            // Maintain maximum frontier size
            if frontier.len() > self.config.max_pareto_size {
                // Sort by quality score and keep the best
                frontier.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap_or(Ordering::Equal));
                frontier.truncate(self.config.max_pareto_size);
            }
            
            self.metrics.pareto_updates += 1;
        }
        
        Ok(())
    }
    
    /// Update coverage information
    fn update_coverage(&mut self, test_case: &[u8]) -> TargetingResult<()> {
        // Extract branch information from test case
        // This is a simplified implementation - in practice, this would integrate
        // with actual coverage instrumentation
        let branches = self.extract_coverage_branches(test_case);
        
        let mut coverage = self.coverage_state.lock()
            .map_err(|_| TargetingError::CoverageError("Failed to acquire coverage lock".to_string()))?;
        
        let new_coverage = coverage.update_coverage(&branches, test_case);
        
        if new_coverage {
            self.metrics.coverage_updates += 1;
        }
        
        Ok(())
    }
    
    /// Extract coverage branch information from test case
    fn extract_coverage_branches(&self, test_case: &[u8]) -> Vec<String> {
        // Simplified branch extraction based on test case content
        // In a real implementation, this would integrate with coverage instrumentation
        let mut branches = Vec::new();
        
        for (i, &byte) in test_case.iter().enumerate() {
            if i % 8 == 0 {
                branches.push(format!("BRANCH_0x{:04X}_{:02X}", i, byte));
            }
            
            if byte > 128 {
                branches.push(format!("HIGH_VALUE_BRANCH_{}", i));
            }
            
            if byte == 0 {
                branches.push(format!("ZERO_BRANCH_{}", i));
            }
        }
        
        // Add length-based branches
        match test_case.len() {
            0..=10 => branches.push("SHORT_TEST".to_string()),
            11..=100 => branches.push("MEDIUM_TEST".to_string()),
            _ => branches.push("LONG_TEST".to_string()),
        }
        
        branches
    }
    
    /// Generate targeting suggestions for guided test case generation
    pub fn get_targeting_suggestions(&mut self) -> TargetingResult<TargetingSuggestions> {
        let mut pareto_suggestions = Vec::new();
        let mut coverage_suggestions = Vec::new();
        let mut choice_modifications = Vec::new();
        let mut priorities = Vec::new();
        
        // Get suggestions from Pareto frontier
        if let Ok(frontier) = self.pareto_frontier.read() {
            for point in frontier.iter().take(10) {
                pareto_suggestions.push(point.test_case.clone());
                priorities.push(point.quality_score);
            }
        }
        
        // Get suggestions from coverage state
        if self.config.enable_coverage_guidance {
            if let Ok(coverage) = self.coverage_state.lock() {
                for region in coverage.interesting_regions.iter().rev().take(5) {
                    coverage_suggestions.push(region.clone());
                    priorities.push(75.0); // Fixed priority for coverage suggestions
                }
            }
        }
        
        // Generate choice modifications based on Pareto frontier
        if let Ok(frontier) = self.pareto_frontier.read() {
            for point in frontier.iter().take(3) {
                for (i, choice) in point.choice_sequence.iter().enumerate() {
                    if i < 5 { // Limit to first 5 choices
                        let modification = self.suggest_choice_modification(i, choice, point);
                        choice_modifications.push(modification);
                        priorities.push(50.0 + point.quality_score * 0.1);
                    }
                }
            }
        }
        
        self.metrics.suggestions_generated += 1;
        
        Ok(TargetingSuggestions {
            pareto_suggestions,
            coverage_suggestions,
            choice_modifications,
            priorities,
        })
    }
    
    /// Suggest a modification to a choice for improved targeting
    fn suggest_choice_modification(&self, index: usize, choice: &ChoiceNode, point: &ParetoPoint) -> ChoiceModification {
        let suggested_value = match &choice.value {
            ChoiceValue::Integer(val) => {
                // Suggest slight perturbation
                let delta = if point.quality_score > 100.0 { 1 } else { 5 };
                ChoiceValue::Integer(val + delta)
            }
            
            ChoiceValue::Float(val) => {
                // Suggest slight perturbation
                let delta = if point.quality_score > 100.0 { 0.1 } else { 1.0 };
                ChoiceValue::Float(val + delta)
            }
            
            ChoiceValue::Boolean(val) => {
                // Suggest opposite value
                ChoiceValue::Boolean(!val)
            }
            
            ChoiceValue::String(val) => {
                // Suggest string with modified length
                if val.is_empty() {
                    ChoiceValue::String("x".to_string())
                } else {
                    ChoiceValue::String(val[..val.len().saturating_sub(1)].to_string())
                }
            }
            
            ChoiceValue::Bytes(val) => {
                // Suggest bytes with modified length
                if val.is_empty() {
                    ChoiceValue::Bytes(vec![42])
                } else {
                    ChoiceValue::Bytes(val[..val.len().saturating_sub(1)].to_vec())
                }
            }
        };
        
        ChoiceModification {
            choice_index: index,
            suggested_value,
            reason: format!("Perturbation of choice {} for improved targeting", index),
            expected_improvement: point.quality_score * 0.1,
        }
    }
    
    /// Get current Pareto frontier
    pub fn get_pareto_frontier(&self) -> TargetingResult<Vec<ParetoPoint>> {
        self.pareto_frontier.read()
            .map(|frontier| frontier.clone())
            .map_err(|_| TargetingError::ParetoError("Failed to read Pareto frontier".to_string()))
    }
    
    /// Get current coverage statistics
    pub fn get_coverage_stats(&self) -> TargetingResult<CoverageStats> {
        self.coverage_state.lock()
            .map(|coverage| coverage.get_stats())
            .map_err(|_| TargetingError::CoverageError("Failed to read coverage state".to_string()))
    }
    
    /// Get targeting engine metrics
    pub fn get_metrics(&self) -> &TargetingMetrics {
        &self.metrics
    }
    
    /// Clear all targeting data
    pub fn clear(&mut self) -> TargetingResult<()> {
        self.pareto_frontier.write()
            .map_err(|_| TargetingError::ParetoError("Failed to acquire write lock".to_string()))?
            .clear();
        
        self.coverage_state.lock()
            .map_err(|_| TargetingError::CoverageError("Failed to acquire coverage lock".to_string()))?
            .covered_branches.clear();
        
        self.observations_cache.write()
            .map_err(|_| TargetingError::ObservationError("Failed to acquire cache lock".to_string()))?
            .clear();
        
        self.metrics = TargetingMetrics::default();
        
        Ok(())
    }
}

impl Default for TargetingEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Built-in target function for minimizing a value
pub struct MinimizeFunction {
    label: String,
}

impl MinimizeFunction {
    pub fn new(label: &str) -> Self {
        MinimizeFunction {
            label: label.to_string(),
        }
    }
}

impl TargetFunction for MinimizeFunction {
    fn evaluate(&self, data: &ConjectureData) -> TargetingResult<f64> {
        // Simple evaluation: minimize the sum of choice values
        let mut sum = 0.0;
        
        for choice in data.choices() {
            match &choice.value {
                ChoiceValue::Integer(val) => sum += *val as f64,
                ChoiceValue::Float(val) => sum += val,
                ChoiceValue::Boolean(val) => sum += if *val { 1.0 } else { 0.0 },
                ChoiceValue::String(val) => sum += val.len() as f64,
                ChoiceValue::Bytes(val) => sum += val.len() as f64,
            }
        }
        
        Ok(sum)
    }
    
    fn label(&self) -> &str {
        &self.label
    }
    
    fn direction(&self) -> OptimizationDirection {
        OptimizationDirection::Minimize
    }
}

/// Built-in target function for maximizing a value
pub struct MaximizeFunction {
    label: String,
}

impl MaximizeFunction {
    pub fn new(label: &str) -> Self {
        MaximizeFunction {
            label: label.to_string(),
        }
    }
}

impl TargetFunction for MaximizeFunction {
    fn evaluate(&self, data: &ConjectureData) -> TargetingResult<f64> {
        // Simple evaluation: maximize the product of non-zero choice values
        let mut product = 1.0;
        
        for choice in data.choices() {
            match &choice.value {
                ChoiceValue::Integer(val) => {
                    if *val != 0 {
                        product *= (*val as f64).abs();
                    }
                }
                ChoiceValue::Float(val) => {
                    if *val != 0.0 {
                        product *= val.abs();
                    }
                }
                ChoiceValue::Boolean(val) => {
                    if *val {
                        product *= 2.0;
                    }
                }
                ChoiceValue::String(val) => {
                    if !val.is_empty() {
                        product *= val.len() as f64;
                    }
                }
                ChoiceValue::Bytes(val) => {
                    if !val.is_empty() {
                        product *= val.len() as f64;
                    }
                }
            }
        }
        
        Ok(product)
    }
    
    fn label(&self) -> &str {
        &self.label
    }
    
    fn direction(&self) -> OptimizationDirection {
        OptimizationDirection::Maximize
    }
}

/// Target function for maximizing test case complexity
pub struct ComplexityFunction {
    label: String,
}

impl ComplexityFunction {
    pub fn new() -> Self {
        ComplexityFunction {
            label: "complexity".to_string(),
        }
    }
}

impl TargetFunction for ComplexityFunction {
    fn evaluate(&self, data: &ConjectureData) -> TargetingResult<f64> {
        let mut complexity = 0.0;
        
        // Count different choice types
        let mut type_counts = HashMap::new();
        for choice in data.choices() {
            *type_counts.entry(choice.choice_type).or_insert(0u32) += 1;
        }
        
        // Complexity based on diversity of choice types
        complexity += type_counts.len() as f64 * 10.0;
        
        // Complexity based on total number of choices
        complexity += data.choices().len() as f64;
        
        // Complexity based on value ranges
        for choice in data.choices() {
            match &choice.value {
                ChoiceValue::Integer(val) => complexity += (*val as f64).abs().ln_1p(),
                ChoiceValue::Float(val) => complexity += val.abs().ln_1p(),
                ChoiceValue::String(val) => complexity += (val.len() as f64).sqrt(),
                ChoiceValue::Bytes(val) => complexity += (val.len() as f64).sqrt(),
                ChoiceValue::Boolean(_) => complexity += 1.0,
            }
        }
        
        Ok(complexity)
    }
    
    fn label(&self) -> &str {
        &self.label
    }
    
    fn direction(&self) -> OptimizationDirection {
        OptimizationDirection::Maximize
    }
}

impl Default for ComplexityFunction {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::{ChoiceType, Constraints, BooleanConstraints, IntegerConstraints};
    use crate::data::{Status, Example};
    
    #[test]
    fn test_target_observation() {
        let obs = TargetObservation::new(42.0, "test_label".to_string())
            .with_metadata("key1".to_string(), "value1".to_string());
        
        assert_eq!(obs.value, 42.0);
        assert_eq!(obs.label, "test_label");
        assert_eq!(obs.metadata.get("key1"), Some(&"value1".to_string()));
    }
    
    #[test]
    fn test_pareto_dominance() {
        let point1 = ParetoPoint::new(
            vec![1.0, 2.0],
            vec![1, 2, 3],
            vec![],
            vec![],
        );
        
        let point2 = ParetoPoint::new(
            vec![2.0, 1.0],
            vec![4, 5, 6],
            vec![],
            vec![],
        );
        
        let directions = vec![OptimizationDirection::Minimize, OptimizationDirection::Minimize];
        
        // Neither should dominate the other (trade-off between objectives)
        assert!(!point1.dominates(&point2, &directions));
        assert!(!point2.dominates(&point1, &directions));
    }
    
    #[test]
    fn test_pareto_dominance_clear_winner() {
        let point1 = ParetoPoint::new(
            vec![1.0, 1.0],
            vec![1, 2, 3],
            vec![],
            vec![],
        );
        
        let point2 = ParetoPoint::new(
            vec![2.0, 2.0],
            vec![4, 5, 6],
            vec![],
            vec![],
        );
        
        let directions = vec![OptimizationDirection::Minimize, OptimizationDirection::Minimize];
        
        // Point1 should dominate point2 (better in all objectives)
        assert!(point1.dominates(&point2, &directions));
        assert!(!point2.dominates(&point1, &directions));
    }
    
    #[test]
    fn test_coverage_state() {
        let mut coverage = CoverageState::new();
        
        let branches1 = vec!["branch1".to_string(), "branch2".to_string()];
        let test_case1 = vec![1, 2, 3];
        
        let new_coverage = coverage.update_coverage(&branches1, &test_case1);
        assert!(new_coverage);
        assert_eq!(coverage.covered_branches.len(), 2);
        
        // Same branches again should not be new coverage
        let new_coverage = coverage.update_coverage(&branches1, &test_case1);
        assert!(!new_coverage);
        
        // New branch should be new coverage
        let branches2 = vec!["branch3".to_string()];
        let test_case2 = vec![4, 5, 6];
        let new_coverage = coverage.update_coverage(&branches2, &test_case2);
        assert!(new_coverage);
        assert_eq!(coverage.covered_branches.len(), 3);
    }
    
    #[test]
    fn test_targeting_engine() {
        let mut engine = TargetingEngine::new();
        
        // Add target functions
        engine.add_target(Box::new(MinimizeFunction::new("minimize_test"))).unwrap();
        engine.add_target(Box::new(MaximizeFunction::new("maximize_test"))).unwrap();
        
        // Create test result
        let choices = vec![
            ChoiceNode {
                choice_type: ChoiceType::Boolean,
                value: ChoiceValue::Boolean(true),
                constraints: Constraints::Boolean(BooleanConstraints::default()),
                was_forced: false,
                index: Some(0),
            }
        ];
        
        let result = ConjectureResult {
            status: Status::Valid,
            nodes: choices,
            length: 3,
            events: HashMap::new(),
            buffer: vec![1, 2, 3],
            examples: Vec::new(),
            interesting_origin: None,
            output: Vec::new(),
            extra_information: ExtraInformation::new(),
            expected_exception: None,
            expected_traceback: None,
            has_discards: false,
            target_observations: HashMap::new(),
            tags: HashSet::new(),
            spans: Vec::new(),
            arg_slices: Vec::new(),
            slice_comments: HashMap::new(),
            misaligned_at: None,
            cannot_proceed_scope: None,
        };
        
        // Update engine with result
        engine.update_with_result(&result).unwrap();
        
        // Check that metrics were updated
        assert_eq!(engine.get_metrics().total_evaluations, 1);
        
        // Get targeting suggestions
        let suggestions = engine.get_targeting_suggestions().unwrap();
        assert!(!suggestions.pareto_suggestions.is_empty() || !suggestions.coverage_suggestions.is_empty());
    }
    
    #[test]
    fn test_minimize_function() {
        let func = MinimizeFunction::new("test_minimize");
        
        let choices = vec![
            ChoiceNode {
                choice_type: ChoiceType::Boolean,
                value: ChoiceValue::Boolean(true),
                constraints: Constraints::Boolean(BooleanConstraints::default()),
                was_forced: false,
                index: Some(0),
            }
        ];
        
        let data = ConjectureData::from_choices(&choices, 12345);
        let result = func.evaluate(&data).unwrap();
        
        assert_eq!(result, 1.0); // Boolean true = 1.0
        assert_eq!(func.direction(), OptimizationDirection::Minimize);
    }
    
    #[test]
    fn test_maximize_function() {
        let func = MaximizeFunction::new("test_maximize");
        
        let choices = vec![
            ChoiceNode {
                choice_type: ChoiceType::Boolean,
                value: ChoiceValue::Boolean(true),
                constraints: Constraints::Boolean(BooleanConstraints::default()),
                was_forced: false,
                index: Some(0),
            }
        ];
        
        let data = ConjectureData::from_choices(&choices, 12345);
        let result = func.evaluate(&data).unwrap();
        
        assert_eq!(result, 2.0); // Boolean true = 2.0 for maximize
        assert_eq!(func.direction(), OptimizationDirection::Maximize);
    }
    
    #[test]
    fn test_complexity_function() {
        let func = ComplexityFunction::new();
        
        let choices = vec![
            ChoiceNode {
                choice_type: ChoiceType::Boolean,
                value: ChoiceValue::Boolean(true),
                constraints: Constraints::Boolean(BooleanConstraints::default()),
                was_forced: false,
                index: Some(0),
            },
            ChoiceNode {
                choice_type: ChoiceType::Integer,
                value: ChoiceValue::Integer(42),
                constraints: Constraints::Integer(crate::choice::IntegerConstraints::default()),
                was_forced: false,
                index: Some(1),
            },
        ];
        
        let data = ConjectureData::from_choices(&choices, 12345);
        let result = func.evaluate(&data).unwrap();
        
        // Should have complexity from 2 different choice types + 2 choices + value complexity
        assert!(result > 20.0); // 2 types * 10 + 2 choices + value complexity
        assert_eq!(func.direction(), OptimizationDirection::Maximize);
    }
}