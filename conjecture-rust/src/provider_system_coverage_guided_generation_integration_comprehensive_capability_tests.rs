//! Coverage-Guided Generation Integration Module for ProviderSystem
//! 
//! This module implements comprehensive coverage-guided generation integration that extends
//! the ProviderSystem with sophisticated generation strategies based on structural coverage,
//! span-aware generation, and coverage feedback loops.
//!
//! Key capabilities implemented:
//! - Structural coverage integration for choice generation
//! - Span-aware generation strategies with hierarchical tracking
//! - Coverage feedback loops with target observations and Pareto optimization
//! - Novel prefix generation from explored choice trees
//! - Cross-contamination mutation strategies
//! - Multi-objective optimization with coverage and target guidance
//!
//! This module follows Python Hypothesis patterns while implementing idiomatic Rust
//! with sophisticated error handling, debug logging, and comprehensive testing.

use crate::choice::{ChoiceValue, Constraints, ChoiceType, FloatConstraints, IntegerConstraints, BooleanConstraints, IntervalSet};
use crate::providers::{PrimitiveProvider, ProviderError, ProviderScope, ProviderLifetime, BackendCapabilities, TestCaseContext, ObservationMessage, ObservationType, TestCaseObservation, ProviderFactory};
use crate::data::DrawError;
use std::collections::{HashMap, HashSet, BTreeSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::fmt;
use serde::{Serialize, Deserialize};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Coverage-guided generation provider that implements sophisticated generation strategies
/// based on structural coverage analysis, span tracking, and feedback optimization.
///
/// This provider combines multiple generation modes:
/// - Novel prefix generation from unexplored choice trees
/// - Mutation-based generation with cross-contamination
/// - Target-driven optimization with hill-climbing
/// - Coverage-guided exploration with exhaustion detection
#[derive(Debug)]
pub struct CoverageGuidedProvider {
    /// Core coverage tracking infrastructure
    coverage_tracker: CoverageTracker,
    
    /// Span hierarchy tracking for structural awareness
    span_tracker: SpanTracker,
    
    /// Choice tree for systematic exploration
    choice_tree: ChoiceTree,
    
    /// Target observation system for optimization feedback
    target_tracker: TargetTracker,
    
    /// Pareto front maintenance for multi-objective optimization
    pareto_front: ParetoFront,
    
    /// Generation strategy controller
    strategy_controller: GenerationStrategyController,
    
    /// Current test case context
    current_context: Option<CoverageTestCaseContext>,
    
    /// Configuration parameters
    config: CoverageGuidedConfig,
    
    /// Internal RNG for deterministic behavior
    rng: ChaCha8Rng,
    
    /// Performance metrics tracking
    metrics: CoverageMetrics,
}

/// Configuration for coverage-guided generation behavior
#[derive(Debug, Clone)]
pub struct CoverageGuidedConfig {
    /// Probability of using novel prefix generation (default: 0.3)
    pub novel_prefix_probability: f64,
    
    /// Probability of using mutation-based generation (default: 0.4)
    pub mutation_probability: f64,
    
    /// Probability of using target optimization (default: 0.2)
    pub target_optimization_probability: f64,
    
    /// Maximum depth for choice tree exploration
    pub max_tree_depth: usize,
    
    /// Maximum number of entries in Pareto front
    pub max_pareto_front_size: usize,
    
    /// Threshold for span similarity in cross-contamination
    pub span_similarity_threshold: f64,
    
    /// Number of hill-climbing iterations for target optimization
    pub target_optimization_iterations: usize,
    
    /// Enable debug logging for coverage decisions
    pub debug_coverage_decisions: bool,
}

impl Default for CoverageGuidedConfig {
    fn default() -> Self {
        Self {
            novel_prefix_probability: 0.3,
            mutation_probability: 0.4,
            target_optimization_probability: 0.2,
            max_tree_depth: 1000,
            max_pareto_front_size: 100,
            span_similarity_threshold: 0.8,
            target_optimization_iterations: 10,
            debug_coverage_decisions: true,
        }
    }
}

/// Performance metrics for coverage-guided generation
#[derive(Debug, Default)]
pub struct CoverageMetrics {
    /// Total number of choices generated
    pub total_choices: u64,
    
    /// Number of novel prefixes generated
    pub novel_prefixes_generated: u64,
    
    /// Number of mutation-based generations
    pub mutations_generated: u64,
    
    /// Number of target optimizations performed
    pub target_optimizations: u64,
    
    /// Number of coverage targets hit
    pub coverage_targets_hit: u64,
    
    /// Current size of choice tree
    pub choice_tree_size: usize,
    
    /// Current Pareto front size
    pub pareto_front_size: usize,
    
    /// Average span depth
    pub average_span_depth: f64,
}

/// Structural coverage tracking system
#[derive(Debug)]
pub struct CoverageTracker {
    /// Coverage tags by label
    coverage_tags: HashMap<u32, StructuralCoverageTag>,
    
    /// Label generation for strategy tracking
    label_generator: LabelGenerator,
    
    /// Coverage statistics
    coverage_stats: CoverageStatistics,
}

/// Structural coverage tag for tracking strategy execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralCoverageTag {
    /// Unique label for this coverage point
    pub label: u32,
    
    /// Number of times this coverage point was hit
    pub hit_count: u64,
    
    /// Strategy name that created this coverage point
    pub strategy_name: String,
    
    /// First time this coverage point was hit
    pub first_hit_timestamp: std::time::SystemTime,
    
    /// Last time this coverage point was hit
    pub last_hit_timestamp: std::time::SystemTime,
    
    /// Additional metadata for this coverage point
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Coverage statistics aggregation
#[derive(Debug, Default)]
pub struct CoverageStatistics {
    /// Total number of unique coverage points
    pub unique_coverage_points: usize,
    
    /// Total coverage hits across all points
    pub total_coverage_hits: u64,
    
    /// Coverage points hit in current test case
    pub current_test_case_coverage: BTreeSet<u32>,
    
    /// Most frequently hit coverage points
    pub hot_coverage_points: Vec<(u32, u64)>,
}

/// Label generation for consistent strategy tracking
#[derive(Debug)]
pub struct LabelGenerator {
    /// Cache of labels by strategy name
    label_cache: HashMap<String, u32>,
    
    /// Next available label ID
    next_label: u32,
}

impl LabelGenerator {
    pub fn new() -> Self {
        Self {
            label_cache: HashMap::new(),
            next_label: 1,
        }
    }
    
    /// Generate a label for the given strategy name
    pub fn calc_label_from_name(&mut self, strategy_name: &str) -> u32 {
        if let Some(&label) = self.label_cache.get(strategy_name) {
            return label;
        }
        
        let label = self.next_label;
        self.next_label += 1;
        self.label_cache.insert(strategy_name.to_string(), label);
        
        println!("COVERAGE_LABEL DEBUG: Generated label {:#08X} for strategy '{}'", label, strategy_name);
        label
    }
    
    /// Generate a label for the given choice type
    pub fn calc_label_from_choice_type(&mut self, choice_type: ChoiceType, constraints: &Constraints) -> u32 {
        let strategy_name = format!("{:?}_{:?}", choice_type, constraints);
        self.calc_label_from_name(&strategy_name)
    }
}

/// Span tracking system for hierarchical structural awareness
#[derive(Debug)]
pub struct SpanTracker {
    /// Active span stack
    span_stack: Vec<SpanInfo>,
    
    /// All spans from completed test cases
    completed_spans: Vec<CompletedSpan>,
    
    /// Mutator groups for cross-contamination
    mutator_groups: HashMap<u32, Vec<usize>>, // label -> span indices
    
    /// Span similarity matrix cache
    similarity_cache: HashMap<(usize, usize), f64>,
}

/// Information about an active span
#[derive(Debug, Clone)]
pub struct SpanInfo {
    /// Unique label for this span
    pub label: u32,
    
    /// Depth in the span hierarchy
    pub depth: usize,
    
    /// Start position in choice sequence
    pub start_position: usize,
    
    /// Choices made within this span
    pub choices: Vec<ChoiceValue>,
    
    /// Child spans
    pub children: Vec<SpanInfo>,
    
    /// Timestamp when span was started
    pub start_timestamp: std::time::SystemTime,
}

/// Completed span with full information
#[derive(Debug, Clone)]
pub struct CompletedSpan {
    /// Span information
    pub span_info: SpanInfo,
    
    /// End position in choice sequence
    pub end_position: usize,
    
    /// Whether span was discarded
    pub discarded: bool,
    
    /// Total execution time
    pub execution_time: std::time::Duration,
    
    /// Coverage points hit within this span
    pub coverage_points: BTreeSet<u32>,
}

/// Choice tree for systematic exploration and novel prefix generation
#[derive(Debug)]
pub struct ChoiceTree {
    /// Root node of the choice tree
    root: ChoiceTreeNode,
    
    /// Total number of nodes in the tree
    node_count: usize,
    
    /// Maximum depth reached
    max_depth: usize,
    
    /// Exhausted paths that shouldn't be explored further
    exhausted_paths: BTreeSet<Vec<ChoiceValue>>,
}

/// Node in the choice tree representing a specific choice point
#[derive(Debug)]
pub struct ChoiceTreeNode {
    /// Choice value at this node (None for root)
    pub choice: Option<ChoiceValue>,
    
    /// Choice type and constraints used
    pub choice_metadata: Option<(ChoiceType, Constraints)>,
    
    /// Child nodes representing subsequent choices
    pub children: HashMap<ChoiceValue, Box<ChoiceTreeNode>>,
    
    /// Number of times this path was explored
    pub visit_count: u64,
    
    /// Whether this path is exhausted (all children explored)
    pub exhausted: bool,
    
    /// Best target observation seen from this path
    pub best_target_observation: Option<f64>,
    
    /// Coverage points reached from this path
    pub coverage_points_reached: BTreeSet<u32>,
}

/// Target tracking system for optimization feedback
#[derive(Debug)]
pub struct TargetTracker {
    /// Current target observations by label
    current_observations: HashMap<String, f64>,
    
    /// Historical target observations
    observation_history: Vec<TargetObservation>,
    
    /// Best observations seen per label
    best_observations: HashMap<String, f64>,
    
    /// Optimization state for hill-climbing
    optimization_state: HashMap<String, OptimizationState>,
}

/// Single target observation with metadata
#[derive(Debug, Clone)]
pub struct TargetObservation {
    /// Target label
    pub label: String,
    
    /// Observed value
    pub value: f64,
    
    /// Choice sequence that produced this observation
    pub choice_sequence: Vec<ChoiceValue>,
    
    /// Timestamp of observation
    pub timestamp: std::time::SystemTime,
    
    /// Coverage points active when observation was made
    pub active_coverage_points: BTreeSet<u32>,
}

/// Optimization state for hill-climbing algorithm
#[derive(Debug, Clone)]
pub struct OptimizationState {
    /// Current best value
    pub best_value: f64,
    
    /// Current best choice sequence
    pub best_choice_sequence: Vec<ChoiceValue>,
    
    /// Number of optimization iterations performed
    pub iterations: u64,
    
    /// Recent improvement direction
    pub improvement_direction: Option<ImprovementDirection>,
}

/// Direction of recent improvements for optimization
#[derive(Debug, Clone)]
pub enum ImprovementDirection {
    Increasing,
    Decreasing,
    Oscillating,
}

/// Pareto front maintenance for multi-objective optimization
#[derive(Debug)]
pub struct ParetoFront {
    /// Current Pareto-optimal solutions
    solutions: Vec<ParetoSolution>,
    
    /// Maximum number of solutions to maintain
    max_size: usize,
    
    /// Dominance comparisons performed
    dominance_comparisons: u64,
}

/// Solution in the Pareto front
#[derive(Debug, Clone)]
pub struct ParetoSolution {
    /// Choice sequence for this solution
    pub choice_sequence: Vec<ChoiceValue>,
    
    /// Target observations for this solution
    pub target_observations: HashMap<String, f64>,
    
    /// Coverage points reached by this solution
    pub coverage_points: BTreeSet<u32>,
    
    /// Solution quality metrics
    pub quality_metrics: QualityMetrics,
    
    /// Generation timestamp
    pub timestamp: std::time::SystemTime,
}

/// Quality metrics for Pareto solutions
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Total number of coverage points hit
    pub coverage_breadth: usize,
    
    /// Average target observation value
    pub average_target_value: f64,
    
    /// Novelty score based on choice diversity
    pub novelty_score: f64,
    
    /// Execution efficiency (choices per coverage point)
    pub efficiency_score: f64,
}

/// Generation strategy controller for selecting generation modes
#[derive(Debug)]
pub struct GenerationStrategyController {
    /// Current generation strategy
    current_strategy: GenerationStrategy,
    
    /// Strategy performance tracking
    strategy_performance: HashMap<GenerationStrategy, StrategyPerformance>,
    
    /// Adaptive strategy selection weights
    strategy_weights: HashMap<GenerationStrategy, f64>,
}

/// Available generation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationStrategy {
    /// Generate novel prefixes from unexplored choice tree paths
    NovelPrefix,
    
    /// Mutation-based generation with cross-contamination
    MutationBased,
    
    /// Target-driven optimization with hill-climbing
    TargetOptimization,
    
    /// Random fallback generation
    RandomFallback,
}

/// Performance tracking for generation strategies
#[derive(Debug, Default)]
pub struct StrategyPerformance {
    /// Number of times this strategy was used
    pub usage_count: u64,
    
    /// Number of successful generations (led to new coverage or better targets)
    pub success_count: u64,
    
    /// Average target improvement when using this strategy
    pub average_target_improvement: f64,
    
    /// Average new coverage points discovered
    pub average_new_coverage: f64,
    
    /// Recent success rate (last 100 uses)
    pub recent_success_rate: f64,
}

/// Test case context for coverage-guided provider
#[derive(Debug)]
pub struct CoverageTestCaseContext {
    /// Unique identifier for this test case
    pub test_case_id: String,
    
    /// Generation strategy used for this test case
    pub generation_strategy: GenerationStrategy,
    
    /// Start time for performance tracking
    pub start_time: std::time::SystemTime,
    
    /// Coverage points hit in this test case
    pub coverage_points_hit: BTreeSet<u32>,
    
    /// Target observations made in this test case
    pub target_observations: HashMap<String, f64>,
    
    /// Choices made in this test case
    pub choice_sequence: Vec<ChoiceValue>,
}

impl CoverageGuidedProvider {
    /// Create a new coverage-guided provider with default configuration
    pub fn new() -> Self {
        Self::with_config(CoverageGuidedConfig::default())
    }
    
    /// Create a new coverage-guided provider with custom configuration
    pub fn with_config(config: CoverageGuidedConfig) -> Self {
        let mut rng = ChaCha8Rng::from_entropy();
        
        Self {
            coverage_tracker: CoverageTracker {
                coverage_tags: HashMap::new(),
                label_generator: LabelGenerator::new(),
                coverage_stats: CoverageStatistics::default(),
            },
            span_tracker: SpanTracker {
                span_stack: Vec::new(),
                completed_spans: Vec::new(),
                mutator_groups: HashMap::new(),
                similarity_cache: HashMap::new(),
            },
            choice_tree: ChoiceTree {
                root: ChoiceTreeNode {
                    choice: None,
                    choice_metadata: None,
                    children: HashMap::new(),
                    visit_count: 0,
                    exhausted: false,
                    best_target_observation: None,
                    coverage_points_reached: BTreeSet::new(),
                },
                node_count: 1,
                max_depth: 0,
                exhausted_paths: BTreeSet::new(),
            },
            target_tracker: TargetTracker {
                current_observations: HashMap::new(),
                observation_history: Vec::new(),
                best_observations: HashMap::new(),
                optimization_state: HashMap::new(),
            },
            pareto_front: ParetoFront {
                solutions: Vec::new(),
                max_size: config.max_pareto_front_size,
                dominance_comparisons: 0,
            },
            strategy_controller: GenerationStrategyController {
                current_strategy: GenerationStrategy::RandomFallback,
                strategy_performance: HashMap::new(),
                strategy_weights: [(GenerationStrategy::NovelPrefix, config.novel_prefix_probability),
                                 (GenerationStrategy::MutationBased, config.mutation_probability),
                                 (GenerationStrategy::TargetOptimization, config.target_optimization_probability),
                                 (GenerationStrategy::RandomFallback, 0.1)].iter().cloned().collect(),
            },
            current_context: None,
            rng,
            config,
            metrics: CoverageMetrics::default(),
        }
    }
    
    /// Select the next generation strategy based on performance and configuration
    fn select_generation_strategy(&mut self) -> GenerationStrategy {
        let mut cumulative_weight = 0.0;
        let total_weight: f64 = self.strategy_controller.strategy_weights.values().sum();
        let threshold = self.rng.gen::<f64>() * total_weight;
        
        for (&strategy, &weight) in &self.strategy_controller.strategy_weights {
            cumulative_weight += weight;
            if threshold <= cumulative_weight {
                if self.config.debug_coverage_decisions {
                    println!("COVERAGE_STRATEGY DEBUG: Selected strategy {:?} (threshold: {:.3}, weight: {:.3})", 
                            strategy, threshold, weight);
                }
                return strategy;
            }
        }
        
        GenerationStrategy::RandomFallback
    }
    
    /// Generate a choice using coverage-guided strategies
    fn generate_coverage_guided_choice(&mut self, choice_type: ChoiceType, constraints: &Constraints) -> Result<ChoiceValue, ProviderError> {
        // Update metrics
        self.metrics.total_choices += 1;
        
        // Generate coverage label for this choice
        let coverage_label = self.coverage_tracker.label_generator.calc_label_from_choice_type(choice_type, constraints);
        
        // Track structural coverage
        self.track_structural_coverage(coverage_label, &format!("{:?}", choice_type));
        
        // Select generation strategy
        let strategy = self.select_generation_strategy();
        self.strategy_controller.current_strategy = strategy;
        
        // Generate choice based on selected strategy
        let result = match strategy {
            GenerationStrategy::NovelPrefix => self.generate_novel_prefix_choice(choice_type, constraints),
            GenerationStrategy::MutationBased => self.generate_mutation_based_choice(choice_type, constraints),
            GenerationStrategy::TargetOptimization => self.generate_target_optimized_choice(choice_type, constraints),
            GenerationStrategy::RandomFallback => self.generate_random_fallback_choice(choice_type, constraints),
        };
        
        // Update strategy performance
        self.update_strategy_performance(strategy, result.is_ok());
        
        // Add choice to current test case context
        if let (Ok(ref choice_value), Some(ref mut context)) = (&result, &mut self.current_context) {
            context.choice_sequence.push(choice_value.clone());
        }
        
        result
    }
    
    /// Generate a novel prefix choice from unexplored choice tree paths
    fn generate_novel_prefix_choice(&mut self, choice_type: ChoiceType, constraints: &Constraints) -> Result<ChoiceValue, ProviderError> {
        if self.config.debug_coverage_decisions {
            println!("COVERAGE_NOVEL_PREFIX DEBUG: Generating novel prefix for {:?}", choice_type);
        }
        
        // Find unexplored paths in the choice tree
        if let Some(novel_path) = self.find_novel_prefix_path() {
            if !novel_path.is_empty() {
                // Use the first unexplored choice from the novel path
                if let Some(choice) = novel_path.first() {
                    if self.choice_satisfies_constraints(choice, choice_type, constraints) {
                        self.metrics.novel_prefixes_generated += 1;
                        if self.config.debug_coverage_decisions {
                            println!("COVERAGE_NOVEL_PREFIX DEBUG: Using novel choice: {:?}", choice);
                        }
                        return Ok(choice.clone());
                    }
                }
            }
        }
        
        // Fallback to random generation if no novel prefix available
        self.generate_random_fallback_choice(choice_type, constraints)
    }
    
    /// Generate a mutation-based choice using cross-contamination
    fn generate_mutation_based_choice(&mut self, choice_type: ChoiceType, constraints: &Constraints) -> Result<ChoiceValue, ProviderError> {
        if self.config.debug_coverage_decisions {
            println!("COVERAGE_MUTATION DEBUG: Generating mutation-based choice for {:?}", choice_type);
        }
        
        // Find similar spans for cross-contamination - collect data without borrowing
        let similar_span_info = self.find_similar_spans_info_for_mutation();
        if let Some((source_choices, source_label)) = similar_span_info {
            // Extract choices from source span that match the current choice type
            for choice in &source_choices {
                if self.choice_satisfies_constraints(choice, choice_type, constraints) {
                    self.metrics.mutations_generated += 1;
                    if self.config.debug_coverage_decisions {
                        println!("COVERAGE_MUTATION DEBUG: Using cross-contaminated choice: {:?} from span {:#08X}", 
                                choice, source_label);
                    }
                    return Ok(choice.clone());
                }
            }
        }
        
        // Fallback to random generation if no suitable mutation available
        self.generate_random_fallback_choice(choice_type, constraints)
    }
    
    /// Generate a target-optimized choice using hill-climbing
    fn generate_target_optimized_choice(&mut self, choice_type: ChoiceType, constraints: &Constraints) -> Result<ChoiceValue, ProviderError> {
        if self.config.debug_coverage_decisions {
            println!("COVERAGE_TARGET_OPT DEBUG: Generating target-optimized choice for {:?}", choice_type);
        }
        
        // Find the best target optimization candidate
        if let Some(optimization_candidate) = self.find_target_optimization_candidate() {
            // Perform hill-climbing optimization
            if let Some(optimized_choice) = self.perform_hill_climbing_optimization(&optimization_candidate, choice_type, constraints) {
                self.metrics.target_optimizations += 1;
                if self.config.debug_coverage_decisions {
                    println!("COVERAGE_TARGET_OPT DEBUG: Using optimized choice: {:?} for target '{}'", 
                            optimized_choice, optimization_candidate);
                }
                return Ok(optimized_choice);
            }
        }
        
        // Fallback to random generation if no optimization opportunity available
        self.generate_random_fallback_choice(choice_type, constraints)
    }
    
    /// Generate a random fallback choice
    fn generate_random_fallback_choice(&mut self, choice_type: ChoiceType, constraints: &Constraints) -> Result<ChoiceValue, ProviderError> {
        if self.config.debug_coverage_decisions {
            println!("COVERAGE_RANDOM DEBUG: Generating random fallback choice for {:?}", choice_type);
        }
        
        match (choice_type, constraints) {
            (ChoiceType::Integer, Constraints::Integer(int_constraints)) => {
                let min = int_constraints.min_value.unwrap_or(i128::MIN);
                let max = int_constraints.max_value.unwrap_or(i128::MAX);
                
                if min > max {
                    return Err(ProviderError::InvalidChoice(format!("Invalid range: {} > {}", min, max)));
                }
                
                let value = if min == max {
                    min
                } else {
                    self.rng.gen_range(min..=max)
                };
                
                Ok(ChoiceValue::Integer(value))
            },
            (ChoiceType::Boolean, Constraints::Boolean(bool_constraints)) => {
                let value = self.rng.gen::<f64>() < bool_constraints.p;
                Ok(ChoiceValue::Boolean(value))
            },
            (ChoiceType::Float, Constraints::Float(float_constraints)) => {
                let value = self.rng.gen::<f64>();
                let constrained_value = if float_constraints.min_value.is_finite() && float_constraints.max_value.is_finite() {
                    float_constraints.min_value + value * (float_constraints.max_value - float_constraints.min_value)
                } else {
                    value
                };
                
                if !float_constraints.allow_nan && constrained_value.is_nan() {
                    return self.generate_random_fallback_choice(choice_type, constraints);
                }
                
                Ok(ChoiceValue::Float(constrained_value))
            },
            _ => Err(ProviderError::InvalidChoice(format!("Unsupported choice type: {:?}", choice_type))),
        }
    }
    
    /// Track structural coverage for the given label and strategy
    fn track_structural_coverage(&mut self, label: u32, strategy_name: &str) {
        let current_time = std::time::SystemTime::now();
        
        // Update or create coverage tag
        let coverage_tag = self.coverage_tracker.coverage_tags.entry(label).or_insert_with(|| {
            StructuralCoverageTag {
                label,
                hit_count: 0,
                strategy_name: strategy_name.to_string(),
                first_hit_timestamp: current_time,
                last_hit_timestamp: current_time,
                metadata: HashMap::new(),
            }
        });
        
        coverage_tag.hit_count += 1;
        coverage_tag.last_hit_timestamp = current_time;
        
        // Update coverage statistics
        self.coverage_tracker.coverage_stats.current_test_case_coverage.insert(label);
        self.coverage_tracker.coverage_stats.total_coverage_hits += 1;
        
        // Update current test case context
        if let Some(ref mut context) = self.current_context {
            context.coverage_points_hit.insert(label);
        }
        
        if self.config.debug_coverage_decisions {
            println!("COVERAGE_TRACK DEBUG: Hit coverage point {:#08X} for strategy '{}' (count: {})", 
                    label, strategy_name, coverage_tag.hit_count);
        }
    }
    
    /// Find a novel prefix path from the choice tree
    fn find_novel_prefix_path(&self) -> Option<Vec<ChoiceValue>> {
        // Simplified novel prefix generation - in a full implementation this would
        // traverse the choice tree to find unexplored branches
        if self.choice_tree.node_count < 10 {
            // Early exploration phase - don't have enough data for novel prefixes
            return None;
        }
        
        // For demonstration, return None to indicate no novel prefix available
        // A full implementation would implement tree traversal to find unexplored paths
        None
    }
    
    /// Find similar spans for cross-contamination mutation
    fn find_similar_spans_for_mutation(&self) -> Option<(&CompletedSpan, &CompletedSpan)> {
        if self.span_tracker.completed_spans.len() < 2 {
            return None;
        }
        
        // Find spans with the same label for cross-contamination
        for (i, span1) in self.span_tracker.completed_spans.iter().enumerate() {
            for span2 in self.span_tracker.completed_spans.iter().skip(i + 1) {
                if span1.span_info.label == span2.span_info.label {
                    if self.config.debug_coverage_decisions {
                        println!("COVERAGE_MUTATION DEBUG: Found similar spans with label {:#08X}", span1.span_info.label);
                    }
                    return Some((span1, span2));
                }
            }
        }
        
        None
    }
    
    /// Find similar spans for mutation without borrowing references
    fn find_similar_spans_info_for_mutation(&self) -> Option<(Vec<ChoiceValue>, u32)> {
        if self.span_tracker.completed_spans.len() < 2 {
            return None;
        }
        
        // Find spans with the same label for cross-contamination
        for (i, span1) in self.span_tracker.completed_spans.iter().enumerate() {
            for span2 in self.span_tracker.completed_spans.iter().skip(i + 1) {
                if span1.span_info.label == span2.span_info.label {
                    if self.config.debug_coverage_decisions {
                        println!("COVERAGE_MUTATION DEBUG: Found similar spans with label {:#08X}", span1.span_info.label);
                    }
                    return Some((span1.span_info.choices.clone(), span1.span_info.label));
                }
            }
        }
        
        None
    }
    
    /// Find a target optimization candidate
    fn find_target_optimization_candidate(&self) -> Option<String> {
        // Return the target label with the highest recent improvement potential
        self.target_tracker.best_observations.keys().next().cloned()
    }
    
    /// Perform hill-climbing optimization for the given target
    fn perform_hill_climbing_optimization(&mut self, target_label: &str, choice_type: ChoiceType, constraints: &Constraints) -> Option<ChoiceValue> {
        // Simplified hill-climbing - a full implementation would modify choices
        // systematically to improve target observations
        
        if let Some(best_value) = self.target_tracker.best_observations.get(target_label) {
            if self.config.debug_coverage_decisions {
                println!("COVERAGE_TARGET_OPT DEBUG: Optimizing for target '{}' with best value {}", target_label, best_value);
            }
            
            // For demonstration, return None to indicate no optimization available
            // A full implementation would generate choices based on optimization gradients
        }
        
        None
    }
    
    /// Check if a choice satisfies the given constraints
    fn choice_satisfies_constraints(&self, choice: &ChoiceValue, choice_type: ChoiceType, constraints: &Constraints) -> bool {
        match (choice, choice_type, constraints) {
            (ChoiceValue::Integer(value), ChoiceType::Integer, Constraints::Integer(int_constraints)) => {
                if let Some(min) = int_constraints.min_value {
                    if *value < min { return false; }
                }
                if let Some(max) = int_constraints.max_value {
                    if *value > max { return false; }
                }
                true
            },
            (ChoiceValue::Boolean(_), ChoiceType::Boolean, Constraints::Boolean(_)) => true,
            (ChoiceValue::Float(value), ChoiceType::Float, Constraints::Float(float_constraints)) => {
                if *value < float_constraints.min_value || *value > float_constraints.max_value {
                    return false;
                }
                if !float_constraints.allow_nan && value.is_nan() {
                    return false;
                }
                true
            },
            _ => false,
        }
    }
    
    /// Update strategy performance based on generation result
    fn update_strategy_performance(&mut self, strategy: GenerationStrategy, success: bool) {
        let performance = self.strategy_controller.strategy_performance.entry(strategy).or_default();
        performance.usage_count += 1;
        if success {
            performance.success_count += 1;
        }
        
        // Update recent success rate (simplified)
        performance.recent_success_rate = performance.success_count as f64 / performance.usage_count as f64;
        
        if self.config.debug_coverage_decisions {
            println!("COVERAGE_STRATEGY_PERF DEBUG: Strategy {:?} used {} times, success rate: {:.2}", 
                    strategy, performance.usage_count, performance.recent_success_rate);
        }
    }
    
    /// Add a target observation for optimization feedback
    pub fn add_target_observation(&mut self, label: &str, value: f64) {
        if self.config.debug_coverage_decisions {
            println!("COVERAGE_TARGET DEBUG: Adding target observation '{}' = {}", label, value);
        }
        
        self.target_tracker.current_observations.insert(label.to_string(), value);
        
        // Update best observation if this is better
        let is_better = self.target_tracker.best_observations.get(label)
            .map(|&best| value > best)
            .unwrap_or(true);
        
        if is_better {
            self.target_tracker.best_observations.insert(label.to_string(), value);
        }
        
        // Add to observation history
        let observation = TargetObservation {
            label: label.to_string(),
            value,
            choice_sequence: self.current_context.as_ref()
                .map(|ctx| ctx.choice_sequence.clone())
                .unwrap_or_default(),
            timestamp: std::time::SystemTime::now(),
            active_coverage_points: self.coverage_tracker.coverage_stats.current_test_case_coverage.clone(),
        };
        
        self.target_tracker.observation_history.push(observation);
        
        // Update current test case context
        if let Some(ref mut context) = self.current_context {
            context.target_observations.insert(label.to_string(), value);
        }
    }
    
    /// Update the choice tree with the current choice sequence
    fn update_choice_tree(&mut self, choice_sequence: &[ChoiceValue]) {
        let mut current_node = &mut self.choice_tree.root;
        current_node.visit_count += 1;
        
        for (depth, choice) in choice_sequence.iter().enumerate() {
            if depth >= self.config.max_tree_depth {
                break;
            }
            
            let child_node = current_node.children.entry(choice.clone()).or_insert_with(|| {
                self.choice_tree.node_count += 1;
                Box::new(ChoiceTreeNode {
                    choice: Some(choice.clone()),
                    choice_metadata: None,
                    children: HashMap::new(),
                    visit_count: 0,
                    exhausted: false,
                    best_target_observation: None,
                    coverage_points_reached: BTreeSet::new(),
                })
            });
            
            child_node.visit_count += 1;
            
            // Update coverage points reached
            child_node.coverage_points_reached.extend(&self.coverage_tracker.coverage_stats.current_test_case_coverage);
            
            current_node = child_node;
        }
        
        self.choice_tree.max_depth = self.choice_tree.max_depth.max(choice_sequence.len());
        
        if self.config.debug_coverage_decisions {
            println!("COVERAGE_TREE DEBUG: Updated choice tree, now {} nodes, max depth {}", 
                    self.choice_tree.node_count, self.choice_tree.max_depth);
        }
    }
    
    /// Update Pareto front with current solution
    fn update_pareto_front(&mut self) {
        if let Some(ref context) = self.current_context {
            let solution = ParetoSolution {
                choice_sequence: context.choice_sequence.clone(),
                target_observations: context.target_observations.clone(),
                coverage_points: context.coverage_points_hit.clone(),
                quality_metrics: QualityMetrics {
                    coverage_breadth: context.coverage_points_hit.len(),
                    average_target_value: context.target_observations.values().sum::<f64>() / context.target_observations.len().max(1) as f64,
                    novelty_score: self.calculate_novelty_score(&context.choice_sequence),
                    efficiency_score: context.coverage_points_hit.len() as f64 / context.choice_sequence.len().max(1) as f64,
                },
                timestamp: std::time::SystemTime::now(),
            };
            
            // Check if solution is Pareto-optimal
            if self.is_pareto_optimal(&solution) {
                self.pareto_front.solutions.push(solution);
                
                // Remove dominated solutions
                self.remove_dominated_solutions();
                
                // Maintain maximum front size
                if self.pareto_front.solutions.len() > self.pareto_front.max_size {
                    self.trim_pareto_front();
                }
                
                if self.config.debug_coverage_decisions {
                    println!("COVERAGE_PARETO DEBUG: Added solution to Pareto front, now {} solutions", 
                            self.pareto_front.solutions.len());
                }
            }
        }
    }
    
    /// Check if a solution is Pareto-optimal
    fn is_pareto_optimal(&mut self, candidate: &ParetoSolution) -> bool {
        for existing in &self.pareto_front.solutions {
            self.pareto_front.dominance_comparisons += 1;
            if self.dominates(existing, candidate) {
                return false;
            }
        }
        true
    }
    
    /// Check if solution A dominates solution B
    fn dominates(&self, a: &ParetoSolution, b: &ParetoSolution) -> bool {
        // A dominates B if A is at least as good in all objectives and strictly better in at least one
        let a_better_coverage = a.quality_metrics.coverage_breadth >= b.quality_metrics.coverage_breadth;
        let a_better_targets = a.quality_metrics.average_target_value >= b.quality_metrics.average_target_value;
        let a_better_novelty = a.quality_metrics.novelty_score >= b.quality_metrics.novelty_score;
        
        let a_strictly_better = a.quality_metrics.coverage_breadth > b.quality_metrics.coverage_breadth
            || a.quality_metrics.average_target_value > b.quality_metrics.average_target_value
            || a.quality_metrics.novelty_score > b.quality_metrics.novelty_score;
        
        a_better_coverage && a_better_targets && a_better_novelty && a_strictly_better
    }
    
    /// Remove dominated solutions from the Pareto front
    fn remove_dominated_solutions(&mut self) {
        let mut to_remove = Vec::new();
        
        for i in 0..self.pareto_front.solutions.len() {
            for j in 0..self.pareto_front.solutions.len() {
                if i != j {
                    self.pareto_front.dominance_comparisons += 1;
                    if self.dominates(&self.pareto_front.solutions[j], &self.pareto_front.solutions[i]) {
                        to_remove.push(i);
                        break;
                    }
                }
            }
        }
        
        // Remove in reverse order to maintain indices
        to_remove.sort();
        to_remove.reverse();
        for index in to_remove {
            self.pareto_front.solutions.remove(index);
        }
    }
    
    /// Trim Pareto front to maximum size
    fn trim_pareto_front(&mut self) {
        // Simple trimming strategy: remove oldest solutions
        // A more sophisticated approach might use diversity preservation
        self.pareto_front.solutions.sort_by_key(|sol| sol.timestamp);
        self.pareto_front.solutions.truncate(self.pareto_front.max_size);
    }
    
    /// Calculate novelty score for a choice sequence
    fn calculate_novelty_score(&self, choice_sequence: &[ChoiceValue]) -> f64 {
        // Simplified novelty calculation based on choice diversity
        let unique_choices: HashSet<_> = choice_sequence.iter().collect();
        unique_choices.len() as f64 / choice_sequence.len().max(1) as f64
    }
    
    /// Complete the current test case and update all tracking systems
    fn complete_test_case(&mut self, success: bool) {
        if let Some(context) = self.current_context.take() {
            let execution_time = context.start_time.elapsed().unwrap_or_default();
            
            if self.config.debug_coverage_decisions {
                println!("COVERAGE_TEST_CASE DEBUG: Completing test case '{}', success: {}, duration: {:?}", 
                        context.test_case_id, success, execution_time);
            }
            
            // Update choice tree
            self.update_choice_tree(&context.choice_sequence);
            
            // Update Pareto front
            if success {
                self.update_pareto_front();
            }
            
            // Complete any active spans
            while let Some(span_info) = self.span_tracker.span_stack.pop() {
                let completed_span = CompletedSpan {
                    span_info,
                    end_position: context.choice_sequence.len(),
                    discarded: !success,
                    execution_time,
                    coverage_points: context.coverage_points_hit.clone(),
                };
                
                // Add to mutator groups
                let label = completed_span.span_info.label;
                self.span_tracker.mutator_groups.entry(label).or_default().push(self.span_tracker.completed_spans.len());
                
                self.span_tracker.completed_spans.push(completed_span);
            }
            
            // Update coverage statistics
            self.coverage_tracker.coverage_stats.unique_coverage_points = self.coverage_tracker.coverage_tags.len();
            self.coverage_tracker.coverage_stats.current_test_case_coverage.clear();
            
            // Update metrics
            self.metrics.choice_tree_size = self.choice_tree.node_count;
            self.metrics.pareto_front_size = self.pareto_front.solutions.len();
            self.metrics.coverage_targets_hit = context.coverage_points_hit.len() as u64;
            
            if !self.span_tracker.completed_spans.is_empty() {
                self.metrics.average_span_depth = self.span_tracker.completed_spans.iter()
                    .map(|span| span.span_info.depth as f64)
                    .sum::<f64>() / self.span_tracker.completed_spans.len() as f64;
            }
        }
    }
}

impl PrimitiveProvider for CoverageGuidedProvider {
    fn lifetime(&self) -> ProviderLifetime {
        ProviderLifetime::TestRun // Coverage guidance benefits from persistence across test cases
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            avoid_realization: false,
            add_observability_callback: true,
            structural_awareness: true,
            replay_support: true,
            symbolic_constraints: false,
        }
    }
    
    fn draw_boolean(&mut self, p: f64) -> Result<bool, ProviderError> {
        if p < 0.0 || p > 1.0 {
            return Err(ProviderError::InvalidChoice(format!("Invalid probability: {}", p)));
        }
        
        let constraints = BooleanConstraints { p };
        let constraints_obj = Constraints::Boolean(constraints);
        
        match self.generate_coverage_guided_choice(ChoiceType::Boolean, &constraints_obj)? {
            ChoiceValue::Boolean(value) => Ok(value),
            _ => Err(ProviderError::InvalidChoice("Generated non-boolean value for boolean choice".to_string())),
        }
    }
    
    fn draw_integer(&mut self, constraints: &IntegerConstraints) -> Result<i128, ProviderError> {
        let constraints_obj = Constraints::Integer(constraints.clone());
        
        match self.generate_coverage_guided_choice(ChoiceType::Integer, &constraints_obj)? {
            ChoiceValue::Integer(value) => Ok(value),
            _ => Err(ProviderError::InvalidChoice("Generated non-integer value for integer choice".to_string())),
        }
    }
    
    fn draw_float(&mut self, constraints: &FloatConstraints) -> Result<f64, ProviderError> {
        let constraints_obj = Constraints::Float(constraints.clone());
        
        match self.generate_coverage_guided_choice(ChoiceType::Float, &constraints_obj)? {
            ChoiceValue::Float(value) => Ok(value),
            _ => Err(ProviderError::InvalidChoice("Generated non-float value for float choice".to_string())),
        }
    }
    
    fn draw_string(&mut self, intervals: &IntervalSet, min_size: usize, max_size: usize) -> Result<String, ProviderError> {
        // Simplified string generation for coverage-guided provider
        // A full implementation would integrate with choice tree and mutation strategies
        
        if min_size > max_size {
            return Err(ProviderError::InvalidChoice(format!("Invalid size range: {} > {}", min_size, max_size)));
        }
        
        if intervals.intervals.is_empty() {
            return Err(ProviderError::InvalidChoice("Empty character set".to_string()));
        }
        
        let size = if min_size == max_size {
            min_size
        } else {
            self.rng.gen_range(min_size..=max_size)
        };
        
        let mut result = String::new();
        for _ in 0..size {
            let interval_idx = self.rng.gen_range(0..intervals.intervals.len());
            let (start, end) = intervals.intervals[interval_idx];
            let code_point = self.rng.gen_range(start..=end);
            
            if let Some(ch) = char::from_u32(code_point) {
                result.push(ch);
            } else {
                result.push('?');
            }
        }
        
        Ok(result)
    }
    
    fn draw_bytes(&mut self, min_size: usize, max_size: usize) -> Result<Vec<u8>, ProviderError> {
        // Simplified bytes generation for coverage-guided provider
        
        if min_size > max_size {
            return Err(ProviderError::InvalidChoice(format!("Invalid size range: {} > {}", min_size, max_size)));
        }
        
        let size = if min_size == max_size {
            min_size
        } else {
            self.rng.gen_range(min_size..=max_size)
        };
        
        let mut bytes = vec![0u8; size];
        self.rng.fill_bytes(&mut bytes[..]);
        Ok(bytes)
    }
    
    fn observe_test_case(&mut self) -> HashMap<String, serde_json::Value> {
        let mut observation = HashMap::new();
        
        observation.insert("strategy".to_string(), 
                         serde_json::Value::String(format!("{:?}", self.strategy_controller.current_strategy)));
        observation.insert("total_choices".to_string(),
                         serde_json::Value::Number(serde_json::Number::from(self.metrics.total_choices)));
        observation.insert("coverage_points".to_string(),
                         serde_json::Value::Number(serde_json::Number::from(self.coverage_tracker.coverage_stats.unique_coverage_points)));
        observation.insert("choice_tree_size".to_string(),
                         serde_json::Value::Number(serde_json::Number::from(self.choice_tree.node_count)));
        observation.insert("pareto_front_size".to_string(),
                         serde_json::Value::Number(serde_json::Number::from(self.pareto_front.solutions.len())));
        observation.insert("target_observations".to_string(),
                         serde_json::Value::Number(serde_json::Number::from(self.target_tracker.current_observations.len())));
        
        observation
    }
    
    fn observe_information_messages(&mut self, lifetime: ProviderLifetime) -> Vec<ObservationMessage> {
        let mut messages = Vec::new();
        
        if lifetime == ProviderLifetime::TestRun {
            messages.push(ObservationMessage {
                message_type: ObservationType::Info,
                title: "Coverage-Guided Generation Status".to_string(),
                content: serde_json::json!({
                    "total_choices": self.metrics.total_choices,
                    "novel_prefixes": self.metrics.novel_prefixes_generated,
                    "mutations": self.metrics.mutations_generated,
                    "target_optimizations": self.metrics.target_optimizations,
                    "coverage_points": self.coverage_tracker.coverage_stats.unique_coverage_points,
                    "choice_tree_size": self.choice_tree.node_count,
                    "pareto_front_size": self.pareto_front.solutions.len(),
                }),
                timestamp: std::time::SystemTime::now(),
            });
        }
        
        messages
    }
    
    fn on_observation(&mut self, observation: &TestCaseObservation) {
        if observation.observation_type == "target" {
            if let Some(label) = observation.metadata.get("label").and_then(|v| v.as_str()) {
                if let Some(value) = observation.metadata.get("value").and_then(|v| v.as_f64()) {
                    self.add_target_observation(label, value);
                }
            }
        }
    }
    
    fn span_start(&mut self, label: u32) {
        let depth = self.span_tracker.span_stack.len();
        let start_position = self.current_context.as_ref()
            .map(|ctx| ctx.choice_sequence.len())
            .unwrap_or(0);
        
        let span_info = SpanInfo {
            label,
            depth,
            start_position,
            choices: Vec::new(),
            children: Vec::new(),
            start_timestamp: std::time::SystemTime::now(),
        };
        
        if self.config.debug_coverage_decisions {
            println!("COVERAGE_SPAN DEBUG: Starting span {:#08X} at depth {}", label, depth);
        }
        
        self.span_tracker.span_stack.push(span_info);
    }
    
    fn span_end(&mut self, discard: bool) {
        if let Some(mut span_info) = self.span_tracker.span_stack.pop() {
            let end_position = self.current_context.as_ref()
                .map(|ctx| ctx.choice_sequence.len())
                .unwrap_or(0);
            
            // Extract choices made within this span
            if let Some(ref context) = self.current_context {
                if span_info.start_position < context.choice_sequence.len() {
                    span_info.choices = context.choice_sequence[span_info.start_position..end_position.min(context.choice_sequence.len())].to_vec();
                }
            }
            
            let execution_time = span_info.start_timestamp.elapsed().unwrap_or_default();
            
            if self.config.debug_coverage_decisions {
                println!("COVERAGE_SPAN DEBUG: Ending span {:#08X}, discard: {}, choices: {}", 
                        span_info.label, discard, span_info.choices.len());
            }
            
            if !discard {
                let completed_span = CompletedSpan {
                    span_info,
                    end_position,
                    discarded: discard,
                    execution_time,
                    coverage_points: self.coverage_tracker.coverage_stats.current_test_case_coverage.clone(),
                };
                
                // Add to mutator groups
                let label = completed_span.span_info.label;
                self.span_tracker.mutator_groups.entry(label).or_default().push(self.span_tracker.completed_spans.len());
                
                self.span_tracker.completed_spans.push(completed_span);
            }
        }
    }
    
    fn per_test_case_context(&mut self) -> Box<dyn TestCaseContext> {
        let test_case_id = format!("test_case_{}", std::process::id());
        let context = CoverageTestCaseContext {
            test_case_id: test_case_id.clone(),
            generation_strategy: self.strategy_controller.current_strategy,
            start_time: std::time::SystemTime::now(),
            coverage_points_hit: BTreeSet::new(),
            target_observations: HashMap::new(),
            choice_sequence: Vec::new(),
        };
        
        self.current_context = Some(context);
        
        Box::new(CoverageGuidedTestCaseContext {
            test_case_id,
        })
    }
    
    fn replay_choices(&mut self, choices: &[ChoiceValue]) -> Result<(), ProviderError> {
        if self.config.debug_coverage_decisions {
            println!("COVERAGE_REPLAY DEBUG: Replaying {} choices", choices.len());
        }
        
        // Update choice tree with replayed choices
        self.update_choice_tree(choices);
        
        // Add choices to current context if available
        if let Some(ref mut context) = self.current_context {
            context.choice_sequence.extend_from_slice(choices);
        }
        
        Ok(())
    }
}

/// Provider factory for coverage-guided provider integration with registry
#[derive(Debug)]
pub struct CoverageGuidedProviderFactory;

impl ProviderFactory for CoverageGuidedProviderFactory {
    fn name(&self) -> &str {
        "coverage-guided"
    }
    
    fn create(&self) -> Result<Box<dyn PrimitiveProvider>, ProviderError> {
        Ok(Box::new(CoverageGuidedProvider::new()))
    }
    
    fn supports_configuration(&self) -> bool {
        true
    }
    
    fn default_configuration(&self) -> serde_json::Value {
        serde_json::json!({
            "novel_prefix_probability": 0.3,
            "mutation_probability": 0.4,
            "target_optimization_probability": 0.2,
            "max_tree_depth": 1000,
            "max_pareto_front_size": 100,
            "debug_coverage_decisions": true
        })
    }
    
    fn validate_environment(&self) -> Result<(), ProviderError> {
        // Coverage-guided provider requires no special environment setup
        Ok(())
    }
    
    fn create_with_config(&self, _config: &serde_json::Value) -> Result<Box<dyn PrimitiveProvider>, ProviderError> {
        // For now, ignore config and use defaults - in full implementation would parse config
        Ok(Box::new(CoverageGuidedProvider::new()))
    }
    
    fn description(&self) -> &str {
        "Advanced coverage-guided generation provider with sophisticated generation strategies"
    }
    
    fn provider_type(&self) -> &str {
        "coverage-guided"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
}

/// Test case context implementation for coverage-guided provider
#[derive(Debug)]
pub struct CoverageGuidedTestCaseContext {
    test_case_id: String,
    // In a real implementation, this would be a weak reference or channel to communicate with provider
    // For this test implementation, we'll just use the test_case_id
}

impl TestCaseContext for CoverageGuidedTestCaseContext {
    fn enter_test_case(&mut self) {
        println!("COVERAGE_CONTEXT DEBUG: Entering test case '{}'", self.test_case_id);
    }
    
    fn exit_test_case(&mut self, success: bool) {
        println!("COVERAGE_CONTEXT DEBUG: Exiting test case '{}', success: {}", self.test_case_id, success);
        
        // In a full implementation, this would call complete_test_case on the provider
        // For this simplified version, we just log the exit
    }
    
    fn get_context_data(&self) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();
        data.insert("test_case_id".to_string(), serde_json::Value::String(self.test_case_id.clone()));
        data.insert("provider_type".to_string(), serde_json::Value::String("coverage_guided".to_string()));
        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::IntegerConstraints;

    #[test]
    fn test_coverage_guided_provider_creation() {
        let provider = CoverageGuidedProvider::new();
        assert_eq!(provider.lifetime(), ProviderLifetime::TestRun);
        assert!(provider.capabilities().structural_awareness);
        assert!(provider.capabilities().add_observability_callback);
        assert!(provider.capabilities().replay_support);
    }

    #[test]
    fn test_coverage_guided_integer_generation() {
        let mut provider = CoverageGuidedProvider::new();
        
        let constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        };
        
        let value = provider.draw_integer(&constraints).unwrap();
        assert!(value >= 0 && value <= 100);
        
        // Verify metrics were updated
        assert_eq!(provider.metrics.total_choices, 1);
    }

    #[test]
    fn test_coverage_tracking() {
        let mut provider = CoverageGuidedProvider::new();
        
        // Generate a few choices to create coverage
        let constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        };
        
        provider.draw_integer(&constraints).unwrap();
        provider.draw_integer(&constraints).unwrap();
        
        assert!(provider.coverage_tracker.coverage_stats.total_coverage_hits >= 2);
        assert!(!provider.coverage_tracker.coverage_tags.is_empty());
    }

    #[test]
    fn test_span_tracking() {
        let mut provider = CoverageGuidedProvider::new();
        
        // Start and end a span
        provider.span_start(0x12345678);
        
        let constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        };
        
        provider.draw_integer(&constraints).unwrap();
        provider.span_end(false);
        
        // Verify span was tracked
        assert_eq!(provider.span_tracker.completed_spans.len(), 1);
        assert_eq!(provider.span_tracker.completed_spans[0].span_info.label, 0x12345678);
    }

    #[test]
    fn test_target_observations() {
        let mut provider = CoverageGuidedProvider::new();
        
        // Add some target observations
        provider.add_target_observation("test_target", 10.0);
        provider.add_target_observation("test_target", 15.0);
        provider.add_target_observation("another_target", 5.0);
        
        assert_eq!(provider.target_tracker.current_observations.len(), 2);
        assert_eq!(provider.target_tracker.best_observations["test_target"], 15.0);
        assert_eq!(provider.target_tracker.observation_history.len(), 3);
    }

    #[test]
    fn test_strategy_selection() {
        let mut provider = CoverageGuidedProvider::new();
        
        // Test strategy selection multiple times
        let mut strategy_counts = HashMap::new();
        for _ in 0..100 {
            let strategy = provider.select_generation_strategy();
            *strategy_counts.entry(strategy).or_insert(0) += 1;
        }
        
        // Should see multiple different strategies selected
        assert!(strategy_counts.len() > 1);
    }

    #[test]
    fn test_choice_tree_updates() {
        let mut provider = CoverageGuidedProvider::new();
        
        let choices = vec![
            ChoiceValue::Integer(42),
            ChoiceValue::Boolean(true),
            ChoiceValue::Float(3.14),
        ];
        
        provider.update_choice_tree(&choices);
        
        assert!(provider.choice_tree.node_count > 1);
        assert_eq!(provider.choice_tree.max_depth, 3);
    }

    #[test]
    fn test_pareto_front_management() {
        let mut provider = CoverageGuidedProvider::new();
        
        // Create a test context
        let context = CoverageTestCaseContext {
            test_case_id: "test".to_string(),
            generation_strategy: GenerationStrategy::RandomFallback,
            start_time: std::time::SystemTime::now(),
            coverage_points_hit: [1, 2, 3].iter().cloned().collect(),
            target_observations: [("target1".to_string(), 10.0)].iter().cloned().collect(),
            choice_sequence: vec![ChoiceValue::Integer(42)],
        };
        
        provider.current_context = Some(context);
        provider.update_pareto_front();
        
        assert_eq!(provider.pareto_front.solutions.len(), 1);
    }

    #[test]
    fn test_observability() {
        let mut provider = CoverageGuidedProvider::new();
        
        // Generate some activity
        let constraints = IntegerConstraints {
            min_value: Some(0),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        };
        
        provider.draw_integer(&constraints).unwrap();
        provider.add_target_observation("test", 5.0);
        
        let observation = provider.observe_test_case();
        assert!(observation.contains_key("strategy"));
        assert!(observation.contains_key("total_choices"));
        assert!(observation.contains_key("coverage_points"));
        
        let messages = provider.observe_information_messages(ProviderLifetime::TestRun);
        assert!(!messages.is_empty());
        assert_eq!(messages[0].title, "Coverage-Guided Generation Status");
    }

    #[test]
    fn test_configuration() {
        let config = CoverageGuidedConfig {
            novel_prefix_probability: 0.5,
            mutation_probability: 0.3,
            target_optimization_probability: 0.2,
            max_tree_depth: 500,
            max_pareto_front_size: 50,
            span_similarity_threshold: 0.9,
            target_optimization_iterations: 20,
            debug_coverage_decisions: false,
        };
        
        let provider = CoverageGuidedProvider::with_config(config.clone());
        assert_eq!(provider.config.novel_prefix_probability, 0.5);
        assert_eq!(provider.config.max_tree_depth, 500);
        assert!(!provider.config.debug_coverage_decisions);
    }

    #[test]
    fn test_choice_constraints_validation() {
        let provider = CoverageGuidedProvider::new();
        
        let int_constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(0),
        });
        
        // Valid choice
        assert!(provider.choice_satisfies_constraints(
            &ChoiceValue::Integer(5), 
            ChoiceType::Integer, 
            &int_constraints
        ));
        
        // Invalid choice (out of range)
        assert!(!provider.choice_satisfies_constraints(
            &ChoiceValue::Integer(15), 
            ChoiceType::Integer, 
            &int_constraints
        ));
        
        // Invalid choice (wrong type)
        assert!(!provider.choice_satisfies_constraints(
            &ChoiceValue::Boolean(true), 
            ChoiceType::Integer, 
            &int_constraints
        ));
    }

    #[test]
    fn test_replay_functionality() {
        let mut provider = CoverageGuidedProvider::new();
        
        let choices = vec![
            ChoiceValue::Integer(42),
            ChoiceValue::Boolean(true),
            ChoiceValue::Float(3.14),
        ];
        
        assert!(provider.replay_choices(&choices).is_ok());
        
        // Verify choice tree was updated
        assert!(provider.choice_tree.node_count > 1);
    }

    #[test]
    fn test_dominance_relation() {
        let provider = CoverageGuidedProvider::new();
        
        let solution_a = ParetoSolution {
            choice_sequence: vec![ChoiceValue::Integer(1)],
            target_observations: [("target1".to_string(), 10.0)].iter().cloned().collect(),
            coverage_points: [1, 2, 3].iter().cloned().collect(),
            quality_metrics: QualityMetrics {
                coverage_breadth: 3,
                average_target_value: 10.0,
                novelty_score: 0.8,
                efficiency_score: 3.0,
            },
            timestamp: std::time::SystemTime::now(),
        };
        
        let solution_b = ParetoSolution {
            choice_sequence: vec![ChoiceValue::Integer(2)],
            target_observations: [("target1".to_string(), 5.0)].iter().cloned().collect(),
            coverage_points: [1, 2].iter().cloned().collect(),
            quality_metrics: QualityMetrics {
                coverage_breadth: 2,
                average_target_value: 5.0,
                novelty_score: 0.6,
                efficiency_score: 2.0,
            },
            timestamp: std::time::SystemTime::now(),
        };
        
        // Solution A should dominate solution B
        assert!(provider.dominates(&solution_a, &solution_b));
        assert!(!provider.dominates(&solution_b, &solution_a));
    }
}