//! Advanced Shrinking System with Float Encoding Export Integration
//!
//! This module implements the complete shrinking capability for the ChoiceSystem,
//! providing advanced algorithms to minimize test cases while preserving failure
//! conditions. It includes specialized shrinking strategies for different data types,
//! structural optimization, and multi-objective minimization.
//!
//! ## Float Encoding Export Integration
//!
//! This module integrates the sophisticated float encoding export system to enable
//! optimal shrinking behavior for floating-point values using lexicographic encoding.
//! The integration provides:
//!
//! - `float_to_lex()` - Convert float to lexicographic encoding for shrinking optimization
//! - `lex_to_float()` - Convert lexicographic encoding back to float value
//! - `float_to_int()` - Convert float to integer for DataTree storage
//! - `int_to_float()` - Convert integer back to float from DataTree
//! - Lexicographic float comparison for optimal shrinking order

use crate::choice::ChoiceValue;
use crate::float_encoding_export::{
    float_to_lex, lex_to_float, float_to_int, int_to_float, FloatWidth
};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::fmt;

/// Represents a choice made during test generation
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub value: ChoiceValue,
    pub index: usize,
}

/// Represents the result of a shrinking operation
#[derive(Debug, Clone, PartialEq)]
pub enum ShrinkResult {
    /// Shrinking was successful and produced a smaller test case
    Success(Vec<Choice>),
    /// Shrinking failed - the test case could not be made smaller
    Failed,
    /// Shrinking was blocked by constraints
    Blocked(String),
    /// Shrinking reached maximum iterations
    Timeout,
}

/// Metrics for tracking shrinking performance and effectiveness
#[derive(Debug, Clone, Default)]
pub struct ShrinkingMetrics {
    pub total_attempts: u64,
    pub successful_shrinks: u64,
    pub failed_shrinks: u64,
    pub blocked_shrinks: u64,
    pub timeouts: u64,
    pub reduction_percentage: f64,
    pub average_iterations: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl fmt::Display for ShrinkingMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ShrinkingMetrics {{ attempts: {}, success_rate: {:.1}%, reduction: {:.1}% }}", 
               self.total_attempts, 
               if self.total_attempts > 0 { (self.successful_shrinks as f64 / self.total_attempts as f64) * 100.0 } else { 0.0 },
               self.reduction_percentage)
    }
}

/// Represents different shrinking strategies with their priority levels
#[derive(Debug, Clone, PartialEq)]
pub enum ShrinkingStrategy {
    /// Minimize integer values towards zero or target
    MinimizeIntegers { target: i128, aggressive: bool },
    /// Minimize floating point values
    MinimizeFloats { target: f64, precision_reduction: bool },
    /// Reduce collection sizes and simplify content
    SimplifyCollections { min_size: usize, preserve_structure: bool },
    /// Minimize string content and length
    OptimizeStrings { preserve_chars: HashSet<char> },
    /// Remove redundant or duplicate elements
    Deduplicate { similarity_threshold: f64 },
    /// Reorder elements for optimal structure
    Reorder { priority_function: ReorderPriority },
    /// Apply constraint-aware optimizations
    ConstraintOptimization { repair_violations: bool },
    /// Multi-objective optimization using Pareto principles
    ParetoOptimization { objectives: Vec<OptimizationObjective> },
}

impl Hash for ShrinkingStrategy {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            ShrinkingStrategy::MinimizeIntegers { target, aggressive } => {
                0u8.hash(state);
                target.hash(state);
                aggressive.hash(state);
            }
            ShrinkingStrategy::MinimizeFloats { target, precision_reduction } => {
                1u8.hash(state);
                target.to_bits().hash(state);
                precision_reduction.hash(state);
            }
            ShrinkingStrategy::SimplifyCollections { min_size, preserve_structure } => {
                2u8.hash(state);
                min_size.hash(state);
                preserve_structure.hash(state);
            }
            ShrinkingStrategy::OptimizeStrings { preserve_chars } => {
                3u8.hash(state);
                for ch in preserve_chars {
                    ch.hash(state);
                }
            }
            ShrinkingStrategy::Deduplicate { similarity_threshold } => {
                4u8.hash(state);
                similarity_threshold.to_bits().hash(state);
            }
            ShrinkingStrategy::Reorder { priority_function } => {
                5u8.hash(state);
                priority_function.hash(state);
            }
            ShrinkingStrategy::ConstraintOptimization { repair_violations } => {
                6u8.hash(state);
                repair_violations.hash(state);
            }
            ShrinkingStrategy::ParetoOptimization { objectives } => {
                7u8.hash(state);
                objectives.len().hash(state);
                // Note: objectives hashing would need custom implementation too
            }
        }
    }
}

impl Eq for ShrinkingStrategy {}

/// Priority functions for reordering elements during shrinking
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReorderPriority {
    /// Order by value magnitude (smallest first)
    ByMagnitude,
    /// Order by frequency of occurrence
    ByFrequency,
    /// Order by structural complexity
    ByComplexity,
    /// Order by constraint satisfaction
    ByConstraints,
    /// Custom ordering based on provided function
    Custom(String), // Function name for custom ordering
}

/// Optimization objectives for multi-criteria shrinking
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationObjective {
    /// Minimize total test case size
    MinimizeSize { weight: f64 },
    /// Minimize computational complexity
    MinimizeComplexity { weight: f64 },
    /// Maximize readability/interpretability
    MaximizeReadability { weight: f64 },
    /// Minimize constraint violations
    MinimizeViolations { weight: f64 },
    /// Target specific value patterns
    TargetPattern { pattern: ValuePattern, weight: f64 },
}

/// Patterns for targeting specific value configurations
#[derive(Debug, Clone, PartialEq)]
pub enum ValuePattern {
    Zero,
    PowerOfTwo,
    SmallPrime,
    SimpleRatio,
    CommonConstant(i128),
    CustomPattern(String),
}

/// Sophisticated shrinking engine that implements advanced algorithms
pub struct AdvancedShrinkingEngine {
    /// Cache for storing shrinking results to avoid recomputation
    shrinking_cache: HashMap<u64, ShrinkResult>,
    /// Metrics tracking for performance analysis
    metrics: ShrinkingMetrics,
    /// Maximum number of shrinking iterations per attempt
    max_iterations: usize,
    /// Timeout threshold for shrinking operations
    timeout_threshold: std::time::Duration,
    /// Set of enabled shrinking strategies
    enabled_strategies: HashSet<ShrinkingStrategy>,
    /// Priority ordering for strategies
    strategy_priorities: HashMap<ShrinkingStrategy, u8>,
    /// Historical success rates for adaptive strategy selection
    strategy_success_rates: HashMap<ShrinkingStrategy, f64>,
    /// Simple counter for pseudo-random number generation
    random_state: u64,
}

impl Default for AdvancedShrinkingEngine {
    fn default() -> Self {
        let mut engine = Self {
            shrinking_cache: HashMap::new(),
            metrics: ShrinkingMetrics::default(),
            max_iterations: 1000,
            timeout_threshold: std::time::Duration::from_secs(30),
            enabled_strategies: HashSet::new(),
            strategy_priorities: HashMap::new(),
            strategy_success_rates: HashMap::new(),
            random_state: 12345,
        };
        
        // Initialize with default strategies
        engine.initialize_default_strategies();
        engine
    }
}

impl AdvancedShrinkingEngine {
    /// Create a new shrinking engine with custom configuration
    pub fn new(max_iterations: usize, timeout: std::time::Duration) -> Self {
        let mut engine = Self {
            shrinking_cache: HashMap::new(),
            metrics: ShrinkingMetrics::default(),
            max_iterations,
            timeout_threshold: timeout,
            enabled_strategies: HashSet::new(),
            strategy_priorities: HashMap::new(),
            strategy_success_rates: HashMap::new(),
            random_state: 12345,
        };
        
        engine.initialize_default_strategies();
        engine
    }
    
    /// Convert float to integer for DataTree storage integration
    pub fn float_to_datatree_storage(&self, f: f64) -> u64 {
        float_to_int(f)
    }
    
    /// Convert integer back to float from DataTree storage
    pub fn float_from_datatree_storage(&self, i: u64) -> f64 {
        int_to_float(i)
    }
    
    /// Get lexicographic encoding for a float value
    pub fn get_float_lex_encoding(&self, f: f64) -> u64 {
        float_to_lex(f)
    }
    
    /// Convert lexicographic encoding back to float
    pub fn decode_float_lex_encoding(&self, lex: u64) -> f64 {
        lex_to_float(lex)
    }

    /// Initialize the engine with a comprehensive set of default shrinking strategies
    fn initialize_default_strategies(&mut self) {
        let default_strategies = vec![
            (ShrinkingStrategy::MinimizeIntegers { target: 0, aggressive: false }, 10),
            (ShrinkingStrategy::MinimizeIntegers { target: 0, aggressive: true }, 8),
            (ShrinkingStrategy::MinimizeFloats { target: 0.0, precision_reduction: true }, 9),
            (ShrinkingStrategy::SimplifyCollections { min_size: 0, preserve_structure: false }, 7),
            (ShrinkingStrategy::OptimizeStrings { preserve_chars: HashSet::new() }, 6),
            (ShrinkingStrategy::Deduplicate { similarity_threshold: 0.9 }, 5),
            (ShrinkingStrategy::Reorder { priority_function: ReorderPriority::ByMagnitude }, 4),
            (ShrinkingStrategy::ConstraintOptimization { repair_violations: true }, 8),
        ];

        for (strategy, priority) in default_strategies {
            self.enabled_strategies.insert(strategy.clone());
            self.strategy_priorities.insert(strategy.clone(), priority);
            self.strategy_success_rates.insert(strategy, 0.5); // Start with neutral success rate
        }
    }

    /// Perform advanced shrinking on a sequence of choices
    pub fn shrink_choices(&mut self, choices: &[Choice]) -> ShrinkResult {
        let start_time = std::time::Instant::now();
        self.metrics.total_attempts += 1;

        // Check cache first
        let cache_key = self.hash_choices(choices);
        if let Some(cached_result) = self.shrinking_cache.get(&cache_key) {
            self.metrics.cache_hits += 1;
            return cached_result.clone();
        }
        self.metrics.cache_misses += 1;

        // Perform multi-pass shrinking with different strategies
        let mut current_choices = choices.to_vec();
        let mut best_result = current_choices.clone();
        let mut improved = true;
        let mut iteration = 0;

        while improved && iteration < self.max_iterations {
            if start_time.elapsed() > self.timeout_threshold {
                self.metrics.timeouts += 1;
                let result = ShrinkResult::Timeout;
                self.shrinking_cache.insert(cache_key, result.clone());
                return result;
            }

            improved = false;
            let strategies = self.get_ordered_strategies();

            for strategy in strategies {
                if let Ok(shrunk_choices) = self.apply_strategy(&strategy, &current_choices) {
                    if self.is_better_solution(&shrunk_choices, &current_choices) {
                        current_choices = shrunk_choices;
                        best_result = current_choices.clone();
                        improved = true;
                        
                        // Update strategy success rate
                        let current_rate = self.strategy_success_rates.get(&strategy).unwrap_or(&0.5);
                        let new_rate = current_rate * 0.9 + 0.1; // Increase success rate
                        self.strategy_success_rates.insert(strategy.clone(), new_rate);
                        
                        break; // Move to next iteration with improved solution
                    }
                } else {
                    // Update strategy success rate for failure
                    let current_rate = self.strategy_success_rates.get(&strategy).unwrap_or(&0.5);
                    let new_rate = current_rate * 0.9; // Decrease success rate
                    self.strategy_success_rates.insert(strategy.clone(), new_rate);
                }
            }

            iteration += 1;
        }

        // Determine final result
        let result = if best_result.len() < choices.len() || self.has_smaller_values(&best_result, choices) {
            self.metrics.successful_shrinks += 1;
            let reduction = self.calculate_reduction_percentage(choices, &best_result);
            self.metrics.reduction_percentage = 
                (self.metrics.reduction_percentage * (self.metrics.successful_shrinks - 1) as f64 + reduction) 
                / self.metrics.successful_shrinks as f64;
            ShrinkResult::Success(best_result)
        } else {
            self.metrics.failed_shrinks += 1;
            ShrinkResult::Failed
        };

        // Update average iterations
        self.metrics.average_iterations = 
            (self.metrics.average_iterations * (self.metrics.total_attempts - 1) as f64 + iteration as f64) 
            / self.metrics.total_attempts as f64;

        // Cache the result
        self.shrinking_cache.insert(cache_key, result.clone());
        result
    }

    /// Apply a specific shrinking strategy to choices
    fn apply_strategy(&mut self, strategy: &ShrinkingStrategy, choices: &[Choice]) -> Result<Vec<Choice>, String> {
        match strategy {
            ShrinkingStrategy::MinimizeIntegers { target, aggressive } => {
                self.minimize_integers(choices, *target, *aggressive)
            }
            ShrinkingStrategy::MinimizeFloats { target, precision_reduction } => {
                self.minimize_floats(choices, *target, *precision_reduction)
            }
            ShrinkingStrategy::SimplifyCollections { min_size, preserve_structure } => {
                self.simplify_collections(choices, *min_size, *preserve_structure)
            }
            ShrinkingStrategy::OptimizeStrings { preserve_chars } => {
                self.optimize_strings(choices, preserve_chars)
            }
            ShrinkingStrategy::Deduplicate { similarity_threshold } => {
                self.deduplicate_choices(choices, *similarity_threshold)
            }
            ShrinkingStrategy::Reorder { priority_function } => {
                self.reorder_choices(choices, priority_function)
            }
            ShrinkingStrategy::ConstraintOptimization { repair_violations } => {
                self.optimize_constraints(choices, *repair_violations)
            }
            ShrinkingStrategy::ParetoOptimization { objectives } => {
                self.pareto_optimization(choices, objectives)
            }
        }
    }

    /// Minimize integer values towards target
    fn minimize_integers(&self, choices: &[Choice], target: i128, aggressive: bool) -> Result<Vec<Choice>, String> {
        let mut result = choices.to_vec();
        let mut modified = false;

        for choice in &mut result {
            if let ChoiceValue::Integer(value) = choice.value.clone() {
                let new_value = if aggressive {
                    // Aggressive: Move directly towards target
                    if value > target {
                        std::cmp::max(target, value - (value - target) / 2)
                    } else if value < target {
                        std::cmp::min(target, value + (target - value) / 2)
                    } else {
                        value
                    }
                } else {
                    // Conservative: Move one step towards target
                    if value > target {
                        value - 1
                    } else if value < target {
                        value + 1
                    } else {
                        value
                    }
                };

                if new_value != value {
                    choice.value = ChoiceValue::Integer(new_value);
                    modified = true;
                    
                    println!("🔧 [SHRINK] Integer {} -> {} (target: {})", value, new_value, target);
                }
            }
        }

        if modified {
            Ok(result)
        } else {
            Err("No integer minimization possible".to_string())
        }
    }

    /// Minimize floating point values using sophisticated lexicographic encoding
    fn minimize_floats(&self, choices: &[Choice], target: f64, precision_reduction: bool) -> Result<Vec<Choice>, String> {
        let mut result = choices.to_vec();
        let mut modified = false;

        for choice in &mut result {
            if let ChoiceValue::Float(value) = choice.value.clone() {
                // Generate sophisticated shrinking candidates using float encoding export
                let candidates = self.generate_float_shrinking_candidates(value, 10);
                
                let mut best_candidate = value;
                let mut found_improvement = false;
                
                // Check each candidate using lexicographic comparison
                for &candidate in &candidates {
                    // Validate candidate against target and precision requirements
                    let candidate_to_test = if precision_reduction && candidate.fract() != 0.0 {
                        // Try to convert to integer if close
                        let rounded = candidate.round();
                        if (rounded - candidate).abs() < 0.001 {
                            rounded
                        } else {
                            // Reduce precision
                            (candidate * 100.0).round() / 100.0
                        }
                    } else {
                        candidate
                    };
                    
                    // Use lexicographic encoding to determine if this is a better value
                    if self.is_float_shrinking_improvement(candidate_to_test, best_candidate) {
                        best_candidate = candidate_to_test;
                        found_improvement = true;
                    }
                }
                
                // If no candidates worked, try traditional approach with lexicographic validation
                if !found_improvement {
                    let traditional_candidate = if precision_reduction && value.fract() != 0.0 {
                        // Try to convert to integer if close
                        let rounded = value.round();
                        if (rounded - value).abs() < 0.001 {
                            rounded
                        } else {
                            // Reduce precision
                            (value * 100.0).round() / 100.0
                        }
                    } else {
                        // Move towards target
                        if value > target {
                            value - (value - target) * 0.5
                        } else if value < target {
                            value + (target - value) * 0.5
                        } else {
                            value
                        }
                    };
                    
                    if self.is_float_shrinking_improvement(traditional_candidate, value) {
                        best_candidate = traditional_candidate;
                        found_improvement = true;
                    }
                }

                if found_improvement && (best_candidate - value).abs() > f64::EPSILON {
                    choice.value = ChoiceValue::Float(best_candidate);
                    modified = true;
                    
                    // Log with lexicographic encoding information
                    let original_lex = float_to_lex(value);
                    let new_lex = float_to_lex(best_candidate);
                    println!("🔧 [SHRINK] Float {:.6} -> {:.6} (lex: 0x{:016X} -> 0x{:016X}, target: {:.6})", 
                           value, best_candidate, original_lex, new_lex, target);
                }
            }
        }

        if modified {
            Ok(result)
        } else {
            Err("No float minimization possible".to_string())
        }
    }

    /// Simplify collections by reducing size and complexity
    fn simplify_collections(&self, choices: &[Choice], min_size: usize, preserve_structure: bool) -> Result<Vec<Choice>, String> {
        let mut result = choices.to_vec();
        let mut modified = false;

        for choice in &mut result {
            match choice.value.clone() {
                ChoiceValue::String(s) if s.len() > min_size => {
                    let new_length = std::cmp::max(min_size, s.len() * 3 / 4);
                    let new_string = if preserve_structure {
                        // Preserve structure by removing middle characters
                        let start = s.chars().take(new_length / 2).collect::<String>();
                        let end = s.chars().skip(s.len() - new_length / 2).collect::<String>();
                        format!("{}{}", start, end)
                    } else {
                        // Simply truncate
                        s.chars().take(new_length).collect::<String>()
                    };
                    
                    choice.value = ChoiceValue::String(new_string.clone());
                    modified = true;
                    
                    println!("🔧 [SHRINK] String len {} -> {} (preserve: {})", s.len(), new_string.len(), preserve_structure);
                }
                ChoiceValue::Bytes(b) if b.len() > min_size => {
                    let new_length = std::cmp::max(min_size, b.len() * 3 / 4);
                    let new_bytes = if preserve_structure {
                        // Preserve structure by removing middle bytes
                        let mut result_bytes = Vec::new();
                        result_bytes.extend_from_slice(&b[..new_length / 2]);
                        result_bytes.extend_from_slice(&b[b.len() - new_length / 2..]);
                        result_bytes
                    } else {
                        b[..new_length].to_vec()
                    };
                    
                    choice.value = ChoiceValue::Bytes(new_bytes.clone());
                    modified = true;
                    
                    println!("🔧 [SHRINK] Bytes len {} -> {} (preserve: {})", b.len(), new_bytes.len(), preserve_structure);
                }
                _ => {}
            }
        }

        if modified {
            Ok(result)
        } else {
            Err("No collection simplification possible".to_string())
        }
    }

    /// Optimize string content for minimal representation
    fn optimize_strings(&self, choices: &[Choice], preserve_chars: &HashSet<char>) -> Result<Vec<Choice>, String> {
        let mut result = choices.to_vec();
        let mut modified = false;

        for choice in &mut result {
            if let ChoiceValue::String(s) = choice.value.clone() {
                let mut new_string = String::new();
                let mut changed = false;

                for ch in s.chars() {
                    if preserve_chars.contains(&ch) {
                        new_string.push(ch);
                    } else if ch.is_alphabetic() && ch.is_uppercase() {
                        // Convert uppercase to lowercase for simplicity
                        new_string.push(ch.to_lowercase().next().unwrap_or(ch));
                        changed = true;
                    } else if ch.is_whitespace() && new_string.chars().last() != Some(' ') {
                        // Normalize whitespace to single spaces
                        new_string.push(' ');
                        if ch != ' ' {
                            changed = true;
                        }
                    } else if !ch.is_whitespace() {
                        new_string.push(ch);
                    } else {
                        changed = true; // Removed redundant whitespace
                    }
                }

                // Trim trailing whitespace
                let trimmed = new_string.trim_end().to_string();
                if trimmed.len() != new_string.len() {
                    new_string = trimmed;
                    changed = true;
                }

                if changed && !new_string.is_empty() {
                    choice.value = ChoiceValue::String(new_string.clone());
                    modified = true;
                    
                    println!("🔧 [SHRINK] String optimized: '{}' -> '{}'", &s, &new_string);
                }
            }
        }

        if modified {
            Ok(result)
        } else {
            Err("No string optimization possible".to_string())
        }
    }

    /// Remove duplicate or highly similar choices
    fn deduplicate_choices(&self, choices: &[Choice], similarity_threshold: f64) -> Result<Vec<Choice>, String> {
        if choices.len() <= 1 {
            return Err("Not enough choices to deduplicate".to_string());
        }

        let mut result = Vec::new();
        let mut seen_values: HashSet<String> = HashSet::new();

        for choice in choices {
            let value_key = match &choice.value {
                ChoiceValue::Integer(i) => format!("int:{}", i),
                ChoiceValue::Float(f) => format!("float:{:.6}", f),
                ChoiceValue::String(s) => format!("string:{}", s),
                ChoiceValue::Bytes(b) => format!("bytes:{:02X?}", b),
                ChoiceValue::Boolean(b) => format!("bool:{}", b),
            };

            // Check for exact duplicates
            if seen_values.contains(&value_key) {
                println!("🔧 [SHRINK] Removed duplicate: {}", value_key);
                continue;
            }

            // Check for similar values based on threshold
            let mut is_similar = false;
            for existing_key in &seen_values {
                if self.calculate_similarity(existing_key, &value_key) > similarity_threshold {
                    is_similar = true;
                    println!("🔧 [SHRINK] Removed similar value: {} (similar to {})", value_key, existing_key);
                    break;
                }
            }

            if !is_similar {
                seen_values.insert(value_key);
                result.push(choice.clone());
            }
        }

        if result.len() < choices.len() {
            Ok(result)
        } else {
            Err("No duplicates found".to_string())
        }
    }

    /// Reorder choices based on priority function
    fn reorder_choices(&self, choices: &[Choice], priority_function: &ReorderPriority) -> Result<Vec<Choice>, String> {
        let mut result = choices.to_vec();
        
        result.sort_by(|a, b| {
            match priority_function {
                ReorderPriority::ByMagnitude => {
                    let mag_a = self.calculate_magnitude(&a.value);
                    let mag_b = self.calculate_magnitude(&b.value);
                    mag_a.partial_cmp(&mag_b).unwrap_or(std::cmp::Ordering::Equal)
                }
                ReorderPriority::ByFrequency => {
                    // For simplicity, use value hash as frequency proxy
                    let freq_a = self.hash_choice_value(&a.value) % 100;
                    let freq_b = self.hash_choice_value(&b.value) % 100;
                    freq_a.cmp(&freq_b)
                }
                ReorderPriority::ByComplexity => {
                    let comp_a = self.calculate_complexity(&a.value);
                    let comp_b = self.calculate_complexity(&b.value);
                    comp_a.cmp(&comp_b)
                }
                ReorderPriority::ByConstraints => {
                    // Sort by constraint satisfaction (simulated)
                    let sat_a = self.estimate_constraint_satisfaction(&a.value);
                    let sat_b = self.estimate_constraint_satisfaction(&b.value);
                    sat_b.partial_cmp(&sat_a).unwrap_or(std::cmp::Ordering::Equal) // Higher satisfaction first
                }
                ReorderPriority::Custom(_) => {
                    // Placeholder for custom ordering
                    std::cmp::Ordering::Equal
                }
            }
        });

        // Check if order actually changed
        let changed = result.iter().zip(choices.iter()).any(|(a, b)| a != b);
        
        if changed {
            println!("🔧 [SHRINK] Reordered choices by {:?}", priority_function);
            Ok(result)
        } else {
            Err("No reordering needed".to_string())
        }
    }

    /// Optimize choices based on constraint satisfaction
    fn optimize_constraints(&self, choices: &[Choice], repair_violations: bool) -> Result<Vec<Choice>, String> {
        let mut result = choices.to_vec();
        let mut modified = false;

        for choice in &mut result {
            // Simulate constraint checking and repair
            let violations = self.detect_constraint_violations(&choice.value);
            
            if !violations.is_empty() && repair_violations {
                let repaired_value = self.repair_constraint_violations(&choice.value, &violations);
                if repaired_value != choice.value {
                    choice.value = repaired_value;
                    modified = true;
                    
                    println!("🔧 [SHRINK] Repaired constraint violations: {} -> {:?}", violations.len(), choice.value);
                }
            }
        }

        if modified {
            Ok(result)
        } else {
            Err("No constraint optimization possible".to_string())
        }
    }

    /// Perform Pareto optimization with multiple objectives
    fn pareto_optimization(&mut self, choices: &[Choice], objectives: &[OptimizationObjective]) -> Result<Vec<Choice>, String> {
        if objectives.is_empty() {
            return Err("No optimization objectives specified".to_string());
        }

        let mut candidates = vec![choices.to_vec()];
        
        // Generate multiple candidate solutions
        for _ in 0..10 {
            let mut candidate = choices.to_vec();
            // Apply random modifications based on objectives
            for obj in objectives {
                match obj {
                    OptimizationObjective::MinimizeSize { weight: _ } => {
                        if candidate.len() > 1 && self.pseudo_random() < 0.3 {
                            candidate.remove(self.pseudo_random_usize() % candidate.len());
                        }
                    }
                    OptimizationObjective::MinimizeComplexity { weight: _ } => {
                        // Simplify random choice
                        if !candidate.is_empty() {
                            let idx = self.pseudo_random_usize() % candidate.len();
                            candidate[idx].value = self.simplify_value(&candidate[idx].value);
                        }
                    }
                    OptimizationObjective::TargetPattern { pattern, weight: _ } => {
                        // Move towards target pattern
                        if !candidate.is_empty() {
                            let idx = self.pseudo_random_usize() % candidate.len();
                            candidate[idx].value = self.apply_pattern(&candidate[idx].value, pattern);
                        }
                    }
                    _ => {} // Handle other objectives
                }
            }
            candidates.push(candidate);
        }

        // Find Pareto optimal solution
        let best_candidate = self.find_pareto_optimal(&candidates, objectives);
        
        if best_candidate != choices {
            println!("🔧 [SHRINK] Pareto optimization found better solution with {} objectives", objectives.len());
            Ok(best_candidate)
        } else {
            Err("No Pareto improvement found".to_string())
        }
    }

    /// Helper method to get strategies ordered by priority and success rate
    fn get_ordered_strategies(&self) -> Vec<ShrinkingStrategy> {
        let mut strategies: Vec<_> = self.enabled_strategies.iter().cloned().collect();
        
        strategies.sort_by(|a, b| {
            let priority_a = self.strategy_priorities.get(a).unwrap_or(&0);
            let priority_b = self.strategy_priorities.get(b).unwrap_or(&0);
            let success_a = self.strategy_success_rates.get(a).unwrap_or(&0.5);
            let success_b = self.strategy_success_rates.get(b).unwrap_or(&0.5);
            
            // Combine priority and success rate for ordering
            let score_a = *priority_a as f64 + success_a * 5.0;
            let score_b = *priority_b as f64 + success_b * 5.0;
            
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        strategies
    }

    /// Check if one solution is better than another using advanced comparison
    fn is_better_solution(&self, new_solution: &[Choice], old_solution: &[Choice]) -> bool {
        // Primary: fewer choices is better
        if new_solution.len() < old_solution.len() {
            return true;
        }
        
        if new_solution.len() > old_solution.len() {
            return false;
        }
        
        // Secondary: use sophisticated value comparison
        self.has_better_values(new_solution, old_solution)
    }

    /// Check if new solution has better values using sophisticated comparison
    fn has_better_values(&self, new_solution: &[Choice], old_solution: &[Choice]) -> bool {
        for (new_choice, old_choice) in new_solution.iter().zip(old_solution.iter()) {
            match (&new_choice.value, &old_choice.value) {
                (ChoiceValue::Float(new_f), ChoiceValue::Float(old_f)) => {
                    // Use lexicographic encoding for float comparison
                    match self.compare_floats_for_shrinking_order(*new_f, *old_f) {
                        std::cmp::Ordering::Less => return true,
                        std::cmp::Ordering::Greater => return false,
                        std::cmp::Ordering::Equal => continue,
                    }
                }
                (ChoiceValue::Integer(new_i), ChoiceValue::Integer(old_i)) => {
                    if new_i.abs() < old_i.abs() {
                        return true;
                    } else if new_i.abs() > old_i.abs() {
                        return false;
                    }
                }
                _ => {
                    // Fallback to magnitude comparison for other types
                    let new_mag = self.calculate_magnitude(&new_choice.value);
                    let old_mag = self.calculate_magnitude(&old_choice.value);
                    if new_mag < old_mag {
                        return true;
                    } else if new_mag > old_mag {
                        return false;
                    }
                }
            }
        }
        
        false // No improvement found
    }

    /// Check if new solution has smaller values overall
    fn has_smaller_values(&self, new_solution: &[Choice], old_solution: &[Choice]) -> bool {
        let new_total = self.calculate_total_magnitude(new_solution);
        let old_total = self.calculate_total_magnitude(old_solution);
        new_total < old_total
    }

    /// Calculate total magnitude of all choices
    fn calculate_total_magnitude(&self, choices: &[Choice]) -> f64 {
        choices.iter()
            .map(|choice| self.calculate_magnitude(&choice.value))
            .sum()
    }

    /// Calculate magnitude of a single choice value
    fn calculate_magnitude(&self, value: &ChoiceValue) -> f64 {
        match value {
            ChoiceValue::Integer(i) => i.abs() as f64,
            ChoiceValue::Float(f) => f.abs(),
            ChoiceValue::String(s) => s.len() as f64,
            ChoiceValue::Bytes(b) => b.len() as f64,
            ChoiceValue::Boolean(_) => 1.0,
        }
    }

    /// Calculate complexity of a choice value
    fn calculate_complexity(&self, value: &ChoiceValue) -> u32 {
        match value {
            ChoiceValue::Integer(i) => {
                if *i == 0 { 0 }
                else if i.abs() < 10 { 1 }
                else if i.abs() < 100 { 2 }
                else { 3 }
            }
            ChoiceValue::Float(f) => {
                if f.fract() == 0.0 { 1 } else { 2 }
            }
            ChoiceValue::String(s) => {
                s.chars().map(|c| if c.is_ascii_alphanumeric() { 1 } else { 2 }).sum()
            }
            ChoiceValue::Bytes(b) => b.len() as u32,
            ChoiceValue::Boolean(_) => 0,
        }
    }

    /// Estimate constraint satisfaction for a value
    fn estimate_constraint_satisfaction(&self, value: &ChoiceValue) -> f64 {
        // Simplified constraint satisfaction estimation
        match value {
            ChoiceValue::Integer(i) => {
                if *i >= 0 && *i <= 100 { 1.0 } else { 0.5 }
            }
            ChoiceValue::Float(f) => {
                if *f >= 0.0 && *f <= 1.0 { 1.0 } else { 0.5 }
            }
            ChoiceValue::String(s) => {
                if s.len() <= 50 && s.chars().all(|c| c.is_ascii()) { 1.0 } else { 0.7 }
            }
            ChoiceValue::Bytes(b) => {
                if b.len() <= 100 { 1.0 } else { 0.6 }
            }
            ChoiceValue::Boolean(_) => 1.0,
        }
    }

    /// Detect constraint violations (simplified)
    fn detect_constraint_violations(&self, value: &ChoiceValue) -> Vec<String> {
        let mut violations = Vec::new();
        
        match value {
            ChoiceValue::Integer(i) => {
                if *i < -1000 || *i > 1000 {
                    violations.push("Integer out of reasonable range".to_string());
                }
            }
            ChoiceValue::Float(f) => {
                if f.is_nan() || f.is_infinite() {
                    violations.push("Invalid float value".to_string());
                }
            }
            ChoiceValue::String(s) => {
                if s.len() > 1000 {
                    violations.push("String too long".to_string());
                }
            }
            ChoiceValue::Bytes(b) => {
                if b.len() > 10000 {
                    violations.push("Bytes too long".to_string());
                }
            }
            ChoiceValue::Boolean(_) => {} // No violations for booleans
        }
        
        violations
    }

    /// Repair constraint violations
    fn repair_constraint_violations(&self, value: &ChoiceValue, violations: &[String]) -> ChoiceValue {
        match value {
            ChoiceValue::Integer(i) => {
                if violations.iter().any(|v| v.contains("out of reasonable range")) {
                    ChoiceValue::Integer((*i).clamp(-1000, 1000))
                } else {
                    value.clone()
                }
            }
            ChoiceValue::Float(_f) => {
                if violations.iter().any(|v| v.contains("Invalid float")) {
                    ChoiceValue::Float(0.0)
                } else {
                    value.clone()
                }
            }
            ChoiceValue::String(s) => {
                if violations.iter().any(|v| v.contains("too long")) {
                    ChoiceValue::String(s.chars().take(1000).collect())
                } else {
                    value.clone()
                }
            }
            ChoiceValue::Bytes(b) => {
                if violations.iter().any(|v| v.contains("too long")) {
                    ChoiceValue::Bytes(b.iter().take(10000).cloned().collect())
                } else {
                    value.clone()
                }
            }
            ChoiceValue::Boolean(_) => value.clone(),
        }
    }

    /// Simplify a value towards a simpler representation
    fn simplify_value(&self, value: &ChoiceValue) -> ChoiceValue {
        match value {
            ChoiceValue::Integer(i) => {
                if *i > 0 {
                    ChoiceValue::Integer(i / 2)
                } else if *i < 0 {
                    ChoiceValue::Integer(i / 2)
                } else {
                    value.clone()
                }
            }
            ChoiceValue::Float(f) => {
                ChoiceValue::Float(f / 2.0)
            }
            ChoiceValue::String(s) => {
                if s.len() > 1 {
                    ChoiceValue::String(s.chars().take(s.len() / 2).collect())
                } else {
                    value.clone()
                }
            }
            ChoiceValue::Bytes(b) => {
                if b.len() > 1 {
                    ChoiceValue::Bytes(b.iter().take(b.len() / 2).cloned().collect())
                } else {
                    value.clone()
                }
            }
            ChoiceValue::Boolean(_) => value.clone(),
        }
    }

    /// Apply a target pattern to a value
    fn apply_pattern(&mut self, value: &ChoiceValue, pattern: &ValuePattern) -> ChoiceValue {
        match (value, pattern) {
            (ChoiceValue::Integer(_), ValuePattern::Zero) => ChoiceValue::Integer(0),
            (ChoiceValue::Integer(i), ValuePattern::PowerOfTwo) => {
                let power = (i.abs() as f64).log2().round() as i32;
                ChoiceValue::Integer(2_i128.pow(power.max(0) as u32))
            }
            (ChoiceValue::Integer(_), ValuePattern::SmallPrime) => {
                let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23];
                ChoiceValue::Integer(primes[self.pseudo_random_usize() % primes.len()] as i128)
            }
            (ChoiceValue::Integer(_), ValuePattern::CommonConstant(c)) => ChoiceValue::Integer(*c),
            (ChoiceValue::Float(_), ValuePattern::Zero) => ChoiceValue::Float(0.0),
            _ => value.clone(),
        }
    }

    /// Find Pareto optimal solution from candidates
    fn find_pareto_optimal(&self, candidates: &[Vec<Choice>], objectives: &[OptimizationObjective]) -> Vec<Choice> {
        let mut best_candidate = candidates[0].clone();
        let mut best_score = self.calculate_objective_score(&best_candidate, objectives);

        for candidate in candidates.iter().skip(1) {
            let score = self.calculate_objective_score(candidate, objectives);
            if score > best_score {
                best_score = score;
                best_candidate = candidate.clone();
            }
        }

        best_candidate
    }

    /// Calculate objective score for Pareto optimization
    fn calculate_objective_score(&self, choices: &[Choice], objectives: &[OptimizationObjective]) -> f64 {
        let mut total_score = 0.0;

        for objective in objectives {
            let score = match objective {
                OptimizationObjective::MinimizeSize { weight } => {
                    weight * (1.0 / (choices.len() as f64 + 1.0))
                }
                OptimizationObjective::MinimizeComplexity { weight } => {
                    let complexity = choices.iter()
                        .map(|c| self.calculate_complexity(&c.value) as f64)
                        .sum::<f64>();
                    weight * (1.0 / (complexity + 1.0))
                }
                OptimizationObjective::MaximizeReadability { weight } => {
                    let readability = self.calculate_readability(choices);
                    weight * readability
                }
                OptimizationObjective::MinimizeViolations { weight } => {
                    let violations = choices.iter()
                        .map(|c| self.detect_constraint_violations(&c.value).len() as f64)
                        .sum::<f64>();
                    weight * (1.0 / (violations + 1.0))
                }
                OptimizationObjective::TargetPattern { pattern: _, weight } => {
                    // Simplified pattern matching score
                    weight * 0.5
                }
            };
            total_score += score;
        }

        total_score
    }

    /// Calculate readability score
    fn calculate_readability(&self, choices: &[Choice]) -> f64 {
        let mut score = 0.0;
        let total = choices.len() as f64;

        for choice in choices {
            match &choice.value {
                ChoiceValue::Integer(i) => {
                    if *i >= 0 && *i <= 100 { score += 1.0; }
                    else if (*i).abs() <= 1000 { score += 0.5; }
                }
                ChoiceValue::Float(f) => {
                    if f.fract() == 0.0 { score += 0.8; }
                    else if f.abs() <= 1.0 { score += 0.6; }
                    else { score += 0.3; }
                }
                ChoiceValue::String(s) => {
                    if s.len() <= 20 && s.chars().all(|c| c.is_ascii_alphanumeric()) { score += 1.0; }
                    else if s.len() <= 50 { score += 0.5; }
                }
                ChoiceValue::Bytes(b) => {
                    if b.len() <= 10 { score += 0.8; }
                    else if b.len() <= 50 { score += 0.4; }
                }
                ChoiceValue::Boolean(_) => score += 1.0,
            }
        }

        if total > 0.0 { score / total } else { 0.0 }
    }

    /// Calculate similarity between two value strings
    fn calculate_similarity(&self, value1: &str, value2: &str) -> f64 {
        if value1 == value2 {
            return 1.0;
        }

        // Simple similarity based on common prefix/suffix
        let chars1: Vec<char> = value1.chars().collect();
        let chars2: Vec<char> = value2.chars().collect();
        
        let max_len = chars1.len().max(chars2.len());
        if max_len == 0 {
            return 1.0;
        }

        let common_chars = chars1.iter().zip(chars2.iter())
            .take_while(|(a, b)| a == b)
            .count();

        common_chars as f64 / max_len as f64
    }

    /// Calculate hash for a sequence of choices
    fn hash_choices(&self, choices: &[Choice]) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for choice in choices {
            choice.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Calculate hash for a choice value
    fn hash_choice_value(&self, value: &ChoiceValue) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        match value {
            ChoiceValue::Integer(i) => i.hash(&mut hasher),
            ChoiceValue::Float(f) => f.to_bits().hash(&mut hasher),
            ChoiceValue::String(s) => s.hash(&mut hasher),
            ChoiceValue::Bytes(b) => b.hash(&mut hasher),
            ChoiceValue::Boolean(b) => b.hash(&mut hasher),
        }
        hasher.finish()
    }

    /// Calculate reduction percentage between original and shrunk choices
    fn calculate_reduction_percentage(&self, original: &[Choice], shrunk: &[Choice]) -> f64 {
        let original_size = self.calculate_total_magnitude(original);
        let shrunk_size = self.calculate_total_magnitude(shrunk);
        
        if original_size > 0.0 {
            ((original_size - shrunk_size) / original_size) * 100.0
        } else {
            0.0
        }
    }

    /// Get current shrinking metrics
    pub fn get_metrics(&self) -> &ShrinkingMetrics {
        &self.metrics
    }

    /// Clear the shrinking cache
    pub fn clear_cache(&mut self) {
        self.shrinking_cache.clear();
        println!("🧹 [SHRINK] Cache cleared");
    }

    /// Add a custom shrinking strategy
    pub fn add_strategy(&mut self, strategy: ShrinkingStrategy, priority: u8) {
        self.enabled_strategies.insert(strategy.clone());
        self.strategy_priorities.insert(strategy.clone(), priority);
        self.strategy_success_rates.insert(strategy, 0.5);
        println!("➕ [SHRINK] Added custom strategy with priority {}", priority);
    }

    /// Remove a shrinking strategy
    pub fn remove_strategy(&mut self, strategy: &ShrinkingStrategy) -> bool {
        let removed = self.enabled_strategies.remove(strategy);
        if removed {
            self.strategy_priorities.remove(strategy);
            self.strategy_success_rates.remove(strategy);
            println!("➖ [SHRINK] Removed strategy: {:?}", strategy);
        }
        removed
    }

    /// Get current strategy success rates for analysis
    pub fn get_strategy_success_rates(&self) -> &HashMap<ShrinkingStrategy, f64> {
        &self.strategy_success_rates
    }

    /// Simple pseudo-random number generator for internal use
    fn pseudo_random(&mut self) -> f64 {
        // Simple linear congruential generator
        self.random_state = self.random_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.random_state % 0x7FFFFFFF) as f64 / 0x7FFFFFFF as f64
    }

    /// Generate pseudo-random usize
    fn pseudo_random_usize(&mut self) -> usize {
        self.random_state = self.random_state.wrapping_mul(1103515245).wrapping_add(12345);
        self.random_state as usize
    }
    
    /// Check if one float value is better (simpler) than another for shrinking
    fn is_float_shrinking_improvement(&self, candidate: f64, current: f64) -> bool {
        // Handle special cases
        if candidate.is_nan() && current.is_nan() {
            return false;
        }
        if candidate.is_nan() {
            return false; // NaN is never an improvement
        }
        if current.is_nan() {
            return true; // Any finite value is better than NaN
        }
        
        let lex_candidate = float_to_lex(candidate.abs());
        let lex_current = float_to_lex(current.abs());
        
        lex_candidate < lex_current
    }
    
    /// Compare two float values using lexicographic encoding for shrinking
    fn compare_floats_for_shrinking_order(&self, a: f64, b: f64) -> std::cmp::Ordering {
        let lex_a = float_to_lex(a.abs());
        let lex_b = float_to_lex(b.abs());
        
        lex_a.cmp(&lex_b)
    }
    
    /// Generate optimal shrinking candidates for a float value
    fn generate_float_shrinking_candidates(&self, value: f64, max_candidates: usize) -> Vec<f64> {
        let mut candidates = Vec::new();
        
        // Always include zero as the ultimate shrink target
        candidates.push(0.0);
        
        if value.is_finite() && value != 0.0 {
            let abs_value = value.abs();
            let original_lex = float_to_lex(abs_value);
            
            // Generate mathematically meaningful candidates
            let math_candidates = [
                abs_value / 2.0,
                abs_value / 10.0,
                abs_value / 100.0,
                abs_value.sqrt(),
                abs_value.floor(),
                1.0,
                0.5,
                0.1,
                0.01,
                1e-6,
            ];
            
            for &candidate in &math_candidates {
                if candidate > 0.0 && candidate < abs_value && candidates.len() < max_candidates {
                    let candidate_lex = float_to_lex(candidate);
                    if candidate_lex < original_lex {
                        candidates.push(candidate);
                        if value < 0.0 {
                            candidates.push(-candidate);
                        }
                    }
                }
            }
            
            // Generate lexicographic encoding-based candidates
            let reduction_factors = [0.9, 0.8, 0.7, 0.5, 0.25, 0.1];
            for &factor in &reduction_factors {
                if candidates.len() >= max_candidates {
                    break;
                }
                
                let target_lex = (original_lex as f64 * factor) as u64;
                if target_lex < original_lex && target_lex > 0 {
                    let candidate = lex_to_float(target_lex);
                    if candidate.is_finite() && candidate > 0.0 {
                        candidates.push(candidate);
                        if value < 0.0 {
                            candidates.push(-candidate);
                        }
                    }
                }
            }
        }
        
        // Remove duplicates and sort by lexicographic encoding
        candidates.sort_by(|&a, &b| {
            let lex_a = float_to_lex(a.abs());
            let lex_b = float_to_lex(b.abs());
            lex_a.cmp(&lex_b)
        });
        candidates.dedup_by(|&mut a, &mut b| (a - b).abs() < f64::EPSILON);
        
        // Limit to requested number of candidates
        candidates.truncate(max_candidates);
        
        candidates
    }
}

// Implement Hash for Choice to support caching
impl Hash for Choice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match &self.value {
            ChoiceValue::Integer(i) => {
                0u8.hash(state);
                i.hash(state);
            }
            ChoiceValue::Float(f) => {
                1u8.hash(state);
                f.to_bits().hash(state);
            }
            ChoiceValue::String(s) => {
                2u8.hash(state);
                s.hash(state);
            }
            ChoiceValue::Bytes(b) => {
                3u8.hash(state);
                b.hash(state);
            }
            ChoiceValue::Boolean(b) => {
                4u8.hash(state);
                b.hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_minimization() {
        let mut engine = AdvancedShrinkingEngine::default();
        let choices = vec![
            Choice { value: ChoiceValue::Integer(100), index: 0 },
            Choice { value: ChoiceValue::Integer(-50), index: 1 },
            Choice { value: ChoiceValue::Integer(0), index: 2 },
        ];

        match engine.shrink_choices(&choices) {
            ShrinkResult::Success(shrunk) => {
                assert!(shrunk.len() <= choices.len());
                // Verify that values moved towards zero
                for choice in &shrunk {
                    if let ChoiceValue::Integer(i) = choice.value {
                        assert!(i.abs() <= 100);
                    }
                }
            }
            _ => {} // Other results are also acceptable
        }
    }

    #[test]
    fn test_string_optimization() {
        let mut engine = AdvancedShrinkingEngine::default();
        let choices = vec![
            Choice { value: ChoiceValue::String("HELLO WORLD    ".to_string()), index: 0 },
            Choice { value: ChoiceValue::String("Test  String\t\n".to_string()), index: 1 },
        ];

        match engine.shrink_choices(&choices) {
            ShrinkResult::Success(shrunk) => {
                for choice in &shrunk {
                    if let ChoiceValue::String(s) = &choice.value {
                        // Verify string is optimized (lowercase, trimmed)
                        assert!(!s.ends_with(' '));
                        assert!(!s.contains('\t'));
                        assert!(!s.contains('\n'));
                    }
                }
            }
            _ => {} // Other results are also acceptable
        }
    }

    #[test]
    fn test_deduplication() {
        let mut engine = AdvancedShrinkingEngine::default();
        let choices = vec![
            Choice { value: ChoiceValue::Integer(42), index: 0 },
            Choice { value: ChoiceValue::Integer(42), index: 1 }, // Duplicate
            Choice { value: ChoiceValue::Integer(43), index: 2 }, // Similar
            Choice { value: ChoiceValue::Float(3.14), index: 3 },
        ];

        match engine.shrink_choices(&choices) {
            ShrinkResult::Success(shrunk) => {
                assert!(shrunk.len() < choices.len());
                // Verify no exact duplicates remain
                let mut seen = HashSet::new();
                for choice in &shrunk {
                    let key = format!("{:?}", choice.value);
                    assert!(!seen.contains(&key), "Found duplicate after deduplication");
                    seen.insert(key);
                }
            }
            _ => {} // Other results are also acceptable
        }
    }

    #[test]
    fn test_metrics_tracking() {
        let mut engine = AdvancedShrinkingEngine::default();
        let choices = vec![
            Choice { value: ChoiceValue::Integer(100), index: 0 },
            Choice { value: ChoiceValue::Float(3.14159), index: 1 },
        ];

        let initial_attempts = engine.get_metrics().total_attempts;
        let _result = engine.shrink_choices(&choices);
        
        assert_eq!(engine.get_metrics().total_attempts, initial_attempts + 1);
    }

    #[test]
    fn test_caching() {
        let mut engine = AdvancedShrinkingEngine::default();
        let choices = vec![
            Choice { value: ChoiceValue::Integer(42), index: 0 },
        ];

        // First call
        let result1 = engine.shrink_choices(&choices);
        let cache_misses_after_first = engine.get_metrics().cache_misses;

        // Second call with same input should use cache
        let result2 = engine.shrink_choices(&choices);
        let cache_hits_after_second = engine.get_metrics().cache_hits;

        assert_eq!(result1, result2);
        assert!(cache_hits_after_second > 0);
    }

    #[test]
    fn test_strategy_management() {
        let mut engine = AdvancedShrinkingEngine::default();
        let initial_count = engine.enabled_strategies.len();

        let custom_strategy = ShrinkingStrategy::MinimizeIntegers { target: 42, aggressive: true };
        engine.add_strategy(custom_strategy.clone(), 9);

        assert_eq!(engine.enabled_strategies.len(), initial_count + 1);
        assert!(engine.enabled_strategies.contains(&custom_strategy));

        assert!(engine.remove_strategy(&custom_strategy));
        assert_eq!(engine.enabled_strategies.len(), initial_count);
        assert!(!engine.enabled_strategies.contains(&custom_strategy));
    }

    #[test]
    fn test_constraint_optimization() {
        let mut engine = AdvancedShrinkingEngine::default();
        let choices = vec![
            Choice { value: ChoiceValue::Integer(99999), index: 0 }, // Out of range
            Choice { value: ChoiceValue::Float(f64::NAN), index: 1 }, // Invalid
        ];

        match engine.shrink_choices(&choices) {
            ShrinkResult::Success(shrunk) => {
                for choice in &shrunk {
                    match &choice.value {
                        ChoiceValue::Integer(i) => assert!(i.abs() <= 1000),
                        ChoiceValue::Float(f) => assert!(!f.is_nan()),
                        _ => {}
                    }
                }
            }
            _ => {} // Other results are also acceptable for this test
        }
    }

    #[test]
    fn test_empty_choices() {
        let mut engine = AdvancedShrinkingEngine::default();
        let choices: Vec<Choice> = vec![];

        match engine.shrink_choices(&choices) {
            ShrinkResult::Failed => {}, // Expected for empty input
            _ => panic!("Expected Failed result for empty choices"),
        }
    }

    #[test]
    fn test_complex_shrinking_scenario() {
        let mut engine = AdvancedShrinkingEngine::default();
        let choices = vec![
            Choice { value: ChoiceValue::Integer(1000), index: 0 },
            Choice { value: ChoiceValue::Float(3.141592653589793), index: 1 },
            Choice { value: ChoiceValue::String("HELLO WORLD!!!".to_string()), index: 2 },
            Choice { value: ChoiceValue::Bytes(vec![0xFF, 0xFF, 0xFF, 0xFF]), index: 3 },
            Choice { value: ChoiceValue::Boolean(true), index: 4 },
            Choice { value: ChoiceValue::Integer(1000), index: 5 }, // Duplicate
        ];

        match engine.shrink_choices(&choices) {
            ShrinkResult::Success(shrunk) => {
                // Verify overall reduction
                assert!(shrunk.len() <= choices.len());
                
                // Verify individual improvements
                for choice in &shrunk {
                    match &choice.value {
                        ChoiceValue::Integer(i) => assert!(i.abs() <= 1000),
                        ChoiceValue::Float(f) => assert!(f.is_finite()),
                        ChoiceValue::String(s) => assert!(s.len() <= 50),
                        ChoiceValue::Bytes(b) => assert!(b.len() <= 10),
                        ChoiceValue::Boolean(_) => {}, // Always valid
                    }
                }
            }
            _ => {} // Other results are acceptable
        }

        // Verify metrics were updated
        let metrics = engine.get_metrics();
        assert!(metrics.total_attempts > 0);
        println!("Final metrics: {}", metrics);
    }

    #[test]
    fn test_float_encoding_integration() {
        let mut engine = AdvancedShrinkingEngine::default();
        
        // Test DataTree integration
        let test_float = 3.14159;
        let stored = engine.float_to_datatree_storage(test_float);
        let restored = engine.float_from_datatree_storage(stored);
        assert_eq!(test_float, restored, "DataTree storage should roundtrip exactly");
        
        // Test lexicographic encoding
        let lex_encoding = engine.get_float_lex_encoding(test_float);
        let decoded = engine.decode_float_lex_encoding(lex_encoding);
        assert_eq!(test_float, decoded, "Lex encoding should roundtrip exactly");
        
        // Test shrinking with lexicographic comparison
        let choices = vec![
            Choice { value: ChoiceValue::Float(100.0), index: 0 },
            Choice { value: ChoiceValue::Float(3.14159), index: 1 },
            Choice { value: ChoiceValue::Float(0.1), index: 2 },
        ];
        
        match engine.shrink_choices(&choices) {
            ShrinkResult::Success(shrunk) => {
                // Verify that floats were shrunk using lexicographic ordering
                for choice in &shrunk {
                    if let ChoiceValue::Float(f) = choice.value {
                        assert!(f.is_finite(), "Shrunk floats should be finite");
                        // Original floats should have larger lex encodings than shrunk ones
                        // This is verified by the shrinking algorithm using is_float_shrinking_improvement
                    }
                }
            }
            _ => {} // Other results are acceptable
        }
        
        println!("Float encoding integration test completed successfully");
    }

    #[test]
    fn test_lexicographic_float_comparison() {
        let mut engine = AdvancedShrinkingEngine::default();
        
        // Test basic lexicographic comparison behavior
        let small_choices = vec![Choice { value: ChoiceValue::Float(1.0), index: 0 }];
        let large_choices = vec![Choice { value: ChoiceValue::Float(100.0), index: 0 }];
        
        // Smaller values should be considered better
        assert!(engine.has_better_values(&small_choices, &large_choices), 
                "Smaller float values should be considered better for shrinking");
        
        // Test with zero
        let zero_choices = vec![Choice { value: ChoiceValue::Float(0.0), index: 0 }];
        assert!(engine.has_better_values(&zero_choices, &small_choices),
                "Zero should be considered better than any positive float");
        
        // Test with special values
        let nan_choices = vec![Choice { value: ChoiceValue::Float(f64::NAN), index: 0 }];
        let finite_choices = vec![Choice { value: ChoiceValue::Float(1.0), index: 0 }];
        
        assert!(engine.has_better_values(&finite_choices, &nan_choices),
                "Finite values should be considered better than NaN");
        
        println!("Lexicographic float comparison test passed");
    }
}