//! Choice-aware shrinking implementation
//! 
//! This module implements modern shrinking algorithms that leverage choice metadata
//! to produce high-quality minimal examples. Unlike byte-stream shrinking, choice-aware
//! shrinking understands the semantic structure of the data being shrunk.

use crate::choice::{ChoiceNode, ChoiceValue, Constraints, ChoiceType};
use crate::data::{ConjectureResult, Status, ExtraInformation};
use crate::datatree::DataTree;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

/// Hash representation of a choice sequence for deduplication
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChoiceSequenceHash {
    /// Hash of the choice values and types
    pub sequence_hash: u64,
    /// Length of the sequence
    pub length: usize,
}

impl ChoiceSequenceHash {
    /// Create a hash from a choice sequence
    pub fn from_choices(choices: &[ChoiceNode]) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        // Hash each choice's type and value
        for choice in choices {
            choice.choice_type.hash(&mut hasher);
            choice.value.hash(&mut hasher);
        }
        
        Self {
            sequence_hash: hasher.finish(),
            length: choices.len(),
        }
    }
}

/// Cache entry for storing shrinking attempt results
#[derive(Debug, Clone)]
pub struct ShrinkingCacheEntry {
    /// Whether this transformation was successful
    pub successful: bool,
    /// The resulting choice sequence (if successful)
    pub result_hash: Option<ChoiceSequenceHash>,
    /// Timestamp when this was cached
    pub timestamp: std::time::Instant,
    /// How many times this has been hit
    pub hit_count: u32,
    /// Average execution time for this transformation (microseconds)
    pub avg_execution_time: u64,
    /// Priority score for cache eviction (higher = keep longer)
    pub priority_score: f64,
}

/// Metrics for tracking shrinking performance
#[derive(Debug, Clone, Default)]
pub struct ShrinkingMetrics {
    /// Total number of transformation attempts
    pub total_attempts: u32,
    /// Number of transformations skipped due to deduplication
    pub deduplication_skips: u32,
    /// Number of transformations skipped due to cache hits
    pub cache_hits: u32,
    /// Number of successful transformations
    pub successful_transformations: u32,
    /// Time spent in transformation functions
    pub transformation_time: std::time::Duration,
    /// Time spent in test execution
    pub test_time: std::time::Duration,
    /// Number of cache evictions performed
    pub cache_evictions: u32,
    /// Memory saved by deduplication (bytes estimated)
    pub memory_saved_by_deduplication: u64,
    /// Average cache hit ratio
    pub cache_hit_ratio: f64,
}

/// Multi-level cache structure for comprehensive caching optimization
#[derive(Debug)]
pub struct MultiLevelCache {
    /// L1 cache for recently used transformations (fastest access)
    l1_transformation_cache: HashMap<(String, ChoiceSequenceHash), ShrinkingCacheEntry>,
    /// L2 cache for choice sequence results (medium access speed)
    l2_sequence_cache: HashMap<ChoiceSequenceHash, CachedSequenceResult>,
    /// L3 cache for partial transformation patterns (slowest but most comprehensive)
    l3_pattern_cache: HashMap<TransformationPattern, PatternCacheEntry>,
    /// Cache configuration parameters
    config: CacheConfig,
    /// Cache statistics
    stats: CacheStats,
}

/// Configuration for the multi-level cache system
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum entries in L1 cache
    pub l1_max_entries: usize,
    /// Maximum entries in L2 cache
    pub l2_max_entries: usize,
    /// Maximum entries in L3 cache
    pub l3_max_entries: usize,
    /// Time-to-live for cache entries (seconds)
    pub ttl_seconds: u64,
    /// Enable intelligent cache warming
    pub enable_cache_warming: bool,
    /// Cache eviction strategy
    pub eviction_strategy: EvictionStrategy,
}

/// Cache eviction strategies
#[derive(Debug, Clone, Copy)]
pub enum EvictionStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-based with priority scoring
    TimeBasedPriority,
    /// Adaptive based on success rate
    AdaptiveSuccessRate,
}

/// Cached result for a complete choice sequence
#[derive(Debug, Clone)]
pub struct CachedSequenceResult {
    /// Whether the sequence produced a valid result
    pub is_valid: bool,
    /// Number of times this sequence was seen
    pub frequency: u32,
    /// Average time to process this sequence
    pub avg_processing_time: std::time::Duration,
    /// Last time this sequence was accessed
    pub last_accessed: std::time::Instant,
    /// Cached transformations that worked on this sequence
    pub successful_transformations: Vec<String>,
}

/// Pattern for matching similar transformation contexts
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TransformationPattern {
    /// Approximate sequence length range
    pub length_range: (usize, usize),
    /// Dominant choice types in the sequence
    pub dominant_types: Vec<ChoiceType>,
    /// Constraint patterns (simplified)
    pub constraint_signature: u64,
}

/// Cache entry for transformation patterns
#[derive(Debug, Clone)]
pub struct PatternCacheEntry {
    /// Transformations that historically work well for this pattern
    pub recommended_transformations: Vec<(String, f64)>, // (name, success_rate)
    /// Transformations to avoid for this pattern
    pub avoid_transformations: Vec<String>,
    /// Number of times this pattern was matched
    pub match_count: u32,
    /// Average improvement ratio for this pattern
    pub avg_improvement_ratio: f64,
}

/// Statistics for cache performance monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub l3_hits: u64,
    pub l3_misses: u64,
    pub evictions: u64,
    pub memory_usage_bytes: u64,
}

/// Frequency data for sequence tracking
#[derive(Debug, Clone)]
pub struct SequenceFrequencyData {
    /// Number of times this sequence was seen
    pub frequency: u32,
    /// First time this sequence was encountered
    pub first_seen: std::time::Instant,
    /// Last time this sequence was accessed
    pub last_accessed: std::time::Instant,
    /// Whether this sequence led to successful shrinking
    pub led_to_success: bool,
    /// Average processing time for this sequence
    pub avg_processing_time: std::time::Duration,
}

/// Simple Bloom filter for fast sequence existence checks
#[derive(Debug)]
pub struct BloomFilter {
    /// Bit array for the filter
    bits: Vec<bool>,
    /// Size of the bit array
    size: usize,
    /// Number of hash functions
    hash_functions: u32,
    /// Number of items inserted
    item_count: u64,
}

impl BloomFilter {
    /// Create a new bloom filter with given size and hash functions
    pub fn new(size: usize, hash_functions: u32) -> Self {
        Self {
            bits: vec![false; size],
            size,
            hash_functions,
            item_count: 0,
        }
    }
    
    /// Add an item to the bloom filter
    pub fn insert(&mut self, item: &ChoiceSequenceHash) {
        for i in 0..self.hash_functions {
            let hash = self.hash_function(item, i);
            let index = (hash % self.size as u64) as usize;
            self.bits[index] = true;
        }
        self.item_count += 1;
    }
    
    /// Check if an item might be in the set (may have false positives)
    pub fn contains(&self, item: &ChoiceSequenceHash) -> bool {
        for i in 0..self.hash_functions {
            let hash = self.hash_function(item, i);
            let index = (hash % self.size as u64) as usize;
            if !self.bits[index] {
                return false; // Definitely not in set
            }
        }
        true // Probably in set
    }
    
    /// Simple hash function for bloom filter
    fn hash_function(&self, item: &ChoiceSequenceHash, seed: u32) -> u64 {
        // Simple hash combining sequence hash with seed
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        item.sequence_hash.hash(&mut hasher);
        seed.hash(&mut hasher);
        item.length.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Get estimated false positive rate
    pub fn false_positive_rate(&self) -> f64 {
        if self.item_count == 0 {
            return 0.0;
        }
        let k = self.hash_functions as f64;
        let m = self.size as f64;
        let n = self.item_count as f64;
        
        // Standard bloom filter false positive formula: (1 - e^(-kn/m))^k
        (1.0 - (-k * n / m).exp()).powf(k)
    }
}

/// A shrinking transformation that can be applied to a choice sequence
#[derive(Debug)]
pub struct ShrinkingTransformation {
    /// Human-readable description of this transformation
    pub description: String,
    
    /// The transformation function that modifies a choice sequence
    pub transform: fn(&[ChoiceNode]) -> Vec<ChoiceNode>,
}

impl Clone for ShrinkingTransformation {
    fn clone(&self) -> Self {
        Self {
            description: self.description.clone(),
            transform: self.transform,
        }
    }
}

/// Core shrinker that applies transformations to minimize test cases
#[derive(Debug)]
pub struct ChoiceShrinker {
    /// Original failing test result
    pub original_result: ConjectureResult,
    
    /// Best (smallest) result found so far
    pub best_result: ConjectureResult,
    
    /// Number of shrinking attempts made
    pub attempts: u32,
    
    /// Maximum number of shrinking attempts allowed
    pub max_attempts: u32,
    
    /// Available shrinking transformations
    pub transformations: Vec<ShrinkingTransformation>,
    
    /// DataTree for context-aware shrinking (optional)
    pub datatree: Option<DataTree>,
    
    /// Legacy cache for backward compatibility
    pub transformation_cache: HashMap<(String, ChoiceSequenceHash), ShrinkingCacheEntry>,
    
    /// Advanced multi-level cache system
    pub multi_cache: MultiLevelCache,
    
    /// Set of previously seen choice sequences for deduplication
    pub seen_sequences: HashSet<ChoiceSequenceHash>,
    
    /// Advanced deduplication with frequency tracking
    pub sequence_frequency: HashMap<ChoiceSequenceHash, SequenceFrequencyData>,
    
    /// Performance metrics
    pub metrics: ShrinkingMetrics,
    
    /// Transformation success rates for smart ordering
    pub transformation_stats: HashMap<String, (u32, u32)>, // (successes, attempts)
    
    /// Bloom filter for fast sequence existence checks
    pub sequence_bloom_filter: Option<BloomFilter>,
}

impl ChoiceShrinker {
    /// Create a new shrinker for the given failing test result
    pub fn new(original_result: ConjectureResult) -> Self {
        let mut seen_sequences = HashSet::new();
        // Add the original sequence to seen sequences
        seen_sequences.insert(ChoiceSequenceHash::from_choices(&original_result.nodes));
        
        Self {
            best_result: original_result.clone(),
            original_result,
            attempts: 0,
            max_attempts: 10000, // Match Python's default
            transformations: Self::default_transformations(),
            datatree: None,
            transformation_cache: HashMap::new(),
            multi_cache: MultiLevelCache::new(),
            seen_sequences,
            sequence_frequency: HashMap::new(),
            metrics: ShrinkingMetrics::default(),
            transformation_stats: HashMap::new(),
            sequence_bloom_filter: Some(BloomFilter::new(10000, 3)),
        }
    }
    
    /// Create a new shrinker with DataTree context
    pub fn with_datatree(original_result: ConjectureResult, datatree: DataTree) -> Self {
        let mut shrinker = Self::new(original_result);
        shrinker.datatree = Some(datatree);
        shrinker.transformations.insert(0, ShrinkingTransformation {
            description: "datatree_guided_shrinking".to_string(),
            transform: datatree_guided_shrinking,
        });
        shrinker
    }
    
    /// Check if a choice sequence has already been seen (deduplication)
    fn is_duplicate_sequence(&self, choices: &[ChoiceNode]) -> bool {
        let hash = ChoiceSequenceHash::from_choices(choices);
        self.seen_sequences.contains(&hash)
    }
    
    /// Add a choice sequence to the seen set
    fn mark_sequence_seen(&mut self, choices: &[ChoiceNode]) {
        let hash = ChoiceSequenceHash::from_choices(choices);
        self.seen_sequences.insert(hash);
    }
    
    /// Check cache for a transformation result
    fn check_cache(&mut self, transformation_name: &str, choices: &[ChoiceNode]) -> Option<bool> {
        let input_hash = ChoiceSequenceHash::from_choices(choices);
        let cache_key = (transformation_name.to_string(), input_hash);
        
        if let Some(entry) = self.transformation_cache.get_mut(&cache_key) {
            entry.hit_count += 1;
            self.metrics.cache_hits += 1;
            println!("SHRINKING CACHE: Hit for {} (count: {})", transformation_name, entry.hit_count);
            Some(entry.successful)
        } else {
            None
        }
    }
    
    /// Cache a transformation result
    fn cache_result(&mut self, transformation_name: &str, input_choices: &[ChoiceNode], 
                    output_choices: Option<&[ChoiceNode]>, successful: bool) {
        let input_hash = ChoiceSequenceHash::from_choices(input_choices);
        let result_hash = output_choices.map(ChoiceSequenceHash::from_choices);
        let cache_key = (transformation_name.to_string(), input_hash);
        
        let entry = ShrinkingCacheEntry {
            successful,
            result_hash,
            timestamp: std::time::Instant::now(),
            hit_count: 0,
            avg_execution_time: 0,
            priority_score: 1.0,
        };
        
        self.transformation_cache.insert(cache_key, entry);
        
        // Clean old cache entries if cache gets too large
        if self.transformation_cache.len() > 10000 {
            self.clean_cache();
        }
    }
    
    /// Clean old cache entries to prevent memory bloat
    fn clean_cache(&mut self) {
        let now = std::time::Instant::now();
        let cutoff = std::time::Duration::from_secs(300); // 5 minutes
        
        self.transformation_cache.retain(|_, entry| {
            now.duration_since(entry.timestamp) < cutoff
        });
        
        println!("SHRINKING CACHE: Cleaned cache, {} entries remaining", 
                 self.transformation_cache.len());
    }
    
    /// Update transformation statistics for smart ordering
    fn update_transformation_stats(&mut self, transformation_name: &str, successful: bool) {
        let stats = self.transformation_stats.entry(transformation_name.to_string())
            .or_insert((0, 0));
        
        stats.1 += 1; // increment attempts
        if successful {
            stats.0 += 1; // increment successes
        }
    }
    
    /// Get transformation success rate
    fn get_transformation_success_rate(&self, transformation_name: &str) -> f64 {
        if let Some((successes, attempts)) = self.transformation_stats.get(transformation_name) {
            if *attempts > 0 {
                *successes as f64 / *attempts as f64
            } else {
                0.0
            }
        } else {
            0.5 // Default rate for new transformations
        }
    }
    
    /// Sort transformations by success rate for smart ordering
    fn sort_transformations_by_success_rate(&mut self) {
        // Pre-calculate success rates to avoid borrowing issues
        let mut transformation_rates: Vec<(usize, f64)> = self.transformations
            .iter()
            .enumerate()
            .map(|(i, transform)| (i, self.get_transformation_success_rate(&transform.description)))
            .collect();
        
        // Sort by success rate (descending)
        transformation_rates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Reorder transformations based on sorted indices
        let mut sorted_transformations = Vec::new();
        for (original_index, _) in transformation_rates {
            sorted_transformations.push(self.transformations[original_index].clone());
        }
        self.transformations = sorted_transformations;
    }
    
    /// Check if we've reached a minimal form and should terminate early
    fn is_minimal_form(&self) -> bool {
        // Early termination conditions:
        // 1. Only one choice remaining
        if self.best_result.nodes.len() <= 1 {
            println!("SHRINKING TERMINATION: Minimal form reached - single choice");
            return true;
        }
        
        // 2. All integer choices are at their shrink targets
        let all_at_targets = self.best_result.nodes.iter().all(|choice| {
            match (&choice.value, &choice.constraints) {
                (ChoiceValue::Integer(value), Constraints::Integer(constraints)) => {
                    let target = constraints.shrink_towards.unwrap_or(0);
                    *value == target
                },
                (ChoiceValue::Boolean(false), _) => true, // false is minimal for booleans
                (ChoiceValue::Float(value), _) => *value == 0.0, // 0.0 is minimal for floats
                (ChoiceValue::String(s), _) => s.is_empty(), // empty is minimal for strings
                (ChoiceValue::Bytes(b), _) => b.is_empty(), // empty is minimal for bytes
                _ => false,
            }
        });
        
        if all_at_targets {
            println!("SHRINKING TERMINATION: All choices at minimal values");
            return true;
        }
        
        // 3. No progress in the last several attempts
        let recent_window = 50;
        if self.attempts > recent_window && 
           self.metrics.successful_transformations == 0 {
            println!("SHRINKING TERMINATION: No progress in recent attempts");
            return true;
        }
        
        false
    }
    
    /// Get the default set of shrinking transformations
    fn default_transformations() -> Vec<ShrinkingTransformation> {
        vec![
            // Priority 1: Constraint-aware shrinking (should happen first to respect shrink_towards)
            ShrinkingTransformation {
                description: "constraint_repair_shrinking".to_string(),
                transform: constraint_repair_shrinking,
            },
            
            // Priority 2: Structure shrinking (most impactful)
            ShrinkingTransformation {
                description: "delete_leading_nodes".to_string(),
                transform: delete_leading_nodes,
            },
            ShrinkingTransformation {
                description: "delete_trailing_nodes".to_string(),
                transform: delete_trailing_nodes,
            },
            ShrinkingTransformation {
                description: "delete_redundant_nodes".to_string(),
                transform: delete_redundant_nodes,
            },
            ShrinkingTransformation {
                description: "zero_out_nodes".to_string(),
                transform: zero_out_nodes,
            },
            ShrinkingTransformation {
                description: "remove_leading_zeros".to_string(),
                transform: remove_leading_zeros,
            },
            
            // Priority 3: Value minimization
            ShrinkingTransformation {
                description: "minimize_integer_values_conservative".to_string(),
                transform: minimize_integer_values_conservative,
            },
            ShrinkingTransformation {
                description: "minimize_integer_values_aggressive".to_string(),
                transform: minimize_integer_values_aggressive,
            },
            ShrinkingTransformation {
                description: "minimize_to_false".to_string(),
                transform: minimize_booleans_to_false,
            },
            
            // Priority 3: Advanced passes
            ShrinkingTransformation {
                description: "reorder_nodes".to_string(),
                transform: reorder_nodes,
            },
            ShrinkingTransformation {
                description: "split_integer_ranges".to_string(),
                transform: split_integer_ranges,
            },
            ShrinkingTransformation {
                description: "minimize_floating_point".to_string(),
                transform: minimize_floating_point,
            },
            
            // Priority 4: Sophisticated advanced passes
            ShrinkingTransformation {
                description: "minimize_duplicated_nodes".to_string(),
                transform: minimize_duplicated_nodes,
            },
            ShrinkingTransformation {
                description: "minimize_choice_order".to_string(),
                transform: minimize_choice_order,
            },
            ShrinkingTransformation {
                description: "adaptive_delete_nodes".to_string(),
                transform: adaptive_delete_nodes,
            },
            ShrinkingTransformation {
                description: "span_aware_shrinking".to_string(),
                transform: span_aware_shrinking,
            },
        ]
    }
    
    /// Apply all available transformations to find minimal example
    pub fn shrink<F>(&mut self, test_function: F) -> ConjectureResult 
    where
        F: Fn(&ConjectureResult) -> bool,
    {
        println!("SHRINKING DEBUG: Starting shrink with {} nodes", self.original_result.nodes.len());
        
        // Performance optimization: Use early exit conditions and smarter ordering
        let mut improved = true;
        let mut pass_count = 0;
        const MAX_PASSES: u32 = 10; // Limit the number of full passes
        
        while improved && self.attempts < self.max_attempts && pass_count < MAX_PASSES {
            improved = false;
            pass_count += 1;
            
            println!("SHRINKING DEBUG: Starting pass {} (attempts: {})", pass_count, self.attempts);
            
            // Clone transformations to avoid borrow conflicts
            let transformations = self.transformations.clone();
            
            // Performance optimization: Try high-impact transformations first
            for transformation in &transformations {
                if self.attempts >= self.max_attempts {
                    break;
                }
                
                // Early exit if we've made good progress
                let original_nodes = self.original_result.nodes.len();
                let current_nodes = self.best_result.nodes.len();
                if current_nodes <= original_nodes / 4 {
                    println!("SHRINKING DEBUG: Early exit - reduced to 25% of original size");
                    break;
                }
                
                if self.apply_transformation(transformation, &test_function) {
                    improved = true;
                    println!("SHRINKING DEBUG: {} succeeded, new best has {} nodes", 
                             transformation.description, self.best_result.nodes.len());
                    
                    // Performance optimization: If we made significant progress, try this transformation again
                    if current_nodes - self.best_result.nodes.len() > 5 {
                        println!("SHRINKING DEBUG: Significant progress, retrying transformation");
                        // Try the same transformation a few more times
                        for _ in 0..3 {
                            if self.attempts >= self.max_attempts {
                                break;
                            }
                            if !self.apply_transformation(transformation, &test_function) {
                                break; // No more progress with this transformation
                            }
                        }
                    }
                }
            }
        }
        
        println!("SHRINKING DEBUG: Shrinking complete after {} attempts in {} passes", 
                 self.attempts, pass_count);
        println!("SHRINKING DEBUG: Original: {} nodes, Final: {} nodes", 
                 self.original_result.nodes.len(), self.best_result.nodes.len());
        
        self.best_result.clone()
    }
    
    /// Apply a specific transformation and test if it improves the result
    fn apply_transformation<F>(&mut self, transformation: &ShrinkingTransformation, test_function: F) -> bool 
    where
        F: Fn(&ConjectureResult) -> bool,
    {
        self.attempts += 1;
        
        // Performance optimization: Early exit for expensive transformations
        if self.best_result.nodes.len() > 100 && 
           (transformation.description.contains("reorder") || transformation.description.contains("split")) {
            println!("SHRINKING DEBUG: Skipping expensive transformation {} for large input", 
                     transformation.description);
            return false;
        }
        
        println!("SHRINKING DEBUG: Applying transformation: {}", transformation.description);
        
        // Apply transformation to current best nodes
        let transformed_nodes = (transformation.transform)(&self.best_result.nodes);
        
        // Performance optimization: Quick equality check before detailed comparison
        if transformed_nodes.len() == self.best_result.nodes.len() {
            let mut changed = false;
            for (orig, trans) in self.best_result.nodes.iter().zip(transformed_nodes.iter()) {
                if orig.value != trans.value {
                    changed = true;
                    break;
                }
            }
            if !changed {
                println!("SHRINKING DEBUG: No changes detected, skipping");
                return false;
            }
        }
        
        println!("SHRINKING DEBUG: Original nodes: {}", self.best_result.nodes.len());
        println!("SHRINKING DEBUG: Transformed nodes: {}", transformed_nodes.len());
        
        // Performance optimization: Only print detailed debug for small inputs
        if self.best_result.nodes.len() <= 10 {
            for (i, (orig, trans)) in self.best_result.nodes.iter().zip(transformed_nodes.iter()).enumerate() {
                println!("SHRINKING DEBUG: Choice {}: {:?} -> {:?}", i, orig.value, trans.value);
            }
        }
        
        // Performance optimization: Pre-check if this could be better before expensive test
        // For choice-aware shrinking, only skip if we have more nodes (structural regression)
        if transformed_nodes.len() > self.best_result.nodes.len() {
            println!("SHRINKING DEBUG: More nodes than current best, skipping test");
            return false;
        }
        
        // Create new result with transformed nodes
        let test_result = ConjectureResult {
            status: Status::Valid,
            nodes: transformed_nodes.clone(),
            length: transformed_nodes.len(), // Use node count as the primary "length" metric for choice shrinking
            events: HashMap::new(),
            buffer: Vec::new(),
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
        
        println!("SHRINKING DEBUG: Testing if transformation still fails");
        
        // Test if the transformation still produces a failing test
        if test_function(&test_result) {
            println!("SHRINKING DEBUG: Transformation still fails - checking if better");
            // This is a valid shrinking - smaller and still fails
            if self.is_better(&test_result, &self.best_result) {
                println!("SHRINKING DEBUG: Found better result, updating best");
                self.best_result = test_result;
                return true;
            } else {
                println!("SHRINKING DEBUG: Not better than current best");
            }
        } else {
            println!("SHRINKING DEBUG: Transformation passes test - not a valid shrinking");
        }
        
        false
    }
    
    /// Check if one result is better (smaller) than another
    fn is_better(&self, candidate: &ConjectureResult, current: &ConjectureResult) -> bool {
        println!("SHRINKING DEBUG: Comparing candidates:");
        println!("SHRINKING DEBUG: Candidate: {} nodes", candidate.nodes.len());
        println!("SHRINKING DEBUG: Current: {} nodes", current.nodes.len());
        
        // Primarily prefer fewer nodes
        if candidate.nodes.len() != current.nodes.len() {
            let result = candidate.nodes.len() < current.nodes.len();
            println!("SHRINKING DEBUG: Different choice counts -> {}", result);
            return result;
        }
        
        // Secondary: prefer lexicographically smaller choice values
        // This handles value minimization within same structure
        for (cand_choice, curr_choice) in candidate.nodes.iter().zip(current.nodes.iter()) {
            let comparison = self.compare_choice_values(&cand_choice.value, &curr_choice.value);
            if comparison != std::cmp::Ordering::Equal {
                let result = comparison == std::cmp::Ordering::Less;
                println!("SHRINKING DEBUG: Value comparison: {:?} vs {:?} -> {}", 
                         cand_choice.value, curr_choice.value, result);
                return result;
            }
        }
        
        println!("SHRINKING DEBUG: No difference found -> false");
        false
    }
    
    /// Compare two choice values for shrinking purposes 
    fn compare_choice_values(&self, a: &ChoiceValue, b: &ChoiceValue) -> std::cmp::Ordering {
        match (a, b) {
            (ChoiceValue::Integer(a), ChoiceValue::Integer(b)) => {
                // Try to find the constraints to determine the shrink target
                // Default to 0 if no constraint found
                let mut shrink_target = 0i128;
                
                // Look through current best result to find matching constraint
                if let Some(best_choice) = self.best_result.nodes.first() {
                    if let (ChoiceValue::Integer(_), Constraints::Integer(constraints)) = (&best_choice.value, &best_choice.constraints) {
                        shrink_target = constraints.shrink_towards.unwrap_or(0);
                    }
                }
                
                // Prefer values closer to the shrink target
                let dist_a = (*a - shrink_target).abs();
                let dist_b = (*b - shrink_target).abs();
                
                if dist_a != dist_b {
                    dist_a.cmp(&dist_b)
                } else {
                    // If distances are equal, prefer the one closer to the default direction (usually 0)
                    a.abs().cmp(&b.abs())
                }
            },
            (ChoiceValue::Boolean(a), ChoiceValue::Boolean(b)) => {
                // false < true for shrinking purposes
                a.cmp(b)
            },
            (ChoiceValue::Float(a), ChoiceValue::Float(b)) => {
                // Prefer smaller absolute values, handle NaN specially
                if a.is_nan() && b.is_nan() {
                    std::cmp::Ordering::Equal
                } else if a.is_nan() {
                    std::cmp::Ordering::Greater // NaN is "larger" for shrinking
                } else if b.is_nan() {
                    std::cmp::Ordering::Less
                } else {
                    let abs_a = a.abs();
                    let abs_b = b.abs();
                    abs_a.partial_cmp(&abs_b).unwrap_or(std::cmp::Ordering::Equal)
                }
            },
            (ChoiceValue::String(a), ChoiceValue::String(b)) => {
                // Prefer shorter strings, then lexicographic order
                let len_cmp = a.len().cmp(&b.len());
                if len_cmp != std::cmp::Ordering::Equal {
                    len_cmp
                } else {
                    a.cmp(b)
                }
            },
            (ChoiceValue::Bytes(a), ChoiceValue::Bytes(b)) => {
                // Prefer shorter byte arrays, then lexicographic order
                let len_cmp = a.len().cmp(&b.len());
                if len_cmp != std::cmp::Ordering::Equal {
                    len_cmp
                } else {
                    a.cmp(b)
                }
            },
            _ => std::cmp::Ordering::Equal, // Different types, consider equal
        }
    }
    
    /// Calculate total length for a choice sequence
    fn calculate_length(&self, nodes: &[ChoiceNode]) -> usize {
        nodes.iter().map(|choice| match choice.choice_type {
            ChoiceType::Integer => 2,
            ChoiceType::Boolean => 1,
            ChoiceType::Float => 8,
            ChoiceType::String => {
                if let ChoiceValue::String(s) = &choice.value {
                    s.len()
                } else {
                    0
                }
            },
            ChoiceType::Bytes => {
                if let ChoiceValue::Bytes(b) = &choice.value {
                    b.len()
                } else {
                    0
                }
            },
        }).sum()
    }
}

/// Conservative integer minimization - makes small steps towards shrink target
fn minimize_integer_values_conservative(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    nodes.iter().map(|choice| {
        // Only modify non-forced nodes
        if choice.was_forced {
            return choice.clone();
        }
        
        if let (ChoiceType::Integer, ChoiceValue::Integer(value), Constraints::Integer(constraints)) = 
            (&choice.choice_type, &choice.value, &choice.constraints) {
            
            let min_val = constraints.min_value.unwrap_or(i128::MIN);
            let max_val = constraints.max_value.unwrap_or(i128::MAX);
            let shrink_target = constraints.shrink_towards.unwrap_or(0).max(min_val).min(max_val);
            
            // If already at target, no change needed
            if *value == shrink_target {
                return choice.clone();
            }
            
            // Make single-step shrinking towards the target
            let new_value = if *value > shrink_target {
                (*value - 1).max(shrink_target)
            } else {
                (*value + 1).min(shrink_target)
            };
            
            // Check if the new value is within bounds and closer to target
            if new_value >= min_val && new_value <= max_val && new_value != *value {
                let mut new_choice = choice.clone();
                new_choice.value = ChoiceValue::Integer(new_value);
                return new_choice;
            }
        }
        
        choice.clone()
    }).collect()
}

/// Aggressive integer minimization - makes larger jumps towards shrink target
fn minimize_integer_values_aggressive(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    nodes.iter().map(|choice| {
        // Only modify non-forced nodes
        if choice.was_forced {
            return choice.clone();
        }
        
        if let (ChoiceType::Integer, ChoiceValue::Integer(value), Constraints::Integer(constraints)) = 
            (&choice.choice_type, &choice.value, &choice.constraints) {
            
            let min_val = constraints.min_value.unwrap_or(i128::MIN);
            let max_val = constraints.max_value.unwrap_or(i128::MAX);
            let shrink_target = constraints.shrink_towards.unwrap_or(0).max(min_val).min(max_val);
            
            // If already at target, no change needed
            if *value == shrink_target {
                return choice.clone();
            }
            
            // Calculate the distance to the target and make larger jumps
            let distance = (*value - shrink_target).abs();
            let reduction = distance / 2; // Try to cut distance in half
            
            let new_value = if *value > shrink_target {
                (*value - reduction).max(shrink_target)
            } else {
                (*value + reduction).min(shrink_target)
            };
            
            // Check if the new value is within bounds and closer to target
            if new_value >= min_val && new_value <= max_val && new_value != *value {
                let mut new_choice = choice.clone();
                new_choice.value = ChoiceValue::Integer(new_value);
                return new_choice;
            }
        }
        
        choice.clone()
    }).collect()
}

/// Minimize boolean values to false where possible
fn minimize_booleans_to_false(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    nodes.iter().map(|choice| {
        // Only modify non-forced boolean nodes that are true
        if let (ChoiceType::Boolean, ChoiceValue::Boolean(true)) = (&choice.choice_type, &choice.value) {
            if !choice.was_forced {
                let mut new_choice = choice.clone();
                new_choice.value = ChoiceValue::Boolean(false);
                return new_choice;
            }
        }
        choice.clone()
    }).collect()
}

/// Delete leading nodes to remove unnecessary initial elements
fn delete_leading_nodes(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    if nodes.is_empty() {
        return nodes.to_vec();
    }
    
    // Remove the first choice - this is especially effective for lists with leading zeros
    nodes[1..].to_vec()
}

/// Delete trailing nodes to reduce sequence length
fn delete_trailing_nodes(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    if nodes.is_empty() {
        return nodes.to_vec();
    }
    
    // Remove the last choice
    nodes[..nodes.len() - 1].to_vec()
}

/// Delete redundant nodes that don't affect test behavior
fn delete_redundant_nodes(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    if nodes.len() <= 1 {
        return nodes.to_vec();
    }
    
    // Try removing each choice individually to find redundant ones
    // For now, try removing the choice with the highest absolute value
    let mut max_abs_value = 0i128;
    let mut max_index = 0;
    
    for (i, choice) in nodes.iter().enumerate() {
        let abs_value = match &choice.value {
            ChoiceValue::Integer(val) => val.abs(),
            ChoiceValue::Boolean(true) => 1,
            ChoiceValue::Boolean(false) => 0,
            ChoiceValue::Float(val) => val.abs() as i128,
            _ => 0,
        };
        
        if abs_value > max_abs_value && !choice.was_forced {
            max_abs_value = abs_value;
            max_index = i;
        }
    }
    
    // Remove the choice with maximum absolute value (if non-zero)
    if max_abs_value > 0 {
        let mut result = nodes.to_vec();
        result.remove(max_index);
        result
    } else {
        nodes.to_vec()
    }
}

/// Remove leading nodes that are zero, which often represent unnecessary list elements
fn remove_leading_zeros(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    if nodes.is_empty() {
        return nodes.to_vec();
    }
    
    // Find the first non-zero choice
    for (i, choice) in nodes.iter().enumerate() {
        let is_zero = match &choice.value {
            ChoiceValue::Integer(0) => true,
            ChoiceValue::Boolean(false) => true,
            _ => false,
        };
        
        // If we found a non-zero choice and it's not the first one, remove leading zeros
        if !is_zero && i > 0 {
            println!("SHRINKING DEBUG: Removing {} leading zero nodes", i);
            return nodes[i..].to_vec();
        }
    }
    
    // If all nodes are zero or we only have one choice, keep as is
    nodes.to_vec()
}

/// Zero out nodes by setting them to their shrink targets
fn zero_out_nodes(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    nodes.iter().map(|choice| {
        if choice.was_forced {
            return choice.clone();
        }
        
        match (&choice.choice_type, &choice.value, &choice.constraints) {
            (ChoiceType::Integer, ChoiceValue::Integer(_), Constraints::Integer(constraints)) => {
                let min_val = constraints.min_value.unwrap_or(i128::MIN);
                let max_val = constraints.max_value.unwrap_or(i128::MAX);
                let target = constraints.shrink_towards.unwrap_or(0).max(min_val).min(max_val);
                
                let mut new_choice = choice.clone();
                new_choice.value = ChoiceValue::Integer(target);
                new_choice
            },
            (ChoiceType::Boolean, ChoiceValue::Boolean(_), _) => {
                let mut new_choice = choice.clone();
                new_choice.value = ChoiceValue::Boolean(false);
                new_choice
            },
            (ChoiceType::Float, ChoiceValue::Float(_), _) => {
                let mut new_choice = choice.clone();
                new_choice.value = ChoiceValue::Float(0.0);
                new_choice
            },
            _ => choice.clone(),
        }
    }).collect()
}

/// Reorder nodes to optimize shrinking
fn reorder_nodes(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    let mut result = nodes.to_vec();
    
    // Sort nodes by their "complexity" - simpler nodes first
    result.sort_by_key(|choice| {
        match &choice.value {
            ChoiceValue::Boolean(_) => 0,
            ChoiceValue::Integer(val) => val.abs() as u64,
            ChoiceValue::Float(val) => val.abs() as u64,
            ChoiceValue::String(s) => s.len() as u64,
            ChoiceValue::Bytes(b) => b.len() as u64,
        }
    });
    
    result
}

/// Split integer ranges for more granular shrinking
fn split_integer_ranges(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    nodes.iter().map(|choice| {
        if choice.was_forced {
            return choice.clone();
        }
        
        if let (ChoiceType::Integer, ChoiceValue::Integer(value), Constraints::Integer(constraints)) = 
            (&choice.choice_type, &choice.value, &choice.constraints) {
            
            let min_val = constraints.min_value.unwrap_or(i128::MIN);
            let max_val = constraints.max_value.unwrap_or(i128::MAX);
            let shrink_target = constraints.shrink_towards.unwrap_or(0).max(min_val).min(max_val);
            
            // If value is far from target, try moving to midpoint
            let distance = (*value - shrink_target).abs();
            if distance > 2 {
                let midpoint = shrink_target + ((*value - shrink_target) / 2);
                
                if midpoint >= min_val && midpoint <= max_val && midpoint != *value {
                    let mut new_choice = choice.clone();
                    new_choice.value = ChoiceValue::Integer(midpoint);
                    return new_choice;
                }
            }
        }
        
        choice.clone()
    }).collect()
}

/// Minimize floating point values
fn minimize_floating_point(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    nodes.iter().map(|choice| {
        if choice.was_forced {
            return choice.clone();
        }
        
        if let (ChoiceType::Float, ChoiceValue::Float(value)) = (&choice.choice_type, &choice.value) {
            // Try to minimize towards 0.0
            if value.is_finite() && *value != 0.0 {
                let abs_val = value.abs();
                
                // Try smaller magnitudes
                let candidates = [
                    0.0,
                    abs_val / 2.0,
                    abs_val / 10.0,
                    1.0,
                    -1.0,
                ];
                
                for &candidate in &candidates {
                    if candidate.abs() < abs_val {
                        let mut new_choice = choice.clone();
                        new_choice.value = ChoiceValue::Float(candidate);
                        return new_choice;
                    }
                }
            }
        }
        
        choice.clone()
    }).collect()
}

/// DataTree-guided shrinking using tree context to find better shrinking candidates
fn datatree_guided_shrinking(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    // For now, implement a simple heuristic that prioritizes nodes that are more likely to matter
    // In a full implementation, this would use the DataTree to understand which paths lead to failures
    
    if nodes.is_empty() {
        return nodes.to_vec();
    }
    
    // Strategy 1: Try reducing the most "complex" choice first
    let mut max_complexity = 0.0f64;
    let mut max_index = 0;
    
    for (i, choice) in nodes.iter().enumerate() {
        let complexity = match &choice.value {
            ChoiceValue::Integer(val) => val.abs() as f64,
            ChoiceValue::Boolean(true) => 1.0,
            ChoiceValue::Boolean(false) => 0.0,
            ChoiceValue::Float(val) => val.abs(),
            ChoiceValue::String(s) => s.len() as f64 * 2.0,
            ChoiceValue::Bytes(b) => b.len() as f64 * 1.5,
        };
        
        if complexity > max_complexity && !choice.was_forced {
            max_complexity = complexity;
            max_index = i;
        }
    }
    
    // Apply the most aggressive shrinking to the most complex choice
    if max_complexity > 0.0 {
        let mut result = nodes.to_vec();
        let choice = &result[max_index];
        
        match (&choice.choice_type, &choice.value, &choice.constraints) {
            (ChoiceType::Integer, ChoiceValue::Integer(value), Constraints::Integer(constraints)) => {
                let min_val = constraints.min_value.unwrap_or(i128::MIN);
                let max_val = constraints.max_value.unwrap_or(i128::MAX);
                let target = constraints.shrink_towards.unwrap_or(0).max(min_val).min(max_val);
                
                // Make aggressive jump towards target
                let distance = (*value - target).abs();
                let jump = distance / 4; // Jump 1/4 of the way to target
                
                let new_value = if *value > target {
                    (*value - jump).max(target)
                } else {
                    (*value + jump).min(target)
                };
                
                if new_value >= min_val && new_value <= max_val && new_value != *value {
                    result[max_index].value = ChoiceValue::Integer(new_value);
                }
            },
            (ChoiceType::Boolean, ChoiceValue::Boolean(true), _) => {
                result[max_index].value = ChoiceValue::Boolean(false);
            },
            (ChoiceType::Float, ChoiceValue::Float(value), _) => {
                if value.is_finite() && *value != 0.0 {
                    result[max_index].value = ChoiceValue::Float(value / 4.0);
                }
            },
            (ChoiceType::String, ChoiceValue::String(s), _) => {
                if !s.is_empty() {
                    let new_len = s.len() / 2;
                    if new_len > 0 {
                        result[max_index].value = ChoiceValue::String(s.chars().take(new_len).collect());
                    } else {
                        result[max_index].value = ChoiceValue::String(String::new());
                    }
                }
            },
            (ChoiceType::Bytes, ChoiceValue::Bytes(b), _) => {
                if !b.is_empty() {
                    let new_len = b.len() / 2;
                    result[max_index].value = ChoiceValue::Bytes(b[..new_len].to_vec());
                }
            },
            _ => {},
        }
        
        result
    } else {
        nodes.to_vec()
    }
}

/// Advanced Shrinking Functions - Implementation of sophisticated transformation passes

/// Detect and remove redundant choice sequences that appear multiple times
fn minimize_duplicated_nodes(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    if nodes.len() < 2 {
        return nodes.to_vec();
    }

    let mut result = nodes.to_vec();
    let mut _seen_patterns: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    
    // Look for patterns of 2-5 consecutive nodes that repeat
    for pattern_length in 2..=std::cmp::min(5, nodes.len() / 2) {
        for start in 0..=(nodes.len() - pattern_length * 2) {
            let pattern = &nodes[start..start + pattern_length];
            
            // Check if this pattern repeats immediately after
            let next_start = start + pattern_length;
            if next_start + pattern_length <= nodes.len() {
                let next_pattern = &nodes[next_start..next_start + pattern_length];
                
                // Compare patterns (simplified equality check)
                let patterns_match = pattern.iter().zip(next_pattern.iter()).all(|(a, b)| {
                    a.choice_type == b.choice_type && a.value == b.value && !a.was_forced && !b.was_forced
                });
                
                if patterns_match {
                    // Remove the duplicate pattern
                    result.drain(next_start..next_start + pattern_length);
                    println!("SHRINKING DEBUG: Removed duplicated pattern of length {} at position {}", pattern_length, next_start);
                    return result; // Only remove one pattern at a time for safety
                }
            }
        }
    }
    
    result
}

/// Reorder nodes to create better shrinking opportunities
fn minimize_choice_order(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    if nodes.len() < 2 {
        return nodes.to_vec();
    }

    let mut result = nodes.to_vec();
    
    // Strategy: Move "simpler" nodes earlier to create better shrinking opportunities
    // This implements a stable sort that preserves order when complexity is equal
    result.sort_by(|a, b| {
        // Don't reorder forced nodes
        if a.was_forced || b.was_forced {
            return std::cmp::Ordering::Equal;
        }
        
        let complexity_a = choice_complexity(a);
        let complexity_b = choice_complexity(b);
        
        complexity_a.partial_cmp(&complexity_b).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    // Only return the reordered result if it's actually different
    if result != nodes {
        println!("SHRINKING DEBUG: Reordered nodes for better shrinking opportunities");
        result
    } else {
        nodes.to_vec()
    }
}

/// Helper function to calculate choice complexity for ordering
fn choice_complexity(choice: &ChoiceNode) -> f64 {
    match &choice.value {
        ChoiceValue::Boolean(false) => 0.0,
        ChoiceValue::Boolean(true) => 1.0,
        ChoiceValue::Integer(val) => val.abs() as f64,
        ChoiceValue::Float(val) => val.abs(),
        ChoiceValue::String(s) => s.len() as f64 * 2.0,
        ChoiceValue::Bytes(b) => b.len() as f64 * 1.5,
    }
}

/// Intelligent choice deletion based on test behavior and choice impact
fn adaptive_delete_nodes(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    if nodes.is_empty() {
        return nodes.to_vec();
    }

    // Strategy 1: Try deleting the choice with highest complexity that isn't forced
    let mut candidates: Vec<(usize, f64)> = nodes.iter().enumerate()
        .filter(|(_, choice)| !choice.was_forced)
        .map(|(i, choice)| (i, choice_complexity(choice)))
        .collect();
    
    // Sort by complexity descending
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    if let Some((index_to_remove, complexity)) = candidates.first() {
        if *complexity > 0.0 {
            let mut result = nodes.to_vec();
            result.remove(*index_to_remove);
            println!("SHRINKING DEBUG: Adaptively deleted choice at index {} with complexity {}", index_to_remove, complexity);
            return result;
        }
    }
    
    // Strategy 2: If no high-complexity nodes, try deleting trailing nodes
    if nodes.len() > 1 {
        let mut result = nodes.to_vec();
        result.pop();
        println!("SHRINKING DEBUG: Adaptively deleted trailing choice");
        return result;
    }
    
    nodes.to_vec()
}

/// Use span structure for contextual shrinking decisions
fn span_aware_shrinking(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    // For now, implement a sophisticated heuristic that would use span information
    // In a full implementation, this would analyze the Example spans in ConjectureResult
    
    if nodes.is_empty() {
        return nodes.to_vec();
    }

    // Strategy: Focus on shrinking nodes that are likely to be in "leaf" spans
    // (nodes that don't affect other nodes) 
    
    // Identify potential leaf nodes by looking for nodes at the end of sequences
    // that don't seem to influence subsequent nodes
    let mut result = nodes.to_vec();
    
    // Look for opportunities to simplify "container" nodes (strings, bytes)
    // that might represent test data structure rather than test logic
    for i in 0..result.len() {
        if result[i].was_forced {
            continue;
        }
        
        let should_shrink = match &result[i].value {
            ChoiceValue::String(s) => s.len() > 1,
            ChoiceValue::Bytes(b) => b.len() > 1,
            ChoiceValue::Integer(val) => val.abs() > 10,
            _ => false,
        };
        
        if should_shrink {
            match &result[i].value {
                ChoiceValue::String(s) => {
                    let original_len = s.len();
                    let new_len = std::cmp::max(1, s.len() / 2);
                    result[i].value = ChoiceValue::String(s.chars().take(new_len).collect());
                    println!("SHRINKING DEBUG: Span-aware string shrinking from {} to {} chars", original_len, new_len);
                    return result;
                },
                ChoiceValue::Bytes(b) => {
                    let original_len = b.len();
                    let new_len = std::cmp::max(1, b.len() / 2);
                    result[i].value = ChoiceValue::Bytes(b[..new_len].to_vec());
                    println!("SHRINKING DEBUG: Span-aware bytes shrinking from {} to {} bytes", original_len, new_len);
                    return result;
                },
                ChoiceValue::Integer(val) => {
                    let original_val = *val;
                    let target = if *val > 0 { val / 3 } else { val / 3 };
                    result[i].value = ChoiceValue::Integer(target);
                    println!("SHRINKING DEBUG: Span-aware integer shrinking from {} to {}", original_val, target);
                    return result;
                },
                _ => {}
            }
        }
    }
    
    nodes.to_vec()
}

/// Fix constraint violations during shrinking transformations
fn constraint_repair_shrinking(nodes: &[ChoiceNode]) -> Vec<ChoiceNode> {
    let mut result = nodes.to_vec();
    let mut repaired = false;
    
    for choice in &mut result {
        if choice.was_forced {
            continue;
        }
        
        let new_value = match (&choice.value, &choice.constraints) {
            (ChoiceValue::Integer(value), Constraints::Integer(constraints)) => {
                let min_val = constraints.min_value.unwrap_or(i128::MIN);
                let max_val = constraints.max_value.unwrap_or(i128::MAX);
                
                if *value < min_val {
                    println!("SHRINKING DEBUG: Repaired integer {} to min constraint {}", value, min_val);
                    Some(ChoiceValue::Integer(min_val))
                } else if *value > max_val {
                    println!("SHRINKING DEBUG: Repaired integer {} to max constraint {}", value, max_val);
                    Some(ChoiceValue::Integer(max_val))
                } else {
                    None
                }
            },
            (ChoiceValue::Float(value), Constraints::Float(constraints)) => {
                if !value.is_finite() {
                    // Repair non-finite floats to a reasonable value within constraints
                    let safe_value = if constraints.min_value > 0.0 {
                        constraints.min_value
                    } else if constraints.max_value < 0.0 {
                        constraints.max_value
                    } else {
                        0.0
                    };
                    println!("SHRINKING DEBUG: Repaired non-finite float {} to {}", value, safe_value);
                    Some(ChoiceValue::Float(safe_value))
                } else {
                    // Check bounds
                    if *value < constraints.min_value {
                        println!("SHRINKING DEBUG: Repaired float {} to min constraint {}", value, constraints.min_value);
                        Some(ChoiceValue::Float(constraints.min_value))
                    } else if *value > constraints.max_value {
                        println!("SHRINKING DEBUG: Repaired float {} to max constraint {}", value, constraints.max_value);
                        Some(ChoiceValue::Float(constraints.max_value))
                    } else {
                        None
                    }
                }
            },
            _ => None,
        };
        
        if let Some(value) = new_value {
            choice.value = value;
            repaired = true;
        }
    }
    
    if repaired {
        result
    } else {
        // If no repairs needed, try to make a conservative shrinking move that respects constraints
        for choice in &mut result {
            if choice.was_forced {
                continue;
            }
            
            let shrink_result = match (&choice.value, &choice.constraints) {
                (ChoiceValue::Integer(value), Constraints::Integer(constraints)) => {
                    let min_val = constraints.min_value.unwrap_or(i128::MIN);
                    let max_val = constraints.max_value.unwrap_or(i128::MAX);
                    let target = constraints.shrink_towards.unwrap_or(0).max(min_val).min(max_val);
                    
                    if *value != target {
                        // Move more aggressively toward the target
                        let distance = (*value - target).abs();
                        let jump_size = if distance > 50 { distance / 4 } else { 1 };
                        
                        let new_value = if *value > target {
                            std::cmp::max(target, *value - jump_size)
                        } else {
                            std::cmp::min(target, *value + jump_size)
                        };
                        
                        if new_value != *value && new_value >= min_val && new_value <= max_val {
                            println!("SHRINKING DEBUG: Constraint-respecting integer shrink from {} to {} (toward target {})", value, new_value, target);
                            Some(ChoiceValue::Integer(new_value))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                },
                _ => None,
            };
            
            if let Some(new_value) = shrink_result {
                choice.value = new_value;
                return result;
            }
        }
        
        nodes.to_vec()
    }
}

/// Individual choice targeting - minimize a specific choice at given index
/// This is a fundamental building block for more sophisticated shrinking algorithms
pub fn minimize_individual_choice_at(nodes: &[ChoiceNode], target_index: usize) -> Vec<ChoiceNode> {
        if nodes.is_empty() || target_index >= nodes.len() {
            return nodes.to_vec();
        }
        
        let mut result = nodes.to_vec();
        let target_node = &nodes[target_index];
        
        // Skip forced nodes to maintain test reproducibility
        if target_node.was_forced {
            return result;
        }
        
        // Apply type-specific minimization to the target choice
        let minimized_value = match (&target_node.value, &target_node.constraints) {
            (ChoiceValue::Integer(value), Constraints::Integer(constraints)) => {
                let shrink_target = constraints.shrink_towards.unwrap_or(0);
                let min_val = constraints.min_value.unwrap_or(i128::MIN);
                let max_val = constraints.max_value.unwrap_or(i128::MAX);
                
                if *value != shrink_target {
                    // Move one step toward shrink target
                    let new_value = if *value > shrink_target {
                        std::cmp::max(shrink_target, *value - 1)
                    } else {
                        std::cmp::min(shrink_target, *value + 1)
                    };
                    
                    if new_value >= min_val && new_value <= max_val {
                        Some(ChoiceValue::Integer(new_value))
                    } else {
                        None
                    }
                } else {
                    None
                }
            },
            (ChoiceValue::Boolean(true), Constraints::Boolean(_)) => {
                // Always try to minimize true to false
                Some(ChoiceValue::Boolean(false))
            },
            (ChoiceValue::Float(value), Constraints::Float(constraints)) => {
                if value.abs() > f64::EPSILON {
                    // Try moving toward 0.0 while respecting bounds
                    let candidates = [0.0, *value / 2.0, *value * 0.9];
                    candidates.iter()
                        .find(|&&candidate| {
                            candidate.abs() < value.abs() 
                                && candidate >= constraints.min_value 
                                && candidate <= constraints.max_value
                        })
                        .map(|&candidate| ChoiceValue::Float(candidate))
                } else {
                    None
                }
            },
            (ChoiceValue::String(s), Constraints::String(constraints)) => {
                if s.len() > constraints.min_size {
                    // Try removing last character
                    let mut new_string = s.clone();
                    new_string.pop();
                    Some(ChoiceValue::String(new_string))
                } else {
                    None
                }
            },
            (ChoiceValue::Bytes(b), Constraints::Bytes(constraints)) => {
                if b.len() > constraints.min_size {
                    // Try removing last byte
                    let mut new_bytes = b.clone();
                    new_bytes.pop();
                    Some(ChoiceValue::Bytes(new_bytes))
                } else {
                    None
                }
            },
            _ => None,
        };
        
        if let Some(new_value) = minimized_value {
            println!("SHRINKING DEBUG: Minimizing choice at index {} from {:?} to {:?}", 
                   target_index, target_node.value, new_value);
            result[target_index].value = new_value;
        }
        
        result
    }
    
    /// Shrink a specific choice toward a custom target value
    /// More flexible than standard minimization - allows custom shrink targets
    pub fn shrink_choice_towards_target(nodes: &[ChoiceNode], target_index: usize, target_value: ChoiceValue) -> Vec<ChoiceNode> {
        if nodes.is_empty() || target_index >= nodes.len() {
            return nodes.to_vec();
        }
        
        let mut result = nodes.to_vec();
        let target_node = &nodes[target_index];
        
        // Skip forced nodes and mismatched types
        if target_node.was_forced || !choice_values_compatible(&target_node.value, &target_value) {
            return result;
        }
        
        let shrunk_value = match (&target_node.value, &target_value, &target_node.constraints) {
            (ChoiceValue::Integer(current), ChoiceValue::Integer(target), Constraints::Integer(constraints)) => {
                let min_val = constraints.min_value.unwrap_or(i128::MIN);
                let max_val = constraints.max_value.unwrap_or(i128::MAX);
                
                if current != target && *target >= min_val && *target <= max_val {
                    // Move toward target, but don't overshoot
                    let distance = (*current - *target).abs();
                    let step_size = std::cmp::max(1, distance / 4);
                    
                    let new_value = if *current > *target {
                        std::cmp::max(*target, *current - step_size)
                    } else {
                        std::cmp::min(*target, *current + step_size)
                    };
                    
                    Some(ChoiceValue::Integer(new_value))
                } else {
                    None
                }
            },
            (ChoiceValue::Float(current), ChoiceValue::Float(target), Constraints::Float(constraints)) => {
                if (current - target).abs() > f64::EPSILON 
                    && *target >= constraints.min_value 
                    && *target <= constraints.max_value {
                    // Move 50% of the way toward target
                    let new_value = *current + (*target - *current) * 0.5;
                    Some(ChoiceValue::Float(new_value))
                } else {
                    None
                }
            },
            _ => None,
        };
        
        if let Some(new_value) = shrunk_value {
            println!("SHRINKING DEBUG: Shrinking choice at index {} toward target {:?} from {:?} to {:?}", 
                   target_index, target_value, target_node.value, new_value);
            result[target_index].value = new_value;
        }
        
        result
    }
    
    /// Minimize a choice within custom bounds (override constraint bounds)
    /// Useful for targeted exploration of specific value ranges
    pub fn minimize_choice_with_bounds(nodes: &[ChoiceNode], target_index: usize, min_bound: ChoiceValue, max_bound: ChoiceValue) -> Vec<ChoiceNode> {
        if nodes.is_empty() || target_index >= nodes.len() {
            return nodes.to_vec();
        }
        
        let mut result = nodes.to_vec();
        let target_node = &nodes[target_index];
        
        // Skip forced nodes and incompatible bounds
        if target_node.was_forced 
            || !choice_values_compatible(&target_node.value, &min_bound)
            || !choice_values_compatible(&target_node.value, &max_bound) {
            return result;
        }
        
        let shrunk_value = match (&target_node.value, &min_bound, &max_bound) {
            (ChoiceValue::Integer(current), ChoiceValue::Integer(min_val), ChoiceValue::Integer(max_val)) => {
                if *current >= *min_val && *current <= *max_val {
                    // Try to move toward the minimum bound
                    if *current > *min_val {
                        let new_value = std::cmp::max(*min_val, *current - 1);
                        Some(ChoiceValue::Integer(new_value))
                    } else {
                        None
                    }
                } else {
                    // Clamp to bounds if outside
                    let clamped = (*current).clamp(*min_val, *max_val);
                    if clamped != *current {
                        Some(ChoiceValue::Integer(clamped))
                    } else {
                        None
                    }
                }
            },
            (ChoiceValue::Float(current), ChoiceValue::Float(min_val), ChoiceValue::Float(max_val)) => {
                if *current >= *min_val && *current <= *max_val {
                    // Try to move toward minimum bound
                    if *current > *min_val {
                        let new_value = *min_val + (*current - *min_val) * 0.5;
                        Some(ChoiceValue::Float(new_value))
                    } else {
                        None
                    }
                } else {
                    // Clamp to bounds if outside
                    let clamped = current.clamp(*min_val, *max_val);
                    if (clamped - *current).abs() > f64::EPSILON {
                        Some(ChoiceValue::Float(clamped))
                    } else {
                        None
                    }
                }
            },
            _ => None,
        };
        
        if let Some(new_value) = shrunk_value {
            println!("SHRINKING DEBUG: Minimizing choice at index {} within bounds [{:?}, {:?}] from {:?} to {:?}", 
                   target_index, min_bound, max_bound, target_node.value, new_value);
            result[target_index].value = new_value;
        }
        
        result
    }
/// Helper function to check if two ChoiceValues are type-compatible
fn choice_values_compatible(a: &ChoiceValue, b: &ChoiceValue) -> bool {
    match (a, b) {
        (ChoiceValue::Integer(_), ChoiceValue::Integer(_)) => true,
        (ChoiceValue::Boolean(_), ChoiceValue::Boolean(_)) => true,
        (ChoiceValue::Float(_), ChoiceValue::Float(_)) => true,
        (ChoiceValue::String(_), ChoiceValue::String(_)) => true,
        (ChoiceValue::Bytes(_), ChoiceValue::Bytes(_)) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::choice::{IntegerConstraints, BooleanConstraints};

    #[test]
    fn test_minimize_integer_values_conservative() {
        let nodes = vec![
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(50),
                Constraints::Integer(IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(100),
                    weights: None,
                    shrink_towards: Some(0),
                }),
                false,
            ),
        ];
        
        let minimized = minimize_integer_values_conservative(&nodes);
        
        if let ChoiceValue::Integer(value) = &minimized[0].value {
            assert_eq!(*value, 49); // Should shrink towards 0 by 1
        } else {
            panic!("Expected integer value");
        }
    }
    
    #[test]
    fn test_minimize_integer_values_aggressive() {
        let nodes = vec![
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(50),
                Constraints::Integer(IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(100),
                    weights: None,
                    shrink_towards: Some(0),
                }),
                false,
            ),
        ];
        
        let minimized = minimize_integer_values_aggressive(&nodes);
        
        if let ChoiceValue::Integer(value) = &minimized[0].value {
            assert_eq!(*value, 25); // Should shrink towards 0 by half distance
        } else {
            panic!("Expected integer value");
        }
    }
    
    #[test]
    fn test_minimize_booleans_to_false() {
        let nodes = vec![
            ChoiceNode::new(
                ChoiceType::Boolean,
                ChoiceValue::Boolean(true),
                Constraints::Boolean(BooleanConstraints { p: 0.5 }),
                false,
            ),
        ];
        
        let minimized = minimize_booleans_to_false(&nodes);
        
        if let ChoiceValue::Boolean(value) = &minimized[0].value {
            assert_eq!(*value, false); // Should shrink to false
        } else {
            panic!("Expected boolean value");
        }
    }
    
    #[test]
    fn test_delete_trailing_nodes() {
        let nodes = vec![
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(1),
                Constraints::Integer(IntegerConstraints::default()),
                false,
            ),
            ChoiceNode::new(
                ChoiceType::Boolean,
                ChoiceValue::Boolean(true),
                Constraints::Boolean(BooleanConstraints { p: 0.5 }),
                false,
            ),
        ];
        
        let result = delete_trailing_nodes(&nodes);
        assert_eq!(result.len(), 1);
        
        if let ChoiceValue::Integer(value) = &result[0].value {
            assert_eq!(*value, 1);
        } else {
            panic!("Expected first choice to remain");
        }
    }

    // Tests for advanced shrinking functions

    #[test]
    fn test_minimize_duplicated_nodes() {
        // Create a pattern that repeats
        let nodes = vec![
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(42),
                Constraints::Integer(IntegerConstraints::default()),
                false,
            ),
            ChoiceNode::new(
                ChoiceType::Boolean,
                ChoiceValue::Boolean(true),
                Constraints::Boolean(BooleanConstraints { p: 0.5 }),
                false,
            ),
            // Duplicate the pattern
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(42),
                Constraints::Integer(IntegerConstraints::default()),
                false,
            ),
            ChoiceNode::new(
                ChoiceType::Boolean,
                ChoiceValue::Boolean(true),
                Constraints::Boolean(BooleanConstraints { p: 0.5 }),
                false,
            ),
        ];
        
        let result = minimize_duplicated_nodes(&nodes);
        
        // Should have removed the duplicate pattern
        assert_eq!(result.len(), 2);
        if let ChoiceValue::Integer(value) = &result[0].value {
            assert_eq!(*value, 42);
        } else {
            panic!("Expected integer value");
        }
    }

    #[test]
    fn test_minimize_choice_order() {
        let nodes = vec![
            // High complexity choice (large integer)
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(1000),
                Constraints::Integer(IntegerConstraints::default()),
                false,
            ),
            // Low complexity choice (false boolean)
            ChoiceNode::new(
                ChoiceType::Boolean,
                ChoiceValue::Boolean(false),
                Constraints::Boolean(BooleanConstraints { p: 0.5 }),
                false,
            ),
            // Medium complexity choice (true boolean)
            ChoiceNode::new(
                ChoiceType::Boolean,
                ChoiceValue::Boolean(true),
                Constraints::Boolean(BooleanConstraints { p: 0.5 }),
                false,
            ),
        ];
        
        let result = minimize_choice_order(&nodes);
        
        // Should reorder by complexity: false (0.0), true (1.0), 1000 (1000.0)
        assert_eq!(result.len(), 3);
        if let ChoiceValue::Boolean(value) = &result[0].value {
            assert_eq!(*value, false); // Simplest should be first
        } else {
            panic!("Expected boolean false to be first after reordering");
        }
    }

    #[test]
    fn test_adaptive_delete_nodes() {
        let nodes = vec![
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(5),
                Constraints::Integer(IntegerConstraints::default()),
                false,
            ),
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(1000), // High complexity
                Constraints::Integer(IntegerConstraints::default()),
                false,
            ),
            ChoiceNode::new(
                ChoiceType::Boolean,
                ChoiceValue::Boolean(false),
                Constraints::Boolean(BooleanConstraints { p: 0.5 }),
                false,
            ),
        ];
        
        let result = adaptive_delete_nodes(&nodes);
        
        // Should delete the highest complexity choice (1000)
        assert_eq!(result.len(), 2);
        // Verify the high complexity choice was removed
        let has_1000 = result.iter().any(|choice| {
            if let ChoiceValue::Integer(val) = &choice.value {
                *val == 1000
            } else {
                false
            }
        });
        assert!(!has_1000, "High complexity choice should have been removed");
    }

    #[test]
    fn test_span_aware_shrinking() {
        let nodes = vec![
            ChoiceNode::new(
                ChoiceType::String,
                ChoiceValue::String("hello world".to_string()),
                Constraints::String(crate::choice::StringConstraints::default()),
                false,
            ),
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(100),
                Constraints::Integer(IntegerConstraints::default()),
                false,
            ),
        ];
        
        let result = span_aware_shrinking(&nodes);
        
        // Should shrink the string length
        if let ChoiceValue::String(s) = &result[0].value {
            assert!(s.len() < "hello world".len(), "String should be shrunk");
            assert!(s.len() >= 1, "String should not be empty");
        } else {
            panic!("Expected string value");
        }
    }

    #[test]
    fn test_constraint_repair_shrinking() {
        let nodes = vec![
            ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(-10), // Below min constraint
                Constraints::Integer(IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(100),
                    weights: None,
                    shrink_towards: Some(0),
                }),
                false,
            ),
        ];
        
        let result = constraint_repair_shrinking(&nodes);
        
        // Should repair the value to respect min constraint
        if let ChoiceValue::Integer(value) = &result[0].value {
            assert_eq!(*value, 0); // Should be repaired to min value
        } else {
            panic!("Expected integer value");
        }
    }

    #[test]
    fn test_constraint_repair_shrinking_float() {
        let nodes = vec![
            ChoiceNode::new(
                ChoiceType::Float,
                ChoiceValue::Float(f64::NAN),
                Constraints::Float(crate::choice::FloatConstraints {
                    min_value: 0.0,
                    max_value: 100.0,
                    allow_nan: false,
                    smallest_nonzero_magnitude: 1e-10,
                }),
                false,
            ),
        ];
        
        let result = constraint_repair_shrinking(&nodes);
        
        // Should repair NaN to a finite value
        if let ChoiceValue::Float(value) = &result[0].value {
            assert!(value.is_finite(), "NaN should be repaired to finite value");
            assert!(*value >= 0.0, "Value should respect min constraint");
        } else {
            panic!("Expected float value");
        }
    }

    #[test]
    fn test_choice_complexity() {
        let boolean_false = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(false),
            Constraints::Boolean(BooleanConstraints { p: 0.5 }),
            false,
        );
        let boolean_true = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(true),
            Constraints::Boolean(BooleanConstraints { p: 0.5 }),
            false,
        );
        let integer_small = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(5),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );
        let integer_large = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(1000),
            Constraints::Integer(IntegerConstraints::default()),
            false,
        );
        
        assert_eq!(choice_complexity(&boolean_false), 0.0);
        assert_eq!(choice_complexity(&boolean_true), 1.0);
        assert_eq!(choice_complexity(&integer_small), 5.0);
        assert_eq!(choice_complexity(&integer_large), 1000.0);
    }
}

impl MultiLevelCache {
    /// Create a new multi-level cache with default configuration
    pub fn new() -> Self {
        Self {
            l1_transformation_cache: HashMap::new(),
            l2_sequence_cache: HashMap::new(),
            l3_pattern_cache: HashMap::new(),
            config: CacheConfig::default(),
            stats: CacheStats::default(),
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_max_entries: 1000,
            l2_max_entries: 5000,
            l3_max_entries: 10000,
            ttl_seconds: 3600, // 1 hour
            enable_cache_warming: true,
            eviction_strategy: EvictionStrategy::LRU,
        }
    }
}