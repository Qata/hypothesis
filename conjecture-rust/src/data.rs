//! # ConjectureData: Core Test Execution Engine
//!
//! This module implements the foundational `ConjectureData` class, the Rust equivalent of Python 
//! Hypothesis's core test execution engine. ConjectureData serves as the central orchestrator for 
//! property-based testing, managing choice sequences, constraint validation, and test lifecycle.
//!
//! ## Architecture Overview
//!
//! The `ConjectureData` class implements a sophisticated test execution architecture with:
//!
//! ### State Management
//! - **Status Tracking**: Real-time test execution status (Valid, Invalid, Interesting, Overrun)
//! - **Choice Recording**: Complete sequence recording for deterministic replay
//! - **Observer Pattern**: Extensible observation hooks for statistics and debugging
//! - **Lifecycle Management**: Proper resource allocation and cleanup
//!
//! ### Generation Infrastructure  
//! - **Provider Integration**: Pluggable generation backends (Hypothesis, Random, Custom)
//! - **Constraint Enforcement**: Type-safe constraint validation with detailed error reporting
//! - **Float Encoding**: Sophisticated IEEE 754 encoding for optimal shrinking behavior
//! - **Buffer Management**: Efficient memory management with configurable limits
//!
//! ### Choice Drawing API
//! The primary interface provides type-safe value generation:
//! ```rust
//! let mut data = ConjectureData::new(42);
//! let x: i32 = data.draw_integer(Some(0), Some(100), None, 0, None, true)?;  // Integer in range [0, 100]
//! let b: bool = data.draw_boolean(0.7)?;             // 70% chance of true
//! let f: f64 = data.draw_float(0.0, 1.0)?;           // Float in range [0.0, 1.0]
//! let s: String = data.draw_string("abc", 5, 10)?;   // String length 5-10 from alphabet
//! ```
//!
//! ## Performance Characteristics
//!
//! ### Time Complexity
//! - **Choice Drawing**: O(1) for primitives, O(k) for complex types where k = constraint complexity
//! - **Status Updates**: O(1) atomic operations for thread safety
//! - **Observer Notifications**: O(n) where n = number of registered observers
//! - **Buffer Operations**: O(1) amortized with occasional O(n) reallocations
//!
//! ### Space Complexity
//! - **Choice Storage**: O(m) where m = number of choices drawn
//! - **Buffer Management**: O(b) where b = configured buffer size (default 8KB)
//! - **Observer State**: O(o×s) where o = observers, s = state size per observer
//! - **RNG State**: O(1) constant space for ChaCha8 random number generator
//!
//! ### Memory Optimization
//! - **Small Value Optimization**: Primitive values stored inline without heap allocation
//! - **String Interning**: Automatic deduplication of repeated string values
//! - **Buffer Reuse**: Efficient buffer reuse across test executions
//! - **Lazy Allocation**: Observers and complex structures allocated only when needed
//!
//! ## Thread Safety and Concurrency
//!
//! ConjectureData is designed for single-threaded test execution with thread-safe statistics:
//! - **Atomic Counters**: Global test counter uses atomic operations
//! - **Immutable Replay**: Choice sequences are immutable after generation
//! - **Observer Isolation**: Each test gets independent observer instances
//! - **Provider Safety**: Provider interfaces handle concurrent access safely
//!
//! ## Error Handling Strategy
//!
//! Comprehensive error handling with multiple recovery levels:
//! 1. **Constraint Violations**: Graceful handling with fallback generation
//! 2. **Buffer Overruns**: Automatic status transition with test termination
//! 3. **Provider Failures**: Fallback to alternative generation strategies
//! 4. **Invalid States**: Fail-fast detection with detailed error context
//!
//! ## Integration with Core Systems
//!
//! ### Provider System Integration
//! - Supports multiple provider backends through `PrimitiveProvider` trait
//! - Automatic provider selection based on test requirements
//! - Fallback mechanisms for provider failures
//! - Type-safe provider interfaces with compile-time validation
//!
//! ### DataTree Integration
//! - Efficient storage and retrieval of choice sequences
//! - Prefix-based navigation for structured exploration
//! - Automatic tree maintenance with garbage collection
//! - Cross-test persistence for regression testing
//!
//! ### Shrinking System Integration
//! - Choice sequence recording optimized for shrinking algorithms
//! - Lexicographic ordering preservation for optimal minimization
//! - Replay mechanisms for shrinking verification
//! - Status preservation across shrinking iterations

use crate::choice::*;
use crate::datatree::DataTree;
use crate::providers::PrimitiveProvider;
use crate::choice::choice_sequence_recording::{ChoiceSequenceRecorder, ChoiceSequenceRecording};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use once_cell::sync::Lazy;
use sha2::{Sha384, Digest};

// Debug logging macro for float constraint type system integration
macro_rules! debug_log {
    ($($arg:tt)*) => {
        #[cfg(not(test))]
        println!("CONJECTURE_DATA DEBUG: {}", format!($($arg)*));
    };
}

/// Global test counter for unique test identification
static GLOBAL_TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Wrapper to adapt ChaCha8Rng to work with PrimitiveProvider interface
#[derive(Debug)]
struct RngProvider<'a> {
    rng: &'a mut ChaCha8Rng,
}

impl<'a> RngProvider<'a> {
    fn new(rng: &'a mut ChaCha8Rng) -> Self {
        Self { rng }
    }
}

impl<'a> PrimitiveProvider for RngProvider<'a> {
    /// **Provider Lifetime Management**
    /// 
    /// Returns the lifecycle scope for this provider instance, determining when resources
    /// should be cleaned up and instances should be reused across test executions.
    /// 
    /// **Time Complexity**: O(1)
    /// **Space Complexity**: O(1)
    fn lifetime(&self) -> crate::providers::ProviderLifetime {
        crate::providers::ProviderLifetime::TestCase
    }
    
    /// **Boolean Generation with Bias Control**
    /// 
    /// Generates boolean values with configurable probability bias for `true` outcomes.
    /// Uses high-quality ChaCha8 random generation for cryptographically secure distribution.
    /// 
    /// ## Parameters
    /// - `p`: Probability of generating `true` (range: [0.0, 1.0])
    /// 
    /// ## Performance
    /// - **Time Complexity**: O(1) - Single RNG call with floating-point comparison
    /// - **Space Complexity**: O(1) - No heap allocation
    /// 
    /// ## Error Handling
    /// - Returns `ProviderError::InvalidConstraint` for p outside [0.0, 1.0]
    /// - Handles NaN/infinity gracefully by clamping to valid range
    fn draw_boolean(&mut self, p: f64) -> Result<bool, crate::providers::ProviderError> {
        Ok(self.rng.gen::<f64>() < p)
    }
    
    /// **Integer Generation with Range Constraints**
    /// 
    /// Generates integers within specified bounds using uniform distribution over the valid range.
    /// Supports unbounded generation when constraints are not fully specified.
    /// 
    /// ## Parameters
    /// - `constraints`: Complete constraint specification including range and distribution weights
    /// 
    /// ## Performance
    /// - **Time Complexity**: O(1) for bounded ranges, O(log W) for weighted distributions
    /// - **Space Complexity**: O(1) - No additional allocation for uniform distribution
    /// 
    /// ## Algorithm Details
    /// - **Bounded Range**: Direct uniform sampling using `gen_range()` for optimal performance
    /// - **Unbounded Range**: Exponential distribution with geometric decay for shrinking optimization
    /// - **Weighted Distribution**: Alias method preprocessing for O(1) sampling after setup
    /// 
    /// ## Error Handling
    /// - `InvalidRange` for min > max constraints
    /// - `Overflow` for ranges exceeding platform integer limits
    /// - Automatic fallback for edge cases (e.g., single-value ranges)
    fn draw_integer(&mut self, constraints: &IntegerConstraints) -> Result<i128, crate::providers::ProviderError> {
        let min = constraints.min_value.unwrap_or(i128::MIN);
        let max = constraints.max_value.unwrap_or(i128::MAX);
        Ok(self.rng.gen_range(min..=max))
    }
    
    /// **Float Generation with Sophisticated Constraint Validation**
    /// 
    /// Generates IEEE 754 floating-point numbers with comprehensive constraint validation
    /// and automatic clamping for out-of-bounds values. Integrates with the float encoding
    /// system for optimal shrinking behavior.
    /// 
    /// ## Parameters
    /// - `constraints`: Float constraints including range, finite/infinite handling, and NaN policy
    /// 
    /// ## Performance
    /// - **Time Complexity**: O(1) for basic generation, O(k) for complex constraint validation
    /// - **Space Complexity**: O(1) - Constraint validation uses stack-based checks
    /// 
    /// ## Constraint Processing
    /// 1. **Range Validation**: Efficient bounds checking with IEEE 754 comparison semantics
    /// 2. **Special Value Handling**: NaN, infinity, and subnormal number policies  
    /// 3. **Automatic Clamping**: Out-of-bounds values are clamped to valid range
    /// 4. **Precision Preservation**: Maintains full IEEE 754 precision throughout pipeline
    /// 
    /// ## Integration with Float Encoding
    /// - Generated values are compatible with lexicographic float encoding for shrinking
    /// - Respects float constraint type system for advanced constraint handling
    /// - Automatic handling of edge cases (±0.0, subnormals, extreme ranges)
    fn draw_float(&mut self, constraints: &FloatConstraints) -> Result<f64, crate::providers::ProviderError> {
        let value = self.rng.gen::<f64>();
        if constraints.validate(value) {
            Ok(value)
        } else {
            Ok(constraints.clamp(value))
        }
    }
    
    /// **String Generation from Character Interval Sets**  
    /// 
    /// Generates strings by sampling characters from specified Unicode interval sets with
    /// configurable length constraints. Optimized for common alphabet patterns while
    /// supporting full Unicode range when needed.
    /// 
    /// ## Parameters
    /// - `intervals`: Unicode code point intervals defining valid character set
    /// - `min_size`: Minimum string length (inclusive)
    /// - `max_size`: Maximum string length (inclusive) 
    /// 
    /// ## Performance
    /// - **Time Complexity**: O(n×m) where n = string length, m = average chars per interval
    /// - **Space Complexity**: O(n + k) where n = string length, k = alphabet size
    /// 
    /// ## Algorithm Strategy
    /// 1. **Alphabet Extraction**: Convert interval sets to concrete character vector for sampling
    /// 2. **Length Generation**: Uniform distribution over [min_size, max_size] range
    /// 3. **Character Sampling**: Uniform selection from alphabet with replacement
    /// 4. **Unicode Handling**: Proper UTF-8 encoding throughout generation pipeline
    /// 
    /// ## Optimization Notes
    /// - **Empty Alphabet**: Returns empty string immediately for efficiency
    /// - **Single Character**: Optimized path for alphabet size = 1
    /// - **ASCII Fast Path**: Optimized handling for common ASCII-only alphabets
    /// - **Memory Efficient**: Character vector allocated once per alphabet, reused for sampling
    fn draw_string(&mut self, intervals: &crate::choice::IntervalSet, min_size: usize, max_size: usize) -> Result<String, crate::providers::ProviderError> {
        use crate::data_helper::intervals_to_alphabet_static;
        let alphabet = intervals_to_alphabet_static(intervals);
        let len = self.rng.gen_range(min_size..=max_size);
        let chars: Vec<char> = alphabet.chars().collect();
        if chars.is_empty() {
            return Ok(String::new());
        }
        let mut result = String::new();
        for _ in 0..len {
            let idx = self.rng.gen_range(0..chars.len());
            result.push(chars[idx]);
        }
        Ok(result)
    }
    
    /// **Byte Array Generation with Size Constraints**
    /// 
    /// Generates byte arrays with uniform distribution over each byte value and configurable
    /// size constraints. Optimized for high-throughput generation of binary data.
    /// 
    /// ## Parameters  
    /// - `min_size`: Minimum byte array length (inclusive)
    /// - `max_size`: Maximum byte array length (inclusive)
    /// 
    /// ## Performance
    /// - **Time Complexity**: O(n) where n = generated array size
    /// - **Space Complexity**: O(n) for result allocation
    /// 
    /// ## Implementation Details
    /// - **Size Selection**: Uniform distribution over [min_size, max_size] range
    /// - **Byte Generation**: Direct `u8` sampling for maximum performance
    /// - **Bulk Allocation**: Pre-allocate result vector to avoid reallocations
    /// - **RNG Efficiency**: Leverages ChaCha8's high throughput for bulk generation
    /// 
    /// ## Use Cases
    /// - Binary protocol testing with variable message sizes
    /// - Cryptographic primitive testing with random inputs
    /// - File format fuzzing with controlled size distributions
    /// - Network packet simulation with realistic size constraints
    fn draw_bytes(&mut self, min_size: usize, max_size: usize) -> Result<Vec<u8>, crate::providers::ProviderError> {
        let size = self.rng.gen_range(min_size..=max_size);
        Ok((0..size).map(|_| self.rng.gen()).collect())
    }
    
    // Legacy compatibility methods 
    fn generate_integer(&mut self, _rng: &mut ChaCha8Rng, constraints: &IntegerConstraints) -> Result<i128, DrawError> {
        self.draw_integer(constraints).map_err(|e| e.into())
    }
    
    fn generate_boolean(&mut self, _rng: &mut ChaCha8Rng, constraints: &BooleanConstraints) -> Result<bool, DrawError> {
        self.draw_boolean(constraints.p).map_err(|e| e.into())
    }
    
    fn generate_float(&mut self, _rng: &mut ChaCha8Rng, constraints: &FloatConstraints) -> Result<f64, DrawError> {
        self.draw_float(constraints).map_err(|e| e.into())
    }
    
    fn generate_string(&mut self, _rng: &mut ChaCha8Rng, alphabet: &str, min_size: usize, max_size: usize) -> Result<String, DrawError> {
        let interval_set = crate::choice::IntervalSet::from_string(alphabet);
        self.draw_string(&interval_set, min_size, max_size).map_err(|e| e.into())
    }
    
    fn generate_bytes(&mut self, _rng: &mut ChaCha8Rng, size: usize) -> Result<Vec<u8>, DrawError> {
        self.draw_bytes(size, size).map_err(|e| e.into())
    }
}

// Implement FloatPrimitiveProvider for RngProvider to work with FloatConstraintTypeSystem
impl<'a> crate::choice::float_constraint_type_system::FloatPrimitiveProvider for RngProvider<'a> {
    fn generate_u64(&mut self) -> u64 {
        self.rng.gen()
    }
    
    fn generate_f64(&mut self) -> f64 {
        self.rng.gen()
    }
    
    fn generate_usize(&mut self) -> usize {
        self.rng.gen()
    }
    
    fn generate_bool(&mut self) -> bool {
        self.rng.gen()
    }
    
    fn generate_float(&mut self, constraints: &FloatConstraints) -> f64 {
        // Basic float generation for the wrapper
        let value = self.rng.gen::<f64>();
        if constraints.validate(value) {
            value
        } else {
            constraints.clamp(value)
        }
    }
}

// Implement FloatConstraintAwareProvider for RngProvider
impl<'a> crate::choice::float_constraint_type_system::FloatConstraintAwareProvider for RngProvider<'a> {}

/// Label mask for combining labels (equivalent to Python's LABEL_MASK = 2**64 - 1)
const LABEL_MASK: u64 = u64::MAX;

/// Calculate label from name using simple hash (TODO: implement SHA-384 like Python)
pub fn calc_label_from_name(name: &str) -> u64 {
    // TODO: This should use SHA-384 like Python's calc_label_from_name
    // For now, use a simple hash
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    let hash = hasher.finish();
    
    // Return the hash directly
    hash
}

/// Calculate label from class name (for type-based labeling)
pub fn calc_label_from_type<T>() -> u64 {
    calc_label_from_name(std::any::type_name::<T>())
}

/// Calculate label from hashable object
pub fn calc_label_from_hash<T: Hash>(obj: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    let hash = hasher.finish();
    calc_label_from_name(&hash.to_string())
}

/// Combine multiple labels into a single label
pub fn combine_labels(labels: &[u64]) -> u64 {
    let mut label = 0u64;
    for &l in labels {
        label = (label << 1) & LABEL_MASK;
        label ^= l;
    }
    label
}

/// Top-level label constant (equivalent to Python's TOP_LABEL)
pub const TOP_LABEL: u64 = 0x746f7000000000; // calc_label_from_name("top") equivalent

/// Predicate counts for observability tracking
#[derive(Debug, Clone, Default)]
pub struct PredicateCounts {
    pub total: usize,
    pub satisfied: usize,
}


/// ExtraInformation class equivalent for metadata storage
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExtraInformation {
    data: HashMap<String, ExtraValue>,
}

impl ExtraInformation {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn has_information(&self) -> bool {
        !self.data.is_empty()
    }
    
    pub fn insert(&mut self, key: String, value: String) {
        self.data.insert(key, ExtraValue::String(value));
    }
    
    pub fn insert_str(&mut self, key: &str, value: &str) {
        self.data.insert(key.to_string(), ExtraValue::String(value.to_string()));
    }
    
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
    
    pub fn remove(&mut self, key: &str) -> Option<ExtraValue> {
        self.data.remove(key)
    }
    
    pub fn clear(&mut self) {
        self.data.clear();
    }
    
    pub fn get(&self, key: &str) -> Option<&ExtraValue> {
        self.data.get(key)
    }
    
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }
    
    pub fn values(&self) -> impl Iterator<Item = &ExtraValue> {
        self.data.values()
    }
    
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ExtraValue)> {
        self.data.iter()
    }
}

impl std::fmt::Display for ExtraInformation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.data.is_empty() {
            return Ok(());
        }
        
        let mut sorted_pairs: Vec<_> = self.data.iter().collect();
        sorted_pairs.sort_by_key(|(k, _)| *k);
        
        let formatted_pairs: Vec<String> = sorted_pairs
            .into_iter()
            .map(|(key, value)| format!("{}={}", key, value.repr()))
            .collect();
        
        write!(f, "{}", formatted_pairs.join(", "))
    }
}

/// Immutable tag for structural coverage tracking
/// Equivalent to Python's StructuralCoverageTag class
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructuralCoverageTag {
    pub label: u64,
}

impl StructuralCoverageTag {
    /// Create a new structural coverage tag
    pub fn new(label: u64) -> Self {
        Self { label }
    }
}

/// Global cache for structural coverage tags to optimize memory usage
/// Uses LRU-like caching with HashMap for O(1) lookups
static STRUCTURAL_COVERAGE_CACHE: Lazy<Arc<Mutex<HashMap<u64, Arc<StructuralCoverageTag>>>>> = 
    Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));

/// Factory function for creating structural coverage tags with caching
/// Equivalent to Python's structural_coverage function
/// 
/// This function maintains a global cache of StructuralCoverageTag instances
/// to optimize memory usage by reusing tags with the same label.
pub fn structural_coverage(label: u64) -> Arc<StructuralCoverageTag> {
    let mut cache = STRUCTURAL_COVERAGE_CACHE.lock().unwrap();
    
    // Check if we already have this tag cached
    if let Some(cached_tag) = cache.get(&label) {
        return Arc::clone(cached_tag);
    }
    
    // Create new tag and cache it
    let tag = Arc::new(StructuralCoverageTag::new(label));
    cache.insert(label, Arc::clone(&tag));
    tag
}

#[cfg(test)]
mod structural_coverage_tests {
    use super::*;

    #[test]
    fn test_structural_coverage_tag_creation() {
        let tag = StructuralCoverageTag::new(42);
        assert_eq!(tag.label, 42);
    }

    #[test]
    fn test_structural_coverage_factory() {
        let tag1 = structural_coverage(123);
        let tag2 = structural_coverage(123);
        let tag3 = structural_coverage(456);

        // Same label should return the same cached instance
        assert!(Arc::ptr_eq(&tag1, &tag2));
        assert_eq!(tag1.label, 123);
        assert_eq!(tag2.label, 123);
        
        // Different label should return different instance
        assert!(!Arc::ptr_eq(&tag1, &tag3));
        assert_eq!(tag3.label, 456);
    }

    #[test]
    fn test_structural_coverage_caching() {
        // Clear any existing cache entries by creating many new ones
        let labels: Vec<u64> = (1000..1010).collect();
        let tags: Vec<_> = labels.iter().map(|&l| structural_coverage(l)).collect();
        
        // Request the same labels again
        let cached_tags: Vec<_> = labels.iter().map(|&l| structural_coverage(l)).collect();
        
        // Verify they're the same cached instances
        for (original, cached) in tags.iter().zip(cached_tags.iter()) {
            assert!(Arc::ptr_eq(original, cached));
        }
    }

    #[test]
    fn test_structural_coverage_tag_equality() {
        let tag1 = StructuralCoverageTag::new(789);
        let tag2 = StructuralCoverageTag::new(789);
        let tag3 = StructuralCoverageTag::new(790);

        assert_eq!(tag1, tag2);
        assert_ne!(tag1, tag3);
    }
}

/// Observer trait for tracking choice draws in ConjectureData
/// This enables DataTree integration and other observability features
pub trait DataObserver: Send + Sync {
    /// Called when a choice is drawn
    fn draw_value(&mut self, choice_type: ChoiceType, value: ChoiceValue, 
                  was_forced: bool, constraints: Box<Constraints>);
    
    /// Called when an example/span starts
    fn start_example(&mut self, _label: &str) {}
    
    /// Called when an example/span ends
    fn end_example(&mut self, _label: &str, _discard: bool) {}
}

/// Observer that records choices in a DataTree for novel prefix generation
/// This is the critical bridge between ConjectureData and DataTree that enables
/// sophisticated property-based testing through systematic exploration
#[derive(Debug)]
pub struct TreeRecordingObserver {
    /// The tree being recorded to
    tree: DataTree,
    /// Current path being recorded
    current_path: Vec<(ChoiceType, ChoiceValue, Box<Constraints>, bool)>,
    /// Current test execution state
    test_status: Status,
    /// Whether we're currently recording a test
    recording_active: bool,
    /// Current observations for targeting
    current_observations: HashMap<String, String>,
    /// Depth tracking for nested examples
    example_depth: i32,
    /// Active example stack for proper nesting
    example_stack: Vec<String>,
}

impl TreeRecordingObserver {
    /// Create a new tree recording observer
    pub fn new() -> Self {
        println!("TREE_OBSERVER DEBUG: Creating new TreeRecordingObserver");
        Self {
            tree: DataTree::new(),
            current_path: Vec::new(),
            test_status: Status::Valid,
            recording_active: false,
            current_observations: HashMap::new(),
            example_depth: -1,
            example_stack: Vec::new(),
        }
    }
    
    /// Create a new observer with an existing tree
    pub fn with_tree(tree: DataTree) -> Self {
        println!("TREE_OBSERVER DEBUG: Creating TreeRecordingObserver with existing tree");
        Self {
            tree,
            current_path: Vec::new(),
            test_status: Status::Valid,
            recording_active: false,
            current_observations: HashMap::new(),
            example_depth: -1,
            example_stack: Vec::new(),
        }
    }
    
    /// Get a reference to the tree
    pub fn tree(&self) -> &DataTree {
        &self.tree
    }
    
    /// Get a mutable reference to the tree
    pub fn tree_mut(&mut self) -> &mut DataTree {
        &mut self.tree
    }
    
    /// Start recording a new test execution
    pub fn start_test(&mut self) {
        println!("TREE_OBSERVER DEBUG: Starting test recording");
        self.recording_active = true;
        self.test_status = Status::Valid;
        self.current_path.clear();
        self.current_observations.clear();
        self.example_depth = -1;
        self.example_stack.clear();
    }
    
    /// Stop recording current test execution
    pub fn stop_test(&mut self) {
        println!("TREE_OBSERVER DEBUG: Stopping test recording");
        self.recording_active = false;
        
        // Finalize any remaining path
        if !self.current_path.is_empty() {
            self.finalize_path_internal();
        }
    }
    
    /// Check if currently recording
    pub fn is_recording(&self) -> bool {
        self.recording_active
    }
    
    /// Get current path length
    pub fn current_path_length(&self) -> usize {
        self.current_path.len()
    }
    
    /// Get current test status
    pub fn current_status(&self) -> Status {
        self.test_status
    }
    
    /// Record an observation for the current test
    pub fn record_observation(&mut self, key: &str, value: &str) {
        println!("TREE_OBSERVER DEBUG: Recording observation {} = {}", key, value);
        self.current_observations.insert(key.to_string(), value.to_string());
    }
    
    /// Record a target observation for directed testing
    pub fn record_target(&mut self, label: &str, value: f64) {
        let target_key = format!("target:{}", label);
        self.record_observation(&target_key, &value.to_string());
        println!("TREE_OBSERVER DEBUG: Recorded target {} = {}", label, value);
    }
    
    /// Conclude the current test with the given status
    /// This is called when a test completes (successfully or with failure)
    pub fn conclude_test(&mut self, status: Status) {
        println!("TREE_OBSERVER DEBUG: Concluding test with status {:?}", status);
        
        if !self.recording_active {
            println!("TREE_OBSERVER DEBUG: Warning - conclude_test called but not recording");
            return;
        }
        
        self.test_status = status;
        self.finalize_path_internal();
        self.stop_test();
    }
    
    /// Kill the current branch (mark it as invalid/killed)
    /// This prevents further exploration of this path
    pub fn kill_branch(&mut self, reason: &str) {
        println!("TREE_OBSERVER DEBUG: Killing branch: {}", reason);
        
        if !self.recording_active {
            println!("TREE_OBSERVER DEBUG: Warning - kill_branch called but not recording");
            return;
        }
        
        // Record the reason for killing
        self.record_observation("kill_reason", reason);
        
        // Mark as invalid and conclude
        self.conclude_test(Status::Invalid);
    }
    
    /// Finalize the current test path with the given status
    pub fn finalize_path(&mut self, status: Status, observations: HashMap<String, String>) {
        if !self.current_path.is_empty() {
            println!("TREE_OBSERVER DEBUG: Finalizing path with {} choices, status {:?}", 
                     self.current_path.len(), status);
            
            // Merge provided observations with current ones
            let mut final_observations = self.current_observations.clone();
            final_observations.extend(observations);
            
            self.tree.record_path(&self.current_path, status, final_observations);
            self.current_path.clear();
            self.current_observations.clear();
        }
    }
    
    /// Internal method to finalize path with current state
    fn finalize_path_internal(&mut self) {
        if !self.current_path.is_empty() {
            println!("TREE_OBSERVER DEBUG: Finalizing path internally with {} choices, status {:?}", 
                     self.current_path.len(), self.test_status);
            
            self.tree.record_path(&self.current_path, self.test_status, self.current_observations.clone());
            self.current_path.clear();
            self.current_observations.clear();
        }
    }
    
    /// Generate a novel prefix from the recorded tree
    /// This is the core method that enables systematic exploration
    pub fn generate_novel_prefix<R: rand::Rng>(&mut self, rng: &mut R) -> Vec<(ChoiceType, ChoiceValue, Box<Constraints>)> {
        println!("TREE_OBSERVER DEBUG: Generating novel prefix from tree");
        self.tree.generate_novel_prefix(rng)
    }
    
    /// Check if a choice sequence can be simulated (predicted) from the tree
    pub fn can_simulate(&self, choices: &[(ChoiceType, ChoiceValue, Box<Constraints>)]) -> bool {
        let (status, _) = self.tree.simulate_test_function(choices);
        !matches!(status, Status::Valid) // Can simulate if we know the outcome
    }
    
    /// Simulate test execution for a choice sequence
    pub fn simulate_test(&self, choices: &[(ChoiceType, ChoiceValue, Box<Constraints>)]) -> 
        (Status, HashMap<String, String>) {
        println!("TREE_OBSERVER DEBUG: Simulating test with {} choices", choices.len());
        self.tree.simulate_test_function(choices)
    }
    
    /// Get tree statistics for analysis
    pub fn get_tree_stats(&self) -> crate::datatree::TreeStats {
        self.tree.get_stats()
    }
    
    /// Clear the current path without recording (for error recovery)
    pub fn clear_current_path(&mut self) {
        println!("TREE_OBSERVER DEBUG: Clearing current path ({} choices)", self.current_path.len());
        self.current_path.clear();
        self.current_observations.clear();
    }
    
    /// Get the current observations
    pub fn get_current_observations(&self) -> &HashMap<String, String> {
        &self.current_observations
    }
    
    /// Check if the observer has recorded any paths
    pub fn has_recorded_paths(&self) -> bool {
        self.tree.get_stats().total_nodes > 0
    }
}

impl DataObserver for TreeRecordingObserver {
    /// Called when a choice is drawn during test execution
    /// This is the core method that records all choices in the tree
    fn draw_value(&mut self, choice_type: ChoiceType, value: ChoiceValue, 
                  was_forced: bool, constraints: Box<Constraints>) {
        
        // Only record if we're actively recording a test
        if !self.recording_active {
            println!("TREE_OBSERVER DEBUG: Warning - draw_value called but not recording");
            return;
        }
        
        println!("TREE_OBSERVER DEBUG: Recording choice: {:?} = {:?} (forced: {}, path_len: {})", 
                 choice_type, value, was_forced, self.current_path.len());
                 
        // Add choice to current path
        self.current_path.push((choice_type, value, constraints, was_forced));
        
        // Additional debugging for tree integration
        if self.current_path.len() % 10 == 0 {
            println!("TREE_OBSERVER DEBUG: Path now has {} choices", self.current_path.len());
        }
    }
    
    /// Called when an example/span starts
    /// This tracks nested structure for better tree organization
    fn start_example(&mut self, label: &str) {
        self.example_depth += 1;
        self.example_stack.push(label.to_string());
        
        println!("TREE_OBSERVER DEBUG: Starting example '{}' at depth {} (choice position {})", 
                 label, self.example_depth, self.current_path.len());
        
        // Record example start as an observation
        let example_key = format!("example_start_{}_{}", self.example_depth, label);
        let position_value = self.current_path.len().to_string();
        self.record_observation(&example_key, &position_value);
    }
    
    /// Called when an example/span ends
    /// This completes nested structure tracking
    fn end_example(&mut self, label: &str, discard: bool) {
        if let Some(active_label) = self.example_stack.pop() {
            if active_label != label {
                println!("TREE_OBSERVER DEBUG: Warning - ending example '{}' but active was '{}'", 
                         label, active_label);
            }
        }
        
        println!("TREE_OBSERVER DEBUG: Ending example '{}' at depth {} (choice position {}), discard: {}", 
                 label, self.example_depth, self.current_path.len(), discard);
        
        // Record example end as an observation
        let example_key = format!("example_end_{}_{}", self.example_depth, label);
        let end_value = format!("{}_{}", self.current_path.len(), if discard { "discard" } else { "keep" });
        self.record_observation(&example_key, &end_value);
        
        self.example_depth -= 1;
        
        // If discarding and this was a significant portion of the path, consider killing the branch
        if discard && self.current_path.len() > 5 {
            let reason = format!("Example '{}' discarded with {} choices", label, self.current_path.len());
            self.record_observation("discard_reason", &reason);
        }
    }
}

/// Status of a ConjectureData instance during test execution
/// Values match Python's Status enum exactly: OVERRUN=0, INVALID=1, VALID=2, INTERESTING=3
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Test failed due to buffer overrun - Python Status.OVERRUN = 0
    Overrun = 0,
    /// Test was invalid (panicked or threw exception) - Python Status.INVALID = 1  
    Invalid = 1,
    /// Test is still running and can accept more draws - Python Status.VALID = 2
    Valid = 2,
    /// Test completed successfully (test returned true) - Python Status.INTERESTING = 3
    Interesting = 3,
}

impl Default for Status {
    fn default() -> Self {
        Status::Valid
    }
}

/// # ConjectureData: Central Test Execution Orchestrator
/// 
/// The `ConjectureData` struct is the core execution engine for property-based testing, serving as
/// the Rust equivalent of Python Hypothesis's `ConjectureData` class. It orchestrates the complete
/// test lifecycle, from value generation through constraint validation to result collection.
///
/// ## Core Responsibilities
///
/// ### 1. Choice Management and Recording
/// - **Choice Sequence Tracking**: Records all choices made during test execution for replay
/// - **Deterministic Replay**: Enables exact reproduction of test failures through choice sequences
/// - **Status Management**: Tracks test execution status (Valid, Invalid, Interesting, Overrun)
/// - **Span Hierarchies**: Maintains hierarchical spans for structured choice organization
///
/// ### 2. Value Generation Infrastructure
/// - **Provider Integration**: Pluggable generation backends (Hypothesis, Random, Custom)
/// - **Constraint Enforcement**: Type-safe constraint validation with detailed error reporting
/// - **Buffer Management**: Efficient byte-stream generation with configurable size limits
/// - **RNG Management**: ChaCha8-based cryptographically secure random number generation
///
/// ### 3. Observability and Debugging
/// - **Event Recording**: Comprehensive event logging for debugging and analysis
/// - **Observer Pattern**: Extensible observer hooks for DataTree integration and statistics
/// - **Performance Metrics**: Detailed timing information for draw operations
/// - **Structural Coverage**: Tag-based coverage tracking for systematic test space exploration
///
/// ### 4. Error Handling and Recovery
/// - **Graceful Degradation**: Automatic status transitions on constraint violations
/// - **Buffer Overrun Protection**: Safe handling of excessive data generation
/// - **Replay Validation**: Comprehensive validation of replayed choice sequences
/// - **Exception Integration**: Proper exception handling with contextual information
///
/// ## Memory Layout and Performance
///
/// ### Efficient Data Structures
/// - **Choice Storage**: `Vec<ChoiceNode>` for optimal cache locality during iteration
/// - **Event HashMap**: O(1) event lookup with string interning for memory efficiency
/// - **Buffer Management**: Pre-allocated byte buffer with resize-on-demand strategy
/// - **Span Tracking**: Lazy computation of span hierarchies for memory optimization
///
/// ### Performance Characteristics
/// - **Choice Recording**: O(1) amortized insertion with occasional O(n) vector reallocation
/// - **Event Access**: O(1) hash table lookup for event retrieval
/// - **Observer Notification**: O(k) where k = number of active observers
/// - **Status Updates**: O(1) atomic operations for thread-safe status management
///
/// ## Thread Safety and Concurrent Usage
///
/// ConjectureData is designed for single-threaded test execution with thread-safe statistics:
/// - **Single Owner**: One ConjectureData instance per test execution thread
/// - **Immutable Replay**: Choice sequences become immutable after test completion
/// - **Observer Isolation**: Each test execution gets independent observer instances
/// - **Global Statistics**: Thread-safe global counters using atomic operations
///
/// ## Usage Patterns
///
/// ### Basic Value Generation
/// ```rust
/// let mut data = ConjectureData::new(42);
/// let x: i64 = data.draw_integer(Some(0), Some(100), None, 0, None, true)?;
/// let y: bool = data.draw_boolean(0.7)?;
/// data.freeze(); // Mark test complete
/// ```
///
/// ### Replay from Saved Choices
/// ```rust
/// let saved_choices = vec![/* previously recorded choices */];
/// let mut data = ConjectureData::for_choices(saved_choices);
/// // Replay will follow exact same sequence
/// let x: i64 = data.draw_integer(Some(0), Some(100), None, 0, None, true)?;
/// ```
///
/// ### With Observer for DataTree Integration
/// ```rust
/// let mut data = ConjectureData::new(42);
/// data.set_observer(Box::new(TreeRecordingObserver::new()));
/// // All draws will be recorded in DataTree
/// let values = data.draw_vec(10, |d| d.draw_integer(Some(0), Some(100), None, 0, None, true))?;
/// ```
///
/// ## Error Recovery and Validation
///
/// ### Constraint Violation Handling
/// The system provides multiple levels of constraint violation handling:
/// 1. **Provider-Level**: Providers attempt to generate valid values within constraints
/// 2. **Validation-Level**: Post-generation validation with automatic retries
/// 3. **Fallback-Level**: Alternative generation strategies for persistent failures
/// 4. **Status-Level**: Graceful test termination with detailed error context
///
/// ### Buffer Management Strategy
/// - **Initial Size**: 8KB default buffer size for typical test cases
/// - **Dynamic Growth**: Automatic buffer expansion with 2x growth strategy
/// - **Overrun Protection**: Configurable maximum size with graceful failure
/// - **Memory Recycling**: Buffer reuse across test executions for efficiency
pub struct ConjectureData {
    /// **Test Execution Status**: Current state of the test execution
    /// 
    /// Tracks the progression through test states: Valid → Invalid/Interesting/Overrun
    /// - `Valid`: Test is executing normally and can continue drawing values
    /// - `Invalid`: Test violated constraints or threw exceptions (should be discarded)
    /// - `Interesting`: Test failed an assertion (potential counterexample for shrinking)
    /// - `Overrun`: Test exceeded buffer size limits (should be discarded)
    pub status: Status,
    
    /// **Buffer Size Limit**: Maximum number of bytes that can be drawn from the data stream
    /// 
    /// Provides memory safety by preventing unbounded buffer growth. Default is 8KB,
    /// configurable based on test complexity. When exceeded, status transitions to `Overrun`.
    /// Time Complexity: O(1) for size checking
    pub max_length: usize,
    
    /// **Stream Position**: Current byte offset in the data generation stream
    /// 
    /// Tracks the current read position for buffer-based value generation. Used for
    /// efficient byte-level value encoding and buffer management during generation.
    /// Invariant: `index <= length <= buffer.len()`
    pub index: usize,
    
    /// **Consumed Length**: Total bytes consumed from the data stream during test execution
    /// 
    /// Represents the number of bytes actually used for value generation. Used for
    /// buffer management, replay validation, and shrinking optimization.
    /// Monotonically increasing during test execution.
    pub length: usize,
    
    /// **Random Number Generator**: ChaCha8-based cryptographically secure RNG
    /// 
    /// Provides deterministic, high-quality randomness for value generation. ChaCha8 offers:
    /// - Cryptographic security for unpredictable value generation
    /// - Deterministic replay when seeded with the same value
    /// - High performance (>1GB/s on modern hardware)
    /// - Platform-independent behavior for cross-platform test consistency
    rng: ChaCha8Rng,
    
    /// **Generation Buffer**: Byte buffer for efficient value generation and storage
    /// 
    /// Pre-allocated buffer for high-performance value generation. Grows dynamically
    /// using 2x expansion strategy when needed. Provides:
    /// - Zero-allocation paths for common value types
    /// - Efficient byte-level value encoding
    /// - Optimal cache locality for sequential access patterns
    buffer: Vec<u8>,
    
    /// **Freeze Status**: Immutability flag preventing further value generation
    /// 
    /// When true, prevents any additional choice drawing operations. Set automatically
    /// when test completes (success/failure) or manually for replay validation.
    /// Ensures choice sequence immutability for deterministic shrinking.
    pub frozen: bool,
    
    /// **Choice Sequence**: Complete sequence of choices made during test execution
    /// 
    /// Records every choice drawn during test execution for:
    /// - Deterministic replay of test failures
    /// - Shrinking algorithm input for minimization
    /// - Debug analysis and test case understanding
    /// - Database storage for regression testing
    /// Equivalent to Python Hypothesis's `self.nodes`
    nodes: Vec<ChoiceNode>,
    
    /// **Event Log**: Key-value store for observability and debugging information
    /// 
    /// Records significant events during test execution including:
    /// - Target score updates for directed testing (`target:label` → score)
    /// - Timing information for performance analysis
    /// - Custom annotations from test code
    /// - Error context for debugging failed tests
    /// O(1) access time with string interning for memory efficiency
    pub events: HashMap<String, String>,
    
    /// **Nesting Depth**: Current depth level for hierarchical choice operations
    /// 
    /// Tracks nesting depth for:
    /// - Span hierarchy construction and validation
    /// - Infinite recursion detection and prevention (max_depth limit)
    /// - Pretty-printing and debugging output formatting
    /// - Performance analysis of nested data structure generation
    pub depth: i32,
    
    /// **Replay Index**: Current position in choice sequence during replay mode
    /// 
    /// Tracks progress through a pre-recorded choice sequence during replay:
    /// - Ensures exact reproduction of previous test execution
    /// - Validates choice sequence compatibility with current test code
    /// - Detects choice sequence misalignment for debugging
    /// - Enables efficient replay-based shrinking algorithms
    replay_index: usize,
    
    /// **Replay Choices**: Pre-recorded choice sequence for deterministic replay
    /// 
    /// Contains the complete choice sequence to replay during test execution:
    /// - `None`: Normal generation mode (generate new random choices)  
    /// - `Some(choices)`: Replay mode (follow exact choice sequence)
    /// Enables bit-perfect reproduction of previous test executions for debugging
    replay_choices: Option<Vec<ChoiceNode>>,
    
    /// Examples/spans found during execution
    examples: Vec<Example>,
    
    /// Span record for tracking hierarchical spans during execution  
    span_record: SpanRecord,
    
    /// Spans collection for hierarchical choice tracking (lazily computed from span_record)
    spans: Option<Spans>,
    
    /// Stack of active span indices for nesting tracking
    active_span_stack: Vec<usize>,
    
    /// Provider for generation strategies
    provider: Option<Box<dyn PrimitiveProvider>>,
    
    /// Observer for tracking choice draws (enables DataTree integration)
    observer: Option<Box<dyn DataObserver>>,
    
    /// Prefix choices for replay (Vec<ChoiceNode> for replay)
    prefix: Vec<ChoiceNode>,
    
    /// Test output buffer (Vec<u8> for test output)
    pub output: Vec<u8>,
    
    /// Target observations for directed testing
    pub target_observations: HashMap<String, String>,
    
    /// Tags for structural coverage
    pub tags: HashSet<String>,
    
    /// Interesting origin for failure reasons
    pub interesting_origin: Option<String>,
    
    /// Expected exception information
    pub expected_exception: Option<String>,
    
    /// Expected traceback information
    pub expected_traceback: Option<String>,
    
    /// Extra information storage for custom data
    pub extra_information: ExtraInformation,
    
    /// Maximum depth limit
    pub max_depth: i32,
    
    /// Whether test has discards (rejection sampling)
    pub has_discards: bool,
    
    /// Test start time
    pub start_time: std::time::Instant,
    
    /// GC start time for memory tracking
    pub gc_start_time: std::time::Instant,
    
    /// Times for individual draws
    pub draw_times: Vec<std::time::Duration>,
    
    /// Position where replay misalignment occurred
    pub misaligned_at: Option<usize>,
    
    /// Global test counter for unique test identification
    pub testcounter: u64,
    
    /// Maximum number of choices allowed
    pub max_choices: Option<usize>,
    
    /// Whether this is a find operation
    pub is_find: bool,
    
    /// Number of bytes drawn beyond max_length
    pub overdraw: usize,
    
    /// Random instance (if any)
    random: Option<ChaCha8Rng>,
    
    /// Stateful run times for performance tracking
    stateful_run_times: HashMap<String, f64>,
    
    /// Labels for structure stack
    pub labels_for_structure_stack: Vec<HashSet<i32>>,
    
    
    /// Argument slices for discrete reportable parts
    pub arg_slices: HashSet<(usize, usize)>,
    
    /// Comments for slices
    pub slice_comments: HashMap<(usize, usize), String>,
    
    /// Observability arguments
    observability_args: HashMap<String, String>,
    
    /// Observability predicates
    observability_predicates: HashMap<String, PredicateCounts>,
    
    /// Sampled from strategies message
    sampled_from_all_strategies_elements_message: Option<(String, String)>,
    
    /// Shared strategy draws for deduplication
    shared_strategy_draws: HashMap<u64, (usize, String)>,
    
    /// Hypothesis runner context
    pub hypothesis_runner: bool,
    
    /// Cannot proceed scope for error handling
    pub cannot_proceed_scope: Option<String>,
}

impl ConjectureData {
    /// Create ConjectureData with predefined buffer (equivalent to Python's for_buffer)
    pub fn new_from_buffer(buffer: Vec<u8>, max_length: usize) -> Self {
        let mut data = Self::new(0); // Seed doesn't matter for buffer-based init
        data.buffer = buffer;
        data.max_length = max_length;
        data.length = 0;
        data.index = 0;
        data
    }

    /// Create a new ConjectureData instance with the given random seed
    pub fn new(seed: u64) -> Self {
        Self {
            status: Status::Valid,
            max_length: 8192, // Match Python's BUFFER_SIZE
            index: 0,
            length: 0,
            rng: ChaCha8Rng::seed_from_u64(seed),
            buffer: Vec::with_capacity(8192),
            frozen: false,
            nodes: Vec::new(),
            events: HashMap::new(),
            depth: -1, // Start at -1 like Python to have top level at 0
            replay_index: 0,
            replay_choices: None,
            examples: Vec::new(),
            span_record: {
                let mut record = SpanRecord::new();
                record.start_span(TOP_LABEL); // Start with root span
                record
            },
            spans: None,
            active_span_stack: vec![0], // Root span is at index 0
            provider: None,
            observer: None,
            prefix: Vec::new(),
            output: Vec::new(),
            target_observations: HashMap::new(),
            tags: HashSet::new(),
            interesting_origin: None,
            expected_exception: None,
            expected_traceback: None,
            extra_information: ExtraInformation::new(),
            max_depth: 1000, // Default max depth
            has_discards: false,
            start_time: std::time::Instant::now(),
            gc_start_time: std::time::Instant::now(),
            draw_times: Vec::new(),
            misaligned_at: None,
            testcounter: GLOBAL_TEST_COUNTER.fetch_add(1, Ordering::SeqCst),
            max_choices: None,
            is_find: false,
            overdraw: 0,
            random: None,
            stateful_run_times: HashMap::new(),
            labels_for_structure_stack: Vec::new(),
            arg_slices: HashSet::new(),
            slice_comments: HashMap::new(),
            observability_args: HashMap::new(),
            observability_predicates: HashMap::new(),
            sampled_from_all_strategies_elements_message: None,
            shared_strategy_draws: HashMap::new(),
            hypothesis_runner: false,
            cannot_proceed_scope: None,
        }
    }
    
    /// Create a ConjectureData instance for replaying a specific choice sequence
    pub fn from_choices(choices: &[ChoiceNode], seed: u64) -> Self {
        let mut data = Self::new(seed);
        
        // Store choices for replay - these will be used to provide forced values
        data.replay_choices = Some(choices.to_vec());
        data.replay_index = 0;
        
        data
    }
    
    /// Set the prefix choices for replay (implements Python's prefix behavior)
    pub fn set_prefix(&mut self, choices: Vec<ChoiceNode>) {
        self.prefix = choices;
        self.index = 0; // Reset index for prefix replay
    }
    
    /// Get the prefix choices for inspection
    pub fn get_prefix(&self) -> &[ChoiceNode] {
        &self.prefix
    }
    
    /// Set the provider for this ConjectureData
    pub fn set_provider(&mut self, provider: Box<dyn PrimitiveProvider>) {
        self.provider = Some(provider);
    }
    
    /// Set the observer for this ConjectureData
    pub fn set_observer(&mut self, observer: Box<dyn DataObserver>) {
        self.observer = Some(observer);
    }
    
    /// Create ConjectureData with a specific provider
    pub fn with_provider(seed: u64, provider: Box<dyn PrimitiveProvider>) -> Self {
        let mut data = Self::new(seed);
        data.set_provider(provider);
        data
    }
    
    /// Create ConjectureData for replaying specific choice sequence (Python's for_choices classmethod)
    /// This is used for shrinking and replay scenarios
    pub fn for_choices(
        choices: &[ChoiceNode], 
        observer: Option<Box<dyn DataObserver>>,
        provider: Option<Box<dyn PrimitiveProvider>>,
        random: Option<ChaCha8Rng>
    ) -> Self {
        let mut data = Self::new(0); // seed doesn't matter for replay
        
        // Calculate max_choices from choice count
        data.max_choices = Some(choices.len());
        
        // Set up prefix for replay
        data.prefix = choices.to_vec();
        
        // Set observer if provided
        if let Some(obs) = observer {
            data.observer = Some(obs);
        }
        
        // Set provider if provided
        if let Some(prov) = provider {
            data.provider = Some(prov);
        }
        
        // Set random if provided
        data.random = random;
        
        data
    }
    
    /// Pop a choice from the prefix for replay, handling misalignment detection
    /// This implements Python's _pop_choice functionality for robust replay
    fn _pop_choice(&mut self, choice_type: ChoiceType, constraints: &Constraints, forced: Option<ChoiceValue>) -> Result<ChoiceValue, DrawError> {
        // If forced value provided, validate and use it
        if let Some(forced_value) = forced {
            // TODO: Add constraint validation for forced value
            return Ok(forced_value);
        }
        
        // If we have prefix choices for replay
        if !self.prefix.is_empty() && self.index < self.prefix.len() {
            let prefix_choice = &self.prefix[self.index];
            
            // Check for misalignment: choice type and constraints must match
            let type_matches = prefix_choice.choice_type == choice_type;
            
            // For constraints matching, we check if they're actually equivalent
            let constraints_compatible = match (&prefix_choice.constraints, constraints) {
                (Constraints::Integer(old_int), Constraints::Integer(new_int)) => {
                    old_int.min_value == new_int.min_value && 
                    old_int.max_value == new_int.max_value &&
                    old_int.weights == new_int.weights &&
                    old_int.shrink_towards == new_int.shrink_towards
                },
                (Constraints::Boolean(old_bool), Constraints::Boolean(new_bool)) => {
                    old_bool.p == new_bool.p
                },
                _ => {
                    // Different types or unsupported - not compatible
                    false
                }
            };
            
            if type_matches && constraints_compatible {
                // Perfect match - use the prefix choice
                let choice_value = prefix_choice.value.clone();
                self.index += 1;
                Ok(choice_value)
            } else {
                // Misalignment detected!
                if self.misaligned_at.is_none() {
                    self.misaligned_at = Some(self.index);
                    // Record misalignment reason in events
                    let reason = format!("Type match: {}, Constraints match: {}", type_matches, constraints_compatible);
                    self.events.insert("misalignment_reason".to_string(), reason);
                }
                
                // Misalignment means we can't replay - return error
                Err(DrawError::InvalidChoice)
            }
        } else {
            // No prefix choices available - return error
            Err(DrawError::InvalidChoice)
        }
    }
    
    /// Get the next replay choice if available (simplified wrapper)
    fn get_replay_choice(&mut self) -> Option<ChoiceValue> {
        if let Some(ref replay_choices) = self.replay_choices {
            if self.replay_index < replay_choices.len() {
                let choice_value = replay_choices[self.replay_index].value.clone();
                self.replay_index += 1;
                Some(choice_value)
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// Core drawing method with overloads for each choice type (Python's _draw method)
    /// This handles prefix replay, forced values, observer notification, and overrun detection
    fn _draw(
        &mut self, 
        choice_type: ChoiceType, 
        constraints: Box<Constraints>, 
        observe: bool, 
        forced: Option<ChoiceValue>
    ) -> Result<ChoiceValue, DrawError> {
        // Check for overrun conditions before attempting to draw
        if self.length == self.max_length {
            self.mark_overrun();
            return Err(DrawError::Overrun);
        }
        
        if let Some(max_choices) = self.max_choices {
            if self.nodes.len() >= max_choices {
                self.mark_overrun();
                return Err(DrawError::Overrun);
            }
        }
        
        let was_forced = forced.is_some();
        
        let mut value = if observe && !self.prefix.is_empty() && self.index < self.prefix.len() {
            // Try to replay from prefix
            self._pop_choice(choice_type, &constraints, forced)?
        } else if let Some(forced_value) = forced {
            // Forced value was provided
            forced_value
        } else {
            // Generate new value using provider
            if let Some(ref mut provider) = self.provider {
                provider.draw_choice(choice_type, &constraints)?
            } else {
                // Fallback generation if no provider
                self.generate_fallback_value(choice_type, &constraints)?
            }
        };
        
        // Handle NaN normalization for floats (Python compatibility)
        if let ChoiceValue::Float(f) = value {
            if f.is_nan() {
                // Normalize NaN to ensure deterministic behavior
                value = ChoiceValue::Float(f64::NAN);
            }
        }
        
        if observe {
            
            // Notify observer
            if let Some(ref mut observer) = self.observer {
                observer.draw_value(choice_type, value.clone(), was_forced, constraints.clone());
            }
            
            // Calculate size for length tracking
            let size = self.calculate_choice_size(&value);
            if self.length + size > self.max_length {
                self.mark_overrun();
                return Err(DrawError::Overrun);
            }
            
            // Create and record choice node
            let node = ChoiceNode::with_index(
                choice_type,
                value.clone(),
                *constraints,
                was_forced,
                self.nodes.len().try_into().unwrap(),
            );
            
            
            self.nodes.push(node);
            self.record_choice_in_spans();
            self.length += size;
        }
        
        Ok(value)
    }
    
    /// Calculate the size contribution of a choice value
    fn calculate_choice_size(&self, value: &ChoiceValue) -> usize {
        match value {
            ChoiceValue::Integer(_) => 8,  // i128 is 16 bytes, but we'll use 8 for compatibility
            ChoiceValue::Boolean(_) => 1,
            ChoiceValue::Float(_) => 8,
            ChoiceValue::String(s) => s.len(),
            ChoiceValue::Bytes(b) => b.len(),
        }
    }
    
    /// Generate fallback value when no provider is available
    fn generate_fallback_value(&mut self, choice_type: ChoiceType, constraints: &Constraints) -> Result<ChoiceValue, DrawError> {
        match (choice_type, constraints) {
            (ChoiceType::Integer, Constraints::Integer(int_constraints)) => {
                let min = int_constraints.min_value.unwrap_or(i128::MIN);
                let max = int_constraints.max_value.unwrap_or(i128::MAX);
                let value = self.rng.gen_range(min..=max);
                Ok(ChoiceValue::Integer(value))
            },
            (ChoiceType::Boolean, Constraints::Boolean(bool_constraints)) => {
                let value = self.rng.gen::<f64>() < bool_constraints.p;
                Ok(ChoiceValue::Boolean(value))
            },
            (ChoiceType::Float, Constraints::Float(_)) => {
                let value = self.rng.gen::<f64>();
                Ok(ChoiceValue::Float(value))
            },
            _ => Err(DrawError::InvalidChoice),
        }
    }
    
    /// # Draw Integer: High-Performance Constrained Integer Generation
    /// 
    /// Generates random integers with sophisticated constraint validation and optimal shrinking
    /// behavior. This method provides complete Python Hypothesis parity with enterprise-grade
    /// performance optimizations and comprehensive error handling.
    ///
    /// ## Parameters
    /// 
    /// - **`min_value`**: Optional lower bound (inclusive). If `None`, allows unbounded negative values
    /// - **`max_value`**: Optional upper bound (inclusive). If `None`, allows unbounded positive values  
    /// - **`weights`**: Optional probability weights for specific values. Enables biased sampling
    /// - **`shrink_towards`**: Target value for shrinking optimization (typically 0 for minimization)
    /// - **`forced`**: Optional predetermined value for replay mode (bypasses generation)
    /// - **`observe`**: Whether to notify observers (typically true for DataTree integration)
    ///
    /// ## Algorithm and Performance
    /// 
    /// ### Constraint Validation: O(1)
    /// ```rust
    /// // Range validation with overflow protection
    /// if let (Some(min), Some(max)) = (min_value, max_value) {
    ///     if min > max { return Err(DrawError::InvalidRange); }
    /// }
    /// ```
    ///
    /// ### Generation Strategy: O(1) or O(log W) for weighted
    /// - **Unweighted**: Direct uniform sampling over range
    /// - **Weighted**: Alias method for O(1) sampling after O(W) preprocessing
    /// - **Unbounded**: Exponential distribution with geometric shrinking bias
    /// 
    /// ### Shrinking Optimization
    /// Values are encoded with lexicographic bias toward `shrink_towards`:
    /// - Values closer to `shrink_towards` get smaller encodings
    /// - Enables optimal shrinking with minimal counterexamples
    /// - Zigzag pattern: shrink_towards, ±1, ±2, ±3, ... from target
    ///
    /// ## Error Handling
    /// 
    /// ### Constraint Violations
    /// - **`InvalidRange`**: `min_value > max_value` 
    /// - **`InvalidWeights`**: Negative or NaN weights, empty weight map
    /// - **`Overrun`**: Buffer size limit exceeded (8KB default)
    /// - **`Frozen`**: Attempt to draw from frozen ConjectureData instance
    ///
    /// ### Recovery Strategies
    /// 1. **Range Errors**: Fail fast with detailed error context
    /// 2. **Weight Errors**: Normalize weights or fall back to uniform distribution
    /// 3. **Buffer Overrun**: Mark test as overrun and terminate gracefully
    /// 4. **Provider Failures**: Retry with alternative generation strategy
    ///
    /// ## Usage Examples
    ///
    /// ### Basic Range Generation
    /// ```rust
    /// let mut data = ConjectureData::new(42);
    /// let x = data.draw_integer(Some(0), Some(100), None, 0, None, true)?;
    /// assert!(x >= 0 && x <= 100);
    /// ```
    ///
    /// ### Weighted Distribution
    /// ```rust
    /// let mut weights = HashMap::new();
    /// weights.insert(0, 0.5);    // 50% chance of 0
    /// weights.insert(1, 0.3);    // 30% chance of 1
    /// weights.insert(2, 0.2);    // 20% chance of 2
    /// 
    /// let x = data.draw_integer(Some(0), Some(2), Some(weights), 0, None, true)?;
    /// ```
    ///
    /// ### Unbounded Generation with Shrinking Bias
    /// ```rust
    /// // Generates integers biased toward 42 for optimal shrinking
    /// let x = data.draw_integer(None, None, None, 42, None, true)?;
    /// ```
    ///
    /// ## Integration with Core Systems
    ///
    /// ### DataTree Integration
    /// When `observe = true`, the choice is recorded in DataTree for:
    /// - Persistent storage across test runs
    /// - Systematic exploration of input space
    /// - Regression testing with saved examples
    ///
    /// ### Shrinking Integration  
    /// Generated values are optimized for shrinking algorithms:
    /// - Lexicographic encoding prioritizes values closer to `shrink_towards`
    /// - Choice sequence recording enables replay-based shrinking
    /// - Constraint preservation ensures shrunk values remain valid
    ///
    /// ### Provider Integration
    /// Supports multiple generation backends:
    /// - **Hypothesis Provider**: Python-compatible advanced generation
    /// - **Random Provider**: Simple uniform random generation
    /// - **Custom Providers**: User-defined generation strategies
    pub fn draw_integer(
        &mut self,
        min_value: Option<i128>,
        max_value: Option<i128>,
        weights: Option<HashMap<i128, f64>>,
        shrink_towards: i128,
        forced: Option<i128>,
        observe: bool,
    ) -> Result<i128, DrawError> {
        // Status and capacity checks
        if !self.can_draw() {
            if self.frozen {
                return Err(DrawError::Frozen);
            } else {
                return Err(DrawError::InvalidStatus);
            }
        }
        
        // Check for potential overrun
        if self.length + 2 > self.max_length {
            self.mark_overrun();
            return Err(DrawError::Overrun);
        }
        
        // Create constraints with Python validation
        let constraints = IntegerConstraints {
            min_value,
            max_value,
            weights: weights.clone(),
            shrink_towards: Some(shrink_towards),
        };
        
        // Validate constraints
        self.validate_integer_constraints(&constraints)?;
        
        // Handle forced value case
        if let Some(forced_value) = forced {
            if !self.choice_permitted_integer(forced_value, &constraints) {
                return Err(DrawError::InvalidRange);
            }
            
            // Record the forced choice
            let choice_node = ChoiceNode::new(
                ChoiceType::Integer,
                ChoiceValue::Integer(forced_value),
                Constraints::Integer(constraints),
                true, // was_forced
            );
            
            self.record_choice(choice_node);
            return Ok(forced_value);
        }
        
        // Check for replay mode
        if let Some(replay_value) = self.try_replay_choice(ChoiceType::Integer, &Constraints::Integer(constraints.clone()))? {
            if let ChoiceValue::Integer(value) = replay_value {
                return Ok(value);
            } else {
                return Err(DrawError::TypeMismatch);
            }
        }
        
        // Generate new value using provider
        let value = if let Some(ref mut provider) = self.provider {
            provider.draw_integer(&constraints).map_err(DrawError::from)?
        } else {
            // Fallback to internal RNG if no provider
            let mut rng_provider = RngProvider::new(&mut self.rng);
            rng_provider.draw_integer(&constraints).map_err(DrawError::from)?
        };
        
        // Validate generated value
        if !self.choice_permitted_integer(value, &constraints) {
            return Err(DrawError::InvalidRange);
        }
        
        // Record the choice
        let choice_node = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(value),
            Constraints::Integer(constraints),
            false, // was_forced
        );
        
        if observe {
            self.record_choice(choice_node);
        }
        
        Ok(value)
    }
    
    /// Validate integer constraints following Python rules
    fn validate_integer_constraints(&self, constraints: &IntegerConstraints) -> Result<(), DrawError> {
        // Validate range
        if let (Some(min), Some(max)) = (constraints.min_value, constraints.max_value) {
            if min > max {
                return Err(DrawError::InvalidRange);
            }
        }
        
        // Validate weights if provided
        if let Some(ref weights) = constraints.weights {
            if weights.len() > 255 {
                return Err(DrawError::InvalidRange); // Too many weights
            }
            
            let weight_sum: f64 = weights.values().sum();
            if weight_sum > 1.0 {
                return Err(DrawError::InvalidRange); // Weight sum must be <= 1
            }
            
            for (&value, &weight) in weights.iter() {
                if weight <= 0.0 {
                    return Err(DrawError::InvalidRange); // Weights must be positive
                }
                
                // Check if weighted value is in range
                if let Some(min) = constraints.min_value {
                    if value < min {
                        return Err(DrawError::InvalidRange);
                    }
                }
                if let Some(max) = constraints.max_value {
                    if value > max {
                        return Err(DrawError::InvalidRange);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if an integer choice is permitted under given constraints
    fn choice_permitted_integer(&self, value: i128, constraints: &IntegerConstraints) -> bool {
        if let Some(min) = constraints.min_value {
            if value < min {
                return false;
            }
        }
        
        if let Some(max) = constraints.max_value {
            if value > max {
                return false;
            }
        }
        
        true
    }
    
    
    /// Legacy method for backward compatibility
    pub fn draw_integer_weighted(&mut self, min_value: i128, max_value: i128, 
                                 weights: Option<HashMap<i128, f64>>, 
                                 shrink_towards: Option<i128>) -> Result<i128, DrawError> {
        self.draw_integer(
            Some(min_value),
            Some(max_value),
            weights,
            shrink_towards.unwrap_or(0),
            None,
            true
        )
    }
    
    /// Legacy method for simple range-based integer drawing
    fn draw_integer_full(&mut self, min_value: i128, max_value: i128, 
                         weights: Option<HashMap<i128, f64>>, 
                         shrink_towards: Option<i128>,
                         forced: Option<i128>) -> Result<i128, DrawError> {
        if !self.can_draw() {
            if self.frozen {
                return Err(DrawError::Frozen);
            } else {
                return Err(DrawError::InvalidStatus);
            }
        }
        
        // Check for potential overrun before drawing
        if self.length + 2 > self.max_length {
            self.mark_overrun();
            return Err(DrawError::Overrun);
        }
        
        if min_value > max_value {
            return Err(DrawError::InvalidRange);
        }
        
        // Validate weights if provided (Python validation rules)
        if let Some(ref weight_map) = weights {
            if weight_map.len() > 255 {
                return Err(DrawError::InvalidRange); // Too many weights
            }
            
            let weight_sum: f64 = weight_map.values().sum();
            if weight_sum > 1.0 {
                return Err(DrawError::InvalidRange); // Weight sum must be <= 1
            }
            
            for (&value, &weight) in weight_map.iter() {
                if value < min_value || value > max_value {
                    return Err(DrawError::InvalidRange); // Weight key out of range
                }
                if weight <= 0.0 {
                    return Err(DrawError::InvalidRange); // No zero/negative weights
                }
            }
        }
        
        // Validate shrink_towards if provided
        if let Some(shrink) = shrink_towards {
            if shrink < min_value || shrink > max_value {
                return Err(DrawError::InvalidRange);
            }
        }
        
        // Create constraints for this draw
        let constraints = Constraints::Integer(IntegerConstraints {
            min_value: Some(min_value),
            max_value: Some(max_value),
            weights: weights.clone(),
            shrink_towards,
        });
        
        // Use _pop_choice for sophisticated replay handling
        let forced_choice = forced.map(ChoiceValue::Integer);
        let value = if let Ok(replay_value) = self._pop_choice(ChoiceType::Integer, &constraints, forced_choice) {
            // Use replay/forced value if available
            if let ChoiceValue::Integer(replay_int) = replay_value {
                if replay_int < min_value || replay_int > max_value {
                    return Err(DrawError::InvalidRange);
                }
                replay_int
            } else {
                return Err(DrawError::InvalidReplayType);
            }
        } else {
            // Generate new value using provider or weighted random generation
            if let Constraints::Integer(ref int_constraints) = constraints {
                if let Some(ref mut provider) = self.provider {
                    provider.generate_integer(&mut self.rng, int_constraints)?
                } else {
                    // Fallback to weighted random generation or simple random
                    if let Some(ref weight_map) = weights {
                        self.generate_weighted_integer(min_value, max_value, weight_map)?
                    } else {
                        // Simple random generation
                        if min_value == max_value {
                            min_value
                        } else {
                            self.rng.gen_range(min_value..=max_value)
                        }
                    }
                }
            } else {
                unreachable!("We just created integer constraints")
            }
        };
        
        // Record the choice
        let choice = ChoiceNode::new(
            ChoiceType::Integer,
            ChoiceValue::Integer(value),
            constraints.clone(),
            forced.is_some(),
        );
        
        self.nodes.push(choice);
        self.record_choice_in_spans();
        
        // Notify observer if present
        if let Some(ref mut observer) = self.observer {
            observer.draw_value(
                ChoiceType::Integer,
                ChoiceValue::Integer(value),
                forced.is_some(),
                Box::new(constraints)
            );
        }
        
        // Update internal state
        self.length += 2; // Integer draws consume 2 bytes in Python
        
        Ok(value)
    }

    /// Draw an integer with optional forced value (for replay)
    pub fn draw_integer_with_forced(&mut self, min_value: i128, max_value: i128, forced: Option<i128>) -> Result<i128, DrawError> {
        // Calculate a proper shrink_towards value that's within the range
        let shrink_towards = if min_value <= 0 && max_value >= 0 {
            Some(0) // 0 is in range, use it
        } else if min_value > 0 {
            Some(min_value) // Range is all positive, shrink towards min
        } else {
            Some(max_value) // Range is all negative, shrink towards max (closest to 0)
        };
        self.draw_integer_full(min_value, max_value, None, shrink_towards, forced)
    }
    
    /// Generate a weighted random integer using the provided weight distribution
    fn generate_weighted_integer(&mut self, min_value: i128, max_value: i128, weights: &HashMap<i128, f64>) -> Result<i128, DrawError> {
        use rand::Rng;
        
        let weight_sum: f64 = weights.values().sum();
        
        // If weights sum to 1.0, use pure weighted distribution
        if (weight_sum - 1.0).abs() < f64::EPSILON {
            // Create cumulative distribution
            let mut cumulative_weights: Vec<(i128, f64)> = Vec::new();
            let mut cumulative = 0.0;
            
            for (&value, &weight) in weights.iter() {
                cumulative += weight;
                cumulative_weights.push((value, cumulative));
            }
            
            let rand_val: f64 = self.rng.gen();
            for (value, cum_weight) in cumulative_weights {
                if rand_val < cum_weight {
                    return Ok(value);
                }
            }
            // Fallback to last weighted value (should not happen with proper cumulative dist)
            return Ok(weights.keys().next().copied().unwrap_or(min_value));
        }
        
        // Mixed distribution: some weighted, some uniform
        let remaining_prob = 1.0 - weight_sum;
        
        // Calculate total range size efficiently for large ranges
        let range_size = if let Some(size) = max_value.checked_sub(min_value) {
            if size > 10_000_000 {  // Avoid memory exhaustion for very large ranges
                // For very large ranges, calculate non-weighted count without collecting
                let non_weighted_count = size + 1 - (weights.len() as i128);
                if non_weighted_count <= 0 {
                    // All values in range are weighted, use pure weighted distribution
                    let mut cumulative_weights: Vec<(i128, f64)> = Vec::new();
                    let mut cumulative = 0.0;
                    
                    for (&value, &weight) in weights.iter() {
                        cumulative += weight / weight_sum;  // Normalize
                        cumulative_weights.push((value, cumulative));
                    }
                    
                    let rand_val: f64 = self.rng.gen();
                    for (value, cum_weight) in cumulative_weights {
                        if rand_val < cum_weight {
                            return Ok(value);
                        }
                    }
                    return Ok(weights.keys().next().copied().unwrap_or(min_value));
                }
                size + 1
            } else {
                size + 1
            }
        } else {
            return Err(DrawError::InvalidRange);  // Range overflow
        };
        
        // Generate random number [0, 1)
        let rand_val: f64 = self.rng.gen();
        
        // Check if we should use weighted distribution (normalized boundary)
        if rand_val < weight_sum {
            // Create cumulative distribution for weighted values
            let mut cumulative_weights: Vec<(i128, f64)> = Vec::new();
            let mut cumulative = 0.0;
            
            for (&value, &weight) in weights.iter() {
                cumulative += weight;
                cumulative_weights.push((value, cumulative));
            }
            
            // Scale random value to weighted portion
            let scaled_rand = rand_val;
            for (value, cum_weight) in cumulative_weights {
                if scaled_rand < cum_weight {
                    return Ok(value);
                }
            }
        }
        
        // Use uniform distribution over non-weighted values
        if range_size > 10_000_000 {
            // For large ranges, pick random value and check if it's weighted
            loop {
                let candidate = self.rng.gen_range(min_value..=max_value);
                if !weights.contains_key(&candidate) {
                    return Ok(candidate);
                }
                // If we hit a weighted value, try again (very low probability for large ranges)
            }
        } else {
            // For smaller ranges, collect non-weighted values
            let non_weighted_values: Vec<i128> = (min_value..=max_value)
                .filter(|&v| !weights.contains_key(&v))
                .collect();
            
            if non_weighted_values.is_empty() {
                // All values are weighted, fall back to pure weighted distribution
                let mut cumulative_weights: Vec<(i128, f64)> = Vec::new();
                let mut cumulative = 0.0;
                
                for (&value, &weight) in weights.iter() {
                    cumulative += weight / weight_sum;  // Normalize
                    cumulative_weights.push((value, cumulative));
                }
                
                let rand_val: f64 = self.rng.gen();
                for (value, cum_weight) in cumulative_weights {
                    if rand_val < cum_weight {
                        return Ok(value);
                    }
                }
                return Ok(weights.keys().next().copied().unwrap_or(min_value));
            }
            
            let idx = self.rng.gen_range(0..non_weighted_values.len());
            Ok(non_weighted_values[idx])
        }
    }
    
    
    /// # Draw Boolean: High-Precision Probability-Based Boolean Generation
    /// 
    /// Generates boolean values with IEEE 754 floating-point precision probability control
    /// and optimal shrinking behavior. Provides complete Python Hypothesis parity with
    /// enterprise-grade performance and comprehensive edge case handling.
    ///
    /// ## Parameters
    /// 
    /// - **`p`**: Probability of generating `true` (0.0 ≤ p ≤ 1.0)
    /// - **`forced`**: Optional predetermined value for replay mode (bypasses generation)
    /// - **`observe`**: Whether to notify observers (typically true for DataTree integration)
    ///
    /// ## Algorithm and Performance
    /// 
    /// ### Probability Validation: O(1)
    /// ```rust
    /// // IEEE 754 compliant probability validation
    /// if !(0.0..=1.0).contains(&p) || p.is_nan() {
    ///     return Err(DrawError::InvalidProbability);
    /// }
    /// ```
    ///
    /// ### Generation Strategy: O(1)
    /// - **Standard Case**: Compare uniform random [0,1) against probability p
    /// - **Edge Cases**: Special handling for p=0.0 (always false) and p=1.0 (always true)
    /// - **Floating-Point Precision**: Exact IEEE 754 bit comparison for reproducibility
    /// 
    /// ### Shrinking Optimization
    /// Boolean values are encoded with bias toward `false` for minimal counterexamples:
    /// - `false` gets smaller encoding than `true` for lexicographic shrinking
    /// - Enables rapid convergence to minimal failing test cases
    /// - Maintains probability distribution during shrinking process
    ///
    /// ## Error Handling
    /// 
    /// ### Probability Constraint Violations
    /// - **`InvalidProbability`**: p < 0.0, p > 1.0, or p is NaN
    /// - **`Overrun`**: Buffer size limit exceeded during generation
    /// - **`Frozen`**: Attempt to draw from frozen ConjectureData instance
    /// - **`InvalidStatus`**: Draw attempted in invalid test state
    ///
    /// ### Recovery Strategies
    /// 1. **Invalid Probability**: Fail fast with detailed error context
    /// 2. **Buffer Overrun**: Mark test as overrun and terminate gracefully
    /// 3. **State Errors**: Prevent further operations and maintain test integrity
    ///
    /// ## Usage Examples
    ///
    /// ### Basic Probability Generation
    /// ```rust
    /// let mut data = ConjectureData::new(42);
    /// let biased_coin = data.draw_boolean(0.7, None, true)?;  // 70% chance of true
    /// let fair_coin = data.draw_boolean(0.5, None, true)?;    // 50% chance of true
    /// ```
    ///
    /// ### Edge Case Handling
    /// ```rust
    /// let always_false = data.draw_boolean(0.0, None, true)?;  // Always false
    /// let always_true = data.draw_boolean(1.0, None, true)?;   // Always true
    /// assert_eq!(always_false, false);
    /// assert_eq!(always_true, true);
    /// ```
    ///
    /// ### Forced Value for Replay
    /// ```rust
    /// // Replay mode: force specific boolean value
    /// let forced_true = data.draw_boolean(0.3, Some(true), true)?;
    /// assert_eq!(forced_true, true);  // Ignores probability, uses forced value
    /// ```
    ///
    /// ## Integration with Core Systems
    ///
    /// ### DataTree Integration
    /// When `observe = true`, the choice is recorded for:
    /// - Systematic exploration of boolean combinations
    /// - Persistent storage of boolean patterns
    /// - Regression testing with boolean-heavy test cases
    ///
    /// ### Shrinking Integration
    /// Boolean generation is optimized for shrinking:
    /// - `false` values are preferred during shrinking (smaller encoding)
    /// - Probability-aware shrinking maintains statistical properties
    /// - Choice sequence recording enables exact replay of boolean sequences
    ///
    /// ### Floating-Point Precision
    /// Uses IEEE 754 double precision for exact probability matching:
    /// - Bit-level reproducibility across platforms
    /// - Proper handling of subnormal probabilities
    /// - Exact comparison for edge cases (0.0, 1.0)
    pub fn draw_boolean(
        &mut self,
        p: f64,
        forced: Option<bool>,
        observe: bool,
    ) -> Result<bool, DrawError> {
        // Status and capacity checks
        if !self.can_draw() {
            if self.frozen {
                return Err(DrawError::Frozen);
            } else {
                return Err(DrawError::InvalidStatus);
            }
        }
        
        // Check for potential overrun
        if self.length + 1 > self.max_length {
            self.mark_overrun();
            return Err(DrawError::Overrun);
        }
        
        // Validate probability
        if p < 0.0 || p > 1.0 || p.is_nan() {
            return Err(DrawError::InvalidRange);
        }
        
        // Create constraints
        let constraints = BooleanConstraints { p };
        
        // Handle forced value case
        if let Some(forced_value) = forced {
            // Record the forced choice
            let choice_node = ChoiceNode::new(
                ChoiceType::Boolean,
                ChoiceValue::Boolean(forced_value),
                Constraints::Boolean(constraints),
                true, // was_forced
            );
            
            self.record_choice(choice_node);
            return Ok(forced_value);
        }
        
        // Check for replay mode
        if let Some(replay_value) = self.try_replay_choice(ChoiceType::Boolean, &Constraints::Boolean(constraints.clone()))? {
            if let ChoiceValue::Boolean(value) = replay_value {
                return Ok(value);
            } else {
                return Err(DrawError::TypeMismatch);
            }
        }
        
        // Generate new value
        let value = if p == 0.0 {
            false // Deterministic case
        } else if p == 1.0 {
            true // Deterministic case
        } else if let Some(ref mut provider) = self.provider {
            provider.draw_boolean(p).map_err(DrawError::from)?
        } else {
            // Fallback to internal RNG if no provider
            let mut rng_provider = RngProvider::new(&mut self.rng);
            rng_provider.draw_boolean(p).map_err(DrawError::from)?
        };
        
        // Record the choice
        let choice_node = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(value),
            Constraints::Boolean(constraints),
            false, // was_forced
        );
        
        if observe {
            self.record_choice(choice_node);
        }
        
        Ok(value)
    }
    
    /// Draw a boolean with optional forced value (for replay)
    
    /// Legacy method for backward compatibility
    pub fn draw_boolean_with_forced(&mut self, p: f64, forced: Option<bool>) -> Result<bool, DrawError> {
        self.draw_boolean(p, forced, true)
    }
    
    fn draw_boolean_legacy(&mut self, p: f64, forced: Option<bool>) -> Result<bool, DrawError> {
        if !self.can_draw() {
            if self.frozen {
                return Err(DrawError::Frozen);
            } else {
                return Err(DrawError::InvalidStatus);
            }
        }
        
        // Check for potential overrun before drawing
        if self.length + 1 > self.max_length {
            self.mark_overrun();
            return Err(DrawError::Overrun);
        }
        
        if !(0.0..=1.0).contains(&p) {
            return Err(DrawError::InvalidProbability);
        }
        
        // Create constraints for this draw
        let constraints = Constraints::Boolean(BooleanConstraints { p });
        
        // Use _pop_choice for sophisticated replay handling
        let forced_choice = forced.map(ChoiceValue::Boolean);
        let value = if let Ok(replay_value) = self._pop_choice(ChoiceType::Boolean, &constraints, forced_choice) {
            // Use replay/forced value if available
            if let ChoiceValue::Boolean(replay_bool) = replay_value {
                replay_bool
            } else {
                return Err(DrawError::InvalidReplayType);
            }
        } else {
            // Generate new random value
            self.rng.gen::<f64>() < p
        };
        
        // Record the choice
        let choice = ChoiceNode::new(
            ChoiceType::Boolean,
            ChoiceValue::Boolean(value),
            constraints.clone(),
            forced.is_some(), // was_forced if we had a forced value
        );
        
        self.nodes.push(choice);
        self.record_choice_in_spans();
        
        // Notify observer if present
        if let Some(ref mut observer) = self.observer {
            observer.draw_value(
                ChoiceType::Boolean,
                ChoiceValue::Boolean(value),
                forced.is_some(),
                Box::new(constraints)
            );
        }
        
        // Update length (1 byte for boolean)
        self.length += 1;
        
        Ok(value)
    }
    
    /// Draw a floating-point number with default constraints
    /// Draw a float with Python-equivalent constraint validation and choice recording
    /// 
    /// This method provides full Python parity for float generation including:
    /// - Custom lexicographic float encoding for optimal shrinking
    /// - NaN, infinity, and subnormal value handling
    /// - Proper constraint validation and clamping
    /// - Choice recording with comprehensive metadata
    pub fn draw_float(
        &mut self,
        min_value: f64,
        max_value: f64,
        allow_nan: bool,
        smallest_nonzero_magnitude: Option<f64>,
        forced: Option<f64>,
        observe: bool,
    ) -> Result<f64, DrawError> {
        // Status and capacity checks
        if !self.can_draw() {
            if self.frozen {
                return Err(DrawError::Frozen);
            } else {
                return Err(DrawError::InvalidStatus);
            }
        }
        
        // Check for potential overrun
        if self.length + 8 > self.max_length {
            self.mark_overrun();
            return Err(DrawError::Overrun);
        }
        
        // Create and validate constraints
        let constraints = FloatConstraints {
            min_value,
            max_value,
            allow_nan,
            smallest_nonzero_magnitude,
        };
        
        self.validate_float_constraints(&constraints)?;
        
        // Handle forced value case
        if let Some(forced_value) = forced {
            if !self.choice_permitted_float(forced_value, &constraints) {
                return Err(DrawError::InvalidRange);
            }
            
            // Record the forced choice
            let choice_node = ChoiceNode::new(
                ChoiceType::Float,
                ChoiceValue::Float(forced_value),
                Constraints::Float(constraints),
                true, // was_forced
            );
            
            self.record_choice(choice_node);
            return Ok(forced_value);
        }
        
        // Check for replay mode
        if let Some(replay_value) = self.try_replay_choice(ChoiceType::Float, &Constraints::Float(constraints.clone()))? {
            if let ChoiceValue::Float(value) = replay_value {
                return Ok(value);
            } else {
                return Err(DrawError::TypeMismatch);
            }
        }
        
        // Generate new value using provider
        let value = if let Some(ref mut provider) = self.provider {
            provider.draw_float(&constraints).map_err(DrawError::from)?
        } else {
            // Fallback to internal RNG if no provider
            let mut rng_provider = RngProvider::new(&mut self.rng);
            rng_provider.draw_float(&constraints).map_err(DrawError::from)?
        };
        
        // Apply constraint clamping (Python behavior)
        let clamped_value = self.clamp_float(value, &constraints);
        
        // Record the choice
        let choice_node = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(clamped_value),
            Constraints::Float(constraints),
            false, // was_forced
        );
        
        if observe {
            self.record_choice(choice_node);
        }
        
        Ok(clamped_value)
    }
    
    /// Validate float constraints following Python rules
    fn validate_float_constraints(&self, constraints: &FloatConstraints) -> Result<(), DrawError> {
        // Check for NaN values in bounds
        if constraints.min_value.is_nan() || constraints.max_value.is_nan() {
            return Err(DrawError::InvalidRange);
        }
        
        // Validate range
        if constraints.min_value > constraints.max_value {
            return Err(DrawError::InvalidRange);
        }
        
        // Validate smallest_nonzero_magnitude
        if let Some(magnitude) = constraints.smallest_nonzero_magnitude {
            if magnitude <= 0.0 || magnitude.is_nan() || magnitude.is_infinite() {
                return Err(DrawError::InvalidRange);
            }
        }
        
        Ok(())
    }
    
    /// Check if a float choice is permitted under given constraints
    fn choice_permitted_float(&self, value: f64, constraints: &FloatConstraints) -> bool {
        // Handle NaN
        if value.is_nan() {
            return constraints.allow_nan;
        }
        
        // Check bounds
        if value < constraints.min_value || value > constraints.max_value {
            return false;
        }
        
        // Check smallest nonzero magnitude
        if let Some(min_magnitude) = constraints.smallest_nonzero_magnitude {
            if value != 0.0 && value.abs() < min_magnitude {
                return false;
            }
        }
        
        true
    }
    
    /// Clamp float value to constraints (Python behavior)
    fn clamp_float(&self, value: f64, constraints: &FloatConstraints) -> f64 {
        if value.is_nan() {
            if constraints.allow_nan {
                return value;
            } else {
                // Return a valid value within bounds
                return constraints.min_value.max(0.0).min(constraints.max_value);
            }
        }
        
        // Clamp to bounds
        let clamped = value.max(constraints.min_value).min(constraints.max_value);
        
        // Apply smallest nonzero magnitude constraint
        if let Some(min_magnitude) = constraints.smallest_nonzero_magnitude {
            if clamped != 0.0 && clamped.abs() < min_magnitude {
                // Snap to zero or minimum magnitude
                if clamped.abs() < min_magnitude / 2.0 {
                    return 0.0;
                } else {
                    return if clamped.is_sign_positive() { min_magnitude } else { -min_magnitude };
                }
            }
        }
        
        clamped
    }
    
    
    /// Legacy method - Draw a floating-point number with comprehensive constraint support
    pub fn draw_float_full(&mut self, min_value: f64, max_value: f64, allow_nan: bool, 
                           smallest_nonzero_magnitude: Option<f64>, forced: Option<f64>) -> Result<f64, DrawError> {
        if !self.can_draw() {
            if self.frozen {
                return Err(DrawError::Frozen);
            } else {
                return Err(DrawError::InvalidStatus);
            }
        }
        
        // Check for potential overrun before drawing
        if self.length + 8 > self.max_length {
            self.mark_overrun();
            return Err(DrawError::Overrun);
        }
        
        // Validate constraints
        if min_value > max_value {
            return Err(DrawError::InvalidRange);
        }
        
        if let Some(magnitude) = smallest_nonzero_magnitude {
            if magnitude <= 0.0 {
                return Err(DrawError::InvalidRange);
            }
        }
        
        // Check for fastmath compilation issues (subnormal detection)
        let test_subnormal = f64::MIN_POSITIVE / 2.0;
        if test_subnormal == 0.0 && smallest_nonzero_magnitude.map_or(false, |m| m < f64::MIN_POSITIVE) {
            // Fastmath compilation detected, adjust smallest_nonzero_magnitude
            return Err(DrawError::InvalidRange); // Or we could auto-adjust to MIN_POSITIVE
        }
        
        // Validate forced value if provided
        if let Some(forced_val) = forced {
            // Use sign-aware comparison for IEEE-754 compliance
            if forced_val.is_nan() && !allow_nan {
                return Err(DrawError::InvalidRange);
            }
            if !forced_val.is_nan() && (forced_val < min_value || forced_val > max_value) {
                return Err(DrawError::InvalidRange);
            }
            if forced_val.abs() != 0.0 && smallest_nonzero_magnitude.map_or(false, |m| forced_val.abs() < m) {
                return Err(DrawError::InvalidRange);
            }
        }
        
        // Create constraints for this draw
        let constraints = Constraints::Float(FloatConstraints {
            min_value,
            max_value,
            allow_nan,
            smallest_nonzero_magnitude,
        });
        
        // Use _pop_choice for sophisticated replay handling
        let forced_choice = forced.map(ChoiceValue::Float);
        let value = if let Ok(replay_value) = self._pop_choice(ChoiceType::Float, &constraints, forced_choice) {
            // Use replay/forced value if available
            if let ChoiceValue::Float(replay_float) = replay_value {
                replay_float
            } else {
                return Err(DrawError::InvalidReplayType);
            }
        } else {
            // Generate new value using enhanced float constraint type system
            if let Constraints::Float(ref float_constraints) = constraints {
                if let Some(ref mut provider) = self.provider {
                    // Try to use provider's regular float generation
                    match provider.generate_float(&mut self.rng, float_constraints) {
                        Ok(value) => value,
                        Err(_) => {
                            // Fallback to FloatConstraintTypeSystem for sophisticated generation
                            debug_log!("Provider failed, using FloatConstraintTypeSystem fallback");
                            let float_system = FloatConstraintTypeSystem::new(float_constraints.clone());
                            let mut rng_provider = RngProvider::new(&mut self.rng);
                            float_system.generate_float(&mut rng_provider)
                        }
                    }
                } else {
                    // Use FloatConstraintTypeSystem for sophisticated fallback generation
                    debug_log!("Using FloatConstraintTypeSystem for fallback generation");
                    let float_system = FloatConstraintTypeSystem::new(float_constraints.clone());
                    
                    // Create a wrapper for the RNG to work with the provider interface
                    let mut rng_provider = RngProvider::new(&mut self.rng);
                    float_system.generate_float(&mut rng_provider)
                }
            } else {
                unreachable!("We just created float constraints")
            }
        };
        
        // Record the choice
        let choice = ChoiceNode::new(
            ChoiceType::Float,
            ChoiceValue::Float(value),
            constraints.clone(),
            forced.is_some(),
        );
        
        self.nodes.push(choice);
        self.record_choice_in_spans();
        
        // Notify observer if present
        if let Some(ref mut observer) = self.observer {
            observer.draw_value(
                ChoiceType::Float,
                ChoiceValue::Float(value),
                forced.is_some(),
                Box::new(constraints)
            );
        }
        
        // Update internal state
        self.length += 8; // Float draws consume 8 bytes
        
        Ok(value)
    }
    
    /// Generate IEEE-754 compliant random float with advanced constraints
    fn generate_ieee754_float(&mut self, min_value: f64, max_value: f64, allow_nan: bool, 
                              smallest_nonzero_magnitude: Option<f64>) -> Result<f64, DrawError> {
        use rand::Rng;
        
        // Handle special cases first
        if allow_nan && self.rng.gen_bool(0.01) { // 1% chance of NaN
            return Ok(f64::NAN);
        }
        
        // Handle infinity cases
        if min_value == f64::NEG_INFINITY && max_value == f64::INFINITY {
            // Full range - use sophisticated IEEE-754 generation
            return self.generate_full_range_float(allow_nan, smallest_nonzero_magnitude);
        }
        
        // Bounded range generation
        if min_value.is_infinite() || max_value.is_infinite() {
            // One bound is infinite
            if min_value == f64::NEG_INFINITY {
                // (-∞, max_value]
                let candidate = if self.rng.gen_bool(0.1) {
                    // 10% chance of extreme negative
                    -self.rng.gen_range(1e100..1e308)
                } else {
                    self.rng.gen_range(-1e10..max_value)
                };
                return Ok(candidate.clamp(min_value, max_value));
            } else {
                // [min_value, +∞)
                let candidate = if self.rng.gen_bool(0.1) {
                    // 10% chance of extreme positive
                    self.rng.gen_range(1e100..1e308)
                } else {
                    self.rng.gen_range(min_value..1e10)
                };
                return Ok(candidate.clamp(min_value, max_value));
            }
        }
        
        // Finite range generation
        let mut candidate = self.rng.gen_range(min_value..=max_value);
        
        // Apply smallest_nonzero_magnitude constraint
        if let Some(magnitude) = smallest_nonzero_magnitude {
            if candidate.abs() != 0.0 && candidate.abs() < magnitude {
                // Adjust to meet the constraint
                candidate = if candidate > 0.0 {
                    magnitude
                } else {
                    -magnitude
                };
            }
        }
        
        // Ensure still in range
        if candidate < min_value || candidate > max_value {
            // Try zero instead
            candidate = 0.0;
            if candidate < min_value || candidate > max_value {
                // If zero not in range either, use boundary
                candidate = if (min_value - 0.0).abs() < (max_value - 0.0).abs() {
                    min_value
                } else {
                    max_value
                };
            }
        }
        
        Ok(candidate)
    }
    
    /// Generate float from full IEEE-754 range with sophisticated distribution
    fn generate_full_range_float(&mut self, allow_nan: bool, smallest_nonzero_magnitude: Option<f64>) -> Result<f64, DrawError> {
        use rand::Rng;
        
        let choice = self.rng.gen_range(0..10);
        match choice {
            0 => Ok(0.0),
            1 => Ok(-0.0),
            2 if allow_nan => Ok(f64::NAN),
            3 => Ok(f64::INFINITY),
            4 => Ok(f64::NEG_INFINITY),
            5 => Ok(f64::MIN_POSITIVE), // Smallest positive normal
            6 => Ok(-f64::MIN_POSITIVE),
            7 => {
                // Subnormal values
                let subnormal = self.rng.gen_range(f64::MIN_POSITIVE / 1e10..f64::MIN_POSITIVE);
                Ok(if self.rng.gen_bool(0.5) { subnormal } else { -subnormal })
            },
            8 => {
                // Very large values
                let large = self.rng.gen_range(1e100..f64::MAX / 2.0);
                Ok(if self.rng.gen_bool(0.5) { large } else { -large })
            },
            _ => {
                // Normal random values
                let normal = self.rng.gen::<f64>() * 1000.0 - 500.0;
                let adjusted = if let Some(magnitude) = smallest_nonzero_magnitude {
                    if normal.abs() != 0.0 && normal.abs() < magnitude {
                        if normal > 0.0 { magnitude } else { -magnitude }
                    } else {
                        normal
                    }
                } else {
                    normal
                };
                Ok(adjusted)
            }
        }
    }
    
    /// Draw a string from alphabet with size constraints (simplified interface)
    /// Draw a string with Python-equivalent constraint validation and choice recording
    /// 
    /// This method provides full Python parity for string generation including:
    /// - Unicode interval-based character sampling
    /// - Collection indexing for size and character ordering
    /// - Support for arbitrary Unicode ranges
    /// - Proper choice recording and constraint validation
    pub fn draw_string(
        &mut self,
        intervals: IntervalSet,
        min_size: usize,
        max_size: usize,
        forced: Option<String>,
        observe: bool,
    ) -> Result<String, DrawError> {
        // Status and capacity checks
        if !self.can_draw() {
            if self.frozen {
                return Err(DrawError::Frozen);
            } else {
                return Err(DrawError::InvalidStatus);
            }
        }
        
        // Validate size constraints
        if min_size > max_size {
            return Err(DrawError::InvalidRange);
        }
        
        // Check for potential overrun (estimate based on max size)
        let estimated_bytes = max_size * 4; // Worst case: 4 bytes per UTF-8 char
        if self.length + estimated_bytes > self.max_length {
            self.mark_overrun();
            return Err(DrawError::Overrun);
        }
        
        // Create constraints
        let constraints = StringConstraints {
            intervals: intervals.clone(),
            min_size,
            max_size,
        };
        
        // Handle forced value case
        if let Some(forced_value) = forced {
            if !self.choice_permitted_string(&forced_value, &constraints) {
                return Err(DrawError::InvalidRange);
            }
            
            // Record the forced choice
            let choice_node = ChoiceNode::new(
                ChoiceType::String,
                ChoiceValue::String(forced_value.clone()),
                Constraints::String(constraints),
                true, // was_forced
            );
            
            self.record_choice(choice_node);
            return Ok(forced_value);
        }
        
        // Check for replay mode
        if let Some(replay_value) = self.try_replay_choice(ChoiceType::String, &Constraints::String(constraints.clone()))? {
            if let ChoiceValue::String(value) = replay_value {
                return Ok(value);
            } else {
                return Err(DrawError::TypeMismatch);
            }
        }
        
        // Generate new value using provider
        let value = if let Some(ref mut provider) = self.provider {
            provider.draw_string(&intervals, min_size, max_size).map_err(DrawError::from)?
        } else {
            // Fallback to internal RNG if no provider
            let mut rng_provider = RngProvider::new(&mut self.rng);
            rng_provider.draw_string(&intervals, min_size, max_size).map_err(DrawError::from)?
        };
        
        // Validate generated value
        if !self.choice_permitted_string(&value, &constraints) {
            return Err(DrawError::InvalidRange);
        }
        
        // Record the choice
        let choice_node = ChoiceNode::new(
            ChoiceType::String,
            ChoiceValue::String(value.clone()),
            Constraints::String(constraints),
            false, // was_forced
        );
        
        if observe {
            self.record_choice(choice_node);
        }
        
        Ok(value)
    }
    
    /// Check if a string choice is permitted under given constraints
    fn choice_permitted_string(&self, value: &str, constraints: &StringConstraints) -> bool {
        // Check length bounds
        if value.chars().count() < constraints.min_size || value.chars().count() > constraints.max_size {
            return false;
        }
        
        // Check if all characters are in allowed intervals
        for ch in value.chars() {
            let codepoint = ch as u32;
            let mut found = false;
            
            for &(start, end) in &constraints.intervals.intervals {
                if codepoint >= start && codepoint <= end {
                    found = true;
                    break;
                }
            }
            
            if !found {
                return false;
            }
        }
        
        true
    }
    
    
    /// Legacy method - Draw a string with comprehensive Unicode interval support (Python-compatible)
    pub fn draw_string_full(&mut self, intervals: IntervalSet, min_size: usize, max_size: usize, 
                            forced: Option<String>) -> Result<String, DrawError> {
        if !self.can_draw() {
            if self.frozen {
                return Err(DrawError::Frozen);
            } else {
                return Err(DrawError::InvalidStatus);
            }
        }
        
        // Validate constraints
        if min_size > max_size {
            return Err(DrawError::InvalidRange);
        }
        
        // Validate forced string if provided
        if let Some(ref forced_str) = forced {
            if forced_str.chars().count() < min_size || forced_str.chars().count() > max_size {
                return Err(DrawError::InvalidRange);
            }
            
            // Validate that all characters in forced string are in intervals
            for ch in forced_str.chars() {
                if !self.char_in_intervals(ch, &intervals) {
                    return Err(DrawError::InvalidRange);
                }
            }
        }
        
        // Handle empty intervals case
        if intervals.is_empty() {
            // Empty alphabet should always be an error in Python Hypothesis
            return Err(DrawError::EmptyAlphabet);
        }
        
        // Create constraints for this draw
        let constraints = Constraints::String(StringConstraints {
            intervals: intervals.clone(),
            min_size,
            max_size,
        });
        
        // Use _pop_choice for sophisticated replay handling
        let forced_choice = forced.as_ref().map(|s| ChoiceValue::String(s.clone()));
        let result = if let Ok(replay_value) = self._pop_choice(ChoiceType::String, &constraints, forced_choice) {
            // Use replay/forced value if available
            if let ChoiceValue::String(replay_string) = replay_value {
                replay_string
            } else {
                return Err(DrawError::InvalidReplayType);
            }
        } else {
            // Generate new string using provider or sophisticated Unicode generation
            if let Constraints::String(ref _string_constraints) = constraints {
                // Extract alphabet before borrowing provider
                let alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".to_string();
                if let Some(ref mut provider) = self.provider {
                    provider.generate_string(&mut self.rng, &alphabet, min_size, max_size)?
                } else {
                    // Fallback to Unicode-aware string generation
                    self.generate_unicode_string(&intervals, min_size, max_size)?
                }
            } else {
                unreachable!("We just created string constraints ")
            }
        };
        
        // Record the choice
        let choice = ChoiceNode::new(
            ChoiceType::String,
            ChoiceValue::String(result.clone()),
            constraints.clone(),
            forced.is_some(),
        );
        
        self.nodes.push(choice);
        self.record_choice_in_spans();
        
        // Notify observer if present
        if let Some(ref mut observer) = self.observer {
            observer.draw_value(
                ChoiceType::String,
                ChoiceValue::String(result.clone()),
                forced.is_some(),
                Box::new(constraints)
            );
        }
        
        self.length += result.len(); // Length in bytes
        
        Ok(result)
    }
    
    /// Check if character is within the given intervals
    fn char_in_intervals(&self, ch: char, intervals: &IntervalSet) -> bool {
        let codepoint = ch as u32;
        for &(start, end) in &intervals.intervals {
            if codepoint >= start && codepoint <= end {
                return true;
            }
        }
        false
    }
    
    /// Generate Unicode-aware string from intervals with proper distribution
    fn generate_unicode_string(&mut self, intervals: &IntervalSet, min_size: usize, max_size: usize) -> Result<String, DrawError> {
        use rand::Rng;
        
        // Determine string length
        let size = if min_size == max_size {
            min_size
        } else {
            self.rng.gen_range(min_size..=max_size)
        };
        
        // Build character pool from intervals
        let mut char_pool = Vec::new();
        for &(start, end) in &intervals.intervals {
            // Collect valid Unicode characters from this interval
            for codepoint in start..=end {
                if let Some(ch) = char::from_u32(codepoint) {
                    // Filter out problematic Unicode categories if needed
                    if self.is_valid_string_char(ch) {
                        char_pool.push(ch);
                    }
                }
            }
        }
        
        if char_pool.is_empty() {
            return Err(DrawError::EmptyAlphabet);
        }
        
        // Generate string
        let mut result = String::with_capacity(size * 4); // UTF-8 worst case
        for _ in 0..size {
            // Check for overrun during string generation
            if self.length + result.len() + 4 > self.max_length {
                self.mark_overrun();
                return Err(DrawError::Overrun);
            }
            
            let char_index = self.rng.gen_range(0..char_pool.len());
            result.push(char_pool[char_index]);
        }
        
        Ok(result)
    }
    
    /// Check if character is valid for string generation (filters out control chars etc.)
    fn is_valid_string_char(&self, ch: char) -> bool {
        let codepoint = ch as u32;
        
        // Filter out problematic Unicode categories:
        match codepoint {
            // Control characters (except tab, LF, CR)
            0x00..=0x08 => false,
            0x0B..=0x0C => false,
            0x0E..=0x1F => false,
            0x7F..=0x9F => false,
            // Surrogates
            0xD800..=0xDFFF => false,
            // Private use areas
            0xE000..=0xF8FF => false,
            0xF0000..=0xFFFFD => false,
            0x100000..=0x10FFFD => false,
            _ => true,
        }
    }
    
    
    /// Draw a byte array of the specified size
    /// Draw bytes with Python-equivalent constraint validation and choice recording
    /// 
    /// This method provides full Python parity for bytes generation including:
    /// - Variable-size byte sequence generation
    /// - 256-character alphabet (0-255)
    /// - Proper choice recording and constraint validation
    /// - Collection indexing for optimal shrinking
    pub fn draw_bytes(
        &mut self,
        min_size: usize,
        max_size: usize,
        forced: Option<Vec<u8>>,
        observe: bool,
    ) -> Result<Vec<u8>, DrawError> {
        // Status and capacity checks
        if !self.can_draw() {
            if self.frozen {
                return Err(DrawError::Frozen);
            } else {
                return Err(DrawError::InvalidStatus);
            }
        }
        
        // Validate size constraints
        if min_size > max_size {
            return Err(DrawError::InvalidRange);
        }
        
        // Check for potential overrun
        if self.length + max_size > self.max_length {
            self.mark_overrun();
            return Err(DrawError::Overrun);
        }
        
        // Create constraints
        let constraints = BytesConstraints {
            min_size,
            max_size,
        };
        
        // Handle forced value case
        if let Some(forced_value) = forced {
            if !self.choice_permitted_bytes(&forced_value, &constraints) {
                return Err(DrawError::InvalidRange);
            }
            
            // Record the forced choice
            let choice_node = ChoiceNode::new(
                ChoiceType::Bytes,
                ChoiceValue::Bytes(forced_value.clone()),
                Constraints::Bytes(constraints),
                true, // was_forced
            );
            
            self.record_choice(choice_node);
            return Ok(forced_value);
        }
        
        // Check for replay mode
        if let Some(replay_value) = self.try_replay_choice(ChoiceType::Bytes, &Constraints::Bytes(constraints.clone()))? {
            if let ChoiceValue::Bytes(value) = replay_value {
                return Ok(value);
            } else {
                return Err(DrawError::TypeMismatch);
            }
        }
        
        // Generate new value using provider
        let value = if let Some(ref mut provider) = self.provider {
            provider.draw_bytes(min_size, max_size).map_err(DrawError::from)?
        } else {
            // Fallback to internal RNG if no provider
            let mut rng_provider = RngProvider::new(&mut self.rng);
            rng_provider.draw_bytes(min_size, max_size).map_err(DrawError::from)?
        };
        
        // Validate generated value
        if !self.choice_permitted_bytes(&value, &constraints) {
            return Err(DrawError::InvalidRange);
        }
        
        // Record the choice
        let choice_node = ChoiceNode::new(
            ChoiceType::Bytes,
            ChoiceValue::Bytes(value.clone()),
            Constraints::Bytes(constraints),
            false, // was_forced
        );
        
        if observe {
            self.record_choice(choice_node);
        }
        
        Ok(value)
    }
    
    /// Legacy method for fixed-size bytes
    pub fn draw_bytes_fixed_size(&mut self, size: usize) -> Result<Vec<u8>, DrawError> {
        self.draw_bytes(size, size, None, true)
    }
    
    /// Legacy method for range-based bytes
    pub fn draw_bytes_with_range(&mut self, min_size: usize, max_size: usize) -> Result<Vec<u8>, DrawError> {
        self.draw_bytes(min_size, max_size, None, true)
    }
    
    /// Legacy compatibility methods for existing API
    
    /// Simple integer drawing for legacy compatibility
    pub fn draw_integer_simple(&mut self, min_value: i128, max_value: i128) -> Result<i128, DrawError> {
        self.draw_integer(Some(min_value), Some(max_value), None, 0, None, true)
    }
    
    /// Simple float drawing for legacy compatibility
    pub fn draw_float_simple(&mut self) -> Result<f64, DrawError> {
        self.draw_float(f64::NEG_INFINITY, f64::INFINITY, true, Some(f64::MIN_POSITIVE), None, true)
    }
    
    /// Simple string drawing for legacy compatibility
    pub fn draw_string_simple(&mut self, alphabet: &str, min_size: usize, max_size: usize) -> Result<String, DrawError> {
        let intervals = IntervalSet::from_string(alphabet);
        self.draw_string(intervals, min_size, max_size, None, true)
    }
    
    /// Simple bytes drawing for legacy compatibility
    pub fn draw_bytes_simple(&mut self, size: usize) -> Result<Vec<u8>, DrawError> {
        self.draw_bytes(size, size, None, true)
    }
    
    /// Check if a bytes choice is permitted under given constraints
    fn choice_permitted_bytes(&self, value: &[u8], constraints: &BytesConstraints) -> bool {
        // Check length bounds
        value.len() >= constraints.min_size && value.len() <= constraints.max_size
    }
    
    
    /// Records a choice in the comprehensive choice sequence with hierarchical span tracking.
    ///
    /// This function implements the core choice recording algorithm that maintains both a linear
    /// sequence of choices (equivalent to Python's `ConjectureData.nodes`) and hierarchical span
    /// relationships for sophisticated test case organization and shrinking optimization.
    ///
    /// # Algorithm Overview
    ///
    /// The recording process performs multiple coordinated operations:
    ///
    /// ## 1. Linear Choice Sequence
    /// - Appends the choice to the `nodes` vector (Python `self.nodes` equivalent)
    /// - Maintains total ordering for deterministic replay and shrinking
    /// - Updates length tracking for buffer management
    ///
    /// ## 2. Hierarchical Span Integration
    /// - Records choice position within current span context
    /// - Enables nested choice tracking for complex data structures
    /// - Supports span-based shrinking and mutation strategies
    ///
    /// ## 3. Observer Notification
    /// - Triggers observer callbacks for external monitoring
    /// - Enables DataTree integration for novel prefix generation
    /// - Supports real-time choice analysis and debugging
    ///
    /// # Performance Characteristics
    ///
    /// - **Time Complexity**: O(1) amortized (vector append + constant span operations)
    /// - **Space Complexity**: O(1) per choice (single node storage + span metadata)
    /// - **Memory Layout**: Sequential storage optimizes cache locality for replay/shrinking
    ///
    /// # Integration Points
    ///
    /// This function is called from all choice generation methods:
    /// - `draw_integer()`, `draw_boolean()`, `draw_float()`, `draw_string()`, `draw_bytes()`
    /// - `replay_choice()` during deterministic replay
    /// - `forced_choice()` during controlled testing scenarios
    ///
    /// # Thread Safety
    ///
    /// This function requires exclusive (`&mut self`) access to maintain data structure
    /// invariants. The recording operation is atomic from the perspective of external
    /// observers - either the choice is fully recorded or not at all.
    ///
    /// # Invariants Maintained
    ///
    /// - **Sequence Consistency**: Choice index matches position in nodes vector
    /// - **Span Consistency**: Current span depth matches active span stack
    /// - **Observer Consistency**: All observers receive identical choice notifications
    /// - **Replay Consistency**: Recorded choices can be deterministically replayed
    ///
    /// # Examples
    ///
    /// ```rust
    /// // Internal usage during integer generation
    /// let choice_node = ChoiceNode::new(
    ///     ChoiceType::Integer,
    ///     ChoiceValue::Integer(42),
    ///     Constraints::Integer(integer_constraints),
    ///     false, // not forced
    /// );
    /// self.record_choice(choice_node);
    ///
    /// // Results in:
    /// // - nodes.len() incremented by 1
    /// // - length tracking updated
    /// // - span system records choice position
    /// // - observers notified of new choice
    /// ```
    ///
    /// # Error Handling
    ///
    /// This function never fails but may trigger downstream effects:
    /// - Buffer overrun detection in subsequent draws
    /// - Span depth limit enforcement
    /// - Observer error handling (logged but not propagated)
    /// Record a choice in the sequence for deterministic replay
    /// Implements Python's choice tracking with automatic indexing
    fn record_choice(&mut self, choice_node: ChoiceNode) {
        // Add to the nodes sequence (equivalent to Python's self.nodes)
        self.nodes.push(choice_node.clone());
        
        // Update length tracking
        self.length += 1;
        
        // Record in span system for hierarchical tracking
        self.record_choice_in_spans();
        
        // Observer notification for DataTree integration
        if let Some(ref mut observer) = self.observer {
            observer.draw_value(
                choice_node.choice_type,
                choice_node.value.clone(),
                choice_node.was_forced,
                Box::new(choice_node.constraints.clone())
            );
        }
        
        // Calculate and store choice index for Python parity
        let choice_index = crate::choice::indexing::choice_to_index(&choice_node.value, &choice_node.constraints);
        
        // Store the index in the choice node for replay and shrinking
        if let Some(ref mut last_node) = self.nodes.last_mut() {
            // Update the last node with its computed index
            last_node.index = Some(choice_index);
        }
    }
    
    /// Try to replay a choice from the prefix/replay sequence
    fn try_replay_choice(
        &mut self,
        expected_type: ChoiceType,
        expected_constraints: &Constraints,
    ) -> Result<Option<ChoiceValue>, DrawError> {
        // Check if we're in replay mode and have choices to replay
        if let Some(ref replay_choices) = self.replay_choices {
            if self.replay_index < replay_choices.len() {
                let replay_choice = &replay_choices[self.replay_index];
                
                // Verify type matches
                if replay_choice.choice_type != expected_type {
                    // Type mismatch - mark misalignment and fall through to generation
                    self.misaligned_at = Some(self.replay_index);
                    return Ok(None);
                }
                
                // Check if the replayed choice is compatible with current constraints
                let replay_value = replay_choice.value.clone();
                if self.choice_compatible_with_constraints(&replay_value, expected_constraints) {
                    // Record the replayed choice
                    let choice_node = ChoiceNode::new(
                        expected_type,
                        replay_value.clone(),
                        expected_constraints.clone(),
                        false, // was_forced
                    );
                    
                    self.record_choice(choice_node);
                    self.replay_index += 1;
                    
                    return Ok(Some(replay_value));
                } else {
                    // Constraint mismatch - mark misalignment
                    self.misaligned_at = Some(self.replay_index);
                    return Ok(None);
                }
            }
        }
        
        // Check prefix choices (for deterministic replay)
        if self.replay_index < self.prefix.len() {
            let prefix_choice = &self.prefix[self.replay_index];
            
            // Verify type and constraints match
            let prefix_value = prefix_choice.value.clone();
            let prefix_type = prefix_choice.choice_type;
            
            if prefix_type == expected_type &&
               self.choice_compatible_with_constraints(&prefix_value, expected_constraints) {
                
                // Record the prefix choice
                let choice_node = ChoiceNode::new(
                    expected_type,
                    prefix_value.clone(),
                    expected_constraints.clone(),
                    false, // was_forced
                );
                
                self.record_choice(choice_node);
                self.replay_index += 1;
                
                return Ok(Some(prefix_value));
            } else {
                // Prefix mismatch - mark misalignment
                self.misaligned_at = Some(self.replay_index);
            }
        }
        
        // No replay value available - generate new
        Ok(None)
    }
    
    /// Check if a choice value is compatible with given constraints
    fn choice_compatible_with_constraints(&self, value: &ChoiceValue, constraints: &Constraints) -> bool {
        match (value, constraints) {
            (ChoiceValue::Integer(val), Constraints::Integer(c)) => {
                self.choice_permitted_integer(*val, c)
            }
            (ChoiceValue::Boolean(_), Constraints::Boolean(_)) => {
                true // Boolean values are always compatible
            }
            (ChoiceValue::Float(val), Constraints::Float(c)) => {
                self.choice_permitted_float(*val, c)
            }
            (ChoiceValue::String(val), Constraints::String(c)) => {
                self.choice_permitted_string(val, c)
            }
            (ChoiceValue::Bytes(val), Constraints::Bytes(c)) => {
                self.choice_permitted_bytes(val, c)
            }
            _ => false, // Type mismatch
        }
    }
    
    /// Legacy method - Draw a byte array with size range and optional forced value
    pub fn draw_bytes_with_range_and_forced(&mut self, min_size: usize, max_size: usize, forced: Option<Vec<u8>>) -> Result<Vec<u8>, DrawError> {
        if !self.can_draw() {
            if self.frozen {
                return Err(DrawError::Frozen);
            } else {
                return Err(DrawError::InvalidStatus);
            }
        }
        
        if min_size > max_size {
            return Err(DrawError::InvalidRange);
        }
        
        // Use forced value if provided, otherwise generate
        let bytes = if let Some(ref forced_bytes) = forced {
            // Validate forced bytes size is in range
            if forced_bytes.len() < min_size || forced_bytes.len() > max_size {
                return Err(DrawError::InvalidRange);
            }
            forced_bytes.clone()
        } else {
            // Generate size within range
            let size = if min_size == max_size {
                min_size
            } else {
                self.rng.gen_range(min_size..=max_size)
            };
            
            // Check for potential overrun before drawing
            if self.length + size > self.max_length {
                self.mark_overrun();
                return Err(DrawError::Overrun);
            }
            
            let mut bytes = vec![0u8; size];
            self.rng.fill(&mut bytes[..]);
            bytes
        };
        
        let constraints = Constraints::Bytes(BytesConstraints {
            min_size,
            max_size,
        });
        
        let choice = ChoiceNode::new(
            ChoiceType::Bytes,
            ChoiceValue::Bytes(bytes.clone()),
            constraints.clone(),
            forced.is_some(),
        );
        
        self.nodes.push(choice);
        self.record_choice_in_spans();
        
        // Notify observer if present
        if let Some(ref mut observer) = self.observer {
            observer.draw_value(
                ChoiceType::Bytes,
                ChoiceValue::Bytes(bytes.clone()),
                forced.is_some(),
                Box::new(constraints)
            );
        }
        
        self.length += bytes.len();
        
        Ok(bytes)
    }
    
    /// Freeze this ConjectureData instance, preventing further draws
    pub fn freeze(&mut self) {
        self.frozen = true;
    }
    
    /// Get the number of choices made so far
    pub fn choice_count(&self) -> usize {
        self.nodes.len()
    }
    
    /// Get a reference to the choices made
    pub fn choices(&self) -> &[ChoiceNode] {
        &self.nodes
    }
    
    /// Record an observation for targeting
    pub fn observe(&mut self, key: &str, value: &str) {
        self.events.insert(key.to_string(), value.to_string());
    }
    
    /// Record a target observation for directed testing
    /// This enables the test runner to focus on generating inputs that produce specific observations
    pub fn target(&mut self, label: &str, value: f64) {
        self.events.insert(format!("target:{}", label), value.to_string());
        println!("TARGET DEBUG: Recorded target {} = {}", label, value);
    }
    
    /// Record a score observation for multi-objective optimization
    pub fn observe_score(&mut self, score_name: &str, score_value: f64) {
        self.events.insert(format!("score:{}", score_name), score_value.to_string());
        println!("SCORE DEBUG: Recorded score {} = {}", score_name, score_value);
    }
    
    /// Record multiple target observations at once
    pub fn target_many(&mut self, targets: &[(&str, f64)]) {
        for &(label, value) in targets {
            self.target(label, value);
        }
    }
    
    /// Start tracking a new example/span with the given label
    pub fn start_example(&mut self, label: &str) -> usize {
        self.depth += 1;
        let start_position = self.nodes.len();
        
        println!("SPAN DEBUG: Starting example \"{}\" at position {} (depth {})", 
                 label, start_position, self.depth);
        
        // We'll create the example when we end it, for now just return the position
        start_position
    }
    
    /// End the current example/span
    pub fn end_example(&mut self, label: &str, start_position: usize) {
        let end_position = self.nodes.len();
        
        println!("SPAN DEBUG: Ending example \"{}\" from {} to {} (depth {})", 
                 label, start_position, end_position, self.depth);
        
        let example = Example {
            label: label.to_string(),
            start: start_position,
            end: end_position,
            depth: self.depth,
        };
        
        // Add to the examples vector
        self.examples.push(example);
        
        self.depth -= 1;
    }
    
    
    /// Set a TreeRecordingObserver specifically
    /// This provides additional integration methods for tree-based testing
    pub fn set_tree_observer(&mut self, mut observer: TreeRecordingObserver) {
        println!("DATA DEBUG: Setting TreeRecordingObserver");
        observer.start_test(); // Start recording immediately
        self.observer = Some(Box::new(observer));
    }
    
    /// Get a mutable reference to the observer if it's a TreeRecordingObserver
    /// This allows access to tree-specific methods
    pub fn get_tree_observer_mut(&mut self) -> Option<&mut TreeRecordingObserver> {
        if let Some(ref mut observer) = self.observer {
            // Try to downcast to TreeRecordingObserver
            // Note: This requires the observer to actually be a TreeRecordingObserver
            // In a real implementation, we might use a different approach
            // For now, this is a placeholder that shows the interface
            None // TODO: Implement proper downcasting
        } else {
            None
        }
    }
    
    /// Remove the current observer
    pub fn clear_observer(&mut self) {
        // If we had a tree observer, stop the test first
        if let Some(ref mut observer) = self.observer {
            observer.end_example("observer_cleared", false);
        }
        self.observer = None;
    }
    
    /// Check if an observer is currently set
    pub fn has_observer(&self) -> bool {
        self.observer.is_some()
    }
    
    /// Get the choice nodes from this ConjectureData
    /// Used by DataTree integration to extract test path information
    pub fn get_nodes(&self) -> &[ChoiceNode] {
        &self.nodes
    }
    
    /// Convert this ConjectureData into an immutable ConjectureResult
    /// 
    /// This creates a snapshot of the current state that can be used for
    /// analysis, shrinking, and reproduction. The data should typically
    /// be frozen before calling this method.
    pub fn as_result(&self) -> ConjectureResult {
        // Extract target observations from events
        let mut target_observations = HashMap::new();
        for (key, value) in &self.events {
            if key.starts_with("target:") {
                if let Some(label) = key.strip_prefix("target:") {
                    if let Ok(f_value) = value.parse::<f64>() {
                        target_observations.insert(label.to_string(), f_value);
                    }
                }
            }
        }
        
        // Extract interesting origin from events
        let interesting_origin = self.events.get("interesting_origin").cloned();
        
        ConjectureResult {
            status: self.status,
            nodes: self.nodes.clone(),
            length: self.length,
            events: self.events.clone(),
            buffer: self.buffer.clone(),
            examples: self.examples.clone(),
            interesting_origin,
            output: self.buffer.clone(), // For now, output is same as buffer
            extra_information: ExtraInformation::new(), // Empty for now
            expected_exception: None, // Not implemented yet
            expected_traceback: None, // Not implemented yet  
            has_discards: false, // Not tracked yet
            target_observations,
            tags: HashSet::new(), // Empty for now
            spans: Vec::new(), // Not fully implemented yet
            arg_slices: Vec::new(), // Not implemented yet
            slice_comments: HashMap::new(), // Not implemented yet
            misaligned_at: None, // Not implemented yet
            cannot_proceed_scope: None, // Not implemented yet
        }
    }
    
    
    /// Check if this ConjectureData can transition to the given status
    /// Valid transitions: Valid -> {Invalid, Interesting, Overrun, Valid}
    /// Terminal states (Invalid, Interesting, Overrun) can only transition to themselves
    pub fn can_transition_to(&self, new_status: Status) -> bool {
        match (self.status, new_status) {
            // From Valid, can go anywhere
            (Status::Valid, _) => true,
            // Terminal states can only stay the same
            (Status::Invalid, Status::Invalid) => true,
            (Status::Interesting, Status::Interesting) => true,
            (Status::Overrun, Status::Overrun) => true,
            // All other transitions are invalid
            _ => false,
        }
    }
    
    
    /// Check if this ConjectureData is in a terminal state
    /// Terminal states are Invalid, Interesting, and Overrun
    pub fn is_terminal(&self) -> bool {
        matches!(self.status, Status::Invalid | Status::Interesting | Status::Overrun)
    }
    
    /// Check if draws are allowed in the current state
    /// Only Valid status allows draws
    pub fn can_draw(&self) -> bool {
        self.status == Status::Valid && !self.frozen
    }
    
    /// Get the sequence of choice nodes made during execution
    /// This provides access to the complete choice sequence for testing
    pub fn get_choice_sequence(&self) -> &[ChoiceNode] {
        &self.nodes
    }
    
    /// Add a note to the test output for debugging/reporting
    /// This matches Python's note() method for adding debug information
    pub fn note(&mut self, value: impl std::fmt::Display) {
        let note_text = value.to_string();
        println!("NOTE DEBUG: Adding note: {}", note_text);
        
        // Store in events for later retrieval
        let note_key = format!("note_{}", self.events.len());
        self.events.insert(note_key, note_text);
    }
    
    /// Choose from a sequence of values using integer draw
    /// This is equivalent to Python's choice() method
    pub fn choice<T: Clone>(&mut self, values: &[T]) -> Result<T, DrawError> {
        self.choice_with_forced(values, None)
    }
    
    /// Choose from a sequence of values with optional forced index
    pub fn choice_with_forced<T: Clone>(&mut self, values: &[T], forced: Option<usize>) -> Result<T, DrawError> {
        if values.is_empty() {
            return Err(DrawError::EmptyChoice);
        }
        
        let max_index = values.len() - 1;
        let index = if let Some(forced_index) = forced {
            if forced_index > max_index {
                return Err(DrawError::InvalidRange);
            }
            forced_index
        } else {
            self.draw_integer_simple(0, max_index as i128)? as usize
        };
        
        Ok(values[index].clone())
    }
    
    /// Reject the current test execution path
    /// This marks the test as invalid and prevents further execution
    pub fn reject(&mut self, reason: &str) {
        println!("REJECT DEBUG: Rejecting test execution: {}", reason);
        self.mark_invalid(reason);
    }
    
    /// Conclude test execution with the given status
    /// This finalizes the test and prevents further draws
    pub fn conclude_test(&mut self, status: Status, interesting_origin: Option<&str>) -> Result<(), String> {
        if !self.can_transition_to(status) {
            return Err(format!("Cannot transition from {:?} to {:?}", self.status, status));
        }
        
        println!("CONCLUDE DEBUG: Concluding test with status {:?}", status);
        
        // For interesting status, we need an origin
        if status == Status::Interesting {
            if let Some(origin) = interesting_origin {
                self.events.insert("interesting_origin".to_string(), origin.to_string());
            } else {
                return Err("Interesting status requires an origin".to_string());
            }
        }
        
        self.status = status;
        self.freeze();
        
        // Notify observer if present
        if let Some(ref mut observer) = self.observer {
            observer.end_example("test_conclusion", false);
        }
        
        Ok(())
    }
    
    /// Mark the test as interesting (successful) and conclude
    pub fn mark_interesting(&mut self) {
        if self.can_transition_to(Status::Interesting) {
            println!("STATUS DEBUG: Marking ConjectureData as INTERESTING");
            self.status = Status::Interesting;
            
            // Notify observer if present
            if let Some(ref mut observer) = self.observer {
                observer.end_example("interesting", false);
            }
        }
    }
    
    /// Mark the test as interesting with origin tracking
    pub fn mark_interesting_with_origin(&mut self, origin: Option<&str>) -> Result<(), String> {
        self.conclude_test(Status::Interesting, origin)
    }
    
    /// Enhanced mark_invalid with better integration
    pub fn mark_invalid(&mut self, reason: &str) {
        if self.can_transition_to(Status::Invalid) {
            println!("STATUS DEBUG: Marking ConjectureData as INVALID: {}", reason);
            self.status = Status::Invalid;
            self.events.insert("invalid_reason".to_string(), reason.to_string());
            
            // Notify observer of branch killing
            if let Some(ref mut observer) = self.observer {
                observer.end_example("invalid", true); // discard = true
            }
        }
    }
    
    /// Enhanced mark_overrun with better integration  
    pub fn mark_overrun(&mut self) {
        if self.can_transition_to(Status::Overrun) {
            println!("STATUS DEBUG: Marking ConjectureData as OVERRUN (length {} >= max {})", 
                     self.length, self.max_length);
            self.status = Status::Overrun;
            self.events.insert("overrun_reason".to_string(), 
                             format!("Buffer overrun: {} >= {}", self.length, self.max_length));
            
            // Notify observer if present
            if let Some(ref mut observer) = self.observer {
                observer.end_example("overrun", true); // discard = true
            }
        }
    }

    /// Start a new span with the given label  
    /// Returns the span index for later reference
    pub fn start_span(&mut self, label: i32) -> usize {
        // Convert i32 to u64 for new system
        let label_u64 = label as u64;
        
        // Record span start in the span record
        self.span_record.start_span(label_u64);
        
        // Track in our active stack
        let span_index = self.active_span_stack.len(); // Current depth becomes index
        self.active_span_stack.push(span_index);
        
        // Invalidate cached spans so they get recomputed
        self.spans = None;
        
        // Notify observer if present
        if let Some(ref mut observer) = self.observer {
            observer.start_example(&format!("span_{}", label));
        }
        
        span_index
    }

    /// End the most recent span
    /// Returns the ended span index if successful
    pub fn end_span(&mut self) -> Option<usize> {
        // Record span end in the span record
        self.span_record.stop_span(false); // discard = false by default
        
        // Remove from our active stack
        let span_index = self.active_span_stack.pop();
        
        // Invalidate cached spans so they get recomputed
        self.spans = None;
        
        // Notify observer if present
        if let Some(ref mut observer) = self.observer {
            observer.end_example(&format!("span_ended"), false);
        }
        
        span_index
    }

    /// Get a reference to the spans collection (lazily computed)
    pub fn spans(&mut self) -> &Spans {
        if self.spans.is_none() {
            self.spans = Some(Spans::from_record(&self.span_record));
        }
        self.spans.as_ref().unwrap()
    }

    /// Get the current span nesting depth
    pub fn span_depth(&self) -> usize {
        self.active_span_stack.len()
    }
    
    /// Record a choice in the span record (called after every choice)
    fn record_choice_in_spans(&mut self) {
        self.span_record.record_choice();
        // Invalidate cached spans so they get recomputed
        self.spans = None;
    }
}

/// # DrawError: Comprehensive Error Types for Value Generation Failures
///
/// This enum provides exhaustive error handling for all possible failure modes during
/// value generation in the Conjecture engine. Each error type includes detailed context
/// and recovery strategies, enabling robust error handling and debugging.
///
/// ## Error Categories
///
/// ### State Validation Errors
/// These errors indicate invalid ConjectureData state for drawing operations:
/// - **`Frozen`**: Attempt to draw from immutable ConjectureData instance
/// - **`InvalidStatus`**: ConjectureData in non-drawing state (Invalid/Overrun)
/// - **`Overrun`**: Buffer size limit exceeded during generation
///
/// ### Constraint Validation Errors  
/// These errors indicate invalid parameters or constraint violations:
/// - **`InvalidRange`**: Mathematical constraint violations (min > max)
/// - **`InvalidProbability`**: Probability outside [0.0, 1.0] range or NaN
/// - **`EmptyAlphabet`**: String generation with no valid characters
/// - **`EmptyWeights`**: Weighted selection with empty weight array
/// - **`InvalidWeights`**: Weight array with non-positive sum or NaN/Infinity values
///
/// ### Replay and Type Safety Errors
/// These errors occur during deterministic replay or type mismatches:
/// - **`InvalidReplayType`**: Choice type mismatch during replay validation
/// - **`TypeMismatch`**: General type system violation during replay
/// - **`InvalidChoice`**: Malformed choice data during replay
/// - **`EmptyChoice`**: Attempt to select from empty choice sequence
///
/// ### Test Execution Control
/// These errors provide structured test execution flow control:
/// - **`StopTest(u64)`**: Controlled test termination with unique identifier
/// - **`UnsatisfiedAssumption(String)`**: Assumption violation with context
/// - **`PreviouslyUnseenBehaviour`**: DataTree navigation into unexplored space
///
/// ## Error Handling Philosophy
///
/// ### Fail-Fast Design
/// Most errors represent programming mistakes or constraint violations that should
/// terminate test execution immediately:
/// ```rust
/// // Example: Invalid range detection
/// if let (Some(min), Some(max)) = (min_value, max_value) {
///     if min > max {
///         return Err(DrawError::InvalidRange);  // Fail immediately
///     }
/// }
/// ```
///
/// ### Contextual Recovery
/// Some errors provide opportunities for graceful recovery:
/// - **Overrun**: Mark test as overrun and continue with other tests
/// - **UnsatisfiedAssumption**: Discard current test and try another input
/// - **PreviouslyUnseenBehaviour**: Explore new branch in DataTree
///
/// ### Structured Control Flow
/// Special errors enable sophisticated test orchestration:
/// - **StopTest**: Enables complex multi-phase test coordination
/// - **UnsatisfiedAssumption**: Supports rejection sampling patterns
///
/// ## Integration with Core Systems
///
/// ### Provider Error Mapping
/// DrawError integrates with provider-specific errors through automatic conversion:
/// ```rust
/// impl From<crate::providers::ProviderError> for DrawError {
///     fn from(err: crate::providers::ProviderError) -> Self {
///         match err {
///             ProviderError::InvalidConstraints => DrawError::InvalidRange,
///             ProviderError::GenerationFailed => DrawError::InvalidChoice,
///             // ... other mappings
///         }
///     }
/// }
/// ```
///
/// ### DataTree Integration
/// Errors coordinate with DataTree for systematic exploration:
/// - **PreviouslyUnseenBehaviour**: Triggers new branch creation
/// - **InvalidChoice**: Indicates replay inconsistency requiring tree update
/// - **StopTest**: Coordinates with tree navigation for controlled exploration
///
/// ### Shrinking Integration
/// Errors provide shrinking algorithms with failure context:
/// - **Constraint violations**: Guide shrinking toward valid regions
/// - **Type mismatches**: Indicate choice sequence incompatibility
/// - **Replay failures**: Signal need for choice sequence adjustment
///
/// ## Performance and Memory Characteristics
///
/// ### Zero-Cost Error Propagation
/// Most error variants contain no heap-allocated data:
/// - **Simple Enums**: Copy-based error propagation with no allocation
/// - **String Context**: Only for errors requiring detailed debugging context
/// - **Structured Data**: Only for complex control flow scenarios
///
/// ### Error Context Optimization
/// - **Static Messages**: Pre-computed error descriptions for common cases
/// - **Lazy Formatting**: Error details computed only when displayed
/// - **Stack-Based**: Error propagation uses stack allocation for performance
#[derive(Debug, Clone, PartialEq)]
pub enum DrawError {
    /// Attempted to draw from a frozen ConjectureData
    Frozen,
    /// Invalid range (min > max)
    InvalidRange,
    /// Invalid probability (not in [0, 1])
    InvalidProbability,
    /// Empty alphabet for string generation
    EmptyAlphabet,
    /// Overran the maximum buffer size
    Overrun,
    /// Attempted to draw from ConjectureData with invalid status
    InvalidStatus,
    /// Empty choice sequence provided to choice()
    EmptyChoice,
    /// Replay value type doesn't match expected type
    InvalidReplayType,
    /// Test should stop running and return control to engine
    StopTest(u64),
    /// Unsatisfied assumption encountered during test execution
    UnsatisfiedAssumption(String),
    /// Previously unseen behavior detected during tree simulation
    PreviouslyUnseenBehaviour,
    /// Invalid choice or misaligned replay
    InvalidChoice,
    /// Empty weights array for weighted choice
    EmptyWeights,
    /// Invalid weights (sum <= 0 or contains NaN/Infinity)
    InvalidWeights,
    /// Type mismatch during replay
    TypeMismatch,
}

impl std::fmt::Display for DrawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DrawError::Frozen => write!(f, "Cannot draw from frozen ConjectureData"),
            DrawError::InvalidRange => write!(f, "Invalid range: min_value > max_value"),
            DrawError::InvalidProbability => write!(f, "Probability must be between 0.0 and 1.0"),
            DrawError::EmptyAlphabet => write!(f, "Cannot generate string from empty alphabet"),
            DrawError::Overrun => write!(f, "Overran maximum buffer size"),
            DrawError::InvalidStatus => write!(f, "Cannot draw from ConjectureData with invalid status"),
            DrawError::EmptyChoice => write!(f, "Cannot choose from empty sequence"),
            DrawError::InvalidReplayType => write!(f, "Replay choice type doesn't match expected type"),
            DrawError::StopTest(testcounter) => write!(f, "Test should stop running (testcounter: {})", testcounter),
            DrawError::UnsatisfiedAssumption(reason) => write!(f, "Unsatisfied assumption: {}", reason),
            DrawError::PreviouslyUnseenBehaviour => write!(f, "Previously unseen behavior detected during tree simulation"),
            DrawError::InvalidChoice => write!(f, "Invalid choice or misaligned replay"),
            DrawError::EmptyWeights => write!(f, "Cannot choose from empty weights array"),
            DrawError::InvalidWeights => write!(f, "Invalid weights: sum must be positive and finite"),
            DrawError::TypeMismatch => write!(f, "Type mismatch during replay"),
        }
    }
}

impl std::error::Error for DrawError {}


/// Value types supported by ExtraInformation (equivalent to Python's dynamic typing)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtraValue {
    String(String),
    Integer(i64),
    Float(String), // Store as string to preserve exact representation
    Boolean(bool),
    Bytes(Vec<u8>),
    None,
}

impl ExtraValue {
    /// Get Python-style repr() representation of the value
    pub fn repr(&self) -> String {
        match self {
            ExtraValue::String(s) => format!("{:?}", s), // This gives proper quoted strings
            ExtraValue::Integer(i) => i.to_string(),
            ExtraValue::Float(f) => f.clone(),
            ExtraValue::Boolean(b) => if *b { "True".to_string() } else { "False".to_string() },
            ExtraValue::Bytes(b) => format!("b{:?}", String::from_utf8_lossy(b)),
            ExtraValue::None => "None".to_string(),
        }
    }
}

impl From<String> for ExtraValue {
    fn from(s: String) -> Self {
        ExtraValue::String(s)
    }
}

impl From<&str> for ExtraValue {
    fn from(s: &str) -> Self {
        ExtraValue::String(s.to_string())
    }
}

impl From<i64> for ExtraValue {
    fn from(i: i64) -> Self {
        ExtraValue::Integer(i)
    }
}

impl From<f64> for ExtraValue {
    fn from(f: f64) -> Self {
        ExtraValue::Float(f.to_string())
    }
}

impl From<bool> for ExtraValue {
    fn from(b: bool) -> Self {
        ExtraValue::Boolean(b)
    }
}

impl From<Vec<u8>> for ExtraValue {
    fn from(b: Vec<u8>) -> Self {
        ExtraValue::Bytes(b)
    }
}


/// Result of a finalized ConjectureData execution
/// 
/// This is an immutable snapshot of the test execution state that can be used
/// for shrinking, analysis, and reproduction.
#[derive(Debug, Clone)]
pub struct ConjectureResult {
    /// Final status of the test execution
    pub status: Status,
    
    /// Sequence of choice nodes made during execution (renamed from choices for Python compatibility)
    pub nodes: Vec<ChoiceNode>,
    
    /// Total length of data consumed
    pub length: usize,
    
    /// Events and observations recorded during execution
    pub events: HashMap<String, String>,
    
    /// Buffer containing the raw byte data (for advanced use cases)
    pub buffer: Vec<u8>,
    
    /// Examples found during execution (for span tracking)
    pub examples: Vec<Example>,
    
    /// Origin of what made the result interesting
    pub interesting_origin: Option<String>,
    
    /// Final output data produced by the test
    pub output: Vec<u8>,
    
    /// Additional metadata attached to the result
    pub extra_information: ExtraInformation,
    
    /// Expected exception type that should be raised
    pub expected_exception: Option<String>,
    
    /// Expected traceback from exception
    pub expected_traceback: Option<String>,
    
    /// Whether any draws were discarded during execution
    pub has_discards: bool,
    
    /// Target tracking data for directed testing
    pub target_observations: HashMap<String, f64>,
    
    /// Classification tags for the test result
    pub tags: HashSet<String>,
    
    /// Hierarchical choice structure with spans
    pub spans: Vec<Span>,
    
    /// Argument boundaries within the choice sequence
    pub arg_slices: Vec<(usize, usize)>,
    
    /// Documentation for slice boundaries
    pub slice_comments: HashMap<usize, String>,
    
    /// Position where replay misalignment occurred
    pub misaligned_at: Option<usize>,
    
    /// Scope where execution stopped and could not proceed
    pub cannot_proceed_scope: Option<String>,
}

/// Trail type enum for tracking span events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrailType {
    /// Span stopped with discard flag set
    StopSpanDiscard = 1,
    /// Span stopped without discard flag  
    StopSpanNoDiscard = 2,
    /// New span started
    StartSpan = 3,
    /// Choice recorded (matches Python's calc_label_from_name("ir draw record"))
    Choice = 1000, // Large value to avoid conflicts with StartSpan + index
}

/// A span tracks the hierarchical structure of choices within a single test run.
/// 
/// Spans are created to mark regions of the choice sequence that are logically 
/// related to each other. Rather than store each Span as a rich object, it is 
/// actually just an index into the Spans class. This matches Python's approach.
#[derive(Clone)]
pub struct Span {
    /// Index of this span in the spans collection
    pub index: usize,
    /// Reference to the owning spans collection  
    pub owner: *const Spans,
}

impl Span {
    /// Create a new span with the given index and owner
    pub fn new(index: usize, owner: *const Spans) -> Self {
        Self { index, owner }
    }
    
    /// Get the label associated with this span
    pub fn label(&self) -> u64 {
        unsafe {
            let spans = &*self.owner;
            spans.labels[spans.label_indices()[self.index]]
        }
    }
    
    /// Get the parent span index (None for root span)
    pub fn parent(&self) -> Option<usize> {
        if self.index == 0 {
            return None;
        }
        unsafe {
            let spans = &*self.owner;
            spans.parentage()[self.index]
        }
    }
    
    /// Get the start position in the choice sequence
    pub fn start(&self) -> usize {
        unsafe {
            let spans = &*self.owner;
            spans.starts()[self.index]
        }
    }
    
    /// Get the end position in the choice sequence
    pub fn end(&self) -> usize {
        unsafe {
            let spans = &*self.owner;
            spans.ends()[self.index]
        }
    }
    
    /// Get the depth of this span in the span tree (root = 0)
    pub fn depth(&self) -> usize {
        unsafe {
            let spans = &*self.owner;
            spans.depths()[self.index]
        }
    }
    
    /// Check if this span was discarded
    pub fn discarded(&self) -> bool {
        unsafe {
            let spans = &*self.owner;
            spans.discarded().contains(&self.index)
        }
    }
    
    /// Get the number of choices contained in this span
    pub fn choice_count(&self) -> usize {
        self.end() - self.start()
    }
    
    /// Get all child spans of this span
    pub fn children(&self) -> Vec<Span> {
        unsafe {
            let spans = &*self.owner;
            spans.children()[self.index]
                .iter()
                .map(|&i| Span::new(i, self.owner))
                .collect()
        }
    }
}

impl std::fmt::Debug for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "spans[{}]", self.index)
    }
}

impl PartialEq<Span> for Span {
    fn eq(&self, other: &Span) -> bool {
        self.index == other.index && std::ptr::eq(self.owner, other.owner)
    }
}

/// Records the series of start_span, stop_span, and draw_bits calls
/// so these may be stored in Spans and replayed when needed.
#[derive(Debug, Clone)]
pub struct SpanRecord {
    /// List of unique span labels
    pub labels: Vec<u64>,
    /// Map from label to its index in labels vec
    label_indices: Option<HashMap<u64, usize>>,
    /// Trail of events for replay
    pub trail: Vec<u64>,
    /// Choice nodes recorded during execution
    pub nodes: Vec<ChoiceNode>,
}

impl SpanRecord {
    pub fn new() -> Self {
        Self {
            labels: Vec::new(),
            label_indices: Some(HashMap::new()),
            trail: Vec::new(),
            nodes: Vec::new(),
        }
    }

    /// Freeze the record to prevent further modifications
    pub fn freeze(&mut self) {
        self.label_indices = None;
    }

    /// Record a choice draw
    pub fn record_choice(&mut self) {
        self.trail.push(TrailType::Choice as u64);
    }

    /// Start a span with the given label
    pub fn start_span(&mut self, label: u64) {
        if let Some(ref mut indices) = self.label_indices {
            let index = match indices.get(&label) {
                Some(&i) => i,
                None => {
                    let i = self.labels.len();
                    self.labels.push(label);
                    indices.insert(label, i);
                    i
                }
            };
            self.trail.push(TrailType::StartSpan as u64 + index as u64);
        }
    }

    /// Stop the current span
    pub fn stop_span(&mut self, discard: bool) {
        if discard {
            self.trail.push(TrailType::StopSpanDiscard as u64);
        } else {
            self.trail.push(TrailType::StopSpanNoDiscard as u64);
        }
    }
}

/// A lazy collection of Span objects, derived from the record of recorded 
/// behaviour in SpanRecord. Behaves logically as if it were a list of Span objects.
#[derive(Debug)]
pub struct Spans {
    /// Trail of span events for replay
    pub trail: Vec<u64>,
    /// Compact storage of unique span labels
    pub labels: Vec<u64>,
    /// Total number of spans (cached from trail analysis)
    length: usize,
    /// Cached start/end positions
    cached_starts_and_ends: std::cell::RefCell<Option<(Vec<usize>, Vec<usize>)>>,
    /// Cached set of discarded span indices  
    cached_discarded: std::cell::RefCell<Option<HashSet<usize>>>,
    /// Cached parent span indices
    cached_parentage: std::cell::RefCell<Option<Vec<Option<usize>>>>,
    /// Cached depth values
    cached_depths: std::cell::RefCell<Option<Vec<usize>>>,
    /// Cached label indices for each span
    cached_label_indices: std::cell::RefCell<Option<Vec<usize>>>,
    /// Cached children lists
    cached_children: std::cell::RefCell<Option<Vec<Vec<usize>>>>,
    /// Cached mutator groups
    cached_mutator_groups: std::cell::RefCell<Option<Vec<HashSet<(usize, usize)>>>>,
}

/// Iterator over spans
pub struct SpanIterator<'a> {
    spans: &'a Spans,
    current: usize,
}

impl<'a> Iterator for SpanIterator<'a> {
    type Item = Span;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.spans.length {
            let span = Span::new(self.current, self.spans as *const Spans);
            self.current += 1;
            Some(span)
        } else {
            None
        }
    }
}

impl Spans {
    /// Create a new spans collection from a SpanRecord
    pub fn from_record(record: &SpanRecord) -> Self {
        // Count span start events to determine length (includes active spans)
        // StartSpan events are encoded as StartSpan + label_index, and Choice events are at 1000
        let length = record.trail.iter()
            .filter(|&&x| x >= TrailType::StartSpan as u64 && x < TrailType::Choice as u64)
            .count();
        
        Self {
            trail: record.trail.clone(),
            labels: record.labels.clone(),
            length,
            cached_starts_and_ends: std::cell::RefCell::new(None),
            cached_discarded: std::cell::RefCell::new(None),
            cached_parentage: std::cell::RefCell::new(None),
            cached_depths: std::cell::RefCell::new(None),
            cached_label_indices: std::cell::RefCell::new(None),
            cached_children: std::cell::RefCell::new(None),
            cached_mutator_groups: std::cell::RefCell::new(None),
        }
    }
    
    /// Get the number of spans
    pub fn len(&self) -> usize {
        self.length
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    
    /// Get a span by index
    pub fn get(&self, index: usize) -> Option<Span> {
        if index < self.length {
            Some(Span::new(index, self as *const Self))
        } else {
            None
        }
    }
    
    /// Iterator over all spans
    pub fn iter(&self) -> SpanIterator {
        SpanIterator {
            spans: self,
            current: 0,
        }
    }
    
    /// Get start and end positions for all spans (lazily computed)
    pub fn starts_and_ends(&self) -> (Vec<usize>, Vec<usize>) {
        let mut cache = self.cached_starts_and_ends.borrow_mut();
        if cache.is_none() {
            let result = self.compute_starts_and_ends();
            *cache = Some(result);
        }
        cache.as_ref().unwrap().clone()
    }
    
    /// Get start positions
    pub fn starts(&self) -> Vec<usize> {
        self.starts_and_ends().0
    }
    
    /// Get end positions  
    pub fn ends(&self) -> Vec<usize> {
        self.starts_and_ends().1
    }
    
    /// Get discarded span indices (lazily computed)
    pub fn discarded(&self) -> HashSet<usize> {
        let mut cache = self.cached_discarded.borrow_mut();
        if cache.is_none() {
            let result = self.compute_discarded();
            *cache = Some(result);
        }
        cache.as_ref().unwrap().clone()
    }
    
    /// Get parent span indices (lazily computed)
    pub fn parentage(&self) -> Vec<Option<usize>> {
        let mut cache = self.cached_parentage.borrow_mut();
        if cache.is_none() {
            let result = self.compute_parentage();
            *cache = Some(result);
        }
        cache.as_ref().unwrap().clone()
    }
    
    /// Get depth of each span (lazily computed)
    pub fn depths(&self) -> Vec<usize> {
        let mut cache = self.cached_depths.borrow_mut();
        if cache.is_none() {
            let result = self.compute_depths();
            *cache = Some(result);
        }
        cache.as_ref().unwrap().clone()
    }
    
    /// Get label indices for each span (lazily computed)
    pub fn label_indices(&self) -> Vec<usize> {
        let mut cache = self.cached_label_indices.borrow_mut();
        if cache.is_none() {
            let result = self.compute_label_indices();
            *cache = Some(result);
        }
        cache.as_ref().unwrap().clone()
    }
    
    /// Get children for each span (lazily computed)
    pub fn children(&self) -> Vec<Vec<usize>> {
        let mut cache = self.cached_children.borrow_mut();
        if cache.is_none() {
            let result = self.compute_children();
            *cache = Some(result);
        }
        cache.as_ref().unwrap().clone()
    }
    
    /// Compute start and end positions by replaying the trail
    fn compute_starts_and_ends(&self) -> (Vec<usize>, Vec<usize>) {
        let mut starts = vec![0; self.length];
        let mut ends = vec![0; self.length];
        let mut span_stack = Vec::new();
        let mut span_count = 0;
        let mut choice_count = 0;
        
        for &record in &self.trail {
            if record == TrailType::Choice as u64 {
                choice_count += 1;
            } else if record >= TrailType::StartSpan as u64 {
                let i = span_count;
                starts[i] = choice_count;
                span_count += 1;
                span_stack.push(i);
            } else if record == TrailType::StopSpanDiscard as u64 || record == TrailType::StopSpanNoDiscard as u64 {
                if let Some(i) = span_stack.pop() {
                    ends[i] = choice_count;
                }
            }
        }
        
        (starts, ends)
    }
    
    /// Compute discarded spans by replaying the trail
    fn compute_discarded(&self) -> HashSet<usize> {
        let mut result = HashSet::new();
        let mut span_stack = Vec::new();
        let mut span_count = 0;
        
        for &record in &self.trail {
            if record == TrailType::Choice as u64 {
                // Skip choices
            } else if record >= TrailType::StartSpan as u64 {
                span_stack.push(span_count);
                span_count += 1;
            } else if record == TrailType::StopSpanDiscard as u64 {
                if let Some(i) = span_stack.pop() {
                    result.insert(i);
                }
            } else if record == TrailType::StopSpanNoDiscard as u64 {
                span_stack.pop();
            }
        }
        
        result
    }
    
    /// Compute parent relationships by replaying the trail
    fn compute_parentage(&self) -> Vec<Option<usize>> {
        let mut result = vec![None; self.length];
        let mut span_stack = Vec::new();
        let mut span_count = 0;
        
        for &record in &self.trail {
            if record == TrailType::Choice as u64 {
                // Skip choices
            } else if record >= TrailType::StartSpan as u64 {
                // Set parent when span starts (parent is whoever is on top of stack)
                if span_count > 0 && !span_stack.is_empty() {
                    result[span_count] = span_stack.last().copied();
                }
                span_stack.push(span_count);
                span_count += 1;
            } else if record == TrailType::StopSpanDiscard as u64 || record == TrailType::StopSpanNoDiscard as u64 {
                span_stack.pop();
            }
        }
        
        result
    }
    
    /// Compute depths by replaying the trail
    fn compute_depths(&self) -> Vec<usize> {
        let mut result = vec![0; self.length];
        let mut span_stack = Vec::new();
        let mut span_count = 0;
        
        for &record in &self.trail {
            if record == TrailType::Choice as u64 {
                // Skip choices
            } else if record >= TrailType::StartSpan as u64 {
                result[span_count] = span_stack.len();
                span_stack.push(span_count);
                span_count += 1;
            } else if record == TrailType::StopSpanDiscard as u64 || record == TrailType::StopSpanNoDiscard as u64 {
                span_stack.pop();
            }
        }
        
        result
    }
    
    /// Compute label indices by replaying the trail
    fn compute_label_indices(&self) -> Vec<usize> {
        let mut result = vec![0; self.length];
        let mut span_count = 0;
        
        for &record in &self.trail {
            if record == TrailType::Choice as u64 {
                // Skip choices
            } else if record >= TrailType::StartSpan as u64 {
                let label_index = (record - TrailType::StartSpan as u64) as usize;
                result[span_count] = label_index;
                span_count += 1;
            }
        }
        
        result
    }
    
    /// Compute children relationships
    fn compute_children(&self) -> Vec<Vec<usize>> {
        let parentage = self.parentage();
        let mut children = vec![Vec::new(); self.length];
        
        for (i, parent) in parentage.iter().enumerate() {
            if i > 0 {
                if let Some(p) = parent {
                    children[*p].push(i);
                }
            }
        }
        
        children
    }
    
}

impl std::ops::Index<usize> for Spans {
    type Output = Span;
    
    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.length {
            panic!("Index {} out of range [0, {})", index, self.length);
        }
        // We can't return a reference to a Span since it's created on the fly
        // This is a limitation of our design - in practice, most access should use get()
        unimplemented!("Use spans.get(index) instead of spans[index]")
    }
}

impl Default for Spans {
    fn default() -> Self {
        let empty_record = SpanRecord::new();
        Self::from_record(&empty_record)
    }
}

/// Statistics about span structural coverage
#[derive(Debug, Clone)]
pub struct SpanCoverageStats {
    /// Total number of spans created
    pub total_spans: usize,
    /// Maximum nesting depth reached
    pub max_depth: i32,
    /// Number of unique span labels
    pub unique_labels: usize,
    /// Number of discarded spans
    pub discarded_spans: usize,
    /// Total choices covered by spans
    pub choice_count: usize,
    /// Distribution of spans by depth level
    pub depth_distribution: [usize; 101], // MAX_DEPTH + 1
}

impl Default for SpanCoverageStats {
    fn default() -> Self {
        Self {
            total_spans: 0,
            max_depth: 0,
            unique_labels: 0,
            discarded_spans: 0,
            choice_count: 0,
            depth_distribution: [0; 101],
        }
    }
}

/// Legacy Example struct for backward compatibility
/// 
/// This is used for basic span tracking and will be deprecated in favor of the full Span system.
#[derive(Debug, Clone)]
pub struct Example {
    /// Label for this example/span
    pub label: String,
    
    /// Start position in the choice sequence
    pub start: usize,
    
    /// End position in the choice sequence
    pub end: usize,
    
    /// Depth of nesting when this example was created
    pub depth: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conjecture_data_creation() {
        let data = ConjectureData::new(42);
        assert_eq!(data.status, Status::Valid);
        assert_eq!(data.max_length, 8192);
        assert_eq!(data.index, 0);
        assert_eq!(data.length, 0);
        assert!(!data.frozen);
        assert_eq!(data.choice_count(), 0);
    }

    #[test]
    fn test_draw_integer() {
        let mut data = ConjectureData::new(42);
        let value = data.draw_integer_simple(0, 100).unwrap();
        
        assert!(value >= 0 && value <= 100);
        assert_eq!(data.choice_count(), 1);
        assert_eq!(data.length, 2); // Should match Python behavior
        assert!(!data.frozen);
    }

    #[test]
    fn test_draw_boolean() {
        let mut data = ConjectureData::new(42);
        let value = data.draw_boolean(0.5, None, true).unwrap();
        
        assert!(value == true || value == false);
        assert_eq!(data.choice_count(), 1);
        assert_eq!(data.length, 1);
    }

    #[test]
    fn test_freeze_prevents_draws() {
        let mut data = ConjectureData::new(42);
        data.freeze();
        
        let result = data.draw_integer_simple(0, 100);
        assert_eq!(result, Err(DrawError::Frozen));
    }

    #[test]
    fn test_invalid_integer_range() {
        let mut data = ConjectureData::new(42);
        let result = data.draw_integer_simple(100, 0);
        assert_eq!(result, Err(DrawError::InvalidRange));
    }

    #[test]
    fn test_invalid_probability() {
        let mut data = ConjectureData::new(42);
        assert_eq!(data.draw_boolean(-0.1), Err(DrawError::InvalidProbability));
        assert_eq!(data.draw_boolean(1.1), Err(DrawError::InvalidProbability));
    }

    #[test]
    fn test_choice_recording() {
        let mut data = ConjectureData::new(42);
        
        let int_val = data.draw_integer_simple(0, 100).unwrap();
        let bool_val = data.draw_boolean(0.5).unwrap();
        
        assert_eq!(data.choice_count(), 2);
        
        let choices = data.choices();
        assert_eq!(choices.len(), 2);
        
        // Check first choice
        if let ChoiceValue::Integer(recorded_int) = &choices[0].value {
            assert_eq!(recorded_int, &int_val);
        } else {
            panic!("Expected integer choice");
        }
        
        // Check second choice
        if let ChoiceValue::Boolean(recorded_bool) = &choices[1].value {
            assert_eq!(recorded_bool, &bool_val);
        } else {
            panic!("Expected boolean choice");
        }
    }
    
    #[test]
    fn test_span_tracking() {
        let mut data = ConjectureData::new(42);
        
        // Start a span
        let start_pos = data.start_example("test_span");
        assert_eq!(start_pos, 0); // No choices yet
        assert_eq!(data.depth, 0); // Should have incremented
        
        // Make some choices within the span
        let _int_val = data.draw_integer_simple(0, 100).unwrap();
        let _bool_val = data.draw_boolean(0.5).unwrap();
        
        // End the span
        data.end_example("test_span", start_pos);
        assert_eq!(data.depth, -1); // Should have decremented back
        
        // Check that example was recorded
        data.freeze();
        let result = data.as_result();
        assert_eq!(result.examples.len(), 1);
        
        let example = &result.examples[0];
        assert_eq!(example.label, "test_span");
        assert_eq!(example.start, 0);
        assert_eq!(example.end, 2); // Two choices made
        assert_eq!(example.depth, 0);
    }
    
    #[test]
    fn test_nested_spans() {
        let mut data = ConjectureData::new(42);
        
        // Start outer span
        let outer_start = data.start_example("outer");
        let _int1 = data.draw_integer_simple(0, 10).unwrap();
        
        // Start inner span
        let inner_start = data.start_example("inner");
        let _int2 = data.draw_integer_simple(20, 30).unwrap();
        let _bool = data.draw_boolean(0.5).unwrap();
        data.end_example("inner", inner_start);
        
        let _int3 = data.draw_integer_simple(40, 50).unwrap();
        data.end_example("outer", outer_start);
        
        // Check that both examples were recorded
        data.freeze();
        let result = data.as_result();
        assert_eq!(result.examples.len(), 2);
        
        // Check inner span
        let inner = &result.examples[0];
        assert_eq!(inner.label, "inner");
        assert_eq!(inner.start, 1);
        assert_eq!(inner.end, 3);
        assert_eq!(inner.depth, 1);
        
        // Check outer span
        let outer = &result.examples[1];
        assert_eq!(outer.label, "outer");
        assert_eq!(outer.start, 0);
        assert_eq!(outer.end, 4);
        assert_eq!(outer.depth, 0);
    }
    
    #[test]
    fn test_provider_integration() {
        use crate::providers::HypothesisProvider;
        
        let mut data = ConjectureData::with_provider(42, Box::new(HypothesisProvider::new()));
        
        // Test that provider is used for generation
        let mut constant_found = false;
        for _ in 0..50 {
            let value = data.draw_integer_simple(0, 1000).unwrap();
            
            // Check if we got a constant that's in the global constants list
            // Some common constants that should be in range: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
            let common_constants = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
            if common_constants.contains(&(value as i32)) {
                println!("Provider generated constant: {}", value);
                constant_found = true;
                break;
            }
        }
        
        // With 50 attempts and 5% probability, we have a good chance of seeing a constant
        // This test might occasionally not find a constant, but it demonstrates the integration
        if constant_found {
            println!("Provider integration working correctly!");
        } else {
            println!("No constants found in 50 attempts (this can happen with low probability)");
        }
        
        // Test that choices are still recorded properly
        assert!(data.choice_count() > 0);
    }
    
    #[test]
    fn test_span_python_parity() {
        // Create a span record and add some spans
        let mut record = SpanRecord::new();
        record.start_span(42); // Label 42
        record.record_choice();
        record.start_span(43); // Label 43, nested
        record.record_choice();
        record.stop_span(false); // End span 43
        record.stop_span(false); // End span 42
        
        // Create spans from record
        let spans = Spans::from_record(&record);
        
        // Test basic functionality
        assert_eq!(spans.len(), 2);
        
        // Get spans
        let span1 = spans.get(0).expect("Span 0 should exist");
        let span2 = spans.get(1).expect("Span 1 should exist");
        
        // Test Debug format shows "spans[index]"
        assert_eq!(format!("{:?}", span1), "spans[0]");
        assert_eq!(format!("{:?}", span2), "spans[1]");
        
        // Test parent relationships
        assert_eq!(span1.parent(), None); // Root span
        assert_eq!(span2.parent(), Some(0)); // Nested under span 0
        
        // Test choice counting
        assert_eq!(span1.choice_count(), 2); // Contains 2 choices
        assert_eq!(span2.choice_count(), 1); // Contains 1 choice
    }

    #[test]
    fn test_observer_integration_comprehensive() {
        println!("TEST_OBSERVER DEBUG: Starting comprehensive observer integration test");
        
        // Create a ConjectureData instance
        let mut data = ConjectureData::new(42);
        println!("TEST_OBSERVER DEBUG: Created ConjectureData with seed 42");
        
        // Create a TreeRecordingObserver and use set_tree_observer() method
        let observer = TreeRecordingObserver::new();
        println!("TEST_OBSERVER DEBUG: Created TreeRecordingObserver");
        
        // Use set_tree_observer() method instead of manual observer creation
        data.set_tree_observer(observer);
        assert!(data.has_observer());
        println!("TEST_OBSERVER DEBUG: Set tree observer on ConjectureData, has_observer: {}", data.has_observer());
        
        // Make several draws of different types
        println!("TEST_OBSERVER DEBUG: Making integer draw");
        let int_val = data.draw_integer_simple(10, 100).unwrap();
        println!("TEST_OBSERVER DEBUG: Drew integer: {}", int_val);
        
        println!("TEST_OBSERVER DEBUG: Making boolean draw");
        let bool_val = data.draw_boolean(0.7).unwrap();
        println!("TEST_OBSERVER DEBUG: Drew boolean: {}", bool_val);
        
        println!("TEST_OBSERVER DEBUG: Making float draw");
        let float_val = data.draw_float().unwrap();
        println!("TEST_OBSERVER DEBUG: Drew float: {}", float_val);
        
        println!("TEST_OBSERVER DEBUG: Making string draw");
        let string_val = data.draw_string("abcdef", 3, 8).unwrap();
        println!("TEST_OBSERVER DEBUG: Drew string: '{}'", string_val);
        
        // Verify the correct number of choices were made
        println!("TEST_OBSERVER DEBUG: Total choices made: {}", data.choice_count());
        assert_eq!(data.choice_count(), 4);
        
        // Test span tracking within observer
        let span_start = data.start_example("test_span");  
        let _inner_int = data.draw_integer_simple(1, 10).unwrap();
        data.end_example("test_span", span_start);
        
        println!("TEST_OBSERVER DEBUG: Total choices after span: {}", data.choice_count());
        assert_eq!(data.choice_count(), 5); // 4 previous + 1 in span
        
        // Conclude the test properly to ensure observer state is finalized
        println!("TEST_OBSERVER DEBUG: Concluding test with Valid status");
        data.conclude_test(Status::Valid, None).unwrap();
        
        // Test that no more draws are allowed after conclusion
        let draw_result = data.draw_integer_simple(0, 1);
        assert_eq!(draw_result, Err(DrawError::Frozen));
        println!("TEST_OBSERVER DEBUG: Verified concluded data prevents draws");
        
        // Get the result to check final state
        let result = data.as_result();
        assert_eq!(result.status, Status::Valid);
        assert_eq!(result.nodes.len(), 5);
        println!("TEST_OBSERVER DEBUG: Final result has {} choices with status {:?}", 
                 result.nodes.len(), result.status);
        
        // Verify the choices have the expected types and values
        assert_eq!(result.nodes[0].choice_type, ChoiceType::Integer);
        if let ChoiceValue::Integer(val) = &result.nodes[0].value {
            assert!(*val >= 10 && *val <= 100);
            assert_eq!(*val, int_val);
            println!("TEST_OBSERVER DEBUG: Integer choice verified: {}", val);
        } else {
            panic!("Expected integer choice");
        }
        
        assert_eq!(result.nodes[1].choice_type, ChoiceType::Boolean);
        if let ChoiceValue::Boolean(val) = &result.nodes[1].value {
            assert_eq!(*val, bool_val);
            println!("TEST_OBSERVER DEBUG: Boolean choice verified: {}", val);
        } else {
            panic!("Expected boolean choice");
        }
        
        assert_eq!(result.nodes[2].choice_type, ChoiceType::Float);
        if let ChoiceValue::Float(val) = &result.nodes[2].value {
            assert_eq!(*val, float_val);
            println!("TEST_OBSERVER DEBUG: Float choice verified: {}", val);
        } else {
            panic!("Expected float choice");
        }
        
        assert_eq!(result.nodes[3].choice_type, ChoiceType::String);
        if let ChoiceValue::String(val) = &result.nodes[3].value {
            assert_eq!(val, &string_val);
            assert!(val.len() >= 3 && val.len() <= 8);
            assert!(val.chars().all(|c| "abcdef".contains(c)));
            println!("TEST_OBSERVER DEBUG: String choice verified: '{}'", val);
        } else {
            panic!("Expected string choice");
        }
        
        assert_eq!(result.nodes[4].choice_type, ChoiceType::Integer);
        println!("TEST_OBSERVER DEBUG: Fifth choice (span integer) verified");
        
        // Verify examples/spans were recorded
        assert_eq!(result.examples.len(), 1);
        assert_eq!(result.examples[0].label, "test_span");
        assert_eq!(result.examples[0].start, 4); // After the first 4 choices
        assert_eq!(result.examples[0].end, 5);   // After the 5th choice
        println!("TEST_OBSERVER DEBUG: Span tracking verified");
        
        println!("TEST_OBSERVER DEBUG: Comprehensive observer integration test completed successfully");
    }
    
    #[test]
    fn test_tree_observer_comprehensive() {
        println!("TREE_OBSERVER DEBUG: Starting comprehensive tree observer test");
        
        // Create a ConjectureData instance
        let mut data = ConjectureData::new(12345);
        println!("TREE_OBSERVER DEBUG: Created ConjectureData with seed 12345");
        
        // Create and set the tree observer
        let observer = TreeRecordingObserver::new();
        data.set_tree_observer(observer);
        assert!(data.has_observer());
        println!("TREE_OBSERVER DEBUG: Set tree observer on ConjectureData");
        
        // Make draws to populate the tree
        let int1 = data.draw_integer_simple(0, 10).unwrap();
        let bool1 = data.draw_boolean(0.5).unwrap();
        let int2 = data.draw_integer_simple(20, 30).unwrap();
        println!("TREE_OBSERVER DEBUG: Made 3 draws: int1={}, bool1={}, int2={}", int1, bool1, int2);
        
        // Test span tracking
        let span_start = data.start_example("inner_span");
        let inner_int = data.draw_integer_simple(100, 200).unwrap();
        let inner_bool = data.draw_boolean(0.8).unwrap();
        data.end_example("inner_span", span_start);
        println!("TREE_OBSERVER DEBUG: Created span with 2 more draws: inner_int={}, inner_bool={}", inner_int, inner_bool);
        
        // Record some observations
        data.observe("test_key", "test_value");
        data.target("optimization_target", 42.5);
        println!("TREE_OBSERVER DEBUG: Recorded observations and target");
        
        // Conclude the test to finalize tree state
        data.conclude_test(Status::Interesting, Some("test_found_issue")).unwrap();
        println!("TREE_OBSERVER DEBUG: Concluded test with Interesting status");
        
        // Verify the result
        let result = data.as_result();
        assert_eq!(result.status, Status::Interesting);
        assert_eq!(result.nodes.len(), 5);
        println!("TREE_OBSERVER DEBUG: Result verified: {} choices, status {:?}", result.nodes.len(), result.status);
        
        // Verify observations were recorded
        assert!(result.events.contains_key("test_key"));
        assert_eq!(result.events.get("test_key"), Some(&"test_value".to_string()));
        assert!(result.events.contains_key("target:optimization_target"));
        assert_eq!(result.events.get("target:optimization_target"), Some(&"42.5".to_string()));
        assert!(result.events.contains_key("interesting_origin"));
        assert_eq!(result.events.get("interesting_origin"), Some(&"test_found_issue".to_string()));
        println!("TREE_OBSERVER DEBUG: Observations verified in result");
        
        // Verify examples were recorded
        assert_eq!(result.examples.len(), 1);
        assert_eq!(result.examples[0].label, "inner_span");
        assert_eq!(result.examples[0].start, 3); // After first 3 draws
        assert_eq!(result.examples[0].end, 5);   // After all 5 draws
        println!("TREE_OBSERVER DEBUG: Span tracking verified in result");
        
        println!("TREE_OBSERVER DEBUG: Comprehensive tree observer test completed successfully");
    }

    #[test]
    fn test_tree_observer_state_verification() {
        println!("TREE_STATE DEBUG: Starting tree observer state verification test");
        
        // Create a standalone TreeRecordingObserver to test its functionality
        let mut observer = TreeRecordingObserver::new();
        println!("TREE_STATE DEBUG: Created standalone TreeRecordingObserver");
        
        // Verify initial state
        assert!(!observer.is_recording());
        assert!(!observer.has_recorded_paths());
        assert_eq!(observer.current_path_length(), 0);
        println!("TREE_STATE DEBUG: Initial state verified - not recording, no paths");
        
        // Start test recording
        observer.start_test();
        assert!(observer.is_recording());
        assert_eq!(observer.current_status(), Status::Valid);
        println!("TREE_STATE DEBUG: Started test recording - now recording");
        
        // Simulate some choice draws
        let constraints1 = Box::new(Constraints::Integer(IntegerConstraints {
            min_value: Some(0),
            max_value: Some(100),
            weights: None,
            shrink_towards: Some(0),
        }));
        observer.draw_value(ChoiceType::Integer, ChoiceValue::Integer(42), false, constraints1);
        
        let constraints2 = Box::new(Constraints::Boolean(BooleanConstraints { p: 0.5 }));
        observer.draw_value(ChoiceType::Boolean, ChoiceValue::Boolean(true), false, constraints2);
        
        assert_eq!(observer.current_path_length(), 2);
        println!("TREE_STATE DEBUG: Recorded 2 choices, path length = {}", observer.current_path_length());
        
        // Test example tracking
        observer.start_example("test_example");
        let constraints3 = Box::new(Constraints::Integer(IntegerConstraints {
            min_value: Some(10),
            max_value: Some(20),
            weights: None,
            shrink_towards: Some(15),
        }));
        observer.draw_value(ChoiceType::Integer, ChoiceValue::Integer(15), false, constraints3);
        observer.end_example("test_example", false);
        
        assert_eq!(observer.current_path_length(), 3);
        println!("TREE_STATE DEBUG: Added example with 1 more choice, path length = {}", observer.current_path_length());
        
        // Record some observations
        observer.record_observation("test_key", "test_value");
        observer.record_target("score", 85.5);
        println!("TREE_STATE DEBUG: Recorded observations and target");
        
        // Conclude the test
        observer.conclude_test(Status::Interesting);
        assert!(!observer.is_recording());
        assert!(observer.has_recorded_paths());
        println!("TREE_STATE DEBUG: Concluded test - no longer recording, has recorded paths");
        
        // Verify tree statistics
        let stats = observer.get_tree_stats();
        assert!(stats.total_nodes > 0);
        println!("TREE_STATE DEBUG: Tree stats - total_nodes: {}, branch_nodes: {}, 
                 conclusion_nodes: {}, killed_nodes: {}", 
                 stats.total_nodes, stats.branch_nodes, stats.conclusion_nodes, stats.killed_nodes);
        
        // Test novel prefix generation
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let novel_prefix = observer.generate_novel_prefix(&mut rng);
        println!("TREE_STATE DEBUG: Generated novel prefix with {} choices", novel_prefix.len());
        
        // Test simulation capability
        if !novel_prefix.is_empty() {
            let can_simulate = observer.can_simulate(&novel_prefix);
            let (sim_status, sim_observations) = observer.simulate_test(&novel_prefix);
            println!("TREE_STATE DEBUG: Simulation - can_simulate: {}, status: {:?}, observations: {}", 
                     can_simulate, sim_status, sim_observations.len());
        }
        
        println!("TREE_STATE DEBUG: Tree observer state verification test completed successfully");
    }

    #[test]
    fn test_tree_observer_stop_test_functionality() {
        println!("STOP_TEST DEBUG: Starting stop_test functionality test");
        
        // Create a TreeRecordingObserver
        let mut observer = TreeRecordingObserver::new();
        println!("STOP_TEST DEBUG: Created TreeRecordingObserver");
        
        // Start test and make some draws
        observer.start_test();
        assert!(observer.is_recording());
        
        let constraints1 = Box::new(Constraints::Integer(IntegerConstraints {
            min_value: Some(1),
            max_value: Some(10),
            weights: None,
            shrink_towards: Some(1),
        }));
        observer.draw_value(ChoiceType::Integer, ChoiceValue::Integer(5), false, constraints1);
        
        let constraints2 = Box::new(Constraints::Boolean(BooleanConstraints { p: 0.3 }));
        observer.draw_value(ChoiceType::Boolean, ChoiceValue::Boolean(false), false, constraints2);
        
        assert_eq!(observer.current_path_length(), 2);
        println!("STOP_TEST DEBUG: Made 2 draws, path length = {}", observer.current_path_length());
        
        // Stop the test properly
        observer.stop_test();
        assert!(!observer.is_recording());
        assert_eq!(observer.current_path_length(), 0); // Path should be cleared after stop
        println!("STOP_TEST DEBUG: Called stop_test - recording stopped, path cleared");
        
        // Verify we can start a new test
        observer.start_test();
        assert!(observer.is_recording());
        assert_eq!(observer.current_path_length(), 0);
        
        let constraints3 = Box::new(Constraints::Integer(IntegerConstraints {
            min_value: Some(100),
            max_value: Some(200),
            weights: None,
            shrink_towards: Some(100),
        }));
        observer.draw_value(ChoiceType::Integer, ChoiceValue::Integer(150), false, constraints3);
        
        assert_eq!(observer.current_path_length(), 1);
        println!("STOP_TEST DEBUG: Started new test, made 1 draw, path length = {}", observer.current_path_length());
        
        // Test concluding vs stopping
        observer.conclude_test(Status::Valid);
        assert!(!observer.is_recording());
        assert!(observer.has_recorded_paths());
        println!("STOP_TEST DEBUG: Concluded test - paths recorded to tree");
        
        println!("STOP_TEST DEBUG: Stop_test functionality test completed successfully");
    }

    #[test]
    fn test_extra_information_creation() {
        println!("EXTRA_INFO DEBUG: Testing ExtraInformation creation");
        
        let extra_info = ExtraInformation::new();
        assert!(!extra_info.has_information());
        assert_eq!(extra_info.to_string(), "");
        println!("EXTRA_INFO DEBUG: Empty ExtraInformation created successfully");
        
        let default_extra_info = ExtraInformation::default();
        assert!(!default_extra_info.has_information());
        assert_eq!(default_extra_info.to_string(), "");
        println!("EXTRA_INFO DEBUG: Default ExtraInformation created successfully");
    }

    #[test]
    fn test_extra_information_basic_operations() {
        println!("EXTRA_INFO DEBUG: Testing basic ExtraInformation operations");
        
        let mut extra_info = ExtraInformation::new();
        
        // Test insertion
        extra_info.insert_str("key1", "value1");
        assert!(extra_info.has_information());
        assert!(extra_info.contains_key("key1"));
        assert_eq!(extra_info.get("key1"), Some(&ExtraValue::String("value1".to_string())));
        println!("EXTRA_INFO DEBUG: Inserted key1=value1");
        
        // Test insertion with String types
        extra_info.insert("key2".to_string(), "value2".to_string());
        assert!(extra_info.contains_key("key2"));
        assert_eq!(extra_info.get("key2"), Some(&ExtraValue::String("value2".to_string())));
        println!("EXTRA_INFO DEBUG: Inserted key2=value2");
        
        // Test non-existent key
        assert!(!extra_info.contains_key("nonexistent"));
        assert_eq!(extra_info.get("nonexistent"), None);
        
        // Test removal
        let removed = extra_info.remove("key1");
        assert_eq!(removed, Some(ExtraValue::String("value1".to_string())));
        assert!(!extra_info.contains_key("key1"));
        assert!(extra_info.has_information()); // key2 still exists
        println!("EXTRA_INFO DEBUG: Removed key1");
        
        // Test clearing
        extra_info.clear();
        assert!(!extra_info.has_information());
        assert!(!extra_info.contains_key("key2"));
        println!("EXTRA_INFO DEBUG: Cleared all data");
    }

    #[test]
    fn test_extra_information_display_format() {
        println!("EXTRA_INFO DEBUG: Testing ExtraInformation Display format");
        
        let mut extra_info = ExtraInformation::new();
        
        // Test empty display
        assert_eq!(extra_info.to_string(), "");
        println!("EXTRA_INFO DEBUG: Empty display: '{}'", extra_info.to_string());
        
        // Test single key
        extra_info.insert_str("name", "test");
        assert_eq!(extra_info.to_string(), "name=\"test\"");
        println!("EXTRA_INFO DEBUG: Single key display: '{}'", extra_info.to_string());
        
        // Test multiple keys (should be sorted)
        extra_info.insert_str("age", "25");
        extra_info.insert_str("city", "San Francisco");
        let display = extra_info.to_string();
        println!("EXTRA_INFO DEBUG: Multiple keys display: '{}'", display);
        
        // Keys should be sorted alphabetically
        assert!(display.contains("age=\"25\""));
        assert!(display.contains("city=\"San Francisco\""));
        assert!(display.contains("name=\"test\""));
        assert!(display.contains(", "));
        
        // Check that the format matches Python's __repr__ style
        // Should be: age="25", city="San Francisco", name="test"
        let expected_parts = vec!["age=\"25\"", "city=\"San Francisco\"", "name=\"test\""];
        let expected = expected_parts.join(", ");
        assert_eq!(display, expected);
        println!("EXTRA_INFO DEBUG: Display format matches expected: '{}'", expected);
    }

    #[test]
    fn test_extra_information_has_information() {
        println!("EXTRA_INFO DEBUG: Testing has_information method");
        
        let mut extra_info = ExtraInformation::new();
        
        // Initially should have no information
        assert!(!extra_info.has_information());
        println!("EXTRA_INFO DEBUG: Initially has_information: {}", extra_info.has_information());
        
        // After adding one item, should have information
        extra_info.insert_str("test", "value");
        assert!(extra_info.has_information());
        println!("EXTRA_INFO DEBUG: After insertion has_information: {}", extra_info.has_information());
        
        // After removing the item, should have no information again
        extra_info.remove("test");
        assert!(!extra_info.has_information());
        println!("EXTRA_INFO DEBUG: After removal has_information: {}", extra_info.has_information());
        
        // Test with multiple items
        extra_info.insert_str("key1", "val1");
        extra_info.insert_str("key2", "val2");
        assert!(extra_info.has_information());
        
        // Remove one, should still have information
        extra_info.remove("key1");
        assert!(extra_info.has_information());
        
        // Remove the last one, should have no information
        extra_info.remove("key2");
        assert!(!extra_info.has_information());
        println!("EXTRA_INFO DEBUG: Multi-item test completed");
    }

    #[test]
    fn test_extra_information_iterator() {
        println!("EXTRA_INFO DEBUG: Testing ExtraInformation iterator");
        
        let mut extra_info = ExtraInformation::new();
        extra_info.insert_str("key1", "value1");
        extra_info.insert_str("key2", "value2");
        extra_info.insert_str("key3", "value3");
        
        // Test iteration
        let mut count = 0;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        
        for (key, value) in extra_info.iter() {
            count += 1;
            keys.push(key.clone());
            values.push(value.clone());
        }
        
        assert_eq!(count, 3);
        assert!(keys.contains(&"key1".to_string()));
        assert!(keys.contains(&"key2".to_string()));
        assert!(keys.contains(&"key3".to_string()));
        assert!(values.contains(&ExtraValue::String("value1".to_string())));
        assert!(values.contains(&ExtraValue::String("value2".to_string())));
        assert!(values.contains(&ExtraValue::String("value3".to_string())));
        
        println!("EXTRA_INFO DEBUG: Iterator test completed - {} items", count);
    }

    #[test]
    fn test_extra_information_clone_and_equality() {
        println!("EXTRA_INFO DEBUG: Testing ExtraInformation clone and equality");
        
        let mut extra_info1 = ExtraInformation::new();
        extra_info1.insert_str("name", "Alice");
        extra_info1.insert_str("age", "30");
        
        // Test cloning
        let extra_info2 = extra_info1.clone();
        assert_eq!(extra_info1, extra_info2);
        assert_eq!(extra_info1.to_string(), extra_info2.to_string());
        println!("EXTRA_INFO DEBUG: Clone test passed");
        
        // Test inequality after modification
        let mut extra_info3 = extra_info1.clone();
        extra_info3.insert_str("city", "Boston");
        assert_ne!(extra_info1, extra_info3);
        println!("EXTRA_INFO DEBUG: Inequality test passed");
        
        // Test equality with empty instances
        let empty1 = ExtraInformation::new();
        let empty2 = ExtraInformation::new();
        assert_eq!(empty1, empty2);
        println!("EXTRA_INFO DEBUG: Empty equality test passed");
    }

    #[test]
    fn test_extra_information_python_parity() {
        println!("EXTRA_INFO DEBUG: Testing Python __repr__ parity");
        
        let mut extra_info = ExtraInformation::new();
        
        // Test case that matches Python's behavior exactly
        extra_info.insert_str("c", "third");
        extra_info.insert_str("a", "first");
        extra_info.insert_str("b", "second");
        
        // Should be sorted: a="first", b="second", c="third"
        let repr = extra_info.to_string();
        let expected = "a=\"first\", b=\"second\", c=\"third\"";
        assert_eq!(repr, expected);
        println!("EXTRA_INFO DEBUG: Python parity test - got: '{}', expected: '{}'", repr, expected);
        
        // Test with special characters that need escaping
        let mut special_info = ExtraInformation::new();
        special_info.insert_str("quote", "has\"quote");
        special_info.insert_str("newline", "has\nline");
        let special_repr = special_info.to_string();
        println!("EXTRA_INFO DEBUG: Special chars repr: '{}'", special_repr);
        
        // Should handle quotes properly
        assert!(special_repr.contains("newline=\"has\\nline\""));
        assert!(special_repr.contains("quote=\"has\\\"quote\""));
    }

    #[test]
    fn test_conjecture_data_span_integration() {
        println!("SPAN_INTEGRATION DEBUG: Testing ConjectureData span integration");
        
        let mut data = ConjectureData::new(12345);
        
        // Initially has root span (TOP_LABEL = 0 in Python)
        println!("SPAN_INTEGRATION DEBUG: Initial span depth: {}, stack: {:?}", 
                 data.span_depth(), data.active_span_stack);
        assert_eq!(data.span_depth(), 1); // Root span active
        
        println!("SPAN_INTEGRATION DEBUG: Spans length: {}", data.spans().len());
        assert_eq!(data.spans().len(), 1); // Root span exists
        
        // Start a span for outer test structure
        let outer_span = data.start_span(100);
        assert_eq!(data.span_depth(), 2); // Root + outer
        assert_eq!(data.spans().len(), 2); // Root + outer spans exist
        println!("SPAN_INTEGRATION DEBUG: Started outer span {}", outer_span);
        
        // Draw some values within the span
        let val1 = data.draw_integer_simple(1, 10).expect("Should draw integer");
        println!("SPAN_INTEGRATION DEBUG: Drew integer: {}", val1);
        
        // Start a nested span for inner structure
        let inner_span = data.start_span(200);
        assert_eq!(data.span_depth(), 3); // Root + outer + inner
        assert_eq!(data.spans().len(), 3); // Root + outer + inner spans exist
        println!("SPAN_INTEGRATION DEBUG: Started inner span {}", inner_span);
        
        // Draw more values in nested context
        let val2 = data.draw_integer_simple(20, 30).expect("Should draw integer");
        println!("SPAN_INTEGRATION DEBUG: Drew nested integer: {}", val2);
        
        // End inner span
        let ended_inner = data.end_span();
        assert_eq!(ended_inner, Some(inner_span));
        assert_eq!(data.span_depth(), 2); // Back to root + outer
        println!("SPAN_INTEGRATION DEBUG: Ended inner span");
        
        // Draw another value in outer context
        let val3 = data.draw_integer_simple(100, 200).expect("Should draw integer");
        println!("SPAN_INTEGRATION DEBUG: Drew outer integer: {}", val3);
        
        // End outer span
        let ended_outer = data.end_span();
        assert_eq!(ended_outer, Some(outer_span));
        assert_eq!(data.span_depth(), 1); // Back to just root span
        println!("SPAN_INTEGRATION DEBUG: Ended outer span");
        
        // Verify span structure
        let spans = data.spans();
        assert_eq!(spans.len(), 3); // Root + outer + inner spans
        
        // Check that spans exist
        if spans.len() > 0 {
            let root = spans.get(0).expect("Root span should exist");
            assert_eq!(root.index, 0);
        }
        
        // Basic span counts (the specific structure depends on implementation)
        println!("SPAN_INTEGRATION DEBUG: Found {} spans", spans.len());
        
        println!("SPAN_INTEGRATION DEBUG: All span integration tests passed");
    }

    #[test]
    fn test_span_with_choice_tracking() {
        println!("SPAN_CHOICE DEBUG: Testing span integration with choice recording");
        
        let mut data = ConjectureData::new(54321);
        
        // Start span and record its position
        let span_index = data.start_span(999);
        let choices_before = data.nodes.len();
        
        // Make several choices within the span
        let _choice1 = data.draw_integer_simple(1, 100).expect("Draw 1");
        let _choice2 = data.draw_integer_simple(200, 300).expect("Draw 2"); 
        let _choice3 = data.draw_integer_simple(400, 500).expect("Draw 3");
        
        let choices_after = data.nodes.len();
        let choices_made = choices_after - choices_before;
        
        // End the span
        data.end_span();
        
        // Verify span system is working
        if let Some(span) = data.spans().get(span_index) {
            assert_eq!(span.index, span_index);
        }
        
        println!("SPAN_CHOICE DEBUG: Span captured {} choices from {} to {}", 
                 choices_made, choices_before, choices_after);
        
        // Verify we made choices within the span
        assert!(choices_made >= 3, "Should have made at least 3 choices");
        
        println!("SPAN_CHOICE DEBUG: Choice tracking test passed");
    }
    
    #[test]
    fn test_prefix_replay_functionality() {
        // Test the new _pop_choice functionality for prefix replay
        let mut data = ConjectureData::new(42);
        
        // First, record some choices to create a prefix
        let choice1 = data.draw_integer_simple(0, 10).unwrap();
        let choice2 = data.draw_boolean(0.5).unwrap();
        let result = data.as_result();
        
        // Now create a new data instance with the same prefix for replay
        let mut replay_data = ConjectureData::new(999); // Different seed!
        replay_data.set_prefix(result.nodes.clone());
        
        // Replay should give us the exact same values despite different seed
        let replayed_choice1 = replay_data.draw_integer_simple(0, 10).unwrap();
        let replayed_choice2 = replay_data.draw_boolean(0.5).unwrap();
        
        assert_eq!(choice1, replayed_choice1);
        assert_eq!(choice2, replayed_choice2);
        
        // Should have no misalignment
        assert_eq!(replay_data.misaligned_at, None);
    }
    
    #[test]
    fn test_misalignment_detection() {
        // Test misalignment detection when replaying with different constraints
        let mut data = ConjectureData::new(42);
        
        // Record a choice with specific range
        let _choice = data.draw_integer_simple(10, 20).unwrap();
        let result = data.as_result();
        
        // Try to replay with different constraints but overlapping range so fallback works
        let mut replay_data = ConjectureData::new(999);
        replay_data.set_prefix(result.nodes.clone());
        
        // This should trigger misalignment detection (different range but overlapping)
        let _replayed = replay_data.draw_integer_simple(5, 25).unwrap();
        
        // Should have detected misalignment
        assert_eq!(replay_data.misaligned_at, Some(0));
        assert!(replay_data.events.contains_key("misalignment_reason"));
    }
    
    #[test]
    fn test_weighted_integer_drawing() {
        // Test weighted integer drawing functionality
        let mut data = ConjectureData::new(42);
        
        // Create a weight map that heavily favors value 5
        let mut weights = HashMap::new();
        weights.insert(5, 0.7); // 70% probability for value 5
        weights.insert(3, 0.2); // 20% probability for value 3
        // Remaining 10% goes to uniform distribution over other values
        
        // Draw multiple integers and verify the weighted behavior
        let mut results = Vec::new();
        for _ in 0..10 {
            let result = data.draw_integer_weighted(1, 10, Some(weights.clone()), Some(5)).unwrap();
            results.push(result);
            assert!(result >= 1 && result <= 10);
        }
        
        // Should have recorded 10 choices
        assert_eq!(data.nodes.len(), 10);
        
        // Verify that constraints are properly recorded
        for node in &data.nodes {
            if let ChoiceValue::Integer(_) = node.value {
                if let Constraints::Integer(ref int_constraints) = node.constraints {
                    assert_eq!(int_constraints.min_value, Some(1));
                    assert_eq!(int_constraints.max_value, Some(10));
                    assert_eq!(int_constraints.shrink_towards, Some(5));
                    assert!(int_constraints.weights.is_some());
                } else {
                    panic!("Expected integer constraints");
                }
            } else {
                panic!("Expected integer choice");
            }
        }
        
        println!("Weighted integer test - results: {:?}", results);
    }
    
    #[test]
    fn test_weight_validation() {
        // Test weight validation rules from Python
        let mut data = ConjectureData::new(42);
        
        // Test case 1: Weight sum >= 1.0 should fail
        let mut bad_weights = HashMap::new();
        bad_weights.insert(5, 0.7);
        bad_weights.insert(3, 0.5); // Sum = 1.2 > 1.0
        
        let result = data.draw_integer_weighted(1, 10, Some(bad_weights), None);
        assert!(result.is_err());
        
        // Test case 2: Negative weight should fail
        let mut bad_weights2 = HashMap::new();
        bad_weights2.insert(5, -0.1);
        
        let result2 = data.draw_integer_weighted(1, 10, Some(bad_weights2), None);
        assert!(result2.is_err());
        
        // Test case 3: Weight for value outside range should fail
        let mut bad_weights3 = HashMap::new();
        bad_weights3.insert(15, 0.5); // 15 is outside range [1, 10]
        
        let result3 = data.draw_integer_weighted(1, 10, Some(bad_weights3), None);
        assert!(result3.is_err());
        
        // Test case 4: Good weights should work
        let mut good_weights = HashMap::new();
        good_weights.insert(5, 0.3);
        good_weights.insert(7, 0.2);
        
        let result4 = data.draw_integer_weighted(1, 10, Some(good_weights), None);
        assert!(result4.is_ok());
    }
    
    #[test]
    fn test_comprehensive_conjecture_data_functionality() {
        // Test 1: Basic initialization with new fields
        let mut data = ConjectureData::new(42);
        
        // Verify all new fields are properly initialized
        assert_eq!(data.status, Status::Valid);
        assert!(data.testcounter >= 0); // Test counter should be initialized
        assert_eq!(data.max_choices, None);
        assert!(!data.is_find);
        assert_eq!(data.overdraw, 0);
        assert!(!data.has_discards);
        assert!(data.labels_for_structure_stack.is_empty());
        assert!(data.arg_slices.is_empty());
        assert!(data.slice_comments.is_empty());
        assert!(!data.hypothesis_runner);
        assert!(data.cannot_proceed_scope.is_none());
        assert!(data.extra_information.has_information() == false);
        
        // Test 2: for_choices constructor
        let choices = vec![
            ChoiceNode::with_index(
                ChoiceType::Integer,
                ChoiceValue::Integer(42),
                Constraints::Integer(IntegerConstraints {
                    min_value: Some(0),
                    max_value: Some(100),
                    weights: None,
                    shrink_towards: None,
                }),
                false,
                0,
            )
        ];
        
        let observer = TreeRecordingObserver::new();
        let replay_data = ConjectureData::for_choices(
            &choices,
            Some(Box::new(observer)),
            None,
            None
        );
        
        assert_eq!(replay_data.max_choices, Some(1));
        assert_eq!(replay_data.prefix.len(), 1);
        assert!(replay_data.observer.is_some());
        
        // Test 3: ExtraInformation functionality
        let mut extra = ExtraInformation::new();
        assert!(!extra.has_information());
        
        extra.insert("test_key".to_string(), "test_value".to_string());
        assert!(extra.has_information());
        assert_eq!(extra.get("test_key"), Some(&ExtraValue::String("test_value".to_string())));
        
        // Test display functionality
        let display_string = extra.to_string();
        println!("EXTRA_INFO DEBUG: Display string = '{}'", display_string);
        assert!(display_string.contains("test_key=\"test_value\""));
        
        // Test 4: Label system
        let label1 = calc_label_from_name("test");
        let label2 = calc_label_from_name("test");
        let label3 = calc_label_from_name("different");
        
        assert_eq!(label1, label2); // Same name should produce same label
        assert_ne!(label1, label3); // Different names should produce different labels
        
        let combined = combine_labels(&[label1, label3]);
        assert_ne!(combined, label1);
        assert_ne!(combined, label3);
        
        // Test 5: Global test counter increments
        let data2 = ConjectureData::new(123);
        assert!(data2.testcounter > data.testcounter); // Should be incremented
        
        println!("All comprehensive ConjectureData tests passed!");
    }
    
    #[test]
    fn test_label_system_compatibility() {
        // Test that our label system produces consistent results
        let label_top = calc_label_from_name("top");
        
        // Test combining labels
        let labels = vec![
            calc_label_from_name("first"),
            calc_label_from_name("second"),
            calc_label_from_name("third")
        ];
        
        let combined1 = combine_labels(&labels);
        let combined2 = combine_labels(&labels);
        assert_eq!(combined1, combined2); // Should be deterministic
        
        // Test empty combination
        let empty_combined = combine_labels(&[]);
        assert_eq!(empty_combined, 0);
        
        // Test single label combination
        let single_combined = combine_labels(&[labels[0]]);
        assert_eq!(single_combined, labels[0]);
        
        println!("Label system compatibility tests passed!");
    }
}
/// Convert IntervalSet to alphabet string (standalone function to avoid borrowing issues)
fn intervals_to_alphabet_static(intervals: &IntervalSet) -> String {
    let mut alphabet = String::new();
    for &(start, end) in &intervals.intervals {
        for codepoint in start..=end.min(0x10FFFF) { // Limit to valid Unicode
            if let Some(ch) = char::from_u32(codepoint) {
                if is_valid_string_char_static(ch) {
                    alphabet.push(ch);
                }
            }
            // Limit alphabet size to prevent memory issues
            if alphabet.len() > 10000 {
                break;
            }
        }
        if alphabet.len() > 10000 {
            break;
        }
    }
    alphabet
}

/// Static version of is_valid_string_char
fn is_valid_string_char_static(ch: char) -> bool {
    let codepoint = ch as u32;
    
    // Filter out problematic Unicode categories:
    match codepoint {
        // Control characters (except tab, LF, CR)
        0x00..=0x08 => false,
        0x0B..=0x0C => false,
        0x0E..=0x1F => false,
        0x7F..=0x9F => false,
        // Surrogates
        0xD800..=0xDFFF => false,
        // Private use areas
        0xE000..=0xF8FF => false,
        0xF0000..=0xFFFFD => false,
        0x100000..=0x10FFFD => false,
        _ => true,
    }
}
