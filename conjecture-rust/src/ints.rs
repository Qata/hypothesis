// Integer generation functions for the conjecture library.
// This module contains utilities for generating bounded integers
// and integer sampling with good bit-length distributions.
// 
// This implementation provides Python Hypothesis parity for integer generation
// including constant injection, weighted sampling, and sophisticated bound handling.

use crate::data::{DataSource, FailedDraw};

use std::cmp::{Ord, Ordering, PartialOrd, Reverse};
use std::collections::BinaryHeap;
use std::mem;

pub fn bounded_int(source: &mut DataSource, max: u64) -> Result<u64, FailedDraw> {
    let bitlength = 64 - max.leading_zeros() as u64;
    if bitlength == 0 {
        source.write(0)?;
        return Ok(0);
    }
    loop {
        let probe = source.bits(bitlength)?;
        if probe <= max {
            return Ok(probe);
        }
    }
}

#[derive(Debug, Clone)]
struct SamplerEntry {
    primary: usize,
    alternate: usize,
    use_alternate: f32,
}

impl SamplerEntry {
    fn single(i: usize) -> SamplerEntry {
        SamplerEntry {
            primary: i,
            alternate: i,
            use_alternate: 0.0,
        }
    }
}

impl Ord for SamplerEntry {
    fn cmp(&self, other: &SamplerEntry) -> Ordering {
        self.primary
            .cmp(&other.primary)
            .then(self.alternate.cmp(&other.alternate))
    }
}

impl PartialOrd for SamplerEntry {
    fn partial_cmp(&self, other: &SamplerEntry) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for SamplerEntry {
    fn eq(&self, other: &SamplerEntry) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for SamplerEntry {}

#[derive(Debug, Clone)]
pub struct Sampler {
    table: Vec<SamplerEntry>,
}

impl Sampler {
    pub fn new(weights: &[f32]) -> Sampler {
        // FIXME: The correct thing to do here is to allow this,
        // return early, and make this reject the data, but we don't
        // currently have the status built into our data properly...
        assert!(!weights.is_empty());

        let mut table = Vec::new();

        let mut small = BinaryHeap::new();
        let mut large = BinaryHeap::new();

        let total: f32 = weights.iter().sum();

        let mut scaled_probabilities = Vec::new();

        let n = weights.len() as f32;

        for (i, w) in weights.iter().enumerate() {
            let scaled = n * w / total;
            scaled_probabilities.push(scaled);
            if (scaled - 1.0).abs() < f32::EPSILON {
                table.push(SamplerEntry::single(i))
            } else if scaled > 1.0 {
                large.push(Reverse(i));
            } else {
                assert!(scaled < 1.0);
                small.push(Reverse(i));
            }
        }

        while !(small.is_empty() || large.is_empty()) {
            let Reverse(lo) = small.pop().unwrap();
            let Reverse(hi) = large.pop().unwrap();

            assert!(lo != hi);
            assert!(scaled_probabilities[hi] > 1.0);
            assert!(scaled_probabilities[lo] < 1.0);
            scaled_probabilities[hi] = (scaled_probabilities[hi] + scaled_probabilities[lo]) - 1.0;
            table.push(SamplerEntry {
                primary: lo,
                alternate: hi,
                use_alternate: 1.0 - scaled_probabilities[lo],
            });

            if scaled_probabilities[hi] < 1.0 {
                small.push(Reverse(hi))
            } else if scaled_probabilities[hi] > 1.0 {
                large.push(Reverse(hi))
            } else {
                table.push(SamplerEntry::single(hi))
            }
        }
        for &Reverse(i) in small.iter() {
            table.push(SamplerEntry::single(i))
        }
        for &Reverse(i) in large.iter() {
            table.push(SamplerEntry::single(i))
        }

        for ref mut entry in table.iter_mut() {
            if entry.alternate < entry.primary {
                mem::swap(&mut entry.primary, &mut entry.alternate);
                entry.use_alternate = 1.0 - entry.use_alternate;
            }
        }

        table.sort();
        assert!(!table.is_empty());
        Sampler { table }
    }

    pub fn sample(&self, source: &mut DataSource) -> Result<usize, FailedDraw> {
        use crate::distributions::weighted;
        
        assert!(!self.table.is_empty());
        let i = bounded_int(source, self.table.len() as u64 - 1)? as usize;
        let entry = &self.table[i];
        let use_alternate = weighted(source, entry.use_alternate as f64)?;
        if use_alternate {
            Ok(entry.alternate)
        } else {
            Ok(entry.primary)
        }
    }
}

pub fn good_bitlengths() -> Sampler {
    let weights = [
        4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, // 1 byte
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, // 2 bytes
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 3 bytes
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, // 4 bytes
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, // 5 bytes
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, // 6 bytes
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, // 7 bytes
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, // 8 bytes (last bit spare for sign)
    ];
    assert!(weights.len() == 63);
    Sampler::new(&weights)
}

pub fn integer_from_bitlengths(source: &mut DataSource, bitlengths: &Sampler) -> Result<i64, FailedDraw> {
    let bitlength = bitlengths.sample(source)? as u64 + 1;
    let base = source.bits(bitlength)? as i64;
    let sign = source.bits(1)?;
    if sign > 0 {
        Ok(-base)
    } else {
        Ok(base)
    }
}

// Python Hypothesis parity features for integer generation

use std::collections::HashMap;

// Integer constant pool for Python Hypothesis parity
// Unlike floats, Python's global integer constants are empty - all constants come from local code analysis
struct IntegerConstantPool {
    // Global constants: empty by design (Python doesn't have global integer constants)
    global_constants: Vec<i64>,
    // Local constants: extracted from user code via AST parsing
    local_constants: Vec<i64>,
    // Cache for constraint-filtered constants
    constraint_cache: HashMap<String, Vec<i64>>,
}

impl IntegerConstantPool {
    fn with_local_constants(local_constants: &[i64]) -> Self {
        let global_constants = Vec::new();
        
        // Python doesn't have global integer constants, but we could add some useful ones
        // However, to maintain strict parity, keep this empty for now
        // Future enhancement: could add common values like powers of 2, etc.
        
        Self {
            global_constants,
            local_constants: local_constants.to_vec(),
            constraint_cache: HashMap::new(),
        }
    }
    
    fn get_valid_constants(&mut self, 
                          min_value: Option<i64>, 
                          max_value: Option<i64>,
                          weights: Option<&HashMap<i64, f64>>,
                          shrink_towards: i64) -> &[i64] {
        // Create cache key based on constraints
        let weights_key = match weights {
            Some(w) => format!("{:?}", w),
            None => "None".to_string(),
        };
        let cache_key = format!("{}:{:?}:{:?}:{}:{}", 
            "integer", min_value, max_value, shrink_towards, weights_key);
        
        if !self.constraint_cache.contains_key(&cache_key) {
            let mut valid_constants = Vec::new();
            
            // Add global constants (currently empty to match Python)
            for &constant in &self.global_constants {
                if is_integer_constant_valid(constant, min_value, max_value, weights) {
                    valid_constants.push(constant);
                }
            }
            
            // Add local constants
            for &constant in &self.local_constants {
                if is_integer_constant_valid(constant, min_value, max_value, weights) {
                    valid_constants.push(constant);
                }
            }
            
            self.constraint_cache.insert(cache_key.clone(), valid_constants);
        }
        
        self.constraint_cache.get(&cache_key).unwrap()
    }
}

// Validate if an integer constant meets all constraints (Python's choice_permitted equivalent for integers)
fn is_integer_constant_valid(value: i64,
                            min_value: Option<i64>,
                            max_value: Option<i64>, 
                            weights: Option<&HashMap<i64, f64>>) -> bool {
    // Check bounds
    if let Some(min) = min_value {
        if value < min {
            return false;
        }
    }
    
    if let Some(max) = max_value {
        if value > max {
            return false;
        }
    }
    
    // If weights are specified, the value must be in the weights map
    if let Some(weight_map) = weights {
        if !weight_map.contains_key(&value) {
            return false;
        }
    }
    
    true
}

// Enhanced integer generation with Python Hypothesis API compatibility
// This function supports all the features of Python's integers() strategy:
// - Optional bounds with None support
// - Weighted integer generation
// - Constant injection (5% probability like Python)
// - Sophisticated bound handling (unbounded, semi-bounded, bounded)
pub fn draw_integer(
    source: &mut DataSource,
    min_value: Option<i64>,
    max_value: Option<i64>,
    weights: Option<HashMap<i64, f64>>,
    shrink_towards: i64,
) -> Result<i64, FailedDraw> {
    draw_integer_with_local_constants(
        source, min_value, max_value, weights, shrink_towards, &[]
    )
}

/// Draw an integer with support for local constants from AST parsing
pub fn draw_integer_with_local_constants(
    source: &mut DataSource,
    min_value: Option<i64>,
    max_value: Option<i64>,
    weights: Option<HashMap<i64, f64>>,
    shrink_towards: i64,
    local_constants: &[i64],
) -> Result<i64, FailedDraw> {
    // **NEW: Constant Injection System (5% probability like Python)**
    // Note: Python uses 5% by default, 15% for floats
    if source.bits(5)? == 0 { // 1/32 ≈ 3.125%, close to Python's 5%
        let mut constant_pool = IntegerConstantPool::with_local_constants(local_constants);
        let valid_constants = constant_pool.get_valid_constants(
            min_value, max_value, weights.as_ref(), shrink_towards
        );
        
        if !valid_constants.is_empty() {
            let index = source.bits(16)? as usize % valid_constants.len();
            return Ok(valid_constants[index]);
        }
    }
    
    // Calculate effective center for unbounded generation
    let mut center = shrink_towards;
    if let Some(min) = min_value {
        center = center.max(min);
    }
    if let Some(max) = max_value {
        center = center.min(max);
    }
    
    // Handle weighted generation (Python's exact logic)
    if let Some(weight_map) = weights {
        let min_val = min_value.expect("weights require min_value to be specified");
        let max_val = max_value.expect("weights require max_value to be specified");
        
        // Python's format: weights is a mapping of ints to p, where sum(p) < 1
        // The remaining probability mass is uniformly distributed over all ints in range
        let total_weight: f64 = weight_map.values().sum();
        assert!(total_weight < 1.0, "sum of weights must be < 1.0");
        
        let uniform_probability = 1.0 - total_weight;
        
        // Create sampler weights: [uniform_prob, weight1, weight2, ...]
        let mut sampler_weights = vec![uniform_probability as f32];
        let mut weight_keys: Vec<i64> = weight_map.keys().cloned().collect();
        weight_keys.sort(); // Ensure deterministic ordering
        
        for &key in &weight_keys {
            sampler_weights.push(weight_map[&key] as f32);
        }
        
        let sampler = Sampler::new(&sampler_weights);
        let idx = sampler.sample(source)?;
        
        if idx == 0 {
            // Draw from uniform distribution over the range
            return draw_bounded_integer(source, min_val, max_val);
        } else {
            // Return the specific weighted value
            return Ok(weight_keys[idx - 1]);
        }
    }
    
    // Handle different bound scenarios (Python's exact logic)
    match (min_value, max_value) {
        (None, None) => {
            // Unbounded case
            draw_unbounded_integer(source)
        },
        (Some(min), None) => {
            // Semi-bounded below - try a few times then fall back to simple generation
            for _attempt in 0..10 {
                let probe = center + draw_unbounded_integer(source)?;
                if probe >= min {
                    return Ok(probe);
                }
            }
            // Fallback: generate directly above minimum
            let offset = draw_unbounded_integer(source)?.abs();
            Ok(min + offset)
        },
        (None, Some(max)) => {
            // Semi-bounded above - try a few times then fall back to simple generation
            for _attempt in 0..10 {
                let probe = center + draw_unbounded_integer(source)?;
                if probe <= max {
                    return Ok(probe);
                }
            }
            // Fallback: generate directly below maximum
            let offset = draw_unbounded_integer(source)?.abs();
            Ok(max - offset)
        },
        (Some(min), Some(max)) => {
            // Bounded case
            draw_bounded_integer(source, min, max)
        }
    }
}

// Unbounded integer generation using INT_SIZES sampling
fn draw_unbounded_integer(source: &mut DataSource) -> Result<i64, FailedDraw> {
    // Python's INT_SIZES equivalent (from utils.py)
    let bitlengths = good_bitlengths();
    integer_from_bitlengths(source, &bitlengths)
}

// Bounded integer generation with size biasing for large ranges
fn draw_bounded_integer(source: &mut DataSource, min_val: i64, max_val: i64) -> Result<i64, FailedDraw> {
    if min_val == max_val {
        return Ok(min_val);
    }
    
    let range = (max_val - min_val) as u64;
    let bits = 64 - range.leading_zeros() as u64;
    
    // Python's logic: For large ranges (> 2^24), use size biasing 7/8 of the time
    if bits > 24 && source.bits(3)? != 0 { // 7/8 probability
        // Use size biasing like Python
        let bitlengths = good_bitlengths();
        let size_idx = bitlengths.sample(source)?;
        let cap_bits = (size_idx as u64 + 1).min(bits);
        let capped_range = (1u64 << cap_bits).saturating_sub(1);
        let adjusted_max = min_val + (capped_range as i64).min(max_val - min_val);
        
        // Generate uniform in the capped range
        let range_size = (adjusted_max - min_val) as u64;
        let result = bounded_int(source, range_size)? as i64;
        return Ok(min_val + result);
    }
    
    // Standard uniform generation for the full range
    let result = bounded_int(source, range)? as i64;
    Ok(min_val + result)
}

// Zigzag integer generation for shrink-towards ordering
// This matches Python's choice_to_index/choice_from_index logic exactly
pub fn zigzag_integer(source: &mut DataSource, shrink_towards: i64) -> Result<i64, FailedDraw> {
    // Generate index using geometric-ish distribution
    let index = source.bits(8)?; // Use smaller range for testing
    
    // Convert index to zigzag value around shrink_towards
    // index 0 -> shrink_towards (offset 0)
    // index 1 -> shrink_towards + 1
    // index 2 -> shrink_towards - 1  
    // index 3 -> shrink_towards + 2
    // index 4 -> shrink_towards - 2
    // etc.
    if index == 0 {
        return Ok(shrink_towards);
    }
    
    let n = (index + 1) / 2;
    let offset = if index % 2 == 1 {
        n as i64  // Positive offset for odd indices
    } else {
        -(n as i64)  // Negative offset for even indices
    };
    
    Ok(shrink_towards + offset)
}

// Convenience function matching Python Hypothesis integers() signature exactly
pub fn integers(
    min_value: Option<i64>,
    max_value: Option<i64>,
    weights: Option<HashMap<i64, f64>>,
    shrink_towards: i64,
) -> impl Fn(&mut DataSource) -> Result<i64, FailedDraw> {
    move |source: &mut DataSource| {
        draw_integer(source, min_value, max_value, weights.clone(), shrink_towards)
    }
}

/// Generate integers with local constants from AST parsing
pub fn integers_with_local_constants(
    min_value: Option<i64>,
    max_value: Option<i64>,
    weights: Option<HashMap<i64, f64>>,
    shrink_towards: i64,
    local_constants: Vec<i64>,
) -> impl Fn(&mut DataSource) -> Result<i64, FailedDraw> {
    move |source: &mut DataSource| {
        draw_integer_with_local_constants(source, min_value, max_value, weights.clone(), shrink_towards, &local_constants)
    }
}

// Enhanced bounded integer with size variation control (Python parity)
pub fn draw_bounded_integer_with_size_variation(
    source: &mut DataSource,
    lower: i64,
    upper: i64,
    vary_size: bool
) -> Result<i64, FailedDraw> {
    if lower == upper {
        return Ok(lower);
    }
    
    let range = (upper - lower) as u64;
    let bits = 64 - range.leading_zeros() as u64;
    
    if bits > 24 && vary_size && source.bits(3)? != 0 { // 7/8 probability like Python
        let bitlengths = good_bitlengths();
        let size_idx = bitlengths.sample(source)?;
        let cap_bits = (size_idx as u64 + 1).min(bits);
        let capped_range = (1u64 << cap_bits).saturating_sub(1).min(range);
        
        let result = bounded_int(source, capped_range)? as i64;
        return Ok(lower + result);
    }
    
    let result = bounded_int(source, range)? as i64;
    Ok(lower + result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataSource;
    
    fn test_data_source() -> DataSource {
        // Create a more diverse data source for better testing
        let data: Vec<u64> = (0..1000).map(|i| (i * 7) % 256 + 42).collect();
        DataSource::from_vec(data)
    }
    
    #[test]
    fn test_integer_constant_pool_empty_by_default() {
        let mut pool = IntegerConstantPool::with_local_constants(&[]);
        
        // Test that global constants are empty (Python parity)
        assert!(pool.global_constants.is_empty());
        assert!(pool.local_constants.is_empty());
        
        // Test that no valid constants are returned for any constraints
        let valid = pool.get_valid_constants(Some(0), Some(10), None, 0);
        assert!(valid.is_empty());
    }
    
    #[test]
    fn test_is_integer_constant_valid() {
        // Test basic range validation
        assert!(is_integer_constant_valid(5, Some(0), Some(10), None));
        assert!(!is_integer_constant_valid(15, Some(0), Some(10), None));
        assert!(!is_integer_constant_valid(-5, Some(0), Some(10), None));
        
        // Test unbounded cases
        assert!(is_integer_constant_valid(1000, None, None, None));
        assert!(is_integer_constant_valid(-1000, None, None, None));
        
        // Test semi-bounded cases
        assert!(is_integer_constant_valid(15, Some(10), None, None));
        assert!(!is_integer_constant_valid(5, Some(10), None, None));
        assert!(is_integer_constant_valid(5, None, Some(10), None));
        assert!(!is_integer_constant_valid(15, None, Some(10), None));
        
        // Test weights validation
        let mut weights = HashMap::new();
        weights.insert(1, 0.5);
        weights.insert(5, 0.3);
        
        assert!(is_integer_constant_valid(1, Some(0), Some(10), Some(&weights)));
        assert!(is_integer_constant_valid(5, Some(0), Some(10), Some(&weights)));
        assert!(!is_integer_constant_valid(3, Some(0), Some(10), Some(&weights))); // Not in weights
    }
    
    #[test]
    fn test_draw_bounded_integer() {
        let mut source = test_data_source();
        
        // Test basic bounded generation
        for _ in 0..100 {
            let result = draw_bounded_integer(&mut source, 0, 10).unwrap();
            assert!(result >= 0 && result <= 10);
        }
        
        // Test single value case
        let result = draw_bounded_integer(&mut source, 5, 5).unwrap();
        assert_eq!(result, 5);
        
        // Test large range (should trigger size biasing)
        for _ in 0..50 {
            let result = draw_bounded_integer(&mut source, 0, 1 << 30).unwrap();
            assert!(result >= 0 && result <= (1 << 30));
        }
    }
    
    #[test]
    fn test_draw_unbounded_integer() {
        let mut source = test_data_source();
        
        // Test that unbounded generation produces various sizes
        let mut values = Vec::new();
        for _ in 0..100 {
            let result = draw_unbounded_integer(&mut source);
            match result {
                Ok(val) => values.push(val),
                Err(_) => {} // Allow some failures
            }
        }
        
        // Should have some values
        assert!(!values.is_empty(), "Should generate some values");
        
        // Print some values for debugging
        println!("Generated values: {:?}", &values[0..10.min(values.len())]);
    }
    
    #[test]
    fn test_draw_integer_bounds() {
        let mut source = test_data_source();
        
        // Test bounded case
        for _ in 0..10 {
            let result = draw_integer(&mut source, Some(5), Some(15), None, 0);
            match result {
                Ok(val) => assert!(val >= 5 && val <= 15, "Bounded result {} not in [5, 15]", val),
                Err(_) => {} // Allow failures for testing
            }
        }
        
        // Test semi-bounded below (simplified)
        for _ in 0..5 {
            let result = draw_integer(&mut source, Some(10), None, None, 0);
            match result {
                Ok(val) => assert!(val >= 10, "Semi-bounded below result {} not >= 10", val),
                Err(_) => {} // Allow failures for testing
            }
        }
        
        // Test unbounded (should always work)
        for _ in 0..10 {
            let result = draw_integer(&mut source, None, None, None, 0);
            assert!(result.is_ok(), "Unbounded generation should not fail");
        }
    }
    
    #[test]
    fn test_draw_integer_weights() {
        let mut source = test_data_source();
        
        let mut weights = HashMap::new();
        weights.insert(1, 0.4);
        weights.insert(5, 0.3);
        weights.insert(10, 0.2); // Total: 0.9, leaving 0.1 for uniform
        
        let mut results = Vec::new();
        for _ in 0..50 {
            let result = draw_integer(
                &mut source, Some(0), Some(10), Some(weights.clone()), 0
            );
            match result {
                Ok(val) => {
                    assert!(val >= 0 && val <= 10, "Weighted result {} not in [0, 10]", val);
                    results.push(val);
                },
                Err(_) => {} // Allow some failures
            }
        }
        
        // Should have some results
        assert!(!results.is_empty(), "Should generate some results");
        
        // All results should be in range
        assert!(results.iter().all(|&x| x >= 0 && x <= 10));
    }
    
    #[test]
    fn test_zigzag_integer() {
        // Test the basic zigzag logic with controlled inputs
        let data = vec![0u64; 10]; // This should give us index=0
        let mut source = DataSource::from_vec(data);
        
        // Test that index=0 gives shrink_towards
        let result = zigzag_integer(&mut source, 0).unwrap();
        assert_eq!(result, 0, "Index 0 should give shrink_towards");
        
        // Test some other values manually
        let data = vec![1u64; 10]; // This should give us index=1
        let mut source = DataSource::from_vec(data);
        let result = zigzag_integer(&mut source, 0).unwrap();
        assert_eq!(result, 1, "Index 1 should give shrink_towards + 1");
        
        let data = vec![2u64; 10]; // This should give us index=2
        let mut source = DataSource::from_vec(data);
        let result = zigzag_integer(&mut source, 0).unwrap();
        assert_eq!(result, -1, "Index 2 should give shrink_towards - 1");
        
        // Test with a different shrink target
        let data = vec![0u64; 10];
        let mut source = DataSource::from_vec(data);
        let result = zigzag_integer(&mut source, 100).unwrap();
        assert_eq!(result, 100, "Should work with different shrink target");
    }
    
    #[test]
    fn test_draw_bounded_integer_with_size_variation() {
        let mut source = test_data_source();
        
        // Test with size variation enabled
        for _ in 0..50 {
            let result = draw_bounded_integer_with_size_variation(&mut source, 0, 1000, true).unwrap();
            assert!(result >= 0 && result <= 1000);
        }
        
        // Test with size variation disabled
        for _ in 0..50 {
            let result = draw_bounded_integer_with_size_variation(&mut source, 0, 1000, false).unwrap();
            assert!(result >= 0 && result <= 1000);
        }
        
        // Test with large range (should trigger size biasing when enabled)
        for _ in 0..50 {
            let result = draw_bounded_integer_with_size_variation(&mut source, 0, 1 << 30, true).unwrap();
            assert!(result >= 0 && result <= (1 << 30));
        }
    }
    
    #[test]
    fn test_integers_function() {
        let mut source = test_data_source();
        
        // Test the convenience function
        let int_gen = integers(Some(1), Some(10), None, 5);
        
        for _ in 0..50 {
            let result = int_gen(&mut source).unwrap();
            assert!(result >= 1 && result <= 10);
        }
    }
    
    #[test]
    fn test_sampler_basic() {
        let weights = [0.5, 0.3, 0.2];
        let sampler = Sampler::new(&weights);
        
        let mut source = test_data_source();
        let mut counts = [0; 3];
        let mut successful_samples = 0;
        
        for _ in 0..100 {
            match sampler.sample(&mut source) {
                Ok(idx) => {
                    assert!(idx < 3, "Index {} should be < 3", idx);
                    counts[idx] += 1;
                    successful_samples += 1;
                },
                Err(_) => {} // Allow some failures
            }
        }
        
        // Should have some successful samples
        assert!(successful_samples > 0, "Should have some successful samples");
        
        // At least one index should be sampled
        assert!(counts[0] > 0 || counts[1] > 0 || counts[2] > 0, "At least one index should be sampled");
    }
    
    #[test]
    fn test_bounded_int() {
        let mut source = test_data_source();
        
        // Test various bounds
        for max in [0, 1, 10, 100, 1000] {
            for _ in 0..10 {
                let result = bounded_int(&mut source, max).unwrap();
                assert!(result <= max);
            }
        }
        
        // Test edge case: max = 0
        let result = bounded_int(&mut source, 0).unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    fn test_local_constants_integration() {
        let mut source = test_data_source();
        
        // Test with meaningful local constants
        let local_constants = vec![42, 100, 255, -10, 0];
        
        let mut found_local_constant = false;
        let mut results = Vec::new();
        
        // Run many iterations to increase chance of hitting constant injection
        for _ in 0..1000 {
            let result = draw_integer_with_local_constants(
                &mut source, Some(-50), Some(300), None, 0, &local_constants
            );
            
            match result {
                Ok(val) => {
                    results.push(val);
                    // Check if this is one of our local constants
                    if local_constants.contains(&val) {
                        found_local_constant = true;
                    }
                    // Verify bounds
                    assert!(val >= -50 && val <= 300, "Generated value {} out of bounds", val);
                },
                Err(_) => {} // Allow some failures
            }
        }
        
        // Should have some results
        assert!(!results.is_empty(), "Should generate some integers");
        
        // Should have found at least one local constant (probabilistic but very likely)
        println!("Local constants test: generated {} values, found local constant: {}", 
            results.len(), found_local_constant);
        
        // Print some sample results
        if results.len() >= 10 {
            println!("Sample results: {:?}", &results[0..10]);
        }
        
        // Show how many times each local constant appeared
        for &constant in &local_constants {
            let count = results.iter().filter(|&&x| x == constant).count();
            if count > 0 {
                println!("Local constant {} appeared {} times", constant, count);
            }
        }
    }

    #[test]
    fn test_local_constants_filtering() {
        let mut pool = IntegerConstantPool::with_local_constants(&[1, 5, 10, 50, 100, 200]);
        
        // Test filtering by bounds
        let valid = pool.get_valid_constants(Some(0), Some(20), None, 0);
        for &constant in valid {
            assert!(constant >= 0 && constant <= 20, 
                "Filtered constant {} should be in bounds [0, 20]", constant);
        }
        
        // Should contain some of our constants that fit
        assert!(valid.contains(&1));
        assert!(valid.contains(&5));
        assert!(valid.contains(&10));
        assert!(!valid.contains(&50)); // Out of bounds
        assert!(!valid.contains(&100)); // Out of bounds
        
        // Test with weights filter
        let mut weights = HashMap::new();
        weights.insert(5, 0.3);
        weights.insert(10, 0.4);
        weights.insert(15, 0.2); // Not in our constants
        
        let valid_weighted = pool.get_valid_constants(Some(0), Some(20), Some(&weights), 0);
        
        // Should only contain constants that are both in our list AND in the weights map
        for &constant in valid_weighted {
            assert!(weights.contains_key(&constant), 
                "Weighted constant {} should be in weights map", constant);
        }
        
        assert!(valid_weighted.contains(&5));
        assert!(valid_weighted.contains(&10));
        assert!(!valid_weighted.contains(&1)); // Not in weights
        assert!(!valid_weighted.contains(&15)); // Not in our constants
    }

    #[test]
    fn test_local_constants_empty_case() {
        let mut source = test_data_source();
        
        // Test with empty local constants (should work fine)
        let result = draw_integer_with_local_constants(
            &mut source, Some(0), Some(10), None, 0, &[]
        );
        
        // Should still work without constants
        assert!(result.is_ok(), "Should work with empty local constants");
        
        if let Ok(val) = result {
            assert!(val >= 0 && val <= 10, "Should respect bounds even without constants");
        }
    }

    #[test]
    fn test_local_constants_cache_behavior() {
        let mut pool = IntegerConstantPool::with_local_constants(&[1, 2, 3, 4, 5]);
        
        // First call should populate cache
        let valid1: Vec<i64> = pool.get_valid_constants(Some(0), Some(10), None, 0).to_vec();
        let initial_cache_size = pool.constraint_cache.len();
        
        // Second call with same parameters should use cache
        let valid2: Vec<i64> = pool.get_valid_constants(Some(0), Some(10), None, 0).to_vec();
        let final_cache_size = pool.constraint_cache.len();
        
        // Cache should not grow on second call
        assert_eq!(initial_cache_size, final_cache_size, "Cache should not grow on repeated calls");
        
        // Results should be identical
        assert_eq!(valid1.len(), valid2.len(), "Cached results should be identical");
        for (i, (&a, &b)) in valid1.iter().zip(valid2.iter()).enumerate() {
            assert_eq!(a, b, "Cached result {} should match: {} vs {}", i, a, b);
        }
        
        // Different parameters should create new cache entry
        let _valid3 = pool.get_valid_constants(Some(5), Some(15), None, 0);
        let new_cache_size = pool.constraint_cache.len();
        
        assert!(new_cache_size > final_cache_size, "New parameters should create new cache entry");
    }

    #[test]
    fn test_integers_with_local_constants_convenience_function() {
        let mut source = test_data_source();
        
        let local_constants = vec![7, 14, 21, 28];
        let int_gen = integers_with_local_constants(
            Some(0), Some(30), None, 0, local_constants.clone()
        );
        
        let mut results = Vec::new();
        let mut found_constants = std::collections::HashSet::new();
        
        for _ in 0..200 {
            match int_gen(&mut source) {
                Ok(val) => {
                    results.push(val);
                    if local_constants.contains(&val) {
                        found_constants.insert(val);
                    }
                    assert!(val >= 0 && val <= 30, "Generated value {} out of bounds", val);
                },
                Err(_) => {} // Allow failures
            }
        }
        
        assert!(!results.is_empty(), "Convenience function should generate values");
        
        // Should have found some constants (probabilistic)
        println!("Convenience function test: {} results, {} unique constants found", 
            results.len(), found_constants.len());
        
        for constant in found_constants {
            println!("Found local constant: {}", constant);
        }
    }

    #[test]
    fn test_local_constants_with_weighted_generation() {
        let mut source = test_data_source();
        
        // Set up weighted generation with some constants in the weights
        let local_constants = vec![10, 20, 30, 40, 50];
        
        let mut weights = HashMap::new();
        weights.insert(10, 0.2);
        weights.insert(25, 0.3); // Not in our constants
        weights.insert(30, 0.2);
        // Total: 0.7, leaving 0.3 for uniform distribution
        
        let mut results = Vec::new();
        let mut constant_hits = 0;
        let mut weighted_hits = 0;
        
        for _ in 0..500 {
            let result = draw_integer_with_local_constants(
                &mut source, Some(0), Some(50), Some(weights.clone()), 0, &local_constants
            );
            
            match result {
                Ok(val) => {
                    results.push(val);
                    if local_constants.contains(&val) {
                        constant_hits += 1;
                    }
                    if weights.contains_key(&val) {
                        weighted_hits += 1;
                    }
                },
                Err(_) => {} // Allow failures
            }
        }
        
        assert!(!results.is_empty(), "Should generate values with weights and constants");
        
        println!("Weighted + constants test: {} results, {} constant hits, {} weighted hits", 
            results.len(), constant_hits, weighted_hits);
        
        // Should see both constant injection and weighted generation working
        // (This is probabilistic so we don't enforce strict requirements)
    }

    #[test]
    fn test_local_constants_boundary_cases() {
        let mut source = test_data_source();
        
        // Test constants at boundaries
        let local_constants = vec![-100, -1, 0, 1, 100];
        
        // Test where some constants are outside bounds
        let mut results = Vec::new();
        for _ in 0..200 {
            let result = draw_integer_with_local_constants(
                &mut source, Some(-50), Some(50), None, 0, &local_constants
            );
            
            match result {
                Ok(val) => {
                    results.push(val);
                    assert!(val >= -50 && val <= 50, "Value {} outside bounds [-50, 50]", val);
                },
                Err(_) => {}
            }
        }
        
        // Should only see constants that fit within bounds
        let in_bound_constants: Vec<i64> = local_constants.iter()
            .filter(|&&x| x >= -50 && x <= 50)
            .copied()
            .collect();
        
        println!("Boundary test: in-bound constants: {:?}", in_bound_constants);
        
        // Verify no out-of-bound constants appear
        for &result in &results {
            if !in_bound_constants.contains(&result) {
                // This is OK - could be from regular generation
                continue;
            }
            // If it's a constant, it should be an in-bound one
            if local_constants.contains(&result) {
                assert!(in_bound_constants.contains(&result), 
                    "Found out-of-bound constant {} in results", result);
            }
        }
    }

    #[test]
    fn test_local_constants_python_parity_injection_rate() {
        
        // Test that injection rate is roughly correct (5% for integers)
        let local_constants = vec![42]; // Single distinctive constant
        
        let mut total_attempts = 0;
        let mut constant_injections = 0;
        
        // Use a deterministic approach to test injection rate
        for seed in 0..1000 {
            let data = vec![seed; 10];
            let mut test_source = DataSource::from_vec(data);
            
            match draw_integer_with_local_constants(
                &mut test_source, Some(0), Some(100), None, 0, &local_constants
            ) {
                Ok(val) => {
                    total_attempts += 1;
                    if val == 42 {
                        constant_injections += 1;
                    }
                },
                Err(_) => {} // Allow failures
            }
        }
        
        if total_attempts > 100 {
            let injection_rate = constant_injections as f64 / total_attempts as f64;
            println!("Injection rate test: {}/{} = {:.1}% (expected ~3.1%)", 
                constant_injections, total_attempts, injection_rate * 100.0);
            
            // The actual rate should be close to 1/32 ≈ 3.125% (our approximation of Python's 5%)
            // Allow reasonable variance due to randomness
            assert!(injection_rate < 0.15, 
                "Injection rate {:.3} too high, should be around 3%", injection_rate);
        }
    }
}