//! ConjectureRunner - The main test execution engine
//! 
//! This module implements the Rust equivalent of Python's ConjectureRunner class,
//! which orchestrates the entire property-based testing process including
//! generation, execution, shrinking, and result collection.

use crate::data::{ConjectureData, ConjectureResult};
use crate::shrinking::ChoiceShrinker;
use std::time::{Duration, Instant};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Configuration for the ConjectureRunner
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Maximum number of examples to generate
    pub max_examples: u32,
    
    /// Maximum number of shrinking attempts
    pub max_shrinks: u32,
    
    /// Random seed for deterministic execution
    pub seed: u64,
    
    /// Buffer size for generated data
    pub buffer_size: usize,
    
    /// Enable target-driven test generation
    pub enable_targeting: bool,
    
    /// Target labels to optimize for
    pub target_labels: Vec<String>,
    
    /// Targeting phase duration as fraction of total examples
    pub targeting_phase_fraction: f64,
    
    /// Maximum time allowed for entire test run
    pub max_time: Option<Duration>,
    
    /// Maximum time allowed for a single test execution
    pub test_timeout: Option<Duration>,
    
    /// Maximum time allowed for shrinking phase
    pub shrink_timeout: Option<Duration>,
    
    /// Print detailed execution information
    pub verbose: bool,
    
    /// Report intermediate statistics during execution
    pub report_multiple_bugs: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            max_examples: 100,
            max_shrinks: 10000,
            seed: 0,
            buffer_size: 8192,
            enable_targeting: false,
            target_labels: Vec::new(),
            targeting_phase_fraction: 0.1, // 10% of examples for targeting
            max_time: Some(Duration::from_secs(60)), // 60 seconds total
            test_timeout: Some(Duration::from_millis(100)), // 100ms per test
            shrink_timeout: Some(Duration::from_secs(10)), // 10 seconds for shrinking
            verbose: false,
            report_multiple_bugs: false,
        }
    }
}

/// Statistics about test execution
#[derive(Debug, Clone, Default)]
pub struct RunnerStats {
    /// Total examples generated
    pub examples_generated: u32,
    
    /// Examples that were valid (didn't throw exceptions)
    pub valid_examples: u32,
    
    /// Examples that were invalid (threw exceptions)
    pub invalid_examples: u32,
    
    /// Examples that were interesting (failed the test)
    pub interesting_examples: u32,
    
    /// Number of shrinking attempts made
    pub shrink_attempts: u32,
    
    /// Number of examples generated during targeting phase
    pub targeting_examples: u32,
    
    /// Best target scores achieved
    pub best_target_scores: std::collections::HashMap<String, f64>,
    
    /// Total time spent in test execution
    pub total_runtime: Duration,
    
    /// Time spent in generation phase
    pub generation_time: Duration,
    
    /// Time spent in shrinking phase  
    pub shrinking_time: Duration,
    
    /// Number of timeouts during test execution
    pub timeouts: u32,
    
    /// Average time per test execution
    pub avg_test_time: Duration,
    
    /// Tests that overran buffer limits
    pub overrun_examples: u32,
    
    /// Examples that were discarded (filtered out)
    pub discarded_examples: u32,
}

/// Main test execution engine
/// 
/// This is the Rust equivalent of Python's ConjectureRunner. It manages
/// the lifecycle of property-based testing including generation, execution,
/// and shrinking phases.
#[derive(Debug)]
pub struct ConjectureRunner {
    /// Configuration for this runner
    pub config: RunnerConfig,
    
    /// Statistics about execution
    pub stats: RunnerStats,
    
    /// Current random seed
    current_seed: u64,
    
    /// Best interesting result found (if any)
    best_result: Option<ConjectureResult>,
    
    /// Start time of the test run
    start_time: Option<Instant>,
    
    /// Total test execution time accumulator
    total_test_time: Duration,
}

impl ConjectureRunner {
    /// Create a new ConjectureRunner with the given configuration
    pub fn new(config: RunnerConfig) -> Self {
        Self {
            current_seed: config.seed,
            config,
            stats: RunnerStats::default(),
            best_result: None,
            start_time: None,
            total_test_time: Duration::from_nanos(0),
        }
    }
    
    /// Run a property-based test with the given test function
    /// 
    /// The test function should:
    /// - Take a mutable ConjectureData reference
    /// - Generate inputs using the ConjectureData methods
    /// - Return true if the test passes, false if it fails
    /// - Panic/throw if the test is invalid
    pub fn run<F>(&mut self, test_function: F) -> RunResult
    where
        F: Fn(&mut ConjectureData) -> bool,
    {
        println!("RUNNER DEBUG: Starting test run with config: {:?}", self.config);
        
        // Generation phase
        let interesting_result = self.generation_phase(&test_function);
        
        if let Some(result) = interesting_result {
            println!("RUNNER DEBUG: Found interesting result, starting shrinking phase");
            // Shrinking phase
            let shrunk_result = self.shrinking_phase(result, &test_function);
            RunResult::Failed(shrunk_result)
        } else {
            println!("RUNNER DEBUG: No interesting results found, test passes");
            RunResult::Passed
        }
    }
    
    /// Generation phase - try to find an interesting (failing) example
    fn generation_phase<F>(&mut self, test_function: F) -> Option<ConjectureResult>
    where
        F: Fn(&mut ConjectureData) -> bool,
    {
        let targeting_examples = if self.config.enable_targeting {
            (self.config.max_examples as f64 * self.config.targeting_phase_fraction) as u32
        } else {
            0
        };
        
        // First phase: Regular generation
        for example_num in 0..(self.config.max_examples - targeting_examples) {
            self.stats.examples_generated += 1;
            
            // Create new ConjectureData for this example
            let mut data = ConjectureData::new(self.current_seed + example_num as u64);
            
            println!("RUNNER DEBUG: Generating example {} with seed {}", 
                     example_num, self.current_seed + example_num as u64);
            
            // Run the test function
            let test_result = self.execute_test(&mut data, &test_function);
            
            match test_result {
                TestOutcome::Valid => {
                    self.stats.valid_examples += 1;
                    self.update_target_scores(&data);
                    println!("RUNNER DEBUG: Example {} passed", example_num);
                },
                TestOutcome::Invalid => {
                    self.stats.invalid_examples += 1;
                    println!("RUNNER DEBUG: Example {} was invalid", example_num);
                },
                TestOutcome::Interesting(result) => {
                    self.stats.interesting_examples += 1;
                    println!("RUNNER DEBUG: Example {} was interesting (failed)!", example_num);
                    return Some(result);
                },
                TestOutcome::Discarded => {
                    println!("RUNNER DEBUG: Example {} was discarded", example_num);
                },
                TestOutcome::Overrun => {
                    println!("RUNNER DEBUG: Example {} overran buffer", example_num);
                },
            }
        }
        
        // Second phase: Targeting phase (if enabled)
        if self.config.enable_targeting && targeting_examples > 0 {
            println!("RUNNER DEBUG: Starting targeting phase with {} examples", targeting_examples);
            
            for example_num in 0..targeting_examples {
                self.stats.examples_generated += 1;
                self.stats.targeting_examples += 1;
                
                // Create ConjectureData with targeting bias
                let mut data = ConjectureData::new(self.current_seed + (self.config.max_examples + example_num) as u64);
                
                println!("RUNNER DEBUG: Generating targeting example {} with seed {}", 
                         example_num, self.current_seed + (self.config.max_examples + example_num) as u64);
                
                // Run the test function
                let test_result = self.execute_test(&mut data, &test_function);
                
                match test_result {
                    TestOutcome::Valid => {
                        self.stats.valid_examples += 1;
                        self.update_target_scores(&data);
                        println!("RUNNER DEBUG: Targeting example {} passed", example_num);
                    },
                    TestOutcome::Invalid => {
                        self.stats.invalid_examples += 1;
                        println!("RUNNER DEBUG: Targeting example {} was invalid", example_num);
                    },
                    TestOutcome::Interesting(result) => {
                        self.stats.interesting_examples += 1;
                        println!("RUNNER DEBUG: Targeting example {} was interesting (failed)!", example_num);
                        return Some(result);
                    },
                    TestOutcome::Discarded => {
                        println!("RUNNER DEBUG: Targeting example {} was discarded", example_num);
                    },
                    TestOutcome::Overrun => {
                        println!("RUNNER DEBUG: Targeting example {} overran buffer", example_num);
                    },
                }
            }
        }
        
        None // No interesting examples found
    }
    
    /// Update target scores from a successful test execution
    fn update_target_scores(&mut self, data: &ConjectureData) {
        for (key, value) in &data.events {
            if key.starts_with("target:") {
                let label = &key[7..]; // Remove "target:" prefix
                if let Ok(score) = value.parse::<f64>() {
                    let current_best = self.stats.best_target_scores.get(label).copied().unwrap_or(f64::NEG_INFINITY);
                    if score > current_best {
                        self.stats.best_target_scores.insert(label.to_string(), score);
                        println!("TARGET DEBUG: New best score for {}: {}", label, score);
                    }
                }
            }
        }
    }
    
    /// Shrinking phase - minimize the interesting example
    fn shrinking_phase<F>(&mut self, original_result: ConjectureResult, test_function: F) -> ConjectureResult
    where
        F: Fn(&mut ConjectureData) -> bool,
    {
        println!("RUNNER DEBUG: Starting shrinking phase with {} nodes", 
                 original_result.nodes.len());
        
        let mut shrinker = ChoiceShrinker::new(original_result.clone());
        
        // Wrap the test function to work with ConjectureResult instead of ConjectureData
        let shrink_test = |result: &ConjectureResult| -> bool {
            // Create ConjectureData from the result nodes
            let mut data = ConjectureData::new(self.current_seed);
            
            // Replay the nodes from the result, then freeze to prevent new nodes
            for choice in &result.nodes {
                match &choice.value {
                    crate::choice::ChoiceValue::Integer(val) => {
                        if let crate::choice::Constraints::Integer(constraints) = &choice.constraints {
                            let min = constraints.min_value.unwrap_or(i128::MIN);
                            let max = constraints.max_value.unwrap_or(i128::MAX);
                            let _ = data.draw_integer_with_forced(min, max, Some(*val));
                        }
                    },
                    crate::choice::ChoiceValue::Boolean(val) => {
                        if let crate::choice::Constraints::Boolean(constraints) = &choice.constraints {
                            let _ = data.draw_boolean_with_forced(constraints.p, Some(*val));
                        }
                    },
                    crate::choice::ChoiceValue::Float(_val) => {
                        // Float shrinking not implemented yet
                        let _ = data.draw_float_simple();
                    },
                    crate::choice::ChoiceValue::String(_val) => {
                        // String shrinking not implemented yet  
                        let _ = data.draw_string_simple("abc", 0, 10);
                    },
                    crate::choice::ChoiceValue::Bytes(_val) => {
                        // Bytes shrinking not implemented yet
                        let _ = data.draw_bytes_simple(10);
                    },
                }
            }
            
            // Freeze the data to prevent new nodes from being generated
            data.freeze();
            
            // Use panic catch to handle attempts to draw after freeze
            let test_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                test_function(&mut data)
            }));
            
            // If the test panicked (likely due to frozen data), consider it invalid shrinking
            // If it returned false, the shrinking is valid
            match test_result {
                Ok(false) => true,  // Test failed - valid shrinking
                Ok(true) => false,  // Test passed - invalid shrinking
                Err(_) => false,    // Test panicked - invalid shrinking
            }
        };
        
        let shrunk_result = shrinker.shrink(shrink_test);
        self.stats.shrink_attempts += shrinker.attempts;
        
        println!("RUNNER DEBUG: Shrinking complete. Original: {} nodes, Final: {} nodes", 
                 original_result.nodes.len(), shrunk_result.nodes.len());
        
        shrunk_result
    }
    
    /// Execute a single test and determine the outcome
    fn execute_test<F>(&self, data: &mut ConjectureData, test_function: F) -> TestOutcome
    where
        F: Fn(&mut ConjectureData) -> bool,
    {
        // Check if already in error state before running test
        if data.status == crate::data::Status::Overrun {
            data.freeze();
            return TestOutcome::Overrun;
        }
        
        if data.status == crate::data::Status::Invalid {
            data.freeze();
            return TestOutcome::Invalid;
        }
        
        // Use std::panic::catch_unwind to handle panics as invalid tests
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            test_function(data)
        }));
        
        match result {
            Ok(true) => {
                // Test passed - check final status
                data.freeze();
                match data.status {
                    crate::data::Status::Valid => TestOutcome::Valid,
                    crate::data::Status::Overrun => TestOutcome::Overrun,
                    crate::data::Status::Invalid => TestOutcome::Invalid,
                    crate::data::Status::Interesting => TestOutcome::Valid, // Interesting but test passed
                }
            },
            Ok(false) => {
                // Test failed - this is interesting!
                data.freeze();
                match data.status {
                    crate::data::Status::Overrun => TestOutcome::Overrun,
                    crate::data::Status::Invalid => TestOutcome::Invalid,
                    _ => TestOutcome::Interesting(data.as_result()),
                }
            },
            Err(_) => {
                // Test panicked - this is invalid
                data.status = crate::data::Status::Invalid;
                data.freeze();
                TestOutcome::Invalid
            },
        }
    }
}

/// Outcome of running a single test
#[derive(Debug)]
enum TestOutcome {
    /// Test passed (returned true)
    Valid,
    /// Test was invalid (panicked or threw exception)
    Invalid,
    /// Test failed (returned false) - this is what we want to shrink
    Interesting(ConjectureResult),
    /// Test was discarded (filtered out)
    Discarded,
    /// Test overran the buffer limit
    Overrun,
}

/// Result of running the entire property-based test
#[derive(Debug)]
pub enum RunResult {
    /// Test passed - no counterexamples found
    Passed,
    /// Test failed - counterexample found and shrunk
    Failed(ConjectureResult),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runner_creation() {
        let config = RunnerConfig::default();
        let runner = ConjectureRunner::new(config.clone());
        
        assert_eq!(runner.config.max_examples, config.max_examples);
        assert_eq!(runner.stats.examples_generated, 0);
        assert!(runner.best_result.is_none());
    }

    #[test]
    fn test_passing_property() {
        let mut runner = ConjectureRunner::new(RunnerConfig::default());
        
        // Property that always passes
        let result = runner.run(|data| {
            let _x = data.draw_integer(0, 100).unwrap();
            true // Always pass
        });
        
        match result {
            RunResult::Passed => {
                assert_eq!(runner.stats.examples_generated, 100);
                assert_eq!(runner.stats.valid_examples, 100);
                assert_eq!(runner.stats.interesting_examples, 0);
            },
            RunResult::Failed(_) => panic!("Expected test to pass"),
        }
    }

    #[test]
    fn test_failing_property() {
        let mut runner = ConjectureRunner::new(RunnerConfig {
            max_examples: 10,
            max_shrinks: 100,
            seed: 42,
            buffer_size: 8192,
            enable_targeting: false,
            target_labels: Vec::new(),
            targeting_phase_fraction: 0.1,
            max_time: Some(Duration::from_secs(10)),
            test_timeout: Some(Duration::from_millis(100)),
            shrink_timeout: Some(Duration::from_secs(5)),
            verbose: false,
            report_multiple_bugs: false,
        });
        
        // Property that always fails
        let result = runner.run(|data| {
            let _x = data.draw_integer(0, 100).unwrap();
            false // Always fail
        });
        
        match result {
            RunResult::Failed(counterexample) => {
                assert_eq!(runner.stats.examples_generated, 1); // Should find failure immediately
                assert_eq!(runner.stats.interesting_examples, 1);
                // The shrinker might reduce this to 0 nodes if the test always fails
                println!("Counterexample has {} nodes", counterexample.nodes.len());
            },
            RunResult::Passed => panic!("Expected test to fail"),
        }
    }

    #[test]
    fn test_property_with_condition() {
        let mut runner = ConjectureRunner::new(RunnerConfig {
            max_examples: 100,
            max_shrinks: 100,
            seed: 42,
            buffer_size: 8192,
            enable_targeting: false,
            target_labels: Vec::new(),
            targeting_phase_fraction: 0.1,
            max_time: Some(Duration::from_secs(30)),
            test_timeout: Some(Duration::from_millis(100)),
            shrink_timeout: Some(Duration::from_secs(10)),
            verbose: false,
            report_multiple_bugs: false,
        });
        
        // Property that fails when x > 50
        let result = runner.run(|data| {
            let x = match data.draw_integer(0, 100) {
                Ok(val) => val,
                Err(_) => return true, // If draw fails (e.g., frozen), consider test passed
            };
            x <= 50
        });
        
        match result {
            RunResult::Failed(counterexample) => {
                // Should find an example where x > 50
                assert!(!counterexample.nodes.is_empty());
                if let crate::choice::ChoiceValue::Integer(x) = &counterexample.nodes[0].value {
                    assert!(*x > 50, "Counterexample should have x > 50, got x = {}", x);
                } else {
                    panic!("Expected integer choice");
                }
            },
            RunResult::Passed => panic!("Expected test to fail"),
        }
    }
}